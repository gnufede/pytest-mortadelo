"""File dependency collection via ``sys.monitoring`` (Python 3.12+).

Instead of coverage.py-style per-line coverage, the monitor registers a single
*PY_START* event: every time a code block starts executing it records

* the file the block lives in (file-level coverage -- we deliberately do not
  track lines), and
* the targets of any import bytecode found inside the block.  This is the
  conservative approximation requested for dynamic imports: an ``import``
  inside a branch that never runs is still recorded as a dependency of
  whatever test executed the surrounding code block.  Imports that *do* run
  are additionally captured through the imported module's own PY_START event.

Dependencies are bucketed by execution phase, which the plugin drives:

* ``import``  -- everything executed before collection starts (pytest plugins,
  conftest imports).  These become dependencies of *every* recorded test.
* ``collect`` -- everything executed while collecting (test module imports
  and their module-level imports).
* ``test``    -- everything executed while a given test runs, attributed to
  that test's node id.

Code that lives outside the repository is ignored entirely, as are the
exclusion roots (the plugin's own package) and virtual environments.

Non-Python data files are tracked by wrapping ``open`` (on by default;
``--less-no-data-files`` opts out): every in-repo file opened while the monitor
is active is bucketed like code dependencies -- per test during a test, into
the shared import-time bucket before collection.  Python source and bytecode
opens are deliberately *not* recorded here: they are tracked at block
precision by PY_START, and a file-level wildcard dep on them would throw that
precision away.

*Subprocess boundaries* are detected by wrapping the functions that create
processes (``subprocess.Popen``, ``os.system``/``fork``/``exec*``/
``posix_spawn``, ``_posixsubprocess.fork_exec``): a process spawned during a
test marks that test as a spawner.  Coverage cannot follow the child, so
collection for that test stops, its dependencies are never written, and it
re-runs every time.  Reflective stdlib entry points (``importlib.import_module``
and friends) mark the running test *dynamic* through a PY_START
pseudo-dependency.  Neither uses ``sys.addaudithook``: an audit hook is called
for every audit event in the process (``builtins.id`` alone fires millions of
times in a large suite), which cost seconds for nothing.

Spawns are only attributed during the collect and test phases: during
pytest's own startup (phase ``import``) our *own* git subprocesses run --
attributing those would mark every test a spawner and destroy precision.
A collect-phase spawn conservatively becomes a shared (all-tests) mark; a
spawn during the run's teardown (phase ``done``) is ignored.

Performance notes (the hot path runs once per code block start -- millions of
times per suite; profiling numbers from the sqlalchemy benchmark):

* the callback is flattened into a single function: a dict lookup plus a
  tuple comparison for code objects already analyzed in the current bucket;
* dependencies of a code object are re-attributed only when the attribution
  bucket changes (i.e. when the next test starts), not on every start;
* file resolutions are memoized per file name (``to_repo_rel`` uses an
  LRU cache; exclusion checks and import resolutions are memoized on the
  monitor).
"""

from __future__ import annotations

import dis
import importlib.util
import opcode
import os
import sys
from pathlib import Path

from .blocks import DYNAMIC_DEP, WILDCARD, Block, BlockCache, ParseCache
from .util import _EXCLUDED_SEGMENTS, to_repo_rel

try:
    from sys import monitoring as _sys_monitoring
except ImportError:  # pragma: no cover - Python < 3.12
    _sys_monitoring = None

# PEP 669 defines six tool ids (0-5).  0, 1, 2 and 5 belong to the standard
# debugger/coverage/profiler/optimizer tools; 3 and 4 are unnamed.  We prefer
# the unnamed ones, then fall back to ANY free id: the allocation is
# dynamic so that several monitors can coexist in one process (e.g. our
# own plugin running on our own tests via an in-process pytester, where the
# outer monitor already holds an id).
# Reflective stdlib entry points: starting one of these in a test means the
# test reaches code (or files) *by computed name*, which the dependency
# recording can only follow for what actually ran.  Matched by
# (qualified name -> file basename) on out-of-repo code objects at their
# first start; the hit becomes the DYNAMIC_DEP pseudo-dependency.
_REFLECTIVE_ENTRY_POINTS = {
    "import_module": ("__init__.py",),  # importlib
    "reload": ("__init__.py",),  # importlib
    "spec_from_file_location": ("util.py",),  # importlib.util
    # runpy/pkgutil may be frozen (co_filename "<frozen runpy>") on 3.13+.
    "run_module": ("runpy.py", "<frozen runpy>"),
    "run_path": ("runpy.py", "<frozen runpy>"),
    "iter_modules": ("pkgutil.py", "<frozen pkgutil>"),
    "walk_packages": ("pkgutil.py", "<frozen pkgutil>"),
    "entry_points": ("__init__.py",),  # importlib.metadata
}

_PREFERRED_TOOL_IDS = (4, 3)
_ALL_TOOL_IDS = tuple(range(6))


def _candidate_tool_ids():
    yield from _PREFERRED_TOOL_IDS
    for tool_id in _ALL_TOOL_IDS:
        if tool_id not in _PREFERRED_TOOL_IDS:
            yield tool_id

# Sentinel for "no cache cell for this code object yet" (distinct from a
# cached empty dependency set, which is a legitimate cell).
_NO_CELL = object()

try:
    _IMPORT_NAME_OP = opcode.opmap["IMPORT_NAME"]
    _IMPORT_FROM_OP = opcode.opmap["IMPORT_FROM"]
except KeyError:  # pragma: no cover - future bytecode without import ops
    _IMPORT_NAME_OP = None
    _IMPORT_FROM_OP = None

# On Python 3.13+ code.co_code returns the original (unadaptive) bytecode
# snapshot, so a byte-level presence test for the import opcodes is sound:
# if an import opcode is absent from the bytes, the code object provably
# contains no import bytecode and can skip disassembly entirely (roughly
# three quarters of code objects).  On 3.12 co_code may reflect the live
# adaptive array; there we always disassemble.
_FAST_IMPORT_PRECHECK = sys.version_info >= (3, 13) and _IMPORT_NAME_OP is not None

# Whether the "silence after attribution, re-arm on bucket change" scheme is
# available: re-arming needs sys.monitoring.restart_events(), which clears the
# DISABLE state of every code object in one call.  Without it, disabling a code
# object after attribution would lose its events for all later tests, so the
# callback keeps the always-firing behavior instead.
_REARM = _sys_monitoring is not None and hasattr(_sys_monitoring, "restart_events")


def _use_tool(tool_id: int, name: str) -> None:
    # Python 3.13 renamed sys.monitoring.use_tool -> use_tool_id.
    fn = getattr(_sys_monitoring, "use_tool_id", None) or getattr(
        _sys_monitoring, "use_tool", None
    )
    fn(tool_id, name)


def _free_tool(tool_id: int) -> None:
    fn = getattr(_sys_monitoring, "free_tool_id", None) or getattr(
        _sys_monitoring, "free_tool", None
    )
    fn(tool_id)


def _operand_name(code, instruction) -> str | None:
    """The co_names entry referenced by an IMPORT_NAME/IMPORT_FROM operand."""
    try:
        value = code.co_names[instruction.arg]
    except (IndexError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _static_level(instrs, index: int) -> int:
    """Best-effort static read of the relative-import *level* stack operand.

    ``IMPORT_NAME`` pops ``fromlist`` (TOS) and ``level`` (TOS1); the compiler
    emits ``LOAD_CONST <level>; LOAD_CONST <fromlist>; IMPORT_NAME`` for every
    import statement, so the level is (almost always) the instruction two
    slots before the IMPORT_NAME.  If we cannot find it we assume 0 (absolute
    import), which keeps the approximation conservative.
    """
    if index >= 2:
        prev = instrs[index - 2]
        if prev.opname == "LOAD_CONST" and isinstance(prev.argval, int):
            return prev.argval
    return 0


def _build_module_index(root: Path, exclude_roots: tuple[Path, ...]) -> dict[str, str]:
    """Map importable dotted names to repo-relative files, conservatively.

    A file's name is its package-qualified path (walking up while each
    directory has an ``__init__.py``); files not inside any package are
    indexed by their stem, as if their directory were on ``sys.path``.

    The map over-approximates importability, which is exactly what the
    conservative import scan wants: a scan-resolved dependency can only
    ever be an *extra* dependency, because an import that actually executes
    is always recorded directly through the imported module's own PY_START
    event.  So an imprecise index can over-record (a few extra reruns) but
    never under-record.
    """
    excluded = {"__pycache__", ".git"} | _EXCLUDED_SEGMENTS
    excl_roots = tuple(os.fspath(p) for p in exclude_roots)

    def excluded_dirpath(dp: str) -> bool:
        return any(dp == e or dp.startswith(e + os.sep) for e in excl_roots)

    is_pkg: dict[Path, bool] = {}

    def dir_is_package(d: Path) -> bool:
        if d not in is_pkg:
            is_pkg[d] = (d / "__init__.py").is_file()
        return is_pkg[d]

    index: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        if excluded_dirpath(dirpath):
            dirnames[:] = []
            continue
        dirnames[:] = [
            d
            for d in dirnames
            if d not in excluded
            and not d.startswith(".")
            and not excluded_dirpath(os.path.join(dirpath, d))
        ]
        directory = Path(dirpath)
        parts: list[str] = []
        cur = directory
        while cur != root and dir_is_package(cur):
            parts.insert(0, cur.name)
            cur = cur.parent
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            stem = fn[:-3]
            rel = (directory / fn).relative_to(root).as_posix()
            if stem == "__init__":
                if not parts:
                    continue
                name = ".".join(parts)
            else:
                name = ".".join([*parts, stem]) if parts else stem
            index.setdefault(name, rel)
    return index


class FileMonitor:
    """Collects repo-relative file dependencies, bucketed by execution phase."""

    def __init__(
        self, root: Path, exclude_roots: tuple[Path, ...] = (),
        *, data_files: bool = False,
        cosmetic: bool = True,
        parse_cache: ParseCache | None = None,
    ):
        self.root = Path(root).resolve()
        # Whether non-Python files opened by tests are recorded as
        # file-level dependencies (wrapped ``open``; on by default).
        self._data_files = data_files
        # (owner, name, original, wrapper) of every installed interceptor.
        self._undo: list[tuple] = []
        # Absolute directories whose files are never recorded as dependencies
        # (e.g. the pytest_less package itself, which would otherwise show
        # up as a dependency of every test when installed editable in-repo).
        self._exclude_roots = tuple(Path(p).resolve() for p in exclude_roots)
        self._exclude_cache: dict[str, bool] = {}
        # Per-instance file-resolution cache (absolute paths only; relative
        # ones depend on the cwd).  No module-level state: two monitors in
        # one process never share (or stale-share) anything.
        self._rel_cache: dict[str, str | None] = {}
        # Buckets.
        self._phase = "import"  # import | collect | test | done
        self._nodeid: str | None = None
        self._key: tuple = ("import",)  # current attribution bucket
        self.import_deps: set[Block] = set()
        self.collect_deps: set[Block] = set()
        self.test_deps: dict[str, set[Block]] = {}
        # Subprocess boundary: tests that spawned a process during
        # their run, plus a shared mark for collect-phase spawns (affects
        # every test -- same policy as the import-time bucket).
        self.spawn_tests: set[str] = set()
        self.spawn_shared = False
        # Block analysis (one AST parse per file, lazily).  ``cosmetic``
        # strips docstrings before hashing (on by default).
        self._blocks = BlockCache(
            self.root, cosmetic=cosmetic, parse_cache=parse_cache
        )
        # Internals.
        self._active = False
        self._tool_id: int | None = None
        self._busy = False  # re-entrancy guard (find_spec imports fire PY_START)
        # Code objects that first started inside a busy region (the stdlib
        # helpers of our own analysis: pathlib, ast, ...): analyzed after the
        # outer analysis completes instead of being dropped.  Dropping them
        # left them uncached, so every later execution re-entered the cold
        # path (~7M redundant dispatches per sqlalchemy run), and could lose
        # deps for code that only ever executes in busy regions.
        self._pending: list = []
        # code object -> mutable [deps, last_bucket] cell.  Strong refs keep
        # the code objects (and their ids) alive for the whole run.  Deps are
        # re-attributed only when the bucket changes, so a code block that
        # starts a thousand times inside one test costs a tuple comparison
        # per event after the first.
        self._code_cache: dict = {}
        self._spec_cache: dict[str, str | None] = {}
        self._module_index: dict[str, str] | None = None
        self._resolve_cache: dict[
            tuple[str | None, int, str | None], tuple[frozenset[str], str | None]
        ] = {}
        self._pkg_prefix_cache: dict[str, list[str]] = {}

    # -- phase / attribution target ------------------------------------

    @property
    def phase(self) -> str:
        return self._phase

    @phase.setter
    def phase(self, value: str) -> None:
        self._phase = value
        self._update_key()

    @property
    def current_nodeid(self) -> str | None:
        return self._nodeid

    @current_nodeid.setter
    def current_nodeid(self, value: str | None) -> None:
        self._nodeid = value
        self._update_key()

    def _update_key(self) -> None:
        if self._phase == "test" and self._nodeid is not None:
            new_key = ("test", self._nodeid)
        else:
            new_key = (self._phase,)
        if new_key != self._key:
            self._key = new_key
            # Attribution bucket changed (the next test started): re-arm
            # every silenced code object so its dependency is attributed to
            # the new test on its next start.  Note that restart_events()
            # clears the per-code state of *all* tools -- acceptable for our
            # use, but a co-resident sys.monitoring tool (coverage sysmon, a
            # debugger) would see its per-code state reset as well.
            if self._active and _REARM:
                try:
                    _sys_monitoring.restart_events()
                except Exception:  # pragma: no cover - defensive
                    pass

    # -- lifecycle -----------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> bool:
        """Register the sys.monitoring tool and the PY_START callback.

        Grabs the first free tool id, preferring the unnamed ones (4, 3) so
        that co-resident standard tools (debugger/coverage/profiler) are
        only claimed as a last resort.
        """
        if not _sys_monitoring:
            return False
        for tool_id in _candidate_tool_ids():
            try:
                _use_tool(tool_id, "pytest-less")
            except ValueError:
                continue
            self._tool_id = tool_id
            break
        else:
            return False
        monitoring_events = _sys_monitoring.events
        # Interceptors first: installing them runs Python code (functools,
        # imports) that the freshly armed tool would otherwise observe.
        self._install_interceptors()
        try:
            _sys_monitoring.set_events(self._tool_id, monitoring_events.PY_START)
            _sys_monitoring.register_callback(
                self._tool_id, monitoring_events.PY_START, self._on_py_start
            )
        except ValueError:
            self._remove_interceptors()
            self._release_tool()
            return False
        self._active = True
        return True

    def deactivate(self) -> None:
        if not self._active:
            return
        monitoring_events = _sys_monitoring.events
        try:
            _sys_monitoring.register_callback(
                self._tool_id, monitoring_events.PY_START, None
            )
            _sys_monitoring.set_events(self._tool_id, monitoring_events.NO_EVENTS)
        except ValueError:  # pragma: no cover - tool already gone
            pass
        finally:
            self._release_tool()
        self._active = False
        self._tool_id = None
        self._remove_interceptors()

    def _release_tool(self) -> None:
        if self._tool_id is None:
            return
        try:
            _free_tool(self._tool_id)
        except ValueError:  # pragma: no cover
            pass

    # -- spawn / data-file interception ----------------------------------
    #
    # No ``sys.addaudithook``: an audit hook is a Python call for *every*
    # audit event in the process, and some of them are extremely hot
    # (``builtins.id`` fires on every ``id()`` call: 22M events -- ~7s -- in
    # the sqlalchemy ORM suite, for a hook that ended up caring about <1k of
    # them).  Wrapping the few functions we care about costs nothing until
    # they are called.  Trade-off: only the Python-level entry points are
    # seen (see ``_SPAWN_TARGETS``); a C extension that forks or opens files
    # on its own is invisible.

    # Repository-relative locations that are never dependencies: VCS
    # internals and pytest's own cache (the latter is written on every
    # run and would otherwise pin "the last test that ran" forever).
    _AUDIT_SKIP_PREFIXES = (".git/", ".pytest_cache/", "__pycache__/")

    # (module, attribute) pairs that create a process or replace the image.
    # ``Popen.__init__`` is the choke point of ``subprocess``; the others
    # cover direct os-level use and ``multiprocessing`` (``os.fork`` /
    # ``_posixsubprocess.fork_exec``).  Names missing on a platform are
    # skipped.
    _SPAWN_TARGETS = (
        ("subprocess", "Popen.__init__"),
        ("subprocess", "_fork_exec"),
        ("_posixsubprocess", "fork_exec"),
        ("os", "system"),
        ("os", "fork"),
        ("os", "forkpty"),
        ("os", "posix_spawn"),
        ("os", "posix_spawnp"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "execvpe"),
    )

    def _install_interceptors(self) -> None:
        """Wrap spawn functions (always) and ``open`` (data files only).

        Wrappers are removed again by :meth:`deactivate`; one that cannot be
        removed (someone wrapped on top of it) no-ops through the ``_active``
        check.
        """
        import functools

        def wrap(owner, name, make):
            orig = getattr(owner, name, None)
            if orig is None:
                return
            wrapper = make(orig)
            try:
                setattr(owner, name, wrapper)
            except (AttributeError, TypeError):  # pragma: no cover
                return
            self._undo.append((owner, name, orig, wrapper))

        def spawn_wrapper(orig):
            note = self._note_spawn

            @functools.wraps(orig)
            def wrapper(*a, **k):
                note()
                return orig(*a, **k)

            return wrapper

        for modname, attr in self._SPAWN_TARGETS:
            try:
                module = __import__(modname)
            except ImportError:  # pragma: no cover - platform specific
                continue
            owner, _, leaf = (module, None, attr)
            if "." in attr:  # "Popen.__init__"
                cls, _, leaf = attr.partition(".")
                owner = getattr(module, cls, None)
                if owner is None:  # pragma: no cover
                    continue
            wrap(owner, leaf, spawn_wrapper)

        if self._data_files:
            import builtins
            import io

            def open_wrapper(orig):
                track = self._track_open

                @functools.wraps(orig)
                def wrapper(file, *a, **k):
                    track(file)
                    return orig(file, *a, **k)

                return wrapper

            # ``builtins.open`` and ``io.open`` are the same object; wrap
            # both bindings (pathlib and most libraries go through one of
            # them at call time).
            wrap(builtins, "open", open_wrapper)
            wrap(io, "open", open_wrapper)

    def _remove_interceptors(self) -> None:
        while self._undo:
            owner, name, orig, wrapper = self._undo.pop()
            try:
                if getattr(owner, name, None) is wrapper:
                    setattr(owner, name, orig)
            except (AttributeError, TypeError):  # pragma: no cover
                pass

    def _note_spawn(self) -> None:
        """A process is being created: attribute it to the running test.

        * test phase: coverage cannot follow the child, so collection for
          this test stops (see ``_bucket``) and the recorder never gives it
          dependencies -- it re-runs every time.
        * collect phase: the culprit is unknown (module import) -> shared
          mark.
        * import phase: pytest's own startup -- and with it our *own* git
          subprocesses during setup/planning -- must not mark every test a
          spawner.  Done phase: recording is over.
        """
        if not self._active:
            return
        try:
            if self._phase == "test" and self._nodeid is not None:
                self.spawn_tests.add(self._nodeid)
            elif self._phase == "collect":
                self.spawn_shared = True
        except Exception:  # pragma: no cover - never break the call we wrap
            pass

    def _track_open(self, path) -> None:
        """Record a non-Python in-repo file opened while the monitor is active
        as a file-level wildcard dependency -- a data file has no code
        structure, so ``(rel, "*", "*")`` means "any content change re-runs
        the tests (or, outside a test, everything) that opened it".  git only
        reports content changes, so mtime-only churn never reaches selection.
        The suffix reject-first filter runs before any path resolution.
        """
        if not self._active:
            return
        try:
            if isinstance(path, str):
                fname = path
            elif isinstance(path, (bytes, os.PathLike)):
                fname = os.fspath(path)
                if not isinstance(fname, str):
                    fname = os.fsdecode(fname)
            else:
                return  # int fd or exotic object: no path to record
            # Python source/bytecode is tracked at block precision by
            # PY_START; a file-level wildcard dep on it would throw that
            # precision away.
            if fname.endswith((".py", ".pyc", ".pyo")):
                return
            rel = self.repo_rel(fname)  # memoized; None outside the repo
            if rel is None:
                return
            if rel == ".git" or rel.startswith(self._AUDIT_SKIP_PREFIXES):
                return
            if self._exclude_roots and self._is_excluded(fname):
                return
            # GIL note: set.add is atomic; bucket attribution under
            # concurrent threads follows the current test (documented
            # over/under-attribution caveat, same as the code tracker).
            self._bucket(((rel, WILDCARD, WILDCARD),))
        except Exception:
            pass  # tracking must never break the open() it observes

    # -- event handling --------------------------------------------------

    def _on_py_start(self, code, instruction_offset) -> None:
        """Hot path: called for every code block start in the process.

        Flattened to a dict lookup plus an identity comparison for the
        overwhelmingly common case (a code object already analyzed, still
        executing within the same attribution bucket).  Code objects whose
        analysis yielded no dependencies are *permanently* silenced by
        returning ``sys.monitoring.DISABLE`` -- CPython honors DISABLE for
        globally-set events (verified on 3.13) and dep-less code can never
        contribute to any bucket, so this is a pure win: out-of-repo,
        venv and excluded code stops firing events after its first start.
        Never let a monitoring bug break the user's test run.
        """
        try:
            cell = self._code_cache.get(code, _NO_CELL)
            if cell is _NO_CELL:
                return self._first_start(code)
            if cell[1] is self._key:
                # Already attributed to this bucket: re-silence.  A foreign
                # restart_events() (another monitor in this process, or a
                # co-resident tool) may have re-armed us; this restores the
                # one-event-per-bucket invariant without extra work.
                return _sys_monitoring.DISABLE if _REARM else None
            deps = cell[0]
            if not deps:
                return _sys_monitoring.DISABLE
            self._bucket(deps)
            cell[1] = self._key
            # Attributed to the current bucket: silence this code object
            # until the bucket changes (the next test start re-arms everything
            # via restart_events()).  A code block that runs a thousand times
            # inside one test now fires exactly once for it.
            if _REARM:
                return _sys_monitoring.DISABLE
            return None
        except Exception:  # pragma: no cover - defensive
            return None

    def _first_start(self, code):
        """Cold path: first start of this code object."""
        if self._busy or self._phase == "done":
            # Nested event while resolving/analyzing for an outer block (or
            # after the run ended): queue it for analysis once the outer one
            # completes.  Analyzing inline would re-enter the machinery that
            # triggered the event; dropping it would both lose dependencies
            # for code that only ever executes in busy regions and leave the
            # object uncached, so every later execution would pay the cold
            # path again.
            if self._phase == "done":
                # Recording is over: no code executing now can ever become a
                # dependency, so cache the object as empty and silence it.
                # Without this, every execution during teardown -- and during
                # our own finish() serialization, which alone makes millions
                # of Python-level calls -- re-entered the cold path as an
                # uncached miss (~7M redundant dispatches per sqlalchemy
                # run).
                self._code_cache[code] = [frozenset(), None]
                return _sys_monitoring.DISABLE
            self._pending.append(code)
            return None
        self._busy = True
        try:
            deps = self._analyze(code)
        except Exception:  # pragma: no cover - defensive
            deps = frozenset()
        finally:
            self._busy = False
        cell = [deps, None]
        self._code_cache[code] = cell
        if deps:
            self._bucket(deps)
            cell[1] = self._key
            result = _sys_monitoring.DISABLE if _REARM else None
        else:
            # No dependencies ever: silence this code object for good.
            result = _sys_monitoring.DISABLE
        self._drain_pending()
        return result

    def _drain_pending(self) -> None:
        """Analyze code objects that first started inside a busy region."""
        while self._pending:
            code = self._pending.pop()
            cell = self._code_cache.get(code)
            if cell is not None:
                # Queued and analyzed since (nested drains): just keep the
                # attribution current.
                if cell[0] and cell[1] is not self._key:
                    self._bucket(cell[0])
                    cell[1] = self._key
                continue
            self._first_start(code)  # busy is False here; nested events re-queue

    def _bucket(self, deps) -> None:
        phase = self._phase
        if phase == "test":
            nodeid = self._nodeid
            # A test that spawned a process is no longer recorded: its
            # dependencies would be incomplete, so collection stops and the
            # recorder drops it (it is simply re-run next time).
            if nodeid is not None and nodeid not in self.spawn_tests:
                self.test_deps.setdefault(nodeid, set()).update(deps)
        elif phase == "collect":
            self.collect_deps.update(deps)
        elif phase == "import":
            self.import_deps.update(deps)

    # -- analysis ------------------------------------------------------

    def _analyze(self, code) -> frozenset[Block]:
        """Deps of a code block: the block itself plus the module skeletons
        of anything its bytecode may import."""
        if self._exclude_roots and self._is_excluded(code.co_filename):
            return frozenset()
        own = self.repo_rel(code.co_filename)
        if own is None:
            # Code outside the repository cannot import repository modules
            # in any way we care about; skip the (relatively expensive)
            # scan.  The one exception: reflective stdlib entry points are
            # remembered (as a pseudo-dependency) so the test that reached
            # code by computed name can be marked dynamic.
            names = _REFLECTIVE_ENTRY_POINTS.get(code.co_qualname)
            if names is not None and (
                os.path.basename(code.co_filename) in names
                or code.co_filename in names
            ):
                return frozenset({DYNAMIC_DEP})
            return frozenset()
        deps: set[Block] = set()
        fb = self._blocks.file_blocks(own)
        if fb is None:
            # File not parseable (deleted mid-run, mid-edit): file-level dep.
            deps.add((own, WILDCARD, WILDCARD))
        else:
            resolved = fb.resolve(
                getattr(code, "co_qualname", code.co_name), code.co_firstlineno
            )
            if resolved is None or resolved[1] == WILDCARD:
                # Unmappable code object (dynamic code, redefined name):
                # file-level fallback.
                deps.add((own, WILDCARD, WILDCARD))
            else:
                block_id, content = resolved
                deps.add((own, block_id, content))
        if _FAST_IMPORT_PRECHECK:
            co_code = code.co_code
            if (
                _IMPORT_NAME_OP not in co_code
                and _IMPORT_FROM_OP not in co_code
            ):
                # No import bytecode anywhere in this block (byte-level
                # check over the original snapshot): nothing to resolve.
                return frozenset(deps)
        last_full: str | None = None
        instrs = list(dis.get_instructions(code, show_caches=False))
        for i, ins in enumerate(instrs):
            op = ins.opname
            if op == "IMPORT_NAME":
                name = _operand_name(code, ins)
                level = _static_level(instrs, i)
                resolved, full = self._resolve_module(name, level, own)
                for rel in resolved:
                    # Importing a module executes its module-level code: the
                    # dependency is the module skeleton, not the whole file
                    # (function bodies of the imported module do not run).
                    deps.add(self._blocks.module_dep(rel))
                last_full = full
            elif op == "IMPORT_FROM":
                # ``from pkg import attr`` where attr is a submodule.  If it
                # is a plain attribute the resolution simply fails and we
                # skip it -- conservative in the right direction.
                attr = _operand_name(code, ins)
                if attr and last_full:
                    for rel in self._resolve_dotted(f"{last_full}.{attr}"):
                        deps.add(self._blocks.module_dep(rel))
        return frozenset(deps)

    def _resolve_module(
        self, name: str | None, level: int, own_rel: str
    ) -> tuple[frozenset[str], str | None]:
        """Resolve an IMPORT_NAME operand to (dep files, absolute dotted name).

        Results are memoized per (name, level, importing file): the same
        import statement appears in many code blocks of the same module.

        *level* > 0 indicates a relative import, resolved against the package
        of the importing file.  Unresolvable imports return no dependencies
        (if they actually execute, the imported module's PY_START still
        captures them at runtime).
        """
        key = (name, level, own_rel)
        cached = self._resolve_cache.get(key)
        if cached is not None:
            return cached
        if level > 0:
            prefix = self._pkg_prefix(own_rel)
            keep = len(prefix) - (level - 1)
            if keep < 0:
                result = (frozenset(), None)
            else:
                parts = list(prefix[:keep]) if keep > 0 else []
                if name:
                    parts.extend(p for p in name.split(".") if p)
                if not parts:
                    result = (frozenset(), None)
                else:
                    full = ".".join(parts)
                    result = (self._resolve_dotted(full), full)
        elif not name:
            result = (frozenset(), None)
        else:
            result = (self._resolve_dotted(name), name)
        self._resolve_cache[key] = result
        return result

    def _resolve_dotted(self, dotted: str) -> frozenset[str]:
        """Resolve a dotted module path, including every ancestor package."""
        files: set[str] = set()
        parts = [p for p in dotted.split(".") if p]
        for i in range(1, len(parts) + 1):
            origin = self._spec_origin(".".join(parts[:i]))
            if origin:
                files.add(origin)
        return frozenset(files)

    def module_dep(self, rel: str) -> Block:
        """Import-time dependency on a module (its skeleton)."""
        return self._blocks.module_dep(rel)

    @property
    def parse_cache(self) -> ParseCache | None:
        return self._blocks.parse_cache

    @property
    def parse_stats(self) -> tuple[int, int]:
        """(parse-cache hits, files actually parsed)."""
        return self._blocks.hits, self._blocks.misses

    def file_blocks(self, rel: str):
        """Block table of a repo file (None if unparseable); cached."""
        return self._blocks.file_blocks(rel)

    def repo_rel(self, filename: str) -> str | None:
        """Repo-relative path for a filename, memoized per monitor.

        The hot file-resolution entry point (once per code object start).
        Only absolute paths are cached; relative ones depend on the cwd.
        """
        if not os.path.isabs(filename):
            return to_repo_rel(filename, self.root)
        try:
            return self._rel_cache[filename]
        except KeyError:
            rel = to_repo_rel(filename, self.root)
            self._rel_cache[filename] = rel
            return rel

    def _is_excluded(self, filename: str) -> bool:
        """Whether *filename* lives under one of the excluded roots.

        Memoized per (absolute) file name: the check involves a resolve()
        syscall and runs once per code object otherwise.
        """
        if not self._exclude_roots:
            return False
        if not os.path.isabs(filename):
            # Rare (relative co_filename): depends on the cwd, do not cache.
            return self._compute_excluded(Path.cwd() / filename)
        cached = self._exclude_cache.get(filename)
        if cached is None:
            cached = self._compute_excluded(Path(filename))
            self._exclude_cache[filename] = cached
        return cached

    def _compute_excluded(self, p: Path) -> bool:
        try:
            p = p.resolve()
        except OSError:
            return False
        return any(p.is_relative_to(root) for root in self._exclude_roots)

    def _spec_origin(self, dotted: str) -> str | None:
        if dotted in self._spec_cache:
            return self._spec_cache[dotted]
        origin = None
        if self._module_index is None:
            self._module_index = _build_module_index(self.root, self._exclude_roots)
        origin = self._module_index.get(dotted)
        if origin is None:
            # Not in the repository index: fall back to importlib for
            # sys.path subtleties (e.g. a module importable only via a
            # directory that pytest put on sys.path).  The index values are
            # already repo-relative; importlib needs the conversion.
            try:
                spec = importlib.util.find_spec(dotted)
                # Only real files count: builtin modules report "built-in",
                # frozen modules report synthetic locations, namespace
                # packages report None. (pytest's assertion rewriter
                # injects ``import builtins`` into rewritten modules, so
                # "built-in" origins are common in practice.)
                if (
                    spec is not None
                    and spec.origin
                    and not spec.origin.startswith("<")
                    and os.path.isfile(spec.origin)
                    and not self._is_excluded(spec.origin)
                ):
                    origin = self.repo_rel(spec.origin)
            except Exception:
                origin = None
        self._spec_cache[dotted] = origin
        return origin

    def _pkg_prefix(self, own_rel: str) -> list[str]:
        """Dotted package prefix for a repo file, e.g. ``pkg/sub/mod.py`` ->
        ``["pkg", "sub"]`` when each level has an ``__init__.py``."""
        cached = self._pkg_prefix_cache.get(own_rel)
        if cached is not None:
            return cached
        parts = own_rel.split("/")[:-1]
        prefix: list[str] = []
        for i in range(1, len(parts) + 1):
            if (self.root.joinpath(*parts[:i]) / "__init__.py").exists():
                prefix = parts[:i]
            else:
                break
        self._pkg_prefix_cache[own_rel] = prefix
        return prefix
