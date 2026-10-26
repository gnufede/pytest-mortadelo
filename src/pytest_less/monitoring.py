"""File dependency collection via ``sys.monitoring`` (Python 3.12+).

Instead of coverage.py-style per-line coverage, the monitor registers the
*PY_START* and *PY_RESUME* events: every time a code block starts executing --
or resumes after a suspension (a generator or coroutine primed elsewhere runs
its code in whoever's window resumes it) -- it records

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

Non-Python data files are tracked by wrapping ``open``/``os.open`` (on by
default; ``--less-no-data-files`` opts out): every in-repo file opened while
the monitor is active is bucketed like code dependencies -- per test during a
test, into the shared import-time bucket before collection.  Python source and
bytecode opens are deliberately *not* recorded here: they are tracked at block
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

# The untouched functools originals, saved at import time (before any
# monitor patches them).  Wraps always delegate to THESE, not to whatever
# is currently on the functools module -- two coexisting monitors each
# install their own wrapper, and each must call the real original, never
# the other monitor's wrapper (which for cached_property is a function,
# not a class -- subclassing it would crash).
import functools as _ft_mod
import importlib.util
import opcode
import os
import sys
from pathlib import Path

from .blocks import DYNAMIC_DEP, WILDCARD, Block, BlockCache
from .util import _EXCLUDED_SEGMENTS, to_repo_rel

_ORIG_LRU = _ft_mod.lru_cache
_ORIG_CACHE = getattr(_ft_mod, "cache", None)
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

try:
    _IMPORT_NAME_OP = opcode.opmap["IMPORT_NAME"]
    _IMPORT_FROM_OP = opcode.opmap["IMPORT_FROM"]
    _EXTENDED_ARG_OP = opcode.opmap["EXTENDED_ARG"]
    _LOAD_CONST_OP = opcode.opmap["LOAD_CONST"]
except KeyError:  # pragma: no cover - future bytecode without import ops
    _IMPORT_NAME_OP = None
    _IMPORT_FROM_OP = None
    _EXTENDED_ARG_OP = None
    _LOAD_CONST_OP = None
# 3.14 loads small ints (0..255, so every import level) with LOAD_SMALL_INT, whose
# argument *is* the value; before that they were LOAD_CONST (an index).
_LOAD_SMALL_INT_OP = opcode.opmap.get("LOAD_SMALL_INT")
# 3.15 (PEP 810) packs lazy/eager flags into the low two bits of IMPORT_NAME's
# operand; the co_names index is above them.
_IMPORT_NAME_SHIFT = 2 if sys.version_info >= (3, 15) else 0

# On Python 3.13+ code.co_code returns the original (unadaptive) bytecode
# snapshot, so a byte-level presence test for the import opcodes is sound:
# if an import opcode is absent from the bytes, the code object provably
# contains no import bytecode and can skip disassembly entirely (roughly
# three quarters of code objects).  On 3.12 co_code may reflect the live
# adaptive array; there we always disassemble.
# (Validated on 3.13 - 3.15, see tests/test_import_scan.py; on anything newer
# the bytecode may have changed shape again, so it stays on ``dis``.)
_FAST_IMPORT_PRECHECK = (
    (3, 13) <= sys.version_info[:2] <= (3, 15) and _IMPORT_NAME_OP is not None
)

# Whether the "silence after attribution, re-arm on bucket change" scheme is
# available: re-arming needs sys.monitoring.restart_events(), which clears the
# DISABLE state of every code object in one call.  Without it, disabling a code
# object after attribution would lose its events for all later tests, so the
# callback keeps the always-firing behavior instead.
_REARM = _sys_monitoring is not None and hasattr(_sys_monitoring, "restart_events")
# What the callback returns: DISABLE silences a code object until the next
# restart_events().  Dep-less code is silenced unconditionally; code with
# dependencies only when the per-test re-arm is available (otherwise it must
# keep firing).  Module constants: one global lookup instead of an attribute
# chain and a branch per call.
_DISABLE = _sys_monitoring.DISABLE if _sys_monitoring is not None else None
_DISABLE_ATTRIBUTED = _DISABLE if _REARM else None

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
    """The module / attribute name an IMPORT_NAME / IMPORT_FROM refers to.

    ``dis`` has already decoded the operand (``argval``).  Indexing
    ``co_names[arg]`` by hand is wrong from 3.15 on, where IMPORT_NAME carries
    lazy/eager flags in its low two bits and the name index above them: the
    index then falls off the table and the import silently disappears.
    """
    value = instruction.argval
    return value if isinstance(value, str) else None

def _imports_via_dis(code) -> list[tuple[str, str | None, int]]:
    """``[(op, name, level)]`` for every IMPORT_NAME / IMPORT_FROM, via ``dis``.

    The reference implementation: correct on every supported version, but it
    decodes the *whole* code object in pure Python (~1 ms for a module-level
    block) just to find a handful of instructions.
    """
    instrs = list(dis.get_instructions(code, show_caches=False))
    out: list[tuple[str, str | None, int]] = []
    for i, ins in enumerate(instrs):
        if ins.opname == "IMPORT_NAME":
            out.append(("IMPORT_NAME", _operand_name(code, ins), _static_level(instrs, i)))
        elif ins.opname == "IMPORT_FROM":
            out.append(("IMPORT_FROM", _operand_name(code, ins), 0))
    return out

def _imports_via_bytes(code) -> list[tuple[str, str | None, int]] | None:
    """The same list, found by scanning the raw bytecode for the two opcodes.

    Cost is proportional to the number of imports, not to the size of the
    code.  Python 3.13 - 3.15 only (``co_code`` is the un-instrumented,
    un-adapted original there, which the byte-level import precheck already
    relies on).
    Returns ``None`` -- use ``dis`` -- whenever the surrounding instructions
    are not the plain ``LOAD_CONST level; LOAD_CONST fromlist; IMPORT_NAME``
    shape, so an unusual code path can never produce a different answer.
    """
    co = code.co_code
    names = code.co_names
    consts = code.co_consts
    hits: list[tuple[int, int]] = []  # (byte offset, opcode)
    for op in (_IMPORT_NAME_OP, _IMPORT_FROM_OP):
        marker = bytes((op,))
        pos = co.find(marker)
        while pos != -1:
            if pos % 2 == 0:  # an opcode, not the argument byte of the previous unit
                hits.append((pos, op))
            pos = co.find(marker, pos + 1)
    hits.sort()
    out: list[tuple[str, str | None, int]] = []
    for pos, op in hits:
        arg = co[pos + 1]
        start = pos
        shift = 8
        while start >= 2 and co[start - 2] == _EXTENDED_ARG_OP:
            arg |= co[start - 1] << shift
            shift += 8
            start -= 2
        try:
            name = names[arg >> _IMPORT_NAME_SHIFT if op == _IMPORT_NAME_OP else arg]
        except IndexError:
            return None
        if not isinstance(name, str):
            return None
        if op == _IMPORT_FROM_OP:
            out.append(("IMPORT_FROM", name, 0))
            continue
        # The level is an int load among the two units before IMPORT_NAME,
        # nearest first: the fromlist (None or a tuple, never an int) may or
        # may not be pushed depending on the version, and the level is loaded
        # with LOAD_CONST (an index) or LOAD_SMALL_INT (the value itself).
        # Only single-unit loads are decoded; anything else defers to dis.
        level = _level_before(co, start, consts)
        if level is None:
            return None
        out.append(("IMPORT_NAME", name, level))
    return out

def _level_before(co: bytes, start: int, consts: tuple) -> int | None:
    """The relative-import level pushed before the IMPORT_NAME at byte *start*,
    or None if the two units before it are not a plain int load.  Mirrors
    :func:`_static_level` (which reads the same fact from ``dis``)."""
    for back in (2, 4):
        at = start - back
        if at < 0:
            return None
        if at >= 2 and co[at - 2] == _EXTENDED_ARG_OP:
            return None  # a multi-unit load: not decoded here
        op = co[at]
        if op == _LOAD_SMALL_INT_OP and _LOAD_SMALL_INT_OP is not None:
            return co[at + 1]
        if op == _LOAD_CONST_OP:
            value = consts[co[at + 1]]
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        # Anything else in the fromlist slot (a LOAD_CONST of None / a tuple,
        # 3.15's LOAD_COMMON_CONSTANT, ...): the level is one load further
        # back.  Past that, defer to dis.
    return None

def _static_level(instrs, index: int) -> int:
    """Best-effort static read of the relative-import *level* stack operand.

    ``IMPORT_NAME`` pops the fromlist and the level; the level is an int load
    (``LOAD_CONST`` before 3.14, ``LOAD_SMALL_INT`` after) among the two
    instructions before it, nearest first.  The fromlist is ``None`` or a
    tuple, never an int, so the first int load is the level -- on every
    version (3.15 pushes a ``None`` fromlist with ``LOAD_COMMON_CONSTANT``).
    If none is found, assume 0 (an absolute import).  ``EXTENDED_ARG`` prefixes
    (a load whose operand needs more than one byte, e.g. in a module with more
    than 255 constants) are separate instructions in ``dis`` and are skipped,
    as the stdlib's own ``dis._find_imports`` does; counting them made
    ``from ._path import x`` read as level 0 in such modules.
    """
    seen = 0
    for k in range(index - 1, -1, -1):
        prev = instrs[k]
        if prev.opname == "EXTENDED_ARG":
            continue
        if prev.opname in ("LOAD_CONST", "LOAD_SMALL_INT") and isinstance(
            prev.argval, int
        ) and not isinstance(prev.argval, bool):
            return prev.argval
        seen += 1
        if seen == 2:
            break
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
    ):
        self.root = Path(root).resolve()
        # (owner, name, original, wrapper) of every installed interceptor.
        self._undo: list[tuple] = []
        # In-repo files this run only *wrote*: outputs, never inputs (see
        # ``_track_open``).
        self._written: set[str] = set()
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
        self._test_deps: dict[str, set[Block]] = {}
        # Where the callback puts dependencies right now: the current bucket's
        # set, held directly so the hot path is ``sink.update(deps)`` -- no
        # method call, no phase checks, no ``setdefault(nodeid, set())`` that
        # allocated a throwaway set on every event.  Swapped when the bucket
        # changes (``_update_key``) or the test spawns a process.
        self._trash: set[Block] = set()  # attribution that is not recorded
        self._cur_nid: str | None = None
        self._cur_set: set[Block] | None = None
        self._sink: set[Block] = self.import_deps
        # Subprocess boundary: tests that spawned a process during
        # their run, plus a shared mark for collect-phase spawns (affects
        # every test -- same policy as the import-time bucket).
        self.spawn_tests: set[str] = set()
        self.spawn_shared = False
        # Block analysis (one compile per file, lazily).
        self._blocks = BlockCache(self.root)
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
        # id(code object) -> mutable [deps, last_bucket, code] cell.  The cell
        # holds the code object, so its id stays valid (never reused) for the
        # whole run.  Keyed by id, not by the code object: ``dict.get(code)``
        # *hashes the code object*, which walks its constants, names and
        # nested code -- 226 ns on average for real sqlalchemy code, 2-5 us for
        # the heavy functions that run in nearly every test, against 37 ns for
        # the int.  Deps are re-attributed only when the bucket changes, so a
        # code block that starts a thousand times inside one test costs a tuple
        # comparison per event after the first.
        self._code_cache: dict[int, list] = {}
        self._spec_cache: dict[str, str | None] = {}
        self._module_index: dict[str, str] | None = None
        self._resolve_cache: dict[
            tuple[str | None, int, str | None], tuple[frozenset[str], str | None]
        ] = {}
        self._pkg_prefix_cache: dict[str, list[str]] = {}
        self._dotted_cache: dict[str, frozenset[str]] = {}

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

    @property
    def test_deps(self) -> dict[str, set[Block]]:
        """Per-test dependency sets.  An entry exists once its test has
        recorded something (a test that executed no repository code has none,
        exactly as before); the current test's set is published on access."""
        self._publish_current()
        return self._test_deps

    def _publish_current(self) -> None:
        nodeid, current = self._cur_nid, self._cur_set
        if nodeid is not None and current and nodeid not in self._test_deps:
            self._test_deps[nodeid] = current

    def _select_sink(self) -> None:
        """Point ``_sink`` at the bucket for the current phase / test."""
        phase, nodeid = self._phase, self._nodeid
        if phase == "test" and nodeid is not None:
            if nodeid in self.spawn_tests:
                self._cur_nid, self._cur_set = nodeid, None
                self._sink = self._trash
            else:
                existing = self._test_deps.get(nodeid)
                current = existing if existing is not None else set()
                self._cur_nid, self._cur_set = nodeid, current
                self._sink = current
            return
        self._cur_nid = self._cur_set = None
        if phase == "collect":
            self._sink = self.collect_deps
        elif phase == "import":
            self._sink = self.import_deps
        else:  # "done", or a test phase between tests
            self._sink = self._trash

    def _update_key(self) -> None:
        if self._phase == "test" and self._nodeid is not None:
            new_key = ("test", self._nodeid)
        else:
            new_key = (self._phase,)
        if new_key != self._key:
            self._publish_current()  # the finished test's set, before moving on
            self._key = new_key
            self._select_sink()
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
        # Repo modules already imported (pytest's own plugin loading, before
        # the earliest hook) ran before any event could see them.
        self._note_pre_armed_modules()
        try:
            # PY_START covers a code object's first execution in a window;
            # PY_RESUME covers the resumption of a suspended one (generators,
            # coroutines).  A generator primed by one test and resumed by
            # another executes its code in the later test's window: without
            # PY_RESUME that test holds no dependency on the block (seen on
            # sqlalchemy's class-scoped yield fixtures: 186 tests).  Both
            # events honor DISABLE per code object until restart_events(),
            # so the hot path and its once-per-window cost are identical.
            _sys_monitoring.set_events(
                self._tool_id, monitoring_events.PY_START | monitoring_events.PY_RESUME
            )
            _sys_monitoring.register_callback(
                self._tool_id, monitoring_events.PY_START, self._on_py_start
            )
            _sys_monitoring.register_callback(
                self._tool_id, monitoring_events.PY_RESUME, self._on_py_start
            )
        except ValueError:
            self._remove_interceptors()
            self._release_tool()
            return False
        self._active = True
        return True

    def _note_pre_armed_modules(self) -> None:
        """Skeletons of repo modules that were imported before arming.

        pytest imports its plugins -- entry points, ``-p``, ``PYTEST_PLUGINS``
        -- before the earliest hook a plugin can use, so the module-level code
        of a repo-resident plugin has already run when the monitor arms: no
        event can see the past.  What ran is import phase, so these skeletons
        become shared import-time dependencies of every test, the same
        treatment conftests imported before a *late* activation get.  Their
        functions still fire PY_START when called in a window and stay
        block-precise; only the already-executed module level is covered
        here.  Called before the tool is armed so neither the past nor the
        sweep itself is recorded.
        """
        seen: set[str] = set()
        for module in list(sys.modules.values()):
            file = getattr(module, "__file__", None)
            if not file or not file.endswith(".py"):
                continue  # builtins, frozen, namespace packages, .pyc-only
            rel = self.repo_rel(file)
            if rel is not None and rel not in seen:
                seen.add(rel)
                self.import_deps.add(self.module_dep(rel))

    def deactivate(self) -> None:
        if not self._active:
            return
        monitoring_events = _sys_monitoring.events
        try:
            _sys_monitoring.register_callback(
                self._tool_id, monitoring_events.PY_START, None
            )
            _sys_monitoring.register_callback(
                self._tool_id, monitoring_events.PY_RESUME, None
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
        """Wrap the runtime entry points whose effects no event can see.

        All wraps share one pattern: a thin ``functools.wraps`` wrapper that
        calls a tracking method before (or, when the original must run
        first, after) the original, and the shared ``wrap`` helper that
        stashes the undo entry.  A wrapper that cannot be installed, or a
        wrap someone stacked on top of, no-ops through the ``_active``
        check in the tracking method.  Removed by :meth:`deactivate`.

        What is wrapped and why (the gap each one closes):
        * *spawn functions* (``subprocess.Popen.__init__``, ``os.system``,
          ``os.fork``/``exec*``, ``_posixsubprocess.fork_exec``): coverage
          cannot follow a child process, so a spawning test gets no
          dependencies and always re-runs (``always``).
        * ``open`` (data files only): a non-Python in-repo file read by a
          test is a file-level dependency of that test.
        * ``ctypes.CDLL.__init__``: dlopen is not an import (no
          ``IMPORT_NAME`` bytecode, no Python code objects), so the artifact
          is invisible to both the event model and the import scan.  All
          public ctypes loaders (``CDLL``, ``PyDLL``, ``WinDLL``, ``OleDLL``,
          ``cdll.LoadLibrary``) go through ``CDLL.__init__``, and patching
          the *class method* is immune to ``from ctypes import CDLL``
          binding-order issues.
        * ``importlib.import_module``: the reflective-import API produces
          no ``IMPORT_NAME`` opcode in the caller's bytecode (it is a plain
          function call), so the bytecode import scan cannot see it.  The
          ``import`` *statement* is already covered (the scan reads its
          opcodes; ``PY_START`` fires the module's code objects).
          ``sys.meta_path`` was considered and rejected: it fires for every
          import in the process (stdlib, plugins, everything — millions of
          times on a test suite) against 57 calls for the ``import_module``
          wrapper on sqlalchemy.
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

        def intercept(track, *, after=False):
            """A wrapper factory: call ``track(*args)`` and then the
            original (or the original first and then the track, when the
            tracking needs the original's result — ``import_module`` must
            return the module before we can resolve its file)."""

            def make(orig):
                if after:
                    @functools.wraps(orig)
                    def wrapper(*a, **k):
                        result = orig(*a, **k)
                        track(*a, **k)
                        return result
                else:
                    @functools.wraps(orig)
                    def wrapper(*a, **k):
                        track(*a, **k)
                        return orig(*a, **k)
                return wrapper

            return make

        def note_spawn(*a, **k):
            self._note_spawn()

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
            wrap(owner, leaf, intercept(note_spawn))

        # ctypes.CDLL.__init__: the class method, so every subclass
        # constructor (PyDLL, WinDLL, OleDLL) and every LibraryLoader
        # (cdll.LoadLibrary) goes through it.

        # importlib.import_module: resolved after the call (the module must
        # exist before we can ask it where it lives).
        import importlib

        def track_import_module(name, package=None):
            self._track_import_module(name, package)

        wrap(importlib, "import_module", intercept(track_import_module, after=True))

        # functools.lru_cache / cache / cached_property: a memoized function
        # consumes the original's result without executing it, so PY_START
        # never fires on later calls.  Scoped to REPO functions: the wrap
        # checks co_filename at decoration time, and only repo functions
        # get the Python tracking wrapper -- stdlib and third-party cached
        # functions keep the untouched C wrapper (zero overhead on their
        # hot paths).  The dependency is pre-computed at decoration time
        # (resolve the code object to a block once) and added to the
        # current bucket at every call, hit or miss, so every consumer
        # gets the edge.
        def make_scoped_lru(orig_lru):
            """A drop-in for functools.lru_cache that tracks repo functions."""

            def scoped_lru(*args, **kwargs):
                # bare form: @lru_cache (the function is the first arg)
                if args and callable(args[0]) and not isinstance(args[0], int):
                    return _apply(orig_lru, args[0], (), kwargs)
                # parameterized: @lru_cache(maxsize=N) returns a decorator

                def dec(fn):
                    return _apply(orig_lru, fn, args, kwargs)

                return dec

            scoped_lru.__wrapped__ = orig_lru
            return scoped_lru

        def _apply(orig_lru, func, dargs, dkwargs):
            dep = self._resolve_code_dep(func.__code__) if self._is_repo_code(func) else None
            if dargs or dkwargs:
                cached = orig_lru(*dargs, **dkwargs)(func)  # @lru_cache(maxsize=N)
            else:
                cached = orig_lru(func)  # bare @lru_cache or @cache
            if dep is None:
                return cached

            def tracked(*a, **k):
                result = cached(*a, **k)
                if self._active:
                    self._bucket((dep,))
                return result

            tracked.__wrapped__ = func
            tracked.cache_info = cached.cache_info
            tracked.cache_clear = cached.cache_clear
            for attr in ("__name__", "__qualname__", "__doc__", "__module__"):
                try:
                    setattr(tracked, attr, getattr(func, attr))
                except (AttributeError, TypeError):
                    pass
            tracked.__dict__.update(getattr(func, "__dict__", {}))
            return tracked

        wrap(_ft_mod, "lru_cache", lambda orig: make_scoped_lru(_ORIG_LRU))
        if _ORIG_CACHE is not None:
            wrap(_ft_mod, "cache", lambda orig: make_scoped_lru(_ORIG_CACHE))

        # functools.cached_property: a descriptor, not a function wrapper.
        # __get__ is called on EVERY attribute access (hit or miss: it
        # checks the instance __dict__ first), so the tracking fires for
        # every consumer, not just the first one that computed the value.

    def _is_repo_code(self, func) -> bool:
        """Whether *func*'s code object lives in the repository."""
        try:
            return self.repo_rel(func.__code__.co_filename) is not None
        except Exception:
            return False

    def _resolve_code_dep(self, code) -> Block | None:
        """Pre-compute the dependency triple for a code object's block.

        Called once at decoration time; the result is captured in the
        tracking wrapper's closure and added to the bucket at every call,
        hit or miss.  A file-level wildcard if the block table cannot
        resolve the code object."""
        try:
            rel = self.repo_rel(code.co_filename)
            if rel is None:
                return None
            fb = self._blocks.file_blocks(rel)
            if fb is None:
                return (rel, WILDCARD, WILDCARD)
            block_id, content = fb.resolve(
                getattr(code, "co_qualname", code.co_name), code.co_firstlineno
            )
            return (rel, block_id, content)
        except Exception:
            return None

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
                self._publish_current()  # what it recorded before the spawn stays
                self.spawn_tests.add(self._nodeid)
                self._sink = self._trash  # collection stops for this test
            elif self._phase == "collect":
                self.spawn_shared = True
        except Exception:  # pragma: no cover - never break the call we wrap
            pass

    def _track_import_module(self, name, package) -> None:
        """The module a reflective import resolved to is a dependency of
        whoever imported it -- its *skeleton* if it is a .py (exactly what
        the bytecode import scan records for a plain ``import``), a
        file-level wildcard if it is an extension module.  This is the
        ``sys.modules`` gap: only the first import runs the module's code,
        every later one is a silent dict hit."""
        if not self._active:
            return
        try:
            module = sys.modules.get(
                importlib.util.resolve_name(name, package) if isinstance(name, str) else name
            )
            if module is None:
                return  # the import failed (the original already raised)
            f = getattr(module, "__file__", None)
            if f:
                rel = self.repo_rel(f)
                if rel is not None:
                    self._bucket((self.module_dep(rel),))
        except Exception:
            pass  # tracking must never break the import it observes

    # -- event handling --------------------------------------------------

    def _on_py_start(self, code, instruction_offset) -> None:
        """Hot path: called for every code block start *or resumption* in the
        process (PY_START and PY_RESUME share this callback).

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
            cell = self._code_cache.get(id(code))
            if cell is None:
                return self._first_start(code)
            key = self._key
            if cell[1] is key:
                # Already attributed to this bucket: re-silence.  A foreign
                # restart_events() (another monitor in this process, or a
                # co-resident tool) may have re-armed us; this restores the
                # one-event-per-bucket invariant without extra work.
                return _DISABLE_ATTRIBUTED
            deps = cell[0]
            if not deps:
                return _DISABLE
            self._sink.update(deps)
            cell[1] = key
            # Attributed to the current bucket: silence this code object
            # until the bucket changes (the next test start re-arms everything
            # via restart_events()).  A code block that runs a thousand times
            # inside one test now fires exactly once for it.
            return _DISABLE_ATTRIBUTED
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
                self._code_cache[id(code)] = [frozenset(), None, code]
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
        cell = [deps, None, code]
        self._code_cache[id(code)] = cell
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
            cell = self._code_cache.get(id(code))
            if cell is not None:
                # Queued and analyzed since (nested drains): just keep the
                # attribution current.
                if cell[0] and cell[1] is not self._key:
                    self._bucket(cell[0])
                    cell[1] = self._key
                continue
            self._first_start(code)  # busy is False here; nested events re-queue

    def _bucket(self, deps) -> None:
        """Attribute *deps* to the current bucket (cold paths; the hot path
        updates ``_sink`` directly)."""
        self._sink.update(deps)

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
        events = _imports_via_bytes(code) if _FAST_IMPORT_PRECHECK else None
        if events is None:
            events = _imports_via_dis(code)
        for op, name, level in events:
            if op == "IMPORT_NAME":
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
                attr = name
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
        cached = self._dotted_cache.get(dotted)
        if cached is not None:
            return cached
        files: set[str] = set()
        parts = [p for p in dotted.split(".") if p]
        for i in range(1, len(parts) + 1):
            origin = self._spec_origin(".".join(parts[:i]))
            if origin:
                files.add(origin)
        result = self._dotted_cache[dotted] = frozenset(files)
        return result

    def module_dep(self, rel: str) -> Block:
        """Import-time dependency on a module (its skeleton)."""
        return self._blocks.module_dep(rel)


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
                # find_spec("a.b") imports "a" when it is not loaded.  The
                # recorder must not execute imports no test ran: if the parent
                # is not loaded, nothing it could resolve to is a repo file
                # (repo files are in the index above), so leave it.
                parent = dotted.rpartition(".")[0]
                if parent and parent not in sys.modules:
                    raise LookupError(dotted)
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
        """Dotted package prefix for a repo file: ``pkg/sub/mod.py`` ->
        ``["pkg", "sub"]``, and ``lib/pkg/sub/mod.py`` -> ``["pkg", "sub"]``
        too when ``lib`` is a source root (no ``__init__.py``).

        Walks **up** from the file's directory while each one is a package --
        the rule :func:`_build_module_index` names files by.  (Walking down
        from the repo root stops at ``lib/`` and yields ``[]``, so every
        relative import in a ``src/`` or ``lib/`` layout resolved to a
        nonexistent top-level name and contributed no dependency.)"""
        cached = self._pkg_prefix_cache.get(own_rel)
        if cached is not None:
            return cached
        parts = own_rel.split("/")[:-1]
        top = len(parts)
        while top > 0 and (self.root.joinpath(*parts[:top]) / "__init__.py").exists():
            top -= 1
        prefix = parts[top:]
        self._pkg_prefix_cache[own_rel] = prefix
        return prefix
