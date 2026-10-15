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

from .util import _EXCLUDED_SEGMENTS, to_repo_rel

try:
    from sys import monitoring as _sys_monitoring
except ImportError:  # pragma: no cover - Python < 3.12
    _sys_monitoring = None

# PEP 669 defines six tool ids (0-5).  0-2 and 5 are debugger/coverage/
# profiler/optimizer; 3 and 4 are currently unnamed.  We prefer 4 and fall
# back to 3 (and give up if both are taken by another tool).
_TOOL_IDS = (4, 3)

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
        self, root: Path, exclude_roots: tuple[Path, ...] = ()
    ):
        self.root = Path(root).resolve()
        # Absolute directories whose files are never recorded as dependencies
        # (e.g. the pytest_filemon package itself, which would otherwise show
        # up as a dependency of every test when installed editable in-repo).
        self._exclude_roots = tuple(Path(p).resolve() for p in exclude_roots)
        self._exclude_cache: dict[str, bool] = {}
        # Buckets.
        self._phase = "import"  # import | collect | test | done
        self._nodeid: str | None = None
        self._key: tuple = ("import",)  # current attribution bucket
        self.import_deps: set[str] = set()
        self.collect_deps: set[str] = set()
        self.test_deps: dict[str, set[str]] = {}
        # Internals.
        self._active = False
        self._tool_id: int | None = None
        self._busy = False  # re-entrancy guard (find_spec imports fire PY_START)
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
        """Register the sys.monitoring tool and the PY_START callback."""
        if not _sys_monitoring:
            return False
        for tool_id in _TOOL_IDS:
            try:
                _use_tool(tool_id, "pytest-filemon")
            except ValueError:
                continue
            self._tool_id = tool_id
            break
        else:
            return False
        monitoring_events = _sys_monitoring.events
        try:
            _sys_monitoring.set_events(self._tool_id, monitoring_events.PY_START)
            _sys_monitoring.register_callback(
                self._tool_id, monitoring_events.PY_START, self._on_py_start
            )
        except ValueError:
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

    def _release_tool(self) -> None:
        if self._tool_id is None:
            return
        try:
            _free_tool(self._tool_id)
        except ValueError:  # pragma: no cover
            pass

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
                return None
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
            # Nested event while resolving imports for an outer block (or
            # after the run ended): skip analysis and, crucially, cache
            # nothing -- this code object gets a normal analysis the next
            # time it starts outside a busy region.
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
            return _sys_monitoring.DISABLE if _REARM else None
        # No dependencies ever: silence this code object for good.
        return _sys_monitoring.DISABLE

    def _bucket(self, deps) -> None:
        phase = self._phase
        if phase == "test":
            if self._nodeid is not None:
                self.test_deps.setdefault(self._nodeid, set()).update(deps)
        elif phase == "collect":
            self.collect_deps.update(deps)
        elif phase == "import":
            self.import_deps.update(deps)

    # -- analysis ------------------------------------------------------

    def _analyze(self, code) -> frozenset[str]:
        """Deps of a code block: its own file plus anything it may import."""
        if self._exclude_roots and self._is_excluded(code.co_filename):
            return frozenset()
        own = to_repo_rel(code.co_filename, self.root)
        if own is None:
            # Code outside the repository cannot import repository modules in
            # any way we care about; skip the (relatively expensive) scan.
            return frozenset()
        files: set[str] = set()
        files.add(own)
        if _FAST_IMPORT_PRECHECK:
            co_code = code.co_code
            if (
                _IMPORT_NAME_OP not in co_code
                and _IMPORT_FROM_OP not in co_code
            ):
                # No import bytecode anywhere in this block (byte-level
                # check over the original snapshot): nothing to resolve.
                return frozenset(files)
        last_full: str | None = None
        instrs = list(dis.get_instructions(code, show_caches=False))
        for i, ins in enumerate(instrs):
            op = ins.opname
            if op == "IMPORT_NAME":
                name = _operand_name(code, ins)
                level = _static_level(instrs, i)
                resolved, full = self._resolve_module(name, level, own)
                files |= resolved
                last_full = full
            elif op == "IMPORT_FROM":
                # ``from pkg import attr`` where attr is a submodule.  If it
                # is a plain attribute the resolution simply fails and we
                # skip it -- conservative in the right direction.
                attr = _operand_name(code, ins)
                if attr and last_full:
                    files |= self._resolve_dotted(f"{last_full}.{attr}")
        return frozenset(files)

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
                    origin = to_repo_rel(spec.origin, self.root)
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
