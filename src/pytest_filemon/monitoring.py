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

Code that lives outside the repository is ignored entirely.
"""

from __future__ import annotations

import dis
import importlib.util
import os
from pathlib import Path

from .util import to_repo_rel

try:
    from sys import monitoring as _sys_monitoring
except ImportError:  # pragma: no cover - Python < 3.12
    _sys_monitoring = None

# PEP 669 defines six tool ids (0-5).  0-2 and 5 are debugger/coverage/
# profiler/optimizer; 3 and 4 are currently unnamed.  We prefer 4 and fall
# back to 3 (and give up if both are taken by another tool).
_TOOL_IDS = (4, 3)


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


class FileMonitor:
    """Collects repo-relative file dependencies, bucketed by execution phase."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        # Buckets.
        self.phase = "import"  # import | collect | test | done
        self.current_nodeid: str | None = None
        self.import_deps: set[str] = set()
        self.collect_deps: set[str] = set()
        self.test_deps: dict[str, set[str]] = {}
        # Internals.
        self._active = False
        self._tool_id: int | None = None
        self._busy = False  # re-entrancy guard (find_spec imports fire PY_START)
        self._code_cache: dict = {}  # code object -> frozenset[str]; strong refs
        self._spec_cache: dict[str, str | None] = {}
        self._pkg_prefix_cache: dict[str, list[str]] = {}

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

    # -- event handling ------------------------------------------------

    def _on_py_start(self, code, instruction_offset) -> None:
        # Never let a monitoring bug break the user's test run.
        try:
            self._handle(code)
        except Exception:  # pragma: no cover - defensive
            pass

    def _handle(self, code) -> None:
        if self._busy or self.phase == "done":
            return
        cached = self._code_cache.get(code, False)
        if cached is False:
            self._busy = True
            try:
                cached = self._analyze(code)
            finally:
                self._busy = False
            self._code_cache[code] = cached
        if not cached:
            return
        phase = self.phase
        if phase == "test":
            if self.current_nodeid is not None:
                self.test_deps.setdefault(self.current_nodeid, set()).update(cached)
        elif phase == "collect":
            self.collect_deps.update(cached)
        elif phase == "import":
            self.import_deps.update(cached)

    # -- analysis ------------------------------------------------------

    def _analyze(self, code) -> frozenset[str]:
        """Deps of a code block: its own file plus anything it may import."""
        own = to_repo_rel(code.co_filename, self.root)
        files: set[str] = set()
        if own is None:
            # Code outside the repository cannot import repository modules in
            # any way we care about; skip the (relatively expensive) scan.
            return frozenset()
        files.add(own)
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

        *level* > 0 indicates a relative import, resolved against the package
        of the importing file.  Unresolvable imports return no dependencies
        (if they actually execute, the imported module's PY_START still
        captures them at runtime).
        """
        if level > 0:
            prefix = self._pkg_prefix(own_rel)
            keep = len(prefix) - (level - 1)
            if keep < 0:
                return frozenset(), None
            parts = list(prefix[:keep]) if keep > 0 else []
            if name:
                parts.extend(p for p in name.split(".") if p)
            if not parts:
                return frozenset(), None
            full = ".".join(parts)
            return self._resolve_dotted(full), full
        if not name:
            return frozenset(), None
        return self._resolve_dotted(name), name

    def _resolve_dotted(self, dotted: str) -> frozenset[str]:
        """Resolve a dotted module path, including every ancestor package."""
        files: set[str] = set()
        parts = [p for p in dotted.split(".") if p]
        for i in range(1, len(parts) + 1):
            origin = self._spec_origin(".".join(parts[:i]))
            if origin:
                files.add(origin)
        return frozenset(files)

    def _spec_origin(self, dotted: str) -> str | None:
        if dotted in self._spec_cache:
            return self._spec_cache[dotted]
        origin = None
        try:
            spec = importlib.util.find_spec(dotted)
            # Only real files count: builtin modules report "built-in",
            # frozen modules report synthetic locations, namespace packages
            # report None. (pytest's assertion rewriter injects
            # ``import builtins`` into rewritten modules, so "built-in"
            # origins are common in practice.)
            if (
                spec is not None
                and spec.origin
                and not spec.origin.startswith("<")
                and os.path.isfile(spec.origin)
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
