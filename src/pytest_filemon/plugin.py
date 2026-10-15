"""The pytest plugin: wires the monitor, the git store and selection together.

Lifecycle (``pytest --filemon``):

1. ``pytest_load_initial_conftests`` -- activate the sys.monitoring tool as
   early as possible (before any conftest is imported) so import-time
   dependencies are captured.  Options are not parsed yet at this point, so
   the raw command line / ``PYTEST_FILEMON`` env var are peeked at.
2. ``pytest_configure`` -- decide whether filemon is enabled, find the git
   root, load the shard for this exact pytest command.
3. ``pytest_collection`` -- monitor switches to the *collect* phase (test
   module imports); afterwards to the *test* phase.
4. ``pytest_collection_modifyitems`` -- select tests via git diff against the
   shard's base commit; deselect the rest.
5. ``pytest_runtest_logstart`` / ``logreport`` -- attribute PY_START events to
   the running test; track failures and completion.
6. ``pytest_sessionfinish`` -- merge recorded dependencies (per-test deps +
   import-time deps) and store the shard as a git ref.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from .gitstore import GitStore
from .monitoring import FileMonitor
from .selection import Selection, Shard, load_shard, select_tests
from .util import command_key, git_root, to_repo_rel

# Module-level state.  The plugin runs once per pytest process; hooks that do
# not receive a config object read these globals.
_EARLY_MONITOR: FileMonitor | None = None
_STATE: FilemonState | None = None

# The plugin's own package directory: never recorded as a test dependency
# (relevant when the plugin is installed editable inside the tested repo).
_PLUGIN_PACKAGE_DIR = Path(__file__).resolve().parent

_ENV_TRUTHY = ("1", "true", "yes")


def _env_enabled() -> bool:
    return os.environ.get("PYTEST_FILEMON", "").strip().lower() in _ENV_TRUTHY


# ---------------------------------------------------------------- options


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("filemon", "test impact analysis by file monitoring")
    group.addoption(
        "--filemon",
        action="store_true",
        default=False,
        help="Enable filemon TIA: record file-level test dependencies and "
        "select only tests affected by changed files (git diff).",
    )
    group.addoption(
        "--filemon-force",
        action="store_true",
        default=False,
        help="Run all tests but still record dependencies.",
    )
    group.addoption(
        "--filemon-clean",
        action="store_true",
        default=False,
        help="Discard the stored dependency data for this command before running.",
    )


# ---------------------------------------------------------------- early activation


@pytest.hookimpl(wrapper=True)
def pytest_load_initial_conftests(early_config, parser, args):
    # Options are not parsed yet: peek at the raw command line and the env
    # var.  Activating before conftest loading is what makes import-time
    # dependency tracking complete.  (``--filemon`` coming from ini addopts
    # instead of the command line is handled by a late activation fallback in
    # pytest_configure.)
    if "--filemon" in args or _env_enabled():
        root = git_root(Path.cwd())
        if root is not None:
            monitor = FileMonitor(root, exclude_roots=(_PLUGIN_PACKAGE_DIR,))
            if monitor.activate():
                global _EARLY_MONITOR
                _EARLY_MONITOR = monitor
    return (yield)


# ---------------------------------------------------------------- state


class FilemonState:
    """Per-run state of the plugin, attached to the config object."""

    def __init__(self) -> None:
        self.enabled = False
        self.requested = False
        self.notes: list[str] = []
        self.root: Path | None = None
        self.monitor: FileMonitor | None = None
        self.store: GitStore | None = None
        self.shard: Shard | None = None
        self.selection: Selection | None = None
        # Filled during the run.
        self.item_map: dict[str, str | None] = {}  # nodeid -> module rel path
        self.module_files: set[str] = set()
        self.conftest_deps: set[str] = set()
        self.failed: set[str] = set()
        self.finished: set[str] = set()

    # -- setup ----------------------------------------------------------

    def setup(self, config: pytest.Config, early_monitor: FileMonitor | None) -> None:
        self.requested = True
        self.root = git_root(config.rootpath)
        if self.root is None:
            self.notes.append("filemon: not a git repository; running and recording nothing")
            self._drop_early_monitor(early_monitor)
            return

        if early_monitor is not None and early_monitor.root == self.root:
            self.monitor = early_monitor
        else:
            self._drop_early_monitor(early_monitor)
            if early_monitor is None:
                # Late activation (e.g. --filemon came from ini addopts):
                # root conftest imports already happened unwatched, so add
                # the conftest files themselves as import-time deps.
                self.monitor = FileMonitor(
                    self.root, exclude_roots=(_PLUGIN_PACKAGE_DIR,)
                )
                self.monitor.activate()
                self.notes.append(
                    "filemon: monitoring activated late; conftest files added "
                    "as import-time dependencies"
                )
        if self.monitor is None or not self.monitor.active:
            self.notes.append(
                "filemon: could not register a sys.monitoring tool "
                "(already in use?); running all tests without recording"
            )
            self.monitor = None
            return

        self._collect_conftest_deps(config)
        self.store = GitStore(self.root, command_key(config.invocation_params.args, self.root))
        if config.option.filemon_clean:
            self.store.delete()
            self.notes.append(f"filemon: cleared stored data for ref {self.store.ref}")
        data, meta = self.store.load()
        self.shard = load_shard(data, meta)
        self.enabled = True

    def _drop_early_monitor(self, monitor: FileMonitor | None) -> None:
        if monitor is not None:
            monitor.deactivate()

    def _collect_conftest_deps(self, config: pytest.Config) -> None:
        assert self.root is not None
        # Every loaded conftest is a dependency of every test: changing a
        # conftest re-runs everything.  (Normally the monitor captures
        # conftest execution anyway; this covers late activation and empty
        # conftest files whose module still gets imported.)
        for plugin in config.pluginmanager.get_plugins():
            filename = getattr(plugin, "__file__", None)
            if filename and os.path.basename(filename) == "conftest.py":
                rel = to_repo_rel(filename, self.root)
                if rel:
                    self.conftest_deps.add(rel)

    # -- collection / selection ------------------------------------------

    def remember_items(self, items: list[pytest.Item]) -> None:
        assert self.root is not None
        for item in items:
            rel = to_repo_rel(item.path, self.root)
            self.item_map[item.nodeid] = rel
            if rel:
                self.module_files.add(rel)

    def select(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        assert self.store is not None
        if config.option.filemon_force:
            self.selection = Selection(
                [i.nodeid for i in items], [], set(), "--filemon-force: running everything"
            )
            return
        collected = [i.nodeid for i in items]
        shard = self.shard
        changed = None
        if shard is not None:
            changed = self.store.changed_since(shard.base_commit)
        selection = select_tests(collected, shard, changed)
        self.selection = selection
        if selection.deselected:
            keep = [i for i in items if i.nodeid in selection.selected_set]
            drop = [i for i in items if i.nodeid not in selection.selected_set]
            config.hook.pytest_deselected(items=drop)
            items[:] = keep

    # -- recording -------------------------------------------------------

    def record_report(self, report) -> None:
        if report.failed:
            self.failed.add(report.nodeid)
        if report.when in ("call", "teardown"):
            # The test body finished; deps recorded for it are trustworthy.
            self.finished.add(report.nodeid)

    def finish(self, config: pytest.Config) -> None:
        """Merge monitor buckets into per-test deps and store the shard."""
        if self.monitor is None or self.store is None or self.root is None:
            return
        monitor = self.monitor
        monitor.phase = "done"
        monitor.current_nodeid = None
        global_deps = (
            monitor.import_deps
            | (monitor.collect_deps - self.module_files)
            | self.conftest_deps
        )
        old_tests = self.shard.tests if self.shard is not None else {}
        tests: dict[str, list[str]] = {}
        for nodeid, module_rel in self.item_map.items():
            base = set(global_deps)
            if module_rel:
                base.add(module_rel)
            ran = monitor.test_deps.get(nodeid)
            if ran is not None and nodeid in self.finished:
                tests[nodeid] = sorted(base | ran)
            elif nodeid in old_tests:
                # Not run this time (deselected or interrupted): keep the
                # previous dependency set, conservatively unioned with the
                # current import-time deps.
                tests[nodeid] = sorted(base | old_tests[nodeid])
            # else: never ran and no history -> omit, so it is always
            # re-run in the future ("new test").
        data = {"version": 1, "tests": tests, "failed": sorted(self.failed)}
        meta = {
            "version": 1,
            "shard": self.store.shard,
            "base_commit": self.store.head(),
            "created": datetime.now(UTC).isoformat(),
        }
        self.store.save(data, meta)


# ---------------------------------------------------------------- hooks


def pytest_configure(config: pytest.Config) -> None:
    global _EARLY_MONITOR, _STATE
    option = config.option
    enabled = bool(getattr(option, "filemon", False)) or _env_enabled()
    state = FilemonState()
    _STATE = state
    config._filemon_state = state
    if not enabled:
        if _EARLY_MONITOR is not None:
            _EARLY_MONITOR.deactivate()
            _EARLY_MONITOR = None
        return
    state.setup(config, _EARLY_MONITOR)
    _EARLY_MONITOR = None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    state = _state_from(config)
    if state is None or not state.enabled:
        return
    state.remember_items(items)
    state.select(config, items)


@pytest.hookimpl(wrapper=True)
def pytest_collection(session: pytest.Session):
    state = _state_from(session.config)
    if state is not None and state.monitor is not None:
        state.monitor.phase = "collect"
    result = yield
    if state is not None and state.monitor is not None:
        state.monitor.phase = "test"
    return result


def pytest_runtest_logstart(nodeid: str, location) -> None:
    state = _STATE
    if state is not None and state.monitor is not None:
        state.monitor.phase = "test"
        state.monitor.current_nodeid = nodeid


def pytest_runtest_logreport(report) -> None:
    state = _STATE
    if state is not None and state.enabled:
        state.record_report(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus) -> None:
    state = _state_from(session.config)
    if state is None or not state.enabled:
        return
    try:
        state.finish(session.config)
    except Exception as e:  # never break the test run
        state.notes.append(f"filemon: failed to store dependency data: {e!r}")


def pytest_unconfigure(config: pytest.Config) -> None:
    global _EARLY_MONITOR, _STATE
    state = _state_from(config)
    if state is not None and state.monitor is not None:
        state.monitor.deactivate()
    if _EARLY_MONITOR is not None:
        _EARLY_MONITOR.deactivate()
        _EARLY_MONITOR = None
    _STATE = None


def pytest_report_header(config: pytest.Config):
    state = _state_from(config)
    if state is None or not state.enabled or state.store is None:
        return None
    return f"filemon: enabled, shard {state.store.shard} ({state.store.ref})"


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config) -> None:
    state = _state_from(config)
    if state is None or not state.requested:
        return
    reporter = terminalreporter
    reporter.write_sep("=", "filemon")
    if state.store is not None:
        reporter.write_line(f"shard: {state.store.shard} ({state.store.ref})")
    if state.selection is not None:
        sel = state.selection
        reporter.write_line(
            f"selected {len(sel.selected)}/{len(sel.selected) + len(sel.deselected)} tests: {sel.reason}"
        )
    else:
        reporter.write_line("selection: n/a (plugin disabled during this run)")
    for note in state.notes:
        reporter.write_line(note)


def _state_from(config: pytest.Config) -> FilemonState | None:
    return getattr(config, "_filemon_state", None)
