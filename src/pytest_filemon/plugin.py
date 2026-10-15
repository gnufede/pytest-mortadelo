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
5. ``pytest_runtest_protocol`` / ``pytest_runtest_makereport`` (both
   wrappers) -- bracket each test to attribute PY_START events to it, and
   track failures and completion from the outcome reports.
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

# The plugin's own package directory: never recorded as a test dependency
# (relevant when the plugin is installed editable inside the tested repo).
_PLUGIN_PACKAGE_DIR = Path(__file__).resolve().parent

_ENV_TRUTHY = ("1", "true", "yes")


def _env_enabled() -> bool:
    return os.environ.get("PYTEST_FILEMON", "").strip().lower() in _ENV_TRUTHY


def _selectors_active(config: pytest.Config) -> bool:
    """Whether test selectors (-k, -m, --lf/--ff, explicit node ids) are in play.

    When they are, selection is disabled entirely (like testmon's forced
    ``--testmon-noselect``): otherwise an explicitly requested but unaffected
    test would be silently deselected -- or, with collection skipping, its
    file would not even be collected.
    """
    opts = config.option
    if getattr(opts, "keyword", None) or getattr(opts, "markexpr", None):
        return True
    if getattr(opts, "lf", None) or getattr(opts, "failedfirst", None):
        return True
    return any("::" in arg for arg in getattr(config, "args", []))


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
                # Stash on the config (the same instance pytest_configure
                # receives) instead of a module global.
                early_config._filemon_early_monitor = monitor
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
        # Collection plan (computed at the start of the collection phase).
        self.changed: set[str] | None = None
        self.skip_collect_files: set[str] = set()
        self.skipped_test_ids: list[str] = []
        self.selectors_active: bool = False

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

    def plan(self, config: pytest.Config) -> None:
        """Pre-compute the change set and the per-test-file collection plan.

        Runs at the start of the collection phase, before ``pytest_ignore_collect``
        fires: a test file whose recorded tests are *all* unaffected (and that
        is itself unchanged -- its tests always depend on their own module)
        can skip collection entirely, avoiding the module import.  Adding a
        test to a file changes the file, so its tests are selected and it is
        collected: new tests can never be missed.
        """
        if not self.enabled or self.store is None or self.shard is None:
            return
        self.selectors_active = _selectors_active(config)
        if config.option.filemon_force or self.selectors_active:
            return
        try:
            changed = self.store.changed_since(self.shard.base_commit)
        except Exception as e:
            self.notes.append(f"filemon: could not compute changes ({e!r}); running everything")
            return
        self.changed = changed
        if changed is None:
            return
        # Which files still have at least one test that will run?
        file_tests: dict[str, list[str]] = {}
        for nodeid in self.shard.tests:
            file_tests.setdefault(nodeid.split("::")[0], []).append(nodeid)
        for file, nodeids in file_tests.items():
            if any(
                nodeid in self.shard.failed or self.shard.tests[nodeid] & changed
                for nodeid in nodeids
            ):
                continue
            self.skip_collect_files.add(file)
            self.skipped_test_ids.extend(nodeids)
        # pytest exempts explicit command-line paths ("init paths") from
        # pytest_ignore_collect: files given as arguments are always
        # collected, so planning to skip them would only corrupt the report.
        explicit = self._explicit_arg_files(config)
        if explicit & self.skip_collect_files:
            dropped = explicit & self.skip_collect_files
            self.skip_collect_files -= dropped
            self.skipped_test_ids = [
                n for n in self.skipped_test_ids if n.split("::")[0] not in dropped
            ]

    def _explicit_arg_files(self, config: pytest.Config) -> set[str]:
        """Repo-relative files explicitly given as pytest arguments."""
        assert self.root is not None
        out: set[str] = set()
        for arg in getattr(config, "args", []):
            p = Path(str(arg).split("::")[0])
            if p.is_file():
                rel = to_repo_rel(p, self.root)
                if rel:
                    out.add(rel)
        return out

    def remember_items(self, items: list[pytest.Item]) -> None:
        assert self.root is not None
        for item in items:
            rel = to_repo_rel(item.path, self.root)
            self.item_map[item.nodeid] = rel
            if rel:
                self.module_files.add(rel)

    def select(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        """Select tests among the *collected* items (files whose tests are
        all unaffected were already skipped at collection time)."""
        assert self.store is not None
        collected = [i.nodeid for i in items]
        shard = self.shard
        if shard is None:
            self.selection = Selection(
                collected, [], set(),
                "no usable stored data for this command; running and recording everything",
            )
            return
        if config.option.filemon_force:
            self.selection = Selection(
                collected, [], set(), "--filemon-force: running everything"
            )
            return
        if self.selectors_active:
            self.selection = Selection(
                collected, [], set(),
                "test selectors (-k/-m/--lf/explicit node ids) present: running everything",
            )
            return
        selection = select_tests(collected, shard, self.changed)
        if self.skipped_test_ids:
            # Only report tests whose files were genuinely not collected;
            # a planned skip can be overridden by pytest (e.g. a file given
            # as an explicit argument), in which case those tests are
            # already in the collected deselected list.
            collected_set = set(collected)
            actually_skipped = [
                n for n in self.skipped_test_ids if n not in collected_set
            ]
            skipped_files = {
                f
                for f in self.skip_collect_files
                if not any(n.startswith(f + "::") for n in collected_set)
            }
            if actually_skipped or skipped_files:
                selection.deselected.extend(actually_skipped)
                selection.reason += (
                    f"; skipped collection of {len(skipped_files)} file(s)"
                )
        self.selection = selection
        if selection.deselected:
            selected_set = set(selection.selected)
            keep = [i for i in items if i.nodeid in selected_set]
            drop = [i for i in items if i.nodeid not in selected_set]
            if drop:
                config.hook.pytest_deselected(items=drop)
                items[:] = keep

    # -- recording -------------------------------------------------------

    def record_report(self, report) -> None:
        """Track failures and completion from a test's outcome reports
        (TestReport: setup/call/teardown, via pytest_runtest_makereport)."""
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
        # Tests in files whose collection was skipped never ran and are not
        # in item_map: keep their recorded dependencies, otherwise the shard
        # would silently lose them.
        for nodeid, old_deps in old_tests.items():
            if nodeid in tests:
                continue
            file = nodeid.split("::")[0]
            if file in self.skip_collect_files:
                base = set(global_deps)
                if file.endswith(".py"):
                    base.add(file)
                tests[nodeid] = sorted(base | old_deps)
            # else: the file was collected but the test is gone -> drop
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
    option = config.option
    enabled = bool(getattr(option, "filemon", False)) or _env_enabled()
    early_monitor = getattr(config, "_filemon_early_monitor", None)
    config._filemon_early_monitor = None
    state = FilemonState()
    config._filemon_state = state
    if not enabled:
        if early_monitor is not None:
            early_monitor.deactivate()
        return
    state.setup(config, early_monitor)


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
    if state is not None:
        try:
            state.plan(session.config)
        except Exception as e:  # never break the run
            state.notes.append(f"filemon: planning failed ({e!r}); running everything")
    if state is not None and state.monitor is not None:
        state.monitor.phase = "collect"
    result = yield
    if state is not None and state.monitor is not None:
        state.monitor.phase = "test"
    return result


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    """Skip collecting test files whose recorded tests are all unaffected.

    testmon-style optimization: unchanged, fully-deselected test modules are
    not even imported.  Safe because a test always depends on its own module
    file: any edit (including adding tests) selects the file's tests, and a
    brand-new file has no recorded tests and is always collected.
    """
    state = _state_from(config)
    if state is None or not state.skip_collect_files:
        return None
    if collection_path.suffix != ".py":
        return None
    rel = to_repo_rel(collection_path, state.root)
    if rel is not None and rel in state.skip_collect_files:
        return True
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem):
    """Bracket each test's whole runtest protocol (setup/call/teardown).

    This is the canonical per-test lifecycle hook; ``logstart`` is only a
    notification hook and does not carry the item or config.  Setting the
    attribution bucket here is also what re-arms the silenced code objects
    (via the monitor's bucket-change hook) for the new test.
    """
    state = _state_from(item.config)
    if state is not None and state.monitor is not None:
        state.monitor.phase = "test"
        state.monitor.current_nodeid = item.nodeid
    result = yield
    # current_nodeid deliberately stays set after the protocol: teardown of
    # session-scoped fixtures and other inter-test machinery still attribute
    # to the test that just ran, until the next protocol starts.
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    """Observe each phase's outcome report (firstresult wrapper).

    The canonical hook for acting on outcomes; ``logreport`` is a
    notification hook without the item/config.
    """
    report = yield
    state = _state_from(item.config)
    if state is not None and state.enabled and report is not None:
        state.record_report(report)
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus) -> None:
    state = _state_from(session.config)
    if state is None or not state.enabled:
        return
    if exitstatus == pytest.ExitCode.INTERRUPTED:
        state.notes.append(
            "filemon: run interrupted; tests that never finished will be "
            "re-run next time"
        )
    try:
        state.finish(session.config)
    except Exception as e:  # never break the test run
        state.notes.append(f"filemon: failed to store dependency data: {e!r}")


def pytest_unconfigure(config: pytest.Config) -> None:
    state = _state_from(config)
    early_monitor = getattr(config, "_filemon_early_monitor", None)
    if early_monitor is not None:
        early_monitor.deactivate()
        config._filemon_early_monitor = None
    if state is not None and state.monitor is not None:
        state.monitor.deactivate()


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
