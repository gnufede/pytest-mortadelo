"""pytest-less: test impact analysis.

Records per-test block-level dependencies via ``sys.monitoring`` and
selects only tests affected by changed files.  Dependencies are stored
as JSON under ``.pytest_less/`` in the repository; ``rm -rf .pytest_less``
resets the cache.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from .blocks import WILDCARD, Block
from .fingerprint import build_snapshot
from .monitoring import FileMonitor
from .selection import Shard, load_shard, select_tests
from .store import Store
from .util import git_root, to_repo_rel

_ENV_TRUTHY = {"1", "true", "yes", "on"}

_DATA_VERSION = 2  # must match selection.py


def _env_enabled() -> bool:
    return os.environ.get("PYTEST_LESS", "").strip().lower() in _ENV_TRUTHY


def _selectors_active(config: pytest.Config) -> bool:
    """Whether test selectors (-k, -m, --lf, explicit node ids) are in play.

    When they are, selection is disabled entirely: an explicitly
    requested but unaffected test must not be silently deselected."""
    opts = config.option
    if getattr(opts, "keyword", None) or getattr(opts, "markexpr", None):
        return True
    if getattr(opts, "lf", None) or getattr(opts, "failedfirst", None):
        return True
    return any("::" in arg for arg in getattr(config, "args", []))




# ---------------------------------------------------------------- options


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("less", "test impact analysis")
    group.addoption(
        "--less",
        action="store_true",
        default=False,
        help="Enable less: record block-level test dependencies and "
        "select only tests affected by changed files (file fingerprints).",
    )
    group.addoption(
        "--less-force",
        action="store_true",
        default=False,
        help="Run all tests, but still record dependencies.",
    )
    group.addoption(
        "--less-clean",
        action="store_true",
        default=False,
        help="Discard the stored data before running.",
    )


# ---------------------------------------------------------------- helpers


def _fast_conditions(store, meta):
    """``(changed, fast, pure)`` from meta.json alone.

    If no file in ``dep_files`` changed, HEAD did not move, and nothing
    failed, no recorded dependency can be stale and the whole monitoring
    machinery is not needed at all.
    """
    if meta is None:
        return None, False, False
    dep_files = meta.get("dep_files")
    if not isinstance(dep_files, list) or meta.get("failed"):
        return None, False, False
    changed = store.changed_files(meta.get("base_commit"))
    if changed is None or changed:
        return changed, False, False
    if store.head() != meta.get("base_commit"):
        return changed, False, False
    return changed, True, True


def _shared_git_root(config, cwd: Path) -> Path | None:
    root = getattr(config, "_less_git_root", None)
    if root is not None:
        return root
    root = git_root(cwd)
    if root is not None:
        config._less_git_root = root
    return root


def _early_probe_pure_fast(args, invocation_args, early_config) -> bool:
    """Whether the run can skip the monitor entirely (no-change rerun)."""
    if "--less-force" in args or "--less-clean" in args:
        return False
    for flag in ("--lf", "--ff", "-k", "-m"):
        if flag in args:
            return False
    root = _shared_git_root(early_config, Path.cwd())
    if root is None:
        return False
    store = Store(root)
    _, fast, pure = _fast_conditions(store, store.load_meta())
    return fast and pure


# ---------------------------------------------------------------- hooks


def pytest_load_initial_conftests(early_config, parser, args):
    """Arm the monitor before conftest loading (import-time tracking)."""
    if ("--less" in args or _env_enabled()) and not _early_probe_pure_fast(
        args, early_config.invocation_params.args, early_config
    ):
        root = _shared_git_root(early_config, Path.cwd())
        if root is not None:
                monitor = FileMonitor(root)
                if monitor.activate():
                    early_config._less_early_monitor = monitor
    return (yield)


# ---------------------------------------------------------------- state


class LessState:
    """Per-session recording and selection state."""

    def __init__(self) -> None:
        self.enabled = False
        self.requested = False
        self.root: Path | None = None
        self.store: Store | None = None
        self.monitor: FileMonitor | None = None
        self.shard: Shard | None = None
        self.selection = None
        self.failed: set[str] = set()
        self.finished: set[str] = set()
        self.item_map: dict[str, str | None] = {}
        self.module_files: set[str] = set()
        self.conftest_deps: set[str] = set()
        self.fixture_deps: dict[str, frozenset[Block]] = {}
        self._fixture_names: dict[str, frozenset[Block]] = {}
        self._fixture_closures: dict[tuple[str, ...], frozenset[Block]] = {}
        self.notes: list[str] = []
        self._meta = None
        self.fast = False
        self.fast_pure = False
        self.fast_recorded: set[str] = set()
        self.changed: set[str] | None = None

    # -- setup ----------------------------------------------------------

    def setup(self, config: pytest.Config, early_monitor: FileMonitor | None) -> None:
        self.requested = True
        self.root = _shared_git_root(config, config.rootpath)
        if self.root is None:
            self.notes.append("less: not a git repository; running and recording nothing")
            self._drop_early_monitor(early_monitor)
            return
        self.store = Store(self.root)
        self._collect_conftest_deps(config)
        if config.option.less_clean:
            self.store.delete()
            self.notes.append("less: cleared stored data")
        meta = self.store.load_meta()
        fast = self._try_fast_path(config, meta)
        if not fast:
            self.shard = load_shard(self.store.load_data(), meta)
        if not (fast and self.fast_pure):
            self._activate_late(config, early_monitor)

    def _try_fast_path(self, config: pytest.Config, meta) -> bool:
        """Decide the run without reading data.json."""
        self._meta = meta
        if meta is None or config.option.less_force:
            return False
        if _selectors_active(config):
            return False
        changed, fast, pure = _fast_conditions(self.store, meta)
        if changed is not None:
            self.changed = changed
        if not fast:
            return False
        nodeids = [n for n in meta.get("nodeids", []) if isinstance(n, str)]
        self.fast_recorded = set(nodeids)
        self.fast = True
        self.fast_pure = pure
        self.notes.append("less: no recorded dependency changed; fast path")
        return True

    def _activate_late(self, config, early_monitor):
        """Arm the monitor now (recovery path when the early probe skipped)."""
        self.monitor = early_monitor
        if self.monitor is None or not self.monitor.active:
            self.monitor = FileMonitor(self.root)
            self.monitor.activate()

    def _drop_early_monitor(self, monitor):
        if monitor is not None:
            monitor.deactivate()

    def _collect_conftest_deps(self, config: pytest.Config) -> None:
        assert self.root is not None
        for plugin in config.pluginmanager.get_plugins():
            filename = getattr(plugin, "__file__", None)
            if filename and os.path.basename(filename) == "conftest.py":
                rel = to_repo_rel(filename, self.root)
                if rel:
                    self.conftest_deps.add(rel)
        # The active config file: parsed before the earliest hook, so no
        # event ever sees the read.  Any change re-runs everything.
        inifile = getattr(config, "inifile", None)
        if inifile is not None:
            rel = to_repo_rel(os.fspath(inifile), self.root)
            if rel:
                self.conftest_deps.add(rel)

    # -- collection / selection -------------------------------------------

    def plan(self, config: pytest.Config) -> None:
        """Compute the changed set before collection starts."""
        if not self.enabled or self.store is None or self.shard is None:
            return
        if config.option.less_force or _selectors_active(config):
            return
        try:
            self.changed = self.store.changed_files(self.shard.base_commit)
        except Exception as e:
            self.notes.append(f"less: could not compute changes ({e!r}); running everything")

    def _new_maps(self, changed: set[str]) -> dict:
        """Current block tables of the changed .py files."""
        maps: dict = {}
        if self.monitor is None:
            return maps
        for f in changed:
            if f.endswith(".py"):
                maps[f] = self.monitor.file_blocks(f)
        return maps

    def repo_rel(self, path) -> str | None:
        assert self.root is not None
        if self.monitor is not None:
            return self.monitor.repo_rel(os.fspath(path))
        return to_repo_rel(path, self.root)

    def remember_items(self, items: list[pytest.Item]) -> None:
        assert self.root is not None
        rel_of: dict = {}
        for item in items:
            path = item.path
            if path not in rel_of:
                rel_of[path] = to_repo_rel(path, self.root)
            rel = rel_of[path]
            self.item_map[item.nodeid] = rel
            if rel:
                self.module_files.add(rel)
            if self.monitor is not None:
                deps = self._fixture_deps(item)
                if deps:
                    self.fixture_deps[item.nodeid] = deps

    def _fixture_deps(self, item: pytest.Item) -> frozenset[Block]:
        """Blocks of the >function-scoped fixtures this test consumes."""
        try:
            arg2 = item.session._fixturemanager._arg2fixturedefs
        except AttributeError:
            return frozenset()
        key = tuple(getattr(item, "fixturenames", ()))
        hit = self._fixture_closures.get(key)
        if hit is None:
            deps: set[Block] = set()
            for name in key:
                ndeps = self._fixture_names.get(name)
                if ndeps is None:
                    ndeps = self._fixture_name_deps(arg2, name)
                    self._fixture_names[name] = ndeps
                deps |= ndeps
            hit = frozenset(deps)
            self._fixture_closures[key] = hit
        return hit

    def _fixture_name_deps(self, arg2, name: str) -> frozenset[Block]:
        deps: set[Block] = set()
        for fdef in arg2.get(name) or ():
            try:
                if getattr(fdef, "scope", "function") == "function":
                    continue
                code = getattr(fdef.func, "__code__", None)
                if code is None:
                    continue
                rel = self.monitor.repo_rel(code.co_filename)
                if rel is None:
                    continue
                deps.add(self._fixture_dep(rel, code))
            except Exception:  # noqa: S112
                continue
        return frozenset(deps)

    def _fixture_dep(self, rel: str, code) -> Block:
        fb = self.monitor.file_blocks(rel)
        if fb is None:
            return (rel, WILDCARD, WILDCARD)
        block_id, content = fb.resolve(
            getattr(code, "co_qualname", code.co_name), code.co_firstlineno
        )
        if block_id.split("~")[0] != getattr(code, "co_qualname", code.co_name):
            return (rel, WILDCARD, WILDCARD)
        return (rel, block_id, content)

    def select(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        """Select tests among the collected items."""
        assert self.store is not None
        collected = [i.nodeid for i in items]
        if self.fast:
            recorded = self.fast_recorded
            self.selection = _Selection(
                [n for n in collected if n not in recorded],
                [n for n in collected if n in recorded],
                "no recorded dependency changed; fast path",
            )
        elif self.shard is None:
            self.selection = _Selection(collected, [], "no stored data; running everything")
            return
        elif config.option.less_force:
            self.selection = _Selection(collected, [], "--less-force: running everything")
            return
        elif _selectors_active(config):
            self.selection = _Selection(collected, [], "selectors present: running everything")
            return
        else:
            maps = self._new_maps(self.changed or set())
            self.selection = select_tests(collected, self.shard, self.changed, maps)
        sel = self.selection
        if sel.deselected:
            selected_set = set(sel.selected)
            keep = [i for i in items if i.nodeid in selected_set]
            drop = [i for i in items if i.nodeid not in selected_set]
            if drop:
                config.hook.pytest_deselected(items=drop)
                items[:] = keep

    # -- recording ---------------------------------------------------------

    def record_report(self, report) -> None:
        if report.failed:
            self.failed.add(report.nodeid)
        if report.when in ("call", "teardown"):
            self.finished.add(report.nodeid)

    def finish(self, config: pytest.Config, interrupted: bool = False) -> None:
        if self.store is None or self.root is None:
            return
        if self.fast and not self.finished and not self.failed:
            self.notes.append("less: nothing ran; stored data left unchanged")
            return
        if self.fast and self.monitor is None:
            self._store_failures_only()
            return
        monitor = self.monitor
        if monitor is not None:
            monitor.phase = "done"
            monitor.current_nodeid = None
        if (
            monitor is not None
            and not self.fast
            and self.shard is not None
            and not self.finished
            and not self.failed
            and not interrupted
        ):
            self.notes.append("less: nothing ran; stored data left unchanged")
            return
        if self.fast:
            self.shard = load_shard(self.store.load_data(), self._meta)
        if monitor is None:
            return
        old_shard = self.shard
        import_blocks, module_collect = self._shared_blocks(monitor, old_shard)
        tests = self._merge_tests(monitor, module_collect, old_shard, interrupted)
        always = self._marks(monitor, tests, old_shard)
        data, meta, dep_files = _shard_documents(
            self.store, tests, import_blocks, self.failed, always
        )
        self._save(config, monitor, data, meta, dep_files, bool(always))

    def _store_failures_only(self) -> None:
        data = self.store.load_data() or {
            "version": _DATA_VERSION, "blocks": [], "tests": {}, "import_deps": [],
        }
        data["failed"] = sorted(self.failed)
        meta = dict(self._meta or {})
        meta.update(
            failed=sorted(self.failed),
            base_commit=self.store.head(),
            created=datetime.now(UTC).isoformat(),
        )
        self.store.save(data, meta)
        self.notes.append("less: fast path; failure list updated")

    def _shared_blocks(self, monitor, old_shard):
        """``(import-time blocks, collect-time blocks per test module)``."""
        module_collect: dict[str, set[Block]] = {}
        import_blocks: set[Block] = set(monitor.import_deps)
        for block in monitor.collect_deps:
            rel = block[0]
            if rel in self.module_files:
                module_collect.setdefault(rel, set()).add(block)
            else:
                import_blocks.add(block)
        for rel in self.conftest_deps:
            import_blocks.add(monitor.module_dep(rel))
        if old_shard is not None:
            import_blocks |= old_shard.import_deps
        return import_blocks, module_collect

    def _merge_tests(self, monitor, module_collect, old_shard, interrupted):
        old_tests = old_shard.tests if old_shard is not None else {}
        tests: dict[str, set[Block]] = {}
        for nodeid, module_rel in self.item_map.items():
            own = monitor.test_deps.get(nodeid)
            if own is not None and nodeid in self.finished:
                extra = module_collect.get(module_rel, set()) if module_rel else set()
                if nodeid in self.fixture_deps:
                    extra = extra | self.fixture_deps[nodeid]
                tests[nodeid] = own | extra if extra else own
            elif nodeid in old_tests:
                tests[nodeid] = set(old_tests[nodeid])
        for nodeid, old_deps in old_tests.items():
            if nodeid not in tests and nodeid in self.item_map:
                tests[nodeid] = set(old_deps)
        if interrupted or (self.finished and len(self.finished) < len(self.item_map)):
            for nodeid in self.item_map:
                if nodeid not in self.finished:
                    tests.setdefault(nodeid, set())
                    self.failed.add(nodeid)
        for nodeid in monitor.spawn_tests:
            if nodeid in self.item_map:
                tests[nodeid] = set()
        return tests

    def _marks(self, monitor, tests, old_shard):
        alive = set(tests) | set(self.item_map)
        observed = self.finished
        old_always = old_shard.always if old_shard is not None else set()
        return (set(monitor.spawn_tests) | (old_always - observed)) & alive

    def _save(self, config, monitor, data, meta, dep_files, spawners):
        """Write the JSON files (with fingerprints) to .pytest_less/."""
        digests = {}
        tracked = self.store.tracked_files(30000) if spawners else None
        snapshot = build_snapshot(
            self.root, dep_files, digests,
            previous=self.store.load_files(),
            extra_files=tracked,
        )
        self.store.save(data, meta, files=snapshot)


def _shard_documents(store, tests, import_blocks, failed, always):
    """``(data, meta, dep_files)`` for the .pytest_less/ files."""
    distinct: set[Block] = set(import_blocks)
    for deps in tests.values():
        distinct.update(deps)
    dep_files = {b[0] for b in distinct}
    ordered = sorted(distinct)
    lookup = {b: i for i, b in enumerate(ordered)}.__getitem__
    data = {
        "version": _DATA_VERSION,
        "blocks": [list(b) for b in ordered],
        "tests": {nid: sorted(map(lookup, tests[nid])) for nid in sorted(tests)},
        "import_deps": sorted(map(lookup, import_blocks)),
        "failed": sorted(failed),
        "always": sorted(always),
    }
    meta = {
        "version": 1,
        "base_commit": store.head(),
        "created": datetime.now(UTC).isoformat(),
        "failed": sorted(failed),
        "nodeids": sorted(tests),
        "dep_files": sorted(dep_files),
        "always": sorted(always),
    }
    return data, meta, dep_files


class _Selection:
    """Outcome of the selection for one run."""

    __slots__ = ("deselected", "reason", "selected")

    def __init__(self, selected, deselected, reason=""):
        self.selected = selected
        self.deselected = deselected
        self.reason = reason


# ---------------------------------------------------------------- hooks


def pytest_configure(config: pytest.Config) -> None:
    option = config.option
    enabled = bool(getattr(option, "less", False)) or _env_enabled()
    early_monitor = getattr(config, "_less_early_monitor", None)
    config._less_early_monitor = None
    state = LessState()
    config._less_state = state
    if not enabled:
        if early_monitor is not None:
            early_monitor.deactivate()
        return
    state.enabled = True
    state.setup(config, early_monitor)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    state = _state_from(config)
    if state is None or not state.enabled:
        return
    state.remember_items(items)
    state.select(config, items)


@pytest.hookimpl(wrapper=True)
def pytest_collection(session):
    state = _state_from(session.config)
    if state is not None:
        try:
            state.plan(session.config)
        except Exception as e:
            state.notes.append(f"less: planning failed ({e!r}); running everything")
    if state is not None and state.monitor is not None:
        state.monitor.phase = "collect"
    result = yield
    if state is not None and state.monitor is not None:
        state.monitor.phase = "test"
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    state = _state_from(item.config)
    if state is not None and state.monitor is not None:
        state.monitor.phase = "test"
        state.monitor.current_nodeid = item.nodeid
    result = yield
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    state = _state_from(item.config)
    if state is not None and state.enabled and report is not None:
        state.record_report(report)
    return report


def pytest_sessionfinish(session, exitstatus):
    state = _state_from(session.config)
    if state is None or not state.enabled:
        return
    try:
        state.finish(session.config, interrupted=exitstatus == pytest.ExitCode.INTERRUPTED)
    except Exception as e:
        state.notes.append(f"less: failed to store dependency data: {e!r}")


def pytest_unconfigure(config):
    state = _state_from(config)
    early_monitor = getattr(config, "_less_early_monitor", None)
    if early_monitor is not None:
        early_monitor.deactivate()
        config._less_early_monitor = None
    if state is not None and state.monitor is not None:
        state.monitor.deactivate()


def pytest_report_header(config):
    state = _state_from(config)
    if state is None or not state.enabled or state.store is None:
        return None
    return f"less: enabled, data in {state.store.dir}"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = _state_from(config)
    if state is None or not state.requested:
        return
    reporter = terminalreporter
    reporter.write_sep("=", "less")
    if state.store is not None:
        reporter.write_line(f"data: {state.store.dir}")
    if state.selection is not None:
        sel = state.selection
        reporter.write_line(
            f"selected {len(sel.selected)}/{len(sel.selected) + len(sel.deselected)} tests: {sel.reason}"
        )
    for note in state.notes:
        reporter.write_line(note)


def _state_from(config) -> LessState | None:
    return getattr(config, "_less_state", None)
