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

from .blocks import Block
from .gitstore import GitStore
from .monitoring import FileMonitor
from .selection import (
    DATA_VERSION,
    Selection,
    Shard,
    load_shard,
    select_tests,
    stale_dep,
)
from .util import _EXCLUDED_SEGMENTS, command_key, git_root, to_repo_rel

_ENV_TRUTHY = ("1", "true", "yes")


def _env_enabled() -> bool:
    return os.environ.get("PYTEST_FILEMON", "").strip().lower() in _ENV_TRUTHY


def _sync_remote(config: pytest.Config) -> tuple[str, str] | None:
    """The (remote, mode) to sync shards with, if any.

    ``PYTEST_FILEMON_SYNC`` takes precedence over the ``filemon_sync`` ini
    option.  Both accept a remote name (pull *and* push), ``1``/``true``/
    ``yes`` meaning ``origin``, or ``pull:<remote>`` / ``push:<remote>`` for
    one-directional sync.  ``pull:`` is the CI mode: it consumes shards
    with the same read-only credentials the checkout already uses, needing
    no write token at all.
    """
    value = os.environ.get("PYTEST_FILEMON_SYNC")
    if value is None:
        value = config.getini("filemon_sync") or ""
    value = value.strip()
    if not value or value.lower() in ("0", "false", "no"):
        return None
    if value.lower() in ("1", "true", "yes"):
        return ("origin", "both")
    mode = "both"
    remote = value
    left, sep, right = value.partition(":")
    if sep and left.lower() in ("pull", "push") and right.strip():
        mode, remote = left.lower(), right.strip()
    return (remote, mode)


def _fast_conditions(store, meta) -> tuple[set[str] | None, bool, bool]:
    """Probe the run using only meta.json (the first-level cache).

    Returns (changed-file set, fast-path eligible, monitor-less eligible).
    Fast path: no recorded dependency can be stale.  Monitor-less ("pure"):
    additionally, no unrecorded .py file exists that pytest could collect
    tests from -- if one did, it might run and need recording, so the
    monitor must stay armed.
    """
    if (
        meta is None
        or not isinstance(meta.get("nodeids"), list)
        or not isinstance(meta.get("dep_files"), list)
    ):
        return None, False, False
    try:
        changed = store.changed_since(meta.get("base_commit"))
    except Exception:
        return None, False, False
    if changed is None:
        return changed, False, False
    if meta.get("failed") or changed & {
        f for f in meta["dep_files"] if isinstance(f, str)
    }:
        return changed, False, False
    dep_files = {f for f in meta["dep_files"] if isinstance(f, str)}
    candidates = {
        f
        for f in changed - dep_files
        if f.endswith(".py")
        and not _EXCLUDED_SEGMENTS.intersection(Path(f).parts)
    }
    return changed, True, not candidates


def _early_probe_pure_fast(args: list[str]) -> bool:
    """Heuristic pre-configure check: can we skip monitor activation?

    Conservative in the right direction: any raw-argument that could mean
    selectors, discards or forced runs makes the probe fail (the monitor is
    activated normally).  A wrong probe verdict only costs the early
    import-time capture -- pytest_configure re-runs the exact check and
    late-activates the monitor when needed.
    """
    if any(
        flag in args
        for flag in ("--filemon-force", "--filemon-clean", "--lf", "--ff",
                     "--last-failed", "--failed-first")
    ) or any(
        a.startswith(("-k", "-m", "--lf", "--ff", "-x", "--maxfail"))
        or "::" in a
        for a in args
    ):
        return False
    root = git_root(Path.cwd())
    if root is None:
        return False
    store = GitStore(root, command_key(args, root))
    _, fast, pure = _fast_conditions(store, store.load_meta())
    return fast and pure


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
    parser.addini(
        "filemon_sync",
        help="git remote to sync the dependency shard with (e.g. 'origin'): "
        "pulled before selection, pushed after recording",
        default="",
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
        if _early_probe_pure_fast(args):
            # No-change rerun (decided from meta.json alone, before any
            # conftest is imported): no recorded dependency can be stale
            # and nothing new can run, so the whole monitoring machinery is
            # not needed at all.  pytest_configure re-checks with parsed
            # options and can still activate it late if this was wrong.
            early_config._filemon_probe_pure = True
        else:
            root = git_root(Path.cwd())
            if root is not None:
                monitor = FileMonitor(root)
                if monitor.activate():
                    # Stash on the config (the same instance
                    # pytest_configure receives) instead of a module global.
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
        # Shard sync (see _sync_remote).
        self.sync_remote: str | None = None
        self.sync_mode: str = "both"
        # No-change fast path (meta.json is the first-level cache).
        self.fast = False
        self.fast_recorded: set[str] = set()
        self.fast_pure = False  # fast path AND the monitor can stay off
        self._meta: dict | None = None
        # Current block tables of the changed files (selection input).
        self.new_maps: dict = {}

    # -- setup ----------------------------------------------------------

    def setup(self, config: pytest.Config, early_monitor: FileMonitor | None) -> None:
        self.requested = True
        self.root = git_root(config.rootpath)
        if self.root is None:
            self.notes.append("filemon: not a git repository; running and recording nothing")
            self._drop_early_monitor(early_monitor)
            return

        self._collect_conftest_deps(config)
        self.store = GitStore(self.root, command_key(config.invocation_params.args, self.root))
        sync = _sync_remote(config)
        if sync is not None:
            self.sync_remote, self.sync_mode = sync
        if config.option.filemon_clean:
            self.store.delete()
            self.notes.append(f"filemon: cleared stored data for ref {self.store.ref}")
        elif self.sync_remote and self.sync_mode in ("pull", "both"):
            # Pull before loading the shard: a newer record on the remote is
            # used for this run's selection.  (Skipped under --filemon-clean,
            # so "discard" really discards.)
            try:
                self.notes.append(f"filemon: {self.store.sync_pull(self.sync_remote)}")
            except Exception as e:  # never break the run
                self.notes.append(f"filemon: sync pull failed ({e!r})")
        meta = self.store.load_meta()
        fast = self._try_fast_path(config, meta)
        if not fast:
            # Slow path: load the full dependency blob.
            self.shard = load_shard(self.store.load_data(), meta)
        if not (fast and self.fast_pure):
            # The monitor is only armed when something might actually be
            # recorded (the pure fast path runs nothing worth recording).
            if early_monitor is not None and early_monitor.root == self.root:
                self.monitor = early_monitor
            else:
                self._drop_early_monitor(early_monitor)
                # Late activation (also the recovery path when the early
                # probe skipped arming but this run turned out to need it,
                # and --filemon from ini addopts): root conftest imports
                # already happened unwatched, so add the conftest files
                # themselves as import-time deps.
                self.monitor = FileMonitor(self.root)
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
        else:
            # Pure fast path: drop any early monitor the probe did not
            # prevent (probe false-positives) and stay disarmed.
            self._drop_early_monitor(early_monitor)
        self.enabled = True

    def _try_fast_path(self, config: pytest.Config, meta) -> bool:
        """Decide the run without reading data.json at all.

        meta.json doubles as the first-level cache: it carries the recorded
        node ids, the failed list, and -- crucially -- ``dep_files``, the set
        of every file any recorded dependency references.  git itself is the
        cache-validity oracle: if no file in that set changed since the
        recorded base commit (working tree + untracked), no recorded
        dependency can be stale, so the 40MB data blob is never read and no
        per-test block expansion ever happens.  Runs everything through the
        normal path when the summary is absent (old shards), anything
        relevant changed, or tests failed (they must re-run).
        """
        self._meta = meta
        if meta is None or config.option.filemon_force:
            return False
        if _selectors_active(config):
            self.selectors_active = True
            return False
        changed, fast, pure = _fast_conditions(self.store, meta)
        if changed is not None:
            self.changed = changed
        if not fast:
            return False
        # Nothing any recorded test depends on changed.
        nodeids = [n for n in meta["nodeids"] if isinstance(n, str)]
        self.fast = True
        self.fast_pure = pure
        self.fast_recorded = set(nodeids)
        explicit = self._explicit_arg_files(config)
        self.skip_collect_files = {n.split("::")[0] for n in nodeids} - explicit
        self.skipped_test_ids = [
            n for n in nodeids if n.split("::")[0] in self.skip_collect_files
        ]
        base = (meta.get("base_commit") or "?")[:8]
        how = (
            "fast path (monitoring not armed, dependency data not read)"
            if self.fast_pure
            else "fast path (dependency data not read)"
        )
        self.notes.append(f"filemon: no recorded dependency changed since {base}; {how}")
        return True

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
        self.new_maps = self._new_maps(changed)
        # If any shared import-time dependency went stale, every test runs:
        # nothing is skippable.
        if any(stale_dep(b, changed, self.new_maps) for b in self.shard.import_deps):
            return
        # Which files still have at least one test that will run?  A test
        # runs if it previously failed or any of its recorded blocks went
        # stale in the changed files.
        file_tests: dict[str, list[str]] = {}
        for nodeid in self.shard.tests:
            file_tests.setdefault(nodeid.split("::")[0], []).append(nodeid)
        for file, nodeids in file_tests.items():
            if any(
                nodeid in self.shard.failed
                or any(stale_dep(b, changed, self.new_maps) for b in self.shard.tests[nodeid])
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

    def _new_maps(self, changed: set[str]) -> dict:
        """Current block tables of the changed .py files (selection input).

        Files that no longer parse (deleted, mid-edit) map to None, which
        the stale check treats as "changed in a way we cannot resolve".
        """
        maps: dict = {}
        if self.monitor is None:
            return maps
        for f in changed:
            if f.endswith(".py"):
                maps[f] = self.monitor.file_blocks(f)
        return maps

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

    def repo_rel(self, path) -> str | None:
        """Repo-relative path, via the monitor's per-instance cache when
        available (hot during collection traversal)."""
        assert self.root is not None
        if self.monitor is not None:
            return self.monitor.repo_rel(os.fspath(path))
        return to_repo_rel(path, self.root)

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
        if self.fast:
            # Decision was already made from meta.json: recorded tests are
            # deselected (nothing they depend on changed); anything not in
            # the recorded set is new and runs.
            recorded = self.fast_recorded
            selection = Selection(
                [n for n in collected if n not in recorded],
                [n for n in collected if n in recorded],
                set(self.changed or ()),
                "no recorded dependency changed; fast path",
            )
        elif self.shard is None:
            self.selection = Selection(
                collected, [], set(),
                "no usable stored data for this command; running and recording everything",
            )
            return
        elif config.option.filemon_force:
            self.selection = Selection(
                collected, [], set(), "--filemon-force: running everything"
            )
            return
        elif self.selectors_active:
            self.selection = Selection(
                collected, [], set(),
                "test selectors (-k/-m/--lf/explicit node ids) present: running everything",
            )
            return
        else:
            selection = select_tests(collected, self.shard, self.changed, self.new_maps)
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
        """Merge monitor buckets into per-test deps and store the shard.

        Block-level model (data version 2): each test keeps the blocks that
        executed while it ran (plus the collect-time blocks of its own test
        module); import-time blocks (module skeletons of everything imported
        before the tests, conftests, and the bodies of anything that
        executed during import) are shared by every test and stored once.
        """
        if self.store is None or self.root is None:
            return
        if self.fast and not self.finished and not self.failed:
            # Fast path and nothing ran: the stored shard is still exactly
            # valid (same base commit, same content) -- leave it untouched.
            self.notes.append("filemon: nothing ran; stored shard left unchanged")
            return
        if self.fast and self.monitor is None:
            # Pure fast path (the monitor was never armed) but something
            # ran or failed.  No dependency can have changed -- that is why
            # we took the fast path -- so only the failure list can differ;
            # deps of anything that ran without a monitor stay as recorded
            # (or, for unrecorded tests, absent: they simply re-run next
            # time, the conservative "new test" behavior).
            data = self.store.load_data() or {
                "version": DATA_VERSION, "blocks": [], "tests": {}, "import_deps": [],
            }
            data["failed"] = sorted(self.failed)
            meta = dict(self._meta or {})
            meta.update(
                failed=sorted(self.failed),
                base_commit=self.store.head(),
                created=datetime.now(UTC).isoformat(),
            )
            self.store.save(data, meta)
            self.notes.append(
                "filemon: monitor was not armed (fast path); failure list "
                "updated, dependencies left as recorded"
            )
            return
        if self.fast:
            # Something ran (a new test, or a previously failed one):
            # lazily load the full data now to preserve everything else.
            self.shard = load_shard(self.store.load_data(), self._meta)
        monitor = self.monitor
        if monitor is None:
            return
        monitor.phase = "done"
        monitor.current_nodeid = None

        # Partition collect-phase blocks: those executed in test-module
        # files belong to that module's tests only; everything else is a
        # shared import-time dependency.
        module_collect: dict[str, set[Block]] = {}
        import_blocks: set[Block] = set(monitor.import_deps)
        for block in monitor.collect_deps:
            rel = block[0]
            if rel in self.module_files:
                module_collect.setdefault(rel, set()).add(block)
            else:
                import_blocks.add(block)
        # Conftest skeletons: dependency of every test. (Normally captured
        # by the monitor at import time anyway; covers late activation and
        # empty conftest files.)
        for rel in self.conftest_deps:
            import_blocks.add(monitor.module_dep(rel))

        old_shard = self.shard
        old_tests = old_shard.tests if old_shard is not None else {}
        # Keep previously recorded import-time deps (conservative: a module
        # that stopped being imported still re-runs its dependents).
        import_blocks |= old_shard.import_deps if old_shard is not None else set()

        tests: dict[str, set[Block]] = {}
        for nodeid, module_rel in self.item_map.items():
            own = monitor.test_deps.get(nodeid)
            if own is not None and nodeid in self.finished:
                extra = module_collect.get(module_rel, set()) if module_rel else set()
                tests[nodeid] = own | extra
            elif nodeid in old_tests:
                # Not run this time (deselected or interrupted): keep the
                # previous block set; the shared import deps cover the rest.
                tests[nodeid] = set(old_tests[nodeid])
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
                tests[nodeid] = set(old_deps)
            # else: the file was collected but the test is gone -> drop

        # Serialize with a dedup table: shared blocks (import-time and
        # common helpers) are stored once.
        table: dict[Block, int] = {}
        ordered: list[Block] = []

        def index(block: Block) -> int:
            i = table.get(block)
            if i is None:
                i = len(ordered)
                ordered.append(block)
                table[block] = i
            return i

        data = {
            "version": DATA_VERSION,
            "blocks": [list(b) for b in ordered],  # filled below
            "tests": {
                nid: sorted({index(b) for b in tests[nid]})
                for nid in sorted(tests)
            },
            "import_deps": sorted({index(b) for b in import_blocks}),
            "failed": sorted(self.failed),
        }
        data["blocks"] = [list(b) for b in ordered]
        dep_files: set[str] = {b[0] for b in import_blocks}
        for deps in tests.values():
            dep_files.update(b[0] for b in deps)
        meta = {
            "version": 1,
            "shard": self.store.shard,
            "base_commit": self.store.head(),
            "created": datetime.now(UTC).isoformat(),
            # First-level cache for the no-change rerun (see
            # _try_fast_path): everything needed to decide "can anything
            # be stale?" without reading data.json.
            "failed": sorted(self.failed),
            "nodeids": sorted(tests),
            "dep_files": sorted(dep_files),
        }
        self.store.save(data, meta)
        if self.sync_remote and self.sync_mode in ("push", "both"):
            try:
                self.notes.append(f"filemon: {self.store.sync_push(self.sync_remote)}")
            except Exception as e:  # never break the run
                self.notes.append(f"filemon: sync push failed ({e!r})")


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
    rel = state.repo_rel(collection_path)
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
