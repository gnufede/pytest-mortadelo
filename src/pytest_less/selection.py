"""Test selection: recorded per-test block dependencies vs. changed files.

A dependency is a ``Block`` triple ``(file, block id, hash)``.  A changed
file does not automatically affect a test: the file is re-parsed and only
stale blocks (missing, or with a different content hash) re-run their
dependents.  A ``WILDCARD`` block id (dynamic code, redefined names,
unparseable files, recorded data files) means "any change
to this file re-runs".

Tests are selected when they have no recorded data, failed in the recorded
run, depend on a stale block, or (via the shared ``import_deps``) when any
import-time dependency of every test went stale.  Widening rules sit on top
of that:

* **subprocess spawns** -- coverage cannot follow a child process, so a
  test that spawned one is *never given dependencies* and always re-runs
  (``always`` lists them).  ``spawn_unknown`` (old shards, or a spawn
  during collection, where the culprit test is unknown) falls back to the
  weaker rule: every test is selected when a changed ``.py`` file is stale
  or was never recorded at all.
* **runtime-dynamic tests** -- tests that went through a reflective entry
  point (``importlib.import_module``, ``pkgutil.walk_packages``, ``runpy``,
  ...) at test time (``dynamic``) are selected when a changed ``.py`` file
  was *never recorded* (a new module, or one no test ever executed): the one
  edge a computed name can reach that the recording never saw.  A *stale*
  recorded file needs no widening: whatever the test executed of it is
  already a dependency, and what it did not execute cannot change its
  result.  (String-compiled code -- ``eval``/``exec`` -- is deliberately not
  a signal: the generated code comes from generators that are themselves
  recorded, and flagging it selected 69% of sqlalchemy's ORM tests on every
  edit.)
* everything else is block-precise; each widening is reported in the run
  summary, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .blocks import WILDCARD, Block, FileBlocks
from .util import _EXCLUDED_SEGMENTS

# Shard data version this code reads and writes.
DATA_VERSION = 2


@dataclass
class Shard:
    """Parsed shard data for one pytest command."""

    base_commit: str | None
    # node id -> the test's own recorded blocks
    tests: dict[str, set[Block]]
    # import-time blocks shared by every test (skeletons + anything that
    # executed before the first test: conftests, plugins, module-level calls)
    import_deps: set[Block]
    failed: set[str]
    # Tests observed spawning a process during their run: they carry no
    # dependencies and re-run every time.
    always: set[str] = field(default_factory=set)
    # True when the spawners are unknown (shard recorded before this
    # feature, or a spawn during collection): every test is conservatively
    # treated as a spawner under the "stale or unrecorded .py" rule.
    spawn_unknown: bool = False
    # Tests that ran string-compiled code / reflective entry points.  None =
    # unknown (old shard): every test is conservatively treated as dynamic.
    dynamic: set[str] | None = field(default_factory=set)


@dataclass
class Selection:
    """Outcome of the selection for one run."""

    selected: list[str]
    deselected: list[str]
    changed: set[str] = field(default_factory=set)
    reason: str = ""

    @property
    def selected_set(self) -> set[str]:
        return set(self.selected)


def load_shard(data, meta) -> Shard | None:
    """Parse a shard, or return None if the data is unusable (fresh/corrupt)."""
    if not isinstance(data, dict) or not isinstance(meta, dict):
        return None
    try:
        version = data.get("version")
        if version == DATA_VERSION:
            table = [tuple(b) for b in data["blocks"]]
            raw_tests = data["tests"]
            if not isinstance(raw_tests, dict):
                return None
            tests = {}
            for nid, indices in raw_tests.items():
                if not isinstance(indices, list):
                    return None
                # ~1.65M index lookups on a 2,480-test shard: min/max
                # validate the whole list at C speed, map builds the set.
                # Anything min/max cannot order (a non-int) raises, which the
                # outer except turns into "unusable" exactly as the per-element
                # check did; a bool is an int here as it was there.
                if indices:
                    low, high = min(indices), max(indices)
                    if not (
                        isinstance(low, int)
                        and 0 <= low
                        and isinstance(high, int)
                        and high < len(table)
                    ):
                        return None
                    tests[nid] = set(map(table.__getitem__, indices))
                else:
                    tests[nid] = set()
            import_deps = set()
            for i in data.get("import_deps", []):
                if not isinstance(i, int) or not (0 <= i < len(table)):
                    return None
                import_deps.add(table[i])
        else:
            return None
        raw_failed = data.get("failed") or []
        if not isinstance(raw_failed, list):
            return None
        failed = {f for f in raw_failed if isinstance(f, str)}
        # Subprocess marks.  ``always``: tests that spawned a process (no
        # dependencies, re-run every time).  Absent ``always`` (shard from
        # before this feature) or ``spawn_unknown`` (a spawn during
        # collection: the culprit is unknown) -> conservatively treat all
        # tests as spawners.  The next full record run self-heals.
        raw_always = data.get("always")
        always = (
            {a for a in raw_always if isinstance(a, str)}
            if isinstance(raw_always, list)
            else set()
        )
        spawn_unknown = not isinstance(raw_always, list) or bool(
            data.get("spawn_unknown")
        )
        raw_dynamic = data.get("dynamic")
        dynamic: set[str] | None = (
            {d for d in raw_dynamic if isinstance(d, str)}
            if isinstance(raw_dynamic, list)
            else None
        )
        base_commit = meta.get("base_commit")
        if base_commit is not None and not isinstance(base_commit, str):
            return None
    except Exception:
        return None
    return Shard(
        base_commit=base_commit,
        tests=tests,
        import_deps=import_deps,
        failed=failed,
        always=always,
        spawn_unknown=spawn_unknown,
        dynamic=dynamic,
    )


def stale_dep(dep: Block, changed: set[str], maps: dict[str, FileBlocks | None]) -> bool:
    """Whether a recorded dependency went stale.

    Unchanged files never stale.  For changed files the current block table
    decides: wildcards and missing/unparseable files always stale; a block
    stales only if it is gone or its content hash changed.
    """
    rel, block_id, recorded_hash = dep
    if rel not in changed:
        return False
    if block_id == WILDCARD:
        return True
    fb = maps.get(rel)
    if fb is None:
        return True  # deleted or unparseable
    return fb.blocks.get(block_id) != recorded_hash


@dataclass
class _Context:
    """Per-run selection facts computed once from the whole shard.

    Holds the global conditions (import-time staleness, subprocess-trigger
    availability) and memoizes ``stale_dep``
    per block triple: every consumer (collection planning, selection) asks
    the same questions.
    """

    changed: set[str]
    maps: dict[str, FileBlocks | None]
    import_stale: bool
    spawn_active: bool  # some changed .py is stale or was never recorded
    unrecorded_py: bool = False  # some changed .py was never recorded
    # Nothing changed since the recording, at the same commit, with everything
    # a child process could read watched: tests that always re-run do not.
    pristine: bool = False
    _memo: dict[Block, bool] = field(default_factory=dict)

    def stale(self, dep: Block) -> bool:
        hit = self._memo.get(dep)
        if hit is None:
            hit = stale_dep(dep, self.changed, self.maps)
            self._memo[dep] = hit
        return hit


def _is_candidate_py(rel: str) -> bool:
    """A .py file the spawn rule cares about (production/test code, not venvs)."""
    return rel.endswith(".py") and not _EXCLUDED_SEGMENTS.intersection(
        Path(rel).parts
    )


def build_context(
    shard: Shard,
    changed: set[str],
    maps: dict[str, FileBlocks | None],
    pristine: bool = False,
) -> _Context:
    """Compute the per-run widening facts (one pass over the shard)."""
    import_stale = False
    recorded: set[str] = set()
    stale_files: set[str] = set()

    def note(block: Block) -> None:
        rel = block[0]
        if rel in changed:
            recorded.add(rel)
            if stale_dep(block, changed, maps):
                stale_files.add(rel)

    for b in shard.import_deps:
        if not import_stale and stale_dep(b, changed, maps):
            import_stale = True
        note(b)
    for deps in shard.tests.values():
        for b in deps:
            note(b)

    changed_py = {f for f in changed if _is_candidate_py(f)}
    spawn_active = bool(changed_py) and bool(
        (changed_py & stale_files) | (changed_py - recorded)
    )
    return _Context(
        pristine=pristine,
        unrecorded_py=bool(changed_py - recorded),
        changed=changed,
        maps=maps,
        import_stale=import_stale,
        spawn_active=spawn_active,
    )


def affected_reason(
    nodeid: str, deps: set[Block], shard: Shard, ctx: _Context
) -> str | None:
    """Why *nodeid* must run, or None if it is unaffected.

    The single predicate shared by collection planning and selection, so a
    test can never run under one and be skipped under the other.
    """
    if nodeid in shard.always and not ctx.pristine:
        return "always"
    if ctx.import_stale:
        return "import"
    if any(ctx.stale(b) for b in deps):
        return "stale"
    if ctx.spawn_active and shard.spawn_unknown:
        return "spawn"
    if ctx.unrecorded_py and (shard.dynamic is None or nodeid in shard.dynamic):
        return "runtime"
    return None


def select_tests(
    collected: list[str],
    shard: Shard | None,
    changed: set[str] | None,
    maps: dict[str, FileBlocks | None] | None = None,
    ctx: _Context | None = None,
    pristine: bool = False,
) -> Selection:
    """Decide which of *collected* tests should run."""
    maps = maps or {}
    if shard is None:
        return Selection(
            list(collected),
            [],
            set(),
            "no usable stored data for this command; running and recording everything",
        )
    if changed is None:
        return Selection(
            list(collected),
            [],
            set(),
            "cannot compute changes since the recorded base commit "
            "(rebased/rewritten history?); running everything",
        )
    # Import-time deps are shared by every test: if any went stale, all
    # tests run. This is the old "global dependency" behavior, now
    # block-granular: a body edit of a function that only runs at test time
    # no longer re-runs the world -- only module skeletons and the blocks
    # that actually executed at import time (module-level calls included) do.
    if ctx is None:
        ctx = build_context(shard, changed, maps, pristine=pristine)
    selected: list[str] = []
    deselected: list[str] = []
    n_affected = n_failed = n_new = n_spawn = 0
    n_always = n_runtime = 0
    always = shard.always
    for nodeid in collected:
        deps = shard.tests.get(nodeid)
        if nodeid in always and not ctx.pristine:
            selected.append(nodeid)
            n_always += 1
        elif deps is None:
            selected.append(nodeid)
            n_new += 1
        elif nodeid in shard.failed:
            selected.append(nodeid)
            n_failed += 1
        else:
            why = affected_reason(nodeid, deps, shard, ctx)
            if why is None:
                deselected.append(nodeid)
                continue
            selected.append(nodeid)
            if why == "spawn":
                n_spawn += 1
            elif why == "runtime":
                n_runtime += 1
            else:
                n_affected += 1
    base = (shard.base_commit or "?")[:8]
    reason = (
        f"{len(changed)} file(s) changed since {base}: "
        f"{n_affected} affected, {n_failed} previously failed, {n_new} new"
    )
    if ctx.import_stale:
        reason += "; import-time dependencies changed"
    if n_always:
        reason += f"; {n_always} always re-run (spawn subprocesses)"
    elif ctx.pristine and always & set(collected):
        reason += (
            f"; {len(always & set(collected))} always-run test(s) skipped: "
            "nothing changed since the recorded commit"
        )
    if n_spawn:
        reason += f"; {n_spawn} selected by the subprocess rule"
    if n_runtime:
        reason += f"; {n_runtime} selected as runtime-dynamic (exec/import_module/...)"
    return Selection(selected, deselected, set(changed), reason)
