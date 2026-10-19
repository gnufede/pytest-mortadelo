"""Test selection: recorded per-test block dependencies vs. changed files.

A dependency is a ``Block`` triple ``(file, block id, hash)``.  A changed
file does not automatically affect a test: the file is re-parsed and only
stale blocks (missing, or with a different content hash) re-run their
dependents.  A ``WILDCARD`` block id (dynamic code, redefined names, v1
shards, unparseable files, audit-recorded data files) means "any change
to this file re-runs".

Tests are selected when they have no recorded data, failed in the recorded
run, depend on a stale block, or (via the shared ``import_deps``) when any
import-time dependency of every test went stale.  Three widening rules sit
on top of that:

* **dynamic markers** -- a changed file that both *contains* reflective
  constructs (computed ``getattr``, ``__getattr__``, ``eval``/``exec``, ...)
  *and* has a stale block is widened back to file level: the recorded run
  may not hold every edge reflective dispatch creates (Ley 1 net).
* **subprocess spawns (Tier A)** -- a test that spawned a process is
  selected when any changed ``.py`` file is stale or was never recorded at
  all: the child may have executed code the parent's recording never saw
  (the e2b blind spot).  ``spawning is None`` (old shards, or a spawn
  during collection) conservatively treats every test as a spawner.
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
    # Tests observed spawning a process during their run.  None = unknown
    # (shard recorded before Tier A, or a spawn during collection): every
    # test is conservatively treated as a spawner.
    spawning: set[str] | None = None


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
    """Parse a shard, or return None if the data is unusable (fresh/corrupt).

    Version 1 shards (file-level dependencies) are migrated to wildcard
    blocks: ``file`` becomes ``(file, "*", "*")`` -- exactly the old
    any-change-to-the-file behavior.
    """
    if not isinstance(data, dict) or not isinstance(meta, dict):
        return None
    try:
        version = data.get("version")
        if version == 1:
            raw_tests = data["tests"]
            if not isinstance(raw_tests, dict):
                return None
            tests = {
                nid: {(f, WILDCARD, WILDCARD) for f in files if isinstance(f, str)}
                for nid, files in raw_tests.items()
                if isinstance(nid, str) and isinstance(files, list)
            }
            import_deps: set[Block] = set()
        elif version == DATA_VERSION:
            table = [tuple(b) for b in data["blocks"]]
            raw_tests = data["tests"]
            if not isinstance(raw_tests, dict):
                return None
            tests = {}
            for nid, indices in raw_tests.items():
                if not isinstance(indices, list):
                    return None
                own = set()
                for i in indices:
                    if not isinstance(i, int) or not (0 <= i < len(table)):
                        return None
                    own.add(table[i])
                tests[nid] = own
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
        # Tier-A subprocess marks.  Absent (shard recorded before Tier A),
        # null (a spawn during collection: every test may be affected) or
        # corrupt -> None: conservatively treat all tests as spawners.
        # The next record run re-derives the precise list and self-heals.
        if "spawning" in data and isinstance(data["spawning"], list):
            spawning: set[str] | None = {s for s in data["spawning"] if isinstance(s, str)}
        else:
            spawning = None
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
        spawning=spawning,
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

    Holds the global conditions (import-time staleness, dynamic-marker
    widenings, subprocess-trigger availability) and memoizes ``stale_dep``
    per block triple: every consumer (collection planning, selection) asks
    the same questions.
    """

    changed: set[str]
    maps: dict[str, FileBlocks | None]
    import_stale: bool
    dynamic_files: set[str]  # marker-bearing files with a stale block
    dynamic_global: bool  # ... of which one is in the shared import bucket
    spawn_active: bool  # some changed .py is stale or was never recorded
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
    shard: Shard, changed: set[str], maps: dict[str, FileBlocks | None]
) -> _Context:
    """Compute the per-run widening facts (one pass over the shard)."""
    import_stale = False
    import_files: set[str] = set()
    recorded: set[str] = set()
    stale_files: set[str] = set()

    def note(block: Block) -> None:
        rel = block[0]
        if rel in changed:
            recorded.add(rel)
            if stale_dep(block, changed, maps):
                stale_files.add(rel)

    for b in shard.import_deps:
        import_files.add(b[0])
        if not import_stale and stale_dep(b, changed, maps):
            import_stale = True
        note(b)
    for deps in shard.tests.values():
        for b in deps:
            note(b)

    # Dynamic-marker widening: reflective construct + a real (stale) change
    # in the same file.  Cosmetic/format-only changes produce no stale block
    # and do not widen (interlock with --less-cosmetic).  A marker file
    # everyone imported widens to *all* tests (file-level semantics: any
    # test could reach code through an edge the recording never saw).
    dynamic_files = {
        rel
        for rel in stale_files
        if (fb := maps.get(rel)) is not None and fb.markers
    }
    changed_py = {f for f in changed if _is_candidate_py(f)}
    spawn_active = bool(changed_py) and bool(
        (changed_py & stale_files) | (changed_py - recorded)
    )
    return _Context(
        changed=changed,
        maps=maps,
        import_stale=import_stale,
        dynamic_files=dynamic_files,
        dynamic_global=bool(dynamic_files & import_files),
        spawn_active=spawn_active,
    )


def affected_reason(
    nodeid: str, deps: set[Block], shard: Shard, ctx: _Context
) -> str | None:
    """Why *nodeid* must run, or None if it is unaffected.

    The single predicate shared by collection planning and selection, so a
    test can never run under one and be skipped under the other.
    """
    if ctx.import_stale:
        return "import"
    if any(ctx.stale(b) for b in deps):
        return "stale"
    if ctx.dynamic_global or any(b[0] in ctx.dynamic_files for b in deps):
        return "dynamic"
    if ctx.spawn_active and (shard.spawning is None or nodeid in shard.spawning):
        return "spawn"
    return None


def select_tests(
    collected: list[str],
    shard: Shard | None,
    changed: set[str] | None,
    maps: dict[str, FileBlocks | None] | None = None,
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
    ctx = build_context(shard, changed, maps)
    selected: list[str] = []
    deselected: list[str] = []
    n_affected = n_failed = n_new = n_dynamic = n_spawn = 0
    for nodeid in collected:
        deps = shard.tests.get(nodeid)
        if deps is None:
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
            if why == "dynamic":
                n_dynamic += 1
            elif why == "spawn":
                n_spawn += 1
            else:
                n_affected += 1
    base = (shard.base_commit or "?")[:8]
    reason = (
        f"{len(changed)} file(s) changed since {base}: "
        f"{n_affected} affected, {n_failed} previously failed, {n_new} new"
    )
    if ctx.import_stale:
        reason += "; import-time dependencies changed"
    if n_dynamic:
        reason += (
            f"; {n_dynamic} widened by dynamic markers in "
            f"{sorted(ctx.dynamic_files)}"
        )
    if n_spawn:
        reason += f"; {n_spawn} selected by the subprocess rule"
    return Selection(selected, deselected, set(changed), reason)
