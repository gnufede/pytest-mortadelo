"""Test selection: recorded per-test block dependencies vs. changed files.

A dependency is a ``Block`` triple ``(file, block id, hash)``.  A changed
file does not automatically affect a test: the file is re-parsed and only
stale blocks (missing, or with a different content hash) re-run their
dependents.  A ``WILDCARD`` block id (dynamic code, redefined names, v1
shards, unparseable files, audit-recorded data files) means "any change
to this file re-runs".

Tests are selected when they have no recorded data, failed in the recorded
run, depend on a stale block, or (via the shared ``import_deps``) when any
import-time dependency of every test went stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .blocks import WILDCARD, Block, FileBlocks

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
        base_commit = meta.get("base_commit")
        if base_commit is not None and not isinstance(base_commit, str):
            return None
    except Exception:
        return None
    return Shard(base_commit=base_commit, tests=tests, import_deps=import_deps, failed=failed)


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
    import_stale = any(stale_dep(b, changed, maps) for b in shard.import_deps)
    selected: list[str] = []
    deselected: list[str] = []
    n_affected = n_failed = n_new = 0
    for nodeid in collected:
        deps = shard.tests.get(nodeid)
        if deps is None:
            selected.append(nodeid)
            n_new += 1
        elif nodeid in shard.failed:
            selected.append(nodeid)
            n_failed += 1
        elif import_stale or any(stale_dep(b, changed, maps) for b in deps):
            selected.append(nodeid)
            n_affected += 1
        else:
            deselected.append(nodeid)
    base = (shard.base_commit or "?")[:8]
    reason = (
        f"{len(changed)} file(s) changed since {base}: "
        f"{n_affected} affected, {n_failed} previously failed, {n_new} new"
    )
    if import_stale:
        reason += "; import-time dependencies changed"
    return Selection(selected, deselected, set(changed), reason)
