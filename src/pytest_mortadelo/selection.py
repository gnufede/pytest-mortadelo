"""Test selection: recorded per-test block dependencies vs. changed files.

A dependency is a ``Block`` triple ``(file, block id, content hash)``.
A changed file does not automatically affect a test: the file is
re-analyzed and only stale blocks (missing, or with a different content
hash) re-run their dependents.

A test runs when it has no recorded data, failed in the recorded run,
depends on a stale block, or (via the shared ``import_deps``) when any
import-time dependency of every test went stale.  Tests that spawned a
process are re-run every time (``always``): coverage cannot follow a
child.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .blocks import WILDCARD, FileBlocks

# A dependency: (repo-relative file, block id, content hash).
Block = tuple[str, str, str]

DATA_VERSION = 2  # must match the plugin; bump to invalidate stored data


@dataclass
class Graph:
    """The recorded dependency graph from the last run."""

    base_commit: str | None
    tests: dict[str, set[Block]]
    import_deps: set[Block]
    failed: set[str]
    # Tests that spawned a process: no dependencies, re-run every time.
    always: set[str]


def load_graph(data, meta) -> Graph | None:
    """Parse the stored data, or None if it is unusable (fresh/corrupt)."""
    if not isinstance(data, dict) or not isinstance(meta, dict):
        return None
    try:
        version = data.get("version")
        if version != DATA_VERSION:
            return None
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
        raw_failed = data.get("failed") or []
        if not isinstance(raw_failed, list):
            return None
        failed = {f for f in raw_failed if isinstance(f, str)}
        raw_always = data.get("always")
        always = (
            {a for a in raw_always if isinstance(a, str)}
            if isinstance(raw_always, list)
            else set()
        )
        base_commit = meta.get("base_commit")
        if base_commit is not None and not isinstance(base_commit, str):
            return None
    except Exception:
        return None
    return Graph(
        base_commit=base_commit,
        tests=tests,
        import_deps=import_deps,
        failed=failed,
        always=always,
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
    """Per-run selection facts computed once from the whole graph."""

    changed: set[str]
    maps: dict[str, FileBlocks | None]
    import_stale: bool
    stale_blocks: frozenset[Block]


def build_context(
    graph: Graph,
    changed: set[str],
    maps: dict[str, FileBlocks | None],
) -> _Context:
    """Compute the per-run facts (one pass over the distinct blocks).

    A graph keeps one dependency set per test, but the blocks repeat:
    ~1.65M ``(test, block)`` edges on a 2,480-test suite over ~9k
    distinct blocks.  Every fact depends only on the distinct blocks.
    """
    distinct: set[Block] = set(graph.import_deps)
    for deps in graph.tests.values():
        distinct.update(deps)

    stale = frozenset(
        b for b in distinct if b[0] in changed and stale_dep(b, changed, maps)
    )
    return _Context(
        changed=changed,
        maps=maps,
        import_stale=not stale.isdisjoint(graph.import_deps),
        stale_blocks=stale,
    )


def affected_reason(nodeid: str, deps: set[Block], graph: Graph, ctx: _Context) -> str | None:
    """Why *nodeid* must run, or None if it is unaffected.

    The single predicate shared by collection planning and selection, so a
    test can never run under one and be skipped under the other.
    """
    if nodeid in graph.always:
        return "always"
    if ctx.import_stale:
        return "import"
    if not deps.isdisjoint(ctx.stale_blocks):
        return "stale"
    return None


@dataclass
class Selection:
    """Outcome of the selection for one run."""

    selected: list[str]
    deselected: list[str]
    changed: set[str] = field(default_factory=set)
    reason: str = ""


def select_tests(
    collected: list[str],
    graph: Graph | None,
    changed: set[str] | None,
    maps: dict[str, FileBlocks | None] | None = None,
    ctx: _Context | None = None,
) -> Selection:
    """Decide which of *collected* tests should run."""
    maps = maps or {}
    if graph is None:
        return Selection(
            list(collected), [], set(),
            "no stored data; running everything",
        )
    if changed is None:
        return Selection(
            list(collected), [], set(),
            "cannot compute changes since the recorded base commit; "
            "running everything",
        )
    if ctx is None:
        ctx = build_context(graph, changed, maps)
    selected: list[str] = []
    deselected: list[str] = []
    n_affected = n_failed = n_new = n_always = 0
    for nodeid in collected:
        deps = graph.tests.get(nodeid)
        if nodeid in graph.always:
            selected.append(nodeid)
            n_always += 1
        elif deps is None:
            selected.append(nodeid)
            n_new += 1
        elif nodeid in graph.failed:
            selected.append(nodeid)
            n_failed += 1
        else:
            why = affected_reason(nodeid, deps, graph, ctx)
            if why is None:
                deselected.append(nodeid)
                continue
            selected.append(nodeid)
            n_affected += 1
    base = (graph.base_commit or "?")[:8]
    reason = (
        f"{len(changed)} file(s) changed since {base}: "
        f"{n_affected} affected, {n_failed} previously failed, {n_new} new"
    )
    if ctx.import_stale:
        reason += "; import-time dependencies changed"
    if n_always:
        reason += f"; {n_always} always re-run (spawn subprocesses)"
    return Selection(selected, deselected, set(changed), reason)
