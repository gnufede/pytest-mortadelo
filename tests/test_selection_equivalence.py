"""The selection machinery against straightforward reference implementations.

The production ``load_shard`` / ``build_context`` / ``affected_reason`` are
optimized to work per *distinct* block and with set intersections (C speed),
because a shard holds ~1.65M ``(test, block)`` edges and the stale set is tiny.
The references here do the same job the slow, obvious way -- per element, per
edge -- and randomized shards must give identical answers.  They double as the
statement of what the optimizations are allowed to change: nothing.
"""

import random

import pytest
from blocktable import table_of

from pytest_less.blocks import MODULE_BLOCK, WILDCARD, FileBlocks
from pytest_less.selection import (
    DATA_VERSION,
    Shard,
    affected_reason,
    build_context,
    load_shard,
    select_tests,
    stale_dep,
)

# --------------------------------------------------------------------------
# The references.


def reference_load_shard(data, meta):
    """The current decoder, spelled out per element."""
    if not isinstance(data, dict) or not isinstance(meta, dict):
        return None
    try:
        if data.get("version") != DATA_VERSION:
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
        raw_always = data.get("always")
        always = (
            {a for a in raw_always if isinstance(a, str)}
            if isinstance(raw_always, list)
            else set()
        )
        raw_dynamic = data.get("dynamic")
        dynamic = (
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
        failed={f for f in raw_failed if isinstance(f, str)},
        always=always,
        spawn_unknown=not isinstance(raw_always, list) or bool(data.get("spawn_unknown")),
        dynamic=dynamic,
    )


def reference_reason(nodeid, deps, shard, changed, maps, pristine):
    """affected_reason, spelled out per block."""
    if nodeid in shard.always and not pristine:
        return "always"
    if any(stale_dep(b, changed, maps) for b in shard.import_deps):
        return "import"
    if any(stale_dep(b, changed, maps) for b in deps):
        return "stale"
    changed_py = {f for f in changed if f.endswith(".py")}
    recorded = {b[0] for deps_of in [shard.import_deps, *shard.tests.values()] for b in deps_of if b[0] in changed}
    stale_files = {
        b[0]
        for b in [b for d in [shard.import_deps, *shard.tests.values()] for b in d]
        if b[0] in changed and stale_dep(b, changed, maps)
    }
    if changed_py and ((changed_py & stale_files) or (changed_py - recorded)):
        if shard.spawn_unknown:
            return "spawn"
        if changed_py - recorded and (shard.dynamic is None or nodeid in shard.dynamic):
            return "runtime"
    return None


def reference_selection(collected, shard, changed, maps, pristine):
    selected, deselected = [], []
    for nodeid in collected:
        deps = shard.tests.get(nodeid)
        if nodeid in shard.always and not pristine or deps is None or nodeid in shard.failed:
            selected.append(nodeid)
        else:
            why = reference_reason(nodeid, deps, shard, changed, maps, pristine)
            if why is None:
                deselected.append(nodeid)
            else:
                selected.append(nodeid)
    return selected, deselected


# --------------------------------------------------------------------------
# Randomized shards and change sets.


def random_shard(rng):
    n_files = rng.randint(1, 6)
    files = [f"src/mod{i}.py" for i in range(n_files)] + ["src/plain.txt"]
    blocks = []
    for f in files:
        for j in range(rng.randint(0, 4)):
            kind = rng.choice(["fn", "cls", "wild", "module"])
            bid = {"fn": f"f{j}", "cls": f"C{j}", "wild": WILDCARD, "module": MODULE_BLOCK}[kind]
            blocks.append((f, bid, f"h{rng.randint(0, 3)}"))
    n = len(blocks)
    tests = {}
    for t in range(rng.randint(0, 12)):
        nid = f"src/test{t % 3}.py::test_{t}"
        # duplicates in a list must collapse like a set would
        tests[nid] = sorted({rng.randrange(n) for _ in range(rng.randint(0, 5))}) if n else []
    import_deps = sorted({rng.randrange(n) for _ in range(rng.randint(0, 4))}) if n else []
    always = [f"src/test{t}.py::test_{t}" for t in range(3) if rng.random() < 0.3]
    failed = [f"src/test{t}.py::test_{t}" for t in range(3) if rng.random() < 0.2]
    dynamic = rng.choice([None, [], [f"src/test0.py::test_{rng.randint(0, 11)}"]])
    data = {
        "version": DATA_VERSION,
        "blocks": [list(b) for b in blocks],
        "tests": {k: v for k, v in tests.items()},
        "import_deps": import_deps,
        "failed": failed,
        "always": always,
        "spawn_unknown": rng.random() < 0.3,
        "dynamic": dynamic,
    }
    meta = {"base_commit": rng.choice([None, "c0ffee"])}
    return data, meta, blocks


def random_maps(rng, blocks):
    """Current block tables for the changed files: sometimes matching, sometimes
    not, sometimes absent (deleted/unparseable)."""
    changed = {
        b[0] for b in blocks if rng.random() < 0.4
    } | {f"src/new{rng.randint(0, 2)}.py"}  # a never-recorded file, sometimes .py
    maps = {}
    for f in sorted(changed):
        r = rng.random()
        if r < 0.2:
            maps[f] = None  # deleted / unparseable
        elif r < 0.7:
            fresh = (
                f"class C:\n    def f{rng.randint(0, 3)}(self):\n"
                f"        return {rng.randint(0, 9)}\n"
            )
            maps[f] = table_of(fresh)  # a table that shares nothing with the recording
        else:
            # a current table in which some blocks kept their recorded hash
            # and some changed (the third branch of stale_dep)
            keep = {(b[0], b[1]): b[2] for b in blocks if b[0] == f}
            fb_blocks = {
                bid: (h if rng.random() < 0.5 else f"z{rng.randint(0, 3)}")
                for (_, bid), h in keep.items()
            } or {MODULE_BLOCK: "fresh"}
            maps[f] = FileBlocks(fb_blocks, [(1, bid.split("~")[0]) for bid in fb_blocks])
    return changed, maps


@pytest.mark.parametrize("seed", range(300))
def test_select_tests_matches_the_reference(seed):
    rng = random.Random(seed)
    data, meta, blocks = random_shard(rng)
    shard = load_shard(data, meta)
    ref = reference_load_shard(data, meta)
    assert (shard is None) == (ref is None)
    if shard is None:
        assert ref is None
        return
    for field in ("base_commit", "tests", "import_deps", "failed", "always", "spawn_unknown", "dynamic"):
        assert getattr(shard, field) == getattr(ref, field), field
    changed, maps = random_maps(rng, blocks)
    pristine = not changed and rng.random() < 0.5
    collected = sorted(set(shard.tests) | {f"src/test{i}.py::test_new{i}" for i in range(2)})
    got = select_tests(collected, shard, changed, maps, pristine=pristine)
    want_selected, want_deselected = reference_selection(collected, shard, changed, maps, pristine)
    assert got.selected == want_selected
    assert got.deselected == want_deselected
    # the shared predicate agrees with itself for every test, both ways
    ctx = build_context(shard, changed, maps, pristine=pristine)
    for nodeid in collected:
        if nodeid not in shard.tests:
            continue  # a new test is selected before the predicate is asked
        deps = shard.tests[nodeid]
        # always/failed short-circuit the predicate in select_tests
        must_run = (
            nodeid in shard.always
            or nodeid in shard.failed
            or affected_reason(nodeid, deps, shard, ctx) is not None
        )
        assert must_run == (nodeid in want_selected), nodeid


@pytest.mark.parametrize(
    "tests_field",
    [
        {"a": [0]},  # one valid index
        {"a": [0, 0]},  # duplicates collapse
        {"a": [-1]},  # negative: silently wrong with __getitem__, must be rejected
        {"a": [True]},  # bool is an int subclass: table[True] is table[1]
        {"a": ["1"]},  # not an int
        {"a": [1.0]},  # not an int
        {"a": [[]]},  # not an int
    ],
)
def test_load_shard_indices(tests_field):
    data = {"version": DATA_VERSION, "blocks": [["f.py", "f", "h0"], ["f.py", "C", "h1"]], "tests": tests_field}
    ref = reference_load_shard(data, {"base_commit": None})
    got = load_shard(data, {"base_commit": None})
    assert (got is None) == (ref is None)
    if got is not None:
        assert got.tests == ref.tests


@pytest.mark.parametrize("imp", [[-1], ["x"], [True], [0, 99], [1]])
def test_load_shard_import_deps(imp):
    data = {"version": DATA_VERSION, "blocks": [["f.py", "f", "h0"], ["g.py", "C", "h1"]], "tests": {}, "import_deps": imp}
    ref = reference_load_shard(data, {"base_commit": None})
    got = load_shard(data, {"base_commit": None})
    assert (got is None) == (ref is None)
    if got is not None:
        assert got.import_deps == ref.import_deps
