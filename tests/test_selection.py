"""Selection over block-level dependencies."""

from typing import ClassVar

from blocktable import table_of

from pytest_less.blocks import MODULE_BLOCK, WILDCARD
from pytest_less.selection import (
    Shard,
    load_shard,
    select_tests,
    stale_dep,
)


def dep(file, block, hash_="h1"):
    return (file, block, hash_)


def make_shard(**overrides):
    shard = Shard(
        base_commit="ab" * 20,
        tests={
            "tests/test_a.py::test_a": {dep("src/lib.py", "ping"), dep("tests/test_a.py", "test_a")},
            "tests/test_b.py::test_b": {dep("tests/test_b.py", "test_b")},
        },
        import_deps={dep("tests/conftest.py", MODULE_BLOCK)},
        failed=set(),
    )
    for key, value in overrides.items():
        setattr(shard, key, value)
    return shard


class TestLoadShard:
    def test_v2_roundtrip_shape(self):
        # the writer's format (dedup table + indices) parses back
        shard = load_shard(
            {
                "version": 2,
                "blocks": [["src/lib.py", "ping", "h1"], ["tests/conftest.py", MODULE_BLOCK, "h0"]],
                "tests": {"a": [0], "b": [0, 1]},
                "import_deps": [1],
                "failed": ["b"],
            },
            {"base_commit": "c0ffee"},
        )
        assert shard is not None
        assert shard.tests == {"a": {("src/lib.py", "ping", "h1")}, "b": {("src/lib.py", "ping", "h1"), ("tests/conftest.py", MODULE_BLOCK, "h0")}}
        assert shard.import_deps == {("tests/conftest.py", MODULE_BLOCK, "h0")}
        assert shard.failed == {"b"}

    def test_v1_files_migrate_to_wildcards(self):
        # v1 shards (plain file lists) keep working with the old semantics:
        # any change to the file re-runs.
        shard = load_shard(
            {"version": 1, "tests": {"a": ["x.py"]}, "failed": []},
            {"base_commit": "c0ffee"},
        )
        assert shard.tests == {"a": {("x.py", WILDCARD, WILDCARD)}}
        assert shard.import_deps == set()

    def test_bad_version(self):
        assert load_shard({"version": 3, "tests": {}}, {"base_commit": None}) is None

    def test_corrupt(self):
        assert load_shard({"version": 2, "blocks": [], "tests": {}}, {}) is not None
        assert load_shard({"version": 2, "blocks": [], "tests": {"a": [7]}}, {}) is None
        assert load_shard({"version": 1}, {}) is None  # missing "tests"
        assert load_shard(None, None) is None

    def test_missing_base_commit_ok(self):
        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}, "import_deps": [], "failed": []},
            {},
        )
        assert shard is not None
        assert shard.base_commit is None


class TestStaleDep:
    def test_unchanged_file_never_stales(self):
        assert not stale_dep(dep("src/lib.py", "ping"), set(), {})

    def test_changed_file_matching_hash_not_stale(self):
        fb = table_of("def ping():\n    return 1\n")
        # recorded hash from the same content:
        recorded = ("src/lib.py", "ping", fb.blocks["ping"])
        maps = {"src/lib.py": fb}
        assert not stale_dep(recorded, {"src/lib.py"}, maps)

    def test_changed_body_stales(self):
        old = table_of("def ping():\n    return 1\n")
        new = table_of("def ping():\n    return 2\n")
        recorded = ("src/lib.py", "ping", old.blocks["ping"])
        assert stale_dep(recorded, {"src/lib.py"}, {"src/lib.py": new})

    def test_removed_block_stales(self):
        recorded = dep("src/lib.py", "ping")
        new = table_of("def other():\n    return 1\n")
        assert stale_dep(recorded, {"src/lib.py"}, {"src/lib.py": new})

    def test_wildcard_stales_on_any_change(self):
        assert stale_dep(("src/lib.py", WILDCARD, WILDCARD), {"src/lib.py"}, {"src/lib.py": table_of("x = 1\n")})

    def test_unparseable_changed_file_stales(self):
        assert stale_dep(dep("src/lib.py", "ping"), {"src/lib.py"}, {"src/lib.py": None})

    def test_module_skeleton_granularity(self):
        old = table_of("X = 1\n\ndef f():\n    return 1\n")
        # body edit: skeleton unchanged, f changed
        body_edit = table_of("X = 1\n\ndef f():\n    return 2\n")
        # module-level edit: skeleton changed, f unchanged
        module_edit = table_of("X = 2\n\ndef f():\n    return 1\n")
        rec_skel = ("lib.py", MODULE_BLOCK, old.blocks[MODULE_BLOCK])
        rec_f = ("lib.py", "f", old.blocks["f"])
        assert not stale_dep(rec_skel, {"lib.py"}, {"lib.py": body_edit})
        assert stale_dep(rec_skel, {"lib.py"}, {"lib.py": module_edit})
        assert stale_dep(rec_f, {"lib.py"}, {"lib.py": body_edit})
        assert not stale_dep(rec_f, {"lib.py"}, {"lib.py": module_edit})


class TestSelectTests:
    COLLECTED: ClassVar[list[str]] = [
        "tests/test_a.py::test_a",
        "tests/test_b.py::test_b",
        "tests/test_c.py::test_c",
    ]

    def test_no_shard_runs_all(self):
        sel = select_tests(self.COLLECTED, None, set(), {})
        assert sel.selected == self.COLLECTED
        assert sel.deselected == []

    def test_unknown_changes_runs_all(self):
        sel = select_tests(self.COLLECTED, make_shard(), None, {})
        assert sel.selected == self.COLLECTED

    def test_new_tests_always_run(self):
        sel = select_tests(self.COLLECTED, make_shard(), set(), {})
        assert "tests/test_c.py::test_c" in sel.selected
        assert "tests/test_a.py::test_a" in sel.deselected

    def test_stale_block_selects_dependents(self):
        shard = make_shard()
        old = table_of("def ping():\n    return 1\n")
        shard.tests["tests/test_a.py::test_a"] = {("src/lib.py", "ping", old.blocks["ping"])}
        new = table_of("def ping():\n    return 2\n")
        sel = select_tests(self.COLLECTED, shard, {"src/lib.py"}, {"src/lib.py": new})
        # the ping-dependent test and the new test are selected, nothing else
        assert sel.selected == ["tests/test_a.py::test_a", "tests/test_c.py::test_c"]
        assert "tests/test_b.py::test_b" in sel.deselected
        assert "tests/test_c.py::test_c" in sel.selected  # new

    def test_same_hash_does_not_select(self):
        shard = make_shard()
        fb = table_of("def ping():\n    return 1\n")
        shard.tests["tests/test_a.py::test_a"] = {("src/lib.py", "ping", fb.blocks["ping"])}
        sel = select_tests(self.COLLECTED, shard, {"src/lib.py"}, {"src/lib.py": fb})
        assert "tests/test_a.py::test_a" in sel.deselected

    def test_import_time_stale_selects_everyone(self):
        shard = make_shard()
        old_conf = table_of("X = 1\n")
        new_conf = table_of("X = 2\n")
        shard.import_deps = {("tests/conftest.py", MODULE_BLOCK, old_conf.blocks[MODULE_BLOCK])}
        sel = select_tests(
            self.COLLECTED,
            shard,
            {"tests/conftest.py"},
            {"tests/conftest.py": new_conf},
        )
        # both recorded tests run (import-time dep of every test), plus new:
        assert sel.selected == self.COLLECTED
        assert "import-time" in sel.reason

    def test_import_time_fresh_deselects(self):
        shard = make_shard()
        conf = table_of("X = 1\n")
        shard.import_deps = {("tests/conftest.py", MODULE_BLOCK, conf.blocks[MODULE_BLOCK])}
        sel = select_tests(
            self.COLLECTED,
            shard,
            {"tests/conftest.py"},
            {"tests/conftest.py": conf},
        )
        assert "tests/test_a.py::test_a" in sel.deselected

    def test_failed_tests_always_run(self):
        shard = make_shard(failed={"tests/test_b.py::test_b"})
        sel = select_tests(self.COLLECTED, shard, set(), {})
        assert "tests/test_b.py::test_b" in sel.selected

    def test_irrelevant_change_deselects_all_known(self):
        sel = select_tests(self.COLLECTED, make_shard(), {"README.md"}, {})
        assert sel.selected == ["tests/test_c.py::test_c"]
        assert sel.deselected == [
            "tests/test_a.py::test_a",
            "tests/test_b.py::test_b",
        ]
