from typing import ClassVar

from pytest_less.selection import Shard, load_shard, select_tests


def make_shard(**overrides):
    shard = Shard(
        base_commit="ab" * 20,
        tests={
            "tests/test_a.py::test_a": {"src/lib.py", "tests/test_a.py"},
            "tests/test_b.py::test_b": {"tests/test_b.py"},
        },
        failed=set(),
    )
    for key, value in overrides.items():
        setattr(shard, key, value)
    return shard


class TestLoadShard:
    def test_valid(self):
        shard = load_shard(
            {
                "version": 1,
                "tests": {"a": ["x.py"]},
                "failed": ["a"],
            },
            {"base_commit": "c0ffee"},
        )
        assert shard is not None
        assert shard.tests == {"a": {"x.py"}}
        assert shard.failed == {"a"}
        assert shard.base_commit == "c0ffee"

    def test_bad_version(self):
        assert load_shard({"version": 2, "tests": {}}, {"base_commit": None}) is None

    def test_corrupt(self):
        assert load_shard({"version": 1}, {}) is None  # missing "tests"
        assert load_shard(None, None) is None

    def test_missing_base_commit_ok(self):
        shard = load_shard({"version": 1, "tests": {}, "failed": []}, {})
        assert shard is not None
        assert shard.base_commit is None


class TestSelectTests:
    COLLECTED: ClassVar[list[str]] = [
        "tests/test_a.py::test_a",
        "tests/test_b.py::test_b",
        "tests/test_c.py::test_c",
    ]

    def test_no_shard_runs_all(self):
        sel = select_tests(self.COLLECTED, None, set())
        assert sel.selected == self.COLLECTED
        assert sel.deselected == []

    def test_unknown_changes_runs_all(self):
        sel = select_tests(self.COLLECTED, make_shard(), None)
        assert sel.selected == self.COLLECTED

    def test_new_tests_always_run(self):
        sel = select_tests(self.COLLECTED, make_shard(), set())
        assert "tests/test_c.py::test_c" in sel.selected
        assert "tests/test_a.py::test_a" in sel.deselected

    def test_affected_tests_run(self):
        sel = select_tests(self.COLLECTED, make_shard(), {"src/lib.py"})
        assert sel.selected == [
            "tests/test_a.py::test_a",
            "tests/test_c.py::test_c",  # no data: always runs
        ]
        assert sel.deselected == ["tests/test_b.py::test_b"]

    def test_failed_tests_always_run(self):
        shard = make_shard(failed={"tests/test_b.py::test_b"})
        sel = select_tests(self.COLLECTED, shard, set())
        assert "tests/test_b.py::test_b" in sel.selected

    def test_irrelevant_change_deselects_all_known(self):
        sel = select_tests(self.COLLECTED, make_shard(), {"README.md"})
        assert sel.selected == ["tests/test_c.py::test_c"]
        assert sel.deselected == ["tests/test_a.py::test_a", "tests/test_b.py::test_b"]
# dogfood demo: touch one test file
