"""Subprocess boundary: tests that spawn processes are never recorded.

The recorded run only sees what the *parent* process executed.  A test
whose child reads or runs repo code has dependencies no PY_START event will
ever attribute to it (the e2b blind spot: edit the child-only ``worker.py``,
watch the test fail with "0 tests selected").  Coverage cannot follow the
child, so the recorder stops collecting for a test the moment it spawns a
process, writes **no dependencies** for it, and selection re-runs it every
time.

A spawn *during collection* has an unknown culprit: the shard is flagged
``spawn_unknown`` and falls back to the weaker rule (every test runs when a
changed ``.py`` file is stale or was never recorded).  The same fallback
covers shards from before this feature.

Scope, deliberately: spawns are attributed during collect/test phases only
(pytest's own startup git subprocesses live in the import phase and must not
mark every test a spawner).
"""

from typing import ClassVar

import pytest
from test_plugin import commit, decode_shard, dep_ids, git, run_less, shard_data


@pytest.fixture
def spawn_project(pytester):
    """A git repo with one test that spawns ``worker.py`` in a child process
    and one test that touches nothing.  ``worker.py`` is never imported by
    the parent -- it exists only for the child."""
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (pytester.path / "worker.py").write_text(
        "def main():\n"
        '    return "v1"\n'
        'if __name__ == "__main__":\n'
        "    print(main())\n"
    )
    (tests / "test_spawn.py").write_text(
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "def test_spawn():\n"
        "    root = pathlib.Path(__file__).resolve().parent.parent\n"
        "    out = subprocess.run(\n"
        "        [sys.executable, str(root / 'worker.py')],\n"
        "        capture_output=True, text=True,\n"
        "    )\n"
        '    assert "v1" in out.stdout\n'
    )
    (tests / "test_observer.py").write_text(
        "def test_observer():\n    assert True\n"
    )
    (pytester.path / ".gitignore").write_text(
        ".pytest_cache/\n__pycache__/\nrunpytest-*\nstdout\nstderr\n"
    )
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


def read_meta(path):
    import json
    import subprocess

    refs = [
        line
        for line in git(path, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.splitlines()
        if line
    ]
    assert len(refs) == 1
    return json.loads(
        subprocess.run(
            ["git", "-C", str(path), "cat-file", "blob", f"{refs[0]}:meta.json"],
            capture_output=True, text=True, check=True,
        ).stdout
    )


class TestSpawnRecording:
    def test_record_writes_no_dependencies_for_the_spawner(self, spawn_project):
        result = run_less(spawn_project, "-q")
        result.assert_outcomes(passed=2)

        data = shard_data(spawn_project.path)
        spawner = "tests/test_spawn.py::test_spawn"
        assert data["always"] == [spawner]
        assert data["spawn_unknown"] is False
        meta = read_meta(spawn_project.path)
        assert meta["always"] == [spawner]
        tests, _ = decode_shard(data)
        # Collection stopped at the spawn and nothing was written.
        assert tests[spawner] == set()
        # The observer is recorded normally.
        assert "tests/test_observer.py:test_observer" in dep_ids(
            tests["tests/test_observer.py::test_observer"]
        )
        # worker.py is a child-only file: no recorded dependency knows it.
        assert "worker.py" not in meta["dep_files"]

    def test_no_change_rerun_skips_the_spawner_too(self, spawn_project):
        # Nothing changed, same commit, and the snapshot watches every tracked
        # file (a child could read any of them): the answer is the answer
        # from last time -- "no change" beats "always run".
        run_less(spawn_project, "-q")
        result = run_less(spawn_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])
        result.stdout.fnmatch_lines(["*fast path*"])
        data = shard_data(spawn_project.path)
        assert data["always"] == ["tests/test_spawn.py::test_spawn"]  # still a spawner

    def test_a_test_that_stops_spawning_is_recorded_again(self, spawn_project):
        run_less(spawn_project, "-q")
        (spawn_project.path / "tests" / "test_spawn.py").write_text(
            "def test_spawn():\n    assert True\n"
        )
        result = run_less(spawn_project, "-q")
        result.assert_outcomes(passed=1)
        data = shard_data(spawn_project.path)
        assert data["always"] == []
        tests, _ = decode_shard(data)
        assert "tests/test_spawn.py:test_spawn" in dep_ids(
            tests["tests/test_spawn.py::test_spawn"]
        )


class TestChildOnlyEdit:
    def test_edit_reruns_the_spawner_and_only_the_spawner(self, spawn_project):
        # The e2b blind spot: worker.py changed, but it is not a dependency
        # of any test -- only the spawner's child can be affected by it.
        run_less(spawn_project, "-q")
        (spawn_project.path / "worker.py").write_text(
            "def main():\n"
            '    return "v1.1"\n'
            'if __name__ == "__main__":\n'
            "    print(main())\n"
        )
        result = run_less(spawn_project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
        result.stdout.fnmatch_lines(["*test_spawn.py::test_spawn PASSED*"])

    def test_child_only_edit_that_breaks_the_child_is_caught(self, spawn_project):
        run_less(spawn_project, "-q")
        (spawn_project.path / "worker.py").write_text(
            "def main():\n"
            '    return "broken"\n'
            'if __name__ == "__main__":\n'
            "    print(main())\n"
        )
        result = run_less(spawn_project, "-q")
        result.assert_outcomes(failed=1)

    def test_non_py_change_still_reruns_the_spawner(self, spawn_project):
        run_less(spawn_project, "-q")
        (spawn_project.path / "README.md").write_text("docs\n")
        result = run_less(spawn_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])


@pytest.fixture
def collect_spawn_project(pytester):
    """A test module that spawns a process at *import* (collection) time."""
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "test_one.py").write_text(
        "import subprocess\n"
        "import sys\n"
        "subprocess.run([sys.executable, '-c', 'pass'])\n"
        "\n"
        "def test_one():\n"
        "    assert True\n"
    )
    (tests / "test_two.py").write_text("def test_two():\n    assert True\n")
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


class TestCollectPhaseSpawn:
    def test_unknown_culprit_falls_back_to_the_weak_rule(self, collect_spawn_project):
        run_less(collect_spawn_project, "-q").assert_outcomes(passed=2)
        data = shard_data(collect_spawn_project.path)
        assert data["spawn_unknown"] is True
        assert data["always"] == []

        # No change: nothing to do (the weak rule needs a change to fire).
        result = run_less(collect_spawn_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

        # A never-recorded .py file appears: everyone is a suspect.
        (collect_spawn_project.path / "extra.py").write_text("X = 1\n")
        result = run_less(collect_spawn_project, "-q")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*subprocess rule*"])


class TestSpawnAttribution:
    """Unit: the audit hook's phase policy (no real tool id needed)."""

    @staticmethod
    def make(tmp_path):
        from pytest_less.monitoring import FileMonitor

        m = FileMonitor(tmp_path)
        # White-box: the hook body only needs the active flag; no real
        # sys.monitoring tool is registered for this unit test.
        m._active = True
        return m

    def test_test_phase_marks_the_current_test(self, tmp_path):
        m = self.make(tmp_path)
        m.phase = "test"
        m.current_nodeid = "tests/test_x.py::test_x"
        m._note_spawn()
        assert m.spawn_tests == {"tests/test_x.py::test_x"}
        assert not m.spawn_shared

    def test_collection_stops_for_a_spawner(self, tmp_path):
        from pytest_less.blocks import Block

        m = self.make(tmp_path)
        m.phase = "test"
        m.current_nodeid = "t"
        before: Block = ("a.py", "f", "h1")
        after: Block = ("b.py", "g", "h2")
        m._bucket((before,))
        m._note_spawn()
        m._bucket((after,))
        assert m.test_deps["t"] == {before}  # nothing recorded after the spawn
        m.current_nodeid = "other"
        m._bucket((after,))
        assert m.test_deps["other"] == {after}  # other tests are unaffected

    def test_real_spawns_are_intercepted_and_restored(self, tmp_path):
        # End to end on the wrappers (a real tool id, real subprocesses):
        # every Python-level way to create a process is seen, and the
        # originals are back after deactivate().
        import os
        import subprocess
        import sys

        from pytest_less.monitoring import FileMonitor

        originals = (subprocess.Popen.__init__, os.system, os.fork)
        m = FileMonitor(tmp_path)
        assert m.activate()
        try:
            for i, run in enumerate(
                (
                    lambda: subprocess.run([sys.executable, "-c", "pass"], check=False),
                    lambda: os.system("true"),
                )
            ):
                m.phase = "test"
                m.current_nodeid = f"t{i}"
                run()
            assert m.spawn_tests == {"t0", "t1"}
            assert subprocess.Popen.__init__ is not originals[0]  # wrapped
        finally:
            m.deactivate()
        assert (subprocess.Popen.__init__, os.system, os.fork) == originals

    def test_no_audit_hook_is_installed(self, tmp_path):
        # The whole point of the redesign: zero per-event cost.
        import sys

        from pytest_less.monitoring import FileMonitor

        seen = []
        real = sys.addaudithook
        sys.addaudithook = lambda f: seen.append(f)
        try:
            m = FileMonitor(tmp_path)
            assert m.activate()
            m.deactivate()
        finally:
            sys.addaudithook = real
        assert seen == []

    def test_import_phase_is_ignored(self, tmp_path):
        # pytest's own startup git subprocesses run in this phase: marking
        # shared here would make every test a spawner and kill precision.
        m = self.make(tmp_path)
        m.phase = "import"
        m._note_spawn()
        assert m.spawn_tests == set()
        assert not m.spawn_shared

    def test_collect_phase_marks_shared(self, tmp_path):
        m = self.make(tmp_path)
        m.phase = "collect"
        m._note_spawn()
        assert m.spawn_shared is True
        assert m.spawn_tests == set()

    def test_done_phase_and_inactive_hook_ignore(self, tmp_path):
        m = self.make(tmp_path)
        m.phase = "test"
        m.current_nodeid = "t"
        m.phase = "done"
        m._note_spawn()
        assert m.spawn_tests == set()
        inactive = self.make(tmp_path)
        inactive.phase = "test"
        inactive.current_nodeid = "t"
        inactive._active = False
        inactive._note_spawn()
        assert inactive.spawn_tests == set()


class TestSpawnRuleUnit:
    """Unit: the selection side (always-rerun set + weak unknown rule)."""

    @staticmethod
    def make_shard(always=(), unknown=False):
        from pytest_less.blocks import MODULE_BLOCK
        from pytest_less.selection import Shard

        return Shard(
            base_commit="ab" * 20,
            tests={
                "tests/test_s.py::test_s": set(),  # spawner: no deps
                "tests/test_o.py::test_o": {("tests/test_o.py", "test_o", "h0")},
            },
            import_deps={("tests/conftest.py", MODULE_BLOCK, "h0")},
            failed=set(),
            always=set(always),
            spawn_unknown=unknown,
        )

    COLLECTED: ClassVar[list[str]] = [
        "tests/test_s.py::test_s",
        "tests/test_o.py::test_o",
        "tests/test_new.py::test_new",
    ]

    @staticmethod
    def maps_for(path, source="def main():\n    return 1\n"):
        from pytest_less.blocks import parse_blocks

        return {path: parse_blocks(source)}

    def test_spawner_runs_even_with_no_changes(self):
        from pytest_less.selection import select_tests

        shard = self.make_shard(always={"tests/test_s.py::test_s"})
        sel = select_tests(self.COLLECTED, shard, set(), {})
        assert sel.selected == ["tests/test_s.py::test_s", "tests/test_new.py::test_new"]
        assert sel.deselected == ["tests/test_o.py::test_o"]
        assert "always re-run" in sel.reason

    def test_unknown_shard_is_conservatively_all_spawners(self):
        from pytest_less.selection import load_shard, select_tests

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}, "import_deps": [], "failed": []},
            {"base_commit": None},
        )
        assert shard is not None and shard.spawn_unknown  # pre-feature shard
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, self.maps_for("worker.py")
        )
        # everyone is a suspect: the observer is dragged in too (Ley-1
        # direction), and the next full record run self-heals the flag.
        assert set(sel.selected) == set(self.COLLECTED)

    def test_unknown_flag_needs_a_py_change_to_fire(self):
        from pytest_less.selection import select_tests

        shard = self.make_shard(unknown=True)
        sel = select_tests(self.COLLECTED, shard, {"notes.md"}, {})
        assert "tests/test_o.py::test_o" in sel.deselected

    def test_no_spawners_means_no_selection(self):
        from pytest_less.selection import select_tests

        shard = self.make_shard()
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, self.maps_for("worker.py")
        )
        assert "tests/test_o.py::test_o" in sel.deselected

    def test_fresh_recorded_py_file_never_triggers_the_weak_rule(self):
        from pytest_less.blocks import parse_blocks
        from pytest_less.selection import select_tests

        shard = self.make_shard(unknown=True)
        fb = parse_blocks("def main():\n    return 1\n")
        shard.tests["tests/test_o.py::test_o"] = {
            ("worker.py", "main", fb.blocks["main"])
        }
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, {"worker.py": fb}
        )
        # changed but byte-identical at block level: not stale, not unknown
        assert "tests/test_o.py::test_o" in sel.deselected

    def test_stale_recorded_py_file_triggers_the_weak_rule(self):
        from pytest_less.blocks import parse_blocks
        from pytest_less.selection import select_tests

        shard = self.make_shard(unknown=True)
        old = parse_blocks("def main():\n    return 1\n")
        new = parse_blocks("def main():\n    return 2\n")
        shard.tests["tests/test_o.py::test_o"] = {
            ("worker.py", "main", old.blocks["main"])
        }
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, {"worker.py": new}
        )
        assert "tests/test_s.py::test_s" in sel.selected  # weak rule
        assert "tests/test_o.py::test_o" in sel.selected  # stale dep


class TestLoadSpawning:
    def test_always_list_parses(self):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}, "always": ["a", 3]},
            {"base_commit": None},
        )
        assert shard is not None
        assert shard.always == {"a"}  # non-strings dropped
        assert shard.spawn_unknown is False

    def test_spawn_unknown_flag_parses(self):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {
                "version": 2,
                "blocks": [],
                "tests": {},
                "always": [],
                "spawn_unknown": True,
            },
            {"base_commit": None},
        )
        assert shard is not None and shard.spawn_unknown is True

    @pytest.mark.parametrize("value", [None, 7, "x"])
    def test_corrupt_always_is_conservative(self, value):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}, "always": value},
            {"base_commit": None},
        )
        assert shard is not None
        assert shard.spawn_unknown is True

    def test_absent_key_is_conservative(self):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}},
            {"base_commit": None},
        )
        assert shard is not None
        assert shard.spawn_unknown is True
