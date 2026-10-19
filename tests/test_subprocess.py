"""Tier-A subprocess boundary: tests that spawn processes (closes e2b).

The recorded run only sees what the *parent* process executed.  A test
whose child reads or runs repo code has dependencies no PY_START event will
ever attribute to it: the child-only ``worker.py`` edit selected 0/1 while
the test failed.  Tier A marks spawning tests at record time (audit event)
and widens them at selection: a spawner runs when any changed ``.py`` file
is stale *or was never recorded at all* -- the child may have executed code
this recording never saw.

Scope, deliberately: .py files only (child *code* execution; child reads of
data files remain the --less-data-files escape hatch), attributed during
collect/test phases (pytest's own startup git subprocesses live in the
import phase and must not mark every test a spawner).
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
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
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
    def test_record_marks_only_the_spawning_test(self, spawn_project):
        result = run_less(spawn_project, "-q")
        result.assert_outcomes(passed=2)

        data = shard_data(spawn_project.path)
        assert data["spawning"] == ["tests/test_spawn.py::test_spawn"]
        meta = read_meta(spawn_project.path)
        assert meta["spawning"] == ["tests/test_spawn.py::test_spawn"]
        # worker.py is a child-only file: no recorded dependency knows it.
        assert "worker.py" not in meta["dep_files"]
        tests, _ = decode_shard(data)
        assert "worker.py:*" not in dep_ids(tests["tests/test_spawn.py::test_spawn"])

    def test_no_change_rerun_still_takes_the_fast_path(self, spawn_project):
        # The spawn rule needs changes to fire at all: a no-change rerun
        # must stay on the fast path even with spawners recorded.
        run_less(spawn_project, "-q")
        result = run_less(spawn_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*", "*fast path*"])


class TestChildOnlyEdit:
    def test_edit_selects_the_spawner_and_only_the_spawner(self, spawn_project):
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
        result.stdout.fnmatch_lines(["*selected by the subprocess rule*"])
        result.stdout.fnmatch_lines(["*test_spawn.py::test_spawn PASSED*"])

    def test_recorded_change_also_triggers_every_spawner(self, spawn_project):
        # Policy (documented precision price): once *any* changed .py is
        # stale, spawners run -- their children may execute the stale code
        # through paths the parent recording never saw.
        tests_dir = spawn_project.path / "tests"
        (tests_dir / "helper.py").write_text("def val():\n    return 1\n")
        (tests_dir / "test_third.py").write_text(
            "def test_third():\n"
            "    import helper\n"
            "    assert helper.val() == 1\n"
        )
        commit(spawn_project.path, "add helper")
        run_less(spawn_project, "-q").assert_outcomes(passed=3)

        (tests_dir / "helper.py").write_text("def val():\n    return 1 + 0\n")
        result = run_less(spawn_project, "-q")
        # test_third via staleness, test_spawn via the spawn rule,
        # test_observer (never touched helper, non-spawner) stays deselected.
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])

    def test_non_py_change_never_fires_the_rule(self, spawn_project):
        run_less(spawn_project, "-q")
        (spawn_project.path / "README.md").write_text("docs\n")
        result = run_less(spawn_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])


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
        m._on_audit("subprocess.Popen", (None,))
        assert m.spawn_tests == {"tests/test_x.py::test_x"}
        assert not m.spawn_shared

    def test_spawn_events_cover_system_and_fork(self, tmp_path):
        m = self.make(tmp_path)
        m.phase = "test"
        m.current_nodeid = "t"
        for event in ("os.system", "os.fork", "pty.spawn"):
            m._on_audit(event, ())
        assert m.spawn_tests == {"t"}

    def test_import_phase_is_ignored(self, tmp_path):
        # pytest's own startup git subprocesses run in this phase: marking
        # shared here would make every test a spawner and kill precision.
        m = self.make(tmp_path)
        m.phase = "import"
        m._on_audit("subprocess.Popen", (None,))
        assert m.spawn_tests == set()
        assert not m.spawn_shared

    def test_collect_phase_marks_shared(self, tmp_path):
        m = self.make(tmp_path)
        m.phase = "collect"
        m._on_audit("subprocess.Popen", (None,))
        assert m.spawn_shared is True
        assert m.spawn_tests == set()

    def test_done_phase_and_inactive_hook_ignore(self, tmp_path):
        m = self.make(tmp_path)
        m.phase = "test"
        m.current_nodeid = "t"
        m.phase = "done"
        m._on_audit("subprocess.Popen", (None,))
        assert m.spawn_tests == set()
        inactive = self.make(tmp_path)
        inactive.phase = "test"
        inactive.current_nodeid = "t"
        inactive._active = False
        inactive._on_audit("subprocess.Popen", (None,))
        assert inactive.spawn_tests == set()


class TestSpawnRuleUnit:
    """Unit: the selection-side rule (stale/unknown .py + spawner set)."""

    @staticmethod
    def make_shard(spawning):
        from pytest_less.blocks import MODULE_BLOCK
        from pytest_less.selection import Shard

        return Shard(
            base_commit="ab" * 20,
            tests={
                "tests/test_s.py::test_s": {("tests/test_s.py", "test_s", "h0")},
                "tests/test_o.py::test_o": {("tests/test_o.py", "test_o", "h0")},
            },
            import_deps={("tests/conftest.py", MODULE_BLOCK, "h0")},
            failed=set(),
            spawning=spawning,
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

    def test_unknown_py_file_selects_only_the_spawner(self):
        from pytest_less.selection import select_tests

        shard = self.make_shard({"tests/test_s.py::test_s"})
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, self.maps_for("worker.py")
        )
        assert sel.selected == ["tests/test_s.py::test_s", "tests/test_new.py::test_new"]
        assert sel.deselected == ["tests/test_o.py::test_o"]
        assert "subprocess rule" in sel.reason

    def test_unknown_shard_is_conservatively_all_spawners(self):
        from pytest_less.selection import load_shard, select_tests

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}, "import_deps": [], "failed": []},
            {"base_commit": None},
        )
        assert shard is not None and shard.spawning is None  # pre-Tier-A shard
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, self.maps_for("worker.py")
        )
        # everyone spawns: the observer is dragged in too (Ley-1 direction),
        # and the next record run self-heals the list.
        assert set(sel.selected) == set(self.COLLECTED)

    def test_no_spawners_means_no_selection(self):
        from pytest_less.selection import select_tests

        shard = self.make_shard(set())
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, self.maps_for("worker.py")
        )
        assert sel.deselected == [
            "tests/test_s.py::test_s",
            "tests/test_o.py::test_o",
        ]

    def test_non_py_change_never_triggers(self):
        from pytest_less.selection import select_tests

        shard = self.make_shard({"tests/test_s.py::test_s"})
        sel = select_tests(self.COLLECTED, shard, {"notes.md"}, {})
        assert "tests/test_s.py::test_s" in sel.deselected

    def test_fresh_recorded_py_file_never_triggers(self):
        from pytest_less.blocks import parse_blocks
        from pytest_less.selection import select_tests

        shard = self.make_shard({"tests/test_s.py::test_s"})
        fb = parse_blocks("def main():\n    return 1\n")
        shard.tests["tests/test_o.py::test_o"] = {
            ("worker.py", "main", fb.blocks["main"])
        }
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, {"worker.py": fb}
        )
        # changed but byte-identical at block level: not stale, not unknown
        assert "tests/test_s.py::test_s" in sel.deselected

    def test_stale_recorded_py_file_triggers_spawners(self):
        from pytest_less.blocks import parse_blocks
        from pytest_less.selection import select_tests

        shard = self.make_shard({"tests/test_s.py::test_s"})
        old = parse_blocks("def main():\n    return 1\n")
        new = parse_blocks("def main():\n    return 2\n")
        shard.tests["tests/test_o.py::test_o"] = {
            ("worker.py", "main", old.blocks["main"])
        }
        sel = select_tests(
            self.COLLECTED, shard, {"worker.py"}, {"worker.py": new}
        )
        assert "tests/test_s.py::test_s" in sel.selected  # spawn rule
        assert "tests/test_o.py::test_o" in sel.selected  # stale dep


class TestLoadSpawning:
    def test_spawning_list_parses(self):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}, "spawning": ["a", 3]},
            {"base_commit": None},
        )
        assert shard is not None
        assert shard.spawning == {"a"}  # non-strings dropped

    @pytest.mark.parametrize("value", [None, 7, "x"])
    def test_null_or_corrupt_spawning_is_conservative(self, value):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}, "spawning": value},
            {"base_commit": None},
        )
        assert shard is not None
        assert shard.spawning is None  # -> treat every test as a spawner

    def test_absent_spawning_key_is_conservative(self):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}},
            {"base_commit": None},
        )
        assert shard is not None
        assert shard.spawning is None
