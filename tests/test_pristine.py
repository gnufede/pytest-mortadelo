"""At a pristine state "no change" beats "always run".

A test that spawns a process is never given dependencies (coverage cannot
follow the child), so it used to re-run on *every* run.  On this repo's own
suite that was 198 of 400 tests: a rerun with nothing changed took 41.9 s.
The reason for the rule -- a child may read any file -- also says when it can be
relaxed: if **nothing observable changed**, the outcome is the outcome from last
time.  "Nothing" must be provable, so a state is *pristine* only if

* the change check reports nothing,
* ``HEAD`` is still the commit the shard was recorded at, and
* that check covered everything a child could read: the snapshot was widened to
  every tracked file (plus every file name in the watched directories), or the
  check was ``git diff`` + untracked, which covers the whole repository.

Anything else keeps the old behavior.
"""

import os
import shutil
import subprocess
import sys
import time

import pytest
from test_plugin import commit, git, run_less, shard_data  # noqa: F401

from pytest_less import fingerprint
from pytest_less.blocks import MODULE_BLOCK
from pytest_less.fingerprint import build_snapshot, detect_changes
from pytest_less.gitstore import GitStore
from pytest_less.selection import Shard, affected_reason, build_context, select_tests

SPAWNER = "tests/test_spawn.py::test_spawn"
OBSERVER = "tests/test_observer.py::test_observer"


@pytest.fixture
def child_project(pytester):
    """A test that spawns ``scripts/child.py`` -- a directory no recorded
    dependency lives in -- and one that touches nothing."""
    root = pytester.path
    (root / "tests").mkdir()
    (root / "scripts").mkdir()
    (root / "tests" / "conftest.py").touch()
    (root / "scripts" / "child.py").write_text('print("v1")\n')
    (root / "docs.txt").write_text("docs v1\n")
    (root / "tests" / "test_spawn.py").write_text(
        "import pathlib, subprocess, sys\n"
        "\n"
        "def test_spawn():\n"
        "    root = pathlib.Path(__file__).resolve().parent.parent\n"
        "    out = subprocess.run(\n"
        "        [sys.executable, str(root / 'scripts' / 'child.py')],\n"
        "        capture_output=True, text=True, check=False,\n"
        "    )\n"
        '    assert "v1" in out.stdout\n'
    )
    (root / "tests" / "test_observer.py").write_text(
        "def test_observer():\n    assert True\n"
    )
    (root / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n*.log\nrunpytest-*\nstdout\nstderr\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "T")
    commit(root, "initial")
    return pytester


class TestSkippedWhenNothingChanged:
    def test_no_change_rerun_runs_nothing_at_all(self, child_project):
        run_less(child_project, "-q").assert_outcomes(passed=2)
        for _ in range(2):
            result = run_less(child_project, "-q")
            assert result.ret == 5
            result.stdout.fnmatch_lines(["*selected 0/2 tests*", "*fast path*"])

    def test_the_snapshot_was_widened_because_a_test_spawns(self, child_project):
        run_less(child_project, "-q")
        store = GitStore(child_project.path, _shard(child_project))
        snapshot = store.load_files()
        assert snapshot["all"] is True
        # everything tracked is fingerprinted, including files no test imports
        assert {"scripts/child.py", "docs.txt", ".gitignore"} <= set(snapshot["files"])
        assert "docs.txt" in snapshot["names"][""]

    def test_without_spawners_the_snapshot_stays_narrow(self, pytester):
        (pytester.path / "tests").mkdir()
        (pytester.path / "tests" / "conftest.py").touch()
        (pytester.path / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")
        (pytester.path / "unrelated.txt").write_text("x\n")
        (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
        git(pytester.path, "init", "-q")
        git(pytester.path, "config", "user.email", "t@example.com")
        git(pytester.path, "config", "user.name", "T")
        commit(pytester.path, "initial")
        run_less(pytester, "-q").assert_outcomes(passed=1)
        snapshot = GitStore(pytester.path, _shard(pytester)).load_files()
        assert snapshot["all"] is False and "unrelated.txt" not in snapshot["files"]

    def test_pristine_rerun_spawns_no_git_process(self, child_project, tmp_path, monkeypatch):
        run_less(child_project, "-q")
        real = shutil.which("git")
        log = tmp_path / "calls.log"
        shim = tmp_path / "shim"
        shim.mkdir()
        script = shim / "git"
        script.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexec "{real}" "$@"\n')
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
        result = child_project.run(sys.executable, "-m", "pytest", "--less", "-q")
        assert result.ret == 5
        assert not log.exists() or log.read_text().strip() == ""


class TestStillWokenByAnyChange:
    def test_child_only_code_in_a_directory_nothing_depends_on(self, child_project):
        # The e2b blind spot, in the one place only the widened snapshot can
        # see: scripts/ holds no dependency, so it is not a watched directory.
        run_less(child_project, "-q").assert_outcomes(passed=2)
        (child_project.path / "scripts" / "child.py").write_text('print("v1")  # x\n')
        result = run_less(child_project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_spawn.py::test_spawn PASSED*"])

    def test_a_child_that_now_fails_is_caught(self, child_project):
        run_less(child_project, "-q")
        (child_project.path / "scripts" / "child.py").write_text('print("broken")\n')
        run_less(child_project, "-q").assert_outcomes(failed=1)

    def test_an_edit_to_a_tracked_non_python_file(self, child_project):
        run_less(child_project, "-q")
        (child_project.path / "docs.txt").write_text("docs v2, longer\n")
        result = run_less(child_project, "-q")
        result.assert_outcomes(passed=1)

    def test_a_new_untracked_file_beside_the_dependencies(self, child_project):
        run_less(child_project, "-q")
        (child_project.path / "notes.md").write_text("new\n")
        run_less(child_project, "-q").assert_outcomes(passed=1)

    def test_a_new_ignored_file_does_not_wake_it(self, child_project):
        run_less(child_project, "-q")
        (child_project.path / "debug.log").write_text("noise\n")  # *.log is ignored
        result = run_less(child_project, "-q")
        assert result.ret == 5

    def test_a_new_hidden_file_does_not_wake_it(self, child_project):
        run_less(child_project, "-q")
        (child_project.path / ".scratch").write_text("noise\n")
        assert run_less(child_project, "-q").ret == 5

    def test_a_new_commit_wakes_it_once_and_is_then_pristine_again(self, child_project):
        run_less(child_project, "-q")
        git(child_project.path, "commit", "-q", "--allow-empty", "-m", "an empty commit")
        woke = run_less(child_project, "-q")
        woke.assert_outcomes(passed=1)  # HEAD moved: not provably the same state
        again = run_less(child_project, "-q")
        assert again.ret == 5  # re-recorded at the new commit: pristine again

    def test_a_previously_failed_test_still_reruns(self, child_project):
        (child_project.path / "tests" / "test_observer.py").write_text(
            "def test_observer():\n    assert False\n"
        )
        commit(child_project.path, "break the observer")
        run_less(child_project, "-q")  # records; the observer fails
        result = run_less(child_project, "-v")
        result.assert_outcomes(failed=1)  # the failure reruns; the spawner does not
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])


def drop_fingerprints(project):
    """Rewrite the shard as an older version did: no ``files.json``, so change
    detection has to ask ``git diff``."""
    store = GitStore(project.path, _shard(project))
    data, meta, parse = store.load_data(), store.load_meta(), store.load_parse()
    store.delete()
    store.save(data, meta, parse=parse)


class TestGitBackend:
    def test_git_diff_covers_the_whole_repository_so_it_is_pristine_too(
        self, child_project
    ):
        run_less(child_project, "-q").assert_outcomes(passed=2)
        drop_fingerprints(child_project)
        result = run_less(child_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_and_an_edit_anywhere_wakes_it(self, child_project):
        run_less(child_project, "-q")
        drop_fingerprints(child_project)
        (child_project.path / "scripts" / "child.py").write_text('print("v1")  # y\n')
        run_less(child_project, "-q").assert_outcomes(passed=1)


# ---------------------------------------------------------------------------
# Unit: the pieces.


def _shard(project):
    ref = git(project.path, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.split()[0]
    return ref.rsplit("/", 1)[-1]


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "w"
    for rel in ("pkg/a.py", "tests/test_t.py", "tools/run.py"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n")
        old = time.time() - 3600
        os.utime(path, (old, old))
    (root / "etc").mkdir()
    (root / "etc" / "conf.ini").write_text("k=v\n")
    os.utime(root / "etc" / "conf.ini", (time.time() - 3600,) * 2)
    return root


DEPS = ["pkg/a.py", "tests/test_t.py"]
TRACKED = ["pkg/a.py", "tests/test_t.py", "tools/run.py", "etc/conf.ini"]


class TestWidenedSnapshot:
    def test_everything_tracked_is_fingerprinted_and_flagged(self, tree):
        s = build_snapshot(tree, DEPS, extra_files=TRACKED)
        assert s["all"] is True
        assert set(TRACKED) <= set(s["files"])
        assert "conf.ini" in s["names"]["etc"] if "etc" in s["names"] else True

    def test_the_narrow_snapshot_is_not_flagged(self, tree):
        s = build_snapshot(tree, DEPS)
        assert s["all"] is False and "names" not in s and "tools/run.py" not in s["files"]

    def test_an_edit_to_a_file_no_test_imports_is_seen_only_when_widened(self, tree):
        wide = build_snapshot(tree, DEPS, extra_files=TRACKED)
        narrow = build_snapshot(tree, DEPS)
        (tree / "tools" / "run.py").write_text("x = 2  # changed\n")
        (tree / "etc" / "conf.ini").write_text("k=changed\n")
        assert detect_changes(tree, wide, lambda p: set()) == {"tools/run.py", "etc/conf.ini"}
        assert detect_changes(tree, narrow, lambda p: set()) == set()

    def test_a_new_non_python_file_in_a_watched_directory(self, tree):
        wide = build_snapshot(tree, DEPS, extra_files=TRACKED)
        (tree / "pkg" / "new.dat").write_text("d\n")
        asked = []

        def confirm(paths):
            asked.append(paths)
            return set(paths)

        assert detect_changes(tree, wide, confirm) == {"pkg/new.dat"}
        assert asked == [["pkg/new.dat"]]
        # the narrow snapshot never cares about non-Python newcomers
        narrow = build_snapshot(tree, DEPS)
        assert detect_changes(tree, narrow, lambda p: set(p)) == set()

    def test_hidden_newcomers_and_ignored_ones_do_not_count(self, tree):
        wide = build_snapshot(tree, DEPS, extra_files=TRACKED)
        (tree / "pkg" / ".hidden").write_text("h\n")
        (tree / "pkg" / "ignored.dat").write_text("i\n")
        assert detect_changes(tree, wide, lambda p: set()) == set()  # git says: ignored

    def test_large_files_are_fingerprinted_by_stat_alone(self, tree, monkeypatch):
        monkeypatch.setattr(fingerprint, "EXTRA_HASH_LIMIT", 3)
        s = build_snapshot(tree, DEPS, extra_files=TRACKED)
        assert s["files"]["etc/conf.ini"][2] == ""  # too big to read: stat-only
        assert s["files"]["pkg/a.py"][2] != ""  # dependencies are always hashed
        assert detect_changes(tree, s, lambda p: set()) == set()  # untouched: clean
        later = time.time() + 30
        os.utime(tree / "etc" / "conf.ini", (later, later))  # touched: unverifiable
        assert detect_changes(tree, s, lambda p: set()) == {"etc/conf.ini"}


class TestIsPristine:
    @pytest.fixture
    def repo(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        git(root, "init", "-q")
        git(root, "config", "user.email", "t@example.com")
        git(root, "config", "user.name", "T")
        (root / "f.py").write_text("x = 1\n")
        commit(root, "initial")
        return root

    def head(self, repo):
        return git(repo, "rev-parse", "HEAD").stdout.strip()

    def test_needs_an_empty_change_set_at_the_same_commit(self, repo):
        store = GitStore(repo, "k")
        store.change_source = "git"
        head = self.head(repo)
        assert store.is_pristine(head, set()) is True
        assert store.is_pristine(head, {"f.py"}) is False
        assert store.is_pristine(head, None) is False
        assert store.is_pristine(None, set()) is False
        assert store.is_pristine("0" * 40, set()) is False  # another commit

    def test_a_new_commit_is_not_pristine(self, repo):
        store = GitStore(repo, "k")
        store.change_source = "git"
        base = self.head(repo)
        git(repo, "commit", "-q", "--allow-empty", "-m", "next")
        assert store.is_pristine(base, set()) is False

    def test_fingerprints_only_count_when_the_snapshot_was_widened(self, repo):
        store = GitStore(repo, "k")
        head = self.head(repo)
        store.change_source = "fingerprints"
        store.save({"version": 2}, {"version": 1}, files=build_snapshot(repo, ["f.py"]))
        assert store.is_pristine(head, set()) is False  # narrow: a child might read more
        store.save(
            {"version": 2}, {"version": 1},
            files=build_snapshot(repo, ["f.py"], extra_files=["f.py"]),
        )
        assert store.is_pristine(head, set()) is True

    def test_tracked_files_respects_its_limit(self, repo):
        store = GitStore(repo, "k")
        assert store.tracked_files(10) == ["f.py"]
        assert store.tracked_files(0) is None


class TestSelectionRule:
    def shard(self):
        return Shard(
            base_commit="ab" * 20,
            tests={
                "t::spawns": set(),
                "t::plain": {("m.py", "f", "h")},
            },
            import_deps={("c.py", MODULE_BLOCK, "h")},
            failed=set(),
            always={"t::spawns"},
        )

    def test_always_wins_over_nothing_changed_only_when_not_pristine(self):
        shard = self.shard()
        ctx = build_context(shard, set(), {}, pristine=False)
        assert affected_reason("t::spawns", set(), shard, ctx) == "always"
        ctx = build_context(shard, set(), {}, pristine=True)
        assert affected_reason("t::spawns", set(), shard, ctx) is None

    def test_select_tests_reports_the_skip(self):
        shard = self.shard()
        collected = ["t::spawns", "t::plain"]
        running = select_tests(collected, shard, set(), {}, pristine=False)
        assert running.selected == ["t::spawns"] and "always re-run" in running.reason
        skipped = select_tests(collected, shard, set(), {}, pristine=True)
        assert skipped.selected == []
        assert "always-run test(s) skipped" in skipped.reason

    def test_pristine_does_not_hide_a_failed_spawner(self):
        shard = self.shard()
        shard.failed = {"t::spawns"}
        sel = select_tests(["t::spawns", "t::plain"], shard, set(), {}, pristine=True)
        assert sel.selected == ["t::spawns"]


def test_subprocess_import_guard():
    # importing the module must not need git
    out = subprocess.run(
        [sys.executable, "-c", "import pytest_less.fingerprint, pytest_less.selection"],
        capture_output=True,
        check=False,
    )
    assert out.returncode == 0
