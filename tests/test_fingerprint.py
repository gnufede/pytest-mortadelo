# ruff: noqa: F811  (the shared ``project`` fixture is imported, then requested by name)
"""Change detection from file fingerprints instead of ``git diff``.

The shard stores ``{size, mtime, hash}`` of every dependency file (and of the
``.py`` files next to them) plus the watched directory listing.  A run stats
those files and lists those directories; git is only asked -- once, about those
paths alone -- when something *new* appears.  The contract pinned here:

* it reports exactly what matters (dependency files and project ``.py`` files)
  and agrees with ``git diff`` + ``git ls-files --others`` on all of it;
* a ``touch``, a checkout that rewrites identical bytes, or a shard recorded on
  another machine (different mtimes) is not a change;
* a file modified within the racy window is never trusted on stat alone;
* it is independent of any base commit and of whether files are tracked.
"""

import os
import random
import subprocess
import sys
import time

import pytest
from test_plugin import commit, git, project, run_less  # noqa: F401

from pytest_less.blocks import _content_hash
from pytest_less.fingerprint import RACY_NS, build_snapshot, detect_changes
from pytest_less.gitstore import GitStore


@pytest.fixture(autouse=True)
def _fingerprints_by_default(monkeypatch):
    """These tests are about the fingerprint path: whatever backend the
    surrounding run was forced to, each test chooses its own."""
    monkeypatch.delenv("PYTEST_LESS_GIT_BACKEND", raising=False)


def write(root, rel, text="x = 1\n", age=None):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if age is not None:  # make it comfortably older than any snapshot
        old = time.time() - age
        os.utime(path, (old, old))
    return path


@pytest.fixture
def tree(tmp_path):
    """A small tree of *old* files: pkg/{a,b}.py, pkg/sub/c.py, tests/test_t.py,
    data/x.txt, setup.py."""
    root = tmp_path / "w"
    for rel in ("pkg/a.py", "pkg/b.py", "pkg/sub/c.py", "tests/test_t.py", "setup.py"):
        write(root, rel, age=3600)
    write(root, "data/x.txt", "payload\n", age=3600)
    return root


DEPS = ["pkg/a.py", "pkg/sub/c.py", "tests/test_t.py", "data/x.txt"]


def snap(root, deps=DEPS, digests=None):
    return build_snapshot(root, deps, digests)


class TestSnapshot:
    def test_records_dependencies_and_the_py_files_beside_them(self, tree):
        s = snap(tree)
        assert set(s["files"]) == {
            "pkg/a.py", "pkg/b.py", "pkg/sub/c.py", "tests/test_t.py", "data/x.txt", "setup.py",
        }
        size, mtime, digest = s["files"]["pkg/a.py"]
        assert size == len("x = 1\n") and digest == _content_hash(b"x = 1\n")
        assert mtime == (tree / "pkg/a.py").stat().st_mtime_ns

    def test_watches_every_ancestor_up_to_the_root(self, tree):
        s = snap(tree)
        assert set(s["dirs"]) == {"", "pkg", "pkg/sub", "tests", "data"}
        assert s["dirs"][""] == ["data", "pkg", "tests"]
        assert s["dirs"]["pkg"] == ["sub"]

    def test_never_lists_hidden_cache_or_environment_directories(self, tree):
        for d in (".git", ".hidden", "__pycache__", "node_modules", ".venv", "x.egg-info"):
            write(tree, f"{d}/z.py")
        assert snap(tree)["dirs"][""] == ["data", "pkg", "tests"]

    def test_a_dependency_that_vanished_is_simply_absent(self, tree):
        (tree / "data/x.txt").unlink()
        assert "data/x.txt" not in snap(tree)["files"]


class TestRebuildReusesHashes:
    """Rewriting a shard must cost stats, not reads, for files that provably
    did not change since the snapshot being replaced."""

    @pytest.fixture
    def reads(self, monkeypatch):
        import pytest_less.fingerprint as fp

        count = {"n": 0}
        real = fp._content_hash

        def counting(data):
            count["n"] += 1
            return real(data)

        monkeypatch.setattr(fp, "_content_hash", counting)
        return count

    def old_snapshot(self, tree):
        first = snap(tree)
        first["at"] = time.time_ns() + 3 * RACY_NS  # as if taken well after the files
        return first

    def test_unchanged_files_are_not_read_again(self, tree, reads):
        first = self.old_snapshot(tree)
        reads["n"] = 0
        second = build_snapshot(tree, DEPS, previous=first)
        assert reads["n"] == 0
        assert second["files"] == first["files"]

    def test_only_the_edited_file_is_read(self, tree, reads):
        first = self.old_snapshot(tree)
        write(tree, "pkg/a.py", "x = 2  # edited\n")
        reads["n"] = 0
        second = build_snapshot(tree, DEPS, previous=first)
        assert reads["n"] == 1
        assert second["files"]["pkg/a.py"][2] == _content_hash(b"x = 2  # edited\n")

    def test_files_inside_the_previous_racy_window_are_always_read(self, tree, reads):
        first = snap(tree)  # taken "now": nothing is comfortably older than it
        first["at"] = time.time_ns()
        for rel in first["files"]:  # make every file look just-written
            first["files"][rel][1] = first["at"] - 1
            os.utime(tree / rel, ns=(first["at"] - 1, first["at"] - 1))
        reads["n"] = 0
        build_snapshot(tree, DEPS, previous=first)
        assert reads["n"] == len(first["files"])

    def test_a_different_analyzed_digest_forces_a_read(self, tree, reads):
        first = self.old_snapshot(tree)
        reads["n"] = 0
        second = build_snapshot(
            tree, DEPS, digests={"pkg/a.py": _content_hash(b"what was parsed\n")}, previous=first
        )
        assert reads["n"] == 1 and second["files"]["pkg/a.py"][1] == -1  # poisoned

    def test_another_machines_snapshot_is_not_trusted_for_mtimes(self, tree, reads):
        first = self.old_snapshot(tree)
        for rel in first["files"]:
            first["files"][rel][1] = 1_000_000  # recorded elsewhere: mtimes differ
        reads["n"] = 0
        build_snapshot(tree, DEPS, previous=first)
        assert reads["n"] == len(first["files"])

    def test_a_snapshot_of_an_unknown_version_is_ignored(self, tree, reads):
        reads["n"] = 0
        build_snapshot(tree, DEPS, previous={"v": 99, "files": {}, "at": 1})
        assert reads["n"] == len(snap(tree)["files"])


class TestUnchanged:
    def test_nothing_changed_means_no_changes_and_no_questions(self, tree):
        def confirm(paths):
            raise AssertionError("git must not be asked", paths)

        assert detect_changes(tree, snap(tree), confirm) == set()

    def test_touch_is_not_a_change(self, tree):
        s = snap(tree)
        later = time.time() + 30
        for rel in s["files"]:
            os.utime(tree / rel, (later, later))
        assert detect_changes(tree, s, lambda p: set()) == set()

    def test_other_machine_every_mtime_differs_content_identical(self, tree):
        s = snap(tree)
        for rel in s["files"]:
            os.utime(tree / rel, (1_000_000, 1_000_000))
        assert detect_changes(tree, s, lambda p: set()) == set()


class TestEdits:
    def test_content_change_with_new_mtime(self, tree):
        s = snap(tree)
        write(tree, "pkg/a.py", "x = 2\n")
        assert detect_changes(tree, s, lambda p: set()) == {"pkg/a.py"}

    def test_size_change(self, tree):
        s = snap(tree)
        write(tree, "data/x.txt", "payload and more\n", age=3600)  # mtime restored
        assert detect_changes(tree, s, lambda p: set()) == {"data/x.txt"}

    def test_deleted_file_and_file_replaced_by_a_directory(self, tree):
        s = snap(tree)
        (tree / "pkg/a.py").unlink()
        (tree / "data/x.txt").unlink()
        (tree / "data/x.txt").mkdir()
        assert detect_changes(tree, s, lambda p: set()) == {"pkg/a.py", "data/x.txt"}

    def test_a_non_dependency_py_file_beside_a_dependency_counts_too(self, tree):
        s = snap(tree)  # pkg/b.py is not a dependency, but it is next to one
        write(tree, "pkg/b.py", "x = 99\n")
        assert detect_changes(tree, s, lambda p: set()) == {"pkg/b.py"}

    def test_a_vanished_directory_reports_its_files(self, tree):
        s = snap(tree)
        import shutil

        shutil.rmtree(tree / "pkg/sub")
        assert detect_changes(tree, s, lambda p: set()) == {"pkg/sub/c.py"}


class TestRacyClean:
    def test_same_size_same_mtime_edit_inside_the_racy_window_is_caught(self, tmp_path):
        root = tmp_path / "r"
        path = write(root, "m.py", "v = 1\n")  # written "now": inside the window
        s = snap(root, ["m.py"])
        stamp = path.stat().st_mtime_ns
        path.write_text("v = 2\n")  # same size ...
        os.utime(path, ns=(stamp, stamp))  # ... and the very same mtime
        assert detect_changes(root, s, lambda p: set()) == {"m.py"}

    def test_the_window_is_what_decides_the_stat_shortcut(self, tmp_path):
        root = tmp_path / "r"
        path = write(root, "m.py", "v = 1\n", age=3600)
        s = snap(root, ["m.py"])
        assert s["files"]["m.py"][1] < s["at"] - RACY_NS  # comfortably old -> trusted
        path.write_text("v = 2\n")  # a new mtime: stat disagrees, bytes settle it
        assert detect_changes(root, s, lambda p: set()) == {"m.py"}

    def test_a_file_changed_between_parse_and_snapshot_is_poisoned(self, tree):
        # The parse cache analyzed other bytes than are on disk now: the stored
        # mtime is -1 so the stat shortcut can never hide the difference.
        s = snap(tree, digests={"pkg/a.py": _content_hash(b"what was parsed\n")})
        assert s["files"]["pkg/a.py"][1] == -1
        assert detect_changes(tree, s, lambda p: set()) == {"pkg/a.py"}


class TestNewFiles:
    def test_new_py_in_a_watched_directory_is_confirmed_by_git(self, tree):
        s = snap(tree)
        write(tree, "pkg/new.py")
        asked = []

        def confirm(paths):
            asked.append(paths)
            return set(paths)

        assert detect_changes(tree, s, confirm) == {"pkg/new.py"}
        assert asked == [["pkg/new.py"]]

    def test_an_ignored_new_file_is_dropped_when_git_says_so(self, tree):
        s = snap(tree)
        write(tree, "pkg/_version.py")
        assert detect_changes(tree, s, lambda p: set()) == set()

    def test_new_directory_is_asked_about_as_a_directory_once(self, tree):
        s = snap(tree)
        write(tree, "newpkg/mod.py")
        write(tree, "newpkg/deep/mod2.py")
        calls = []

        def confirm(paths):
            calls.append(paths)
            return {"newpkg/mod.py", "newpkg/deep/mod2.py", "newpkg/data.txt"}

        got = detect_changes(tree, s, confirm)
        assert got == {"newpkg/mod.py", "newpkg/deep/mod2.py"}  # .py only
        assert calls == [["newpkg/"]]

    def test_one_question_for_everything_new(self, tree):
        s = snap(tree)
        write(tree, "pkg/n1.py")
        write(tree, "tests/n2.py")
        write(tree, "brand/n3.py")
        calls = []

        def confirm(paths):
            calls.append(sorted(paths))
            return set()

        detect_changes(tree, s, confirm)
        assert calls == [["brand/", "pkg/n1.py", "tests/n2.py"]]

    def test_a_new_root_level_file_counts_because_the_root_is_watched(self, tree):
        s = snap(tree)
        write(tree, "conftest.py")
        assert detect_changes(tree, s, lambda p: set(p)) == {"conftest.py"}

    @pytest.mark.parametrize(
        "name", [".hidden", "__pycache__", "node_modules", ".venv", "venv", "foo.egg-info"]
    )
    def test_environment_and_cache_directories_are_never_new_code(self, tree, name):
        s = snap(tree)
        write(tree, f"{name}/z.py")

        def confirm(paths):
            raise AssertionError("must not ask about", paths)

        assert detect_changes(tree, s, confirm) == set()

    def test_new_non_python_files_are_not_dependencies(self, tree):
        s = snap(tree)
        write(tree, "data/new.txt")
        write(tree, "pkg/readme.md")
        assert detect_changes(tree, s, lambda p: set()) == set()

    def test_python_files_under_excluded_dirs_are_filtered_from_git_answers(self, tree):
        s = snap(tree)
        write(tree, "newpkg/mod.py")
        got = detect_changes(
            tree, s, lambda p: {"newpkg/mod.py", "newpkg/node_modules/dep.py", "newpkg/.x/h.py"}
        )
        assert got == {"newpkg/mod.py"}


class TestCannotDecide:
    def test_git_unable_to_confirm_means_none(self, tree):
        s = snap(tree)
        write(tree, "pkg/new.py")
        assert detect_changes(tree, s, lambda p: None) is None
        assert detect_changes(tree, s, None) is None  # no way to ask

    @pytest.mark.parametrize("bad", [{}, {"v": 99}, {"v": 1}, {"v": 1, "files": [], "dirs": {}, "at": 1}])
    def test_unknown_or_malformed_snapshot_means_none(self, tree, bad):
        assert detect_changes(tree, bad, lambda p: set()) is None


# ---------------------------------------------------------------------------
# Differential: against git itself.


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "T")
    (root / ".gitignore").write_text("generated_*.py\n__pycache__/\n")
    for rel in ("pkg/a.py", "pkg/b.py", "pkg/sub/c.py", "tests/test_t.py", "setup.py"):
        write(root, rel, f"# {rel}\nx = 1\n", age=3600)
    write(root, "data/x.txt", "payload\n", age=3600)
    commit(root, "initial")
    return root


def git_changed(root, base):
    return GitStore(root, "k").changed_since(base)


OPS = ["edit_py", "edit_data", "touch", "delete", "new_py", "new_py_in_new_dir",
       "new_ignored_py", "new_data", "commit_all", "revert_one", "rewrite_same"]


@pytest.mark.parametrize("seed", range(25))
def test_agrees_with_git_diff_and_ls_files(repo, seed):
    rng = random.Random(seed)
    deps = ["pkg/a.py", "pkg/sub/c.py", "tests/test_t.py", "data/x.txt"]
    base = git(repo, "rev-parse", "HEAD").stdout.strip()
    snapshot = build_snapshot(repo, deps)
    store = GitStore(repo, "k")
    existing = ["pkg/a.py", "pkg/b.py", "pkg/sub/c.py", "tests/test_t.py", "setup.py", "data/x.txt"]
    for counter in range(1, rng.randint(1, 7) + 1):
        op = rng.choice(OPS)
        target = rng.choice([f for f in existing if (repo / f).exists()] or ["pkg/a.py"])
        if op == "edit_py" and target.endswith(".py"):
            (repo / target).write_text(f"x = {counter}\n# edited {seed}\n")
        elif op == "edit_data":
            (repo / "data/x.txt").write_text(f"payload {counter}\n")
        elif op == "touch":
            later = time.time() + 5 + counter
            if (repo / target).exists():
                os.utime(repo / target, (later, later))
        elif op == "delete" and (repo / target).exists():
            (repo / target).unlink()
        elif op == "new_py":
            write(repo, f"{rng.choice(['pkg', 'tests', 'pkg/sub'])}/new{counter}.py")
        elif op == "new_py_in_new_dir":
            write(repo, f"fresh{counter}/inner/mod.py")
        elif op == "new_ignored_py":
            write(repo, f"pkg/generated_{counter}.py")
        elif op == "new_data":
            write(repo, f"pkg/notes{counter}.txt")
        elif op == "commit_all":
            git(repo, "add", "-A")
            git(repo, "commit", "-q", "-m", f"c{counter}", "--allow-empty")
        elif op == "revert_one" and target in git(repo, "ls-files").stdout.split():
            git(repo, "checkout", "--", target)
        elif op == "rewrite_same" and (repo / target).exists():
            (repo / target).write_text((repo / target).read_text())

    expected = {
        f for f in git_changed(repo, base)
        if f.endswith(".py") or f in deps
    }
    got = detect_changes(repo, snapshot, store.project_files)
    assert got == expected, (seed, got ^ expected)


# ---------------------------------------------------------------------------
# End to end.


def shard_key(project):
    ref = git(project.path, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.split()[0]
    return ref.rsplit("/", 1)[-1]


class TestEndToEnd:
    def test_the_shard_carries_its_fingerprints(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        snapshot = GitStore(project.path, shard_key(project)).load_files()
        assert snapshot is not None and snapshot["v"] == 1
        assert {"tests/lib.py", "tests/test_a.py", "tests/conftest.py"} <= set(snapshot["files"])
        assert "tests" in snapshot["dirs"] and "" in snapshot["dirs"]

    def test_amended_history_does_not_invalidate_the_shard(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        base = GitStore(project.path, shard_key(project)).load_meta()["base_commit"]
        git(project.path, "commit", "-q", "--amend", "--allow-empty", "-m", "amended")
        git(project.path, "reflog", "expire", "--expire=now", "--all")
        git(project.path, "gc", "-q", "--prune=now")
        probe = subprocess.run(
            ["git", "-C", str(project.path), "cat-file", "-e", base],
            capture_output=True,
            check=False,
        )
        assert probe.returncode != 0  # the recorded base commit is really gone
        # Under git diff this was "cannot compute changes since the recorded
        # base commit; running everything".
        result = run_less(project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_an_untracked_dependency_is_not_perpetually_changed(self, project):
        (project.path / "tests" / "test_c.py").write_text(
            "from pathlib import Path\n\n"
            "def test_c():\n"
            "    assert (Path(__file__).parent / 'local.txt').read_text() == 'a\\n'\n"
        )
        (project.path / "tests" / "local.txt").write_text("a\n")  # never committed
        commit(project.path, "add test (local.txt stays untracked)", )
        run_less(project, "-q").assert_outcomes(passed=3)
        for _ in range(2):
            result = run_less(project, "-q")
            assert result.ret == 5  # git ls-files --others used to re-select it forever
        (project.path / "tests" / "local.txt").write_text("b\n")
        result = run_less(project, "-v")
        result.assert_outcomes(failed=1)  # the edit is seen: only that test runs
        result.stdout.fnmatch_lines(["*selected 1/3 tests*"])

    def test_reverting_a_dirty_record_is_noticed(self, project):
        # Recorded with an uncommitted edit; the edit is then thrown away.  The
        # tree now equals HEAD, so ``git diff <base>`` sees nothing -- but the
        # shard was recorded from different bytes.
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1 + int(0)\n")
        run_less(project, "-q").assert_outcomes(passed=2)
        git(project.path, "checkout", "--", "tests/lib.py")
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_a.py::test_a PASSED*"])

    def test_a_shard_without_fingerprints_falls_back_to_git_and_gains_them(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        store = GitStore(project.path, shard_key(project))
        data, meta, parse = store.load_data(), store.load_meta(), store.load_parse()
        store.delete()
        store.save(data, meta, parse=parse)  # an old-style shard: no files.json
        assert GitStore(project.path, shard_key(project)).load_files() is None
        result = run_less(project, "-q")
        assert result.ret == 5  # git diff path still decides correctly
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1 + int(0)\n")
        run_less(project, "-q").assert_outcomes(passed=1)
        assert GitStore(project.path, shard_key(project)).load_files() is not None

    def test_forced_git_backend_never_reads_fingerprints(self, project, monkeypatch):
        run_less(project, "-q").assert_outcomes(passed=2)
        monkeypatch.setenv("PYTEST_LESS_GIT_BACKEND", "subprocess")
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1 + int(0)\n")
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_a.py::test_a PASSED*"])

    def test_a_run_that_stops_early_does_not_hide_the_tests_it_never_ran(self, pytester):
        tests = pytester.path / "tests"
        tests.mkdir()
        (tests / "conftest.py").touch()
        (tests / "lib.py").write_text("def v():\n    return 1\n")
        for name in ("a", "b"):
            (tests / f"test_{name}.py").write_text(
                f"def test_{name}():\n    import lib\n    assert lib.v() == 1\n"
            )
        (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
        git(pytester.path, "init", "-q")
        git(pytester.path, "config", "user.email", "t@example.com")
        git(pytester.path, "config", "user.name", "T")
        commit(pytester.path, "initial")
        run_less(pytester, "-q").assert_outcomes(passed=2)

        (tests / "lib.py").write_text("def v():\n    return 22\n")  # both now fail
        first = run_less(pytester, "-q", "-x")  # stops after the first failure
        first.assert_outcomes(failed=1)

        (tests / "lib.py").write_text("def v():\n    return 1\n")  # fixed again
        again = run_less(pytester, "-v")
        # The failed one reruns; so must the one that never got to run: the
        # shard's fingerprints were refreshed by the stopped run.
        again.assert_outcomes(passed=2)


def test_spawn_free_sanity_of_the_subprocess_module_import():
    # fingerprint.py must not drag in anything that needs git at import time
    out = subprocess.run(
        [sys.executable, "-c", "import pytest_less.fingerprint"], capture_output=True, check=False
    )
    assert out.returncode == 0
