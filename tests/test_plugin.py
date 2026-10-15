"""End-to-end tests: real pytest subprocesses inside real git repositories."""

import json
import subprocess

import pytest


def git(path, *args, stdin=None):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
        input=stdin,
    )


def commit(path, message="commit"):
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", message)
    return git(path, "rev-parse", "HEAD").stdout.strip()


def shard_data(path):
    """The (single) filemon shard stored in the test repo, as a dict."""
    refs = [
        line
        for line in git(path, "for-each-ref", "--format=%(refname)", "refs/filemon/").stdout.splitlines()
        if line
    ]
    assert len(refs) == 1, f"expected exactly one shard ref, got {refs}"
    return json.loads(git(path, "cat-file", "blob", f"{refs[0]}:data.json").stdout)


@pytest.fixture
def project(pytester):
    """A small git repository with two tests and two lazily-imported helpers.

    * ``test_a`` imports ``lib`` inside the test body (executed dynamic import)
    * ``test_b`` mentions ``other`` in a dead branch (unexecuted import -- the
      conservative bytecode scan must still record it as a dependency)
    """
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "lib.py").write_text("def ping():\n    return 1\n")
    (tests / "other.py").write_text("def pong():\n    return 2\n")
    (tests / "test_a.py").write_text(
        "def test_a():\n"
        "    import lib\n"
        "    assert lib.ping() == 1\n"
    )
    (tests / "test_b.py").write_text(
        "def test_b(always_false=False):\n"
        "    if always_false:\n"
        "        import other\n"
        "    assert 1 + 1 == 2\n"
    )
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


def run_filemon(pytester, *args):
    return pytester.runpytest_subprocess("--filemon", *args)


class TestRecording:
    def test_first_run_records_everything(self, project):
        result = run_filemon(project, "-q")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(
            ["*filemon*", "shard: *", "selected 2/2 tests: no usable stored data*"]
        )

        data = shard_data(project.path)
        assert set(data["tests"]) == {
            "tests/test_a.py::test_a",
            "tests/test_b.py::test_b",
        }
        assert data["failed"] == []
        # Executed dynamic import recorded for test_a...
        assert "tests/lib.py" in data["tests"]["tests/test_a.py::test_a"]
        # ...and the never-executed import recorded for test_b.
        assert "tests/other.py" in data["tests"]["tests/test_b.py::test_b"]
        # Import-time deps (conftest) and the test's own module are recorded...
        for deps in data["tests"].values():
            assert "tests/conftest.py" in deps
        assert "tests/test_a.py" in data["tests"]["tests/test_a.py::test_a"]
        # ...but other tests' modules must NOT leak into a test's deps.
        assert "tests/test_b.py" not in data["tests"]["tests/test_a.py::test_a"]
        assert "tests/test_a.py" not in data["tests"]["tests/test_b.py::test_b"]
        # And no phantom dependencies from non-file module origins
        # (e.g. pytest's assertion rewriter injects ``import builtins``).
        for deps in data["tests"].values():
            assert "built-in" not in deps

    def test_nothing_changed_deselects_all(self, project):
        run_filemon(project, "-q")
        result = run_filemon(project, "-q")
        result.assert_outcomes(deselected=2)
        assert result.ret == 5  # NO_TESTS_COLLECTED: nothing was selected

    def test_disabled_without_flag_records_nothing(self, project):
        result = project.runpytest_subprocess("-q")
        result.assert_outcomes(passed=2)
        refs = git(project.path, "for-each-ref", "--format=%(refname)", "refs/filemon/").stdout
        assert refs.strip() == ""


class TestSelection:
    def test_changed_helper_selects_only_affected_test(self, project):
        run_filemon(project, "-v")
        with open(project.path / "tests" / "lib.py", "a") as f:
            f.write("\ndef extra():\n    return 3\n")
        result = run_filemon(project, "-v")
        result.assert_outcomes(passed=1, deselected=1)
        # test_a depends on lib (executed import); test_b does not.
        result.stdout.fnmatch_lines(["*test_a.py::test_a PASSED*"])

    def test_changed_helper_selects_only_the_scanned_import_test(self, project):
        run_filemon(project, "-v")
        with open(project.path / "tests" / "other.py", "a") as f:
            f.write("\ndef extra():\n    return 3\n")
        result = run_filemon(project, "-v")
        result.assert_outcomes(passed=1, deselected=1)
        result.stdout.fnmatch_lines(["*test_b.py::test_b PASSED*"])

    def test_failed_tests_always_rerun(self, project):
        # Make test_a fail (lib.ping no longer matches the assertion).
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 2\n")
        result = run_filemon(project, "-v")
        result.assert_outcomes(passed=1, failed=1)

        # lib.py is now "unchanged" for a subsequent run, but test_a failed
        # in the recorded run: it must run again, test_b must not.
        result = run_filemon(project, "-v")
        result.assert_outcomes(failed=1, deselected=1)
        result.stdout.fnmatch_lines(["*test_a.py::test_a FAILED*"])

    def test_new_test_files_always_run(self, project):
        run_filemon(project, "-q")
        (project.path / "tests" / "test_new.py").write_text("def test_new():\n    assert True\n")
        result = run_filemon(project, "-q")
        result.assert_outcomes(passed=1, deselected=2)

    def test_conftest_change_reruns_everything(self, project):
        run_filemon(project, "-q")
        with open(project.path / "tests" / "conftest.py", "a") as f:
            f.write("\nimport pathlib\n")
        result = run_filemon(project, "-q")
        result.assert_outcomes(passed=2)

    def test_shards_are_keyed_by_command(self, project):
        # Record for the "tests" invocation.
        run_filemon(project, "-q", "tests")
        # Same command: everything deselected.
        result = run_filemon(project, "-q", "tests")
        result.assert_outcomes(deselected=2)
        # A different invocation: no shard for it, so everything runs.
        result = run_filemon(project, "-q", "tests/test_a.py")
        result.assert_outcomes(passed=1)


class TestOptions:
    def test_force_runs_all(self, project):
        run_filemon(project, "-q")
        result = run_filemon(project, "-q", "--filemon-force")
        result.assert_outcomes(passed=2)

    def test_clean_discards_data(self, project):
        run_filemon(project, "-q")
        result = run_filemon(project, "-q", "--filemon-clean")
        result.assert_outcomes(passed=2)
