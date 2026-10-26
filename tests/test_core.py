"""The simplification branch's core loop: record → select → rerun.

Each test creates a tiny git repo with pytester, runs pytest --less
via subprocess, edits, and reruns.  The helpers live here because the
old test infrastructure (test_plugin.py) was removed with the dropped
features.
"""

import json
import subprocess
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def git(path, *args, check=True):
    return subprocess.run(["git", "-C", str(path), *args],
                          capture_output=True, text=True, check=check)


def commit(path, msg="commit"):
    git(path, "add", "-A")
    git(path, "commit", "-q", "--allow-empty", "-m", msg)


def run_less(pytester, *args):
    return pytester.runpytest_subprocess("--less", *args)


def shard_data(path):
    return json.loads((Path(path) / ".pytest_less" / "data.json").read_bytes())


@pytest.fixture
def project(pytester):
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "lib.py").write_text("def ping():\n    return 1\n")
    (tests / "other.py").write_text("X = 1\n")
    (tests / "test_a.py").write_text(
        "def test_a():\n    import lib\n    assert lib.ping() == 1\n"
    )
    (tests / "test_b.py").write_text("def test_b():\n    assert 1 + 1 == 2\n")
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "t@e.c")
    git(pytester.path, "config", "user.name", "T")
    commit(pytester.path, "initial")
    return pytester


class TestRecord:
    def test_first_run_records_everything(self, project):
        result = run_less(project, "-q")
        result.assert_outcomes(passed=2)
        data = shard_data(project.path)
        assert set(data["tests"]) == {
            "tests/test_a.py::test_a", "tests/test_b.py::test_b",
        }

    def test_disabled_without_flag(self, project):
        result = project.runpytest_subprocess("-q")
        result.assert_outcomes(passed=2)
        assert not (project.path / ".pytest_less").exists()

    def test_pure_rerun_never_activates_the_monitor(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        result = run_less(project, "-q")
        assert result.ret == 5  # "no tests collected": nothing to run
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_edit_selects_the_affected_test(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        path = project.path / "tests" / "lib.py"
        path.write_text("def ping():\n    return 1 + int(0)\n")
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
        result.stdout.fnmatch_lines(["*test_a.py::test_a PASSED*"])

    def test_new_test_runs(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        (project.path / "tests" / "test_new.py").write_text(
            "def test_new():\n    assert True\n"
        )
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_new.py::test_new PASSED*"])

    def test_previously_failed_test_reruns(self, project):
        path = project.path / "tests" / "lib.py"
        path.write_text("def ping():\n    return 2\n")
        result = run_less(project, "-q")
        result.assert_outcomes(passed=1, failed=1)
        path.write_text("def ping():\n    return 1\n")
        result = run_less(project, "-q")
        result.assert_outcomes(passed=1)  # test_a (was failed) reruns

    def test_force_runs_all(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        result = run_less(project, "-q", "--less-force")
        result.assert_outcomes(passed=2)

    def test_clean_discards(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        result = run_less(project, "-q", "--less-clean")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*selected 2/2 tests*"])


class TestSelection:
    def test_import_time_dep_reruns_everything(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        path = project.path / "tests" / "conftest.py"
        path.write_text("import pathlib\n")
        result = run_less(project, "-q")
        result.assert_outcomes(passed=2)

    def test_selector_disables_selection(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        path = project.path / "tests" / "lib.py"
        path.write_text("def ping():\n    return 1 + int(0)\n")
        result = run_less(project, "-q", "-k", "test_b")
        result.assert_outcomes(passed=1)


class TestFixtureClosure:
    """The >function-scoped fixture's consumers hold its block (kept feature)."""

    @pytest.fixture
    def fixture_project(self, pytester):
        tests = pytester.path / "tests"
        tests.mkdir()
        (tests / "conftest.py").write_text(
            "import pytest\n\n"
            '@pytest.fixture(scope="class")\n'
            "def config():\n    return 1\n"
        )
        (tests / "test_suite.py").write_text(
            "class TestA:\n"
            '    def test_a(self, config):\n        assert config == 1\n'
        )
        (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
        git(pytester.path, "init", "-q")
        git(pytester.path, "config", "user.email", "t@e.c")
        git(pytester.path, "config", "user.name", "T")
        commit(pytester.path, "initial")
        return pytester

    def test_fixture_consumers_are_selected_on_edit(self, fixture_project):
        run_less(fixture_project, "-q").assert_outcomes(passed=1)
        path = fixture_project.path / "tests" / "conftest.py"
        path.write_text(
            "import pytest\n\n"
            '@pytest.fixture(scope="class")\n'
            "def config():\n    return 30\n"
        )
        result = run_less(fixture_project, "-v")
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*selected 1/1 tests*"])
