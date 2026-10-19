"""Data-file tracking via the ``open`` audit hook (on by default).

Non-Python files opened by tests become file-level wildcard deps, closing
the data-file blind spot: without tracking, editing a fixture that only
tests read selects nothing (a red suite behind a green CI).  Opt out with
``--less-no-data-files`` or ``less_data_files = false``.  Python source
opens must NOT be recorded that way -- they are tracked at block precision,
and a wildcard dep on them would throw that precision away.
"""

import json
import subprocess

import pytest
from test_plugin import commit, decode_shard, dep_ids, git, run_less, shard_data


@pytest.fixture
def data_project(pytester):
    """A git repo with three tests:

    * ``test_read``     reads ``fixture.txt`` (a data file)
    * ``test_readsrc``  reads ``sample.py`` as *text* (never imported)
    * ``test_plain``    touches nothing
    """
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "fixture.txt").write_text("payload-v1\n")
    (tests / "sample.py").write_text(
        "# never imported by anything\ndef marker():\n    return 1\n"
    )
    (tests / "test_read.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def test_read():\n"
        "    data = (Path(__file__).parent / 'fixture.txt').read_text()\n"
        "    assert len(data) > 0\n"
    )
    (tests / "test_readsrc.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def test_readsrc():\n"
        "    src = (Path(__file__).parent / 'sample.py').read_text()\n"
        "    assert 'marker' in src\n"
    )
    (tests / "test_plain.py").write_text("def test_plain():\n    assert True\n")
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


def record(project, *extra):
    result = run_less(project, "-q", *extra)
    result.assert_outcomes(passed=3)
    return result


class TestDataFileSelection:
    def test_data_open_recorded_per_test_and_selected_on_change(self, data_project):
        record(data_project)
        tests, _ = decode_shard(shard_data(data_project.path))
        assert "tests/fixture.txt:*" in dep_ids(tests["tests/test_read.py::test_read"])
        # Only the reader knows the data file.
        assert "tests/fixture.txt:*" not in dep_ids(
            tests["tests/test_plain.py::test_plain"]
        )
        assert "tests/fixture.txt:*" not in dep_ids(
            tests["tests/test_readsrc.py::test_readsrc"]
        )

        (data_project.path / "tests" / "fixture.txt").write_text(
            "payload-v2-with-different-content\n"
        )
        result = run_less(data_project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/3 tests*"])
        result.stdout.fnmatch_lines(["*test_read.py::test_read PASSED*"])

    def test_python_text_open_is_not_a_wildcard_dep(self, data_project):
        # Precision guard: reading a .py file as text must not create a
        # file-level wildcard dep (PY_START tracks real code at block
        # precision; a wildcard here would re-run on any body edit).
        record(data_project)
        tests, _ = decode_shard(shard_data(data_project.path))
        assert "tests/sample.py:*" not in dep_ids(
            tests["tests/test_readsrc.py::test_readsrc"]
        )
        # The test still depends on its own module (sanity).
        assert "tests/test_readsrc.py:<module>" in dep_ids(
            tests["tests/test_readsrc.py::test_readsrc"]
        )

        with open(data_project.path / "tests" / "sample.py", "a") as f:
            f.write("\ndef added():\n    return 2\n")
        result = run_less(data_project, "-q")
        assert result.ret == 5  # nothing selected, nothing ran
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])

    def test_tracking_is_on_by_default(self, data_project):
        # No flag at all: the data file is a dep and selects its reader.
        result = run_less(data_project, "-q")
        result.assert_outcomes(passed=3)
        tests, _ = decode_shard(shard_data(data_project.path))
        assert "tests/fixture.txt:*" in dep_ids(tests["tests/test_read.py::test_read"])

        (data_project.path / "tests" / "fixture.txt").write_text(
            "payload-v2-with-different-content\n"
        )
        result = run_less(data_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/3 tests*"])

    def test_cli_opt_out_is_blind_to_data_files(self, data_project):
        # --less-no-data-files restores the old opt-out behavior (and uses
        # its own shard: the flag is part of the command key).
        record(data_project, "--less-no-data-files")
        result = run_less(data_project, "-q", "--less-no-data-files")
        assert result.ret == 5  # nothing selected, nothing ran
        tests, _ = decode_shard(shard_data(data_project.path))
        assert "tests/fixture.txt:*" not in dep_ids(
            tests["tests/test_read.py::test_read"]
        )

    def test_no_change_rerun_still_takes_the_fast_path(self, data_project):
        record(data_project)
        result = run_less(data_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])
        # The data file must be part of the fast path's dep_files oracle:
        # if it changes, the no-change shortcut must not fire.
        refs = git(
            data_project.path,
            "for-each-ref",
            "--format=%(refname)",
            "refs/less/",
        ).stdout.split()
        assert len(refs) == 1
        meta = json.loads(
            subprocess.run(
                ["git", "-C", str(data_project.path), "cat-file", "blob",
                 f"{refs[0]}:meta.json"],
                capture_output=True, text=True, check=True,
            ).stdout
        )
        assert "tests/fixture.txt" in meta["dep_files"]

    def test_ini_opt_out_disables_tracking(self, data_project):
        data_project.makeini("[pytest]\nless_data_files = false\n")
        record(data_project)
        tests, _ = decode_shard(shard_data(data_project.path))
        assert "tests/fixture.txt:*" not in dep_ids(
            tests["tests/test_read.py::test_read"]
        )

        (data_project.path / "tests" / "fixture.txt").write_text(
            "payload-v2-with-different-content\n"
        )
        result = run_less(data_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])


@pytest.fixture
def shared_project(pytester):
    """A git repo whose conftest reads ``shared.cfg`` at import time."""
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "shared.cfg").write_text("flag=1\n")
    (tests / "conftest.py").write_text(
        "from pathlib import Path\n"
        "_CONTENT = (Path(__file__).parent / 'shared.cfg').read_text()\n"
    )
    (tests / "test_one.py").write_text("def test_one():\n    assert True\n")
    (tests / "test_two.py").write_text("def test_two():\n    assert True\n")
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


class TestImportTimeDataFile:
    def test_open_before_first_test_is_shared_by_every_test(self, shared_project):
        result = run_less(shared_project, "-q")
        result.assert_outcomes(passed=2)

        _, import_deps = decode_shard(shard_data(shared_project.path))
        assert "tests/shared.cfg:*" in dep_ids(import_deps)

        (shared_project.path / "tests" / "shared.cfg").write_text("flag=2\n")
        result = run_less(shared_project, "-q")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*selected 2/2 tests*"])
