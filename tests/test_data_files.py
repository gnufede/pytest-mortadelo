"""Data-file tracking via the ``open`` / ``os.open`` wrappers (on by default).

Non-Python files opened by tests become file-level wildcard deps, closing
the data-file blind spot: without tracking, editing a fixture that only
tests read selects nothing (a red suite behind a green CI).  Opt out with
``--less-no-data-files`` or ``less_data_files = false``.  Python source
opens must NOT be recorded that way -- they are tracked at block precision,
and a wildcard dep on them would throw that precision away.
"""

import json
import os
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


@pytest.fixture
def osopen_project(pytester):
    """A repo whose only test reads a data file through ``os.open``.

    ``os.open`` is a separate entry point from ``builtins.open``/``io.open``:
    libraries that want a raw file descriptor bypass the other wrappers, so a
    test using it must still get a wildcard dependency on the data file.
    """
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "fixture.dat").write_text("payload-v1\n")
    (tests / "test_osread.py").write_text(
        "import os\n"
        "\n"
        "def test_osread():\n"
        "    path = os.path.join(os.path.dirname(__file__), 'fixture.dat')\n"
        "    fd = os.open(path, os.O_RDONLY)\n"
        "    try:\n"
        "        assert os.read(fd, 16)\n"
        "    finally:\n"
        "        os.close(fd)\n"
    )
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


class TestOsOpenDataFile:
    def test_os_open_data_file_is_a_wildcard_dep(self, osopen_project):
        run_less(osopen_project, "-q").assert_outcomes(passed=1)
        tests, _ = decode_shard(shard_data(osopen_project.path))
        assert "tests/fixture.dat:*" in dep_ids(
            tests["tests/test_osread.py::test_osread"]
        )

        (osopen_project.path / "tests" / "fixture.dat").write_text("payload-v2\n")
        result = run_less(osopen_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/1 tests*"])


# ---------------------------------------------------------------------------
# Outputs are not inputs.  A file a test only *writes* must never become a
# dependency: once it exists it is an untracked file, which git always lists
# as changed, so the test would re-run on every run (the second record is
# where it bites -- the first one sees the file not yet existing).


class TestOutputOpens:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("r", False), ("rb", False), ("r+", False), ("rt", False),
            ("w", True), ("wb", True), ("w+", True),
            ("a", True), ("ab", True), ("a+", False),
            ("x", True), ("xb", True),
            (None, False), (3, False),
        ],
    )
    def test_mode_classification(self, mode, expected):
        from pytest_less.monitoring import _mode_is_output_only

        assert _mode_is_output_only(mode) is expected

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            (os.O_RDONLY, False),
            (os.O_WRONLY, True),
            (os.O_WRONLY | os.O_CREAT | os.O_APPEND, True),
            (os.O_RDWR, False),
            (os.O_RDWR | os.O_APPEND, False),
            (os.O_RDWR | os.O_TRUNC, True),
            (os.O_RDWR | os.O_CREAT | os.O_EXCL, True),
            (os.O_RDWR | os.O_CREAT, False),
            (None, False),
        ],
    )
    def test_flags_classification(self, flags, expected):
        from pytest_less.monitoring import _flags_are_output_only

        assert _flags_are_output_only(flags) is expected

    @staticmethod
    def recorded(tmp_path, body):
        """Run *body* under a live monitor; return rel paths recorded as deps."""
        from pytest_less.monitoring import FileMonitor

        monitor = FileMonitor(tmp_path, data_files=True)
        assert monitor.activate()
        try:
            monitor.phase = "test"
            monitor.current_nodeid = "t"
            body()
        finally:
            monitor.deactivate()
        return {b[0] for b in monitor.test_deps.get("t", ())}

    def test_reads_are_recorded_writes_are_not(self, tmp_path):
        (tmp_path / "in.txt").write_text("data")

        def body():
            with open(tmp_path / "in.txt") as f:
                f.read()
            with open(tmp_path / "out.txt", "w") as f:
                f.write("x")
            with open(tmp_path / "log.txt", mode="a") as f:  # keyword mode
                f.write("x")
            fd = os.open(tmp_path / "raw.bin", os.O_WRONLY | os.O_CREAT)
            os.close(fd)
            fd = os.open(tmp_path / "raw2.bin", flags=os.O_WRONLY | os.O_CREAT)
            os.close(fd)

        assert self.recorded(tmp_path, body) == {"in.txt"}

    def test_reading_back_your_own_output_is_not_a_dependency(self, tmp_path):
        def body():
            with open(tmp_path / "gen.txt", "w") as f:
                f.write("x")
            with open(tmp_path / "gen.txt") as f:  # read back
                f.read()
            fd = os.open(tmp_path / "gen.txt", os.O_RDONLY)
            os.close(fd)

        assert self.recorded(tmp_path, body) == set()

    def test_read_write_modes_still_count_as_inputs(self, tmp_path):
        (tmp_path / "state.json").write_text("{}")

        def body():
            with open(tmp_path / "state.json", "r+") as f:
                f.read()
            fd = os.open(tmp_path / "state.json", os.O_RDWR)
            os.close(fd)

        assert self.recorded(tmp_path, body) == {"state.json"}


@pytest.fixture
def output_project(pytester):
    """Tests that write output files into the repo, plus one that reads a
    committed input."""
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "input.txt").write_text("v1\n")
    (tests / "test_io.py").write_text(
        "import os, pathlib\n"
        "ROOT = pathlib.Path(__file__).resolve().parent.parent\n"
        "\n"
        "def test_reads_input():\n"
        "    assert (ROOT / 'tests' / 'input.txt').read_text().startswith('v')\n"
        "\n"
        "def test_open_w():\n"
        "    (ROOT / 'out_open.txt').open('w').write('x')\n"
        "\n"
        "def test_os_open_wronly():\n"
        "    fd = os.open(str(ROOT / 'out_os.bin'), os.O_WRONLY | os.O_CREAT)\n"
        "    os.write(fd, b'x'); os.close(fd)\n"
        "\n"
        "def test_append():\n"
        "    (ROOT / 'out_log.txt').open('a').write('x')\n"
        "\n"
        "def test_write_then_read_back():\n"
        "    p = ROOT / 'out_rt.txt'\n"
        "    p.write_text('x')\n"
        "    assert p.read_text() == 'x'\n"
    )
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


class TestOutputFilesEndToEnd:
    OUTPUTS = ("out_open.txt", "out_os.bin", "out_log.txt", "out_rt.txt")

    def test_outputs_never_become_dependencies_or_force_reruns(self, output_project):
        from test_subprocess import read_meta

        run_less(output_project, "-q").assert_outcomes(passed=5)
        # Record again with the outputs already on disk (untracked): the
        # second record is where recording them used to bite.
        result = run_less(output_project, "-q", "--less-clean")
        result.assert_outcomes(passed=5)
        for name in self.OUTPUTS:
            assert (output_project.path / name).exists()

        dep_files = read_meta(output_project.path)["dep_files"]
        assert not [f for f in dep_files if f.startswith("out_")]
        assert "tests/input.txt" in dep_files

        # No change: nothing runs, however many times we ask.
        for _ in range(2):
            result = run_less(output_project, "-q")
            assert result.ret == 5
            result.stdout.fnmatch_lines(["*selected 0/5 tests*"])

    def test_the_input_still_selects_only_its_reader(self, output_project):
        run_less(output_project, "-q")
        run_less(output_project, "-q", "--less-clean")
        (output_project.path / "tests" / "input.txt").write_text("v2-changed\n")
        result = run_less(output_project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_reads_input PASSED*"])
