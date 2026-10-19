"""Runtime detection of reflective / dynamic code (fail on the safe side).

``getattr``/``setattr`` have no audit event, but whatever they *reach* still
executes and is recorded by PY_START.  What a recording cannot know is which
code a **computed name** would reach next time.  Two runtime signals mark a
test *dynamic*:

* code compiled from a string -- ``eval``/``exec``/``compile``, generated
  classes -- (audit event ``compile`` with a ``<...>`` filename), and
* reflective entry points (``importlib.import_module``, ``runpy.run_*``,
  ``pkgutil.walk_packages``, ...), seen through a PY_START pseudo-dependency.

A dynamic test keeps its recorded dependencies (precision for ordinary
edits is unchanged) but is also selected whenever a changed ``.py`` file is
stale or was never recorded -- a new module or an edited block it could have
reached by name.
"""

import importlib

import pytest
from test_plugin import commit, decode_shard, dep_ids, git, run_less, shard_data
from test_subprocess import read_meta

from pytest_less.blocks import DYNAMIC_DEP
from pytest_less.monitoring import FileMonitor

EVAL_ID = "tests/test_eval.py::test_eval"
IMPORT_ID = "tests/test_importmod.py::test_importmod"
PLAIN_ID = "tests/test_plain.py::test_plain"


@pytest.fixture
def dyn_project(pytester):
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "lib.py").write_text("def f():\n    return 1\n")
    (tests / "helper.py").write_text("def g():\n    return 2\n")
    (tests / "test_eval.py").write_text(
        "def test_eval():\n    assert eval('1 + 1') == 2\n"
    )
    (tests / "test_importmod.py").write_text(
        "import importlib\n"
        "\n"
        "def test_importmod():\n"
        "    mod = importlib.import_module('lib')\n"
        "    assert mod.f() == 1\n"
    )
    # Executes repo code for the first time inside the test (lazy block
    # analysis parses source from within the test: must NOT look dynamic).
    (tests / "test_plain.py").write_text(
        "def test_plain():\n    import helper\n    assert helper.g() == 2\n"
    )
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


class TestRecording:
    def test_dynamic_tests_are_marked_and_plain_is_not(self, dyn_project):
        run_less(dyn_project, "-q").assert_outcomes(passed=3)
        data = shard_data(dyn_project.path)
        assert data["dynamic"] == sorted([EVAL_ID, IMPORT_ID])
        assert read_meta(dyn_project.path)["dynamic"] == sorted([EVAL_ID, IMPORT_ID])

    def test_dynamic_tests_keep_their_dependencies(self, dyn_project):
        run_less(dyn_project, "-q").assert_outcomes(passed=3)
        tests, _ = decode_shard(shard_data(dyn_project.path))
        assert "tests/lib.py:f" in dep_ids(tests[IMPORT_ID])
        assert "tests/test_eval.py:test_eval" in dep_ids(tests[EVAL_ID])

    def test_pseudo_dependency_never_reaches_the_shard(self, dyn_project):
        run_less(dyn_project, "-q").assert_outcomes(passed=3)
        data = shard_data(dyn_project.path)
        assert all(b[0] != DYNAMIC_DEP[0] for b in data["blocks"])
        assert "<dynamic>" not in read_meta(dyn_project.path)["dep_files"]


class TestSelection:
    def test_no_change_rerun_takes_the_fast_path(self, dyn_project):
        run_less(dyn_project, "-q")
        result = run_less(dyn_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/3 tests*", "*fast path*"])

    def test_unrecorded_py_file_selects_the_dynamic_tests_only(self, dyn_project):
        run_less(dyn_project, "-q")
        (dyn_project.path / "tests" / "plugin_x.py").write_text("X = 1\n")
        result = run_less(dyn_project, "-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])
        result.stdout.fnmatch_lines(["*runtime-dynamic*"])
        result.stdout.no_fnmatch_line("*test_plain.py::test_plain PASSED*")

    def test_stale_edit_of_a_reachable_module_selects_dynamic_tests(
        self, dyn_project
    ):
        run_less(dyn_project, "-q")
        (dyn_project.path / "tests" / "lib.py").write_text(
            "def f():\n    return 1 + 0\n"
        )
        result = run_less(dyn_project, "-v")
        # test_importmod via staleness, test_eval via the runtime rule;
        # test_plain never touched lib and stays deselected.
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])
        result.stdout.no_fnmatch_line("*test_plain.py::test_plain PASSED*")

    def test_non_py_change_selects_nothing(self, dyn_project):
        run_less(dyn_project, "-q")
        (dyn_project.path / "notes.md").write_text("hi\n")
        result = run_less(dyn_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])

    def test_comment_only_edit_selects_nothing(self, dyn_project):
        run_less(dyn_project, "-q")
        (dyn_project.path / "tests" / "lib.py").write_text(
            "def f():\n    return 1  # now with a comment\n"
        )
        result = run_less(dyn_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])

    def test_old_shard_without_marks_is_conservative(self):
        from pytest_less.selection import load_shard

        shard = load_shard(
            {"version": 2, "blocks": [], "tests": {}}, {"base_commit": None}
        )
        assert shard is not None and shard.dynamic is None  # unknown


class TestAuditHook:
    @staticmethod
    def make(tmp_path):
        m = FileMonitor(tmp_path)
        m._active = True  # white-box: no real tool id for the unit test
        m.phase = "test"
        m.current_nodeid = "t"
        return m

    @pytest.mark.parametrize("filename", ["<string>", "<unknown>", b"<string>"])
    def test_string_compiled_code_marks_the_test(self, tmp_path, filename):
        m = self.make(tmp_path)
        m._on_audit("compile", (b"1+1", filename))
        assert m.dynamic_tests == {"t"}

    @pytest.mark.parametrize("filename", ["/repo/mod.py", "<frozen importlib._bootstrap>"])
    def test_real_files_and_frozen_modules_do_not(self, tmp_path, filename):
        m = self.make(tmp_path)
        m._on_audit("compile", (b"x=1", filename))
        assert m.dynamic_tests == set()

    def test_only_the_test_phase_counts(self, tmp_path):
        # Module-level namedtuple/dataclass at import or collection time is
        # ubiquitous and reaches no unseen edge.
        m = self.make(tmp_path)
        for phase in ("import", "collect", "done"):
            m.phase = phase
            m._on_audit("compile", (b"x", "<string>"))
        assert m.dynamic_tests == set()

    def test_our_own_block_analysis_is_ignored(self, tmp_path):
        # ast.parse compiles with "<unknown>"; the monitor parses source
        # lazily from inside tests.
        m = self.make(tmp_path)
        m._busy = True
        m._on_audit("compile", (b"x", "<unknown>"))
        assert m.dynamic_tests == set()


class TestReflectiveEntryPoints:
    @staticmethod
    def analyze(tmp_path, code):
        return FileMonitor(tmp_path)._analyze(code)

    def test_import_module_leaves_the_pseudo_dependency(self, tmp_path):
        assert self.analyze(tmp_path, importlib.import_module.__code__) == {
            DYNAMIC_DEP
        }

    def test_other_reflective_stdlib_entry_points(self, tmp_path):
        import pkgutil
        import runpy

        for fn in (runpy.run_module, runpy.run_path, pkgutil.iter_modules,
                   pkgutil.walk_packages, importlib.reload):
            assert self.analyze(tmp_path, fn.__code__) == {DYNAMIC_DEP}, fn

    def test_ordinary_stdlib_code_is_not_flagged(self, tmp_path):
        import json
        import os.path

        assert self.analyze(tmp_path, os.path.join.__code__) == frozenset()
        assert self.analyze(tmp_path, json.dumps.__code__) == frozenset()
