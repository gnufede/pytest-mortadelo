"""Resolving what a block's bytecode may import to repository files.

The import scan is what records that a test which executes ``from .helper import
h`` depends on ``helper``'s module skeleton.  Only the *first* test to import a
module runs its module code (and so records it through that module's own
PY_START); every later test gets a ``sys.modules`` hit and relies on the scan.
So a relative import the scan cannot resolve is a dependency nobody records.
"""

import sys
from pathlib import Path

import pytest
from test_plugin import commit, git, run_less

from pytest_less.monitoring import FileMonitor


def make_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def tree(tmp_path):
    make_tree(
        tmp_path,
        {
            # a src/lib layout: the source root itself is not a package
            "lib/pkg/__init__.py": "",
            "lib/pkg/sub/__init__.py": "",
            "lib/pkg/sub/mod.py": "",
            "lib/pkg/sub/base.py": "",
            "lib/pkg/top.py": "",
            # the same package at the repository root
            "flat/__init__.py": "",
            "flat/inner/__init__.py": "",
            "flat/inner/m.py": "",
            "flat/inner/base.py": "",
            # a directory that is not a package, holding a same-stem file
            "plain/base.py": "",
            "scripts/tool.py": "",
            # a gap: an intermediate directory without __init__
            "ns/gap/leaf/__init__.py": "",
            "ns/gap/leaf/m.py": "",
        },
    )
    return tmp_path


class TestPackagePrefix:
    @pytest.mark.parametrize(
        ("rel", "prefix"),
        [
            ("lib/pkg/sub/mod.py", ["pkg", "sub"]),  # was [] -- lib/ has no __init__
            ("lib/pkg/top.py", ["pkg"]),
            ("flat/inner/m.py", ["flat", "inner"]),
            ("scripts/tool.py", []),
            ("plain/base.py", []),
            ("ns/gap/leaf/m.py", ["leaf"]),  # the package ends at the first gap
        ],
    )
    def test_prefix_is_the_package_the_module_index_would_name_it_by(
        self, tree, rel, prefix
    ):
        assert FileMonitor(tree)._pkg_prefix(rel) == prefix

    def test_prefix_agrees_with_the_module_index_for_every_file(self, tree):
        from pytest_less.monitoring import _build_module_index

        monitor = FileMonitor(tree)
        for name, rel in _build_module_index(tree, ()).items():
            if rel.endswith("__init__.py"):
                continue
            expected = name.rsplit(".", 1)[0].split(".") if "." in name else []
            assert monitor._pkg_prefix(rel) == expected, (rel, name)


class TestRelativeImports:
    def test_a_relative_import_in_a_source_root_layout_resolves(self, tree):
        files, full = FileMonitor(tree)._resolve_module("base", 1, "lib/pkg/sub/mod.py")
        assert full == "pkg.sub.base"
        assert "lib/pkg/sub/base.py" in files

    def test_parent_package_imports_resolve(self, tree):
        files, full = FileMonitor(tree)._resolve_module("top", 2, "lib/pkg/sub/mod.py")
        assert full == "pkg.top" and "lib/pkg/top.py" in files

    def test_it_does_not_match_an_unrelated_file_of_the_same_stem(self, tree):
        # ``plain/base.py`` is indexed by its bare stem ``base``; the broken
        # prefix resolved ``from .base import x`` to exactly that name and so
        # recorded a dependency on an unrelated file (seen on sqlalchemy: a
        # dialect's ``from .dml import`` matched a typing test data file).
        files, _ = FileMonitor(tree)._resolve_module("base", 1, "lib/pkg/sub/mod.py")
        assert "plain/base.py" not in files

    def test_the_repository_root_layout_is_unchanged(self, tree):
        files, full = FileMonitor(tree)._resolve_module("base", 1, "flat/inner/m.py")
        assert full == "flat.inner.base" and "flat/inner/base.py" in files


class TestNoImportSideEffects:
    def test_resolving_a_dotted_name_does_not_import_its_parent(self, tmp_path, monkeypatch):
        # importlib.util.find_spec("a.b") imports "a"; the recorder must not
        # run imports that no test ran.
        make_tree(
            tmp_path / "elsewhere",
            {"sidefx_pkg/__init__.py": "import builtins\nbuiltins._SIDEFX_RAN = True\n",
             "sidefx_pkg/sub.py": ""},
        )
        monkeypatch.syspath_prepend(str(tmp_path / "elsewhere"))
        monkeypatch.delitem(sys.modules, "sidefx_pkg", raising=False)
        import builtins

        monkeypatch.setattr(builtins, "_SIDEFX_RAN", False, raising=False)
        repo = tmp_path / "repo"
        repo.mkdir()
        monitor = FileMonitor(repo)
        assert monitor._spec_origin("sidefx_pkg.sub") is None
        assert builtins._SIDEFX_RAN is False
        assert "sidefx_pkg" not in sys.modules


class TestEndToEnd:
    @pytest.fixture
    def src_project(self, pytester):
        make_tree(
            pytester.path,
            {
                "pytest.ini": "[pytest]\npythonpath = src\n",
                "src/pkg/__init__.py": "",
                "src/pkg/core.py": (
                    "def f(flag):\n"
                    "    if flag:\n"
                    "        from .helper import compute\n"
                    "        return compute()\n"
                    "    return 0\n"
                ),
                "src/pkg/helper.py": "FACTOR = 2\n\ndef compute():\n    return FACTOR\n",
                "tests/conftest.py": "",
                # runs f without ever taking the branch that imports the helper
                "tests/test_core.py": "def test_core():\n    from pkg import core\n    assert core.f(False) == 0\n",
                "tests/test_other.py": "def test_other():\n    assert 1 + 1 == 2\n",
                ".gitignore": ".pytest_cache/\n__pycache__/\n",
            },
        )
        git(pytester.path, "init", "-q")
        git(pytester.path, "config", "user.email", "t@example.com")
        git(pytester.path, "config", "user.name", "T")
        commit(pytester.path, "initial")
        return pytester

    def test_an_unexecuted_relative_import_in_a_src_layout_is_a_dependency(
        self, src_project
    ):
        run_less(src_project, "-q").assert_outcomes(passed=2)
        # the module-level constant the guarded import would pull in
        (src_project.path / "src/pkg/helper.py").write_text(
            "FACTOR = 3\n\ndef compute():\n    return FACTOR\n"
        )
        result = run_less(src_project, "-v")
        # test_core executes a block that can import helper: conservatively
        # selected.  Before the fix the scan resolved ``.helper`` to nothing
        # and nothing was selected.
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_core.py::test_core PASSED*"])
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
