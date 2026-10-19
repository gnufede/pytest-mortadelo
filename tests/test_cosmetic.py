"""Cosmetic edit filter: ``--less-cosmetic`` (closes IMPROVEMENTS item 10 / C.2).

Docstring-only edits select 270/270 on both of our models while pytest-tia's
cosmetic filter selects 0 -- its one clean win in the controlled matrix.
With the flag, a block's hash is computed from a docstring-stripped AST, so
prose edits produce identical hashes and select nothing.  Comments and
formatting are already invisible to ``ast.dump`` (free); annotations are
deliberately *kept* (typing changes are observable).  The flag is part of
the shard key, so toggling it starts a fresh record run -- and the default
stays off: code that reads ``__doc__`` (help(), doctests, registries built
from docstrings) would silently miss edits (Ley 1 escape hatch).
"""

import pytest
from test_plugin import commit, git, run_less

from pytest_less.blocks import parse_blocks


@pytest.fixture
def doc_project(pytester):
    """A repo with two tests importing two functions from ``lib.py``."""
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "lib.py").write_text(
        "def documented():\n"
        '    """Old prose."""\n'
        "    return 1\n"
        "\n"
        "def plain():\n"
        "    return 2\n"
    )
    (tests / "test_doc.py").write_text(
        "def test_doc():\n"
        "    import lib\n"
        "    assert lib.documented() == 1\n"
    )
    (tests / "test_plain.py").write_text(
        "def test_plain():\n"
        "    import lib\n"
        "    assert lib.plain() == 2\n"
    )
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


def write_lib(project, documented_body='    """Old prose."""\n    return 1'):
    (project.path / "tests" / "lib.py").write_text(
        "def documented():\n"
        f"{documented_body}\n"
        "\n"
        "def plain():\n"
        "    return 2\n"
    )


DOCSTRING_EDIT = '    """New and improved prose."""\n    return 1'
COMMENT_EDIT = '    """Old prose."""\n    return 1  # a comment, invisible to ast.dump'
BODY_EDIT = '    """Old prose."""\n    return 1 + 0'


class TestCosmeticOn:
    def test_docstring_only_edit_selects_nothing(self, doc_project):
        run_less(doc_project, "-q", "--less-cosmetic").assert_outcomes(passed=2)
        write_lib(doc_project, DOCSTRING_EDIT)
        result = run_less(doc_project, "-q", "--less-cosmetic")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_test_file_docstring_edit_also_selects_nothing(self, doc_project):
        run_less(doc_project, "-q", "--less-cosmetic").assert_outcomes(passed=2)
        (doc_project.path / "tests" / "test_doc.py").write_text(
            "def test_doc():\n"
            '    """The test itself, described."""\n'
            "    import lib\n"
            "    assert lib.documented() == 1\n"
        )
        result = run_less(doc_project, "-q", "--less-cosmetic")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_semantic_edit_still_selects_the_executor(self, doc_project):
        run_less(doc_project, "-q", "--less-cosmetic").assert_outcomes(passed=2)
        write_lib(doc_project, BODY_EDIT)
        result = run_less(doc_project, "-q", "--less-cosmetic")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_ini_enables_the_filter(self, doc_project):
        doc_project.makeini("[pytest]\nless_cosmetic = true\n")
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, DOCSTRING_EDIT)
        result = run_less(doc_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])


class TestCosmeticDefault:
    def test_without_the_flag_prose_is_behavior(self, doc_project):
        # Documents the conservative default (Ley 1): a docstring edit
        # re-runs the tests that executed the documented function.
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, DOCSTRING_EDIT)
        result = run_less(doc_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_flag_is_part_of_the_shard_key(self, doc_project):
        # Toggling the flag must not silently reinterpret an existing
        # shard: it starts a fresh record run instead.
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        result = run_less(doc_project, "-q", "--less-cosmetic")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*no usable stored data*"])

    def test_comment_edits_are_free_even_without_the_flag(self, doc_project):
        # Comments never reach ast.dump: the free half of the cosmetic win.
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, COMMENT_EDIT)
        result = run_less(doc_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])


class TestHashNormalization:
    def test_docstring_change_does_not_change_any_hash(self):
        a = parse_blocks(
            '"""Module prose."""\n'
            "class C:\n"
            '    """Class prose."""\n'
            "    def m(self):\n"
            '        """Method prose."""\n'
            "        return 1\n"
        )
        b = parse_blocks(
            '"""Totally different prose."""\n'
            "class C:\n"
            '    """Other class prose."""\n'
            "    def m(self):\n"
            '        """Other method prose."""\n'
            "        return 1\n"
        )
        ca = parse_blocks(
            '"""Module prose."""\n'
            "class C:\n"
            '    """Class prose."""\n'
            "    def m(self):\n"
            '        """Method prose."""\n'
            "        return 1\n"
            "        # trailing comment\n",
            cosmetic=True,
        )
        cb = parse_blocks(
            '"""Totally different prose."""\n'
            "class C:\n"
            '    """Other class prose."""\n'
            "    def m(self):\n"
            '        """Other method prose."""\n'
            "        return 1\n"
            "        # trailing comment\n",
            cosmetic=True,
        )
        # default: docstrings are part of every hash (module skeleton, class
        # skeleton, method) ...
        assert a.blocks["C.m"] != b.blocks["C.m"]
        assert a.blocks["<module>"] != b.blocks["<module>"]
        # ... with the flag: prose and comments are invisible everywhere.
        assert ca.blocks == cb.blocks
        assert ca.blocks["C"] != a.blocks["C"]  # different normalization

    def test_annotations_are_kept(self):
        # Typing changes are observable behavior: a signature annotation
        # edit must still invalidate the block, flag or no flag.
        a = parse_blocks("def f(x: int):\n    return x\n", cosmetic=True)
        b = parse_blocks("def f(x: str):\n    return x\n", cosmetic=True)
        assert a.blocks["f"] != b.blocks["f"]

    def test_string_values_are_not_docstrings(self):
        # Only the leading bare-string statement is a docstring; a string
        # *value* elsewhere is behavior.
        a = parse_blocks('FLAG = "one"\n', cosmetic=True)
        b = parse_blocks('FLAG = "two"\n', cosmetic=True)
        assert a.blocks["<module>"] != b.blocks["<module>"]

    def test_docstring_only_function_body_is_valid(self):
        fb = parse_blocks('def f():\n    """only a docstring"""\n', cosmetic=True)
        assert fb is not None
        assert "f" in fb.blocks
