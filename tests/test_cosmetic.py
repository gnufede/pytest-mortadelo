"""Cosmetic edit filter: on by default (closes IMPROVEMENTS item 10 / C.2).

A block's hash is computed from a docstring-stripped AST, so prose edits
produce identical hashes and select nothing.  Comments and formatting are
invisible to ``ast.dump`` regardless (free); annotations are deliberately
*kept* (typing changes are observable).  Opt out with ``--less-no-cosmetic``
or ``less_cosmetic = false`` (part of the shard key); it is also off under
``--doctest-modules``/``--doctest-glob``, where docstrings are test input.

Safety net: code that reads docstrings as data (``__doc__``, ``getdoc``,
``doctest``, ``help``) would silently miss edits.  Each shard therefore
records a per-file docstring hash and the files that read docstrings; a
changed file whose docstrings differ still selects the tests that touched it
*and* ran a docstring reader.
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


class TestCosmeticDefault:
    def test_docstring_only_edit_selects_nothing(self, doc_project):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, DOCSTRING_EDIT)
        result = run_less(doc_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_test_file_docstring_edit_also_selects_nothing(self, doc_project):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        (doc_project.path / "tests" / "test_doc.py").write_text(
            "def test_doc():\n"
            '    """The test itself, described."""\n'
            "    import lib\n"
            "    assert lib.documented() == 1\n"
        )
        result = run_less(doc_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_semantic_edit_still_selects_the_executor(self, doc_project):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, BODY_EDIT)
        result = run_less(doc_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_comment_edits_are_free(self, doc_project):
        # Comments never reach ast.dump.
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, COMMENT_EDIT)
        result = run_less(doc_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])


class TestOptOut:
    def test_flag_makes_prose_behavior(self, doc_project):
        run_less(doc_project, "-q", "--less-no-cosmetic").assert_outcomes(passed=2)
        write_lib(doc_project, DOCSTRING_EDIT)
        result = run_less(doc_project, "-q", "--less-no-cosmetic")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_ini_opt_out(self, doc_project):
        doc_project.makeini("[pytest]\nless_cosmetic = false\n")
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, DOCSTRING_EDIT)
        result = run_less(doc_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_flag_is_part_of_the_shard_key(self, doc_project):
        # Toggling the flag must not silently reinterpret an existing
        # shard: it starts a fresh record run instead.
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        result = run_less(doc_project, "-q", "--less-no-cosmetic")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*no usable stored data*"])


class _Cfg:
    """Minimal config stand-in for ``_cosmetic_enabled``."""

    def __init__(self, ini=True, **options):
        self.option = type("O", (), options)()
        self._ini = ini

    def getini(self, name):
        return self._ini


class TestEnabledHelper:
    def test_default_on(self):
        from pytest_less.plugin import _cosmetic_enabled

        assert _cosmetic_enabled(_Cfg()) is True

    @pytest.mark.parametrize(
        "cfg, args",
        [
            (_Cfg(less_no_cosmetic=True), ()),
            (_Cfg(ini=False), ()),
            (_Cfg(doctestmodules=True), ()),
            (_Cfg(doctestglob=["*.txt"]), ()),
            (_Cfg(), ("--doctest-modules",)),
            (_Cfg(), ("--doctest-glob=*.rst",)),
        ],
    )
    def test_opt_outs_and_doctest(self, cfg, args):
        from pytest_less.plugin import _cosmetic_enabled

        assert _cosmetic_enabled(cfg, args) is False


READER = (
    "def read_doc(fn):\n"
    "    return fn.__doc__\n"
)


@pytest.fixture
def reader_project(doc_project):
    """doc_project plus a test that reads ``lib.documented.__doc__`` through
    a docstring-reading helper (``reader.py``)."""
    tests = doc_project.path / "tests"
    (tests / "reader.py").write_text(READER)
    (tests / "test_reader.py").write_text(
        "def test_reader():\n"
        "    import lib, reader\n"
        "    assert 'prose' in reader.read_doc(lib.documented)\n"
    )
    commit(doc_project.path, "add reader")
    return doc_project


class TestDocReaderSafetyNet:
    def test_docstring_edit_selects_only_the_doc_reader(self, reader_project):
        run_less(reader_project, "-q").assert_outcomes(passed=3)
        write_lib(reader_project, DOCSTRING_EDIT)
        result = run_less(reader_project, "-v")
        # test_doc ran lib.documented but reads no docstrings: still
        # deselected.  test_reader read the docstring: selected.
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/3 tests*"])
        result.stdout.fnmatch_lines(["*test_reader.py::test_reader PASSED*"])
        result.stdout.fnmatch_lines(["*docstring readers*"])

    def test_non_readers_stay_deselected(self, reader_project):
        run_less(reader_project, "-q").assert_outcomes(passed=3)
        write_lib(reader_project, DOCSTRING_EDIT)
        result = run_less(reader_project, "-v")
        # test_doc and test_plain execute lib.py but never ran a docstring
        # reader: the safety net must not drag them in.
        result.stdout.no_fnmatch_line("*test_doc.py::test_doc PASSED*")
        result.stdout.no_fnmatch_line("*test_plain.py::test_plain PASSED*")

    def test_semantic_edit_unaffected(self, reader_project):
        run_less(reader_project, "-q").assert_outcomes(passed=3)
        write_lib(reader_project, BODY_EDIT)
        result = run_less(reader_project, "-v")
        # documented()'s body changed: only test_doc *executes* it
        # (test_reader merely reads its docstring) -- block precision holds.
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_doc.py::test_doc PASSED*"])

    def test_record_stores_docs_and_readers(self, reader_project):
        from test_plugin import shard_data

        run_less(reader_project, "-q").assert_outcomes(passed=3)
        data = shard_data(reader_project.path)
        assert "tests/lib.py" in data["docs"]
        assert data["doc_readers"] == ["tests/reader.py"]


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
        # plain parse: docstrings are part of every hash (module skeleton, class
        # skeleton, method) ...
        assert a.blocks["C.m"] != b.blocks["C.m"]
        assert a.blocks["<module>"] != b.blocks["<module>"]
        # ... with cosmetic: prose and comments are invisible everywhere.
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


class TestDocAnalysis:
    def test_doc_hash_tracks_only_docstrings(self):
        a = parse_blocks('def f():\n    """one"""\n    return 1\n', cosmetic=True)
        b = parse_blocks('def f():\n    """two"""\n    return 1\n', cosmetic=True)
        c = parse_blocks('def f():\n    """one"""\n    return 2\n', cosmetic=True)
        assert a.blocks == b.blocks and a.doc_hash != b.doc_hash
        assert a.doc_hash == c.doc_hash

    @pytest.mark.parametrize(
        "src",
        [
            "def f(x):\n    return x.__doc__\n",
            "import inspect\ndef f(x):\n    return inspect.getdoc(x)\n",
            "from inspect import cleandoc\n",
            "import doctest\n",
            "def f(x):\n    help(x)\n",
        ],
    )
    def test_readers_detected(self, src):
        assert parse_blocks(src).reads_docs is True

    def test_ordinary_code_is_not_a_reader(self):
        assert parse_blocks("def f(x):\n    return x + 1\n").reads_docs is False
