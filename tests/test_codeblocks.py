"""Block hashes from compiled code (codeblocks.py).

Blocks are hashed from ``compile(source)``, so these tests pin the invariants
that make that useful (what an edit may and may not change), how a running code
object finds its block, and what happens to files the hasher does not model.
"""

import os
import subprocess
import sys
import types

import pytest
from test_plugin import commit, git, run_less

from pytest_less import codeblocks
from pytest_less.blocks import MODULE_BLOCK, BlockCache, FileBlocks, ParseCache
from pytest_less.codeblocks import TAG, Unsupported, compile_blocks

pytestmark = pytest.mark.skipif(
    not codeblocks.SUPPORTED, reason="bytecode hashing needs Python 3.12 / 3.13"
)


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
COMMENT_EDIT = '    """Old prose."""\n    return 1  # a comment, invisible to the compiler'
BODY_EDIT = '    """Old prose."""\n    return 1 + int(0)'


def blocks(source: str) -> dict[str, str]:
    fb = compile_blocks(source.encode())
    assert fb is not None
    return fb.blocks


def changed(before: str, after: str) -> set[str]:
    """Block ids whose hash differs (including added / removed ones)."""
    a, b = blocks(before), blocks(after)
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


KLASS = '''\
import os

GLOBAL = 5


def helper(a, b=2):
    return a + b


class Widget(object):
    size = 5
    label = "w"

    def method(self, x):
        total = 0
        for i in range(x):
            total += i
        return total

    @property
    def prop(self):
        return self.size


def outer():
    class Local:
        value = 1

    def inner():
        return Local.value

    return inner
'''


class TestInvariants:
    """Edits that must not change a hash: only line numbers moved."""

    def test_blank_lines_and_comments_change_nothing(self):
        shifted = "# a comment\n\n\n" + KLASS.replace(
            "    size = 5\n", "    # explained\n    size = 5\n\n"
        )
        assert changed(KLASS, shifted) == set()

    def test_moving_a_class_down_does_not_change_it(self):
        # CPython 3.13 stores the class's own line in the class body.
        assert changed(KLASS, "\n" * 40 + KLASS) == set()

    def test_the_injected_line_number_may_collide_with_another_constant(self):
        # ``__firstlineno__`` is deduplicated against equal constants: with the
        # class on line 5 and ``x = 5`` inside it the table has one fewer entry
        # than with the class on line 6, which renumbers every later constant.
        source = "class C:\n    x = 5\n    y = 'a'\n    z = 7\n"
        for lead in range(12):
            assert changed("\n" * 4 + source, "\n" * lead + source) == set(), lead

    def test_a_class_crossing_line_255_does_not_change(self):
        # 3.14+ loads the class's own line with LOAD_SMALL_INT up to 255 and
        # with LOAD_CONST beyond: a different opcode, same meaning.
        source = "class C:\n    x = 5\n    def m(self):\n        return 1\n"
        base = blocks("\n" * 240 + source)
        for lead in range(246, 270):
            assert blocks("\n" * lead + source) == base, lead

    def test_a_method_gaining_an_attribute_does_not_change_the_class(self):
        # ``__static_attributes__`` lists self.x assignments of *method bodies*.
        before = "class C:\n    def m(self):\n        return 1\n"
        after = "class C:\n    def m(self):\n        self.x = 1\n        return 1\n"
        assert changed(before, after) == {"C.m"}

    def test_equal_bytecode_is_equal_behavior(self):
        # Whatever the compiler folds or drops cannot change behavior.  (Not
        # everything it drops is invisible: a dead branch leaves its constant
        # in the table and a bare string statement leaves a NOP, so *adding*
        # those over-selects -- the safe side.  Editing them does not.)
        pairs = [
            ("def f(a):\n    return a + 1\n", "def f(a):\n  return (a +\n         1)\n"),
            (  # the text of a no-op string statement
                "def f(a):\n    b = a\n    'one'\n    return a + 1\n",
                "def f(a):\n    b = a\n    'two, longer'\n    return a + 1\n",
            ),
            (  # annotations of local variables are never evaluated
                "def f(a):\n    x: int\n    return a + 1\n",
                "def f(a):\n    x: Dict[str, 'Anything']\n    return a + 1\n",
            ),
            (  # attribute docstrings in a class body
                "class C:\n    x = 1\n    'about x'\n",
                "class C:\n    x = 1\n    'about x, rewritten'\n",
            ),
            (  # ``from x import A`` binds the same name as ``... as A``
                "from os import path\n",
                "from os import path as path\n",
            ),
        ]
        if sys.version_info < (3, 14):
            # 3.14+ folds in the flow graph and leaves the folded operand
            # behind in co_consts, unreferenced: that edit over-selects there.
            pairs.append(
                ("def f(a):\n    return a + 1\n", "def f(a):\n    return a + (0 + 1)\n")
            )
        for before, after in pairs:
            assert changed(before, after) == set(), (before, after)

    def test_toggling_future_annotations_leaves_unrelated_functions_alone(self):
        # The flag leaks into every code object of the file; its real effect
        # is in the parent's bytecode (annotations become strings).
        plain = "def f(a):\n    return a\n\ndef g(b):\n    return b\n"
        future = "from __future__ import annotations\n" + plain
        assert changed(plain, future) == {MODULE_BLOCK}

    def test_hashes_do_not_depend_on_the_string_hash_seed(self):
        source = "def f(x):\n    return x in {'a', 'b', 'c', 'd', 'e', 'f'}\n"
        code = (
            "import sys; from pytest_less.codeblocks import compile_blocks;"
            "print(sorted(compile_blocks(sys.stdin.buffer.read()).blocks.items()))"
        )
        outs = {
            subprocess.run(
                [sys.executable, "-c", code],
                input=source.encode(),
                capture_output=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
                check=True,
            ).stdout
            for seed in ("0", "1", "12345", "999")
        }
        assert len(outs) == 1

    def test_hashes_carry_the_algorithm_and_python_version(self):
        assert TAG == f"c{sys.version_info[1]}"
        assert all(h.startswith(TAG) for h in blocks(KLASS).values())


class TestSensitivity:
    """Edits that must change a hash, and *which* ones (the parent rule)."""

    def test_a_body_edit_changes_only_that_function(self):
        after = KLASS.replace("total += i", "total += i * 2")
        assert changed(KLASS, after) == {"Widget.method"}

    def test_a_class_level_statement_changes_the_class_and_the_module(self):
        after = KLASS.replace("size = 5", "size = 6")
        # the module skeleton contains the class skeleton; method bodies
        # (``self.size`` in the property) are untouched
        assert changed(KLASS, after) == {"Widget", MODULE_BLOCK}

    def test_decorator_and_default_belong_to_the_parent(self):
        # They are evaluated by the enclosing scope, so they change *its*
        # block -- which always runs when the function object is created.
        base = "class C:\n    def m(self, x=1) -> int:\n        return x\n"
        for edit in (
            base.replace("x=1", "x=2"),
            base.replace("    def m", "    @staticmethod\n    def m"),
        ):
            assert "C" in changed(base, edit), edit

    def test_annotation_edits_follow_the_interpreters_evaluation_model(self):
        base = "class C:\n    def m(self, x=1) -> int:\n        return x\n"
        edit = changed(base, base.replace("-> int", "-> str"))
        if sys.version_info < (3, 14):
            # evaluated eagerly by the enclosing scope
            assert "C" in edit
        else:
            # PEP 649: annotations live in their own lazily-run __annotate__
            # function, a block of its own.  Only a test that evaluated it
            # (get_type_hints, dataclasses, ...) depends on it, and it is
            # recorded like any other executed block.
            assert "C" not in edit and MODULE_BLOCK not in edit
            assert edit and all(k.endswith("__annotate__") for k in edit), edit

    def test_adding_an_annotation_changes_the_parent_on_every_version(self):
        # A new __annotate__ child appears (3.14+) / the annotation tuple does
        # (before): either way the enclosing block sees it.
        base = "class C:\n    def m(self, x):\n        return x\n"
        assert "C" in changed(base, base.replace("(self, x)", "(self, x: int)"))

    def test_a_renamed_argument_changes_function_and_class(self):
        base = "class C:\n    def m(self, x):\n        return x\n"
        assert changed(base, base.replace("x", "y")) == {"C", "C.m", MODULE_BLOCK}

    def test_async_is_part_of_the_signature(self):
        base = "class C:\n    def m(self):\n        return 1\n"
        assert "C" in changed(base, base.replace("def m", "async def m"))

    def test_nested_definitions_are_part_of_the_enclosing_function(self):
        after = KLASS.replace("return Local.value", "return Local.value + 1")
        assert changed(KLASS, after) == {"outer", "outer.<locals>.inner"}

    def test_a_docstring_is_behavior_here(self):
        # Docstrings are not ignored (there is no cosmetic filter): a docstring edit
        # selects.  (The safe direction: nothing is ever skipped because of it.)
        base = 'def f():\n    """Old."""\n    return None\n'
        assert changed(base, base.replace("Old", "New")) == {"f"}
        assert changed(base, base.replace('    """Old."""\n', "")) == {"f"}
        mod = '"""Old."""\n' + KLASS
        assert MODULE_BLOCK in changed(mod, mod.replace("Old", "New"))
        cls = 'class C:\n    """Old."""\n    x = 1\n'
        assert "C" in changed(cls, cls.replace("Old", "New"))


class TestResolve:
    def test_exact_match_by_qualname(self):
        fb = compile_blocks(KLASS.encode())
        assert fb.resolve("Widget.method", 15) == ("Widget.method", fb.blocks["Widget.method"])

    def test_lambdas_and_comprehensions_belong_to_their_lexical_parent(self):
        src = (
            "f = lambda: 1\n"
            "class C:\n"
            "    g = lambda self: 2\n"
            "    def m(self):\n"
            "        return list(i for i in range(3)), (lambda: 4)\n"
        )
        fb = compile_blocks(src.encode())
        top = compile(src, "x", "exec")
        found = {}

        def walk(code):
            for c in code.co_consts:
                if isinstance(c, types.CodeType):
                    found[c.co_qualname] = fb.resolve(c.co_qualname, c.co_firstlineno)[0]
                    walk(c)

        walk(top)
        assert found["<lambda>"] == MODULE_BLOCK  # a module-level lambda
        assert found["C.<lambda>"] == "C"
        assert found["C.m.<locals>.<lambda>"] == "C.m"
        assert found["C.m.<locals>.<genexpr>"] == "C.m"

    def test_a_class_declared_global_is_found_under_its_real_qualname(self):
        # The qualname is ``K``, not ``g.<locals>.K``: the real ``co_qualname``.
        src = "def g():\n    global K\n    class K:\n        pass\n    return K\n"
        fb = compile_blocks(src.encode())
        assert "K" in fb.blocks and fb.resolve("K", 3)[0] == "K"

    def test_redefinitions_get_one_id_each_and_resolve_by_line(self):
        src = (
            "class C:\n"
            "    @property\n"
            "    def v(self):\n"
            "        return 1\n"
            "    @v.setter\n"
            "    def v(self, value):\n"
            "        self._v = value\n"
        )
        fb = compile_blocks(src.encode())
        ids = [k for k in fb.blocks if k.startswith("C.v")]
        assert len(ids) == 2 and all("~" in k for k in ids)
        getter, setter = fb.resolve("C.v", 2)[0], fb.resolve("C.v", 5)[0]
        assert getter != setter

    def test_every_def_and_class_resolves_to_its_own_block(self):
        # Including 3.14's ``__annotate__`` functions, which several definitions
        # of one scope share a qualname for (told apart by their first line),
        # and redefinitions with the same qualname.
        src = (
            "def g(a: int, b: str) -> int:\n    return a\n"
            "def h(c: float) -> int:\n    return 1\n"
            "x: int = 1\n"
            "class K:\n    y: int = 2\n"
            "    def m(self, d: bytes) -> None: ...\n"
            "    def n(self, e: list) -> None: ...\n"
            "    @property\n    def p(self): return 1\n"
            "    @p.setter\n    def p(self, v: int): pass\n"
            "def outer():\n    def inner(z: int) -> int:\n        return z\n    return inner\n"
        )
        fb = compile_blocks(src.encode())
        top = compile(src, "m.py", "exec")
        checked = 0
        stack = [top]
        while stack:
            code = stack.pop()
            stack += [c for c in code.co_consts if isinstance(c, types.CodeType)]
            if code is top or code.co_name.startswith("<"):
                continue
            block_id, _ = fb.resolve(code.co_qualname, code.co_firstlineno)
            assert block_id.split("~")[0] == code.co_qualname, (code.co_qualname, block_id)
            checked += 1
        assert checked >= 8

    def test_unknown_code_belongs_to_the_module(self):
        fb = compile_blocks(b"x = 1\n")
        assert fb.resolve("never_defined", 1)[0] == MODULE_BLOCK


class TestMarkersAreLazy:
    SRC = b"def f(o, name):\n    return getattr(o, name)\n"

    def test_not_computed_until_asked(self):
        fb = compile_blocks(self.SRC)
        assert fb.known_markers() is None
        assert fb.markers  # computed now: the computed-attribute call
        assert fb.known_markers() == fb.markers

    def test_without_source_the_answer_is_dynamic_never_empty(self):
        fb = compile_blocks(self.SRC)
        restored = FileBlocks(fb.blocks, fb.spans, None)  # no bytes handed over
        assert restored.markers  # nothing to scan: widen, do not switch off


class TestUnsupportedFiles:
    """What the hasher does not model becomes a file-level dependency."""

    def test_type_statements_are_not_modelled(self):
        with pytest.raises(Unsupported):
            compile_blocks(b"type Alias = int\n")

    def test_generics_are_not_modelled(self):
        with pytest.raises(Unsupported):
            compile_blocks(b"def f[T](x: T) -> T:\n    return x\n")

    def test_a_syntax_error_is_none(self):
        assert compile_blocks(b"def broken(:\n") is None

    def test_block_cache_gives_such_files_a_wildcard_dependency(self, tmp_path):
        (tmp_path / "a.py").write_text("def f():\n    return 1\n")
        (tmp_path / "b.py").write_text("type A = int\n")
        cache = BlockCache(tmp_path)
        assert cache.file_blocks("a.py").blocks[MODULE_BLOCK].startswith(TAG)
        assert cache.file_blocks("b.py") is None
        assert cache.module_dep("b.py") == ("b.py", "*", "*")  # any change re-runs

    def test_an_unsupported_interpreter_is_file_level_throughout(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(codeblocks, "SUPPORTED", False)
        (tmp_path / "a.py").write_text("def f():\n    return 1\n")
        with pytest.raises(Unsupported):
            compile_blocks(b"x = 1\n")
        assert BlockCache(tmp_path).file_blocks("a.py") is None

    def test_an_edit_to_such_a_file_selects_what_executed_it(self, pytester):
        tests = pytester.path / "tests"
        tests.mkdir()
        (tests / "conftest.py").touch()
        (tests / "lib.py").write_text("type A = int\n\ndef f():\n    return 1\n")
        (tests / "other.py").write_text("def g():\n    return 2\n")
        for name, mod, call in (("a", "lib", "f"), ("b", "other", "g")):
            (tests / f"test_{name}.py").write_text(
                f"def test_{name}():\n    import {mod}\n    assert {mod}.{call}()\n"
            )
        (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
        git(pytester.path, "init", "-q")
        git(pytester.path, "config", "user.email", "t@example.com")
        git(pytester.path, "config", "user.name", "T")
        commit(pytester.path, "initial")
        run_less(pytester, "-q").assert_outcomes(passed=2)
        (tests / "lib.py").write_text("type A = int\n\ndef f():\n    return 1  # c\n")
        result = run_less(pytester, "-q")  # file level: even a comment re-runs it
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])


class TestCache:
    def test_round_trip_does_not_force_the_marker_scan(self, tmp_path):
        (tmp_path / "m.py").write_text("def f(o, n):\n    return getattr(o, n)\n")
        cache = ParseCache(None)
        fb = BlockCache(tmp_path, parse_cache=cache).file_blocks("m.py")
        assert fb.known_markers() is None
        stored = cache.dump({"m.py"})
        assert stored["files"]["m.py"]["m"] is None
        again = BlockCache(
            tmp_path, parse_cache=ParseCache(lambda: stored)
        ).file_blocks("m.py")
        assert again.blocks == fb.blocks and again.spans == fb.spans
        assert again.markers  # lazy, from the bytes handed over on the hit

    def test_a_cached_table_resolves_like_a_fresh_one(self, tmp_path):
        (tmp_path / "m.py").write_text(KLASS)
        cache = ParseCache(None)
        fresh = BlockCache(tmp_path, parse_cache=cache).file_blocks("m.py")
        hit = BlockCache(
            tmp_path, parse_cache=ParseCache(lambda: cache.dump({"m.py"}))
        ).file_blocks("m.py")
        for name, line in (("Widget.method", 15), ("outer.<locals>.inner", 27),
                           ("never_defined", 1), ("Widget.<lambda>", 12)):
            assert hit.resolve(name, line) == fresh.resolve(name, line)


class TestAnnotationsEndToEnd:
    """Only a test that *evaluated* an annotation depends on its text (3.14+)."""

    @pytest.fixture
    def annotated(self, pytester):
        tests = pytester.path / "tests"
        tests.mkdir()
        (tests / "conftest.py").touch()
        (tests / "lib.py").write_text("def f(x: int) -> int:\n    return x\n")
        (tests / "test_hints.py").write_text(
            "def test_hints():\n"
            "    import typing, lib\n"
            "    assert typing.get_type_hints(lib.f)['x'] is int\n"
        )
        (tests / "test_calls.py").write_text(
            "def test_calls():\n    import lib\n    assert lib.f(1) == 1\n"
        )
        (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
        git(pytester.path, "init", "-q")
        git(pytester.path, "config", "user.email", "t@example.com")
        git(pytester.path, "config", "user.name", "T")
        commit(pytester.path, "initial")
        return pytester

    def test_the_test_that_read_the_annotation_reruns(self, annotated):
        run_less(annotated, "-q").assert_outcomes(passed=2)
        (annotated.path / "tests" / "lib.py").write_text(
            "def f(x: float) -> int:\n    return x\n"
        )
        result = run_less(annotated, "-v")
        # it changed what get_type_hints returns, so that test must rerun --
        # on every version
        assert "test_hints.py::test_hints" in result.stdout.str()
        result.stdout.fnmatch_lines(["*test_hints.py::test_hints*"])
        if sys.version_info >= (3, 14):
            # test_calls never evaluated an annotation: nothing it ran changed
            result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
            assert "test_calls.py::test_calls PASSED" not in result.stdout.str()
        else:
            # eagerly evaluated by the module: the skeleton changed for everyone
            result.stdout.fnmatch_lines(["*selected 2/2 tests*"])


class TestEndToEnd:
    def test_a_docstring_edit_selects_the_executor(self, doc_project):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, '    """New prose."""\n    return 1')
        result = run_less(doc_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_a_comment_or_folded_edit_selects_nothing(self, doc_project):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        for body in (
            '    """Old prose."""\n    # now with a comment\n    return 1',
            '    """Old prose."""\n    return 1 + 0',  # folded by the compiler
        ):
            write_lib(doc_project, body)
            result = run_less(doc_project, "-q")
            assert result.ret == 5
            result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_a_real_edit_selects_the_executor(self, doc_project):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, '    """Old prose."""\n    return 1 + int(0)')
        result = run_less(doc_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
