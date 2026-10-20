"""Block hashes from compiled code (codeblocks.py).

The point of the bytecode hasher is to give the same answers as the AST hasher
about *what changed* at a fraction of the cost, so these tests pin the
invariants it must keep (what an edit may and may not change) and compare it
with the AST hasher on real code.
"""

# ruff: noqa: F811  (shared fixtures are imported, then requested by name)
import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from test_cosmetic import doc_project, write_lib  # noqa: F401
from test_plugin import run_less

from pytest_less import codeblocks
from pytest_less.blocks import (
    MODULE_BLOCK,
    BlockCache,
    LocalParseStore,
    ParseCache,
    hash_mode,
    parse_blocks,
)
from pytest_less.codeblocks import TAG, Unsupported, compile_blocks

pytestmark = pytest.mark.skipif(
    not codeblocks.SUPPORTED, reason="bytecode hashing needs Python 3.12 / 3.13"
)


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
            ("def f(a):\n    return a + 1\n", "def f(a):\n    return a + (0 + 1)\n"),
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

    def test_decorator_default_and_annotation_belong_to_the_parent(self):
        # They are evaluated by the enclosing scope, so they change *its*
        # block -- which always runs when the function object is created.
        base = "class C:\n    def m(self, x=1) -> int:\n        return x\n"
        for edit in (
            base.replace("x=1", "x=2"),
            base.replace("-> int", "-> str"),
            base.replace("    def m", "    @staticmethod\n    def m"),
        ):
            assert "C" in changed(base, edit), edit

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
        # The AST hasher's cosmetic filter is not available: a docstring edit
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
        # The AST path builds ``g.<locals>.K``, which no code object carries.
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
        restored = type(fb).restore(fb.blocks, fb.spans, None)
        assert restored.markers  # nothing to scan: widen, do not switch off


class TestFallback:
    def test_type_statements_use_the_ast_path(self):
        with pytest.raises(Unsupported):
            compile_blocks(b"type Alias = int\n")

    def test_generics_use_the_ast_path(self):
        with pytest.raises(Unsupported):
            compile_blocks(b"def f[T](x: T) -> T:\n    return x\n")

    def test_a_syntax_error_is_none(self):
        assert compile_blocks(b"def broken(:\n") is None

    def test_block_cache_falls_back_per_file(self, tmp_path, bytecode_hashing):
        (tmp_path / "a.py").write_text("def f():\n    return 1\n")
        (tmp_path / "b.py").write_text("type A = int\n")
        cache = BlockCache(tmp_path)
        assert cache.file_blocks("a.py").kind == "c"
        assert cache.file_blocks("b.py").kind == "a"  # unsupported: AST
        assert cache.file_blocks("b.py").blocks[MODULE_BLOCK]
        assert not cache.file_blocks("b.py").blocks[MODULE_BLOCK].startswith(TAG)


class TestAlgorithmKeys:
    """An entry is only ever served to the algorithm that produced it."""

    def test_mode_switch(self, monkeypatch):
        monkeypatch.setenv("PYTEST_LESS_HASH", "ast")
        assert hash_mode() == "ast"
        monkeypatch.setenv("PYTEST_LESS_HASH", "bytecode")
        assert hash_mode() == "bytecode"

    def test_the_repository_store_is_per_algorithm(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTEST_LESS_HASH", "ast")
        a = LocalParseStore.at(tmp_path, cosmetic=True).directory
        monkeypatch.setenv("PYTEST_LESS_HASH", "bytecode")
        b = LocalParseStore.at(tmp_path, cosmetic=True).directory
        assert a != b

    def test_a_shard_cache_from_the_other_algorithm_is_ignored(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "m.py").write_text("def f():\n    return 1\n")
        monkeypatch.setenv("PYTEST_LESS_HASH", "ast")
        ast_cache = ParseCache(None, cosmetic=False)
        BlockCache(tmp_path, parse_cache=ast_cache).file_blocks("m.py")
        stored = ast_cache.dump({"m.py"})
        monkeypatch.setenv("PYTEST_LESS_HASH", "bytecode")
        cache = ParseCache(lambda: stored, cosmetic=False)
        bc = BlockCache(tmp_path, parse_cache=cache)
        assert bc.file_blocks("m.py").kind == "c"
        assert (bc.hits, bc.misses) == (0, 1)

    def test_round_trip_does_not_force_the_marker_scan(self, tmp_path, bytecode_hashing):
        (tmp_path / "m.py").write_text("def f(o, n):\n    return getattr(o, n)\n")
        cache = ParseCache(None, cosmetic=False)
        bc = BlockCache(tmp_path, parse_cache=cache)
        fb = bc.file_blocks("m.py")
        assert fb.known_markers() is None
        stored = cache.dump({"m.py"})
        assert stored["files"]["m.py"]["m"] is None
        again = BlockCache(
            tmp_path, parse_cache=ParseCache(lambda: stored, cosmetic=False)
        ).file_blocks("m.py")
        assert again.kind == "c" and again.blocks == fb.blocks
        assert again.markers  # lazy, from the bytes handed over on the hit


class TestAgreesWithTheAstHasher:
    """The same questions, asked of this repository's own source."""

    ROOT = Path(__file__).resolve().parent.parent
    FILES = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))

    def test_same_blocks_and_same_resolution_on_every_file(self):
        compared = 0
        for path in self.FILES:
            data = path.read_bytes()
            try:
                c = compile_blocks(data, filename=str(path))
            except Unsupported:
                continue
            a = parse_blocks(data.decode(), cosmetic=False)
            assert c is not None and a is not None, path
            assert {k.split("~")[0] for k in c.blocks} == {
                k.split("~")[0] for k in a.blocks
            }, path
            top = compile(data, str(path), "exec")
            stack = [top]
            while stack:
                code = stack.pop()
                stack += [k for k in code.co_consts if isinstance(k, types.CodeType)]
                if code is top or code.co_name.startswith("<"):
                    continue
                ra = a.resolve(code.co_qualname, code.co_firstlineno)
                rc = c.resolve(code.co_qualname, code.co_firstlineno)
                assert ra[0].split("~")[0] == rc[0].split("~")[0], (path, code.co_qualname)
                compared += 1
        assert compared > 500

    def test_a_real_edit_is_seen_by_both(self):
        """Wrap the value of the first ``return`` in a call (never folded)."""
        seen = 0
        for path in self.FILES[:60]:
            tree = ast.parse(path.read_text())
            node = next(
                (n for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value),
                None,
            )
            if node is None:
                continue
            base = ast.unparse(tree)
            node.value = ast.Call(ast.Name("wrapped_x", ast.Load()), [node.value], [])
            edited = ast.unparse(tree)
            try:
                c_changed = blocks(base) != blocks(edited)
            except Unsupported:
                continue
            a_changed = parse_blocks(base).blocks != parse_blocks(edited).blocks
            assert a_changed and c_changed, path
            seen += 1
        assert seen > 20


class TestEndToEnd:
    def test_default_mode_is_bytecode(self, monkeypatch):
        monkeypatch.delenv("PYTEST_LESS_HASH", raising=False)
        assert hash_mode() == "bytecode"

    def test_a_docstring_edit_selects_the_executor(self, doc_project, bytecode_hashing):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, '    """New prose."""\n    return 1')
        result = run_less(doc_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_a_comment_or_folded_edit_selects_nothing(self, doc_project, bytecode_hashing):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        for body in (
            '    """Old prose."""\n    # now with a comment\n    return 1',
            '    """Old prose."""\n    return 1 + 0',  # folded by the compiler
        ):
            write_lib(doc_project, body)
            result = run_less(doc_project, "-q")
            assert result.ret == 5
            result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_a_real_edit_selects_the_executor(self, doc_project, bytecode_hashing):
        run_less(doc_project, "-q").assert_outcomes(passed=2)
        write_lib(doc_project, '    """Old prose."""\n    return 1 + int(0)')
        result = run_less(doc_project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
