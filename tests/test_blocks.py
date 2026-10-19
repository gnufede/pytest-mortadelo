"""Unit tests for AST block analysis (blocks.py)."""

import textwrap

import pytest

from pytest_less.blocks import (
    MODULE_BLOCK,
    WILDCARD,
    BlockCache,
    parse_blocks,
)

SRC = textwrap.dedent(
    """
    import os

    CONST = compute(1)

    @deco(arg=2)
    def outer(a, b=compute(3)):
        def inner():
            return 1
        return [i for i in range(3)]

    class C:
        attr = compute(4)

        def m(self):
            def n():
                pass
            return 2

    def tail():
        return 3
    """
)


def blocks_of(src: str):
    fb = parse_blocks(src)
    assert fb is not None
    return fb


class TestStructure:
    def test_block_ids_match_qualnames(self):
        fb = blocks_of(SRC)
        assert set(fb.blocks) == {
            "outer",
            "outer.<locals>.inner",
            "C",
            "C.m",
            "C.m.<locals>.n",
            "tail",
            MODULE_BLOCK,
        }

    def test_qualname_resolution_against_real_code_objects(self):
        # The whole point: every code object of a real module must resolve to
        # its own block via co_qualname.
        fb = blocks_of(SRC)
        code = compile(SRC, "mod.py", "exec")

        def walk(c):
            yield c
            for k in c.co_consts:
                if hasattr(k, "co_code"):
                    yield from walk(k)

        for c in walk(code):
            resolved = fb.resolve(c.co_qualname, c.co_firstlineno)
            assert resolved is not None, c.co_qualname
            if c.co_qualname == MODULE_BLOCK:
                assert resolved[0] == MODULE_BLOCK
            else:
                assert resolved[0] == c.co_qualname

    def test_lambda_resolves_to_enclosing_block(self):
        src = "def f(x):\n    g = lambda q: q + 1\n    return g(x)\n"
        fb = blocks_of(src)
        code = compile(src, "m.py", "exec")
        (f_code,) = [k for k in code.co_consts if hasattr(k, "co_code")]
        (lambda_code,) = [
            k for k in f_code.co_consts if hasattr(k, "co_code")
        ]
        assert lambda_code.co_qualname == "f.<locals>.<lambda>"
        resolved = fb.resolve(lambda_code.co_qualname, lambda_code.co_firstlineno)
        assert resolved[0] == "f"

    def test_decorated_function_span_includes_decorators(self):
        # co_firstlineno points at the first decorator, so spans must start
        # there for the fallback to work.
        src = "@deco\ndef f():\n    return 1\n"
        fb = blocks_of(src)
        assert fb.spans[0][0] == 1  # decorator line, not the def line

    def test_duplicate_names_get_their_own_ids(self):
        src = "def f():\n    return 1\nif COND:\n    def f():\n        return 2\n"
        fb = blocks_of(src)
        assert "f" not in fb.blocks  # ambiguous: no bare id
        ids = sorted(k for k in fb.blocks if k.startswith("f~"))
        assert len(ids) == 2 and WILDCARD not in fb.blocks.values()

    def test_identical_duplicates_stay_one_block(self):
        src = "def f():\n    return 1\nif COND:\n    def f():\n        return 1\n"
        fb = blocks_of(src)
        assert "f" in fb.blocks and not [k for k in fb.blocks if "~" in k]

    def test_resolve_picks_the_definition_by_line(self):
        src = "def f():\n    return 1\nif COND:\n    def f():\n        return 2\n"
        fb = blocks_of(src)
        first = fb.resolve("f", 1)
        second = fb.resolve("f", 4)
        assert first[0] != second[0] and first[1] != second[1]
        assert first[0] == f"f~{first[1]}"

    def test_overload_stub_edit_does_not_touch_the_implementation(self):
        template = (
            "from typing import overload\n"
            "@overload\n"
            "def g(x: int) -> int: ...\n"
            "@overload\n"
            "def g(x: str) -> str: {stub}\n"
            "def g(x):\n"
            "    return x\n"
        )
        a = blocks_of(template.format(stub="..."))
        b = blocks_of(template.format(stub="# edited\n    ..."))  # comment only
        c = blocks_of(template.format(stub="...").replace("str) -> str", "bytes) -> bytes"))
        impl = a.resolve("g", 6)
        assert impl is not None
        # comment edit: nothing changes
        assert b.blocks.get(impl[0]) == impl[1]
        # editing an overload *signature* leaves the implementation block intact
        assert c.blocks.get(impl[0]) == impl[1]

    def test_property_getter_and_setter_are_separate_blocks(self):
        src = (
            "class C:\n"
            "    @property\n"
            "    def x(self):\n"
            "        return 1\n"
            "    @x.setter\n"
            "    def x(self, v):\n"
            "        self._x = v\n"
        )
        fb = blocks_of(src)
        getter = fb.resolve("C.x", 2)
        setter = fb.resolve("C.x", 5)
        assert getter[0] != setter[0]

    def test_function_inside_if_is_a_module_level_block(self):
        src = "if COND:\n    def f():\n        return 1\n"
        fb = blocks_of(src)
        assert "f" in fb.blocks
        assert fb.blocks[MODULE_BLOCK] != parse_blocks("x = 1\n").blocks[MODULE_BLOCK]

    def test_syntax_error_is_none(self):
        assert parse_blocks("def f(:\n    pass\n") is None


class TestHashing:
    def test_skeleton_ignores_function_bodies(self):
        changed = SRC.replace("return 2", "return 99")  # inside C.m
        assert blocks_of(changed).blocks[MODULE_BLOCK] == blocks_of(SRC).blocks[
            MODULE_BLOCK
        ]

    def test_skeleton_reacts_to_module_level_changes(self):
        for changed in (
            SRC.replace("CONST = compute(1)", "CONST = compute(2)"),
            SRC.replace("def tail():", "def tail2():"),  # new/removed def
            SRC.replace("def outer(a, b=compute(3)):", "def outer(a, b=compute(4)):"),
            SRC.replace("attr = compute(4)", "attr = compute(5)"),
        ):
            assert blocks_of(changed).blocks[MODULE_BLOCK] != blocks_of(SRC).blocks[
                MODULE_BLOCK
            ]

    def test_function_hash_reacts_to_body_change(self):
        changed = SRC.replace("return 3", "return 4")
        fb1, fb2 = blocks_of(SRC), blocks_of(changed)
        assert fb2.blocks["tail"] != fb1.blocks["tail"]
        # ...and only to that function:
        assert fb2.blocks["outer"] == fb1.blocks["outer"]
        assert fb2.blocks[MODULE_BLOCK] == fb1.blocks[MODULE_BLOCK]

    def test_formatting_and_comments_do_not_matter(self):
        changed = SRC.replace("import os", "# a new comment\nimport os").replace(
            "    return 3", "    return    3"
        )
        fb1, fb2 = blocks_of(SRC), blocks_of(changed)
        assert fb1.blocks == fb2.blocks

    def test_moving_a_function_keeps_its_hash(self):
        # The function's own block is position-insensitive... but the module
        # skeleton is order-sensitive (order is import-time behavior), so
        # moving a top-level function re-runs importers while dependents of
        # the function itself keep their (unchanged) hash.
        moved = SRC.replace(
            "def tail():\n    return 3\n",
            "",
        ).replace(
            "class C:",
            "def tail():\n    return 3\n\n\nclass C:",
        )
        fb1, fb2 = blocks_of(SRC), blocks_of(moved)
        assert fb2.blocks["tail"] == fb1.blocks["tail"]
        assert fb2.blocks["outer"] == fb1.blocks["outer"]
        assert fb2.blocks[MODULE_BLOCK] != fb1.blocks[MODULE_BLOCK]

    def test_class_block_is_a_skeleton(self):
        # method body change: neither class block nor module skeleton change
        changed = SRC.replace("return 2", "return 22")
        fb1, fb2 = blocks_of(SRC), blocks_of(changed)
        assert fb2.blocks["C"] == fb1.blocks["C"]
        assert fb2.blocks[MODULE_BLOCK] == fb1.blocks[MODULE_BLOCK]
        assert fb2.blocks["C.m"] != fb1.blocks["C.m"]
        # class-level change: both change
        changed = SRC.replace("attr = compute(4)", "attr2 = compute(4)")
        fb3 = blocks_of(changed)
        assert fb3.blocks["C"] != fb1.blocks["C"]
        assert fb3.blocks[MODULE_BLOCK] != fb1.blocks[MODULE_BLOCK]

    def test_signature_change_changes_both(self):
        changed = SRC.replace("def tail():", "def tail(x):")
        fb1, fb2 = blocks_of(SRC), blocks_of(changed)
        # 'tail' renamed-in-place keeps its position but new signature:
        assert fb2.blocks["tail"] != fb1.blocks["tail"]
        assert fb2.blocks[MODULE_BLOCK] != fb1.blocks[MODULE_BLOCK]


class TestBlockCache:
    def test_module_dep_and_wildcards(self, tmp_path):
        root = tmp_path
        (root / "good.py").write_text("X = 1\n")
        (root / "bad.py").write_text("def broken(:\n")
        cache = BlockCache(root)
        dep = cache.module_dep("good.py")
        assert dep[0] == "good.py"
        assert dep[1] == MODULE_BLOCK
        assert len(dep[2]) == 16
        assert cache.module_dep("bad.py") == ("bad.py", WILDCARD, WILDCARD)
        assert cache.module_dep("missing.py") == ("missing.py", WILDCARD, WILDCARD)
        # one parse per file (cache)
        assert cache.file_blocks("good.py") is cache.file_blocks("good.py")


# ---------------------------------------------------------------------------
# The skeleton hash used to be computed on a deepcopy with function bodies
# stubbed.  It is now computed in place (swap bodies, dump, restore) -- 5x
# cheaper on big files.  Hashes MUST stay byte-identical, otherwise every
# stored shard would go stale at once.


def _legacy_skeleton_hashes(source: str, cosmetic: bool) -> dict[str, str]:
    import ast
    import copy

    from pytest_less.blocks import _hash, _strip_docstrings

    tree = ast.parse(source)
    if cosmetic:
        _strip_docstrings(tree)

    def stub(t):
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n.body = [ast.Pass()]
        return t

    out: dict[str, str] = {}

    def visit(node, prefix, in_function):
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                visit(child, prefix, in_function)
                continue
            qname = (
                prefix + "." + ("<locals>." if in_function else "") + child.name
                if prefix
                else child.name
            )
            if isinstance(child, ast.ClassDef):
                h = _hash(ast.dump(stub(copy.deepcopy(child)), include_attributes=False))
                out.setdefault(qname, h)
            visit(child, qname, not isinstance(child, ast.ClassDef))

    visit(tree, "", False)
    out["<module>"] = _hash(ast.dump(stub(tree), include_attributes=False))
    return out


def _corpus() -> list[str]:
    import ast
    import dataclasses
    import inspect
    import pathlib
    import typing

    import pytest_less

    files = [ast.__file__, dataclasses.__file__, inspect.__file__, typing.__file__]
    root = pathlib.Path(pytest_less.__file__).parent
    files += [str(p) for p in root.glob("*.py")]
    files += [str(p) for p in pathlib.Path(__file__).parent.glob("test_*.py")]
    return files


@pytest.mark.parametrize("cosmetic", [False, True])
def test_skeleton_hashes_are_unchanged_by_the_in_place_algorithm(cosmetic):
    import pathlib

    from pytest_less.blocks import WILDCARD, parse_blocks

    checked = 0
    for path in _corpus():
        source = pathlib.Path(path).read_text(encoding="utf-8")
        legacy = _legacy_skeleton_hashes(source, cosmetic)
        fb = parse_blocks(source, cosmetic=cosmetic)
        assert fb is not None, path
        for qname, expected in legacy.items():
            got = fb.blocks.get(qname)
            if got is None or got == WILDCARD:  # ambiguous duplicate: its id carries a hash
                continue
            assert got == expected, (path, qname)
            checked += 1
    assert checked > 100  # the corpus really exercised classes


def test_hashing_does_not_mutate_the_tree_it_walks():
    # In-place body swapping must restore everything: a second parse of the
    # same source (and a function hash computed after a class hash) agree.
    from pytest_less.blocks import parse_blocks

    src = (
        "class A:\n"
        "    def m(self):\n"
        "        def inner():\n"
        "            return 1\n"
        "        return inner\n"
        "    class B:\n"
        "        def n(self):\n"
        "            return 2\n"
        "def f():\n"
        "    return A\n"
    )
    assert parse_blocks(src).blocks == parse_blocks(src).blocks
    fb = parse_blocks(src)
    assert {"A", "A.m", "A.m.<locals>.inner", "A.B", "A.B.n", "f", "<module>"} <= set(
        fb.blocks
    )
