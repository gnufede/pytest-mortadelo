"""Unit tests for AST block analysis (blocks.py)."""

import textwrap

from pytest_filemon.blocks import (
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

    def test_duplicate_names_fall_back_to_wildcard(self):
        src = "def f():\n    return 1\nif COND:\n    def f():\n        return 2\n"
        fb = blocks_of(src)
        assert fb.blocks["f"] == WILDCARD

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
