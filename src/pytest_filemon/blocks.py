"""AST block analysis: code-block identity and content hashing.

A *block* is a function, a class body, or the module itself.  Blocks are
identified by their qualified name (constructed to match
``code.co_qualname`` exactly, Python 3.11+) and hashed from their AST dump --
whitespace-, comment- and position-insensitive, so reformatting or moving a
function within a file does not invalidate anything.

The special block ``MODULE_BLOCK`` is the module *skeleton*: the module AST
with every function/method body replaced by ``pass``.  The skeleton captures
exactly what executes at import time -- module-level statements, decorators,
signatures, class-level attributes, *in source order* (order is part of
import-time behavior) -- while ignoring function bodies, which do not
execute at import.  Changing a function body therefore does not
invalidate dependents that merely import the module; changing anything that
runs at import does.

Dependencies are ``Block`` triples ``(repo-relative file, block id, content
hash)``.  A ``WILDCARD`` block id means "any change to this file re-runs"
(the file-level fallback for code objects we cannot map to a block:
redefined names, dynamically compiled code, unparseable files).

Compared to pytest-testmon's block analysis: testmon maps *coverage lines*
to blocks identified by (name, line range); we map the executing *code
object* directly via ``co_qualname`` (precise, immune to code motion) and
treat the module skeleton explicitly as the import-time dependency of every
importer.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

MODULE_BLOCK = "<module>"
WILDCARD = "*"

# A dependency: (repo-relative file, block id, content hash).
Block = tuple[str, str, str]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stub_bodies(tree: ast.AST) -> ast.AST:
    """Replace every function/method body in *tree* with a single ``pass``."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [ast.Pass()]
    return tree


class FileBlocks:
    """Block table of one parsed Python file.

    ``blocks`` maps each qualified name to its content hash; ``MODULE_BLOCK``
    maps to the skeleton hash.  ``spans`` is the list of
    ``(first line incl. decorators, last line, qualified name)`` used as the
    fallback for code objects without a resolvable qualname.  ``resolve``
    maps a code object to its block.
    """

    __slots__ = ("blocks", "spans")

    def __init__(
        self, blocks: dict[str, str], spans: list[tuple[int, int, str]]
    ) -> None:
        self.blocks = blocks
        self.spans = spans

    def resolve(self, qualname: str, firstlineno: int) -> tuple[str, str] | None:
        """Map a code object to ``(block id, hash)``; None if unresolvable."""
        h = self.blocks.get(qualname)
        if h is not None:
            return qualname, h
        # Lambdas ("<...>.<lambda>") and odd code objects: fall back to the
        # innermost block whose source span contains the code object's first
        # line (decorators included in the span: co_firstlineno points at the
        # first decorator when present).
        best_start = -1
        best_qname: str | None = None
        for start, end, qname in self.spans:
            if start <= firstlineno <= end and start > best_start:
                best_start, best_qname = start, qname
        if best_qname is not None:
            return best_qname, self.blocks[best_qname]
        return None


def parse_blocks(source: str) -> FileBlocks | None:
    """Analyze Python source; None if it does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    blocks: dict[str, str] = {}
    spans: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                # Statements like if/try/for/with can define functions and
                # classes at any nesting level: recurse with the same scope.
                visit(child, prefix, in_function)
                continue
            if prefix:
                qname = (
                    prefix + "." + ("<locals>." if in_function else "") + child.name
                )
            else:
                qname = child.name
            if isinstance(child, ast.ClassDef):
                # The class body executes at import time: its hash is the
                # class skeleton (method bodies stubbed, class-level
                # statements intact).
                content = _hash(
                    ast.dump(
                        _stub_bodies(copy.deepcopy(child)), include_attributes=False
                    )
                )
                start = child.lineno
            else:
                content = _hash(ast.dump(child, include_attributes=False))
                start = (
                    child.decorator_list[0].lineno
                    if child.decorator_list
                    else child.lineno
                )
            if qname not in blocks:
                blocks[qname] = content
            elif blocks[qname] != content:
                # Same qualified name defined twice with different content
                # (e.g. branches): unresolvable -- file-level fallback.
                blocks[qname] = WILDCARD
            spans.append((start, child.end_lineno, qname))
            visit(
                child,
                qname,
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)),
            )

    visit(tree, "", False)
    blocks[MODULE_BLOCK] = _hash(ast.dump(_stub_bodies(tree), include_attributes=False))
    return FileBlocks(blocks, spans)


class BlockCache:
    """Lazily parses repo files into :class:`FileBlocks` (one parse per file)."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, FileBlocks | None] = {}

    def file_blocks(self, rel: str) -> FileBlocks | None:
        if rel not in self._cache:
            try:
                source = (self.root / rel).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                self._cache[rel] = None
            else:
                self._cache[rel] = parse_blocks(source)
        return self._cache[rel]

    def module_dep(self, rel: str) -> Block:
        """Import-time dependency on a module: its skeleton."""
        fb = self.file_blocks(rel)
        if fb is None:
            return (rel, WILDCARD, WILDCARD)
        return (rel, MODULE_BLOCK, fb.blocks[MODULE_BLOCK])
