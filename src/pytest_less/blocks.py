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
redefined names, dynamically compiled code, unparseable files, and
non-Python data files recorded through the ``open`` audit hook).

Compared to pytest-testmon's block analysis: testmon maps *coverage lines*
to blocks identified by (name, line range); we map the executing *code
object* directly via ``co_qualname`` (precise, immune to code motion) and
treat the module skeleton explicitly as the import-time dependency of every
importer.

Two cross-cutting analyses ride along on every parse: *dynamic markers*
(reflective constructs such as computed ``getattr`` or ``__getattr__``
definitions, which selection uses to widen marker-bearing files back to
file level) and the *cosmetic normalization* (``--less-cosmetic``: docstrings
stripped before hashing, so prose-only edits do not invalidate blocks).
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

# Dynamic-access markers (the "dynscan" safety net): call sites and method
# definitions whose *targets cannot be resolved statically.  A file
# containing one of these is potentially wired by reflective edges the
# recorded run may not have seen (unstable dispatch, exec'd code), so
# selection widens a semantic change in such a file back to file level
# (see selection.dynamic_widenings).  Format: ``name@Lline``.
_DYNAMIC_CALLS = frozenset({"eval", "exec", "__import__", "import_module"})
_DYNAMIC_ATTR_CALLS = frozenset({"getattr", "setattr", "delattr"})
_DYNAMIC_METHODS = frozenset({"__getattr__", "__getattribute__"})


def find_dynamic_markers(tree: ast.AST) -> tuple[str, ...]:
    """Reflective/dynamic constructs in *tree*, as ``name@Lline`` strings.

    Deliberately ignores *statically resolvable* access: ``getattr(x, "lit")``
    names its attribute, so it cannot hide an unseen edge; a computed second
    argument (anything but a string constant) is flagged.  ``__getattr__`` /
    ``__getattribute__`` definitions are flagged wholesale: every attribute
    read on the object may dispatch to code no recording saw execute.
    """
    markers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name in _DYNAMIC_CALLS:
                markers.add(f"{name}@L{node.lineno}")
            elif name in _DYNAMIC_ATTR_CALLS:
                # Second arg a plain string literal: statically resolvable.
                if len(node.args) >= 2:
                    arg = node.args[1]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        continue
                markers.add(f"{name}@L{node.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _DYNAMIC_METHODS:
                markers.add(f"{node.name}@L{node.lineno}")
    return tuple(sorted(markers))


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstrings from every module/class/function in *tree* (in place).

    The cosmetic edit filter (``--less-cosmetic``): a docstring is prose, not
    behavior -- for code that never reads ``__doc__``.  Only true docstrings
    (a bare string as the first body statement, the ``ast.get_docstring``
    definition) are removed; string values elsewhere are behavior and stay.
    Comments and whitespace are already invisible to ``ast.dump``, so this is
    the entire textual surface the hash needs to ignore.  Annotations are
    deliberately *kept*: typing changes are observable.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                rest = body[1:]
                # A function whose entire body is its docstring still needs
                # a statement to be valid Python (hashes only, never run).
                node.body = rest or [ast.Pass()]
    return tree


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
    maps a code object to its block.  ``markers`` lists the file's dynamic
    constructs (see :func:`find_dynamic_markers`), empty for ordinary code.
    """

    __slots__ = ("blocks", "markers", "spans")

    def __init__(
        self,
        blocks: dict[str, str],
        spans: list[tuple[int, int, str]],
        markers: tuple[str, ...] = (),
    ) -> None:
        self.blocks = blocks
        self.spans = spans
        self.markers = markers

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


def parse_blocks(source: str, *, cosmetic: bool = False) -> FileBlocks | None:
    """Analyze Python source; None if it does not parse.

    With *cosmetic*, docstrings are stripped from the tree before hashing:
    a docstring-only edit then produces identical hashes for every block
    (``--less-cosmetic``; comments are invisible to ``ast.dump`` anyway).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    markers = find_dynamic_markers(tree)
    if cosmetic:
        _strip_docstrings(tree)

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
    return FileBlocks(blocks, spans, markers)


class BlockCache:
    """Lazily parses repo files into :class:`FileBlocks` (one parse per file)."""

    def __init__(self, root: Path, *, cosmetic: bool = False):
        self.root = root
        self.cosmetic = cosmetic
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
                self._cache[rel] = parse_blocks(source, cosmetic=self.cosmetic)
        return self._cache[rel]

    def module_dep(self, rel: str) -> Block:
        """Import-time dependency on a module: its skeleton."""
        fb = self.file_blocks(rel)
        if fb is None:
            return (rel, WILDCARD, WILDCARD)
        return (rel, MODULE_BLOCK, fb.blocks[MODULE_BLOCK])
