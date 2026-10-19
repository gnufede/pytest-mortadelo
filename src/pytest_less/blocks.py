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
non-Python data files recorded by wrapping ``open``).

Blocks are identified by mapping the executing *code object* directly via
``co_qualname`` (precise, immune to code motion, no coverage-line mapping),
and we treat the module skeleton explicitly as the import-time dependency
of every importer.

Two cross-cutting analyses ride along on every parse: *dynamic markers*
(reflective constructs such as computed ``getattr`` or ``__getattr__``
definitions, which selection uses to widen marker-bearing files back to
file level) and the *cosmetic normalization* (on by default: docstrings
stripped before hashing, so prose-only edits do not invalidate blocks; comments
are invisible to the AST anyway).  A per-file docstring hash plus a
"reads docstrings" flag back it with a safety net for code that treats
``__doc__`` as data.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

MODULE_BLOCK = "<module>"
WILDCARD = "*"

# A dependency: (repo-relative file, block id, content hash).
Block = tuple[str, str, str]

# Pseudo-dependency emitted when a *reflective stdlib entry point* starts
# executing in a test (``importlib.import_module``, ``runpy.run_module``,
# ``pkgutil.walk_packages``, ...).  It never reaches a shard: the recorder
# turns it into the test's "dynamic" mark and strips it.
DYNAMIC_DEP: Block = ("<dynamic>", "runtime", "*")


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


def scan_tree(tree: ast.AST) -> tuple[tuple[str, ...], bool]:
    """One walk over *tree*: ``(dynamic markers, reads docstrings)``.

    Dynamic markers are reflective/dynamic constructs as ``name@Lline``
    strings.  Deliberately ignored: *statically resolvable* access --
    ``getattr(x, "lit")`` names its attribute, so it cannot hide an unseen
    edge; a computed second argument (anything but a string constant) is
    flagged.  ``__getattr__`` / ``__getattribute__`` definitions are flagged
    wholesale: every attribute read on the object may dispatch to code no
    recording saw execute.

    "Reads docstrings" is true for a *load* of ``__doc__``/``getdoc``/
    ``cleandoc``/``help`` or an import of ``doctest``/``pydoc`` (see
    ``_DOC_ATTRS``); *assigning* ``fn.__doc__ = ...`` is not reading.
    """
    markers: set[str] = set()
    reads_docs = False
    Load = ast.Load
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
        elif not reads_docs:
            if isinstance(node, ast.Attribute):
                reads_docs = node.attr in _DOC_ATTRS and isinstance(node.ctx, Load)
            elif isinstance(node, ast.Name):
                reads_docs = (
                    node.id in _DOC_ATTRS or node.id == "help"
                ) and isinstance(node.ctx, Load)
            elif isinstance(node, ast.Import):
                reads_docs = any(
                    a.name.split(".")[0] in _DOC_MODULES for a in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                reads_docs = (node.module or "").split(".")[0] in _DOC_MODULES or any(
                    a.name in _DOC_ATTRS for a in node.names
                )
    return tuple(sorted(markers)), reads_docs


def find_dynamic_markers(tree: ast.AST) -> tuple[str, ...]:
    """Reflective/dynamic constructs in *tree* (see :func:`scan_tree`)."""
    return scan_tree(tree)[0]


# Docstring readers: code that treats a docstring as *data* (help text,
# doctest, registries built from ``__doc__``).  With the cosmetic filter on,
# a docstring-only edit is invisible to the block hashes; tests that executed
# a file containing one of these references are re-run anyway.
_DOC_ATTRS = frozenset({"__doc__", "getdoc", "cleandoc", "getcomments"})
_DOC_MODULES = frozenset({"doctest", "pydoc"})


def find_doc_readers(tree: ast.AST) -> bool:
    """Whether *tree* reads docstrings as data (see :func:`scan_tree`)."""
    return scan_tree(tree)[1]


def _scopes(tree: ast.AST):
    """Every Module/ClassDef/FunctionDef node, found by walking *statements*
    only (docstring owners cannot hide inside expressions; skipping them is
    several times cheaper than a full ``ast.walk``)."""
    scope_types = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, scope_types):
            yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)):
                stack.append(child)


def block_docstring_hashes(tree: ast.AST) -> dict[int, str]:
    """``id(node) -> hash of the docstrings that node's block hash covers``.

    Computed on the *unstripped* tree.  A function block covers every
    docstring in its subtree (nested definitions included); a class or module
    block is a skeleton -- function bodies are stubbed out -- so it covers
    only the class/module docstrings outside function bodies.  The cosmetic
    filter hides exactly these texts from the block hashes, so recording them
    lets selection tell "this block's docstring changed" from "this block's
    code changed".
    """
    out: dict[int, str] = {}
    scope_types = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    func_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    def walk(node: ast.AST) -> tuple[list, list]:
        """(docstrings in the full subtree, docstrings of the skeleton)."""
        full: list = []
        skel: list = []
        if isinstance(node, scope_types):
            doc = ast.get_docstring(node, clean=False)
            full.append(doc)
            if not isinstance(node, func_types):
                skel.append(doc)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)):
                child_full, child_skel = walk(child)
                full += child_full
                if not isinstance(child, func_types):
                    skel += child_skel
        if isinstance(node, func_types):
            out[id(node)] = _hash(repr(full))
        elif isinstance(node, scope_types):
            out[id(node)] = _hash(repr(skel))
        return full, skel

    walk(tree)
    return out


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstrings from every module/class/function in *tree* (in place).

    The cosmetic edit filter (on by default): a docstring is prose, not
    behavior -- for code that never reads ``__doc__`` (files that do are
    flagged by :func:`scan_tree` and keep their docstring edits selecting).
    Only true docstrings (a bare string as the first body statement, the
    ``ast.get_docstring`` definition) are removed; string values elsewhere
    are behavior and stay.  Comments and whitespace are already invisible to
    ``ast.dump``, so this is the entire textual surface the hash needs to
    ignore.  Annotations are deliberately *kept*: typing changes are
    observable.
    """
    for node in _scopes(tree):
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            # A function whose entire body is its docstring still needs a
            # statement to be valid Python (hashes only, never run).
            node.body = body[1:] or [ast.Pass()]
    return tree


_PASS_BODY: list[ast.stmt] = [ast.Pass()]


def _skeleton_dump(node: ast.AST) -> str:
    """``ast.dump`` of *node* with every function body replaced by ``pass``.

    The class / module skeleton: what executes at import time (class-level
    statements, decorators, signatures) without the method bodies.  Computed
    **in place** -- swap the bodies, dump, restore -- instead of on a
    ``deepcopy``: copying every class (methods and all) was 60% of the whole
    analysis cost on a large codebase (5s of 11s for 295 sqlalchemy files).
    The traversal does not descend into function bodies (they are being
    replaced anyway), so it only touches the class-level statements.  The
    result is byte-identical to the copy-based version, so recorded shards
    stay valid.
    """
    stash: list[tuple[ast.AST, list[ast.stmt]]] = []
    stack = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stash.append((child, child.body))
                child.body = _PASS_BODY
            elif isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)):
                stack.append(child)
    try:
        return ast.dump(node, include_attributes=False)
    finally:
        for fn, body in stash:
            fn.body = body


class FileBlocks:
    """Block table of one parsed Python file.

    ``blocks`` maps each qualified name to its content hash; ``MODULE_BLOCK``
    maps to the skeleton hash.  ``spans`` is the list of
    ``(first line incl. decorators, last line, qualified name)`` used as the
    fallback for code objects without a resolvable qualname.  ``resolve``
    maps a code object to its block.  ``markers`` lists the file's dynamic
    constructs (see :func:`find_dynamic_markers`), empty for ordinary code.
    """

    __slots__ = ("bdocs", "blocks", "markers", "reads_docs", "spans")

    def __init__(
        self,
        blocks: dict[str, str],
        spans: list[tuple[int, int, str]],
        markers: tuple[str, ...] = (),
        bdocs: dict[str, str] | None = None,
        reads_docs: bool = False,
    ) -> None:
        self.blocks = blocks
        self.spans = spans
        self.markers = markers
        # Per-block docstring hash (what the cosmetic filter hides from the
        # block hash) and whether the file reads docstrings as data: the
        # safety net of the cosmetic filter.
        self.bdocs = bdocs or {}
        self.reads_docs = reads_docs

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
    (comments are invisible to ``ast.dump`` anyway).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    markers, reads_docs = scan_tree(tree)
    node_docs = block_docstring_hashes(tree)
    if cosmetic:
        _strip_docstrings(tree)

    blocks: dict[str, str] = {}
    spans: list[tuple[int, int, str]] = []
    # (qname, first line, last line, hash, docstring hash)
    defs: list[tuple[str, int, int, str, str]] = []

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
                content = _hash(_skeleton_dump(child))
            else:
                content = _hash(ast.dump(child, include_attributes=False))
            # co_firstlineno points at the first decorator when present.
            start = (
                child.decorator_list[0].lineno
                if child.decorator_list
                else child.lineno
            )
            defs.append(
                (qname, start, child.end_lineno, content, node_docs[id(child)])
            )
            visit(
                child,
                qname,
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)),
            )

    visit(tree, "", False)
    # A qualified name defined more than once with *different* content
    # (``@overload`` stubs, property getter/setter pairs, conditional
    # definitions) is ambiguous: ``co_qualname`` cannot tell the definitions
    # apart.  Each gets its own id, ``qname~<hash>``, resolved by source line.
    # An edit then only invalidates the definition that changed (and an
    # ``@overload`` stub, which never executes, invalidates nothing).  The old
    # behaviour -- a file-level wildcard -- made *any* edit to such a file,
    # even a comment, stale for every test that touched one of them: 314
    # blocks over 91 files in sqlalchemy, including ``Session.execute``.
    distinct: dict[str, set[str]] = {}
    for qname, _, _, content, _ in defs:
        distinct.setdefault(qname, set()).add(content)
    bdocs: dict[str, str] = {}
    for qname, start, end, content, doc in defs:
        block_id = qname if len(distinct[qname]) == 1 else f"{qname}~{content}"
        blocks[block_id] = content
        # Identical duplicates share a block id: fold their docstrings.
        bdocs[block_id] = _hash(bdocs[block_id] + doc) if block_id in bdocs else doc
        spans.append((start, end, block_id))
    blocks[MODULE_BLOCK] = _hash(_skeleton_dump(tree))
    bdocs[MODULE_BLOCK] = node_docs[id(tree)]
    return FileBlocks(blocks, spans, markers, bdocs, reads_docs)


# Bump when anything that feeds a block/docstring hash changes (the algorithm
# in parse_blocks / scan_tree / block_docstring_hashes): a stored parse result
# from another version must not be trusted.
PARSE_CACHE_VERSION = 1


def _content_hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _file_to_json(fb: FileBlocks) -> dict:
    return {
        "b": {k: [v, fb.bdocs.get(k, "")] for k, v in fb.blocks.items()},
        "s": [list(span) for span in fb.spans],
        "m": list(fb.markers),
        "r": int(fb.reads_docs),
    }


def _file_from_json(d: dict) -> FileBlocks:
    blocks = {k: v[0] for k, v in d["b"].items()}
    bdocs = {k: v[1] for k, v in d["b"].items() if v[1]}
    spans = [(int(a), int(b), c) for a, b, c in d["s"]]
    return FileBlocks(blocks, spans, tuple(d["m"]), bdocs, bool(d["r"]))


class ParseCache:
    """Parse results keyed by file *content hash*.

    Analyzing a file (parse + block hashes + marker scan) costs ~10ms; reading
    it and hashing the bytes costs ~0.1ms.  ``git diff`` already tells us which
    files changed for *staleness*, but resolving an executed code object to a
    block needs the block table of every file whose code runs -- changed or
    not -- so an incremental run used to re-parse hundreds of untouched files.
    Entries are verified against the current bytes on every use, so a stale
    entry is merely a miss: the cache can never serve a wrong table, whatever
    the working tree looked like when it was written (dirty, reverted, switched
    branch).  It lives in the shard (``parse.json``) and travels with it.
    """

    def __init__(self, loader=None, *, cosmetic: bool = False) -> None:
        self._loader = loader
        self._cosmetic = cosmetic
        self._loaded = loader is None
        self._header_ok = True
        self._had_blob = False
        self.entries: dict[str, dict] = {}
        self.fresh: dict[str, dict] = {}
        # rel -> hash of the bytes whose analysis this run actually used
        # (a hit or a fresh parse): what the block hashes were derived from.
        self.used: dict[str, str] = {}

    def _header(self) -> dict:
        return {
            "v": PARSE_CACHE_VERSION,
            "py": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "cosmetic": self._cosmetic,
        }

    def preload(self) -> None:
        """Read the stored cache now (call outside the monitored region: the
        loader may spawn a git process)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = self._loader()
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            return
        self._had_blob = True
        header = self._header()
        if all(raw.get(k) == v for k, v in header.items()) and isinstance(
            raw.get("files"), dict
        ):
            self.entries = raw["files"]
        else:
            self._header_ok = False

    def lookup(self, rel: str, digest: str) -> FileBlocks | None:
        self.preload()
        entry = self.fresh.get(rel) or self.entries.get(rel)
        if entry is None or entry.get("h") != digest:
            return None
        try:
            fb = _file_from_json(entry)
        except Exception:  # corrupt / foreign entry: a miss, never a crash
            return None
        self.used[rel] = digest
        return fb

    def store(self, rel: str, digest: str, fb: FileBlocks) -> None:
        entry = _file_to_json(fb)
        entry["h"] = digest
        self.fresh[rel] = entry
        self.used[rel] = digest

    def dump(self, keep: set[str]) -> dict | None:
        """The cache to persist (entries for *keep* only), or None when the
        stored blob is already exactly right and can be carried over as is."""
        self.preload()
        files = {rel: e for rel, e in self.entries.items() if rel in keep}
        files.update({rel: e for rel, e in self.fresh.items() if rel in keep})
        unchanged = (
            self._header_ok
            and (self._had_blob or not files)
            and set(files) == set(self.entries)
            and not any(rel in keep for rel in self.fresh)
        )
        if unchanged:
            return None
        return {**self._header(), "files": files}


class BlockCache:
    """Lazily parses repo files into :class:`FileBlocks` (one parse per file)."""

    def __init__(
        self,
        root: Path,
        *,
        cosmetic: bool = False,
        parse_cache: ParseCache | None = None,
    ):
        self.root = root
        self.cosmetic = cosmetic
        self.parse_cache = parse_cache
        self.hits = 0
        self.misses = 0
        self._cache: dict[str, FileBlocks | None] = {}

    def file_blocks(self, rel: str) -> FileBlocks | None:
        if rel in self._cache:
            return self._cache[rel]
        try:
            data = (self.root / rel).read_bytes()
        except OSError:
            self._cache[rel] = None
            return None
        digest = None
        if self.parse_cache is not None:
            digest = _content_hash(data)
            fb = self.parse_cache.lookup(rel, digest)
            if fb is not None:
                self.hits += 1
                self._cache[rel] = fb
                return fb
        source = data.decode("utf-8", errors="replace")
        if b"\r" in data:  # same universal-newline translation as read_text
            source = source.replace("\r\n", "\n").replace("\r", "\n")
        fb = parse_blocks(source, cosmetic=self.cosmetic)
        self.misses += 1
        if self.parse_cache is not None and fb is not None:
            self.parse_cache.store(rel, digest, fb)
        self._cache[rel] = fb
        return fb

    def module_dep(self, rel: str) -> Block:
        """Import-time dependency on a module: its skeleton."""
        fb = self.file_blocks(rel)
        if fb is None:
            return (rel, WILDCARD, WILDCARD)
        return (rel, MODULE_BLOCK, fb.blocks[MODULE_BLOCK])
