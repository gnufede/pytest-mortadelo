"""Code blocks: identity, hashing, and the caches that make it cheap.

A *block* is a function, a class body, or the module itself.  Blocks are
identified by their qualified name -- ``code.co_qualname`` exactly, Python 3.12+
-- and hashed from the **compiled bytecode** of the file (see
:mod:`codeblocks`): whitespace-, comment- and position-insensitive, so
reformatting or moving a function within a file does not invalidate anything.

The special block ``MODULE_BLOCK`` is the module *skeleton*: the module with
every function/method body replaced by its signature.  The skeleton captures
exactly what executes at import time -- module-level statements, decorators,
signatures, class-level attributes, *in source order* (order is part of
import-time behavior) -- while ignoring function bodies, which do not execute
at import.  Changing a function body therefore does not invalidate dependents
that merely import the module; changing anything that runs at import does.

Dependencies are ``Block`` triples ``(repo-relative file, block id, content
hash)``.  A ``WILDCARD`` block id means "any change to this file re-runs" (the
file-level fallback for code we cannot map to a block: files the hasher does
not model -- PEP 695 generics, ``type`` statements, an interpreter other than
newer than 3.15 -- and unparseable or deleted files).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MODULE_BLOCK = "<module>"
WILDCARD = "*"

# A dependency: (repo-relative file, block id, content hash).
Block = tuple[str, str, str]


class FileBlocks:
    """Block table of one Python file.

    ``blocks`` maps each block id to its content hash; ``MODULE_BLOCK`` maps to
    the skeleton hash.  ``spans`` lists ``(first line incl. decorators, block
    id)`` for every definition.  ``resolve`` maps a running code object to its
    block *exactly*: by ``(co_qualname, co_firstlineno)``, and code with no
    block of its own (lambdas, comprehensions) to its **lexical parent** -- the
    qualified name with the last component stripped -- instead of guessing from
    which block's source span contains the line.  A module-level lambda
    therefore maps to the module skeleton (which contains it) rather than to a
    file wildcard, and a class declared ``global`` inside a function is found
    under its real ``co_qualname``.

    """

    __slots__ = ("_by_name", "blocks", "spans")

    def __init__(self, blocks: dict[str, str], spans: list[tuple[int, str]]) -> None:
        self.blocks = blocks
        self.spans = spans
        # qualname -> [(first line, block id)]: redefinitions (``@overload``,
        # property setters) share a qualname and are told apart by line.
        by_name: dict[str, list[tuple[int, str]]] = {}
        for start, block_id in spans:
            by_name.setdefault(block_id.split("~")[0], []).append((start, block_id))
        self._by_name = by_name

    def resolve(self, qualname: str, firstlineno: int) -> tuple[str, str] | None:
        """Map a code object to ``(block id, hash)``."""
        name = qualname
        while True:
            options = self._by_name.get(name)
            if options is not None:
                block_id = options[0][1]
                if len(options) > 1:  # redefinitions: pick by source line
                    best = -1
                    for start, candidate in options:
                        if start == firstlineno:
                            block_id = candidate
                            break
                        if best < start <= firstlineno:
                            best, block_id = start, candidate
                return block_id, self.blocks[block_id]
            cut = name.rfind(".")
            if cut < 0:
                break
            name = name[:cut].removesuffix(".<locals>")
        return MODULE_BLOCK, self.blocks[MODULE_BLOCK]


def _content_hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()



class BlockCache:
    """Lazily analyzes repo files into :class:`FileBlocks` (once per file).

    Remembers the content hash of every file it actually read, so the
    snapshot can tell "the file changed while the run was recording" from
    "the file changed since the run" (see :func:`fingerprint.build_snapshot`).
    """

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, FileBlocks | None] = {}
        self._digests: dict[str, str] = {}

    @property
    def digests(self) -> dict[str, str]:
        """Content hash of the bytes each analyzed file was last read from."""
        return self._digests

    def file_blocks(self, rel: str) -> FileBlocks | None:
        """Analyze *rel* into a block table (once per file)."""
        if rel in self._cache:
            return self._cache[rel]
        try:
            data = (self.root / rel).read_bytes()
        except OSError:
            self._cache[rel] = None
            return None
        self._digests[rel] = _content_hash(data)
        from .codeblocks import Unsupported, compile_blocks

        try:
            fb = compile_blocks(data, filename=rel)
        except Unsupported:
            fb = None  # a construct it does not model: a file-level dep
        self._cache[rel] = fb
        return fb

    def module_dep(self, rel: str) -> Block:
        """Import-time dependency on a module: its skeleton."""
        fb = self.file_blocks(rel)
        if fb is None:
            return (rel, WILDCARD, WILDCARD)
        return (rel, MODULE_BLOCK, fb.blocks[MODULE_BLOCK])
