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
newer than 3.15 -- unparseable files, and non-Python data files recorded by
wrapping ``open``).
"""

from __future__ import annotations

import hashlib
import json
import os
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


# Bump when anything that feeds a block hash or the stored table changes (the
# algorithm in codeblocks, the entry layout): a stored parse result from another
# version must not be trusted.
PARSE_CACHE_VERSION = 4


def _content_hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _file_to_json(fb: FileBlocks) -> dict:
    return {"b": fb.blocks, "s": [list(span) for span in fb.spans]}


def _file_from_json(d: dict) -> FileBlocks:
    return FileBlocks(
        dict(d["b"]), [(int(start), block_id) for start, block_id in d["s"]]
    )


# Per repository, per Python minor -- shared by every shard.
LOCAL_PARSE_LIMIT = 8_000


class LocalParseStore:
    """Parse results shared by **every shard** of a repository.

    The shard's ``parse.json`` only knows what that one command recorded, so a
    different command (``pytest tests/unit`` after ``pytest tests``), a cleaned
    shard, or another branch's first record parsed the same unchanged files
    again -- 2.5 s of a 15 s record on a 220-file set.  This is one tiny file
    per *content hash* under the git directory (``<git-dir>/pytest-less/``),
    so a lookup is one small read (no big blob to load, nothing to rewrite)
    and two paths holding the same bytes share an entry.  Like every other
    cache here it is only ever trusted for the exact bytes it was built from,
    and it is bounded (:data:`LOCAL_PARSE_LIMIT` entries, oldest evicted).
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @classmethod
    def at(cls, base: Path) -> LocalParseStore:
        py = f"{sys.version_info[0]}.{sys.version_info[1]}"
        return cls(Path(base) / "pytest-less" / f"parse-v{PARSE_CACHE_VERSION}-py{py}")

    def _path(self, digest: str) -> Path:
        return self.directory / digest[:2] / f"{digest}.json"

    def get(self, digest: str) -> dict | None:
        try:
            entry = json.loads(self._path(digest).read_bytes())
        except (OSError, ValueError):
            return None
        return entry if isinstance(entry, dict) else None

    def put(self, digest: str, entry: dict) -> bool:
        """Store *entry* atomically; True if a new file was written."""
        path = self._path(digest)
        if path.exists():
            return False
        tmp = path.with_name(f".tmp-{os.getpid()}-{digest[:8]}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(json.dumps(entry).encode())
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return False
        return True

    def evict(self, limit: int = LOCAL_PARSE_LIMIT) -> int:
        """Drop the oldest entries beyond *limit*; returns how many."""
        found: list[tuple[float, Path]] = []
        try:
            for bucket in os.scandir(self.directory):
                if not bucket.is_dir():
                    continue
                for item in os.scandir(bucket.path):
                    if item.name.endswith(".json"):
                        found.append((item.stat().st_mtime, Path(item.path)))
        except OSError:
            return 0
        if len(found) <= limit:
            return 0
        found.sort()
        removed = 0
        for _, path in found[: len(found) - int(limit * 0.9)]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


class ParseCache:
    """Parse results keyed by file *content hash*.

    Analyzing a file (compile + block hashes) costs ~10ms; reading
    it and hashing the bytes costs ~0.1ms.  ``git diff`` already tells us which
    files changed for *staleness*, but resolving an executed code object to a
    block needs the block table of every file whose code runs -- changed or
    not -- so an incremental run used to re-parse hundreds of untouched files.
    Entries are verified against the current bytes on every use, so a stale
    entry is merely a miss: the cache can never serve a wrong table, whatever
    the working tree looked like when it was written (dirty, reverted, switched
    branch).  It lives in the shard (``parse.json``) and travels with it.
    """

    def __init__(
        self,
        loader=None,
        *,
        local: LocalParseStore | None = None,
    ) -> None:
        self._loader = loader
        self._local = local
        self._from_local: set[str] = set()
        self.local_hits = 0
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
            return self._lookup_local(rel, digest)
        try:
            fb = _file_from_json(entry)
        except Exception:  # corrupt / foreign entry: a miss, never a crash
            return None
        self.used[rel] = digest
        return fb

    def _lookup_local(self, rel: str, digest: str) -> FileBlocks | None:
        """The repository-wide store: another shard may have parsed these
        exact bytes.  A hit is adopted into this shard's own blob."""
        if self._local is None:
            return None
        entry = self._local.get(digest)
        if entry is None:
            return None
        try:
            fb = _file_from_json(entry)
        except Exception:
            return None
        self.fresh[rel] = {**entry, "h": digest}
        self.used[rel] = digest
        self._from_local.add(rel)
        self.local_hits += 1
        return fb

    def flush_local(self) -> int:
        """Publish what this run parsed to the repository-wide store."""
        if self._local is None:
            return 0
        written = 0
        for rel, entry in self.fresh.items():
            if rel in self._from_local:
                continue
            body = {k: v for k, v in entry.items() if k != "h"}
            written += self._local.put(entry["h"], body)
        if written >= 25:  # a cold run: keep the store bounded
            self._local.evict()
        return written

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
    """Lazily analyzes repo files into :class:`FileBlocks` (once per file)."""

    def __init__(
        self,
        root: Path,
        *,
        parse_cache: ParseCache | None = None,
    ):
        self.root = root
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
        from .codeblocks import Unsupported, compile_blocks

        try:
            fb = compile_blocks(data, filename=rel)
        except Unsupported:
            fb = None  # a construct it does not model: a file-level dependency
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
