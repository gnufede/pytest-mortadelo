"""Read and write the few git objects pytest-less needs, without spawning git.

Every ``git`` call is a process spawn: a few milliseconds on a quiet machine,
a quarter of a second where endpoint scanning slows ``exec``.  A run used to
spend 4-11 of them on things that are plain file reads: finding the toplevel,
``HEAD``, a ref, and three blobs; and six more writing the shard back (167ms
for 12 MB, against 39ms in-process).  This module does exactly that and
nothing else:

* find the repository (plain ``.git`` directory, linked worktree, submodule);
* read refs (loose files, then ``packed-refs``);
* read **loose** objects (``zlib`` + the ``type size\\0`` header);
* write loose objects, a tree, a parentless commit, and move a ref with git's
  own ``.lock`` protocol.

What it deliberately does **not** do: read packed objects, deltas, alternates,
SHA-256 repositories, reftable, diff, ignore rules, or talk to a remote.
Whenever it meets anything it does not fully understand it raises
:class:`Unsupported` (or returns ``None`` from :meth:`Repo.discover`), and the
caller falls back to the corresponding ``git`` command.  Correctness never
depends on this module being applicable -- only speed does.

``PYTEST_LESS_GIT_BACKEND=subprocess`` disables it (cold-run measurements,
triage, and CI runs of the fallback path).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import zlib
from pathlib import Path

# Variables that move or reshape the repository: anything we do not model.
_UNSUPPORTED_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_NAMESPACE",
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_LOCK_ATTEMPTS = 10
_LOCK_WAIT = 0.02

# Fixed identity: the commit is synthetic bookkeeping, not authorship, and
# reading ``user.name`` would cost a spawn (or a config parse).
_IDENTITY = "pytest-less <less@invalid>"


class Unsupported(Exception):
    """This repository layout or object is outside what this module models."""


def backend_enabled() -> bool:
    return os.environ.get("PYTEST_LESS_GIT_BACKEND", "auto").strip().lower() != "subprocess"


class Repo:
    """A repository seen through its files.  Construct with :meth:`discover`."""

    def __init__(self, worktree: Path, gitdir: Path, commondir: Path) -> None:
        self.worktree = worktree
        self.gitdir = gitdir  # per-worktree files (HEAD)
        self.commondir = commondir  # objects/, refs/, packed-refs
        self._counter = 0

    # -- discovery ---------------------------------------------------------

    @classmethod
    def discover(cls, start: Path | str) -> Repo | None:
        """The repository containing *start*, or ``None`` if there is none or
        its layout is not one this module models (callers then use git)."""
        if any(name in os.environ for name in _UNSUPPORTED_ENV):
            return None
        try:
            here = Path(start).resolve()
            for directory in (here, *here.parents):
                dotgit = directory / ".git"
                if dotgit.is_dir():
                    return cls._checked(directory, dotgit)
                if dotgit.is_file():
                    line = dotgit.read_text(encoding="utf-8", errors="replace").strip()
                    if not line.startswith("gitdir:"):
                        return None
                    gitdir = (directory / line[len("gitdir:"):].strip()).resolve()
                    return cls._checked(directory, gitdir)
        except OSError:
            return None
        return None

    @classmethod
    def _checked(cls, worktree: Path, gitdir: Path) -> Repo | None:
        if not (gitdir / "HEAD").is_file():
            return None
        commondir = gitdir
        pointer = gitdir / "commondir"
        if pointer.is_file():  # linked worktree: shared objects and refs
            commondir = (gitdir / pointer.read_text().strip()).resolve()
        config = commondir / "config"
        text = config.read_text(encoding="utf-8", errors="replace").lower() if config.is_file() else ""
        # SHA-256 object ids, reftable refs, a relocated worktree: not modelled.
        if "sha256" in text or "refstorage" in text or "reftable" in text:
            return None
        if re.search(r"^\s*worktree\s*=", text, re.MULTILINE):
            return None
        if not (commondir / "objects").is_dir():
            return None
        return cls(worktree.resolve(), gitdir, commondir)

    # -- refs --------------------------------------------------------------

    def read_ref(self, name: str, _depth: int = 0) -> str | None:
        """The object id *name* points at (following symbolic refs), or None
        when the ref does not exist."""
        if _depth > 5:
            raise Unsupported("symbolic ref chain")
        home = self.gitdir if name == "HEAD" else self.commondir
        path = home / name
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return self._packed_ref(name)
        except IsADirectoryError:
            return None
        if text.startswith("ref:"):
            return self.read_ref(text[4:].strip(), _depth + 1)
        if _HEX40.match(text):
            return text
        raise Unsupported(f"unreadable ref {name!r}")

    def _packed_ref(self, name: str) -> str | None:
        try:
            lines = (self.commondir / "packed-refs").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return None
        for line in lines:
            if not line or line[0] in "#^":
                continue
            sha, _, ref = line.partition(" ")
            if ref == name and _HEX40.match(sha):
                return sha
        return None

    def head(self) -> str | None:
        """``HEAD``'s commit (None on an unborn branch)."""
        return self.read_ref("HEAD")

    # -- loose objects -----------------------------------------------------

    def _object_path(self, sha: str) -> Path:
        if not _HEX40.match(sha):
            raise Unsupported(f"bad object id {sha!r}")
        return self.commondir / "objects" / sha[:2] / sha[2:]

    def read_object(self, sha: str) -> tuple[str, bytes]:
        """``(type, body)`` of a *loose* object.  Raises :class:`Unsupported`
        when it is not loose (packed, in an alternate, absent)."""
        try:
            raw = zlib.decompress(self._object_path(sha).read_bytes())
        except FileNotFoundError:
            raise Unsupported(f"object {sha} is not loose") from None
        nul = raw.index(b"\0")
        kind, _, size = raw[:nul].partition(b" ")
        body = raw[nul + 1:]
        if int(size) != len(body):
            raise Unsupported(f"corrupt object {sha}")
        return kind.decode(), body

    def _tree_of(self, commit_sha: str) -> str:
        kind, body = self.read_object(commit_sha)
        if kind != "commit" or not body.startswith(b"tree "):
            raise Unsupported(f"{commit_sha} is not a commit")
        return body[5:45].decode()

    @staticmethod
    def _tree_entry(body: bytes, name: str) -> str | None:
        wanted = name.encode()
        i = 0
        while i < len(body):
            space = body.index(b" ", i)
            nul = body.index(b"\0", space)
            if body[space + 1:nul] == wanted:
                return body[nul + 1:nul + 21].hex()
            i = nul + 21
        return None

    def blob_sha_in_ref(self, ref: str, name: str) -> str | None:
        """Object id of the top-level entry *name* of ``ref``'s tree (None if
        the ref or the entry does not exist)."""
        commit = self.read_ref(ref)
        if commit is None:
            return None
        kind, body = self.read_object(self._tree_of(commit))
        if kind != "tree":
            raise Unsupported("not a tree")
        return self._tree_entry(body, name)

    def blob_in_ref(self, ref: str, name: str) -> bytes | None:
        """Content of the top-level blob *name* in ``ref`` (None if absent)."""
        sha = self.blob_sha_in_ref(ref, name)
        if sha is None:
            return None
        kind, body = self.read_object(sha)
        if kind != "blob":
            raise Unsupported("not a blob")
        return body

    # -- writing -----------------------------------------------------------

    def write_object(self, kind: str, body: bytes) -> str:
        """Write a loose object (a no-op if it exists); return its id."""
        raw = f"{kind} {len(body)}".encode() + b"\0" + body
        sha = hashlib.sha1(raw).hexdigest()
        path = self._object_path(sha)
        if path.exists():
            return sha
        path.parent.mkdir(exist_ok=True)
        self._counter += 1
        tmp = path.parent / f"tmp_obj_{os.getpid()}_{self._counter}"
        try:
            # Level 1 is git's own default for loose objects.
            tmp.write_bytes(zlib.compress(raw, 1))
            os.chmod(tmp, 0o444)
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        return sha

    def write_shard(
        self,
        ref: str,
        files: dict[str, bytes],
        message: str,
        reuse: dict[str, str] | None = None,
    ) -> str:
        """Store *files* (plus already-present blobs *reuse*: name -> id) as a
        parentless commit and point *ref* at it.  Returns the commit id."""
        entries = {name: self.write_object("blob", data) for name, data in files.items()}
        entries.update(reuse or {})
        tree = b"".join(
            f"100644 {name}\0".encode() + bytes.fromhex(sha)
            for name, sha in sorted(entries.items())
        )
        tree_sha = self.write_object("tree", tree)
        now = int(time.time())
        who = f"{_IDENTITY} {now} +0000"
        commit = f"tree {tree_sha}\nauthor {who}\ncommitter {who}\n\n{message}\n".encode()
        commit_sha = self.write_object("commit", commit)
        self._update_ref(ref, commit_sha)
        return commit_sha

    def _update_ref(self, name: str, sha: str) -> None:
        """Move *name* with git's lock protocol (``<ref>.lock``, then rename).

        Loose refs shadow ``packed-refs``, so no rewrite of the latter is
        needed.  Everything git does for a plain ``update-ref`` on such a ref
        -- except firing the ``reference-transaction`` hook and a reflog
        entry (``refs/less/*`` has neither by default).
        """
        path = self.commondir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_name(path.name + ".lock")
        for attempt in range(_LOCK_ATTEMPTS):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                break
            except FileExistsError:
                if attempt == _LOCK_ATTEMPTS - 1:
                    raise Unsupported(f"{lock} is held") from None
                time.sleep(_LOCK_WAIT)
        try:
            os.write(fd, f"{sha}\n".encode())
        finally:
            os.close(fd)
        os.replace(lock, path)


def toplevel(start: Path | str) -> Path | None:
    """The repository's working-tree root, or None (no repo / unsupported:
    callers then ask ``git rev-parse --show-toplevel``)."""
    repo = Repo.discover(start) if backend_enabled() else None
    return repo.worktree if repo is not None else None
