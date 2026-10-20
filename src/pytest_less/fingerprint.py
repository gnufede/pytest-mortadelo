"""Change detection without asking git "what differs from a base commit?".

``git diff <base>`` plus ``git ls-files --others`` answer a question we do not
actually have.  They report every file that differs from a commit, and the
second one scans the whole checkout (790 ms on a 10k-file repository with
46k untracked files).  What selection needs is narrower: *do the files this
shard depends on still look like they did when it was recorded?*

At record time we store a **snapshot**:

``files``
    ``{rel: [size, mtime_ns, content_hash]}`` for every dependency file and
    every ``.py`` file in a *watched* directory.
``dirs``
    ``{reldir: [subdirectory names]}`` for the watched directories: every
    directory that holds a dependency file, and all of its ancestors up to the
    repository root.
``names``
    Only in widened mode (see below): ``{reldir: [file names]}``, so that *any*
    new non-hidden file in a watched directory -- not just a ``.py`` -- counts
    as new.
``at``
    When the snapshot was taken (``time.time_ns()``), for the racy-clean rule.

At run time :func:`detect_changes` needs no git at all in the common case:

* ``stat`` every recorded file.  Same size, same mtime, and an mtime older than
  the snapshot by more than :data:`RACY_NS` -> unchanged.  Anything else is
  settled by hashing the bytes (a ``touch`` or a ``git checkout`` that rewrites
  identical content is *not* a change; the same applies to a shard recorded on
  another machine, where every mtime differs).
* list the watched directories.  A ``.py`` file that is not in ``files``, or a
  subdirectory that is not in ``dirs``, is *new*.

Only when something new appears is git asked, once, and only about those
paths (``ls-files --cached --others --exclude-standard -- <paths>``): ignore
rules stay git's job, but a repository with a huge untracked tree no longer
pays for listing it on every run.  Hidden directories, ``__pycache__``,
``*.egg-info`` and the usual environment directories are never reported as new.

When a recorded test spawns a process, the snapshot is **widened to every
tracked file** (``"all": true``): a child may read any file the parent never
touched, so "nothing changed" is only provable if everything is watched.  That
is what lets a rerun at the same commit skip such tests instead of always
re-running them (see :meth:`GitStore.is_pristine`).  Files above
:data:`EXTRA_HASH_LIMIT` are fingerprinted by stat alone (a differing mtime
counts as a change: never read, never hashed), and a repository with more than
:data:`TRACKED_LIMIT` tracked files is not widened at all.

Hashing uses the same function as the parse cache, so the digest of a
dependency file in the snapshot is the digest the block hashes were computed
from.  If the file changed between being parsed and being snapshotted, its
stored mtime is ``-1``: the next run never trusts the stat shortcut for it and
the hash mismatch reports the change.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath

from .blocks import _content_hash
from .util import _EXCLUDED_SEGMENTS

SNAPSHOT_VERSION = 1

# Widening to every tracked file (shards with spawning tests) is bounded: a
# big repository keeps the old "spawners always re-run" behavior rather than
# stat'ing a six-figure file list on every run.
TRACKED_LIMIT = 30_000
# Beyond this size a widened (non-dependency) file is fingerprinted by stat only.
EXTRA_HASH_LIMIT = 1_000_000

# A file modified within this window before the snapshot may have been written
# again within the same timestamp tick (coarse filesystems: 1-2 s): never
# trust "same size and mtime" for it.  Git's "racily clean" rule.
RACY_NS = 2_000_000_000


def _skip_dir(name: str) -> bool:
    """Directories that are never reported as new project code."""
    return (
        name.startswith(".")
        or name == "__pycache__"
        or name in _EXCLUDED_SEGMENTS
        or name.endswith(".egg-info")
    )


def _join(directory: str, name: str) -> str:
    return f"{directory}/{name}" if directory else name


def _fingerprint(
    path: Path,
    digest: str | None,
    previous: list | None = None,
    trusted_before: int = 0,
    max_bytes: int | None = None,
) -> list | None:
    """``[size, mtime_ns, hash]`` for *path*, or None if it cannot be read.

    *digest* is the hash of the content that was actually analyzed (parse
    cache): when the bytes on disk no longer match it the file changed during
    the run, and the entry is poisoned (mtime ``-1``) so the next run re-hashes.

    *previous* is the same file's entry in the snapshot being replaced.  If
    its size and mtime are unchanged and the mtime is comfortably older than
    that snapshot (*trusted_before*), the bytes are the ones it hashed: reuse
    the hash instead of reading the file again -- the very shortcut
    :func:`detect_changes` takes -- so rewriting a shard costs stats, not reads
    (507 ms -> a few ms for 3,250 files).
    """
    try:
        st = path.stat()
        if max_bytes is not None and st.st_size > max_bytes:
            return [st.st_size, st.st_mtime_ns, ""]  # stat-only: never read
        if (
            previous is not None
            and previous[1] >= 0
            and previous[1] == st.st_mtime_ns
            and previous[1] < trusted_before
            and previous[0] == st.st_size
            and (digest is None or digest == previous[2])
        ):
            return [st.st_size, st.st_mtime_ns, previous[2]]
        current = _content_hash(path.read_bytes())
    except OSError:
        return None
    if digest is not None and digest != current:
        return [st.st_size, -1, digest]
    return [st.st_size, st.st_mtime_ns, current]


def build_snapshot(
    root: Path,
    dep_files: Iterable[str],
    digests: Mapping[str, str] | None = None,
    previous: Mapping | None = None,
    extra_files: Iterable[str] | None = None,
) -> dict:
    """The snapshot to store next to a freshly recorded shard.

    *previous* is the snapshot being replaced (if any): its hashes are reused
    for files that provably did not change since (see :func:`_fingerprint`).

    *extra_files* (every tracked file, when some recorded test spawns a
    process) are fingerprinted too and mark the snapshot ``"all": true``.
    """
    at = time.time_ns()
    digests = digests or {}
    old_files: Mapping = {}
    trusted_before = 0
    if (
        previous
        and previous.get("v") == SNAPSHOT_VERSION
        and isinstance(previous.get("files"), dict)
        and isinstance(previous.get("at"), int)
    ):
        old_files = previous["files"]
        trusted_before = previous["at"] - RACY_NS
    deps = {f for f in dep_files if isinstance(f, str) and f}

    watched: set[str] = {""}  # the root is always watched (setup.py, conftest.py)
    for rel in deps:
        for parent in PurePosixPath(rel).parents:
            watched.add("" if str(parent) == "." else str(parent))

    files: dict[str, list] = {}
    dirs: dict[str, list[str]] = {}
    names: dict[str, list[str]] = {}
    widened = extra_files is not None
    for directory in sorted(watched):
        base = root / directory if directory else root
        try:
            with os.scandir(base) as entries:
                subdirs: list[str] = []
                listed: list[str] = []
                for entry in entries:
                    name = entry.name
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_dir:
                        if not _skip_dir(name):
                            subdirs.append(name)
                        continue
                    if widened and not name.startswith("."):
                        listed.append(name)
                    if name.endswith(".py"):
                        rel = _join(directory, name)
                        record = _fingerprint(
                            root / rel, digests.get(rel), old_files.get(rel), trusted_before
                        )
                        if record is not None:
                            files[rel] = record
        except OSError:
            continue
        dirs[directory] = sorted(subdirs)
        if widened:
            names[directory] = sorted(listed)

    for rel in deps:
        if rel not in files:
            record = _fingerprint(
                root / rel, digests.get(rel), old_files.get(rel), trusted_before
            )
            if record is not None:
                files[rel] = record
    if extra_files is not None:
        for rel in extra_files:
            if rel in files or not isinstance(rel, str) or not rel:
                continue
            record = _fingerprint(
                root / rel, None, old_files.get(rel), trusted_before, EXTRA_HASH_LIMIT
            )
            if record is not None:
                files[rel] = record
    snapshot = {
        "v": SNAPSHOT_VERSION,
        "at": at,
        "files": files,
        "dirs": dirs,
        "all": widened,
    }
    if widened:
        snapshot["names"] = names
    return snapshot


def detect_changes(
    root: Path,
    snapshot: Mapping,
    confirm: Callable[[list[str]], set[str] | None] | None = None,
) -> set[str] | None:
    """Repo-relative files that differ from *snapshot*, or ``None`` when that
    cannot be decided (unknown snapshot version, an unreadable directory, git
    unable to confirm new paths): the caller then falls back to ``git diff``.

    *confirm* receives the new ``.py`` files and new directories (directories
    with a trailing ``/``) and returns the project files git knows under them
    (tracked, or untracked and not ignored).  It is called at most once, and
    only when something new showed up.
    """
    if snapshot.get("v") != SNAPSHOT_VERSION:
        return None
    files = snapshot.get("files")
    dirs = snapshot.get("dirs")
    at = snapshot.get("at")
    if not isinstance(files, dict) or not isinstance(dirs, dict) or not isinstance(at, int):
        return None

    changed: set[str] = set()
    trusted_before = at - RACY_NS
    for rel, record in files.items():
        try:
            size, mtime, digest = record
            st = os.stat(root / rel)
        except FileNotFoundError:
            changed.add(rel)
            continue
        except NotADirectoryError:
            changed.add(rel)
            continue
        except (OSError, ValueError, TypeError):
            return None
        if st.st_size != size:
            changed.add(rel)
            continue
        if mtime >= 0 and st.st_mtime_ns == mtime and mtime < trusted_before:
            continue  # same size, same mtime, comfortably older than the snapshot
        if not digest:
            changed.add(rel)  # stat-only entry (large file): cannot be verified
            continue
        try:
            now = _content_hash((root / rel).read_bytes())
        except OSError:
            changed.add(rel)
            continue
        if now != digest:
            changed.add(rel)

    names = snapshot.get("names") if snapshot.get("all") else None
    suspects: list[str] = []
    for directory, known in dirs.items():
        known_dirs = set(known) if isinstance(known, list) else set()
        known_names = (
            set(names.get(directory, ())) if isinstance(names, dict) else None
        )
        try:
            with os.scandir(root / directory if directory else root) as entries:
                for entry in entries:
                    name = entry.name
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_dir:
                        child = _join(directory, name)
                        if name in known_dirs or child in dirs or _skip_dir(name):
                            continue
                        suspects.append(child + "/")
                    elif known_names is not None:
                        # widened: any new non-hidden file counts, not just .py
                        if name not in known_names and not name.startswith("."):
                            suspects.append(_join(directory, name))
                    elif name.endswith(".py") and _join(directory, name) not in files:
                        suspects.append(_join(directory, name))
        except FileNotFoundError:
            continue  # the directory is gone: its recorded files were reported above
        except OSError:
            return None

    if suspects:
        if confirm is None:
            return None
        project = confirm(suspects)
        if project is None:
            return None
        for rel in project:
            parts = PurePosixPath(rel).parts
            if (names is not None or rel.endswith(".py")) and not any(
                _skip_dir(p) for p in parts[:-1]
            ):
                changed.add(rel)
    return changed
