"""Dependency storage: plain JSON files under ``.pytest_less/``.

Three files, all replaced atomically on save:

    meta.json   small: the fast-path summary (dep_files, failed, base_commit)
    data.json   the dependency blob (blocks, tests, import_deps, always)
    files.json  the fingerprint snapshot (change detection)

No git refs, no blobs, no sync.  ``head()`` and the git fallbacks for
change detection are plain subprocess calls.  ``rm -rf .pytest_less``
resets the cache; copying the directory shares it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .fingerprint import detect_changes

_REPO_FILES = ("data.json", "meta.json", "files.json")


class Store:
    """Read/write the dependency data under ``<root>/.pytest_less/``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir = self.root / ".pytest_less"

    # -- repository state --------------------------------------------------

    def head(self) -> str | None:
        """Current HEAD commit sha, or None on an unborn branch."""
        try:
            return self._git("rev-parse", "HEAD").strip() or None
        except (subprocess.CalledProcessError, OSError):
            return None

    def project_files(self, paths: list[str]) -> set[str] | None:
        """Which of *paths* are project files (not ignored, not deleted).

        One ``git ls-files`` call about the candidate paths only; None
        when git cannot answer."""
        if not paths:
            return set()
        try:
            out = self._git("--literal-pathspecs", "ls-files", "--", *paths)
        except (subprocess.CalledProcessError, OSError):
            return None
        return set(out.splitlines())

    def tracked_files(self, limit: int) -> list[str] | None:
        """Every tracked file, or None if there are more than *limit*."""
        try:
            out = self._git("ls-files", "-z")
        except (subprocess.CalledProcessError, OSError):
            return None
        files = [p for p in out.split("\0") if p]
        return files if len(files) <= limit else None

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True, text=True, check=True,
        ).stdout

    # -- change detection --------------------------------------------------

    def changed_files(self, base_commit: str | None) -> set[str] | None:
        """Files that differ from the recorded fingerprints.

        The snapshot is content-keyed (independent of any base commit),
        so a rebase or amend does not invalidate anything.  Shards
        without a snapshot fall back to ``git diff``.  None means
        "cannot decide" (running everything).
        """
        snapshot = self.load_files()
        if snapshot is not None:
            return detect_changes(self.root, snapshot, self.project_files)
        return self.changed_since(base_commit)

    def changed_since(self, base_commit: str | None) -> set[str] | None:
        """``git diff <base_commit>`` fallback: changed + untracked files."""
        if base_commit is None:
            return None
        try:
            changed = set(self._git("diff", "--name-only", base_commit).splitlines())
            changed |= set(self._git(
                "ls-files", "--others", "--exclude-standard"
            ).splitlines())
        except (subprocess.CalledProcessError, OSError):
            return None
        changed.discard("")
        return changed

    # -- persistence -------------------------------------------------------

    def _read(self, name: str):
        try:
            return json.loads((self.dir / name).read_bytes())
        except (OSError, ValueError):
            return None

    def load_meta(self) -> dict | None:
        return self._read("meta.json")

    def load_data(self) -> dict | None:
        return self._read("data.json")

    def load_files(self) -> dict | None:
        return self._read("files.json")

    def load(self) -> tuple[dict | None, dict | None]:
        """(data, meta); (None, None) when there is nothing stored."""
        if self.load_meta() is None:
            return None, None
        return self.load_data(), self.load_meta()

    def save(self, data: dict, meta: dict, files: dict | None = None) -> None:
        """Write the files atomically (``files=None`` leaves it untouched)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, obj in (
            ("data.json", data),
            ("meta.json", meta),
            ("files.json", files),
        ):
            if obj is None:
                continue
            tmp = self.dir / f".{name}.tmp"
            tmp.write_bytes(json.dumps(obj, separators=(",", ":")).encode())
            tmp.replace(self.dir / name)

    def delete(self) -> bool:
        """Remove everything; False when there was nothing to delete."""
        if not self.dir.is_dir():
            return False
        shutil.rmtree(self.dir)
        return True
