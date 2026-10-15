"""Shard storage in git: dedicated refs under ``refs/filemon/``, never commits.

A shard is stored as a lightweight commit whose tree contains two blobs::

    data.json  -- {"version": 1, "tests": {nodeid: [dep, ...]}, "failed": [...]}
    meta.json  -- {"version": 1, "shard": ..., "base_commit": ..., "created": ...}

The ref ``refs/filemon/<shard>`` always points at the latest record for that
pytest command ("latest wins"; overwriting is just an ``update-ref``).  The
ref never touches the working tree, HEAD, or any branch, and plain
``git push``/``git fetch``/``git clone`` do not transfer it.  To share shards
between machines, use an explicit refspec::

    git push   origin 'refs/filemon/*:refs/filemon/*'
    git fetch   origin 'refs/filemon/*:refs/filemon/*'
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .util import run_git


class GitStore:
    """Read/write one dependency shard behind a dedicated git ref."""

    def __init__(self, root: Path, shard: str):
        self.root = Path(root)
        self.shard = shard
        self.ref = f"refs/filemon/{shard}"

    def _git(self, *args: str, stdin: str | None = None, extra_env: dict | None = None) -> str:
        env = None
        if extra_env:
            env = {**os.environ, **extra_env}
        proc = subprocess.run(
            ["git", "-C", str(self.root), *args],
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return proc.stdout

    # -- repository state ------------------------------------------------

    def head(self) -> str | None:
        """Current HEAD commit sha, or None on an unborn branch."""
        try:
            return run_git(self.root, "rev-parse", "HEAD").strip() or None
        except subprocess.CalledProcessError:
            return None

    def commit_exists(self, sha: str | None) -> bool:
        if not sha:
            return False
        try:
            run_git(self.root, "cat-file", "-e", f"{sha}^{{commit}}")
            return True
        except subprocess.CalledProcessError:
            return False

    def changed_since(self, base_commit: str | None) -> set[str] | None:
        """Files changed between *base_commit* and the current working tree.

        Includes staged/unstaged modifications and untracked files (untracked
        files that carry uncommitted edits would be invisible to a plain
        diff, and any untracked file that happens to be a dependency must
        trigger a re-run).  Returns None when the change set cannot be
        computed (e.g. the base commit is gone after a rebase).
        """
        if not base_commit or not self.commit_exists(base_commit):
            return None
        changed: set[str] = set()
        out = run_git(self.root, "diff", "--no-renames", "--name-only", base_commit)
        changed.update(line for line in out.splitlines() if line)
        out = run_git(self.root, "ls-files", "--others", "--exclude-standard")
        changed.update(line for line in out.splitlines() if line)
        return changed

    # -- shard persistence -------------------------------------------------

    def load(self) -> tuple[dict | None, dict | None]:
        """Load the latest shard, or (None, None) if absent/unreadable."""
        try:
            sha = run_git(self.root, "rev-parse", "--verify", "--quiet", self.ref).strip()
        except subprocess.CalledProcessError:
            return None, None
        if not sha:
            return None, None
        try:
            data = json.loads(self._git("cat-file", "blob", f"{self.ref}:data.json"))
            meta = json.loads(self._git("cat-file", "blob", f"{self.ref}:meta.json"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return None, None
        return data, meta

    def save(self, data: dict, meta: dict) -> None:
        """Store the shard as a fresh lightweight commit and move the ref."""
        data_blob = self._git(
            "hash-object", "-w", "--stdin", stdin=json.dumps(data)
        ).strip()
        meta_blob = self._git(
            "hash-object", "-w", "--stdin", stdin=json.dumps(meta)
        ).strip()
        tree_input = (
            f"100644 blob {data_blob}\tdata.json\n"
            f"100644 blob {meta_blob}\tmeta.json\n"
        )
        tree = self._git("mktree", stdin=tree_input).strip()
        message = f"pytest-filemon shard {self.shard}"
        try:
            commit = self._git("commit-tree", tree, "-m", message).strip()
        except subprocess.CalledProcessError:
            # The user may not have a git identity configured (e.g. CI);
            # synthesize one so the synthetic ref can still be written.
            commit = self._git(
                "commit-tree",
                tree,
                "-m",
                message,
                extra_env={
                    "GIT_AUTHOR_NAME": "pytest-filemon",
                    "GIT_AUTHOR_EMAIL": "filemon@invalid",
                    "GIT_COMMITTER_NAME": "pytest-filemon",
                    "GIT_COMMITTER_EMAIL": "filemon@invalid",
                },
            ).strip()
        self._git("update-ref", self.ref, commit)

    def delete(self) -> bool:
        """Delete the shard ref. Returns False if there was nothing to delete."""
        try:
            sha = run_git(self.root, "rev-parse", "--verify", "--quiet", self.ref).strip()
        except subprocess.CalledProcessError:
            return False
        if not sha:
            return False
        try:
            run_git(self.root, "update-ref", "-d", self.ref)
            return True
        except subprocess.CalledProcessError:
            return False

    def list_shards(self) -> list[str]:
        """All shard keys recorded in this repository."""
        try:
            out = run_git(self.root, "for-each-ref", "--format=%(refname)", "refs/filemon/")
        except subprocess.CalledProcessError:
            return []
        return [line.rsplit("/", 1)[-1] for line in out.splitlines() if line]
