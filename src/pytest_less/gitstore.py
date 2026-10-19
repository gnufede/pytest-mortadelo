"""Shard storage in git: dedicated refs under ``refs/less/``, never commits.

A shard is stored as a lightweight commit whose tree contains three blobs::

    data.json  -- {"version": 2, "blocks": [...], "tests": {...}, "failed": [...], ...}
    meta.json  -- {"version": 1, "shard": ..., "base_commit": ..., "created": ...}
    parse.json -- per-file parse results keyed by content hash (optional)

The ref ``refs/less/<shard>`` always points at the latest record for that
pytest command ("latest wins"; overwriting is just an ``update-ref``).  The
ref never touches the working tree, HEAD, or any branch, and plain
``git push``/``git fetch``/``git clone`` do not transfer it.  Sharing is
either automatic (``less_sync`` / ``PYTEST_LESS_SYNC``, see the
README: the plugin pulls the shard before selection and pushes it after
recording) or manual via an explicit refspec::

    git push   origin 'refs/less/*:refs/less/*'
    git fetch   origin 'refs/less/*:refs/less/*'
"""

from __future__ import annotations

import json
import os
import subprocess
import zlib
from datetime import datetime
from pathlib import Path

from .gitobjects import Repo, Unsupported, backend_enabled
from .util import run_git

# What the in-process object layer may raise when it meets something it does
# not model (packed objects, odd layouts, a held lock): the caller then asks
# git instead.  Correctness never depends on the fast path applying.
_FALLBACK = (Unsupported, OSError, ValueError, KeyError, IndexError, zlib.error)

# Sync operations must never hang or prompt for credentials during a test run.
_SYNC_ENV = {"GIT_TERMINAL_PROMPT": "0"}
_SYNC_TIMEOUT = 30.0


def _parse_iso(value) -> object:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class GitStore:
    """Read/write one dependency shard behind a dedicated git ref."""

    def __init__(self, root: Path, shard: str, *, memoize: bool = False):
        self.root = Path(root)
        self.shard = shard
        self.ref = f"refs/less/{shard}"
        # Startup asks the same questions from several places (the early
        # probe, setup, collection planning).  Every answer is a git spawn
        # -- a few ms on a quiet machine, a quarter of a second on one with
        # endpoint scanning -- so the plugin's shared store remembers them
        # until the ref changes (sync pull / save / delete).  Off by default:
        # a store used directly must always see the current state.
        self._memoize = memoize
        self._memo: dict = {}
        # Reads/writes of refs and loose objects without spawning git; None
        # when disabled (PYTEST_LESS_GIT_BACKEND=subprocess) or the layout is
        # not one it models.
        self._repo = Repo.discover(self.root) if backend_enabled() else None

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

    def _git_concurrently(self, *commands: tuple[str, ...]) -> list[str | None]:
        """Run git commands at the same time; ``None`` for each one that failed."""
        procs = [
            subprocess.Popen(
                ["git", "-C", str(self.root), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for args in commands
        ]
        results: list[str | None] = []
        for proc in procs:
            out, _ = proc.communicate()
            results.append(out if proc.returncode == 0 else None)
        return results

    # -- repository state ------------------------------------------------

    def head(self) -> str | None:
        """Current HEAD commit sha, or None on an unborn branch."""
        if self._repo is not None:
            try:
                return self._repo.head()
            except _FALLBACK:
                pass
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
        if not base_commit:
            return None
        key = ("changed", base_commit)
        if self._memoize and key in self._memo:
            return set(self._memo[key])
        changed: set[str] = set()
        # The two queries are independent: start both, then collect, so the
        # latency is the slower one rather than the sum (it matters where a
        # spawn is slow).  No separate "does the commit exist" probe: a
        # missing base makes the diff itself fail.
        diff, others = self._git_concurrently(
            ("diff", "--no-renames", "--name-only", base_commit),
            ("ls-files", "--others", "--exclude-standard"),
        )
        if diff is None or others is None:
            return None
        changed.update(line for line in diff.splitlines() if line)
        changed.update(line for line in others.splitlines() if line)
        if self._memoize:
            self._memo[key] = set(changed)
        return changed

    # -- shard persistence -------------------------------------------------

    def _read_blob(self, name: str) -> bytes | None:
        """Content of ``<ref>:<name>``, or None when the shard or the entry
        does not exist.  In-process for loose objects; git otherwise."""
        if self._repo is not None:
            try:
                return self._repo.blob_in_ref(self.ref, name)
            except _FALLBACK:
                pass
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.root), "cat-file", "blob", f"{self.ref}:{name}"],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return None
        return proc.stdout

    def _read_json(self, name: str) -> dict | None:
        raw = self._read_blob(name)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def load_meta(self) -> dict | None:
        """Load only meta.json (tiny: the fast-path summary)."""
        if self._memoize and "meta" in self._memo:
            return self._memo["meta"]
        meta = self._read_json("meta.json")
        if self._memoize:
            self._memo["meta"] = meta
        return meta

    def load_data(self) -> dict | None:
        """Load only data.json (the potentially large dependency blob)."""
        return self._read_json("data.json")

    def load_parse(self) -> dict | None:
        """Load parse.json (per-file parse results keyed by content hash)."""
        return self._read_json("parse.json")

    def load(self) -> tuple[dict | None, dict | None]:
        """Load the latest shard, or (None, None) if absent/unreadable."""
        if self._ref_sha(self.ref) is None:
            return None, None
        return self.load_data(), self.load_meta()

    def save(self, data: dict, meta: dict, parse: dict | None = None) -> None:
        """Store the shard as a fresh lightweight commit and move the ref.

        *parse* is the new parse.json content; ``None`` carries the existing
        blob over untouched (by object id: it is not even read).

        Written in-process (zlib + sha1 + a ref lock: 39ms for 12 MB against
        165ms and six spawns through ``hash-object`` / ``mktree`` /
        ``commit-tree`` / ``update-ref``); git does it when the layout is not
        one the in-process writer models, or anything goes wrong.
        """
        if self._repo is not None:
            try:
                self._save_in_process(data, meta, parse)
                self._memo.pop("meta", None)
                return
            except _FALLBACK:
                pass
        self._save_with_git(data, meta, parse)

    def _save_in_process(self, data: dict, meta: dict, parse: dict | None) -> None:
        files = {
            "data.json": json.dumps(data).encode(),
            "meta.json": json.dumps(meta).encode(),
        }
        reuse: dict[str, str] = {}
        if parse is not None:
            files["parse.json"] = json.dumps(parse).encode()
        else:
            # Carry the stored blob over by id (it may even be packed).
            existing = self._repo.blob_sha_in_ref(self.ref, "parse.json")
            if existing:
                reuse["parse.json"] = existing
        self._repo.write_shard(
            self.ref, files, f"pytest-less shard {self.shard}", reuse=reuse
        )

    def _save_with_git(self, data: dict, meta: dict, parse: dict | None) -> None:
        data_blob = self._git(
            "hash-object", "-w", "--stdin", stdin=json.dumps(data)
        ).strip()
        meta_blob = self._git(
            "hash-object", "-w", "--stdin", stdin=json.dumps(meta)
        ).strip()
        if parse is not None:
            parse_blob = self._git(
                "hash-object", "-w", "--stdin", stdin=json.dumps(parse)
            ).strip()
        else:
            parse_blob = self._ref_sha(f"{self.ref}:parse.json")
        tree_input = (
            f"100644 blob {data_blob}\tdata.json\n"
            f"100644 blob {meta_blob}\tmeta.json\n"
        )
        if parse_blob:
            tree_input += f"100644 blob {parse_blob}\tparse.json\n"
        tree = self._git("mktree", stdin=tree_input).strip()
        message = f"pytest-less shard {self.shard}"
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
                    "GIT_AUTHOR_NAME": "pytest-less",
                    "GIT_AUTHOR_EMAIL": "less@invalid",
                    "GIT_COMMITTER_NAME": "pytest-less",
                    "GIT_COMMITTER_EMAIL": "less@invalid",
                },
            ).strip()
        self._git("update-ref", self.ref, commit)
        self._memo.pop("meta", None)

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
            self._memo.pop("meta", None)
            return True
        except subprocess.CalledProcessError:
            return False

    # -- shard sync (sharing between machines) ----------------------------

    def sync_pull(self, remote: str) -> str:
        """Pull this shard from *remote*, adopting it if newer than local.

        Returns a human-readable status for the run summary; never raises
        (a missing shard, an unreachable remote or a timeout just means no
        sync, and the run continues with whatever is local).
        """
        scratch = f"refs/less-remote/{self.shard}"
        try:
            run_git(
                self.root,
                "fetch",
                "--quiet",
                remote,
                f"+refs/less/{self.shard}:{scratch}",
                timeout=_SYNC_TIMEOUT,
                extra_env=_SYNC_ENV,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            self._delete_ref_quiet(scratch)
            return (
                f"no shard for this command on '{remote}' (or remote unreachable); "
                "using local data"
            )
        remote_created = _parse_iso(
            (self._json_blob(f"{scratch}:meta.json") or {}).get("created")
        )
        local_created = _parse_iso(
            (self._json_blob(f"{self.ref}:meta.json") or {}).get("created")
        )
        if remote_created is None or (
            local_created is not None and remote_created <= local_created
        ):
            self._delete_ref_quiet(scratch)
            return f"shard on '{remote}' is not newer than local; keeping local"
        run_git(self.root, "update-ref", self.ref, scratch)
        self._memo.pop("meta", None)
        self._delete_ref_quiet(scratch)
        return f"pulled a newer shard from '{remote}'"

    def sync_push(self, remote: str) -> str:
        """Push this shard to *remote* (latest record wins; force is safe:
        the ref is single-purpose, tiny, and never a branch)."""
        if self._ref_sha(self.ref) is None:
            return "nothing recorded locally; skipping push"
        try:
            run_git(
                self.root,
                "push",
                "--quiet",
                "--force",
                remote,
                f"{self.ref}:{self.ref}",
                timeout=_SYNC_TIMEOUT,
                extra_env=_SYNC_ENV,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return f"push to '{remote}' failed ({e}); shard kept locally"
        return f"shard pushed to '{remote}'"

    def _ref_sha(self, ref: str) -> str | None:
        if self._repo is not None and ":" not in ref:
            try:
                return self._repo.read_ref(ref)
            except _FALLBACK:
                pass
        try:
            sha = run_git(self.root, "rev-parse", "--verify", "--quiet", ref)
            return sha.strip() or None
        except subprocess.CalledProcessError:
            return None

    def _json_blob(self, rev: str) -> dict | None:
        try:
            return json.loads(run_git(self.root, "cat-file", "blob", rev))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return None

    def _delete_ref_quiet(self, ref: str) -> None:
        try:
            run_git(self.root, "update-ref", "-d", ref)
        except subprocess.CalledProcessError:
            pass

    def list_shards(self) -> list[str]:
        """All shard keys recorded in this repository."""
        try:
            out = run_git(self.root, "for-each-ref", "--format=%(refname)", "refs/less/")
        except subprocess.CalledProcessError:
            return []
        return [line.rsplit("/", 1)[-1] for line in out.splitlines() if line]
