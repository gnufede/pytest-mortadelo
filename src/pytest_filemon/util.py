"""Path and command-key helpers."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


def run_git(root: Path, *args: str, stdin: str | None = None) -> str:
    """Run a git command in *root*; raise CalledProcessError on failure."""
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def git_root(start: Path | str | None = None) -> Path | None:
    """Toplevel directory of the git repository containing *start* (or None)."""
    start = Path(start) if start is not None else Path.cwd()
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not out:
        return None
    return Path(out).resolve()


# Path segments that never hold first-party repository code: virtual
# environments and third-party packages that happen to live *inside* the
# repository tree (uv's default in-repo .venv, tox envs, vendored deps). A
# repo directory legitimately named like one of these is an accepted loss.
_EXCLUDED_SEGMENTS = frozenset(
    {
        "site-packages",
        "dist-packages",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".nox",
    }
)


def to_repo_rel(filename: str | os.PathLike | None, root: Path) -> str | None:
    """Normalize a filename to a repo-root-relative POSIX path.

    Returns None for synthetic locations (``<string>``, ``<frozen ...>``),
    non-strings, paths outside the repository, and paths inside virtual
    environments / site-packages / similar third-party trees.
    """
    if not filename:
        return None
    if not isinstance(filename, str):
        try:
            filename = os.fspath(filename)
        except TypeError:
            return None
    if not isinstance(filename, str) or filename.startswith("<"):
        return None
    try:
        p = Path(filename)
        if not p.is_absolute():
            p = Path.cwd() / p
        p = p.resolve()
        rel = p.relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return None
    parts = rel.parts
    if not parts or ".." in parts:
        return None
    if _EXCLUDED_SEGMENTS.intersection(parts):
        return None
    return rel.as_posix()


def normalize_arg(arg: str, root: Path) -> str:
    """Normalize a single command-line argument for the shard key.

    Existing paths inside the repository become repo-relative POSIX paths, so
    that ``pytest /abs/repo/tests`` and ``pytest tests`` map to the same shard.
    Everything else is kept verbatim.
    """
    p = Path(arg)
    if p.exists():
        try:
            rel = p.resolve().relative_to(root)
            if rel.parts and ".." not in rel.parts:
                return rel.as_posix()
        except (OSError, ValueError):
            pass
    return arg


# Arguments that must NOT influence the shard key: the plugin's own flags
# (so --filemon-force/--filemon-clean update the same shard as a normal run)
# and pytest's volatile --basetemp (unique per pytester run).
_NON_KEY_FLAGS = {"--filemon", "--filemon-force", "--filemon-clean"}


def _filter_args(args: Sequence[str]) -> list[str]:
    """Drop plugin flags and volatile value-taking options from *args*."""
    out: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in _NON_KEY_FLAGS:
            continue
        if arg == "--basetemp":
            skip_next = True
            continue
        if arg.startswith("--basetemp="):
            continue
        out.append(arg)
    return out


def command_key(args: Sequence[str], root: Path) -> str:
    """Shard key for a pytest invocation: normalized args, order/dup-insensitive.

    The same *set* of normalized arguments always produces the same shard,
    regardless of argument order or repetitions; adding/removing/rewording an
    argument produces a different shard.  The plugin's own flags and volatile
    pytest arguments (``--basetemp``) are excluded from the key.
    """
    filtered = _filter_args(args)
    normalized = sorted({normalize_arg(a, root) for a in filtered})
    blob = "\x00".join(normalized)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
