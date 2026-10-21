"""Path and command-key helpers."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


def run_git(root: Path, *args: str, stdin: str | None = None, timeout: float | None = None, extra_env: dict | None = None) -> str:
    """Run a git command in *root*; raise CalledProcessError on failure."""
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        env=env,
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


def _compute_repo_rel(filename: str, root: Path) -> str | None:
    """Uncached core of :func:`to_repo_rel`."""
    try:
        p = Path(filename)
        if not p.is_absolute():
            p = Path.cwd() / p
        p = p.resolve()
        # Guard against phantom relative co_filenames (e.g. bare
        # "bootstrap.py" from some generated/embedded code): only real files
        # can be dependencies.
        if not p.is_file():
            return None
        rel = p.relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return None
    parts = rel.parts
    if not parts or ".." in parts:
        return None
    if _EXCLUDED_SEGMENTS.intersection(parts):
        return None
    return rel.as_posix()


def to_repo_rel(filename: str | os.PathLike | None, root: Path) -> str | None:
    """Normalize a filename to a repo-root-relative POSIX path.

    Returns None for synthetic locations (``<string>``, ``<frozen ...>``),
    non-strings, paths outside the repository, and paths inside virtual
    environments / site-packages / similar third-party trees.

    Deliberately a pure function with no module-level cache: the monitor
    memoizes per instance (see FileMonitor.repo_rel), so concurrent or
    nested monitors in one process never share (or stale-share) state.
    """
    if not filename:
        return None
    if not isinstance(filename, str):
        try:
            filename = os.fspath(filename)
        except TypeError:
            return None
    if not isinstance(filename, str) or not filename or filename.startswith("<"):
        return None
    return _compute_repo_rel(filename, root)


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
# (so --less-force/--less-clean update the same shard as a normal run)
# and pytest's volatile --basetemp (unique per pytester run).
_NON_KEY_FLAGS = {"--less", "--less-force", "--less-clean"}


# Arguments that never change which tests are collected or executed, only
# how results are presented (or how early the run stops -- partial runs merge
# safely into the cumulative shard).  Excluded from the shard key so that
# e.g. ``pytest tests -q`` and ``pytest tests -v`` share one database.
#
# DELIBERATELY an allowlist, not a denylist: anything not listed stays in
# the key.  Over-merging shards risks stale selections (a wrong test set);
# under-merging only costs a fresh record run.  Notably ``-W`` (warning
# filters can change test outcomes mid-run and thus the executed code),
# ``-p`` (plugin loading), ``--deselect``, ``-k``/``-m`` and positional args
# all remain part of the key.
_PRESENATION_NOARG = frozenset(
    {
        "-q",
        "--quiet",
        "-v",
        "--verbose",
        "-l",
        "--showlocals",
        "--full-trace",
        "--no-header",
        "--no-summary",
        "-x",
        "--exitfirst",
    }
)
_PRESENTATION_VALUE = frozenset(
    {
        "--tb",          # traceback style
        "-r",            # report chars (e.g. -ra)
        "--durations",
        "--durations-min",
        "--verbosity",
        "--color",
        "--maxfail",     # stops early; partial runs merge safely
    }
)


def _filter_args(args: Sequence[str]) -> list[str]:
    """Drop key-irrelevant arguments from *args*.

    Three groups are excluded: the plugin's own flags, volatile value-taking
    options (``--basetemp``), and presentation-only flags (verbosity, report
    formatting, early exit).  Unknown arguments are always kept.
    """
    out: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in _NON_KEY_FLAGS or arg in _PRESENATION_NOARG:
            continue
        if arg in _PRESENTATION_VALUE or arg == "--basetemp":
            skip_next = True
            continue
        stripped = arg.split("=", 1)[0]
        if stripped in _PRESENTATION_VALUE or arg.startswith("--basetemp="):
            continue
        out.append(arg)
    return out


def command_key(args: Sequence[str], root: Path) -> str:
    """Shard key for a pytest invocation: normalized args, order/dup-insensitive.

    The same *set* of normalized arguments always produces the same shard,
    regardless of argument order or repetitions.  Excluded from the key: the
    plugin's own flags, volatile pytest arguments (``--basetemp``), and
    presentation-only flags (``-q``/``-v``/``--tb``/``-x``/... -- see
    ``_filter_args``); everything else, notably selection-affecting arguments
    (``-k``, ``-m``, ``--deselect``, positional paths) and unknown arguments,
    stays in the key.
    """
    filtered = _filter_args(args)
    normalized = sorted({normalize_arg(a, root) for a in filtered})
    blob = "\x00".join(normalized)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
