"""Test selection: recorded per-test file dependencies vs. changed files.

A test is selected when

* it has no recorded data (new test, or the previous run never finished it),
* it failed in the recorded run, or
* one of its dependencies changed since the recorded run's base commit.

Everything else is deselected.  When the recorded data is missing or cannot
be interpreted, everything is selected ("run all").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Shard:
    """Parsed shard data for one pytest command."""

    base_commit: str | None
    tests: dict[str, set[str]]
    failed: set[str]


@dataclass
class Selection:
    """Outcome of the selection for one run."""

    selected: list[str]
    deselected: list[str]
    changed: set[str] = field(default_factory=set)
    reason: str = ""

    @property
    def selected_set(self) -> set[str]:
        return set(self.selected)


def load_shard(data: dict | None, meta: dict | None) -> Shard | None:
    """Parse a shard, or return None if the data is unusable (fresh/corrupt)."""
    if not isinstance(data, dict) or not isinstance(meta, dict):
        return None
    try:
        if data.get("version") != 1:
            return None
        raw_tests = data["tests"]
        if not isinstance(raw_tests, dict):
            return None
        tests: dict[str, set[str]] = {}
        for nodeid, deps in raw_tests.items():
            if not isinstance(deps, list):
                return None
            tests[nodeid] = {d for d in deps if isinstance(d, str)}
        raw_failed = data.get("failed") or []
        if not isinstance(raw_failed, list):
            return None
        failed = {f for f in raw_failed if isinstance(f, str)}
        base_commit = meta.get("base_commit")
        if base_commit is not None and not isinstance(base_commit, str):
            return None
    except Exception:
        return None
    return Shard(base_commit=base_commit, tests=tests, failed=failed)


def select_tests(collected: list[str], shard: Shard | None, changed: set[str] | None) -> Selection:
    """Decide which of *collected* tests should run."""
    if shard is None:
        return Selection(
            list(collected),
            [],
            set(),
            "no usable stored data for this command; running and recording everything",
        )
    if changed is None:
        return Selection(
            list(collected),
            [],
            set(),
            "cannot compute changes since the recorded base commit "
            "(rebased/rewritten history?); running everything",
        )
    selected: list[str] = []
    deselected: list[str] = []
    n_affected = n_failed = n_new = 0
    for nodeid in collected:
        deps = shard.tests.get(nodeid)
        if deps is None:
            selected.append(nodeid)
            n_new += 1
        elif nodeid in shard.failed:
            selected.append(nodeid)
            n_failed += 1
        elif deps & changed:
            selected.append(nodeid)
            n_affected += 1
        else:
            deselected.append(nodeid)
    base = (shard.base_commit or "?")[:8]
    reason = (
        f"{len(changed)} file(s) changed since {base}: "
        f"{n_affected} affected, {n_failed} previously failed, {n_new} new"
    )
    return Selection(selected, deselected, set(changed), reason)
