# pytest-filemon

Minimal file-level **test impact analysis (TIA)** for pytest, inspired by
pytest-testmon but deliberately different:

| | pytest-testmon | pytest-filemon |
|---|---|---|
| dependency storage | sqlite | JSON shards stored in git refs |
| dependency resolution | code-block / line level | **file paths**, relative to the repo root |
| coverage collection | coverage.py | a custom `sys.monitoring` tool using **PY_START** events (no per-line tracking) |
| dynamic imports | — | conservative bytecode scan for import opcodes inside every executed code block |
| import-time dependencies | — | everything imported before tests start is a dependency of *every* test |
| change detection | mtime / sqlite / VCS | **git diff** between the recorded base commit and the current working tree |

Requires **Python 3.12+** (for `sys.monitoring`) and a git repository.

## Usage

```bash
pip install pytest-filemon

pytest --filemon              # first run: everything runs, dependencies are recorded
pytest --filemon              # second run: only tests affected by changes run
```

Options:

| option | effect |
|---|---|
| `--filemon` | enable recording + selection |
| `--filemon-force` | run all tests, but still record dependencies |
| `--filemon-clean` | discard the stored data for this command before running |
| `PYTEST_FILEMON=1` | same as `--filemon` (env form; also enables before conftest loading) |
| `-p no:filemon` | disable the plugin entirely |

If nothing changed and nothing failed, **zero tests run and pytest exits with
code 5** ("no tests collected"), just like a `pytest -k nomatch` run.

## How it works

### 1. Dependency collection (`sys.monitoring`, PY_START)

The plugin registers a monitoring tool (PEP 669) with a single global event:
`PY_START`. Every time a code block starts executing it records the repo
file it lives in — file-level coverage, no lines. On top of that, for every
code block that starts, the plugin scans its bytecode for `IMPORT_NAME` /
`IMPORT_FROM` opcodes and resolves the referenced modules to file paths. This
is the conservative treatment of dynamic imports: an `import` inside a branch
that never runs is still recorded as a dependency of the test that executed
the surrounding block. Relative imports are resolved against the importing
file's package; unresolvable names are skipped (if the import really runs,
the imported module's own `PY_START` captures it anyway).

Events are bucketed by phase:

- **import phase** (everything before collection: pytest plugins, conftest
  imports) → dependencies of **every** test. Changing a conftest re-runs the
  world.
- **collect phase** (test module imports) → shared by all tests, except the
  test modules themselves, which are attributed only to their own tests, so
  editing one test file does not re-run the whole suite.
- **test phase** → attributed to the running test's node id.

Files outside the repository are ignored. So are virtual environments and
third-party trees that happen to live *inside* the repo (`.venv`,
`site-packages`, `.tox`, `node_modules`, ...) and the plugin's own package —
only first-party repository code is ever recorded.

### 2. Storage: JSON shards in git refs (never a commit)

For every pytest **command** the data is stored in a dedicated ref:

```
refs/filemon/<shard>   →  lightweight commit
                            ├── data.json   {"version":1, "tests": {nodeid: [files...]}, "failed": [...]}
                            └── meta.json   {"version":1, "shard":..., "base_commit":..., "created":...}
```

- The shard key is a hash of the **normalized** invocation arguments
  (repo-relative paths, order- and duplicate-insensitive), so only a re-run
  of the same command reuses the data. `pytest tests/ -q` and
  `pytest -q tests/` share a shard; `pytest tests/test_a.py` is a different
  one.
- The ref is updated with git plumbing (`hash-object` / `mktree` /
  `commit-tree` / `update-ref`); it never touches the working tree, `HEAD`,
  or any branch, and nothing is ever committed by normal `git commit`.
- Latest record wins: the ref always points at the most recent run's data.

Plain `git push`, `git fetch` and `git clone` do **not** transfer these refs.
To share shards between machines, use explicit refspecs:

```bash
git push  origin 'refs/filemon/*:refs/filemon/*'
git fetch  origin 'refs/filemon/*:refs/filemon/*'

# inspect a shard
git cat-file blob refs/filemon/<shard>:data.json | python -m json.tool
```

### 3. Selection: git diff + collection skipping

Before collection starts, the changed-file set is computed and test files
whose recorded tests are **all** unaffected are not even *collected* (their
modules are never imported) — the same optimization testmon applies via
``pytest_ignore_collect``.  This is safe because a test always depends on
its own module file: any edit (including adding a test) selects the file's
tests, and a brand-new file has no recorded tests and is always collected.
(Files given as explicit pytest *arguments* are always collected anyway:
pytest exempts command-line paths -- "init paths" -- from
``pytest_ignore_collect``.)

When explicit test selectors are in play (``-k``, ``-m``, ``--lf``/``--ff``,
or a ``file.py::test`` argument), selection is disabled entirely (matching
testmon's forced ``--testmon-noselect``): an explicitly requested but
unaffected test must never be silently skipped.

`changed = (files differing between the shard's base commit and the current
working tree) ∪ (untracked files)`. A test runs when:

- it has **no** recorded data (new test, or the previous run never finished
  it — e.g. it was interrupted),
- it **failed** in the recorded run, or
- one of its dependencies is in `changed`.

Everything else is deselected — or, when its whole file is unaffected, its
file is never collected. If the recorded data is missing/unreadable, or the
base commit no longer exists (e.g. after a rebase), everything runs.

Performance (sqlalchemy `test/orm`, 11,313 tests, Python 3.13, serial):
recording costs ~1.55× a plain run (~28s on top of a 52s suite).  The
sys.monitoring floor itself is tiny (~3%, measured with a no-op callback);
the cost is our per-event Python callback and the first-time analysis of
each code block, so:

- code objects whose analysis yields no dependencies (stdlib, the venv,
  pytest internals) are permanently silenced by returning
  ``sys.monitoring.DISABLE`` -- CPython honors DISABLE even for
  globally-set events, so out-of-repo code fires essentially once;
- in-repo code objects keep firing (attribution to the running test is
  required) but only pay a dict lookup + identity comparison, with deps
  re-attributed only when the test changes;
- code blocks without any import bytecode (about three quarters) are
  detected with a byte-level pre-check and skip disassembly entirely;
- a repository module index (one walk, built lazily) resolves import
  names to files without calling ``find_spec`` -- which not only avoids
  hundreds of microseconds per resolution but also removes its side
  effect of importing parent packages.  That side effect previously cost
  real correctness: module trees first imported *inside* our resolution
  had their one-time PY_START events swallowed, silently dropping
  dependencies (on sqlalchemy, whole ``orm`` submodules went unrecorded).
  Resolving from the index imports nothing, so import-time dependency
  tracking is now complete.  The index over-approximates importability,
  which is safe in the conservative direction: a scan-resolved dep can
  only ever be an *extra* dep, because an import that actually executes
  is always captured directly by the imported module's own PY_START;
- file→repo-relative-path resolution (LRU cache), exclusion checks, and
  per-(module, level, file) import resolutions are each computed once
  and memoized.

A no-change rerun takes ~2s because collection is skipped outright;
editing a test file re-runs only that file's tests (~3s for 3 tests).
Note that in packages with an eager ``__init__`` (sqlalchemy's
``sqlalchemy.orm`` imports most of the package at import time), every
eagerly imported module is legitimately an import-time dependency of
every test: changing those files re-runs everything.  That is the
conservative model working as designed.

## Limitations (by design)

- **pytest-xdist is not supported.** The monitor is process-global and the
  shard is written once per run; with multiple workers the last writer wins
  and selection happens per worker.
- Dependencies are **files**, not code blocks; a change to a file a test
  merely shares with unrelated code re-runs that test (conservative).
- Thread attribution: code executed by threads spawned by a test is
  attributed to the test running at that moment; tests are assumed to run
  sequentially.
- Session-fixture teardowns that run after the last test may be attributed
  to that test.
- Line-level coverage is intentionally not tracked.

## Development

The repo is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # create/update .venv from uv.lock (dev tools included)
uv run pytest      # run the test suite (syncs automatically if needed)
uv run ruff check src tests
uv lock            # re-resolve after editing dependencies
```

The lockfile (`uv.lock`) and `.python-version` are committed; the dev tools
(ruff) live in the PEP 735 `dev` dependency group. Any Python 3.12+ works
(`uv sync` will fetch one if needed).

The test suite uses `pytester` subprocess runs against real git repositories,
so it needs network-free local git only.

## Layout

```
src/pytest_filemon/
├── plugin.py      # pytest hooks, lifecycle, options
├── monitoring.py  # sys.monitoring tool: PY_START + import bytecode scan
├── gitstore.py    # JSON shards behind refs/filemon/<shard>
├── selection.py   # shard parsing + test selection
└── util.py        # path normalization, command shard key
```
