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

Files outside the repository are ignored.

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

### 3. Selection: git diff

`changed = (files differing between the shard's base commit and the current
working tree) ∪ (untracked files)`. A test runs when:

- it has **no** recorded data (new test, or the previous run never finished
  it — e.g. it was interrupted),
- it **failed** in the recorded run, or
- one of its dependencies is in `changed`.

Everything else is deselected (pytest reports them as `deselected`). If the
recorded data is missing/unreadable, or the base commit no longer exists
(e.g. after a rebase), everything runs.

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

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest
```

The test suite uses `pytester` subprocess runs against real git repositories.

## Layout

```
src/pytest_filemon/
├── plugin.py      # pytest hooks, lifecycle, options
├── monitoring.py  # sys.monitoring tool: PY_START + import bytecode scan
├── gitstore.py    # JSON shards behind refs/filemon/<shard>
├── selection.py   # shard parsing + test selection
└── util.py        # path normalization, command shard key
```
