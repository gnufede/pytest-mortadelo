# pytest-less

Minimal **test impact analysis (TIA)** for pytest.

| | pytest-less |
|---|---|
| dependency storage | JSON shards stored in git refs |
| dependency resolution | **code blocks** (qualified name + AST hash) and module skeletons for imports |
| dependency collection | a custom `sys.monitoring` tool using **PY_START** events (no per-line tracking) |
| dynamic imports | conservative bytecode scan for import opcodes inside every executed code block |
| import-time dependencies | everything imported before tests start is a dependency of *every* test |
| change detection | **git diff** between the recorded base commit and the current working tree |

Requires **Python 3.12+** (for `sys.monitoring`) and a git repository.

## Usage

```bash
pip install pytest-less

pytest --less              # first run: everything runs, dependencies are recorded
pytest --less              # second run: only tests affected by changes run
```

Options:

| option | effect |
|---|---|
| `--less` | enable recording + selection |
| `--less-force` | run all tests, but still record dependencies |
| `--less-clean` | discard the stored data for this command before running |
| `--less-no-data-files` | disable data-file tracking (on by default); ini: `less_data_files = false` |
| `--less-no-cosmetic` | hash docstrings as behavior (by default comments *and* docstrings are ignored); ini: `less_cosmetic = false`; automatically off under `--doctest-modules` |
| `PYTEST_LESS=1` | same as `--less` (env form; also enables before conftest loading) |
| `-p no:less` | disable the plugin entirely |

If nothing changed and nothing failed, **zero tests run and pytest exits with
code 5** ("no tests collected"), just like a `pytest -k nomatch` run.

## How it works

### 1. Dependency collection (`sys.monitoring`, PY_START)

The plugin registers a monitoring tool (PEP 669) with a single global event:
`PY_START`. Tool ids are allocated dynamically (preferring the unnamed ids 4
and 3, then any free id), so several monitors can coexist in one process --
e.g. the plugin running its own test suite. Every time a code block starts executing it records the repo
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
`site-packages`, `.tox`, `node_modules`, ...) — only first-party repository
code is ever recorded. When the plugin itself is installed inside the tested
repo (its own dev repo), its files are first-party code and are recorded:
the hooks and monitor callback execute during every test, so any plugin edit
conservatively re-runs everything. This is what makes the plugin safe to run
on its own test suite (`uv run pytest --less tests`).

**Data files** (on by default; opt out via `--less-no-data-files` or
`less_data_files = false`): a
`sys.addaudithook` subscription on the `open` event records every in-repo
non-Python file opened while the monitor is active — fixtures, templates,
profiles, `.proto`/`.dat`/`.sql` assets — bucketed exactly like code
dependencies (per test during a test, shared import-time bucket before
collection). They are stored as file-level wildcard deps: any content change
re-runs the opener(s). Without this flag, editing a data file that only
tests read selects nothing — a red suite with a green CI. Trade-offs: "opened"
≠ "depended on" (tests that open files speculatively widen, in the safe
direction); Python source/bytecode opens are excluded on purpose (they are
tracked at block precision by `PY_START`, and a wildcard dep on them would
throw that precision away); `.so` loading via `dlopen` emits no `open` event
and remains untracked.

**Subprocess boundaries (always on):** the same always-installed audit hook
also watches process-creation events (`subprocess.Popen`, `os.system`,
`os.fork`, `os.posix_spawn`, `pty.spawn`, ...). Coverage cannot follow a
child process, so when a test spawns one the monitor **stops collecting for
that test, writes no dependencies for it, and lists it in the shard's
`always` key: it re-runs every time** (reported as `N always re-run (spawn
subprocesses)`; a recorded spawner also rules out the no-change fast path).
The classic e2b blind spot — edit the child-only `worker.py`, watch the test
fail with "0 tests selected" — cannot happen. A test that stops spawning is
recorded normally again on its next run. A spawn during *collection* has an
unknown culprit: the shard is flagged `spawn_unknown` and falls back to the
weaker rule (every test runs when a changed `.py` file is stale or was never
recorded). Shards recorded before this feature behave the same until a full
record run heals them. Spawns during pytest's own startup are **not**
attributed — our own git subprocesses run in that phase, and attributing them
would mark every test a spawner.
**Reflection and dynamic code (always on):** whatever `getattr`/`setattr`/
`import_module` *reaches* still executes, so `PY_START` records it. What a
recording cannot know is what a **computed name** would reach next time. Two
runtime signals mark a test *dynamic*: code compiled from a string
(`eval`/`exec`/`compile`, generated classes — audit event `compile` with a
`<...>` filename) and reflective entry points (`importlib.import_module`,
`importlib.reload`, `runpy.run_*`, `pkgutil.walk_packages`/`iter_modules`,
`importlib.util.spec_from_file_location`, `importlib.metadata.entry_points`).
Both are only attributed during a test (module-level `namedtuple`/dataclass at
import time reaches no unseen edge). A dynamic test keeps its precise
dependencies, and is *also* selected whenever a changed `.py` file is stale or
was never recorded (reported as `N selected as runtime-dynamic`). The static
markers (below) still cover the constructs with no runtime signal
(`getattr`/`setattr` with computed names, `__getattr__`).

Child reads of *data* files are not tracked (data-file tracking covers
the pytest process only).

### 2. Storage: JSON shards in git refs (never a commit)

For every pytest **command** the data is stored in a dedicated ref:

```
refs/less/<shard>   →  lightweight commit
                            ├── data.json   {"version":2, "blocks":[[file,block,hash]...], "tests":{nodeid:[block-id...]}, "import_deps":[...], "failed":[...], "always":[...], "spawn_unknown":false, "dynamic":[...], "docs":{file:hash}, "doc_readers":[...]}
                            └── meta.json   {"version":1, "shard":..., "base_commit":..., "created":...}
```

- The shard key is a hash of the **normalized** invocation arguments
  (repo-relative paths, order- and duplicate-insensitive), so only a re-run
  of the same command reuses the data. Presentation-only flags are excluded
  from the key — `pytest tests`, `pytest tests -q` and `pytest tests -v
  --tb=short -x --durations 5` all share **one** shard (they collect and
  execute the same tests; early exit just makes a partial run, which the
  cumulative merge handles). Arguments that change the test set — positional
  paths, `-k`, `-m`, `--deselect`, `-p`, `-W` — and any **unknown** argument
  stay in the key: over-merging shards would risk stale selections, while
  under-merging only costs a fresh record run. `pytest tests/test_a.py` is
  therefore a different shard.
- The ref is updated with git plumbing (`hash-object` / `mktree` /
  `commit-tree` / `update-ref`); it never touches the working tree, `HEAD`,
  or any branch, and nothing is ever committed by normal `git commit`.
- Latest record wins: the ref always points at the most recent run's data.

Plain `git push`, `git fetch` and `git clone` do **not** transfer these refs.

#### Automatic sync (recommended)

Let the plugin do the sharing — onboarding is one line. It pulls the shard for
the current command from the remote **before** selection (a fresh clone
starts with your recorded dependencies and can deselect immediately instead
of a full record run) and pushes it after recording:

```toml
# pyproject.toml
[tool.pytest.ini_options]
less_sync = "origin"     # any git remote name
```

or per-invocation, without touching the repo config:

```bash
PYTEST_LESS_SYNC=1 pytest --less -q tests        # '1' means 'origin'
PYTEST_LESS_SYNC=upstream pytest --less -q tests # an explicit remote
```

Sync is bidirectional by default; `pull:<remote>` / `push:<remote>` select
one direction. `pull:` is the CI mode — see below.

Semantics and safety:

- "latest record wins": the shard with the newer `meta.json: created`
  timestamp is kept (local vs. remote, both directions);
- sync failures are **non-fatal** and only surface as notes in the run
  summary (offline, no remote configured, missing shard, auth refused);
  pushes force-push the tiny `refs/less/<shard>` only — never a branch;
- credentials are never prompted (`GIT_TERMINAL_PROMPT=0`) and remote
  operations time out after 30s, so a test run can never hang on them;
- `--less-clean` skips the pull, so "discard" really discards;
- the shard key is the *normalized pytest command*: the machine that
  recorded the data must run a command that normalizes to the same shard
  key (same positional paths and selection arguments; presentation flags
  like `-q`/`-v`/`--tb` don't matter) — relevant for CI reuse.

#### CI

The recommended topology needs **no write credentials in CI at all**:
developers push shards with their code (the runs that record the data), CI
only pulls them — using the same read-only access the checkout already has:

```bash
PYTEST_LESS_SYNC=pull:origin pytest --less -q tests
```

On GitHub Actions, the default token permissions suffice (nothing to
configure; `contents: read` is the default and `actions/checkout` already
authenticates `git fetch`):

```yaml
# no 'permissions:' block needed
- run: PYTEST_LESS_SYNC=pull:origin pytest --less -q tests
```

To also *push* recorded data back from CI (so fresh clones and other
machines benefit from CI runs), grant the built-in token write access:

```yaml
permissions:
  contents: write   # pushes refs/less/* only; never branches
```

Notes for the push direction: pushes made with `GITHUB_TOKEN` do not
trigger new workflow runs (documented GitHub behavior), and parallel jobs
racing on the same command resolve as latest-record-wins.

#### Manual sharing (explicit refspecs)

```bash
git push  origin 'refs/less/*:refs/less/*'
git fetch  origin 'refs/less/*:refs/less/*'
git ls-remote origin 'refs/less/*'   # what is on the remote?

# inspect a shard
git cat-file blob refs/less/<shard>:data.json | python -m json.tool
```

To make plain `git pull`/`git fetch` transfer shards automatically, add a
second fetch refspec:

    git config --add remote.origin.fetch '+refs/less/*:refs/less/*'

For push, note that setting `remote.origin.push` *replaces* the default
branch-pushing behavior (it does not merge with it), so include both
refspecs:

    git config --add remote.origin.push 'refs/heads/*:refs/heads/*'
    git config --add remote.origin.push 'refs/less/*:refs/less/*'

or keep it explicit with an alias:

    git config alias.push-less "push origin refs/less/*:refs/less/*"

Host support: verified against plain/self-hosted git servers and against
github.com itself (a `refs/less/*` ref pushed, listed via `ls-remote`,
and fetched back intact). If some host or push-ruleset policy rejects
custom refs, the push simply fails non-fatally (a note in the run summary)
and everything stays local — pull-only CI is unaffected either way. This
repository dogfoods the whole loop in CI (`.github/workflows/ci.yml`).

### 3. Selection: git diff + collection skipping

Before collection starts, the changed-file set is computed and test files
whose recorded tests are **all** unaffected are not even *collected* (their
modules are never imported) — done via
``pytest_ignore_collect``.  This is safe because a test always depends on
its own module file: any edit (including adding a test) selects the file's
tests, and a brand-new file has no recorded tests and is always collected.
(Files given as explicit pytest *arguments* are always collected anyway:
pytest exempts command-line paths -- "init paths" -- from
``pytest_ignore_collect``.)

When explicit test selectors are in play (``-k``, ``-m``, ``--lf``/``--ff``,
or a ``file.py::test`` argument), selection is disabled entirely: an explicitly requested but
unaffected test must never be silently skipped.

`changed = (files differing between the shard's base commit and the current
working tree) ∪ (untracked files)`. A test runs when:

- it has **no** recorded data (new test, or the previous run never finished
  it — e.g. it was interrupted),
- it **failed** in the recorded run, or
- one of its dependencies is in `changed` (a *stale block*: the file's
  current block table no longer has the recorded content hash),
- one of the **shared import-time** dependencies went stale (that is the
  whole-suite fan-out for conftest/import-time edits — block-granular:
  only blocks that actually executed at import time count),
- a changed file **containing a dynamic marker** (computed `getattr`,
  `eval`/`exec`, `__getattr__` definitions, ...) also has a stale block:
  the file is widened to file level and every test that touched it re-runs
  (reported as `N widened by dynamic markers`) — reflective edges are the
  one place a recording can disagree with the next run, so this widens on
  purpose. The staleness gate keeps cosmetic edits from widening, and
- the test is a **spawner** (`always`): it runs every time (`N always re-run`);
  under `spawn_unknown`, every test runs when a changed `.py` file is stale
  or was never recorded (`N selected by the subprocess rule`),
- the test is **runtime-dynamic** and some changed `.py` file is stale or
  was never recorded (`N selected as runtime-dynamic`), or
- the test ran a **docstring reader** and a file it touched has changed
  docstrings (`N selected by docstring readers`, see cosmetic edits below).

Every widening is printed in the run summary — selection never widens
silently. **Cosmetic edits** (on by default): block hashes are computed from a
docstring-stripped AST, so a comment or docstring-only change produces no
stale block and selects nothing (comments and formatting are invisible to the
AST regardless; annotations are kept — typing changes are observable).
Docstrings are only *prose* for code that never reads them, so each shard also
records a per-file docstring hash and the files that read docstrings as data
(`__doc__`, `getdoc`, `cleandoc`, `doctest`, `pydoc`, `help`): a docstring-only
edit still re-runs the tests that touched the changed file **and** ran a
docstring reader. The filter turns itself off under `--doctest-modules` /
`--doctest-glob` (docstrings are test input); `--less-no-cosmetic` /
`less_cosmetic = false` opts out manually.

- Dependencies are code blocks; precision is bounded by the block unit — a
  hot, branchy method fans out to every test that executed *any* part of it
  (an edit inside one branch re-runs tests exercising another). Deliberate:
  finer units need line-level recording, which is the storage curve that
  killed line-level formats (IMPROVEMENTS item 11 has the full analysis).
- Spawners always re-run — the price of fail-safe: a suite whose tests
  shell out heavily (this repo's own pytester tests: ~half the suite) gets
  little skipping from them. Spawns made by a conftest *at startup* are not
  tracked (their in-process effects still are, via the import-time bucket),
  and child reads of data files are not tracked (data-file tracking covers
  the pytest process only). Monitoring inside the child would be the
  precise version; it is not built.
- The dynamic-marker widening only considers changed files with a *stale
  recorded block*; it cannot widen edits for which no recording exists at
  all (nothing stale → nothing widened — the causal-closure argument).
- The cosmetic filter's docstring-reader net is static and file-level: a
  library that reads docstrings on your behalf (e.g. a CLI framework building
  help text from `__doc__`) is invisible to it — opt out with
  `--less-no-cosmetic` if your tests depend on that.
- Reflection with no runtime signal (`getattr`/`setattr` with a computed
  name, `__getattr__`) is only covered by the static markers: a stale block in
  a marker-bearing file widens to the whole file.
- Thread attribution: code executed by threads spawned by a test is
  attributed to the test running at that moment; tests are assumed to run
  sequentially.
- Session-fixture teardowns that run after the last test may be attributed
  to that test.
- Line-level coverage is intentionally not tracked.
- Non-Python data files are tracked by default (opt out with
  `--less-no-data-files`);
  without it, a change to a data file selects nothing. `dlopen`ed `.so`
  libraries emit no `open` event and are never recorded.

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
src/pytest_less/
├── plugin.py      # pytest hooks, lifecycle, options
├── monitoring.py  # sys.monitoring tool: PY_START + import bytecode scan
├── gitstore.py    # JSON shards behind refs/less/<shard>
├── selection.py   # shard parsing + test selection
└── util.py        # path normalization, command shard key
```
<!-- dogfood demo: a change no test depends on -->
