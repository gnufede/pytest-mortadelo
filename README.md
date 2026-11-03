# pytest-mortadelo

**Block-level test impact analysis (TIA)** for pytest.

**MORTADELO** — **M**ethod-level **O**ptimized **R**untime **T**racer
**A**nd **D**ependency **E**liminator for **L**azy **O**ptimization — and
of course he works at the T.I.A. (here: Test Impact Analysis), with a
disguise for every code block.  (The elimination is the point: he
eliminates the tests that do not need to run, so the lazy among us
optimize by not running them at all.)  His file-level partner in crime is
[pytest-filemon](https://github.com/gnufede/pytest-filemon) (import-only
dependencies, works on any Python).

| | pytest-mortadelo |
|---|---|
| dependency storage | local JSON under `.pytest_mortadelo/` (not versioned; copy the directory to share it) |
| dependency resolution | **code blocks** (qualified name + hash of the compiled bytecode) and module skeletons for imports |
| dependency collection | a custom `sys.monitoring` tool using **PY_START / PY_RESUME** events (no per-line tracking) |
| dynamic imports | conservative bytecode scan for import opcodes inside every executed code block |
| import-time dependencies | everything imported before tests start is a dependency of *every* test |
| change detection | **file fingerprints** (size, mtime, content hash) — git only for new-file questions; `git diff` is the fallback |

Requires **Python 3.12+** (for `sys.monitoring`) and a git repository. Blocks are
hashed on 3.12 – 3.15; on a newer interpreter every file is a file-level
dependency (see below) until the hasher is validated for it.

## Usage

```bash
pip install pytest-mortadelo

pytest --mortadelo              # first run: everything runs, dependencies are recorded
pytest --mortadelo              # second run: only tests affected by changes run
```

Options:

| option | effect |
|---|---|
| `--mortadelo` | enable recording + selection |
| `--mortadelo-force` | run all tests, but still record dependencies |
| `--mortadelo-clean` | discard the stored data before running |
| `PYTEST_MORTADELO=1` | same as `--mortadelo` (env form; also enables before conftest loading) |
| `-p no:mortadelo` | disable the plugin entirely |

If nothing changed and nothing failed, **zero tests run and pytest exits
with code 0**: deselecting everything is a successful selection, not an
error. (A genuinely empty collection — no tests at all — keeps pytest's own
exit code 5; a run under `-k`/`-m`, where mortadelo never deselects, keeps
whatever pytest decides.)

## How it works

### 1. Dependency collection (`sys.monitoring`, PY_START / PY_RESUME)

The plugin registers a monitoring tool (PEP 669) with two global events:
`PY_START` (a code block starts executing) and `PY_RESUME` (a suspended one
resumes). Tool ids are allocated dynamically (preferring the unnamed ids 4
and 3, then any free id), so several monitors can coexist in one process --
e.g. the plugin running its own test suite. Every time a code block starts --
or, after a suspension, resumes: a generator or coroutine primed by one test
runs its code in whichever test's window resumes it, and that test depends on
the block (on sqlalchemy, 186 tests consume class-scoped yield fixtures this
way) -- it records the **block** it starts: a
`(repo-relative file, block id, content hash)` triple. On top of that, for
every code block that starts, the plugin scans its bytecode for
`IMPORT_NAME` / `IMPORT_FROM` opcodes and resolves the referenced modules to
file paths. This is the conservative treatment of dynamic imports: an
`import` inside a branch that never runs is still recorded as a dependency of
the test that executed the surrounding block. Relative imports are resolved
against the importing file's package, found the way the module index names
files (walking *up* from the file while each directory is a package, so
`src/` and `lib/` layouts work as well as a package at the repository root).
Resolution never imports anything: a dotted name whose parent package is not
already loaded is not looked up. Unresolvable names are skipped, and this
matters more than it sounds: only the **first** test to import a module runs
its module code and records it through that module's own `PY_START`; every
later test gets a `sys.modules` hit, so for them this scan is the only record
of the edge. A module imported through `importlib.import_module("name")` is
covered by the same scan plus a wrap around `import_module` itself, so even a
*cached* import (a plain function call; no code runs, no event fires) records
the module's skeleton for the caller.

**Fixture closure (static, like the import scan):** a fixture with scope
wider than `function` runs its setup once, in the first requesting test's
window — later consumers receive the cached value without executing any of
its code, so events alone cannot see their dependency. pytest knows the
closure at collection time (`item.fixturenames`), and the block of every
>function-scoped fixture a test requests is recorded as that test's
dependency. On sqlalchemy, a one-line edit to the autouse class-scoped
`setup_class_methods` testing fixture selected 425 of 2,480 tests before
this rule and all 2,480 with it: the other 2,055 all consume its value.

**Memoized functions (`functools.lru_cache` / `functools.cache`):** a cached
call consumes the original's result without executing it, so `PY_START`
never fires on later calls. At decoration time the plugin pre-computes the
decorated function's block and every call — hit or miss — records that edge.
Scoped to repo functions only: stdlib and third-party cached functions keep
the untouched C wrapper, at zero overhead.

Events are bucketed by phase:

- **import phase** (everything before collection: pytest plugins, conftest
  imports) → dependencies of **every** test. Modules imported *before* the
  monitor arms — pytest loads its plugins (entry points, `-p`) before the
  earliest hook a plugin can use — have no events left to see, so at arming
  every already-imported repo module's skeleton is added to this bucket
  instead (functions called later are still block-precise). Changing a
  conftest re-runs the world.
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
on its own test suite (`uv run pytest --mortadelo tests`).

**Subprocess boundaries (always on):** the functions that create processes
are wrapped (`subprocess.Popen`, `os.system`, `os.fork`/`forkpty`,
`os.posix_spawn`/`posix_spawnp`, `os.exec*`, `_posixsubprocess.fork_exec` —
which also covers `multiprocessing`). Coverage cannot follow a
child process, so when a test spawns one the monitor **stops collecting for
that test, writes no dependencies for it, and lists it in the data's
`always` key: it re-runs whenever anything changed** (reported as `N always
re-run (spawn subprocesses)`); a rerun with *provably nothing changed*
(no dependency file, no new project file, same HEAD, nothing failed) skips
them like everything else via the no-change fast path. The classic e2b blind
spot — edit the child-only `worker.py`, watch the test fail with "0 tests
selected" — cannot happen. A test that stops spawning is recorded normally
again on its next run. Spawns during pytest's own startup are **not**
attributed — our own git subprocesses run in that phase, and attributing
them would mark every test a spawner.

**Reflection and computed dispatch:** whatever `getattr`/`setattr`/
`import_module` *reaches* still executes, so `PY_START` records it. A
computed `getattr(self, "handle_" + name, default)` needs no rule of its
own: it can only resolve to something in a namespace some code built — a
class body, a module body, or a registry filled by executed functions — and
each of those is a block a test already depends on (class and module
skeletons run at import, so they are dependencies of every test). The one way
such a lookup can start resolving to something new, *adding* a definition,
always changes a block the test depends on and selects it. What stays
uncovered is dispatch that depends on state the plugin does not track (the
environment, time, randomness, installed packages) — the same as for any test
impact analysis. String-compiled code (`eval`/`exec`) is deliberately **not**
a signal: the generated code comes from generators that are themselves
recorded, and flagging it selected 69% of sqlalchemy's ORM tests on every
edit. The active configuration file (`pytest.ini` / `pyproject.toml` /
...) is claimed as a shared dependency: a config change selects everything.
`docs/correctness.md` has verified code samples of all of these.

#### Block hashes: compiled bytecode

A block is hashed from **`compile(source)`** — a C call — not from a Python AST.
The 220 files (8.8 MB) a sqlalchemy record run touches take **0.7 s** through
`compile` + a digest of each code object (0.53 s of that is the compiler
itself); the AST this replaced took 2.4 s, and a cold record of the 2,480-test
subset is +4.0 s over plain pytest instead of +5.8 s. The code hashed is always
the file *on disk*, never the running code object: pytest rewrites asserts in
test modules and other import hooks transform code, so live bytecode cannot be
reproduced from source at selection time, but `compile(source)` can.

What the hash does and does not see:

- **Layout, comments and line numbers are invisible.** CPython 3.13+ injects
  `__firstlineno__` (the class's own line) and `__static_attributes__` (the
  attributes assigned in *method bodies*) into every class body; both are masked,
  or adding a line above a class would change it, and adding `self.x = 1` to a
  method would change the skeleton every importer depends on. Constants of class
  and module bodies are hashed by *value*, because an inserted constant renumbers
  every later `LOAD_CONST`.
- **What the compiler discards is invisible** (`x + 0`, local-variable
  annotations, the text of a no-op string statement, `import a as a`): equal
  bytecode is equal behavior. The reverse costs only selection precision —
  adding a dead branch leaves its constant in the table, so it over-selects.
- **A class or module hash is a skeleton**: nested `def` bodies are replaced by
  their signature, everything else is included. Decorators, defaults and
  annotations are evaluated by the *parent*, so editing one changes the parent's
  block (always executed when the function object is created), not the
  function's own.
- **Docstrings are behavior.** Editing one selects the tests that executed that
  block; a class or module docstring changes the skeleton, so it selects
  everything that imported the module.
- **Lookup is exact**: a running code object is matched by
  `(co_qualname, co_firstlineno)`; lambdas and comprehensions belong to their
  *lexical parent*, not to whichever block's source span contains the line, and
  a class declared `global` inside a function is found under its real qualname.
  Of 15,450 sqlalchemy code objects, 101 resolved differently from the AST
  approach this replaced, and all 101 were cases where it had guessed.
- **3.14 and 3.15 changed what a code object looks like**, and each change was
  found by a test rather than by reading release notes:
  - *Annotations became code* (PEP 649): every annotated `def`, class or module
    gets a lazily-run `__annotate__` function, a block of its own. Editing an
    annotation changes that block and nothing else, so it selects exactly the
    tests that **evaluated** it (`get_type_hints`, dataclasses, ...) — more
    precise than before, when the enclosing scope evaluated annotations eagerly
    and the whole skeleton changed. Adding or removing an annotation still
    changes the enclosing block. Decorators and defaults are unchanged.
  - *Small ints* are loaded with `LOAD_SMALL_INT` (the operand is the value), so
    the class line number the compiler injects is no longer a constant. It is
    masked in both forms, and in one canonical form: it switches from
    `LOAD_SMALL_INT` to `LOAD_CONST` at line 256, which otherwise changed a
    class that merely crossed that line.
  - *Import statements:* 3.14 loads the relative-import level with
    `LOAD_SMALL_INT`; 3.15 (PEP 810) packs lazy/eager flags into the low two
    bits of `IMPORT_NAME`'s operand, so `co_names[arg]` is the wrong entry and
    the import silently vanished. A `lazy import` is recorded like any import.
  - Folded arithmetic leaves its operand behind in `co_consts` (unreferenced),
    so adding `x + (0 + 1)` over-selects on 3.14+ (the safe side).

  Validated per version: the suite, ~850 real files re-hashed with every line
  number shifted (no hash may change), exact resolution of every `def`/class,
  5,000 random mutations of sqlalchemy functions (none missed that changes
  behavior), and the sqlalchemy subset recorded and re-selected with the same
  counts as 3.13.
- **Hashes are tagged `c<minor>`** (`c13…`): bytecode differs between Python
  versions, so data recorded on 3.12 never matches 3.13 — it reads as
  "changed", never as "unchanged" (an AST hash would not have been portable
  either: 0 of 14,000 sqlalchemy block hashes match between 3.12 and 3.13,
  since `ast.dump` itself changed).
- **What the hasher does not model is a file-level dependency**: files using
  PEP 695 generics or `type` statements, unparseable files, and every file on an
  interpreter newer than 3.15. Any change to such a file re-runs the tests
  that executed it (the `*` wildcard block); nothing is ever skipped because the
  hasher was unsure.

#### What recording costs, and where it goes

On the sqlalchemy ORM subset (2,480 tests; plain pytest ≈ 9–10 s) a record run
adds **+4.0 s**. Layers measured by ablation (variants of the same run with
parts switched off); the last row is what remains by subtraction:

| layer (2.4M events) | cost |
|---|---|
| CPython itself: dispatch + re-instrumenting after each `restart_events()` (a no-op Python callback, nothing else) | ~1.1 s |
| the callback's lookup | ~0.4 s |
| first-start analysis of 11.6k code objects | ~0.2 s (was 0.6); parsing + hashing the files: 0.7 s (was 2.4 s with the AST) |
| bucket update, `finish` (0.35 s), hooks and wrappers (by subtraction) | ~1.3 s |

What was done, in order of effect:

- **Lookup by `id(code)`, not by the code object** (−0.7 s). `dict.get(code)`
  *hashes the code object*, which walks its constants, names and nested code:
  226 ns on average on real sqlalchemy code, 2–5 µs for the heavy functions
  that run in nearly every test (`assert_compile`, `visit_insert`), against
  37 ns for the int. The cell keeps the code object alive, so an id is never
  reused.
- **A held reference to the current bucket's set** (−0.2 s). The callback does
  `sink.update(deps)`; the old `_bucket()` was a method call with phase checks
  and a `setdefault(nodeid, set())` that allocated a throwaway empty set on
  every event.
- **Byte scan for imports instead of `dis`** (−0.65 s). The analysis looks
  for `IMPORT_NAME` / `IMPORT_FROM` to over-approximate import dependencies;
  `dis.get_instructions` decoded whole module-level code objects in pure Python
  (~1 ms each) to find a few instructions. The scan finds the two opcodes in the
  raw bytecode (cost ∝ number of imports) and defers to `dis` for anything that
  is not the plain `level; fromlist; IMPORT_NAME` shape, and on Python 3.12. It
  agrees with `dis` on every import instruction of the 3.13, 3.14 and 3.15
  standard libraries and of this repository (the byte-level path is limited to
  those versions).
- `finish`: file sets derived from the few thousand distinct blocks instead of
  1.6M generator steps (−0.08 s); compact JSON and a sorted block table.

What did **not** help, so nobody has to try it again:

- Instrumenting only repo code with *local* events (so the third of callbacks
  that are stdlib/pytest never fire): prototyped, saves 0.4 s (−13%), and a
  missed code object would silently lose a dependency. Rejected.
- Attributing at module granularity or sampling tests: unsound or loses the
  per-test precision that is the point.
- Re-arming selectively: `restart_events()` is global and `DISABLE` cannot be
  undone per location, so the fire count (≈ 985 distinct code objects per test)
  is the floor of an event-driven design.
- Batching git object reads in-process (when data lived in git refs): 10–30 ms
  on tiny suites, nothing measurable on large ones — not worth a second code
  path. The git-object layer was removed entirely.
- pytest's capture machinery looked guilty in a sampling profile (`suspend` /
  `resume` absorbed 38% of the extra samples); the tax is identical with
  `-s`, `--capture=sys` and `--capture=fd` — a sampling artifact (syscalls
  release the GIL).

The dependencies recorded are unchanged by all this: re-recording with the old
and new code (hash seed fixed, GC off) agrees on 1.64M (test, block) pairs to
within the run-to-run noise of the *same* code (6 vs 7 tests of 2,480). With the
GC on, which test a weakref/`gc` callback lands in (`_remove_linecache_entry`,
`_collection_gced`, …) depends on allocation timing, so it moves between runs
of any version.

### 2. Storage: local JSON under `.pytest_mortadelo/`

```
.pytest_mortadelo/
├── data.json   {"version":2, "blocks":[[file,block,hash]...], "tests":{nodeid:[block-index...]}, "import_deps":[...], "failed":[...], "always":[...]}
├── meta.json   {"version":1, "base_commit":..., "created":..., "failed":[...], "nodeids":[...], "dep_files":[...]}
└── files.json  {"v":1, "at":ns, "files":{file:[size, mtime_ns, hash]}, "dirs":{dir:[subdirs]}}
```

One dataset for the whole repository — every `--mortadelo` run records into the same
files (selectors like `-k`/`-m`/explicit node ids merely disable selection for
that run). Each file is replaced atomically (`write to .tmp` + `rename`), so a
reader never sees a torn file; `rm -rf .pytest_mortadelo` resets the cache, and
copying the directory to another checkout shares the data (fingerprints are
content-keyed, so a different machine's mtimes do not invalidate anything).

The plugin uses git only for what JSON cannot answer: discovering the
repository root, the current `HEAD`, whether a *new* file is a project file
(one `git ls-files` scoped to the candidate paths), and — when some recorded
test spawns processes — the full tracked-file list, so a change anywhere a
child could read counts as a change. Data recorded before `files.json`
existed falls back to `git diff <base_commit>`. A handful of constant git
calls per run, never per file or per test.

**File fingerprints (`files.json`).** What selection needs is narrower than
"what differs from a commit" — *do the files this data depends on still look
like they did when it was recorded?* — so the data stores a snapshot:
`[size, mtime_ns, content hash]` for every dependency file and every `.py`
file in a **watched** directory (each directory holding a dependency, and
all its ancestors up to the root), plus those directories' subdirectory
names. A run `stat`s the recorded files and lists the watched directories
(26 ms for 3,250 files on a 10k-file repository):
- same size, same mtime, and an mtime older than the snapshot by more than 2 s
  → unchanged; anything else is settled by hashing the bytes. A `touch`, a
  checkout that rewrites identical content, or data recorded on another
  machine (every mtime differs) is **not** a change, and a file modified
  inside the racy 2 s window is never trusted on stat alone (git's
  "racily clean" rule).
- a `.py` file that is not in the snapshot, or a subdirectory that is not in
  the listing, is *new*. Only then is git asked — once, about those paths
  alone (`ls-files --cached --others --exclude-standard -- <paths>`), so
  ignore rules stay git's job while the cost no longer scales with the size
  of the checkout. Hidden directories, `__pycache__`, `*.egg-info` and the
  usual environment directories are never reported as new.
- it needs no base commit: rebasing or amending does not invalidate the data,
  a file that is **untracked** is no longer "changed" on every run, and a
  file recorded **dirty** and later reverted is noticed (a diff against
  `HEAD` cannot see it).
- a dependency that changed **while the run was recording it** (its bytes on
  disk no longer match what the block analysis compiled) is stored with
  mtime `-1`, so the next run always re-hashes it.
- rewriting the data reuses the previous snapshot's hashes for files whose
  size and (old enough) mtime did not change, so it costs stats, not reads.
- A run that stops early (`-x`, `--maxfail`) marks the tests it selected but
  never ran as must-rerun: the snapshot is refreshed by that run, so a test
  that kept its old block hashes would otherwise look fresh.

### 3. Selection

After collection (test modules are always imported — their own tests depend
on them), each collected test runs when:

- it has **no** recorded data (new test, or the previous run never finished
  it — e.g. it was interrupted),
- it **failed** in the recorded run, or
- one of its dependencies is in `changed` (a *stale block*: the file's
  current block table no longer has the recorded content hash),
- one of the **shared import-time** dependencies went stale (the
  whole-suite fan-out for conftest/import-time edits — block-granular:
  only blocks that actually executed at import time count),
- the test is a **spawner** (`always`) and anything changed: it re-runs
  (`N always re-run`); a rerun with nothing changed at all skips it via the
  fast path.

When explicit test selectors are in play (`-k`, `-m`, `--lf`/`--ff`, or a
`file.py::test` argument), selection is disabled entirely: an explicitly
requested but unaffected test must never be silently skipped.

Every widening is printed in the run summary — selection never widens
silently.

- A qualified name defined more than once with different content
  (`@overload` stubs, property getter/setter pairs, conditional definitions)
  is split into per-definition blocks (`name~<hash>`) resolved by source line,
  so editing one definition — or an `@overload` stub that never executes —
  does not invalidate the others.
- Dependencies are code blocks; precision is bounded by the block unit — a
  hot, branchy method fans out to every test that executed *any* part of it
  (an edit inside one branch re-runs tests exercising another). Deliberate:
  finer units need line-level recording, which is the storage curve that
  killed line-level formats.
- Thread attribution: code executed by threads spawned by a test is
  attributed to the test running at that moment; tests are assumed to run
  sequentially.
- Session-fixture teardowns that run after the last test may be attributed
  to that test.
- Line-level coverage is intentionally not tracked.

## Known limitations

`docs/correctness.md` has verified code samples and the full analysis; the
short list:

- **State-dispatched tests** — the target chosen by an env var, the clock,
  randomness, an installed package — are the fundamental limit of any TIA;
  no wrapper can record a counterfactual.
- **Memoized calls** outside `functools.lru_cache`/`cache` (`cached_property`
  on shared instances, hand-rolled `if _CACHED is None` caches): only the
  first caller records the edge.
- **Entry points that reach code by computed name** (`runpy.run_module`,
  `pkgutil.walk_packages`, `importlib.util.spec_from_file_location`):
  what actually *executes* is recorded; a module they would load under a
  different name is not (only `importlib.import_module` is wrapped).
- **Non-Python inputs** (data files, config formats beyond the active pytest
  ini, `ctypes`/`dlopen` artifacts): nothing executes them, so they are not
  dependencies — editing a data file that only tests read selects nothing.
- **Directory listings**: a test depending on a directory's *contents* holds
  no dependency on what it listed.
- **Order-dependent tests** (module globals state, files one test writes and
  another reads): TIA assumes test independence; this is that assumption.
- **pytest-xdist**: unsupported — and safe: workers are spawned processes, so
  the spawner rule takes over.

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

### Live demo

`demo/sqlalchemy_demo.py` runs a live overhead + edit→deselect benchmark
against sqlalchemy's ORM suite (2,480 tests over 12 files), with only
comment or no-op edits — nothing breaks, everything is restored:

```bash
python3 demo/sqlalchemy_demo.py [path-to-sqlalchemy-checkout]
```

If the checkout has no `.venv*` virtualenv, the demo creates
`<checkout>/.venv` and installs sqlalchemy, pytest and this plugin
editable into it. Set `SQLALCHEMY_PYTHON` to use an existing interpreter.

Pytest output is streamed live, each edit is shown as a colored
`git diff`, and the demo waits for Enter between steps, so it can be
driven as a live walkthrough.

It shows the record overhead, the no-change fast path (0 tests, exit 0,
under a second), and where the deselect boundary sits — see the script
header for details.

## Layout

```
src/pytest_mortadelo/
├── plugin.py      # pytest hooks, lifecycle, options
├── monitoring.py  # sys.monitoring tool: PY_START/PY_RESUME + import bytecode scan
├── blocks.py      # block identity, per-run analysis cache
├── codeblocks.py  # bytecode hashing: file -> block table
├── fingerprint.py # [size, mtime_ns, hash] snapshots and change detection
├── selection.py   # stored-data parsing + test selection
├── store.py       # .pytest_mortadelo/ JSON storage
└── util.py        # path normalization
```
