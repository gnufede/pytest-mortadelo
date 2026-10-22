# pytest-less

Minimal **test impact analysis (TIA)** for pytest.

| | pytest-less |
|---|---|
| dependency storage | JSON shards stored in git refs |
| dependency resolution | **code blocks** (qualified name + hash of the compiled bytecode) and module skeletons for imports |
| dependency collection | a custom `sys.monitoring` tool using **PY_START / PY_RESUME** events (no per-line tracking) |
| dynamic imports | conservative bytecode scan for import opcodes inside every executed code block |
| import-time dependencies | everything imported before tests start is a dependency of *every* test |
| change detection | **file fingerprints** recorded with the shard (size, mtime, content hash) — no git in the common case; `git diff` is the fallback |

Requires **Python 3.12+** (for `sys.monitoring`) and a git repository. Blocks are
hashed on 3.12 – 3.15; on a newer interpreter every file is a file-level
dependency (see below) until the hasher is validated for it.

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
| `PYTEST_LESS=1` | same as `--less` (env form; also enables before conftest loading) |
| `-p no:less` | disable the plugin entirely |

If nothing changed and nothing failed, **zero tests run and pytest exits with
code 5** ("no tests collected"), just like a `pytest -k nomatch` run.

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
way) -- it records the repo file it lives in — file-level coverage, no
lines. On top of that, for every
code block that starts, the plugin scans its bytecode for `IMPORT_NAME` /
`IMPORT_FROM` opcodes and resolves the referenced modules to file paths. This
is the conservative treatment of dynamic imports: an `import` inside a branch
that never runs is still recorded as a dependency of the test that executed
the surrounding block. Relative imports are resolved against the importing
file's package, found the way the module index names files (walking *up* from
the file while each directory is a package, so `src/` and `lib/` layouts work
as well as a package at the repository root). Resolution never imports
anything: a dotted name whose parent package is not already loaded is not
looked up. Unresolvable names are skipped, and this matters more than it
sounds: only the **first** test to import a module runs its module code and
records it through that module's own `PY_START`; every later test gets a
`sys.modules` hit, so for them this scan is the only record of the edge.

**Fixture closure (static, like the import scan):** a fixture with scope
wider than `function` runs its setup once, in the first requesting test's
window — later consumers receive the cached value without executing any of
its code, so events alone cannot see their dependency. pytest knows the
closure at collection time (`item.fixturenames`), and the block of every
>function-scoped fixture a test requests is recorded as that test's
dependency. On sqlalchemy, a one-line edit to the autouse class-scoped
`setup_class_methods` testing fixture selected 425 of 2,480 tests before
this rule and all 2,480 with it: the other 2,055 all consume its value.

Events are bucketed by phase:

- **import phase** (everything before collection: pytest plugins, conftest
  imports) → dependencies of **every** test. Modules imported *before* the
  monitor arms — pytest loads its plugins (entry points, `-p`) before the
  earliest hook a plugin can use — have no events left to see, so at arming
  every already-imported repo module's skeleton is added to this bucket
  instead (functions called later are still block-precise). Changing a
  conftest re-runs the
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
`less_data_files = false`): wrapping
`open()` (`builtins.open` / `io.open`) records every in-repo
non-Python file opened while the monitor is active — fixtures, templates,
profiles, `.proto`/`.dat`/`.sql` assets — bucketed exactly like code
dependencies (per test during a test, shared import-time bucket before
collection). They are stored as file-level wildcard deps: any content change
re-runs the opener(s). Without this tracking, editing a data file that only
tests read selects nothing — a red suite with a green CI. Trade-offs: "opened"
≠ "depended on" (tests that open files speculatively widen, in the safe
direction); Python source/bytecode opens are excluded on purpose (they are
tracked at block precision by `PY_START`, and a wildcard dep on them would
throw that precision away); files opened through `FileIO`, `mmap` or a C
extension (including `.so` loading via `dlopen`) are not seen and remain
untracked. `builtins.open`, `io.open` and `os.open` are wrapped. **Outputs are
not inputs:** a file a test only *writes* (`open` modes `w`/`x`/`a` without
`+`; `os.open` with `O_WRONLY`, `O_TRUNC` or `O_CREAT|O_EXCL`) is never a
dependency, and neither is reading it back later in the same run. Once such a
file exists it is untracked — git lists it as changed on every run — so
recording it would make the test re-run forever. `r+`, `a+` and `O_RDWR` stay
dependencies (they can read pre-existing content). No
`sys.addaudithook` is used anywhere: an audit hook is a
Python call for *every* audit event in the process, and `builtins.id` alone
fires 22M times in sqlalchemy's ORM suite (~7s of pure dispatch for a hook
that cared about under a thousand events), whereas wrapping the few functions
involved costs nothing until they are called.

**Subprocess boundaries (always on):** the functions that create processes
are wrapped (`subprocess.Popen`, `os.system`, `os.fork`/`forkpty`,
`os.posix_spawn`/`posix_spawnp`, `os.exec*`, `_posixsubprocess.fork_exec` —
which also covers `multiprocessing`). Coverage cannot follow a
child process, so when a test spawns one the monitor **stops collecting for
that test, writes no dependencies for it, and lists it in the shard's
`always` key: it re-runs whenever anything changed** (reported as `N always
re-run (spawn subprocesses)`) — *unless* the state is **pristine**, in which
case "no change" beats "always run" and it is skipped like everything else
(see below).
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
`import_module` *reaches* still executes, so `PY_START` records it. The one
thing a **computed name** can reach that a recording never saw is a *new*
module. A test that went through a reflective entry point
(`importlib.import_module`, `importlib.reload`, `runpy.run_*`,
`pkgutil.walk_packages`/`iter_modules`, `importlib.util.spec_from_file_location`,
`importlib.metadata.entry_points`) is marked *dynamic* (a `PY_START`
pseudo-dependency; only during a test) and is also selected whenever a changed
`.py` file was **never recorded** (reported as `N selected as
runtime-dynamic`). A *stale* recorded file does not widen: what the test
executed of it is already a dependency, and what it did not execute cannot
change its result. String-compiled code (`eval`/`exec`) is deliberately **not**
a signal: the generated code comes from generators that are themselves
recorded, and flagging it selected 69% of sqlalchemy's ORM tests on every
edit.

A computed `getattr(self, "handle_" + name, default)` needs no rule of its own.
It can only resolve to something in a namespace some code built — a class body,
a module body, or a registry filled by executed functions — and each of those
is a block a test already depends on (class and module skeletons run at import,
so they are dependencies of every test). The one way such a lookup can start
resolving to something new, *adding* a definition, therefore always changes a
block the test depends on and selects it; `tests/test_reflection.py` pins this
for methods, module functions, subclass overrides and registries. A static
"computed `getattr` in this file" scan used to widen every change in such a file
to every test that touched it; on sqlalchemy that made one-function edits run
the whole suite (`Query.filter`: 2,480 of 2,480 tests, against 338 without) while
guarding nothing those rules miss. What stays uncovered is dispatch that depends
on state the plugin does not track (the environment, time, randomness,
installed packages), the same as for any test impact analysis — and
memoized calls (`functools.lru_cache`, hand-rolled caches): only the first
caller records the edge, a later one with the same arguments executes
nothing. `docs/correctness.md` has verified code samples of both.

Child reads of *data* files are not tracked (data-file tracking covers
the pytest process only).

#### Block hashes: compiled bytecode

A block is hashed from **`compile(source)`** — a C call — not from a Python AST.
The 220 files (8.8 MB) a sqlalchemy record run touches take **0.7 s** through
`compile` + a digest of each code object (0.53 s of that is the compiler
itself); the AST this replaced took 2.4 s, and a cold record of the 2,480-test
subset is +4.0 s over plain pytest instead of +5.8 s. The code hashed is always
the file *on disk*, never the running code object: pytest rewrites asserts in
test modules and other import hooks transform code, so live bytecode cannot be
reproduced from source at selection time, but `compile(source)` can.

What the hash does and does not see, each point pinned by `tests/test_codeblocks.py`:

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
  block; a class or module docstring changes the skeleton, so it selects everything
  that imported the module. Ignoring prose needs a per-block docstring hash and a
  "reads `__doc__`" safety net to stay sound, and on a codebase that reads
  docstrings itself it buys nothing: on sqlalchemy a method docstring edit
  selected 1,185/2,480 tests and a class or module docstring edit 2,480/2,480
  *with* the filter. The cost falls on codebases that never read docstrings.
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
  versions, so a shard recorded on 3.12 never matches 3.13 — it reads as
  "changed", never as "unchanged" (an AST hash would not have been portable
  either: 0 of 14,000 sqlalchemy block hashes match between 3.12 and 3.13, since
  `ast.dump` itself changed).
- **What the hasher does not model is a file-level dependency**: files using
  PEP 695 generics or `type` statements, unparseable files, and every file on an
  interpreter newer than 3.15. Any change to such a file re-runs the tests
  that executed it (the `*` wildcard block); nothing is ever skipped because the
  hasher was unsure.
Validation (not just unit tests, and done against the AST hasher this replaced):
on 1,498 real sqlalchemy file edits (179,000
blocks) the two hashers agreed on 99.5% of blocks. Where the AST says "changed"
and bytecode says "same" (853 blocks), 842 are explained by a changed ancestor
(the parent rule) and the other 11 by edits that cannot change behavior; where
bytecode says "changed" and the AST says "same" (8 blocks), the cause is
`module.func()` compiling differently once the module-level `import` changed.
And 5,000 random mutations (constants, operators, dropped statements, renamed
arguments, new decorators, …) of real functions: every one that is not
provably equivalent changes the block or an ancestor.

#### What recording costs, and where it goes

On the sqlalchemy ORM subset (2,480 tests; plain pytest ≈ 9–10 s) a record run
adds **+2.3 s with a warm parse cache and +4.0 s cold** (it was +4.5 s and
+7.5 s before the work below and the bytecode hashing above). Layers measured by ablation (variants of the
same run with parts switched off); the last row is what remains by subtraction:

| layer (warm, 2.4M events) | cost |
|---|---|
| CPython itself: dispatch + re-instrumenting after each `restart_events()` (a no-op Python callback, nothing else) | ~1.1 s |
| the callback's lookup | ~0.4 s |
| first-start analysis of 11.6k code objects | ~0.2 s (was 0.6); a cold parse of the files: 0.7 s (was 2.4 s) |
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
- **Byte scan for imports instead of `dis`** (−0.65 s warm). The analysis looks
  for `IMPORT_NAME` / `IMPORT_FROM` to over-approximate import dependencies;
  `dis.get_instructions` decoded whole module-level code objects in pure Python
  (~1 ms each) to find a few instructions. The scan finds the two opcodes in the
  raw bytecode (cost ∝ number of imports) and defers to `dis` for anything that
  is not the plain `level; fromlist; IMPORT_NAME` shape, and on Python 3.12. It
  agrees with `dis` on every import instruction of the 3.13, 3.14 and 3.15
  standard libraries and of this repository (the byte-level path is limited to
  those versions).
- `finish`: file sets derived from the few thousand distinct blocks instead of
  1.6M generator steps (−0.08 s); compact JSON and a sorted block table, so
  `data.json` compresses to 0.70 MB instead of 0.94 MB.

What did **not** help, so nobody has to try it again:

- Instrumenting only repo code with *local* events (so the third of callbacks
  that are stdlib/pytest never fire): prototyped, saves 0.4 s (−13%), and a
  missed code object would silently lose a dependency. Rejected.
- Attributing at module granularity or sampling tests: unsound or loses the
  per-test precision that is the point.
- Re-arming selectively: `restart_events()` is global and `DISABLE` cannot be
  undone per location, so the fire count (≈ 985 distinct code objects per test)
  is the floor of an event-driven design.
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

### 2. Storage: JSON shards in git refs (never a commit)

For every pytest **command** the data is stored in a dedicated ref:

```
refs/less/<shard>   →  lightweight commit
                            ├── data.json   {"version":2, "blocks":[[file,block,hash]...], "tests":{nodeid:[block-id...]}, "import_deps":[...], "failed":[...], "always":[...], "spawn_unknown":false, "dynamic":[...]}
                            ├── meta.json   {"version":1, "shard":..., "base_commit":..., "created":...}
                            ├── files.json  {"v":1, "at":ns, "files":{file:[size, mtime_ns, hash]}, "dirs":{dir:[subdirs]}}
                            └── parse.json  {"v":3, "py":"3.13", "files":{file:{"h":content-hash, "b":{block:hash}, "s":[[first-line, block]...], "m":markers-or-null}}}
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
- **Parse cache (`parse.json`).** `git diff` tells us which files changed for
  *staleness*, but resolving an executed code object to a block needs the
  block table of every file whose code runs — changed or not — and an
  incremental run used to re-parse hundreds of untouched files (about half the
  wall time of a run that selects nothing). `parse.json` stores each
  dependency file's analysis keyed by the **hash of its bytes**: reading and
  hashing a file costs ~0.1 ms against ~10 ms to parse and hash its blocks.
  Every entry is verified against the current bytes on every use, so a stale
  entry is just a miss — whatever the working tree looked like when it was
  written (dirty, reverted, another branch). The blob is only valid for the
  same Python minor version and analysis version
  (`PARSE_CACHE_VERSION`, bumped whenever hashing changes), is loaded before
  the monitor is armed (never on the no-change fast path) so import-time
  parses already hit it, and is carried over by object id when nothing new was parsed. Because it
  lives in the shard it travels with `pull`/`push`. Cost: about 0.5 MB compressed per shard on a 220-file
  dependency set. A cold first record parses everything, as before.

- **Git spawn budget.** Every git call is a process spawn — about 10 ms from a
  small process, several times that from inside a large pytest process, and a
  quarter of a second where endpoint scanning slows `exec`. The plugin reads and
  writes the shard through a handful of plumbing commands (`rev-parse`,
  `cat-file`, `hash-object`, `mktree`, `commit-tree`, `update-ref`), and change
  detection uses file fingerprints instead of asking git (next bullet). A cold
  record or an edit that re-records spawns 13 git processes, a no-change rerun
  3, and a run that selects nothing 5; the number is a constant, never per file
  or per test, and tests pin that budget and that `git diff` / `git status` /
  unscoped `ls-files` are never asked. `load_meta` and the fingerprints are
  memoized on the run's single `GitStore` (invalidated by save / delete / sync
  pull), and planning and selection share one context. A run that selects
  nothing learned nothing new, so it does not rewrite the shard.

  An earlier version did the plumbing in-process (a directory walk for the
  toplevel, loose-object reads, writing the shard's blobs, tree, commit and ref
  by hand) and spawned none. Measured over whole sessions it saved 30–135 ms per
  run on a 30-test project (a record or a one-test edit ran about twice as fast)
  and 40–500 ms on a 2,480-test suite (1–5%). That is not worth ~300 lines of
  git-object code, a second code path for everything it did not model (packed
  objects, SHA-256, reftable, a held lock) and the tests to keep the two
  identical, so it was removed.
- **File fingerprints (`files.json`).** `git diff <base>` plus `git ls-files
  --others` answer a question selection does not actually have: they report
  every file that differs from a commit, and the second one scans the whole
  checkout (790 ms on a 10k-file repository with 46k untracked files; 363 ms
  for the pair). What selection needs is narrower — *do the files this shard
  depends on still look like they did when it was recorded?* — so the shard
  stores a snapshot: `[size, mtime_ns, content hash]` for every dependency file
  and every `.py` file in a **watched** directory (each directory holding a
  dependency, and all its ancestors up to the root), plus those directories'
  subdirectory names. A run `stat`s the recorded files and lists the watched
  directories (26 ms for 3,250 files on that repository):
  - same size, same mtime, and an mtime older than the snapshot by more than 2 s
    → unchanged; anything else is settled by hashing the bytes. A `touch`, a
    checkout that rewrites identical content, or a shard recorded on another
    machine (every mtime differs) is **not** a change, and a file modified
    inside the racy 2 s window is never trusted on stat alone (git's
    "racily clean" rule). A dependency that changed while it was being parsed
    is stored with mtime `-1`, so the next run always re-hashes it.
  - a `.py` file that is not in the snapshot, or a subdirectory that is not in
    the listing, is *new*. Only then is git asked — once, about those paths
    alone (`ls-files --cached --others --exclude-standard -- <paths>`), so
    ignore rules stay git's job while the cost no longer scales with the size of
    the checkout. Hidden directories, `__pycache__`, `*.egg-info` and the usual
    environment directories are never reported as new.
  - it needs no base commit: rebasing or amending does not invalidate the shard
    (previously "cannot compute changes since the recorded base commit; running
    everything"), a file that is **untracked** is no longer "changed" on every
    run, and a file recorded **dirty** and later reverted is noticed (a diff
    against `HEAD` cannot see it).
  - shards without `files.json` (recorded before this), a snapshot it cannot
    read, and git being unable to answer the scoped question all use `git diff`
    against the shard's base commit as before. A randomized differential test checks the two agree
    on edits, touches, deletions, new and ignored files, and reverts.
  - rewriting a shard reuses the previous snapshot's hashes for files whose size
    and (old enough) mtime did not change, so it costs stats, not reads.
  - A run that stops early (`-x`, `--maxfail`) marks the tests it selected but
    never ran as must-rerun: the snapshot is refreshed by that run, so a test
    that kept its old block hashes would otherwise look fresh.

- **Pristine: "no change" beats "always run".** A test that spawns a process
  has no recorded dependencies (coverage cannot follow a child), so it used to
  re-run on *every* run — on this repo's own suite 198 of 400 tests, which made
  a rerun with nothing changed take 41.9 s, the same as a cold run. The reason
  for the rule also says when it can be relaxed: if nothing observable changed,
  the outcome is last time's outcome. A state is **pristine** only when all of
  these hold, and otherwise the old behavior applies:
  1. the change check reports nothing;
  2. `HEAD` is still the commit the shard was recorded at (a new commit, even an empty one, wakes the spawners once; the re-record
     makes the state pristine again);
  3. that check covered everything a child could read. `git diff` + untracked
     covers the whole repository; fingerprints only cover what they watch, so
     when a recorded test spawns, the snapshot is **widened** at record time to
     every tracked file (`"all": true`, one `git ls-files`, capped at 30,000
     files — beyond that nothing is widened and spawners keep re-running;
     files over 1 MB are fingerprinted by stat alone, a differing mtime counting
     as a change) and lists every file name in the watched directories, so even
     a brand-new untracked file only a child could read counts as a change
     (hidden and gitignored files do not).

  Measured on this suite: rerun with nothing changed 41.9 s → 0.4 s, no git
  process spawned. Honest limits: a file read **only by a child** that is
  untracked *and* sits in a directory no dependency lives in, or a tracked
  file's content changed without its mtime/size changing on an old file
  (the same blind spot as git's stat cache), is not seen.

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

### 3. Selection: file fingerprints + collection skipping

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

`changed` = the dependency files (and the `.py` files beside them) that differ
from the shard's recorded fingerprints, plus new project `.py` files (see
*File fingerprints* above). A test runs when:

- it has **no** recorded data (new test, or the previous run never finished
  it — e.g. it was interrupted),
- it **failed** in the recorded run, or
- one of its dependencies is in `changed` (a *stale block*: the file's
  current block table no longer has the recorded content hash),
- one of the **shared import-time** dependencies went stale (that is the
  whole-suite fan-out for conftest/import-time edits — block-granular:
  only blocks that actually executed at import time count),
- the test is a **spawner** (`always`) and the state is not pristine: it runs
  (`N always re-run`); under `spawn_unknown`, every test runs when a changed `.py` file is stale
  or was never recorded (`N selected by the subprocess rule`),
- the test is **runtime-dynamic** and some changed `.py` file was never
  recorded (`N selected as runtime-dynamic`).

Every widening is printed in the run summary — selection never widens
silently.

- A qualified name defined more than once with different content
  (`@overload` stubs, property getter/setter pairs, conditional definitions)
  is split into per-definition blocks (`name~<hash>`) resolved by source line,
  so editing one definition — or an `@overload` stub that never executes —
  does not invalidate the others (previously a file-level wildcard: any edit,
  even a comment, re-ran every test that touched such a function).
- Dependencies are code blocks; precision is bounded by the block unit — a
  hot, branchy method fans out to every test that executed *any* part of it
  (an edit inside one branch re-runs tests exercising another). Deliberate:
  finer units need line-level recording, which is the storage curve that
  killed line-level formats (IMPROVEMENTS item 11 has the full analysis).
- Spawners re-run on *any* change — the price of fail-safe: a suite whose
  tests shell out heavily (this repo's own pytester tests: ~half the suite)
  gets little skipping from them once something changed (an edit, even a
  comment, wakes them all; only a rerun with nothing changed is free).
  Spawns made by a conftest *at startup* are not
  tracked (their in-process effects still are, via the import-time bucket),
  and child reads of data files are not tracked (data-file tracking covers
  the pytest process only). Monitoring inside the child would be the
  precise version; it is not built.
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
├── monitoring.py  # sys.monitoring tool: PY_START/PY_RESUME + import bytecode scan
├── gitstore.py    # JSON shards behind refs/less/<shard>
├── selection.py   # shard parsing + test selection
└── util.py        # path normalization, command shard key
```
<!-- dogfood demo: a change no test depends on -->
