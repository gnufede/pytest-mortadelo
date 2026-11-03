# Correctness notes: what selection guarantees, and why

The contract: **a test is re-run whenever anything it executed, received,
or imported changed; everything else may be skipped.** Every widening
beyond block-exact staleness is printed in the run summary — selection never
widens silently — and every rule fails on the safe side: when the plugin
cannot decide, the test runs.

A one-paragraph model of the machinery, then the corner cases. The rules
were validated by the repository's full test suite (including 312
randomized equivalence selections and corpus re-hash checks) before the
current minimal core suite; the samples below are runnable by hand.

## The rules

1. **Block identity comes from compiled bytecode** (`compile(source)`), not
   from the running code objects: pytest rewrites asserts in test modules and
   other import hooks transform code, so only the file on disk is
   reproducible on both sides. Equal bytecode is equal behavior: an edit the
   compiler folds away (`x + 0`) selects nothing.
2. **Executed blocks: `PY_START` and `PY_RESUME`.** A code object fires at
   most once per test window (`DISABLE` until the next `restart_events()`),
   which is what makes per-test attribution affordable: 2.45M events on a
   2,480-test suite, ~1,000 per test. `PY_RESUME` covers a generator or
   coroutine primed by one test running its code in whoever resumes it.
3. **Module skeletons** (module-level statements, class bodies, decorators,
   signatures — everything that executes at import) are the import-time
   dependency of every importer.
4. **The import bytecode scan** over-approximates dynamic imports: `import`
   bytecode inside an executed block records the target's skeleton even if
   the import never runs. `importlib.import_module` is wrapped after the
   call, so even a *cached* import (no code runs) records the module's
   skeleton for the caller.
5. **Spawners get no dependencies and re-run whenever anything changed**:
   coverage cannot follow a child process.
6. **Modules imported before the monitor arms** (pytest loads its plugins
   before the earliest hook) get their skeletons claimed at arming.
7. **The fixture closure** — this document's centerpiece below: a
   >function-scoped fixture's setup runs once, in the first requester's
   window, so later consumers are recorded statically, from
   `item.fixturenames`.

## Minimal repro 1: a shared fixture's consumers

The hole rule 8 closes, small enough to show live in a talk (~30 seconds):

```bash
mkdir mortadelo-demo && cd mortadelo-demo && git init -q
cat > conftest.py <<'EOF'
import pytest

@pytest.fixture(scope="class")
def config():
    return {"v": 1}
EOF
cat > test_suite.py <<'EOF'
class TestA:
    def test_first(self, config):
        assert config["v"] == 1

    def test_middle(self, config):
        assert config["v"] == 1

    def test_last(self, config):
        assert config["v"] == 1
EOF
git add -A && git commit -qm init
python -m pytest -q --mortadelo                 # record: 3 passed
python - <<'EOF'
s = open("conftest.py").read()
open("conftest.py", "w").write(s.replace('return {"v": 1}', 'return {"v": 30}'))
EOF
python -m pytest -q --mortadelo
python -m pytest -q                        # ground truth, no selection
```

What each command shows:

| run | before the fixture-closure rule | with it |
|---|---|---|
| `--mortadelo` after the edit | `selected 1/3 tests ... 1 affected` — `test_first` only; the other two are **silently skipped** | `selected 3/3 tests ... 3 affected` |
| outcome of that run | `1 failed, 2 deselected` | `3 failed` |
| ground truth (no `--mortadelo`) | `3 failed` — two skipped tests had failing outcomes | `3 failed` |

Why: the class-scoped fixture's body executes once, in `test_first`'s
window. `test_middle` and `test_last` receive the cached value without
executing any of its code, so runtime events cannot see their dependency;
pytest knows the closure statically, and it is recorded for every requester.

On a real suite: sqlalchemy's testing plugin has an autouse class-scoped
fixture (`setup_class_methods`); a one-line edit to its body selected
**425 of 2,480 tests** before the rule and **all 2,480** with it. The other
2,055 all consume its value. The recorded cost is 4,859 extra
`(test, block)` edges on that suite (+0.3 % of the dependency graph), and
the six standard reference edits select exactly the same tests as before.

## Minimal repro 2: `import` after the first test

The hole rule 4 closes: only the **first** import of a module runs its code —
every later one is a silent `sys.modules` dict hit, so no event can record
the dependency. Same shape as repro 1, small enough for a talk (~30 s):

```bash
mkdir mortadelo-import-demo && cd mortadelo-import-demo && git init -q
cat > lib.py <<'EOF'
CONST = 1

def f():
    return CONST
EOF
cat > test_a.py <<'EOF'
def test_a():
    import lib            # the FIRST import: module code runs in this window
    assert lib.f() == 1
EOF
cat > test_b.py <<'EOF'
def test_b():
    import lib            # sys.modules hit: no event fires in this window
    assert lib.f() == 1
EOF
git add -A && git commit -qm init
python -m pytest -q --mortadelo                 # record: 2 passed
python - <<'EOF'
s = open("lib.py").read()
open("lib.py", "w").write(s.replace("CONST = 1", "CONST = 30"))
EOF
python -m pytest -q --mortadelo
python -m pytest -q                        # ground truth, no selection
```

What each run shows:

| run | without the import scan | with it |
|---|---|---|
| `--mortadelo` after the edit | `selected 1/2 tests ... 1 affected` — `test_b` is **silently skipped** | `selected 2/2 tests ... 2 affected` |
| outcome of that run | `1 failed` — a missed failure | `2 failed` |
| ground truth (no `--mortadelo`) | `2 failed` | `2 failed` |

Why: the edit changes module-level code, so only the module *skeleton*
stales — `f`'s bytecode is identical. `test_a` holds the skeleton because
the module genuinely executed in its window (`PY_START`). `test_b` gets a
`sys.modules` hit and only ever calls `f`, so runtime events give it the
`f` block and nothing else; the scan of its executed block's bytecode sees
the `IMPORT_NAME` and records `lib`'s skeleton for it.

The asymmetry is the talk-friendly part: disable the scan and `test_a`
stays correct — the event fired in its window. The scan exists for
everyone *after* the first importer, whom no event can ever see, because
`sys.modules` caching is fundamental to how Python imports work. Like the
fixture closure (repro 1), this is static knowledge recorded for consumers
whose producer already ran; it never executes anything to resolve a name.

## Checked and covered — corner cases and why they hold

**Attribution (events).**
- *Module-level code and class bodies* are ordinary code objects: `PY_START`
  fires for `<module>` and the class body at import, and for a class body
  *defined inside a test* (its own block, `test_x.<locals>.D`). An import
  executed inside a test attributes the imported module's skeleton, its
  import-time calls and its class bodies to that test.
- *A generator or coroutine resumed in another test's window* runs its code
  there — `PY_RESUME` attributes it. On sqlalchemy, 186 tests consume
  class-scoped yield fixtures this way.
- *A monkeypatched/replaced function* attributes its *definer's* block to
  whoever runs it: the replacement code object lives in the patching test's
  file, and the callee's own block to whichever test calls it.
- *`exec(compile(src, "repo/path.py", "exec"))`* resolves through
  `co_filename` like any other code object of that file.
- *`eval`/`exec` of a string produced in the repo*: the string is a constant
  of the enclosing block — editing it stales that block — and every call the
  compiled code makes fires `PY_START` in the caller's window (verified:
  `exec("import lib\nassert lib.g() == 5")` records `lib:g`; editing the
  string or `lib.g` each selects the test). A string from an untracked source
  (env var, network) is the state-dispatch residual. Flagging `eval`/`exec`
  as static markers — tried once — selected 69 % of sqlalchemy's ORM tests on
  every edit while guarding nothing the events miss.
- *Code a co-resident tool re-arms*: a foreign `restart_events()` (another
  monitor in the process) is answered by re-silencing, one event per window.
- *Threads* are attributed to the test running at the moment their code
  fires; only cross-test state handoff remains, as for any TIA.
- *Plugins loaded before arming* (entry points, `-p`) have their skeletons
  claimed from `sys.modules` at arming.

**Identity and hashing.**
- *Reformatting, comments, docstring-only edits*: invisible to bytecode
  hashes (docstrings are behavior on this path — an honest simplification
  that costs a few extra re-runs on docstring-heavy repos).
- *Redefinitions* (`@overload` stubs, property getter/setter pairs,
  conditional definitions): one block per definition, resolved by source
  line, so an edit invalidates only the definition that changed.
- *PEP 649 (3.14+) annotations* are `__annotate__` functions with their own
  blocks; editing one selects exactly the tests that evaluated it.
- *The same code imported under two names / reached via a symlink* still
  resolves by file content.
- *Per-version validity*: hashes are tagged per Python minor version, so a
  dataset recorded on 3.12 never reads as "unchanged" on 3.13; each supported
  version (3.12–3.15) was validated separately (corpus line-shift check,
  exact resolution of every def/class, 5,000 mutations of real functions,
  end-to-end selection).

**Change detection.**
- *Untracked or dirty-at-record-time files*: the stored fingerprint snapshot
  is keyed by content and independent of any base commit (amend/rebase safe).
- *A file deleted or made unparseable*: the block table becomes `None`, every
  recorded dependency on it stales.
- *New files*: watched directories record the file *names* in the snapshot,
  so a new non-hidden file counts as a change; git is asked once, only about
  new paths.
- *An edit during the recording run itself*: racy-mtime files (modified
  within the snapshot's grace window) are re-hashed, never trusted from stat.
- *A concurrent recording of the same data*: each `.pytest_mortadelo/` file is
  replaced atomically (`write tmp` + `rename`); last writer wins.

**Runs and outcomes.**
- *Interrupted runs, `-x`, `--maxfail`*: selected-but-unrun tests are marked
  must-rerun (their fingerprints are refreshed, so they can never look fresh
  forever).
- *Previously failed tests* always re-run until they pass.
- *Selectors* (`-k`, `-m`, `::`) disable selection entirely — a full run.
- *A runtime `request.getfixturevalue` on an uncached wide fixture*
  executes it in the requesting test's window, where events record it.

## Two kinds of gaps — and why only one is closable

Every blind spot in this document falls into one of two families, and the
distinction is the most useful thing to carry out of it:

### Mechanical gaps — a thing is loaded but no signal sees it

A **specific file, module, or library** is consumed, but the existing
signals (PY_START, PY_RESUME, the import scan, `open`) don't fire for it.
The gap is in the *observation*, not in the model. One call maps to one
dependency, so wrapping the entry point closes it:

| gap | why the signal missed it | closed by |
|---|---|---|
| a `.so` loaded by `import fast` | the import scan resolves via `find_spec` → the artifact's path | already covered (the scan asks the import system itself) |
| a module imported by `import_module("lib")` (cached) | a plain function call, no `IMPORT_NAME` opcode; `sys.modules` hit, no code runs | the `import_module` wrap — one call, one module skeleton |
| the active `pytest.ini` | parsed before the earliest hook, so no event ever sees the read | claimed at collection (pytest itself says which file it parsed) |
| a shared fixture's *value* | the setup ran once, in the first consumer's window | the static closure (`item.fixturenames`) |

In every case the fix is the same shape: find where the system already
knows (or can cheaply learn) the mapping from *what happened* to *which
file or block*, and record it as an ordinary dependency. No
configuration, no widening — just a new edge in the same graph.

### State gaps — the *choice* depends on something that is not a thing

A test's outcome depends on **which of several possible code paths it
took**, and the condition that selected the path is not a file, a module,
or a code object — it is run-time state the repository cannot represent:

```python
def run(self):
    # the target is chosen by an environment variable: not a file,
    # not a fixture, nothing the recording ever sees
    return getattr(self, os.environ.get("DISPATCH", "fast"))()
```

No wrapper can close this, because the gap is not in the *observation* —
it is in the *counterfactual*. The recording correctly saw that
`handle_fast` executed. It cannot know that `handle_slow` would have
executed under a different env var, because the env var is not a
dependency that can be recorded. The possibilities:

- **Record what actually runs** — already done (PY_START on the target).
  Sound for the recorded state; silent about the others.
- **Record all possible targets** — what the removed static `getattr`
  markers did: 2,480 of 2,480 sqlalchemy tests on a one-function edit,
  while guarding nothing the block rules miss. The nuclear option, and
  it makes the tool useless.
- **Record the dispatch logic** — already done (the code that computes
  the name is a block; editing it stales it).
- **Record a new target** — already done (adding a definition changes the
  class body, which every importer holds).

The residual is exactly one case: **the state that chose the path is
outside the repository** — an env var, the clock, randomness, an
installed package version, a network response. That is the fundamental
limit of any test impact analysis, not a wrapping problem.

This is the taxonomy for evaluating any proposed TIA feature: *does it
map a call to a thing?* If yes, wrap the entry point and record the
dependency. If the "thing" it maps to is *state* rather than code or a
file, no wrapper can help — and the honest answer is a full run.

## Known residuals (documented, accepted)

- *Dispatch that depends on state we do not track* — the environment, time,
  randomness, installed packages, external services. The fundamental limit of
  test impact analysis without hermetic environments. (Code sample below.)
- *Memoized calls outside `functools.lru_cache` / `functools.cache`*
  (`functools.cached_property` on shared instances, hand-rolled
  `if _CACHED is None` caches): only the first caller records the edge —
  a later caller with the same arguments executes nothing, so no event
  fires, and no static knowledge exists that could record it. (Sample 3.)
  `lru_cache` and `cache` are wrapped: every call, hit or miss, records
  the decorated function's block for the caller.
  Measured on sqlalchemy (12-file ORM subset, 2,480 tests): an edit to
  `SQLCompiler.visit_column` selects 1,455 tests — every test that
  compiled its own statements — while a plain full run on the same edit
  fails 1,633. The other ~180 hold `Compiled.process` but not
  `visit_column`: they consume SQLAlchemy's cached `Compiled` artifacts
  (a hand-rolled LRU cache), compiled earlier by other tests. A behavior
  change reaches those consumers through the artifact, but the recording
  never saw them execute the block.
- *Configuration files*: fixed — the active one (`config.inifile`) is now
  claimed as a shared file-level dependency at collection (sample 4 below
  tells the story, including the fix).
- *Order-dependent tests*: a test whose outcome depends on another having
  run first (module globals, `setattr` of data, files one test writes and
  another reads). TIA assumes test independence; this is that assumption.
  (Sample 5 below.)
- *Directory contents*: only `open` is wrapped — `os.listdir`, `glob`,
  `rglob`, `os.walk` record no dependency on what they looked at.
  (Sample 6 below.)
- *C extension sources* (`.c`): nothing executes them, so they are not
  dependencies — but the built artifact is. An in-repo `.so` imported by name
  resolves through `find_spec` in the import scan and becomes a file-level
  dependency of every importer, so a rebuild re-runs its tests (verified end
  to end with a compiled extension). Editing the source without rebuilding
  selects nothing — correctly: the behavior, the artifact, has not changed.
  Builds landing outside the repository (site-packages) belong to the
  installed-packages family below.
- *CPython's own `.pyc` same-size same-second race*: CPython may execute
  stale bytecode while source-hash selection says "unchanged" — an
  environment-level hazard (it corrupted an early version of the fixture
  demo above), not something a plugin can close.
- *mtime-preserving copies* (`cp -p`, `rsync -a`): a content change with an
  identical stat can pass the fingerprint check, the same blind spot as
  git's stat cache.
- *Session-scoped fixture finalizers after the last test's window*: their
  code runs after all outcomes are recorded; only retro-failing finalizers
  (rare) could matter.
- *Reads by a child process* of an untracked file in a directory no
  dependency lives in: invisible to the snapshot. Child processes generally
  are covered by the spawner rule (always re-run), not tracked.
- *pytest-xdist*: unsupported — and safe: workers are spawned processes, so
  the spawner rule takes over.
- *Interpreters newer than 3.15*: every file is a file-level dependency
  (any change re-runs its tests) until the hasher is validated for that
  version — never a guess.

### Sample 1: memoized functions — the `functools` ones are closed

A memoized function called by a second test with the same arguments executes
*nothing* in that test's window — and Python has no static knowledge of which
test calls which function (unlike fixture requests, which pytest knows, and
`import` bytecode, which the scan sees). Line-coverage tools miss these calls
too: no line of the function runs.

```python
# lib.py
import functools

@functools.lru_cache
def compute(x):
    return x * 10
```

```python
# test_a.py
def test_a():
    import lib
    assert lib.compute(1) == 10   # executes compute -> recorded by PY_START

# test_b.py
def test_b():
    import lib
    assert lib.compute(1) == 10   # cache hit: nothing executes
```

Without a rule, `test_b` holds only `lib:<module>` and an edit to the body
(`x * 10` → `x * 100`) skips it silently. **Closed by wrapping
`functools.lru_cache`/`cache` at decoration time**: the dependency is
pre-computed once (the decorated function's block) and every call — hit or
miss — records it for the caller, so every consumer gets the edge. The wrap
is scoped to repo functions; stdlib and third-party caches keep the
untouched C wrapper.

What remains open is the same family outside `functools`: hand-rolled
caches, `if _CACHED is None` module globals, and `cached_property` on
instances shared between tests (sample 3).

### Not handled, sample 2: computed `getattr` on untracked state

The repo-resident variant of computed dispatch is covered: adding a
`handle_x` method changes the class body, a block every importer holds, so
a test that could reach it re-runs (that is what made removing the static
`getattr` markers safe — see minimal repro 1). The uncovered variant is
when the *choice itself* depends on state selection never sees:

```python
# lib.py
import os

class Handler:
    def fast(self):
        return "fast"

    def slow(self):
        return "slow"

    def run(self):
        # the target is chosen by an environment variable: not a file,
        # not a fixture, nothing the recording ever sees
        return getattr(self, os.environ.get("DISPATCH", "fast"))()
```

```python
# test_dispatch.py
def test_dispatch():
    import lib
    assert lib.Handler().run() in ("fast", "slow")
```

Recorded with `DISPATCH` unset, the test holds exactly `run`, `fast`, the
`Handler` class block and `lib:<module>` — no `slow`. Edit `slow`'s body
(`"slow"` → `"crawling"`):

- `--mortadelo` rerun, same state: `selected 0/1 tests` — **correct**: nothing
  the test executed changed (this precision is deliberate; a handler the
  test never ran does not re-run it), and the test still passes;
- `DISPATCH=slow python -m pytest` (no `--mortadelo`): `1 failed` — the outcome
  changed on a path no recorded dependency covers.

Every rule the plugin has is about code and files; this dispatch is decided
by process state. Widening every computed `getattr` to file level — the
removed static markers — selected 2,480 of 2,480 sqlalchemy tests on
one-function edits while guarding nothing the block rules miss.

### Not handled, sample 3: `cached_property` on a shared instance

The fourth member of the first-consumer-wins family. A session fixture hands
every test the same object; the first access runs the property body in
*that* test's window, later accesses read the instance `__dict__`. The
fixture closure (minimal repro 1) records the *fixture's* block for every
consumer — but the property body is a different producer, and only its
first consumer holds it:

```python
# lib.py
import functools

class Shared:
    @functools.cached_property
    def config(self):
        return 10
```

```python
# conftest.py
import pytest
import lib

@pytest.fixture(scope="session")
def shared():
    return lib.Shared()
```

```python
# test_a.py and test_b.py
def test_a(shared):
    assert shared.config == 10
```

Recorded: `test_a` holds `conftest:shared` (the closure) **and**
`lib:Shared.config` (the first access); `test_b` holds only
`conftest:shared`. Edit the body (`10` → `100`): `selected 1/2 tests`,
`test_b` silently skipped; ground truth `2 failed`.

### Sample 4: configuration files — a gap, now fixed mechanically

pytest parses `pytest.ini` / `pyproject.toml` / `tox.ini` / `setup.cfg`
before the earliest hook a plugin can use, so no event ever saw the read
and the files never became dependencies. With `test_warn.py` emitting a
warning, changing `pytest.ini` to `filterwarnings = error` produced:

- `--mortadelo`: `selected 0/1 tests: no recorded dependency changed; fast
  path` — the run never even read the stored data;
- ground truth: `1 failed`.

**Fixed:** the *active* ini file — the one pytest itself reports
(`config.inifile`) — is claimed as a shared file-level dependency, so a
config change selects everything, which is what a config change means.
Two deliberate details: no configured list of globs (pytest says which
file it parsed, so nothing to maintain or forget); and a *file-level*
dependency, not a content hash — ini files often happen to parse as
Python (`[pytest]` reads as a one-element list), but Python-visible
equivalence is not ini equivalence (configparser keeps a trailing
`# note` *inside* a value while Python strips it as a comment), so two
different configs could have hashed identically.

This is the shape we prefer over the "tracked files" escape hatch other
TIA tools offer: a rule derived from what the system already knows, exact
by construction, zero configuration.  For inputs no rule can see --
``ctypes.CDLL`` artifacts are the canonical case (dlopen, not import) --
there is no rule; the honest answer is a full run.

### Not handled, sample 5: order-dependent tests

`setattr` of *data* — module globals, attributes — is cross-test state
leakage: the second test executes nothing of the first, so no dependency
between them can exist. `setattr` of *functions* is the covered case above
(definer attribution); data is the independence assumption itself:

```python
# test_1_set.py
def test_1_set():
    import lib
    lib.STATE = 22       # was 1

# test_2_get.py
def test_2_get():
    import lib
    assert lib.get() == 1
```

After the edit: `selected 1/2 tests` — only the setter; ground truth
`test_2_get` fails. The same shape exists through files one test writes
and another reads (non-Python files are not dependencies at all:
nothing executes them).

### Not handled, sample 6: directory listings

No file reads are wrapped, so a test whose outcome depends on a
directory's *contents* holds no dependency on anything it looked at:

```python
# test_1_write.py writes out/a.txt and out/b.txt (not deps: nothing executes them)
# test_2_list.py
def test_2_list():
    import os
    assert len(os.listdir("out")) == 2
```

Editing `test_1_write` to write a third file: `selected 1/2 tests` —
`test_2_list` never touched anything selection can see; ground truth
`1 failed, 1 passed`.
