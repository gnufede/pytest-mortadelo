# Correctness notes: what selection guarantees, and why

The contract: **a test is re-run whenever anything it executed, received,
imported or read changed; everything else may be skipped.** Every widening
beyond block-exact staleness is printed in the run summary — selection never
widens silently — and every rule fails on the safe side: when the plugin
cannot decide, the test runs.

A one-paragraph model of the machinery, then the corner cases. Each rule
below is pinned by tests in this repository.

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
   the import never runs.
5. **Spawners get no dependencies and always re-run**: coverage cannot follow
   a child process.
6. **Runtime-dynamic tests** (through `importlib.import_module`, `runpy`,
   `pkgutil`, ...) re-run when a changed `.py` file was never recorded.
7. **Modules imported before the monitor arms** (pytest loads its plugins
   before the earliest hook) get their skeletons claimed at arming.
8. **The fixture closure** — this document's centerpiece below: a
   >function-scoped fixture's setup runs once, in the first requester's
   window, so later consumers are recorded statically, from
   `item.fixturenames`.

## The minimal repro: a shared fixture's consumers

The hole rule 8 closes, small enough to show live in a talk (~30 seconds):

```bash
mkdir less-demo && cd less-demo && git init -q
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
python -m pytest -q --less                 # record: 3 passed
python - <<'EOF'
s = open("conftest.py").read()
open("conftest.py", "w").write(s.replace('return {"v": 1}', 'return {"v": 30}'))
EOF
python -m pytest -q --less
python -m pytest -q                        # ground truth, no selection
```

What each command shows:

| run | before the fixture-closure rule | with it |
|---|---|---|
| `--less` after the edit | `selected 1/3 tests ... 1 affected` — `test_first` only; the other two are **silently skipped** | `selected 3/3 tests ... 3 affected` |
| outcome of that run | `1 failed, 2 deselected` | `3 failed` |
| ground truth (no `--less`) | `3 failed` — two skipped tests had failing outcomes | `3 failed` |

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

## Checked and covered — corner cases and why they hold

**Attribution (events).**
- *Module-level code and class bodies* are ordinary code objects: `PY_START`
  fires for `<module>` and the class body at import, and for a class body
  *defined inside a test* (its own block, `test_x.<locals>.D`). An import
  executed inside a test attributes the imported module's skeleton, its
  import-time calls and its class bodies to that test.
  (pinned: `TestModuleLevelCodeIsTracked`)
- *A generator or coroutine resumed in another test's window* runs its code
  there — `PY_RESUME` attributes it. On sqlalchemy, 186 tests consume
  class-scoped yield fixtures this way. (pinned:
  `TestAResumedGeneratorIsADependency`)
- *A monkeypatched/replaced function* attributes its *definer's* block to
  whoever runs it: the replacement code object lives in the patching test's
  file, and the callee's own block to whichever test calls it.
- *`exec(compile(src, "repo/path.py", "exec"))`* resolves through
  `co_filename` like any other code object of that file.
- *Code a co-resident tool re-arms*: a foreign `restart_events()` (another
  monitor in the process) is answered by re-silencing, one event per window.
- *Threads* are attributed to the test running at the moment their code
  fires; only cross-test state handoff remains, as for any TIA.
- *Plugins loaded before arming* (entry points, `-p`) have their skeletons
  claimed from `sys.modules` at arming. (pinned: `TestPluginsLoadedBeforeArming`)

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
  shard recorded on 3.12 never reads as "unchanged" on 3.13; each supported
  version (3.12–3.15) was validated separately (corpus line-shift check,
  exact resolution of every def/class, 5,000 mutations of real functions,
  end-to-end selection).

**Change detection.**
- *Untracked or dirty-at-record-time files*: the shard's fingerprint snapshot
  is keyed by content and independent of any base commit (amend/rebase safe).
- *A file deleted or made unparseable*: the block table becomes `None`, every
  recorded dependency on it stales.
- *New files*: watched directories record the file *names* in the snapshot,
  so a new non-hidden file counts as a change; git is asked once, only about
  new paths.
- *An edit during the recording run itself*: racy-mtime files (modified
  within the snapshot's grace window) are re-hashed, never trusted from stat.
- *Write-only opens are not dependencies* (outputs), and reading back what the
  test itself wrote is not either.
- *A concurrent recording of the same shard*: the ref moves atomically;
  last writer wins, each shard self-consistent.

**Runs and outcomes.**
- *Interrupted runs, `-x`, `--maxfail`*: selected-but-unrun tests are marked
  must-rerun (their fingerprints are refreshed, so they can never look fresh
  forever).
- *Previously failed tests* always re-run until they pass.
- *Selectors* (`-k`, `-m`, `::`) disable selection entirely — a full run.
- *A runtime `request.getfixturevalue` on an uncached wide fixture*
  executes it in the requesting test's window, where events record it.

## Known residuals (documented, accepted)

- *Dispatch that depends on state we do not track* — the environment, time,
  randomness, installed packages, external services. The fundamental limit of
  test impact analysis without hermetic environments.
- *C extensions in the repository*: `.c`/`.so` changes never select anything
  (nothing to hash); shared by every Python TIA.
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
