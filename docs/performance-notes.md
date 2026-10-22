# Performance notes: where the time goes, what was tried, what is next

Written to resume the performance investigation later. Everything here was
measured on one Mac (Apple silicon, Python 3.13.5) unless it says otherwise.
Priorities for any change, in order: **correct, fast, simple**. A change with a
small payoff is not kept (see "Decided against").

Status: the selection lead in section 2 **is implemented** (commits `190aaf1`,
`76b19ab`, `ca79dd8`, on top of `718cdb3`); a comment-only edit went from 2.53 s
to 1.53 s and a 338-test edit from 5.01 s to 4.46 s on the benchmark corpus,
with identical selection results.  The PY_RESUME lead (a generator resumed in
another test's window ran its code there with no attribution) is also done
(`6b114d2` + the module-level tests): 186 tests on sqlalchemy were consuming
class-scoped yield fixtures unattributed; the fix costs at most ~0.1 s per
record (within noise).  PY_START + PY_RESUME together track module-level code
end to end (module and class bodies, import-time calls). Sections 1, 2 and 7 are updated to say what
remains; the methodology and traps in section 5 are what made the earlier
numbers wrong, so they stay.

## 1. Where a run's time goes today

Benchmark corpus: the sqlalchemy ORM subset, 2,480 tests, plain pytest ≈ 10–11 s
(see "Reproducing" for how to set it up).

| run | cost over plain pytest | what it is |
|---|---|---|
| record, parse cache warm | **+2.3 s** | ≈2 s of per-event cost (below), `finish` 0.35 s, first-start analysis ≈0.18 s |
| record, cold | **+4.0 s** | the above + compile/digest of every executed file (0.54 s compile + 0.19 s digest for 220 files) and cache writes |
| rerun, nothing changed (fast path) | 0.87 s total, of which `remember_items` was 56 ms (now ~0) | reads meta.json + fingerprints only; `plan`/`select` never run |
| rerun, a file changed but nothing selected | **1.53 s** total (was 2.53 s) | full selection; see section 2 |
| plugin installed, not enabled | **≈ +5 ms** on a 95 ms run | importing the plugin's modules |

The per-event floor (≈2 s of the +2.3 s) is CPython's own dispatch and
re-instrumentation after each `sys.monitoring.restart_events()` plus our callback.
In one record, 2.45M `PY_START` callbacks are delivered: 66% to repo code, **33%
(812k) to non-repo code** (stdlib, site-packages, pytest, plugins) that is only
silenced with `DISABLE` and re-armed every test, 0.7% first-time analyses. None
of this can be cached across runs, because the events fire on every run.

First-start analysis, warm (16.7k new code objects): 0.18 s total, of which
`repo_rel` 0.043 s, `file_blocks` 0.051 s, `FileBlocks.resolve` 0.007 s
(9k calls; **qualname resolution is not worth caching**), `_resolve_dotted` 0.03 s.

Git processes per run (a cold record or an edit that re-records is 13, a
no-change rerun 3, a changed file that selects nothing 5): see section 3.

## 2. Selection was O(total dependency edges) — DONE

**Implemented and measured.** What was wrong: `build_context` iterated every
recorded `(test, block)` edge (≈1.65M on the 2,480-test corpus, 1.65M `note()`
calls) and `affected_reason` scanned each test's every block through a memoized
dict lookup — 3.3M `_Context.stale` calls, twice a run (planning, then
selection). A run where a comment was appended and *nothing* was selected
spent 1.9 s of a 2.0 s session in selection machinery.

The fix, three commits:

1. `load_shard` decodes the per-test index lists at C speed (`min`/`max`
   validate, `map` builds the set): 131 → 96 ms.
2. `build_context` iterates the ~9k **distinct** blocks (a C-speed set union
   per test collects them), precomputes the stale ones into one frozenset, and
   `affected_reason` asks each test a single `deps.isdisjoint(stale_blocks)`.
   The `_Context` per-block memo is deleted. `affected_reason` 870 → 1 ms,
   `plan` 524 → 16 ms, `select_tests` 439 → 1 ms, `build_context` 90 → 14 ms.
3. `remember_items` resolves each item's file once (2,493 `to_repo_rel` calls
   → one per distinct file): 56 → ~0 ms.

Measured after (7 interleaved repetitions, paired, selection line asserted):

| run (sqlalchemy subset) | before | after | paired |
|---|---|---|---|
| no change (fast path) | 0.88 s | 0.87 s | unchanged — selection never runs |
| comment-only edit, nothing selected | 2.53 s | **1.53 s** | −1.0 s, faster in 7/7 |
| `Query.filter` edit, 338 selected | 5.01 s | **4.46 s** | −0.55 s, faster in 7/7 |

Correctness: `tests/test_selection_equivalence.py` compares `load_shard`,
`build_context` facts, `affected_reason` and `select_tests` against per-element
reference implementations on 300 randomized shards (plus negative/bool/non-int
indices, duplicates, WILDCARD, dynamic/always/failed/pristine), and the six
reference selections on the real corpus are identical (0 · 1,185 · 1,455 ·
338 · 1,651 · 0). The shared single predicate (plan and select must never
disagree) is unchanged: both call the same `affected_reason`.

**What remains in a run that selects something** (measured after the fix, of
the 1.53 s comment-only session): `GitStore.load_data` 89 ms (`json.loads` of
the 9.14 MB blob — the floor for stdlib json), `load_shard` 96 ms (building
2,480 sets of ~600 block tuples ≈ 58 ns/element; going lower needs a different
data model — keep index lists and compare index sets — which was rejected as
complexity), collection + pytest startup the rest. Nothing here looks worth
another change.

## 3. Git process spawns

Every git call is a process spawn: ≈7–9 ms from a small process here, and
several times that from inside a large pytest process (13 spawns cost ≈500 ms
inside the sqlalchemy run, ≈120 ms in a tiny project). A quiet Linux machine is
likely cheaper; **not measured**.

Sequence of a cold record / edit-and-re-record (13 processes): `rev-parse
--show-toplevel`, 4 × `cat-file blob` (meta, parse, data, files), `rev-parse
HEAD`, 4 × `hash-object -w --stdin`, `mktree`, `commit-tree`, `update-ref`. No
change: toplevel + 2 × `cat-file` (meta, files). Reads come in two natural pairs:
{meta, files} (the fast-path probe, which must never fetch `data.json`) and
{data, parse}; with no shard at all, all four reads fail.

### Tried and reverted: batching the reads

One `git cat-file --batch` per pair, plus "`meta.json` missing ⇒ no shard ⇒ the
rest are missing", on the memoizing store only (≈60 lines, 9 tests, invalidated in
`_forget_ref_state`). It worked: cold record 13 → 10, no change 3 → 2, edit 13 →
11 processes. Paired over 9 runs it saved **11 – 31 ms** on a 30-test project (a
0.23 s run; every repetition faster) and was **indistinguishable from noise on the
2,480-test suite** (record: −21 ms median, faster in 3 of 9). Reverted as not worth
the code.

### Untested ideas on the write side (7 of the 13 spawns)

- One `git fast-import` instead of `hash-object` ×4 + `mktree` + `commit-tree` +
  `update-ref`. Micro-benchmark from a small process: **faster for a small shard**
  (17.7 ms vs 57.7 ms), **slower for a sqlalchemy-sized one** (187 ms vs 141 ms,
  recompression). Not tried inside a large pytest process, where spawns cost
  more. Needs care to create a parentless commit and to avoid accumulating packs.
- Combining `rev-parse --show-toplevel` and `HEAD` into one call (they are asked
  at different times).

Given the batching result, expect these to be small on large suites as well.

## 4. Decided against (with the numbers, so nobody re-measures)

| idea | payoff | why not |
|---|---|---|
| in-process git layer (`gitobjects.py`, removed) | 30–135 ms/run on a 30-test project (record and one-test edit ran about 2× faster), 40–500 ms (1–5%) on 2,480 tests, every repetition positive | ≈300 lines + a second code path for everything it did not model + tests to keep them identical |
| batching shard reads | see section 3 | tiny |
| repo-only `PY_START` via local events | ≈0.4 s of the +2.3 s warm tax (≈17%), prototyped | needs every repo code object registered before it first runs (import hooks, pytest's assertion rewriting, code created at runtime); a miss is a silently missed dependency |
| lazy-import the plugin's modules | ≈5 ms idle | small |
| cache qualname resolution | 7 ms | already negligible |
| repository-wide parse store, `PYTEST_LESS_PARSE_CACHE` kill switch | cold-only, local-only | removed |
| hooking Python's existing bytecode (running code objects or `.pyc`) | ≈0.4–0.5 s on a **cold** record only; warm runs read the cache | rewritten test modules cannot be reproduced from source at selection time without pytest's private rewrite API; `.pyc` validity is mtime-in-seconds |
| cosmetic docstring filter, static `getattr` markers | none for sqlalchemy (it reads docstrings itself); markers made `Query.filter` select 2,480 tests instead of 338 | removed |

Also measured and **not** worth chasing: `find_spec` is now 0.007 s;
`_build_module_index` (the repo `os.walk`) 0.011 s; `build_snapshot` ≈0.03 s.

Still on the table but unmeasured as a unit: `finish()` (0.35 s on a full
record; serializing 9 MB of compact JSON), `data.json` decode in edit runs
(`json.loads` of ≈9 MB).

## 5. Reproducing

### The corpus

`~/other_repos/sqlalchemy` at commit `36e47d3a2`, with its own venv
(`.venv-bench8`, Python 3.13.5; sqlalchemy runs from source). The test selection
is the list of ORM test files used throughout (12 files: `test/orm/test_session.py
test_query.py test_unitofwork.py test_transaction.py test_scoping.py test_events.py
test_mapper.py test_relationships.py test_eager_relations.py test_cascade.py
test_deferred.py test_unitofworkv2.py`): 2,480 tests, 2,405 pass, 75 skip.

Reference **selection results** (these must not change unless a change is meant to
change them): no edit 0 · body of `Session.execute` 1,185 · body of
`SQLCompiler.visit_column` 1,455 · `Query.filter` 338 · `ColumnElement.operate`
1,651 · `TypeEngine.copy_value` 0. (Edit = insert `_less_edit = 1` as the first
statement of the function body, record, edit, rerun.)

### The harness (it lived in `/tmp/ablate`, which will not survive)

Rebuild it from these rules rather than from memory:

- **Interleave** the variants (A, B, B, A, …) and report **paired differences**
  with how many repetitions favored each side. Single runs mislead.
- **Restore the baseline shard before every run.** A run that selects something
  re-records, so save `refs/less/*` once and put it back (`git update-ref`) before
  each timed run; otherwise later repetitions start from a different state.
- **Assert the run is the scenario you think it is**: parse the `selected N/M
  tests` line of every run and fail if it differs between variants or from the
  expectation.
- Count git spawns with a `git` shim earlier in `PATH` that logs `"$*"` and execs
  the real git; count **lines**, not whitespace-separated tokens.
- Keep wall-clock timing separate from the spawn-counting run (the shim slows
  things down).
- Compare two source trees by `git archive <rev> src` into a temp dir and
  switching `PYTHONPATH`; do not edit the working tree between variants.
- cProfile is blind inside `sys.monitoring` callbacks (CPython suspends events
  while one runs), so it cannot attribute first-start analysis; use wall-clock
  wrappers around the entry points for that.

### Benchmarking traps that cost time (all hit during this investigation)

1. **Any extra pytest argument changes the shard key.** The key is the normalized
   command line, so `-p some_timer_plugin` makes the run look like a different
   command: no shard is found and you benchmark a **full run** (this is how the
   last phase timing went wrong: both rows said `selected 2480/2480`). Inject
   instrumentation without touching the arguments: `PYTEST_ADDOPTS="-p mod"`
   (environment, not part of the key) or a `sitecustomize`/`PYTHONPATH` module.
2. **Run-to-run noise exists even for identical code**: weakref/GC-callback blocks
   (`_remove_linecache_entry`, `is_valid`, `_collection_gced`, …) land in
   different tests when allocation timing shifts. When comparing recorded
   dependencies run `-p nogc` (`gc.disable()`) with `PYTHONHASHSEED=0`, and measure
   the same-code-twice noise floor first.
3. **Small repetition counts lie.** A "+797 ms" from 3 repetitions became "−21 ms,
   faster in 3 of 9" with 9.
4. **Warm vs cold needs each variant to warm its own caches first**, or a cold run
   of one variant erases the warm state of the other.
5. `ls -T`/`stat -f` behave differently here (GNU coreutils on `PATH`); BSD `sed`
   chokes on many patterns: use Python for edits and for log parsing.
6. `date` here reports a different timezone than the file timestamps (10:21 vs
   19:21): compare times with the same tool.
7. A change to how a command is logged (the shim) silently breaks every counter
   built on the old format.

## 6. Environment notes

- Python 3.12 – 3.15 are supported by the hasher; CI runs all four. Local venvs
  used: `/tmp/venv314`, `/tmp/venv315` (recreate with `UV_PROJECT_ENVIRONMENT=...
  uv sync --python <interpreter>`; they will not survive either).
- **Local Homebrew Python 3.15 is currently broken**: its real executable
  (`/opt/homebrew/Cellar/python@3.15/3.15.0/.../bin/python3.15`) was replaced by a
  97-byte shell script that `exec`s itself forever (`FAKE-PYENV-USED (pyenv dir
  3.15-dev)`). Something on the machine keeps rewriting it. Not caused by this
  repo. `brew reinstall python@3.15` after finding the culprit; until then a local
  3.15 test run hangs (CI is unaffected). The working interpreter is inside the
  framework at `.../Versions/3.15/Resources/Python.app/Contents/MacOS/Python`, but
  tests that spawn `sys.executable` would hit the stub again.
- Commits are signed with an SSH hardware key whose agent sometimes refuses or
  hangs: retry, and check `git log` before retrying (a commit that "failed" may
  have landed).

## 7. If the investigation continues

The big lead is done. What is left, in decreasing order of expected value:

1. **Nothing measured looks promising.** The remaining per-run costs are the
   per-event monitoring floor (≈2 s of a warm record — irreducible without the
   rejected repo-only local events), stdlib `json` decode of the shard (~89 ms),
   and git spawns (section 3: 13/3/5 per run, each ~10 ms small-process /
   several times that in a large process).
2. The write-side ideas in section 3 (one `git fast-import`, merging the two
   `rev-parse` calls) — the read-batching result suggests these are also
   small; only try them with the paired-9-repetitions harness.
3. Re-check the numbers here on a Linux CI runner, where process spawns are
   cheaper: the git-spawn conclusions may differ.
4. Revisit the broken local Homebrew Python 3.15 (section 6) before trusting
   any local 3.15 run.
