"""End-to-end tests: real pytest subprocesses inside real git repositories."""

import json
import subprocess

import pytest


def git(path, *args, stdin=None):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
        input=stdin,
    )


def commit(path, message="commit"):
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", message)
    return git(path, "rev-parse", "HEAD").stdout.strip()


def shard_data(path):
    """The (single) less shard stored in the test repo, as a dict."""
    refs = [
        line
        for line in git(path, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.splitlines()
        if line
    ]
    assert len(refs) == 1, f"expected exactly one shard ref, got {refs}"
    return json.loads(git(path, "cat-file", "blob", f"{refs[0]}:data.json").stdout)


def decode_shard(data):
    """v2 shard -> (tests: {nid: set[(file, block, hash)]}, import_deps: set)."""
    table = [tuple(b) for b in data["blocks"]]
    tests = {nid: {table[i] for i in idxs} for nid, idxs in data["tests"].items()}
    import_deps = {table[i] for i in data.get("import_deps", [])}
    return tests, import_deps


def dep_ids(deps):
    return sorted(f"{f}:{b}" for f, b, _ in deps)


@pytest.fixture
def project(pytester):
    """A small git repository with two tests and two lazily-imported helpers.

    * ``test_a`` imports ``lib`` inside the test body (executed dynamic import)
    * ``test_b`` mentions ``other`` in a dead branch (unexecuted import -- the
      conservative bytecode scan must still record it as a dependency)
    """
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "lib.py").write_text("def ping():\n    return 1\n")
    (tests / "other.py").write_text("def pong():\n    return 2\n")
    (tests / "test_a.py").write_text(
        "def test_a():\n"
        "    import lib\n"
        "    assert lib.ping() == 1\n"
    )
    (tests / "test_b.py").write_text(
        "def test_b(always_false=False):\n"
        "    if always_false:\n"
        "        import other\n"
        "    assert 1 + 1 == 2\n"
    )
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


def run_less(pytester, *args):
    return pytester.runpytest_subprocess("--less", *args)


class TestRecording:
    def test_first_run_records_everything(self, project):
        result = run_less(project, "-q")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(
            ["*less*", "shard: *", "selected 2/2 tests: no usable stored data*"]
        )

        data = shard_data(project.path)
        tests, import_deps = decode_shard(data)
        assert set(tests) == {
            "tests/test_a.py::test_a",
            "tests/test_b.py::test_b",
        }
        assert data["failed"] == []
        # Executed dynamic import recorded for test_a: lib's SKELETON plus
        # the executed body of ping...
        assert "tests/lib.py:<module>" in dep_ids(tests["tests/test_a.py::test_a"])
        assert "tests/lib.py:ping" in dep_ids(tests["tests/test_a.py::test_a"])
        # ...while test_b's never-executed (dead branch) import records only
        # other's SKELETON -- a body edit there would NOT re-run test_b.
        assert dep_ids(tests["tests/test_b.py::test_b"]) == [
            "tests/other.py:<module>", "tests/test_b.py:<module>", "tests/test_b.py:test_b",
        ]
        # Conftest skeleton is a shared import-time dep.
        assert "tests/conftest.py:<module>" in dep_ids(import_deps)
        # ...but other tests' modules must NOT leak into a test's deps.
        assert "tests/test_b.py" not in data["tests"]["tests/test_a.py::test_a"]
        assert "tests/test_a.py" not in data["tests"]["tests/test_b.py::test_b"]
        # And no phantom dependencies from non-file module origins
        # (e.g. pytest's assertion rewriter injects ``import builtins``).
        for deps in data["tests"].values():
            assert "built-in" not in deps

    def test_nothing_changed_skips_collection(self, project):
        run_less(project, "-q")
        result = run_less(project, "-q")
        # Both test modules were skipped at collection time: nothing was even
        # imported, and pytest reports "no tests ran" (exit 5).
        assert result.ret == 5
        result.stdout.fnmatch_lines(
            ["selected 0/2 tests:*skipped collection of 2 file(s)*", "*no tests ran*"]
        )
        # The skipped tests must keep their recorded dependencies.
        data = shard_data(project.path)
        assert set(data["tests"]) == {
            "tests/test_a.py::test_a",
            "tests/test_b.py::test_b",
        }
        tests, _ = decode_shard(data)
        assert "tests/lib.py:<module>" in dep_ids(tests["tests/test_a.py::test_a"])
        assert "tests/other.py:<module>" in dep_ids(tests["tests/test_b.py::test_b"])

    def test_disabled_without_flag_records_nothing(self, project):
        result = project.runpytest_subprocess("-q")
        result.assert_outcomes(passed=2)
        refs = git(project.path, "for-each-ref", "--format=%(refname)", "refs/less/").stdout
        assert refs.strip() == ""


class TestSelection:
    def test_changed_helper_selects_only_affected_test(self, project):
        run_less(project, "-v")
        with open(project.path / "tests" / "lib.py", "a") as f:
            f.write("\ndef extra():\n    return 3\n")
        result = run_less(project, "-v")
        # test_a is selected; test_b's module was never even collected.
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(
            ["*test_a.py::test_a PASSED*", "*skipped collection of 1 file(s)*"]
        )

    def test_changed_helper_selects_only_the_scanned_import_test(self, project):
        run_less(project, "-v")
        with open(project.path / "tests" / "other.py", "a") as f:
            f.write("\ndef extra():\n    return 3\n")
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_b.py::test_b PASSED*"])

    def test_explicit_file_arguments_still_collect_and_report_correctly(self, project):
        # pytest exempts explicit command-line paths ("init paths") from
        # pytest_ignore_collect: files given as arguments must still be
        # collected, and their deselected tests must not be double-counted
        # in the report.
        run_less(project, "-q", "tests/test_a.py", "tests/test_b.py")
        result = run_less(project, "-q", "tests/test_a.py", "tests/test_b.py")
        result.assert_outcomes(deselected=2)
        assert result.ret == 5
        result.stdout.fnmatch_lines(["selected 0/2 tests:*"])
        # No bogus "skipped collection" claim, no double counting.
        assert "skipped collection" not in result.stdout.str()

    def test_failed_tests_always_rerun(self, project):
        # Make test_a fail (lib.ping no longer matches the assertion).
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 2\n")
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1, failed=1)

        # lib.py is now "unchanged" for a subsequent run, but test_a failed
        # in the recorded run: it must run again, test_b's file must not even
        # be collected.
        result = run_less(project, "-v")
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*test_a.py::test_a FAILED*"])

    def test_new_test_files_always_run(self, project):
        run_less(project, "-q")
        (project.path / "tests" / "test_new.py").write_text("def test_new():\n    assert True\n")
        result = run_less(project, "-q")
        result.assert_outcomes(passed=1)

    def test_conftest_change_reruns_everything(self, project):
        run_less(project, "-q")
        with open(project.path / "tests" / "conftest.py", "a") as f:
            f.write("\nimport pathlib\n")
        result = run_less(project, "-q")
        result.assert_outcomes(passed=2)

    def test_shards_are_keyed_by_command(self, project):
        # Record for the "tests" invocation.
        run_less(project, "-q", "tests")
        # Same command: everything deselected (collection skipped).
        result = run_less(project, "-q", "tests")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])
        # A different invocation: no shard for it, so everything runs.
        result = run_less(project, "-q", "tests/test_a.py")
        result.assert_outcomes(passed=1)


class TestOptions:
    def test_force_runs_all(self, project):
        run_less(project, "-q")
        result = run_less(project, "-q", "--less-force")
        result.assert_outcomes(passed=2)

    def test_clean_discards_data(self, project):
        run_less(project, "-q")
        result = run_less(project, "-q", "--less-clean")
        result.assert_outcomes(passed=2)


class _FakeConfig:
    def __init__(self, ini=""):
        self._ini = ini

    def getini(self, name):
        return self._ini


class TestSyncRemoteParsing:
    def test_parsing(self, monkeypatch):
        from pytest_less.plugin import _sync_remote

        monkeypatch.delenv("PYTEST_LESS_SYNC", raising=False)
        assert _sync_remote(_FakeConfig()) is None
        assert _sync_remote(_FakeConfig("0")) is None
        assert _sync_remote(_FakeConfig("false")) is None
        assert _sync_remote(_FakeConfig("origin")) == ("origin", "both")
        assert _sync_remote(_FakeConfig("pull:origin")) == ("origin", "pull")
        assert _sync_remote(_FakeConfig("PUSH:Upstream")) == ("Upstream", "push")
        # a value that merely contains ':' is a remote name, not a mode
        assert _sync_remote(_FakeConfig("weird:name")) == ("weird:name", "both")

        monkeypatch.setenv("PYTEST_LESS_SYNC", "1")
        assert _sync_remote(_FakeConfig("pull:origin")) == ("origin", "both")  # env wins
        monkeypatch.setenv("PYTEST_LESS_SYNC", "pull:origin")
        assert _sync_remote(_FakeConfig("origin")) == ("origin", "pull")


class TestSyncEndToEnd:
    """Onboarding: PYTEST_LESS_SYNC / less_sync make the plugin itself
    push and pull the shard, no refspecs needed."""

    def test_env_sync_records_pushes_and_pulls_back(self, project, monkeypatch):
        monkeypatch.delenv("PYTEST_LESS_SYNC", raising=False)
        bare = project.path.parent / (project.path.name + "-remote.git")
        bare.mkdir()
        git(bare, "init", "--bare", "-q")
        git(project.path, "remote", "add", "origin", str(bare))
        monkeypatch.setenv("PYTEST_LESS_SYNC", "1")

        # Run 1: records everything and pushes the shard to the remote.
        result = project.runpytest_subprocess("--less", "-q")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*shard pushed to 'origin'*"])
        assert git(bare, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.split()

        # Simulate a fresh clone: no local shard.
        for ref in git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split():
            git(project.path, "update-ref", "-d", ref)

        # Run 2: pulls the remote shard before selection, so instead of a
        # fresh recording run it deselects everything.
        result = project.runpytest_subprocess("--less", "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(
            ["*selected 0/2 tests*", "*pulled a newer shard from 'origin'*"]
        )

    def test_ini_option_enables_sync(self, project, monkeypatch):
        # The env var takes precedence over the ini option: make sure a
        # dogfood run of our own suite with sync enabled doesn't leak in.
        monkeypatch.delenv("PYTEST_LESS_SYNC", raising=False)
        bare = project.path.parent / (project.path.name + "-remote.git")
        bare.mkdir()
        git(bare, "init", "--bare", "-q")
        git(project.path, "remote", "add", "origin", str(bare))
        (project.path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\nless_sync = \"origin\"\n"
        )
        result = project.runpytest_subprocess("--less", "-q")
        result.stdout.fnmatch_lines(["*shard pushed to 'origin'*"])
        assert git(bare, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.split()


    def test_pull_only_mode_never_pushes(self, project, monkeypatch):
        """The CI topology: consume shards read-only, no write token needed."""
        monkeypatch.delenv("PYTEST_LESS_SYNC", raising=False)
        bare = project.path.parent / (project.path.name + "-remote.git")
        bare.mkdir()
        git(bare, "init", "--bare", "-q")
        git(project.path, "remote", "add", "origin", str(bare))

        # Seed the remote with a full record (a developer's pushed shard).
        monkeypatch.setenv("PYTEST_LESS_SYNC", "1")
        project.runpytest_subprocess("--less", "-q")
        seeded = git(bare, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.split()
        assert len(seeded) == 1

        # Simulate a fresh clone (CI): no local shard.
        for ref in git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split():
            git(project.path, "update-ref", "-d", ref)

        # Pull-only run: uses the pushed shard, never writes back.
        monkeypatch.setenv("PYTEST_LESS_SYNC", "pull:origin")
        result = project.runpytest_subprocess("--less", "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*pulled a newer shard from 'origin'*"])
        assert "pushed" not in result.stdout.str()
        # The remote was not touched.
        assert git(bare, "for-each-ref", "--format=%(refname)", "refs/less/").stdout.split() == seeded


class TestFastPath:
    """The no-change rerun: meta.json as first-level cache, data.json never read."""

    def _shard_sha(self, path):
        return git(
            path, "for-each-ref", "--format=%(objectname)", "refs/less/"
        ).stdout.split()[0]

    def test_no_change_rerun_is_fast_and_leaves_shard_untouched(self, project):
        run_less(project, "-q")
        before = self._shard_sha(project.path)
        result = run_less(project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(
            ["*selected 0/2 tests: no recorded dependency changed; fast path*"]
        )
        # Nothing ran: the shard was not even rewritten.
        assert self._shard_sha(project.path) == before

    def test_irrelevant_change_still_takes_fast_path(self, project):
        run_less(project, "-q")
        (project.path / "README.md").write_text("docs change, not a dependency\n")
        result = run_less(project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*fast path*"])

    def test_new_test_file_runs_via_fast_path_and_preserves_old(self, project):
        run_less(project, "-q")
        (project.path / "tests" / "test_new.py").write_text(
            "def test_new():\n    assert True\n"
        )
        result = run_less(project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*fast path*"])
        data = shard_data(project.path)
        tests, _ = decode_shard(data)
        assert set(tests) == {
            "tests/test_a.py::test_a",
            "tests/test_b.py::test_b",
            "tests/test_new.py::test_new",
        }
        # and the meta summary covers the new file
        ref = git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split()[0]
        meta = json.loads(
            git(project.path, "cat-file", "blob", f"{ref}:meta.json").stdout
        )
        assert "tests/test_new.py" in meta["dep_files"]
        assert "tests/test_new.py::test_new" in meta["nodeids"]

    def test_pure_rerun_never_activates_the_monitor(self, project):
        # The distinguishing note of the monitor-less fast path.
        run_less(project, "-q")
        result = run_less(project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*fast path (monitoring not armed*"])

    def test_unmonitored_test_in_pure_run_failure_is_persisted(self, project):
        # The only thing that can run in a pure (monitor-less) rerun is a
        # test that was never recorded -- e.g. lost to an interrupted run.
        # Simulate that: drop test_a from the recorded shard, then make it
        # fail via an untracked non-dependency, non-.py file.  The failure
        # list must be persisted (so the next run re-runs it) while its deps
        # stay as recorded-or-absent.
        # Make test_a's outcome depend on a marker file (committed first so
        # the edit is not itself a change).
        (project.path / "tests" / "test_a.py").write_text(
            "import pathlib\n\ndef test_a():\n"
            "    import lib\n"
            "    assert lib.ping() == 1\n"
            "    assert not pathlib.Path('flag.txt').exists()\n"
        )
        git(project.path, "add", "-A")
        git(project.path, "commit", "-q", "-m", "outcome marker")
        run_less(project, "-q")

        # Simulate the interruption loss: rewrite the shard without test_a.
        ref = git(project.path, "for-each-ref", "--format=%(refname)",
                  "refs/less/").stdout.split()[0]
        data = json.loads(git(project.path, "cat-file", "blob", f"{ref}:data.json").stdout)
        meta = json.loads(git(project.path, "cat-file", "blob", f"{ref}:meta.json").stdout)
        drop = [i for i, b in enumerate(data["blocks"])
                if b[0] == "tests/test_a.py"]
        keep = [i for i in range(len(data["blocks"])) if i not in drop]
        remap = {old: new for new, old in enumerate(keep)}
        data["blocks"] = [data["blocks"][i] for i in keep]
        data["tests"] = {nid: [remap[i] for i in idxs if i in remap]
                         for nid, idxs in data["tests"].items()
                         if nid != "tests/test_a.py::test_a"}
        data["import_deps"] = [remap[i] for i in data["import_deps"] if i in remap]
        meta["nodeids"] = [n for n in meta["nodeids"]
                           if n != "tests/test_a.py::test_a"]
        meta["dep_files"] = [f for f in meta["dep_files"] if f != "tests/test_a.py"]
        from pytest_less.gitstore import GitStore
        GitStore(project.path, meta["shard"]).save(data, meta)

        # Flip the outcome via the untracked marker (non-.py, not a dep:
        # still pure -- no dep changed, no new .py file).
        (project.path / "flag.txt").write_text("x\n")

        result = run_less(project, "-q")
        result.assert_outcomes(failed=1)  # test_a ran (unrecorded) and failed
        result.stdout.fnmatch_lines(["*monitoring not armed*"])
        # ... and the failure is persisted for the next run
        ref = git(project.path, "for-each-ref", "--format=%(refname)",
                  "refs/less/").stdout.split()[0]
        meta = json.loads(git(project.path, "cat-file", "blob", f"{ref}:meta.json").stdout)
        assert meta["failed"] == ["tests/test_a.py::test_a"]
        # next run: test_a is BOTH unrecorded and previously failed; the
        # new-test branch wins the label (it runs either way).
        result = run_less(project, "-q")
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests:*1 new*"])

    def test_changed_dependency_disables_fast_path(self, project):
        run_less(project, "-q")
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 12345\n")
        result = run_less(project, "-q")
        result.assert_outcomes(failed=1)
        assert "fast path" not in result.stdout.str()

    def test_previous_failure_disables_fast_path_then_reenables(self, project):
        run_less(project, "-q")
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 12345\n")
        run_less(project, "-q")  # test_a fails
        result = run_less(project, "-q")  # failed -> rerun (slow path)
        result.assert_outcomes(failed=1)
        assert "fast path" not in result.stdout.str()
        # it passes again after the fix -> meta.failed empty -> fast again
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1\n")
        result = run_less(project, "-q")
        result.stdout.fnmatch_lines(["*1 previously failed*"])


class TestInterruptedRuns:
    """Interrupted runs must never lose a test to collection skipping."""

    def test_interrupted_tests_are_marked_to_rerun(self, project):
        # test_a fails (a real failure), then test_b raises KeyboardInterrupt
        # mid-run: neither result is trustworthy, so both must re-run next
        # time -- test_b especially, whose result is unknown.
        (project.path / "tests" / "test_a.py").write_text(
            "def test_a():\n    import lib\n    assert lib.ping() == 12345\n"
        )
        (project.path / "tests" / "test_b.py").write_text(
            "def test_b():\n    raise KeyboardInterrupt\n"
        )
        result = run_less(project, "-q")
        assert result.ret in (1, 2)
        ref = git(project.path, "for-each-ref", "--format=%(refname)",
                  "refs/less/").stdout.split()[0]
        meta = json.loads(git(project.path, "cat-file", "blob", f"{ref}:meta.json").stdout)
        assert set(meta["failed"]) == {
            "tests/test_a.py::test_a", "tests/test_b.py::test_b",
        }
        # and both stay in the recorded node ids (with deps for the one that
        # did finish, an entry for the interrupted one)
        assert set(meta["nodeids"]) == {
            "tests/test_a.py::test_a", "tests/test_b.py::test_b",
        }
        # next run: previously-failed/unfinished re-run (fast path disabled)
        result = run_less(project, "-q")
        result.stdout.fnmatch_lines(["*2 previously failed*"])

    def test_interrupted_first_run_never_loses_a_test(self, project):
        # Interrupt the very first run mid-test: the unfinished test keeps a
        # shard entry (empty deps) and is must-rerun, so the next run cannot
        # collection-skip it away.
        (project.path / "tests" / "test_b.py").write_text(
            "def test_b():\n    raise KeyboardInterrupt\n"
        )
        result = run_less(project, "-q")
        assert result.ret in (1, 2)
        ref = git(project.path, "for-each-ref", "--format=%(refname)",
                  "refs/less/").stdout.split()[0]
        meta = json.loads(git(project.path, "cat-file", "blob", f"{ref}:meta.json").stdout)
        assert "tests/test_b.py::test_b" in meta["nodeids"]
        assert "tests/test_b.py::test_b" in meta["failed"]
        # fix the interruption and run again: test_b re-runs
        (project.path / "tests" / "test_b.py").write_text(
            "def test_b():\n    assert True\n"
        )
        git(project.path, "add", "-A")
        git(project.path, "commit", "-q", "-m", "fix interruption")
        result = run_less(project, "-q")
        # test_b (previously interrupted -> must-rerun) runs and passes;
        # test_a finished in the interrupted run with unchanged deps, so it
        # stays deselected.
        result.assert_outcomes(passed=1)


class TestBlockGranularity:
    """The point of block-level deps: edit granularity decides re-runs."""

    def test_body_edit_of_one_test_reruns_only_that_test(self, project):
        # File-level tracking re-ran the whole file's tests on any edit;
        # block-level re-runs only the test whose function body changed.
        run_less(project, "-q")
        content = (project.path / "tests" / "test_a.py").read_text()
        (project.path / "tests" / "test_a.py").write_text(
            content.replace("assert lib.ping() == 1", "assert lib.ping() == 1\n    marker = 1")
        )
        result = run_less(project, "-q")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*1 affected*"])

    def test_comment_edit_of_helper_reruns_nothing(self, project):
        run_less(project, "-q")
        (project.path / "tests" / "lib.py").write_text(
            "# a comment change only\ndef ping():\n    return 1\n"
        )
        result = run_less(project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_body_edit_of_dead_imported_helper_does_not_rerun(self, project):
        # test_b imports `other` in a dead branch: its recorded dep is
        # other's SKELETON. Editing other.pong's BODY must not re-run test_b
        # (file-level tracking would have).
        run_less(project, "-q")
        (project.path / "tests" / "other.py").write_text(
            "def pong():\n    return 99\n"  # body edit
        )
        result = run_less(project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_module_level_edit_of_dead_imported_helper_reruns(self, project):
        run_less(project, "-q")
        (project.path / "tests" / "other.py").write_text(
            "NEW_CONSTANT = 1\ndef pong():\n    return 2\n"  # module-level edit
        )
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_b.py::test_b PASSED*"])

    def test_body_edit_of_executed_helper_reruns_only_its_caller(self, project):
        # lib.ping's body executes inside test_a only.  (The edited body is
        # deliberately a different byte length: a same-length edit inside the
        # same mtime second could otherwise be served from a stale .pyc.)
        run_less(project, "-q")
        (project.path / "tests" / "lib.py").write_text(
            "def ping():\n    return 12345\n"
        )
        result = run_less(project, "-q")
        # test_a re-runs (and fails: the assertion now mismatches).
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*FAILED tests/test_a.py::test_a*"])

    def test_conftest_body_edit_reruns_everything_import_time(self, project):
        # Anything that executes at import time (here: a function body in a
        # conftest called at module level) is a shared import-time dep: a
        # body edit re-runs the world.
        run_less(project, "-q")
        (project.path / "tests" / "conftest.py").write_text(
            "def helper():\n    return 2  # body edit\n\nVAL = helper()\n"
        )
        result = run_less(project, "-q")
        result.assert_outcomes(passed=2)  # everything re-ran (import-time stale)


class TestSelfHosting:
    """The plugin running *inside* the plugin's own pytest session, in one
    process: per-config state and dynamic sys.monitoring tool ids must keep the
    nested run independent of the outer one (and vice versa)."""

    def test_inprocess_nested_less_run(self, project):
        # In-process (not subprocess): the inner session shares this process
        # and, when the outer suite is itself run with --less, a monitor.
        result = project.runpytest("--less", "-q")
        result.assert_outcomes(passed=2)
        result = project.runpytest("--less", "-q")
        # Nothing changed in the project repo: deselect everything.
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])
        # The nested run stored its shard in the project repo, not ours.
        refs = git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split()
        assert len(refs) == 1


class TestSelectorsDisableSelection:
    """Explicitly requested tests must never be silently skipped."""

    def test_explicit_node_id_runs_despite_no_changes(self, project):
        nodeid = "tests/test_a.py::test_a"
        run_less(project, "-q", nodeid)
        result = run_less(project, "-q", nodeid)
        # Unaffected by file changes, but explicitly requested: must run.
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selectors*running everything*"])

    def test_k_selector_disables_selection(self, project):
        run_less(project, "-q", "-k", "test_a or test_b")
        result = run_less(project, "-q", "-k", "test_a or test_b")
        # Nothing changed, so without the selector rule both would be
        # deselected; explicitly selected tests must run.
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*selectors*running everything*"])

    def test_selector_run_still_records(self, project):
        run_less(project, "-q")
        run_less(project, "-q", "-k", "test_a or test_b")
        # The selector run is a different command (different shard), but it
        # still collected and recorded everything it ran.
        refs = git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split()
        assert len(refs) == 2  # one shard per command
        merged: dict = {}
        for ref in refs:
            merged.update(
                json.loads(git(project.path, "cat-file", "blob", f"{ref}:data.json").stdout)[
                    "tests"
                ]
            )
        assert set(merged) == {
            "tests/test_a.py::test_a",
            "tests/test_b.py::test_b",
        }


def run_outside(project, *args):
    """Run ``pytest --less`` without ``runpytest_subprocess``, which creates a
    numbered ``runpytest-N/`` directory *inside* the repository on every call --
    each of which is (correctly) a new-directory event."""
    import sys

    return project.run(sys.executable, "-m", "pytest", "--less", *args)


def git_subcommands(calls: list[str]) -> list[str]:
    """The git subcommand of each logged call (lines look like
    ``-C <dir> [--option ...] <sub> ...``)."""
    return [next(w for w in line.split()[2:] if not w.startswith("-")) for line in calls]


# Change detection compares file fingerprints (fingerprint.py) instead of asking
# ``git diff`` / ``git status`` / ``git ls-files --others``: 98 + 790 ms on a
# 10k-file repository.  None of these may appear in a run, whatever it does.
NEVER_ASKED = {"diff", "status", "diff-index", "diff-files", "add"}


class TestGitSpawnBudget:
    """Every git call is a process spawn: ~10 ms from a small process, several
    times that from inside a large pytest process, a quarter of a second where
    endpoint scanning slows exec.  The shard is read and written through a
    handful of plumbing commands, and change detection compares fingerprints.
    Pin the shape: a small constant per run -- never per file or per test --
    and git is only asked, once and about those paths alone, when a *new* file
    or directory appears."""

    @pytest.fixture
    def counted(self, project, tmp_path, monkeypatch):
        import os
        import shutil

        real = shutil.which("git")
        log = tmp_path / "gitcalls.log"
        shim = tmp_path / "shim"
        shim.mkdir()
        script = shim / "git"
        script.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexec "{real}" "$@"\n')
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")

        def calls():
            text = log.read_text() if log.exists() else ""
            log.write_text("")
            return [line for line in text.splitlines() if line]

        return calls

    def test_a_cold_record_has_a_small_constant_budget(self, project, counted):
        counted()
        result = run_outside(project, "-q")
        result.assert_outcomes(passed=2)
        subs = git_subcommands(counted())
        assert len(subs) <= 15, subs  # measured: 13 (4 reads, 7 for the shard write, 2 rev-parse)
        assert not NEVER_ASKED & set(subs), subs
        assert "ls-files" not in subs, subs

    def test_no_change_rerun_only_reads_the_shard(self, project, counted):
        run_outside(project, "-q")
        counted()  # discard the record run
        result = run_outside(project, "-q")
        assert result.ret == 5
        subs = git_subcommands(counted())
        assert len(subs) <= 4, subs  # measured: 3 (toplevel + two shard blobs)
        assert not NEVER_ASKED & set(subs) and "ls-files" not in subs, subs
        assert not {"hash-object", "mktree", "commit-tree", "update-ref"} & set(subs), subs

    def test_run_that_selects_nothing_rewrites_nothing(self, project, counted):
        run_outside(project, "-q")
        counted()
        # other.py is a dead-branch dependency: editing it stales nothing.
        (project.path / "tests" / "other.py").write_text("def pong():\n    return 3\n")
        result = run_outside(project, "-q")
        assert result.ret == 5
        subs = git_subcommands(counted())
        assert len(subs) <= 6, subs  # measured: 5
        assert not NEVER_ASKED & set(subs) and "ls-files" not in subs, subs
        assert not {"hash-object", "mktree", "commit-tree", "update-ref"} & set(subs), subs

    def test_an_edit_that_runs_tests_and_re_records_has_the_record_budget(
        self, project, counted
    ):
        run_outside(project, "-q")
        counted()
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1 + int(0)\n")
        result = run_outside(project, "-q")
        result.assert_outcomes(passed=1)
        subs = git_subcommands(counted())
        assert len(subs) <= 15, subs  # measured: 13
        assert not NEVER_ASKED & set(subs) and "ls-files" not in subs, subs

    def test_a_new_file_costs_one_scoped_query(self, project, counted):
        run_outside(project, "-q")
        counted()
        (project.path / "tests" / "extra.py").write_text("X = 1\n")
        run_outside(project, "-q")
        calls = counted()
        queries = [c for c in calls if " ls-files " in f" {c} "]
        assert len(queries) == 1, calls
        assert "tests/extra.py" in queries[0]
        assert not NEVER_ASKED & set(git_subcommands(calls)), calls

    def test_an_ignored_new_file_is_confirmed_once_and_not_reported(self, project, counted):
        (project.path / ".gitignore").write_text(
            ".pytest_cache/\n__pycache__/\ngenerated_*.py\n"
        )
        commit(project.path, "ignore generated")
        run_outside(project, "-q")
        counted()
        (project.path / "tests" / "generated_version.py").write_text("V = 1\n")
        result = run_outside(project, "-q")
        assert result.ret == 5  # ignored: no candidate, nothing selected
        subs = git_subcommands(counted())
        assert subs.count("ls-files") == 1, subs
        assert not NEVER_ASKED & set(subs), subs


class TestHermeticEnvironment:
    def test_the_outer_plugin_environment_does_not_leak_into_inner_runs(self):
        import os

        for name in ("PYTEST_LESS", "PYTEST_LESS_SYNC", "PYTEST_LESS_PARSE_CACHE"):
            assert name not in os.environ
