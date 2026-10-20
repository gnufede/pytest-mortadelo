"""Content-hash-keyed parse cache (``parse.json`` in the shard).

``git diff`` says which files changed for *staleness*, but resolving an
executed code object to a block needs the block table of every file whose
code runs.  Without a cache an incremental run re-parses hundreds of
untouched files; with it, reading + hashing the bytes (~0.1ms) replaces
parse + hashing (~10ms).  Every entry is verified against the current bytes,
so the cache can serve a stale table to nobody.
"""

# ruff: noqa: F811  (the shared ``project`` fixture is imported, then requested by name)
import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
from test_plugin import commit, git, project, run_less  # noqa: F401

from pytest_less.blocks import BlockCache, ParseCache

SRC = (
    '"mod"\n'
    "class C:\n"
    '    "cls"\n'
    "    def m(self):\n"
    '        "meth"\n'
    "        return getattr(self, name)\n"
    "def f():\n"
    "    return 1\n"
)


@pytest.fixture(autouse=True, params=["bytecode", "ast"])
def hash_algo(request, monkeypatch):
    """Every test here runs under both block-hashing algorithms."""
    monkeypatch.setenv("PYTEST_LESS_HASH", request.param)
    return request.param


def fresh(source, *, cosmetic=True):
    """The table the active algorithm computes for *source*, with no cache:
    what a cached table must equal."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "o.py"
        path.write_bytes(source if isinstance(source, bytes) else source.encode())
        return BlockCache(Path(d), cosmetic=cosmetic).file_blocks("o.py")


def fields(fb):
    return (fb.blocks, fb.spans, fb.markers, fb.bdocs, fb.reads_docs)


def make(tmp_path, source=SRC, *, cosmetic=True, stored=None):
    (tmp_path / "m.py").write_text(source)
    cache = ParseCache((lambda: stored) if stored is not None else None, cosmetic=cosmetic)
    return BlockCache(tmp_path, cosmetic=cosmetic, parse_cache=cache), cache


class TestRoundTrip:
    def test_hit_returns_an_identical_table(self, tmp_path):
        first, cache = make(tmp_path)
        fb = first.file_blocks("m.py")
        assert (first.hits, first.misses) == (0, 1)
        stored = cache.dump({"m.py"})

        second, _ = make(tmp_path, stored=stored)
        again = second.file_blocks("m.py")
        assert (second.hits, second.misses) == (1, 0)
        assert fields(again) == fields(fb)
        assert again.reads_docs is False and again.markers  # getattr(self, name)

    def test_cached_table_equals_a_fresh_parse(self, tmp_path):
        first, cache = make(tmp_path)
        first.file_blocks("m.py")
        second, _ = make(tmp_path, stored=cache.dump({"m.py"}))
        assert fields(second.file_blocks("m.py")) == fields(fresh(SRC))

    def test_edited_content_is_a_miss_never_a_stale_hit(self, tmp_path):
        first, cache = make(tmp_path)
        first.file_blocks("m.py")
        stored = cache.dump({"m.py"})
        edited = SRC.replace("return 1", "return 2")
        second, _ = make(tmp_path, source=edited, stored=stored)
        fb = second.file_blocks("m.py")
        assert (second.hits, second.misses) == (0, 1)
        assert fb.blocks["f"] == fresh(edited).blocks["f"]

    def test_reverting_to_cached_content_hits_again(self, tmp_path):
        first, cache = make(tmp_path)
        first.file_blocks("m.py")
        stored = cache.dump({"m.py"})
        third, _ = make(tmp_path, stored=stored)  # same bytes as the entry
        third.file_blocks("m.py")
        assert third.hits == 1

    def test_crlf_source_matches_read_text_semantics(self, tmp_path):
        crlf = 'X = """a\r\nb"""\r\ndef f():\r\n    return X\r\n'
        (tmp_path / "w.py").write_bytes(crlf.encode())
        cache = ParseCache(None, cosmetic=True)
        bc = BlockCache(tmp_path, cosmetic=True, parse_cache=cache)
        got = bc.file_blocks("w.py")
        # the same file with universal newlines applied up front
        expected = fresh(crlf.replace("\r\n", "\n"))
        assert fields(got) == fields(expected)


class TestValidity:
    def test_other_python_version_or_cosmetic_flag_is_ignored(self, tmp_path):
        first, cache = make(tmp_path, cosmetic=True)
        first.file_blocks("m.py")
        stored = cache.dump({"m.py"})

        other_mode, _ = make(tmp_path, cosmetic=False, stored=stored)
        other_mode.file_blocks("m.py")
        assert other_mode.hits == 0  # cosmetic flag differs

        stored_py = {**stored, "py": "2.7"}
        other_py, _ = make(tmp_path, stored=stored_py)
        other_py.file_blocks("m.py")
        assert other_py.hits == 0

    def test_old_algorithm_version_is_ignored(self, tmp_path):
        first, cache = make(tmp_path)
        first.file_blocks("m.py")
        stored = {**cache.dump({"m.py"}), "v": -1}
        again, _ = make(tmp_path, stored=stored)
        again.file_blocks("m.py")
        assert again.hits == 0

    @pytest.mark.parametrize(
        "garbage",
        [None, [], "x", {"files": 3}, {"v": 1}, {"v": 1, "files": {"m.py": {"h": "x"}}}],
    )
    def test_corrupt_blob_degrades_to_a_parse(self, tmp_path, garbage):
        bc, _ = make(tmp_path, stored=garbage)
        assert bc.file_blocks("m.py") is not None
        assert bc.hits == 0

    def test_corrupt_entry_with_matching_hash_degrades_to_a_parse(self, tmp_path):
        first, cache = make(tmp_path)
        first.file_blocks("m.py")
        stored = cache.dump({"m.py"})
        stored["files"]["m.py"]["b"] = "garbage"
        again, _ = make(tmp_path, stored=stored)
        assert again.file_blocks("m.py") is not None
        assert again.hits == 0


class TestDump:
    def test_nothing_new_means_carry_the_blob_over(self, tmp_path):
        first, cache = make(tmp_path)
        first.file_blocks("m.py")
        stored = cache.dump({"m.py"})
        second, cache2 = make(tmp_path, stored=stored)
        second.file_blocks("m.py")
        assert cache2.dump({"m.py"}) is None  # all hits, same file set

    def test_unused_files_are_pruned(self, tmp_path):
        first, cache = make(tmp_path)
        first.file_blocks("m.py")
        stored = cache.dump({"m.py"})
        second, cache2 = make(tmp_path, stored=stored)
        second.file_blocks("m.py")
        pruned = cache2.dump(set())  # m.py no longer a dependency
        assert pruned is not None and pruned["files"] == {}

    def test_no_blob_and_nothing_to_store_is_a_noop(self):
        assert ParseCache(None).dump(set()) is None


class TestShard:
    def test_gitstore_carries_the_parse_blob_over(self, tmp_path):
        from pytest_less.gitstore import GitStore

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        store = GitStore(tmp_path, "k")
        store.save({"version": 2}, {"version": 1}, parse={"v": 1, "files": {"a.py": {}}})
        assert store.load_parse() == {"v": 1, "files": {"a.py": {}}}
        store.save({"version": 2, "x": 1}, {"version": 1})  # parse=None
        assert store.load_parse() == {"v": 1, "files": {"a.py": {}}}
        assert store.load_data() == {"version": 2, "x": 1}
        store.save({"version": 2}, {"version": 1}, parse={"v": 1, "files": {}})
        assert store.load_parse() == {"v": 1, "files": {}}

    def test_shard_without_parse_blob_loads_as_none(self, tmp_path):
        from pytest_less.gitstore import GitStore

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        store = GitStore(tmp_path, "k")
        store.save({"version": 2}, {"version": 1})
        assert store.load_parse() is None


def read_parse(project):
    ref = git(
        project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
    ).stdout.split()[0]
    return json.loads(git(project.path, "cat-file", "blob", f"{ref}:parse.json").stdout)


def parse_note(result):
    m = re.search(r"parse cache: (\d+) hit\(s\)(?: \(\d+ from the repository cache\))?, (\d+) file\(s\) parsed", result.stdout.str())
    return (int(m.group(1)), int(m.group(2))) if m else None


class TestEndToEnd:
    def test_record_stores_the_dependency_files(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        parse = read_parse(project)
        assert set(parse["files"]) >= {
            "tests/lib.py",
            "tests/other.py",
            "tests/test_a.py",
            "tests/test_b.py",
        }
        assert parse["cosmetic"] is True

    def test_incremental_run_reparses_only_what_changed(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1 + int(0)\n")
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        hits, parsed = parse_note(result)
        assert parsed == 1  # lib.py (its bytes changed)
        assert hits >= 1  # conftest / test files came from the cache

    def test_new_content_is_cached_for_the_next_run(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1 + int(0)\n")
        run_less(project, "-q").assert_outcomes(passed=1)
        # lib.py's new content was cached by the run above.  Now change a
        # different executed file: only that one is parsed, lib.py is a hit.
        (project.path / "tests" / "test_b.py").write_text(
            "def test_b(always_false=False):\n"
            "    if always_false:\n"
            "        import other\n"
            "    assert 2 == 1 + int(1)\n"
        )
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        hits, parsed = parse_note(result)
        assert parsed == 1  # test_b.py only
        assert hits >= 1  # lib.py (new content, cached) among them

    def test_nothing_ran_leaves_the_shard_untouched(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        ref = git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split()[0]
        before = git(project.path, "rev-parse", ref).stdout
        # other.py is only a dead-branch dependency: editing its body stales
        # nothing, so nothing runs -- and nothing is rewritten.
        (project.path / "tests" / "other.py").write_text("def pong():\n    return 3\n")
        result = run_less(project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*nothing ran; stored shard left unchanged*"])
        assert git(project.path, "rev-parse", ref).stdout == before

    def test_no_change_rerun_keeps_the_parse_blob(self, project):
        run_less(project, "-q")
        ref = git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split()[0]
        before = git(project.path, "rev-parse", f"{ref}:parse.json").stdout
        result = run_less(project, "-q")
        assert result.ret == 5  # fast path: nothing ran
        after = git(project.path, "rev-parse", f"{ref}:parse.json").stdout
        assert before == after

    def test_failure_update_path_keeps_the_parse_blob(self, project):
        (project.path / "tests" / "test_b.py").write_text(
            "def test_b():\n    assert False\n"
        )
        commit(project.path, "make test_b fail")
        run_less(project, "-q")  # records, test_b fails
        ref = git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split()[0]
        before = git(project.path, "rev-parse", f"{ref}:parse.json").stdout
        run_less(project, "-q")  # reruns the failed test (monitor-less update)
        after = git(project.path, "rev-parse", f"{ref}:parse.json").stdout
        assert before == after


class TestKillSwitch:
    """``PYTEST_LESS_PARSE_CACHE=0``: no read, no write, blob untouched."""

    @staticmethod
    def _ref(project):
        return git(
            project.path, "for-each-ref", "--format=%(refname)", "refs/less/"
        ).stdout.split()[0]

    def test_first_record_stores_no_parse_blob(self, project, monkeypatch):
        monkeypatch.setenv("PYTEST_LESS_PARSE_CACHE", "0")
        run_less(project, "-q").assert_outcomes(passed=2)
        ref = self._ref(project)
        with pytest.raises(subprocess.CalledProcessError):
            git(project.path, "cat-file", "-e", f"{ref}:parse.json")

    def test_no_read_no_write_and_existing_blob_is_carried_over(
        self, project, monkeypatch
    ):
        run_less(project, "-q").assert_outcomes(passed=2)  # warm the cache
        ref = self._ref(project)
        before = git(project.path, "rev-parse", f"{ref}:parse.json").stdout.strip()

        (project.path / "tests" / "lib.py").write_text("def ping():\n    return 1 + int(0)\n")
        monkeypatch.setenv("PYTEST_LESS_PARSE_CACHE", "0")
        result = run_less(project, "-v")
        result.assert_outcomes(passed=1)
        # The cache is neither consulted nor reported while disabled.
        assert parse_note(result) is None
        after = git(project.path, "rev-parse", f"{ref}:parse.json").stdout.strip()
        assert after == before


# ---------------------------------------------------------------------------
# The repository-wide store: every shard of a repository shares what any of
# them parsed (one tiny file per content hash under the git directory).


class TestLocalStore:
    @pytest.fixture
    def store(self, tmp_path):
        from pytest_less.blocks import LocalParseStore

        return LocalParseStore(tmp_path / "store")

    def entry(self, source=SRC):
        from pytest_less.blocks import _file_to_json

        return _file_to_json(fresh(source))

    def test_put_get_round_trip_and_atomic_layout(self, store):
        assert store.get("ab" * 16) is None
        assert store.put("ab" * 16, {"x": 1}) is True
        assert store.get("ab" * 16) == {"x": 1}
        assert store.put("ab" * 16, {"x": 2}) is False  # never overwritten
        assert store.get("ab" * 16) == {"x": 1}
        leftovers = list(store.directory.rglob(".tmp-*"))
        assert leftovers == []

    def test_corrupt_file_is_a_miss(self, store):
        store.put("cd" * 16, {"x": 1})
        (store.directory / "cd" / f"{'cd' * 16}.json").write_text("{not json")
        assert store.get("cd" * 16) is None

    def test_eviction_keeps_the_newest(self, store):
        import os

        digests = [f"{i:02x}" * 16 for i in range(10)]
        for n, digest in enumerate(digests):
            store.put(digest, {"n": n})
            os.utime(store.directory / digest[:2] / f"{digest}.json", (1000 + n, 1000 + n))
        assert store.evict(limit=5) == 6  # down to 90% of the limit: 4 left
        assert store.get(digests[0]) is None and store.get(digests[-1]) == {"n": 9}

    def test_python_version_and_mode_get_separate_directories(self, tmp_path):
        from pytest_less.blocks import LocalParseStore

        a = LocalParseStore.at(tmp_path, cosmetic=True).directory
        b = LocalParseStore.at(tmp_path, cosmetic=False).directory
        assert a != b and "pytest-less" in a.parts


class TestSharedAcrossShards:
    def cache(self, store, stored=None, cosmetic=True):
        return ParseCache((lambda: stored) if stored else None, cosmetic=cosmetic, local=store)

    def test_a_second_shard_reuses_what_the_first_parsed(self, tmp_path):
        from pytest_less.blocks import LocalParseStore

        store = LocalParseStore(tmp_path / "store")
        (tmp_path / "w").mkdir()
        (tmp_path / "w" / "m.py").write_text(SRC)
        first = self.cache(store)
        a = BlockCache(tmp_path / "w", cosmetic=True, parse_cache=first)
        fb = a.file_blocks("m.py")
        assert (a.hits, a.misses) == (0, 1)
        assert first.flush_local() == 1

        # a different shard: its own (empty) parse.json -- yet no parse needed
        second = self.cache(store)
        b = BlockCache(tmp_path / "w", cosmetic=True, parse_cache=second)
        again = b.file_blocks("m.py")
        assert (b.hits, b.misses, second.local_hits) == (1, 0, 1)
        assert fields(again) == fields(fb)
        # and it is adopted into that shard's own blob
        assert second.dump({"m.py"})["files"]["m.py"]["h"] == first.used["m.py"]
        assert second.flush_local() == 0  # nothing new to publish

    def test_the_same_bytes_at_another_path_share_an_entry(self, tmp_path):
        from pytest_less.blocks import LocalParseStore

        store = LocalParseStore(tmp_path / "store")
        for name in ("one", "two"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "m.py").write_text(SRC)
        first = self.cache(store)
        BlockCache(tmp_path / "one", cosmetic=True, parse_cache=first).file_blocks("m.py")
        first.flush_local()
        second = self.cache(store)
        other = BlockCache(tmp_path / "two", cosmetic=True, parse_cache=second)
        other.file_blocks("m.py")
        assert other.misses == 0

    def test_changed_bytes_are_a_miss_never_a_stale_hit(self, tmp_path):
        from pytest_less.blocks import LocalParseStore

        store = LocalParseStore(tmp_path / "store")
        (tmp_path / "w").mkdir()
        (tmp_path / "w" / "m.py").write_text(SRC)
        first = self.cache(store)
        BlockCache(tmp_path / "w", cosmetic=True, parse_cache=first).file_blocks("m.py")
        first.flush_local()
        (tmp_path / "w" / "m.py").write_text(SRC.replace("return 1", "return 2"))
        second = self.cache(store)
        b = BlockCache(tmp_path / "w", cosmetic=True, parse_cache=second)
        fb = b.file_blocks("m.py")
        assert (b.hits, b.misses) == (0, 1)
        assert fb.blocks["f"] == fresh(SRC.replace("return 1", "return 2")).blocks["f"]

    def test_a_corrupt_local_entry_degrades_to_a_parse(self, tmp_path):
        from pytest_less.blocks import LocalParseStore, _content_hash

        store = LocalParseStore(tmp_path / "store")
        (tmp_path / "w").mkdir()
        (tmp_path / "w" / "m.py").write_text(SRC)
        digest = _content_hash(SRC.encode())
        store.put(digest, {"b": "garbage"})
        b = BlockCache(tmp_path / "w", cosmetic=True, parse_cache=self.cache(store))
        assert b.file_blocks("m.py") is not None and b.hits == 0


class TestSharedEndToEnd:
    @pytest.fixture(autouse=True)
    def _in_process_layer(self, monkeypatch):
        # The store lives under the git directory, which only the in-process
        # layer knows; with PYTEST_LESS_GIT_BACKEND=subprocess there is none.
        monkeypatch.delenv("PYTEST_LESS_GIT_BACKEND", raising=False)

    def test_a_different_command_does_not_re_parse(self, project):
        run_less(project, "-q").assert_outcomes(passed=2)
        # another shard (the argument is part of the key): cold shard, warm repo
        result = run_less(project, "-v", "tests/test_a.py")
        result.assert_outcomes(passed=1)
        hits, parsed = parse_note(result)
        assert parsed == 0 and hits >= 1
        result.stdout.fnmatch_lines(["*from the repository cache*"])

    def test_the_store_lives_inside_the_git_directory(self, project):
        run_less(project, "-q")
        entries = list((project.path / ".git" / "pytest-less").rglob("*.json"))
        assert entries, "expected parse entries under .git/pytest-less"
        # nothing leaked into the working tree
        status = git(project.path, "status", "--porcelain", "-uall").stdout
        assert "pytest-less" not in status

    def test_the_kill_switch_disables_it_too(self, project, monkeypatch):
        monkeypatch.setenv("PYTEST_LESS_PARSE_CACHE", "0")
        run_less(project, "-q").assert_outcomes(passed=2)
        assert not (project.path / ".git" / "pytest-less").exists()
