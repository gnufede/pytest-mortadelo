"""The in-process git layer, checked against real git.

``gitobjects`` replaces a handful of ``git`` spawns (toplevel, HEAD, refs,
blob reads, the shard write) with file reads and writes.  Its contract is
narrow and these tests pin both halves: where it applies it must agree with
git byte for byte, and wherever it does not (packed objects, odd layouts, a
held lock) it must say so -- never guess -- so the caller asks git.
"""

import os
import subprocess
import threading
import zlib

import pytest

from pytest_less.gitobjects import Repo, Unsupported, toplevel
from pytest_less.gitstore import GitStore


def git(path, *args, stdin=None, check=True, env=None):
    proc = subprocess.run(
        ["git", "-C", str(path), *args],
        input=stdin,
        capture_output=True,
        check=check,
        env=None if env is None else {**os.environ, **env},
    )
    return proc.stdout


def out(path, *args):
    return git(path, *args).decode().strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "T")
    (path / "f.txt").write_text("x")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "init")
    return path


def make_shard_with_git(path, ref, files):
    """The way GitStore used to do it: hash-object / mktree / commit-tree."""
    shas = {
        n: git(path, "hash-object", "-w", "--stdin", stdin=d).decode().strip()
        for n, d in files.items()
    }
    tree_in = "".join(f"100644 blob {shas[n]}\t{n}\n" for n in sorted(shas)).encode()
    tree = git(path, "mktree", stdin=tree_in).decode().strip()
    commit = git(path, "commit-tree", tree, "-m", "m").decode().strip()
    git(path, "update-ref", ref, commit)
    return commit, tree


FILES = {
    "data.json": b'{"k": [1, 2, 3]}' * 5000,
    "meta.json": b'{"created": "2026-01-01T00:00:00+00:00"}',
    "parse.json": os.urandom(4096),  # not even valid JSON: bytes are bytes
}


class TestDiscovery:
    def test_plain_repo_and_subdirectory(self, repo):
        sub = repo / "a" / "b"
        sub.mkdir(parents=True)
        for start in (repo, sub):
            found = Repo.discover(start)
            assert found is not None
            assert str(found.worktree) == out(start, "rev-parse", "--show-toplevel")
            assert toplevel(start) == found.worktree

    def test_not_a_repository(self, tmp_path):
        assert Repo.discover(tmp_path) is None
        assert toplevel(tmp_path) is None

    def test_linked_worktree_shares_objects_and_refs(self, repo, tmp_path):
        linked = tmp_path / "linked"
        git(repo, "worktree", "add", "-q", "-b", "other", str(linked))
        found = Repo.discover(linked)
        assert found is not None
        assert str(found.worktree) == out(linked, "rev-parse", "--show-toplevel")
        assert found.commondir == (repo / ".git").resolve()
        assert found.gitdir != found.commondir  # HEAD is per worktree
        assert found.head() == out(linked, "rev-parse", "HEAD")
        # a shard written from the worktree is visible from the main repo
        found.write_shard("refs/less/w", {"meta.json": b"{}"}, "m")
        assert out(repo, "cat-file", "blob", "refs/less/w:meta.json") == "{}"

    @pytest.mark.parametrize("name", ["GIT_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY"])
    def test_relocating_environment_is_unsupported(self, repo, monkeypatch, name):
        monkeypatch.setenv(name, str(repo / ".git"))
        assert Repo.discover(repo) is None

    def test_sha256_repository_is_unsupported(self, tmp_path):
        path = tmp_path / "sha256"
        path.mkdir()
        proc = subprocess.run(
            ["git", "init", "-q", "--object-format=sha256", str(path)],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            pytest.skip("this git cannot create sha256 repositories")
        assert Repo.discover(path) is None

    def test_reftable_repository_is_unsupported(self, tmp_path):
        path = tmp_path / "reftable"
        path.mkdir()
        proc = subprocess.run(
            ["git", "init", "-q", "--ref-format=reftable", str(path)],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            pytest.skip("this git cannot create reftable repositories")
        assert Repo.discover(path) is None


class TestRefs:
    def test_head_matches_git(self, repo):
        found = Repo.discover(repo)
        assert found.head() == out(repo, "rev-parse", "HEAD")

    def test_detached_head(self, repo):
        sha = out(repo, "rev-parse", "HEAD")
        git(repo, "checkout", "-q", "--detach", sha)
        assert Repo.discover(repo).head() == sha

    def test_unborn_branch_has_no_head(self, tmp_path):
        path = tmp_path / "empty"
        path.mkdir()
        git(path, "init", "-q")
        assert Repo.discover(path).head() is None

    def test_loose_and_packed_refs(self, repo):
        sha = out(repo, "rev-parse", "HEAD")
        git(repo, "update-ref", "refs/less/one", sha)
        found = Repo.discover(repo)
        assert found.read_ref("refs/less/one") == sha
        git(repo, "pack-refs", "--all", "--prune")
        assert not (repo / ".git" / "refs" / "less" / "one").exists()
        assert found.read_ref("refs/less/one") == sha  # now only in packed-refs
        assert found.read_ref("refs/less/none") is None
        assert found.head() == sha  # branch ref also packed

    def test_loose_ref_shadows_a_packed_one(self, repo):
        first = out(repo, "rev-parse", "HEAD")
        git(repo, "update-ref", "refs/less/x", first)
        git(repo, "pack-refs", "--all", "--prune")
        (repo / "g.txt").write_text("y")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "two")
        second = out(repo, "rev-parse", "HEAD")
        Repo.discover(repo)._update_ref("refs/less/x", second)
        assert out(repo, "rev-parse", "refs/less/x") == second  # git agrees


class TestReadingObjects:
    def test_blobs_match_git_byte_for_byte(self, repo):
        make_shard_with_git(repo, "refs/less/r", FILES)
        found = Repo.discover(repo)
        for name, data in FILES.items():
            assert found.blob_in_ref("refs/less/r", name) == data
        assert found.blob_in_ref("refs/less/r", "absent.json") is None
        assert found.blob_in_ref("refs/less/none", "meta.json") is None

    def test_blob_sha_matches_git(self, repo):
        make_shard_with_git(repo, "refs/less/r", FILES)
        found = Repo.discover(repo)
        for name in FILES:
            assert found.blob_sha_in_ref("refs/less/r", name) == out(
                repo, "rev-parse", f"refs/less/r:{name}"
            )

    def test_packed_objects_are_reported_not_guessed(self, repo):
        make_shard_with_git(repo, "refs/less/r", FILES)
        git(repo, "gc", "-q", "--prune=now")
        found = Repo.discover(repo)
        with pytest.raises(Unsupported):
            found.blob_in_ref("refs/less/r", "meta.json")

    def test_corrupt_object_is_never_returned(self, repo):
        make_shard_with_git(repo, "refs/less/r", FILES)
        found = Repo.discover(repo)
        sha = found.blob_sha_in_ref("refs/less/r", "meta.json")
        path = repo / ".git" / "objects" / sha[:2] / sha[2:]
        path.chmod(0o644)
        path.write_bytes(b"not zlib at all")
        # zlib.error is one of the exceptions GitStore falls back on
        with pytest.raises(zlib.error):
            found.blob_in_ref("refs/less/r", "meta.json")


class TestWriting:
    def test_tree_id_is_identical_to_git_mktree(self, repo):
        _, git_tree = make_shard_with_git(repo, "refs/less/g", FILES)
        found = Repo.discover(repo)
        found.write_shard("refs/less/p", FILES, "m")
        assert out(repo, "rev-parse", "refs/less/p^{tree}") == git_tree

    def test_tree_entries_are_sorted_like_git_whatever_the_input_order(self, repo):
        names = ["zeta", "a.json", "a", "a-b", "Alpha", "mid"]
        files = {n: n.encode() for n in names}
        _, git_tree = make_shard_with_git(repo, "refs/less/g", files)
        found = Repo.discover(repo)
        found.write_shard("refs/less/p", dict(reversed(list(files.items()))), "m")
        assert out(repo, "rev-parse", "refs/less/p^{tree}") == git_tree

    def test_git_accepts_what_we_wrote(self, repo):
        found = Repo.discover(repo)
        commit = found.write_shard("refs/less/p", FILES, "pytest-less shard abc")
        assert out(repo, "rev-parse", "refs/less/p") == commit
        assert "pytest-less shard abc" in out(repo, "cat-file", "-p", commit)
        for name, data in FILES.items():
            assert git(repo, "cat-file", "blob", f"refs/less/p:{name}") == data
        git(repo, "fsck", "--no-dangling", "--strict")  # raises on corruption
        assert out(repo, "rev-list", "--parents", "-n1", commit).split() == [commit]  # root

    def test_survives_gc_because_the_ref_keeps_it_reachable(self, repo):
        found = Repo.discover(repo)
        found.write_shard("refs/less/p", FILES, "m")
        git(repo, "gc", "-q", "--prune=now")
        for name, data in FILES.items():
            assert git(repo, "cat-file", "blob", f"refs/less/p:{name}") == data

    def test_loose_objects_have_git_permissions_and_no_temp_leftovers(self, repo):
        found = Repo.discover(repo)
        found.write_shard("refs/less/p", FILES, "m")
        objects = repo / ".git" / "objects"
        assert list(objects.rglob("tmp_obj_*")) == []
        sha = found.blob_sha_in_ref("refs/less/p", "meta.json")
        assert (objects / sha[:2] / sha[2:]).stat().st_mode & 0o222 == 0  # read-only

    def test_reuse_carries_an_existing_blob_by_id(self, repo):
        found = Repo.discover(repo)
        found.write_shard("refs/less/p", FILES, "m")
        keep = found.blob_sha_in_ref("refs/less/p", "parse.json")
        found.write_shard(
            "refs/less/p",
            {"data.json": b"{}", "meta.json": b"{}"},
            "m2",
            reuse={"parse.json": keep},
        )
        assert git(repo, "cat-file", "blob", "refs/less/p:parse.json") == FILES["parse.json"]

    def test_overwriting_is_latest_wins(self, repo):
        found = Repo.discover(repo)
        first = found.write_shard("refs/less/p", {"meta.json": b"1"}, "m")
        assert out(repo, "rev-parse", "refs/less/p") == first
        second = found.write_shard("refs/less/p", {"meta.json": b"2"}, "m")
        assert out(repo, "rev-parse", "refs/less/p") == second
        assert out(repo, "cat-file", "blob", "refs/less/p:meta.json") == "2"

    def test_a_held_lock_is_reported(self, repo, monkeypatch):
        import pytest_less.gitobjects as mod

        monkeypatch.setattr(mod, "_LOCK_ATTEMPTS", 2)
        monkeypatch.setattr(mod, "_LOCK_WAIT", 0.0)
        found = Repo.discover(repo)
        ref_dir = repo / ".git" / "refs" / "less"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "p.lock").write_text("held")
        with pytest.raises(Unsupported):
            found.write_shard("refs/less/p", {"meta.json": b"{}"}, "m")

    def test_concurrent_writers_leave_a_valid_shard(self, repo):
        found = Repo.discover(repo)
        errors = []

        def writer(i):
            try:
                found.write_shard("refs/less/p", {"meta.json": f"{i}".encode()}, "m")
            except Unsupported as e:  # lock contention may legitimately give up
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert out(repo, "cat-file", "blob", "refs/less/p:meta.json") in {str(i) for i in range(8)}
        git(repo, "fsck", "--no-dangling", "--strict")


def not_modelled(monkeypatch):
    """Make every repository look like a layout the in-process layer does not
    model (reftable, SHA-256, ...): GitStore then uses the ``git`` command."""
    monkeypatch.setattr(Repo, "discover", classmethod(lambda cls, start: None))


@pytest.fixture(params=["in-process", "git"])
def store(request, repo, monkeypatch):
    """The same GitStore scenario with and without the in-process layer."""
    if request.param == "git":
        not_modelled(monkeypatch)
    s = GitStore(repo, "abc")
    assert (s._repo is None) == (request.param == "git")
    return s


class TestStoreBackendParity:
    """GitStore behaves identically whichever backend serves it."""

    def test_round_trip(self, store, repo):
        assert store.load_meta() is None and store.load_data() is None
        store.save({"version": 2, "x": [1]}, {"version": 1, "n": 1}, parse={"v": 1})
        assert store.load_data() == {"version": 2, "x": [1]}
        assert store.load_meta() == {"version": 1, "n": 1}
        assert store.load_parse() == {"v": 1}
        assert store.load() == ({"version": 2, "x": [1]}, {"version": 1, "n": 1})

    def test_parse_blob_is_carried_over_when_not_given(self, store):
        store.save({"version": 2}, {"version": 1}, parse={"v": 1, "files": {"a": {}}})
        store.save({"version": 2, "n": 2}, {"version": 1})
        assert store.load_parse() == {"v": 1, "files": {"a": {}}}
        assert store.load_data() == {"version": 2, "n": 2}

    def test_head_and_changed_since(self, store, repo):
        base = out(repo, "rev-parse", "HEAD")
        assert store.head() == base
        (repo / "new.txt").write_text("n")
        (repo / "f.txt").write_text("changed")
        assert store.changed_since(base) == {"new.txt", "f.txt"}
        assert store.changed_since("f" * 40) is None
        assert store.changed_since(None) is None

    def test_unborn_head(self, tmp_path, monkeypatch):
        path = tmp_path / "unborn"
        path.mkdir()
        git(path, "init", "-q")
        assert GitStore(path, "x").head() is None

    def test_delete(self, store, repo):
        store.save({"version": 2}, {"version": 1})
        assert store.delete()
        assert store.load_meta() is None
        assert not store.delete()


class TestMixedBackends:
    def test_shard_written_in_process_is_read_by_git_and_back(self, repo, monkeypatch):
        fast = GitStore(repo, "abc")
        fast.save({"version": 2, "w": "py"}, {"version": 1}, parse={"v": 1})
        with monkeypatch.context() as m:
            not_modelled(m)
            slow = GitStore(repo, "abc")
            assert slow.load_data() == {"version": 2, "w": "py"}
            slow.save({"version": 2, "w": "git"}, {"version": 1})
        assert GitStore(repo, "abc").load_data() == {"version": 2, "w": "git"}
        assert GitStore(repo, "abc").load_parse() == {"v": 1}  # carried over

    def test_packed_shard_is_still_read_via_git(self, repo):
        store = GitStore(repo, "abc")
        store.save({"version": 2, "k": 1}, {"version": 1, "m": 1}, parse={"v": 1})
        git(repo, "gc", "-q", "--prune=now")
        git(repo, "pack-refs", "--all", "--prune")
        fresh = GitStore(repo, "abc")  # objects packed, ref packed
        assert fresh.load_data() == {"version": 2, "k": 1}
        assert fresh.load_meta() == {"version": 1, "m": 1}
        assert fresh.load_parse() == {"v": 1}
        fresh.save({"version": 2, "k": 2}, {"version": 1})  # parse blob is packed
        assert fresh.load_parse() == {"v": 1}
        assert fresh.load_data() == {"version": 2, "k": 2}

    def test_failed_in_process_write_falls_back_to_git(self, repo, monkeypatch):
        store = GitStore(repo, "abc")

        def boom(*a, **k):
            raise Unsupported("simulated")

        monkeypatch.setattr(store._repo, "write_shard", boom)
        store.save({"version": 2, "via": "git"}, {"version": 1})
        assert store.load_data() == {"version": 2, "via": "git"}
