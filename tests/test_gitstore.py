import os
import subprocess

import pytest

from pytest_filemon.gitstore import GitStore


def git(path, *args, check=True, stdin=None):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=check,
        input=stdin,
    )


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    return path


def commit(path, message="commit"):
    path.joinpath("f.txt").write_text(message)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", message)
    return git(path, "rev-parse", "HEAD").stdout.strip()


class TestHeadAndCommits:
    def test_head_unborn(self, repo):
        assert GitStore(repo, "x").head() is None

    def test_head_after_commit(self, repo):
        sha = commit(repo)
        assert GitStore(repo, "x").head() == sha

    def test_commit_exists(self, repo):
        sha = commit(repo)
        store = GitStore(repo, "x")
        assert store.commit_exists(sha)
        assert not store.commit_exists("0" * 40)
        assert not store.commit_exists(None)


class TestChangedSince:
    def test_tracked_and_untracked_changes(self, repo):
        base = commit(repo)
        # Committed change after base.
        repo.joinpath("committed.txt").write_text("new\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "second")
        # Uncommitted change.
        repo.joinpath("f.txt").write_text("dirty\n")
        # Untracked file.
        repo.joinpath("untracked.txt").write_text("new\n")

        store = GitStore(repo, "x")
        assert store.changed_since(base) == {
            "committed.txt",
            "f.txt",
            "untracked.txt",
        }

    def test_no_changes(self, repo):
        base = commit(repo)
        assert GitStore(repo, "x").changed_since(base) == set()

    def test_missing_base(self, repo):
        commit(repo)
        store = GitStore(repo, "x")
        assert store.changed_since(None) is None
        assert store.changed_since("ab" * 20) is None


class TestSync:
    """Pull/push the shard via a git remote (onboarding without refspecs)."""

    def _repo_pair(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "t@t.t")
        git(repo, "config", "user.name", "t")
        bare = tmp_path / "remote.git"
        bare.mkdir()
        git(bare, "init", "--bare", "-q")
        git(repo, "remote", "add", "origin", str(bare))
        return repo, bare

    def _save(self, store, tests, created):
        store.save(
            {"version": 1, "tests": tests, "failed": []},
            {"version": 1, "shard": store.shard, "base_commit": None, "created": created},
        )

    def test_push_then_pull_states(self, tmp_path):
        repo, bare = self._repo_pair(tmp_path)
        store = GitStore(repo, "shard1")

        # Push: the remote gets exactly our shard ref.
        self._save(store, {"a": ["x"]}, "2024-01-01T00:00:00+00:00")
        assert "pushed to 'origin'" in store.sync_push("origin")
        assert git(bare, "for-each-ref", "--format=%(refname)", "refs/filemon/").stdout.split() == [
            "refs/filemon/shard1"
        ]

        # Fresh machine (no local ref): remote is adopted.
        git(repo, "update-ref", "-d", "refs/filemon/shard1")
        assert "pulled a newer shard" in store.sync_pull("origin")
        data, meta = store.load()
        assert data["tests"] == {"a": ["x"]}
        assert meta["created"] == "2024-01-01T00:00:00+00:00"

        # Local newer than remote: keep local.
        self._save(store, {"a": ["x", "y"]}, "2025-01-01T00:00:00+00:00")
        assert "not newer than local" in store.sync_pull("origin")
        data, _ = store.load()
        assert data["tests"]["a"] == ["x", "y"]

        # Local OLDER than remote: adopt the remote record.
        self._save(store, {"a": ["stale"]}, "2023-01-01T00:00:00+00:00")
        assert "pulled a newer shard" in store.sync_pull("origin")
        data, _ = store.load()
        assert data["tests"]["a"] == ["x"]

        # The scratch namespace used during pulls is always cleaned up.
        assert git(repo, "for-each-ref", "--format=%(refname)", "refs/filemon-remote/").stdout.strip() == ""

    def test_missing_shard_and_failures_are_non_fatal(self, tmp_path):
        repo, _bare = self._repo_pair(tmp_path)
        other = GitStore(repo, "noshard")
        assert "no shard" in other.sync_pull("origin")
        assert "nothing recorded locally" in other.sync_push("origin")
        # A bogus remote never raises.
        assert "no shard" in other.sync_pull("no-such-remote")
        self._save(other, {"a": ["x"]}, "2024-01-01T00:00:00+00:00")
        assert "push to 'no-such-remote' failed" in other.sync_push("no-such-remote")


class TestShardPersistence:
    def test_save_load_roundtrip(self, repo):
        commit(repo)
        store = GitStore(repo, "deadbeef")
        data = {"version": 1, "tests": {"tests/test_a.py::test_a": ["src/lib.py"]}, "failed": []}
        meta = {"version": 1, "shard": "deadbeef", "base_commit": store.head(), "created": "now"}
        store.save(data, meta)

        loaded_data, loaded_meta = store.load()
        assert loaded_data == data
        assert loaded_meta == meta

    def test_latest_wins(self, repo):
        commit(repo)
        store = GitStore(repo, "deadbeef")
        store.save({"version": 1, "tests": {}, "failed": []}, {"version": 1})
        store.save({"version": 1, "tests": {"a": ["b"]}, "failed": []}, {"version": 1})
        data, _ = store.load()
        assert data["tests"] == {"a": ["b"]}

    def test_load_missing_ref(self, repo):
        commit(repo)
        assert GitStore(repo, "nope").load() == (None, None)

    def test_corrupt_json(self, repo):
        commit(repo)
        store = GitStore(repo, "deadbeef")
        # Hand-build a shard whose data.json is not valid JSON.
        bad = git(repo, "hash-object", "-w", "--stdin", stdin="not json{{").stdout.strip()
        good = (
            git(repo, "hash-object", "-w", "--stdin", stdin='{"version": 1}')
            .stdout.strip()
        )
        tree = git(
            repo,
            "mktree",
            stdin=f"100644 blob {bad}\tdata.json\n100644 blob {good}\tmeta.json\n",
        ).stdout.strip()
        commit_sha = git(
            repo,
            "commit-tree",
            tree,
            "-m",
            "bad shard",
            stdin=None,
        ).stdout.strip()
        git(repo, "update-ref", store.ref, commit_sha)
        # data.json and meta.json are loadable independently: a corrupt data
        # blob degrades to "no old data" only when something actually runs,
        # while meta (the standalone first-level cache) still answers the
        # "did anything relevant change?" question.
        assert store.load() == (None, {"version": 1})

    def test_delete(self, repo):
        commit(repo)
        store = GitStore(repo, "deadbeef")
        store.save({"version": 1, "tests": {}, "failed": []}, {"version": 1})
        assert store.load() != (None, None)
        assert store.delete()
        assert store.load() == (None, None)
        assert not store.delete()

    def test_list_shards(self, repo):
        commit(repo)
        for shard in ("aa", "bb"):
            GitStore(repo, shard).save(
                {"version": 1, "tests": {}, "failed": []}, {"version": 1}
            )
        assert GitStore(repo, "aa").list_shards() == ["aa", "bb"]

    def test_ref_is_not_a_branch_or_head(self, repo):
        commit(repo)
        store = GitStore(repo, "deadbeef")
        store.save({"version": 1, "tests": {}, "failed": []}, {"version": 1})
        # HEAD unchanged, no new branch, working tree untouched.
        branches = git(repo, "for-each-ref", "--format=%(refname)").stdout.split()
        assert "refs/filemon/deadbeef" in branches
        assert "refs/heads/deadbeef" not in branches
        assert len([b for b in branches if b.startswith("refs/heads/")]) == 1

    def test_save_without_git_identity(self, repo, monkeypatch):
        # Hide any global/system git identity so commit-tree must fail and
        # the fallback identity kicks in.
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
        monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
        commit(repo)  # uses the local identity configured by the fixture
        git(repo, "config", "--unset", "user.email")
        git(repo, "config", "--unset", "user.name")
        store = GitStore(repo, "deadbeef")
        store.save({"version": 1, "tests": {}, "failed": []}, {"version": 1})
        assert store.load() != (None, None)
