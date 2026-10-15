
from pytest_filemon.util import command_key, to_repo_rel


class TestToRepoRel:
    def test_absolute_inside(self, tmp_path):
        root = tmp_path / "repo"
        src = root / "src"
        src.mkdir(parents=True)
        f = src / "a.py"
        f.write_text("")
        assert to_repo_rel(str(f), root) == "src/a.py"
        # pathlib.Path inputs (e.g. item.path) must work too.
        assert to_repo_rel(f, root) == "src/a.py"

    def test_relative_to_cwd(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        (root / "src").mkdir(parents=True)
        (root / "src" / "a.py").write_text("")
        monkeypatch.chdir(root)
        assert to_repo_rel("src/a.py", root) == "src/a.py"

    def test_nonexistent_files_are_none(self, tmp_path):
        # Phantom relative co_filenames (e.g. a bare "bootstrap.py" from
        # generated code) must never become dependencies.
        root = tmp_path / "repo"
        root.mkdir()
        assert to_repo_rel("bootstrap.py", root) is None
        assert to_repo_rel(str(root / "nope.py"), root) is None

    def test_pure_function_no_stale_cache(self, tmp_path):
        # to_repo_rel must be pure (per-instance caching lives on the
        # monitor): a file created after a miss is resolved on the next
        # call, even in the same process.
        root = tmp_path / "repo"
        root.mkdir()
        f = root / "late.py"
        assert to_repo_rel(str(f), root) is None
        f.write_text("")
        assert to_repo_rel(str(f), root) == "late.py"

    def test_outside_repo_is_none(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        other = tmp_path / "other.py"
        other.write_text("")
        assert to_repo_rel(str(other), root) is None

    def test_venv_and_site_packages_excluded(self, tmp_path):
        root = tmp_path / "repo"
        other_root = tmp_path / "other"
        venv = root / ".venv" / "lib" / "python3.13" / "site-packages" / "plg"
        venv.mkdir(parents=True)
        excluded = venv / "x.py"
        excluded.write_text("")
        assert to_repo_rel(str(excluded), root) is None
        # also covers bare venv/ and tox layouts
        bare = other_root / "venv" / "y.py"
        bare.parent.mkdir(parents=True)
        bare.write_text("")
        assert to_repo_rel(str(bare), other_root) is None

    def test_synthetic_locations_are_none(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        assert to_repo_rel("<string>", root) is None
        assert to_repo_rel("<frozen importlib._bootstrap>", root) is None
        assert to_repo_rel("", root) is None
        assert to_repo_rel(None, root) is None

    def test_repo_root_itself_is_none(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        assert to_repo_rel(str(root), root) is None


class TestCommandKey:
    def test_order_and_duplicates_insensitive(self, tmp_path):
        root = tmp_path
        a = command_key(["-q", "tests", "-x", "tests"], root)
        b = command_key(["tests", "-x", "-q"], root)
        assert a == b

    def test_different_args_different_shard(self, tmp_path):
        root = tmp_path
        # selection/semantics-affecting args stay in the key
        assert command_key(["-q"], root) != command_key(["-q", "-m", "foo"], root)
        assert command_key(["tests"], root) != command_key(["tests/unit"], root)
        # warning filters can change outcomes mid-run -> kept
        assert command_key(["-W", "error", "tests"], root) != command_key(
            ["-W", "ignore", "tests"], root
        )
        # unknown args are kept (conservative: never merge what we don't know)
        assert command_key(["--some-plugin-flag", "tests"], root) != command_key(
            ["tests"], root
        )

    def test_presentation_flags_do_not_change_the_shard(self, tmp_path):
        # Verbosity/report/traceback formatting and early exit never change
        # which tests run -> one shard for all of these.
        base = command_key(["tests"], tmp_path)
        assert base == command_key(["-q", "tests"], tmp_path)
        assert base == command_key(["tests", "-v"], tmp_path)
        assert base == command_key(["--quiet", "--verbose", "tests"], tmp_path)
        assert base == command_key(["--tb=short", "-r", "a", "tests"], tmp_path)
        assert base == command_key(["--tb", "long", "--durations", "10", "tests"], tmp_path)
        assert base == command_key(["-x", "--maxfail", "3", "--color", "yes", "tests"], tmp_path)
        assert base == command_key(["-l", "--full-trace", "--no-header", "--no-summary", "tests"], tmp_path)

    def test_paths_normalized_to_repo_relative(self, tmp_path, monkeypatch):
        root = tmp_path
        (root / "tests").mkdir()
        (root / "unit").mkdir()
        monkeypatch.chdir(root)
        # Absolute and relative spellings of the same directory merge.
        a = command_key([str(root / "tests")], root)
        b = command_key(["tests"], root)
        assert a == b

    def test_plugin_and_volatile_flags_excluded(self, tmp_path):
        root = tmp_path
        base = ["tests", "-q"]
        assert command_key(["--filemon", *base, "--basetemp=/tmp/x1"], root) == command_key(
            ["--filemon-force", *base, "--basetemp", "/tmp/x2"], root
        )
        assert command_key(base, root) == command_key(base, root)

    def test_empty_args_stable(self, tmp_path):
        assert command_key([], tmp_path) == command_key([], tmp_path)
