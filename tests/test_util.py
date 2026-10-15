
from pytest_filemon.util import command_key, to_repo_rel


class TestToRepoRel:
    def test_absolute_inside(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        assert to_repo_rel(str(root / "src" / "a.py"), root) == "src/a.py"
        # pathlib.Path inputs (e.g. item.path) must work too.
        assert to_repo_rel(root / "src" / "a.py", root) == "src/a.py"

    def test_relative_to_cwd(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        (root / "src").mkdir(parents=True)
        monkeypatch.chdir(root)
        assert to_repo_rel("src/a.py", root) == "src/a.py"

    def test_outside_repo_is_none(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        other = tmp_path / "other.py"
        other.write_text("")
        assert to_repo_rel(str(other), root) is None

    def test_venv_and_site_packages_excluded(self, tmp_path):
        root = tmp_path / "repo"
        (root / ".venv" / "lib" / "python3.13" / "site-packages" / "plg").mkdir(
            parents=True
        )
        excluded = root / ".venv" / "lib" / "python3.13" / "site-packages" / "plg" / "x.py"
        excluded.write_text("")
        assert to_repo_rel(str(excluded), root) is None
        # also covers bare venv/ and tox layouts
        assert to_repo_rel(str(tmp_path / "other" / "site-packages" / "y.py"), tmp_path / "other") is None

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
        assert command_key(["-q"], root) != command_key(["-q", "-x"], root)
        assert command_key(["tests"], root) != command_key(["tests/unit"], root)

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
