"""Tests for the sys.monitoring-based file monitor.

The monitor registers process-global PY_START events, so each scenario runs in
a subprocess that reports its buckets back as JSON.
"""

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

DRIVER = """
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[4])       # pytest_filemon source tree
sys.path.insert(0, sys.argv[1])       # "repo" root, for module resolution

from pytest_filemon.monitoring import FileMonitor

root = Path(sys.argv[1])
phase = sys.argv[2]
snippet_path = Path(sys.argv[3])

monitor = FileMonitor(root)
assert monitor.activate(), "could not activate a monitoring tool id"
monitor.phase = phase
monitor.current_nodeid = "T"

# Execute like a real module: derive __package__ from the __init__.py chain so
# that relative imports inside the snippet work.
pkg_parts = []
d = snippet_path.parent
while (d / "__init__.py").exists():
    pkg_parts.insert(0, d.name)
    d = d.parent
package = ".".join(pkg_parts)
module_name = (package + "." if package else "") + snippet_path.stem
g = {"__name__": module_name, "__package__": package}
exec(compile(snippet_path.read_text(), str(snippet_path), "exec"), g)
monitor.phase = "done"
monitor.current_nodeid = None

print(json.dumps({
    "import": sorted(monitor.import_deps),
    "collect": sorted(monitor.collect_deps),
    "test": sorted(monitor.test_deps.get("T", [])),
}))
"""


def make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "__init__.py").touch()
    (root / "pkg" / "sub" / "__init__.py").touch()
    return root


def run_monitor(tmp_path, root, phase, snippet_rel, snippet_text):
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    snippet = root / snippet_rel
    snippet.parent.mkdir(parents=True, exist_ok=True)
    snippet.write_text(snippet_text)
    proc = subprocess.run(
        [sys.executable, str(driver), str(root), phase, str(snippet), str(SRC)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestBasicRecording:
    def test_own_file_recorded(self, tmp_path):
        root = make_repo(tmp_path)
        out = run_monitor(tmp_path, root, "test", "probe.py", "x = 1\n")
        assert out["test"] == ["probe.py"]

    def test_import_phase_buckets_elsewhere(self, tmp_path):
        root = make_repo(tmp_path)
        out = run_monitor(tmp_path, root, "import", "probe.py", "x = 1\n")
        assert out["import"] == ["probe.py"]
        assert out["test"] == []

    def test_collect_phase_buckets_elsewhere(self, tmp_path):
        root = make_repo(tmp_path)
        out = run_monitor(tmp_path, root, "collect", "probe.py", "x = 1\n")
        assert out["collect"] == ["probe.py"]
        assert out["test"] == []


class TestImports:
    def test_executed_import(self, tmp_path):
        root = make_repo(tmp_path)
        (root / "lib.py").write_text("def f():\n    return 1\n")
        out = run_monitor(tmp_path, root, "test", "probe.py", "import lib\nassert lib.f() == 1\n")
        assert "lib.py" in out["test"]

    def test_dynamic_import_never_executed_is_still_recorded(self, tmp_path):
        # The conservative guarantee: an import inside a code block that
        # *starts* is a dependency even if the import line never runs.
        # (An `if False:` body would be dead-code-eliminated by the compiler,
        # so the branch condition must be a runtime value.)
        root = make_repo(tmp_path)
        (root / "lib.py").write_text("def f():\n    return 1\n")
        snippet = (
            "def helper(flag):\n"
            "    if flag:\n"
            "        import lib\n"
            "    return 0\n"
            "helper(False)\n"
        )
        out = run_monitor(tmp_path, root, "test", "probe.py", snippet)
        assert "lib.py" in out["test"]

    def test_from_import_submodule(self, tmp_path):
        root = make_repo(tmp_path)
        (root / "pkg" / "mod.py").write_text("v = 1\n")
        out = run_monitor(tmp_path, root, "test", "probe.py", "from pkg import mod\nassert mod.v == 1\n")
        assert "pkg/__init__.py" in out["test"]
        assert "pkg/mod.py" in out["test"]

    def test_relative_imports(self, tmp_path):
        root = make_repo(tmp_path)
        (root / "pkg" / "sub" / "sib.py").write_text("v = 1\n")
        (root / "pkg" / "sub" / "thing.py").write_text("v = 2\n")
        (root / "pkg" / "up.py").write_text("v = 3\n")
        snippet = (
            "from . import sib\n"          # pkg.sub.sib
            "from ..sub import thing\n"    # pkg.sub.thing
            "from .. import up\n"          # pkg.up
            "assert sib.v + thing.v + up.v == 6\n"
        )
        out = run_monitor(tmp_path, root, "test", Path("pkg") / "sub" / "probe.py", snippet)
        assert "pkg/sub/probe.py" in out["test"]
        assert "pkg/sub/sib.py" in out["test"]
        assert "pkg/sub/thing.py" in out["test"]
        assert "pkg/up.py" in out["test"]

    def test_stdlib_and_outside_repo_are_dropped(self, tmp_path):
        root = make_repo(tmp_path)
        snippet = "import json\nfrom pathlib import Path\n"
        out = run_monitor(tmp_path, root, "test", "probe.py", snippet)
        assert out["test"] == ["probe.py"]

    def test_ancestor_packages_recorded(self, tmp_path):
        root = make_repo(tmp_path)
        (root / "pkg" / "sub" / "deep.py").write_text("v = 1\n")
        snippet = "import pkg.sub.deep\nassert pkg.sub.deep.v == 1\n"
        out = run_monitor(tmp_path, root, "test", "probe.py", snippet)
        assert {"pkg/__init__.py", "pkg/sub/__init__.py", "pkg/sub/deep.py"} <= set(out["test"])
