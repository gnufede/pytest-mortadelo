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
exclude = Path(sys.argv[5]) if len(sys.argv) > 5 else None

monitor = FileMonitor(
    root, exclude_roots=(exclude,) if exclude is not None else ()
)
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


def run_monitor(tmp_path, root, phase, snippet_rel, snippet_text, exclude=None):
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    snippet = root / snippet_rel
    snippet.parent.mkdir(parents=True, exist_ok=True)
    snippet.write_text(snippet_text)
    argv = [sys.executable, str(driver), str(root), phase, str(snippet), str(SRC)]
    if exclude is not None:
        argv.append(str(exclude))
    proc = subprocess.run(
        argv,
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


DISABLE_DRIVER = """
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[4])
sys.path.insert(0, sys.argv[1])   # repo root, for module resolution
from pytest_filemon.monitoring import FileMonitor

root = Path(sys.argv[1])
calls = {"n": 0}

m = FileMonitor(root)
orig_cb = m._on_py_start

# Wrap the callback BEFORE activate() so the registered callback counts
# every event dispatch.
def counting_on_py_start(code, offset):
    calls["n"] += 1
    return orig_cb(code, offset)

m._on_py_start = counting_on_py_start
assert m.activate()

# Dep-less hot loop: this driver lives outside the monitored root, so all
# its code objects resolve to no dependencies.
def hot(x):
    return x + 1

for _ in range(20000):
    hot(1)
hot_calls = calls["n"]

# In-repo code must keep firing (attribution per test).
calls["n"] = 0
m.phase = "test"
m.current_nodeid = "T1"
snippet = Path(sys.argv[3])
exec(compile(snippet.read_text(), str(snippet), "exec"), {"__name__": "snippet"})
m.current_nodeid = "T2"
exec(compile(snippet.read_text(), str(snippet), "exec"), {"__name__": "snippet"})
m.phase = "done"

print(json.dumps({
    "hot_loop_events": hot_calls,
    "snippet_events": calls["n"],
    "test": sorted(m.test_deps.get("T1", [])),
    "test2": sorted(m.test_deps.get("T2", [])),
}))
"""


class TestDisable:
    def test_depless_code_is_silenced_and_repo_code_keeps_firing(self, tmp_path):
        # sys.monitoring.DISABLE is honored for globally-set events: a
        # dep-less code object fires (almost) once, not once per start.
        root = make_repo(tmp_path)
        (root / "lib.py").write_text("def f():\n    return 1\n")
        driver = tmp_path / "disable_driver.py"
        driver.write_text(DISABLE_DRIVER)
        snippet = root / "probe.py"
        snippet.write_text("import lib\nassert lib.f() == 1\n")
        proc = subprocess.run(
            [sys.executable, str(driver), str(root), "test", str(snippet), str(SRC)],
            capture_output=True,
            text=True,
            check=True,
        )
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        # 20,000 starts of dep-less code produced only a handful of events.
        assert out["hot_loop_events"] < 10, out["hot_loop_events"]
        # Repo code still fires on every test: both buckets are attributed.
        assert out["test"] == ["lib.py", "probe.py"]
        assert out["test2"] == ["lib.py", "probe.py"]
        assert out["snippet_events"] > 0


class TestExclusions:
    def test_excluded_roots_are_never_recorded(self, tmp_path):
        # E.g. the plugin's own package: code that runs inside it (here, a
        # "plugin-ish" package living inside the repo) must not become a
        # dependency of anything.
        root = make_repo(tmp_path)
        plugin_dir = root / "fake_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "hooks.py").write_text("def hook():\n    return 1\n")
        snippet = (
            "import fake_plugin.hooks\n"
            "assert fake_plugin.hooks.hook() == 1\n"
        )
        out = run_monitor(
            tmp_path, root, "test", "probe.py", snippet, exclude=plugin_dir
        )
        assert "fake_plugin/hooks.py" not in out["test"]
        assert "fake_plugin/hooks.py" not in out["import"]
        assert out["test"] == ["probe.py"]

    def test_venv_inside_repo_is_not_recorded(self, tmp_path):
        # uv-style in-repo virtual environment: code executed from inside
        # .venv must not leak into dependencies even though the paths are
        # inside the repo root.
        root = make_repo(tmp_path)
        venv_rel = ".venv/lib/python3.13/site-packages/dep/mod.py"
        out = run_monitor(
            tmp_path, root, "test", venv_rel, "x = 1\n"
        )
        assert out["test"] == []
