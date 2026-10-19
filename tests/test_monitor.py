"""Tests for the sys.monitoring-based file monitor.

The monitor registers process-global PY_START events, so each scenario runs in
a subprocess that reports its buckets back as JSON.
"""

import json
import subprocess
import sys
from pathlib import Path


def dep_ids(deps):
    """[file:block, ...] for readable assertions over Block triples."""
    return sorted(f"{f}:{b}" for f, b, _ in deps)

SRC = Path(__file__).resolve().parents[1] / "src"

DRIVER = """
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[4])       # pytest_less source tree
sys.path.insert(0, sys.argv[1])       # "repo" root, for module resolution

from pytest_less.monitoring import FileMonitor

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
        assert dep_ids(out["test"]) == ["probe.py:<module>"]

    def test_import_phase_buckets_elsewhere(self, tmp_path):
        root = make_repo(tmp_path)
        out = run_monitor(tmp_path, root, "import", "probe.py", "x = 1\n")
        assert dep_ids(out["import"]) == ["probe.py:<module>"]
        assert out["test"] == []

    def test_collect_phase_buckets_elsewhere(self, tmp_path):
        root = make_repo(tmp_path)
        out = run_monitor(tmp_path, root, "collect", "probe.py", "x = 1\n")
        assert dep_ids(out["collect"]) == ["probe.py:<module>"]
        assert out["test"] == []


class TestImports:
    def test_executed_import(self, tmp_path):
        root = make_repo(tmp_path)
        (root / "lib.py").write_text("def f():\n    return 1\n")
        out = run_monitor(tmp_path, root, "test", "probe.py", "import lib\nassert lib.f() == 1\n")
        assert any(d[0] == "lib.py" for d in out["test"])

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
        assert any(d[0] == "lib.py" for d in out["test"])

    def test_from_import_submodule(self, tmp_path):
        root = make_repo(tmp_path)
        (root / "pkg" / "mod.py").write_text("v = 1\n")
        out = run_monitor(tmp_path, root, "test", "probe.py", "from pkg import mod\nassert mod.v == 1\n")
        assert any(d[0] == "pkg/__init__.py" for d in out["test"])
        assert any(d[0] == "pkg/mod.py" for d in out["test"])

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
        assert any(d[0] == "pkg/sub/probe.py" for d in out["test"])
        assert any(d[0] == "pkg/sub/sib.py" for d in out["test"])
        assert any(d[0] == "pkg/sub/thing.py" for d in out["test"])
        assert any(d[0] == "pkg/up.py" for d in out["test"])

    def test_stdlib_and_outside_repo_are_dropped(self, tmp_path):
        root = make_repo(tmp_path)
        snippet = "import json\nfrom pathlib import Path\n"
        out = run_monitor(tmp_path, root, "test", "probe.py", snippet)
        assert dep_ids(out["test"]) == ["probe.py:<module>"]

    def test_ancestor_packages_recorded(self, tmp_path):
        root = make_repo(tmp_path)
        (root / "pkg" / "sub" / "deep.py").write_text("v = 1\n")
        snippet = "import pkg.sub.deep\nassert pkg.sub.deep.v == 1\n"
        out = run_monitor(tmp_path, root, "test", "probe.py", snippet)
        assert {"pkg/__init__.py", "pkg/sub/__init__.py", "pkg/sub/deep.py"} <= {d[0] for d in out["test"]}


DISABLE_DRIVER = """
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[4])
sys.path.insert(0, sys.argv[1])   # repo root, for module resolution
from pytest_less.monitoring import FileMonitor

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


class TestModuleIndex:
    """The precomputed repo module index (replaces find_spec on hot paths)."""

    def _make_repo(self, tmp_path):
        root = tmp_path / "repo"
        pkg = root / "pkg" / "sub"
        pkg.mkdir(parents=True)
        (root / "pkg" / "__init__.py").touch()
        (pkg / "__init__.py").touch()
        (pkg / "mod.py").touch()
        (root / "plain.py").touch()
        (root / "tests" / "lib.py").parent.mkdir(parents=True)
        (root / "tests" / "lib.py").touch()
        (root / ".venv" / "lib" / "x.py").parent.mkdir(parents=True)
        (root / ".venv" / "lib" / "x.py").touch()
        (root / "secret" / "hide.py").parent.mkdir(parents=True)
        (root / "secret" / "hide.py").touch()
        return root

    def test_index_names_and_exclusions(self, tmp_path):
        from pytest_less.monitoring import _build_module_index

        root = self._make_repo(tmp_path)
        idx = _build_module_index(root, (root / "secret",))
        assert idx["pkg"] == "pkg/__init__.py"
        assert idx["pkg.sub"] == "pkg/sub/__init__.py"
        assert idx["pkg.sub.mod"] == "pkg/sub/mod.py"
        assert idx["plain"] == "plain.py"
        # Files outside any package are indexed by stem (their directory
        # might be on sys.path -- pytest does this for test dirs).
        assert idx["lib"] == "tests/lib.py"
        # venvs and excluded roots are pruned.
        assert not any(".venv" in p for p in idx.values())
        assert not any(p.startswith("secret/") for p in idx.values())

    def test_spec_origin_resolves_via_index_without_importlib(self, tmp_path, monkeypatch):
        import importlib.util

        from pytest_less.monitoring import FileMonitor

        root = self._make_repo(tmp_path)
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda *a: (_ for _ in ()).throw(AssertionError("index miss"))
        )
        mon = FileMonitor(root)
        assert mon._spec_origin("pkg.sub.mod") == "pkg/sub/mod.py"
        assert mon._spec_origin("pkg") == "pkg/__init__.py"
        # Unknown names fall back to importlib (which raises here -> None).
        assert mon._spec_origin("nope.nope") is None

    def test_extended_arg_import_still_resolved(self, tmp_path):
        # 300 module-level names push the import's name index past 255,
        # forcing an EXTENDED_ARG in front of IMPORT_NAME. The disassembly
        # path (and its operand resolution) must handle it.
        root = make_repo(tmp_path)
        (root / "lib.py").write_text("def f():\n    return 1\n")
        names = "\n".join(f"x{i} = {i}" for i in range(300))
        snippet = f"{names}\nimport lib\nassert lib.f() == 1\n"
        out = run_monitor(tmp_path, root, "test", "probe.py", snippet)
        assert any(d[0] == "lib.py" for d in out["test"])


COEXIST_DRIVER = """
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[4])
sys.path.insert(0, sys.argv[1])   # repo root, for module resolution
from sys import monitoring as sm
from pytest_less.monitoring import FileMonitor

root = Path(sys.argv[1])
snippet = Path(sys.argv[3])

# Occupy the preferred tool ids (4, 3) with dummy tools, as another
# monitor (e.g. an outer pytest session running this plugin in-process)
# would.
for tid in (4, 3):
    sm.use_tool_id(tid, "dummy-occupier")
    sm.set_events(tid, sm.events.PY_START)
    sm.register_callback(tid, sm.events.PY_START, lambda code, off: None)

mon = FileMonitor(root)
assert mon.activate(), "dynamic tool id allocation failed with 4/3 taken"
assert mon._tool_id not in (4, 3), mon._tool_id

mon2 = FileMonitor(root)
assert mon2.activate(), "second concurrent monitor failed to activate"
assert mon2._tool_id != mon._tool_id

# Both monitors record independently in the same process.
for m_ in (mon, mon2):
    m_.phase = "test"
    m_.current_nodeid = "T"
    exec(compile(snippet.read_text(), str(snippet), "exec"), {"__name__": "s"})
    m_.phase = "done"
    m_.current_nodeid = None

print(json.dumps({
    "tool_ids": [mon._tool_id, mon2._tool_id],
    "mon": sorted(mon.test_deps.get("T", [])),
    "mon2": sorted(mon2.test_deps.get("T", [])),
}))
"""


class TestToolIdAllocation:
    def test_dynamic_allocation_and_monitor_coexistence(self, tmp_path):
        # Preferred ids taken -> a different id is allocated; two monitors
        # coexist in one process and record independently (the nested-run
        # scenario of running our own plugin on our own tests).
        root = make_repo(tmp_path)
        (root / "lib.py").write_text("def f():\n    return 1\n")
        driver = tmp_path / "coexist_driver.py"
        driver.write_text(COEXIST_DRIVER)
        snippet = root / "probe.py"
        snippet.write_text("import lib\nassert lib.f() == 1\n")
        proc = subprocess.run(
            [sys.executable, str(driver), str(root), "test", str(snippet), str(SRC)],
            capture_output=True,
            text=True,
            check=True,
        )
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert len(set(out["tool_ids"])) == 2
        expected = ["lib.py:<module>", "lib.py:f", "probe.py:<module>"]
        assert dep_ids(out["mon"]) == expected
        assert dep_ids(out["mon2"]) == expected


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
        assert dep_ids(out["test"]) == [
            "lib.py:<module>", "lib.py:f", "probe.py:<module>",
        ]
        assert dep_ids(out["test2"]) == [
            "lib.py:<module>", "lib.py:f", "probe.py:<module>",
        ]
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
        assert not any(d[0] == "fake_plugin/hooks.py" for d in out["test"])
        assert not any(d[0] == "fake_plugin/hooks.py" for d in out["import"])
        assert dep_ids(out["test"]) == ["probe.py:<module>"]

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
