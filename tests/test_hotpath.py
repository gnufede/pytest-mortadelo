"""The per-event hot path of the monitor, driven by hand (no tool registered).

``_on_py_start`` runs 2.4M times on a 2.5k-test suite.  It looks the code
object up by ``id()`` (hashing a code object walks its constants: 226 ns on
average, 2-5 us for the heavy functions that run in nearly every test, against
37 ns for the int), and adds to a *held reference* to the current bucket's set
instead of going through a method that allocated a throwaway ``set()`` per
event.  These tests pin the behavior that must not change while doing so.
"""

import gc
import types

import pytest

from pytest_less import monitoring
from pytest_less.monitoring import FileMonitor


def codes_of(source, path):
    top = compile(source, str(path), "exec")
    return {c.co_name: c for c in top.co_consts if isinstance(c, types.CodeType)}


@pytest.fixture
def mon(tmp_path):
    path = tmp_path / "m.py"
    source = "def f():\n    return 1\n\ndef g():\n    return 2\n"
    path.write_text(source)
    monitor = FileMonitor(tmp_path, data_files=False)
    monitor._active = True  # hand-driven: the callback only needs the flag
    fns = codes_of(source, path)
    outside = compile("pass", "/nowhere/else.py", "exec")
    return monitor, fns["f"], fns["g"], outside


def blocks(monitor, nodeid):
    return {b[1] for b in monitor.test_deps.get(nodeid, ())}


class TestAttribution:
    def test_a_block_is_attributed_to_every_test_that_runs_it(self, mon):
        monitor, f, g, _ = mon
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        monitor._on_py_start(f, 0)
        monitor.current_nodeid = "T2"  # the next test re-arms: f fires again
        monitor._on_py_start(f, 0)
        monitor._on_py_start(g, 0)
        assert blocks(monitor, "T1") == {"f"}
        assert blocks(monitor, "T2") == {"f", "g"}

    def test_a_block_that_fires_twice_in_one_test_is_recorded_once(self, mon):
        monitor, f, _, _ = mon
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        first = monitor._on_py_start(f, 0)
        again = monitor._on_py_start(f, 0)  # a foreign restart_events() re-armed it
        assert first == again == monitoring._DISABLE_ATTRIBUTED
        assert len(monitor.test_deps["T1"]) == 1

    def test_code_outside_the_repo_is_silenced_and_never_recorded(self, mon):
        monitor, _, _, outside = mon
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        assert monitor._on_py_start(outside, 0) == monitoring._DISABLE
        assert monitor._on_py_start(outside, 0) == monitoring._DISABLE
        assert "T1" not in monitor.test_deps

    def test_a_test_that_ran_no_repo_code_has_no_entry(self, mon):
        # (it must stay "new" so it is re-run, exactly as before the rewrite)
        monitor, f, _, outside = mon
        monitor.phase = "test"
        monitor.current_nodeid = "ran"
        monitor._on_py_start(f, 0)
        monitor.current_nodeid = "idle"
        monitor._on_py_start(outside, 0)
        assert set(monitor.test_deps) == {"ran"}


class TestBuckets:
    def test_import_collect_and_test_phases_go_to_their_own_bucket(self, mon):
        monitor, f, g, _ = mon
        monitor._on_py_start(f, 0)  # the initial phase is "import"
        monitor.phase = "collect"
        monitor._on_py_start(g, 0)
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        monitor._on_py_start(f, 0)
        assert {b[1] for b in monitor.import_deps} == {"f"}
        assert {b[1] for b in monitor.collect_deps} == {"g"}
        assert blocks(monitor, "T1") == {"f"}
        assert "T1" not in {b for b in monitor.import_deps}

    def test_the_current_test_is_visible_mid_run_and_keeps_filling(self, mon):
        monitor, f, g, _ = mon
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        monitor._on_py_start(f, 0)
        view = monitor.test_deps  # published while the test is still running
        assert {b[1] for b in view["T1"]} == {"f"}
        monitor._on_py_start(g, 0)  # later events land in the same entry
        assert blocks(monitor, "T1") == {"f", "g"}

    def test_done_phase_records_nothing(self, mon):
        monitor, f, g, _ = mon
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        monitor._on_py_start(f, 0)
        monitor.phase = "done"
        monitor._on_py_start(g, 0)
        assert blocks(monitor, "T1") == {"f"}
        assert not monitor.collect_deps and not {b[1] for b in monitor.import_deps}

    def test_a_spawn_stops_collection_but_keeps_what_came_before(self, mon):
        monitor, f, g, _ = mon
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        monitor._on_py_start(f, 0)
        monitor._note_spawn()
        monitor._on_py_start(g, 0)
        assert blocks(monitor, "T1") == {"f"}
        assert monitor.spawn_tests == {"T1"}
        monitor.current_nodeid = "T2"  # the next test collects normally
        monitor._on_py_start(g, 0)
        assert blocks(monitor, "T2") == {"g"}


class TestCodeCache:
    def test_keyed_by_id_and_the_cell_keeps_the_code_alive(self, mon):
        monitor, f, _, _ = mon
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        monitor._on_py_start(f, 0)
        cell = monitor._code_cache[id(f)]
        assert cell[2] is f  # the id can therefore never be reused

    def test_a_freed_code_object_cannot_alias_a_cached_one(self, tmp_path):
        monitor = FileMonitor(tmp_path, data_files=False)
        monitor._active = True
        monitor.phase = "test"
        monitor.current_nodeid = "T1"
        seen = set()
        for i in range(200):  # many short-lived objects: ids would be recycled
            code = compile(f"x = {i}", "/nowhere/x.py", "exec")
            monitor._on_py_start(code, 0)
            seen.add(id(code))
            del code
            gc.collect()
        assert len(seen) == 200  # every id distinct: the cache pinned each object
