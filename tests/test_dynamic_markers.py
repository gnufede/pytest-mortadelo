"""Dynamic-marker safety net: reflective files widen to file level (C.1).

The block model is causally closed for *executed* code -- PY_START records
every block a test runs, however it was dispatched -- but reflective edges
are the one place a recording can disagree with the next run: which target
``getattr(name)`` lands on depends on state (registries, order, data files),
and ``eval``/``exec`` produce code no block table ever saw.  pytest-tia's
dynscan marks the *marker* at record time; we scan the *changed file at
selection time* (no shard format change): a file that both contains a
dynamic construct *and* has a stale recorded block is widened to file level
-- every test that touched the file re-runs.  The staleness gate matters:
it is what keeps the widening from firing on cosmetic edits (a docstring
change under the (default) cosmetic filter produces no stale block, so Ley 2 never
punches through Ley 1), and it only opens when some test really did execute
the changed code -- the case where other, reflective tests may reach it
through edges the recording never saw.
"""

import ast

import pytest
from test_plugin import commit, git, run_less

from pytest_less.blocks import find_dynamic_markers, parse_blocks
from pytest_less.selection import Shard, select_tests


def markers_of(source: str) -> tuple:
    return find_dynamic_markers(ast.parse(source))


class TestMarkerDetection:
    def test_computed_getattr_is_flagged(self):
        assert markers_of(
            "def dispatch(name):\n    return getattr(registry, name)\n"
        ) == ("getattr@L2",)

    def test_literal_getattr_is_statically_resolvable(self):
        # Their dynscan ignores these too: a string constant names its
        # attribute, so no hidden edge can exist.
        assert markers_of("x = getattr(registry, 'name')\n") == ()

    def test_setattr_and_delattr_with_computed_names(self):
        src = (
            "def put(obj, name, value):\n"
            "    setattr(obj, name, value)\n"
            "    delattr(obj, name)\n"
        )
        assert markers_of(src) == ("delattr@L3", "setattr@L2")

    @pytest.mark.parametrize("call", ["eval(x)", "exec(x)", "__import__(x)"])
    def test_dynamic_builtins_are_flagged(self, call):
        name = call.split("(")[0]
        assert markers_of(f"def f(x):\n    return {call}\n") == (f"{name}@L2",)

    def test_import_module_with_computed_name(self):
        assert markers_of(
            "import importlib\n\ndef load(p):\n    return importlib.import_module(p)\n"
        ) == ("import_module@L4",)

    @pytest.mark.parametrize("name", ["__getattr__", "__getattribute__"])
    def test_dynamic_dunder_method_definitions(self, name):
        assert markers_of(f"class C:\n    def {name}(self, key):\n        ...\n") == (
            f"{name}@L2",
        )

    def test_plain_code_has_no_markers(self):
        fb = parse_blocks("def f(x):\n    return x + 1\n\nclass K:\n    pass\n")
        assert fb is not None
        assert fb.markers == ()


# Sources shared by the widening units: the reflective file and its exact
# twin without markers, so the only difference under test is the marker.
REFLECTIVE_OLD = (
    "class Registry:\n"
    "    value = 42\n"
    "\n"
    "def dispatch(name):\n"
    "    return getattr(Registry, name)\n"
    "\n"
    "def helper():\n"
    "    return 'old'\n"
)
REFLECTIVE_NEW = REFLECTIVE_OLD.replace("'old'", "'new'")
PLAIN_OLD = "def dispatch(name):\n    return name\n\ndef helper():\n    return 'old'\n"
PLAIN_NEW = PLAIN_OLD.replace("'old'", "'new'")

USE = "tests/test_use.py::test_use"
OFF = "tests/test_off.py::test_off"
HELPER = "tests/test_helper.py::test_helper"


def make_shard(tests, import_deps=None) -> Shard:
    return Shard(
        base_commit="ab" * 20,
        tests={nid: set(deps) for nid, deps in tests.items()},
        import_deps=set(import_deps or ()),
        failed=set(),
    )


class TestWidening:
    """Unit: marker file + stale block -> file-level fan-out.

    The staleness gate needs *someone* to have executed the changed block;
    here that is ``tests/test_helper.py``, which called ``helper()`` in the
    recorded run.  ``test_use`` only ever executed ``dispatch``.
    """

    def test_marker_file_stale_widens_to_file_level(self):
        old = parse_blocks(REFLECTIVE_OLD)
        new = parse_blocks(REFLECTIVE_NEW)
        assert new.markers  # sanity: the file really is reflective
        shard = make_shard(
            {
                USE: {("lib.py", "dispatch", old.blocks["dispatch"])},
                OFF: {("tests/test_off.py", "test_off", "h0")},
                HELPER: {("lib.py", "helper", old.blocks["helper"])},
            }
        )
        sel = select_tests([USE, OFF, HELPER], shard, {"lib.py"}, {"lib.py": new})
        # helper's executor via staleness, dispatch's executor via the
        # widening (its own deps are all fresh), the stranger stays out.
        assert set(sel.selected) == {USE, HELPER}
        assert sel.deselected == [OFF]
        assert "dynamic markers" in sel.reason

    def test_same_edit_in_a_plain_file_stays_block_precise(self):
        old = parse_blocks(PLAIN_OLD)
        new = parse_blocks(PLAIN_NEW)
        assert not new.markers
        shard = make_shard(
            {
                USE: {("lib.py", "dispatch", old.blocks["dispatch"])},
                OFF: {("tests/test_off.py", "test_off", "h0")},
                HELPER: {("lib.py", "helper", old.blocks["helper"])},
            }
        )
        sel = select_tests([USE, OFF, HELPER], shard, {"lib.py"}, {"lib.py": new})
        # without a marker only the stale dependency itself re-runs:
        # test_use touched the file but never executed helper.
        assert sel.selected == [HELPER]
        assert USE in sel.deselected

    def test_marker_file_in_the_import_bucket_widens_everyone(self):
        old = parse_blocks(REFLECTIVE_OLD)
        new = parse_blocks(REFLECTIVE_NEW)
        shard = make_shard(
            {
                USE: {("tests/test_use.py", "test_use", "h0")},
                OFF: {("tests/test_off.py", "test_off", "h0")},
                HELPER: {("lib.py", "helper", old.blocks["helper"])},
            },
            # everyone imported lib at startup: its skeleton is a shared
            # import-time dep -- still fresh after the helper body edit, so
            # this is the marker widening and not the import-stale path.
            import_deps=[("lib.py", "<module>", old.blocks["<module>"])],
        )
        sel = select_tests([USE, OFF, HELPER], shard, {"lib.py"}, {"lib.py": new})
        assert set(sel.selected) == {USE, OFF, HELPER}
        assert "dynamic markers" in sel.reason
        assert "import-time" not in sel.reason

    def test_fresh_marker_file_never_widens(self):
        fb = parse_blocks(REFLECTIVE_OLD)
        shard = make_shard(
            {
                USE: {("lib.py", "dispatch", fb.blocks["dispatch"])},
                OFF: {("tests/test_off.py", "test_off", "h0")},
                HELPER: {("lib.py", "helper", fb.blocks["helper"])},
            }
        )
        # changed (reformatted, say) but every recorded hash matches:
        # no stale block -> no widening -> nothing runs.
        sel = select_tests([USE, OFF, HELPER], shard, {"lib.py"}, {"lib.py": fb})
        assert sel.selected == []

    def test_changed_file_without_recorded_baseline_does_not_widen(self):
        # Nobody ever executed lib.py's helper (no recorded baseline at
        # all for it): with nothing stale, the net stays quiet -- the
        # causal-closure argument says unreachable code cannot become
        # reachable through an edit alone.
        fb = parse_blocks(REFLECTIVE_NEW)
        shard = make_shard(
            {
                USE: {("tests/test_use.py", "test_use", "h0")},
                OFF: {("tests/test_off.py", "test_off", "h0")},
                HELPER: {("tests/test_helper.py", "test_helper", "h0")},
            }
        )
        sel = select_tests([USE, OFF, HELPER], shard, {"lib.py"}, {"lib.py": fb})
        assert sel.selected == []


@pytest.fixture
def dynamic_project(pytester):
    """A repo with a reflective lib (computed getattr) and a plain module.

    Three tests: ``test_use`` executes ``lib.dispatch``, ``test_helper``
    executes ``lib.helper`` (the block the e2e edits), ``test_off`` only
    ever touches ``plain.py``.
    """
    tests = pytester.path / "tests"
    tests.mkdir()
    (tests / "conftest.py").touch()
    (tests / "plain.py").write_text(PLAIN_OLD)
    write_lib(pytester, "old")
    (tests / "test_use.py").write_text(
        "def test_use():\n"
        "    import lib\n"
        '    assert lib.dispatch("value") == 42\n'
    )
    (tests / "test_helper.py").write_text(
        "def test_helper():\n"
        "    import lib\n"
        "    assert isinstance(lib.helper(), str)\n"
    )
    (tests / "test_off.py").write_text(
        "def test_off():\n"
        "    import plain\n"
        '    assert plain.dispatch("x") == "x"\n'
    )
    (pytester.path / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n")
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "test@example.com")
    git(pytester.path, "config", "user.name", "Test")
    commit(pytester.path, "initial")
    return pytester


def write_lib(project, helper_value, docstring='"""Look up a registry entry."""'):
    (project.path / "tests" / "lib.py").write_text(
        "class Registry:\n"
        "    value = 42\n"
        "\n"
        "def dispatch(name):\n"
        f"    {docstring}\n"
        "    return getattr(Registry, name)\n"
        "\n"
        "def helper():\n"
        f"    return {helper_value!r}\n"
    )


class TestDynamicE2E:
    def test_helper_edit_in_marker_file_selects_the_executor(self, dynamic_project):
        run_less(dynamic_project, "-q").assert_outcomes(passed=3)
        # edit helper: executed only by test_helper; test_use touched the
        # reflective file but never ran helper -- it is dragged in by the
        # widening, test_off never touched lib.py at all.
        write_lib(dynamic_project, "new")
        result = run_less(dynamic_project, "-q")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])
        result.stdout.fnmatch_lines(["*dynamic markers*"])

    def test_same_edit_in_plain_file_selects_nothing(self, dynamic_project):
        run_less(dynamic_project, "-q").assert_outcomes(passed=3)
        (dynamic_project.path / "tests" / "plain.py").write_text(PLAIN_NEW)
        result = run_less(dynamic_project, "-q")
        # nobody executes plain.helper (no stale dep exists), so nothing
        # runs: no marker, no widening, and the recorded hashes are fresh.
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])

    @pytest.mark.usefixtures("ast_hashing")
    def test_docstring_edit_in_marker_file_stays_cosmetic(self, dynamic_project):
        # The interlock: with the (default) cosmetic filter a prose-only edit
        # produces no stale block, so the widening gate never opens (Ley 2
        # without weakening Ley 1).
        run_less(dynamic_project, "-q").assert_outcomes(passed=3)
        write_lib(dynamic_project, "old", docstring='"A much better description."')
        result = run_less(dynamic_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])

    def test_docstring_edit_without_cosmetic_selects_executor(self, dynamic_project):
        # With --less-no-cosmetic prose is behavior.  dispatch's hash
        # changed (the docstring) -> test_use via staleness, and the marker
        # widening pulls in test_helper's file-mate too.
        run_less(dynamic_project, "-q", "--less-no-cosmetic").assert_outcomes(passed=3)
        write_lib(dynamic_project, "old", docstring='"A much better description."')
        result = run_less(dynamic_project, "-q", "--less-no-cosmetic")
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])
        result.assert_outcomes(passed=2)
