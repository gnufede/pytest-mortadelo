"""Reflective lookups need no special rule.

A computed ``getattr(self, "handle_" + name)`` can only resolve to something in
a namespace that some code built.  For code in the repository that namespace is
a class body, a module body, or a registry filled in by executed functions --
and each of those is a block a test already depends on (class and module
skeletons run at import and are dependencies of every test; the rest are
ordinary executed blocks).  So the one way a lookup can start resolving to
something new, *adding* a definition, always changes a block the test depends
on, and the test is selected by the ordinary staleness rule.

These tests pin that, because the plugin used to carry a static "dynamic
marker" scan that widened any change in a file containing a computed
``getattr`` to every test that touched the file.  On sqlalchemy that turned
a one-function edit into a full run (``Query.filter``: 2,480 of 2,480 tests,
against 338 without the scan) while guarding nothing the rules below miss.
"""

import sys

import pytest
from test_plugin import commit, git, run_less

from pytest_less.blocks import WILDCARD


def make_project(pytester, files: dict[str, str]):
    for rel, text in {"tests/conftest.py": "", ".gitignore": ".pytest_cache/\n__pycache__/\n", **files}.items():
        path = pytester.path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(pytester.path, "init", "-q")
    git(pytester.path, "config", "user.email", "t@example.com")
    git(pytester.path, "config", "user.name", "T")
    commit(pytester.path, "initial")
    return pytester


def selected(result) -> str:
    for line in result.stdout.lines:
        if "selected" in line and "tests" in line:
            return line
    raise AssertionError(result.stdout.str())


class TestAnAddedDefinitionIsSeenByTheOrdinaryRules:
    def test_a_method_added_that_a_computed_getattr_now_finds(self, pytester):
        make_project(
            pytester,
            {
                "tests/handler.py": (
                    "class Handler:\n"
                    "    def default(self):\n"
                    "        return 'default'\n"
                    "    def run(self, name):\n"
                    "        return getattr(self, 'handle_' + name, self.default)()\n"
                ),
                "tests/test_a.py": (
                    "def test_a():\n"
                    "    from handler import Handler\n"
                    "    assert Handler().run('a') in ('default', 'A')\n"
                ),
                "tests/test_b.py": "def test_b():\n    assert True\n",
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=2)
        # test_a looked up handle_a, found nothing, took the default.  Now it exists.
        (pytester.path / "tests/handler.py").write_text(
            "class Handler:\n"
            "    def default(self):\n"
            "        return 'default'\n"
            "    def handle_a(self):\n"
            "        return 'A'\n"
            "    def run(self, name):\n"
            "        return getattr(self, 'handle_' + name, self.default)()\n"
        )
        result = run_less(pytester, "-v")
        assert "test_a.py::test_a" in result.stdout.str()  # selected (the class changed)

    def test_a_module_level_function_added_that_getattr_on_the_module_now_finds(
        self, pytester
    ):
        make_project(
            pytester,
            {
                "tests/ops.py": (
                    "import sys\n\n"
                    "def op_add(a, b):\n    return a + b\n\n"
                    "def call(name, *args):\n"
                    "    fn = getattr(sys.modules[__name__], 'op_' + name, None)\n"
                    "    return fn(*args) if fn else None\n"
                ),
                "tests/test_ops.py": (
                    "def test_ops():\n"
                    "    import ops\n"
                    "    assert ops.call('mul', 2, 3) in (None, 6)\n"
                ),
                "tests/test_other.py": "def test_other():\n    assert True\n",
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=2)
        path = pytester.path / "tests/ops.py"
        path.write_text(path.read_text() + "\ndef op_mul(a, b):\n    return a * b\n")
        result = run_less(pytester, "-v")
        assert "test_ops.py::test_ops" in result.stdout.str()

    def test_an_override_added_in_a_subclass_file(self, pytester):
        make_project(
            pytester,
            {
                "tests/base.py": (
                    "class Base:\n"
                    "    def hook_x(self):\n        return 'base'\n"
                    "    def go(self, name):\n"
                    "        return getattr(self, 'hook_' + name)()\n"
                ),
                "tests/sub.py": "from base import Base\n\nclass Sub(Base):\n    pass\n",
                "tests/test_sub.py": (
                    "def test_sub():\n"
                    "    from sub import Sub\n"
                    "    assert Sub().go('x') in ('base', 'sub')\n"
                ),
                "tests/test_other.py": "def test_other():\n    assert True\n",
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=2)
        (pytester.path / "tests/sub.py").write_text(
            "from base import Base\n\nclass Sub(Base):\n    def hook_x(self):\n        return 'sub'\n"
        )
        result = run_less(pytester, "-v")
        assert "test_sub.py::test_sub" in result.stdout.str()

    def test_a_registry_filled_by_executed_code(self, pytester):
        make_project(
            pytester,
            {
                "tests/reg.py": (
                    "REGISTRY = {}\n\n"
                    "def register(name):\n"
                    "    def deco(fn):\n        REGISTRY[name] = fn\n        return fn\n"
                    "    return deco\n\n"
                    "@register('a')\n"
                    "def handle_a():\n    return 'a'\n\n"
                    "def lookup(name):\n    return REGISTRY.get(name, lambda: 'none')()\n"
                ),
                "tests/test_reg.py": (
                    "def test_reg():\n"
                    "    import reg\n"
                    "    assert reg.lookup('b') in ('none', 'b')\n"
                ),
                "tests/test_other.py": "def test_other():\n    assert True\n",
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=2)
        path = pytester.path / "tests/reg.py"
        path.write_text(path.read_text() + "\n@register('b')\ndef handle_b():\n    return 'b'\n")
        result = run_less(pytester, "-v")
        assert "test_reg.py::test_reg" in result.stdout.str()


class TestPrecision:
    @pytest.fixture
    def dispatch_project(self, pytester):
        return make_project(
            pytester,
            {
                "tests/handler.py": (
                    "class Handler:\n"
                    "    def handle_a(self):\n        return 'a'\n"
                    "    def handle_b(self):\n        return 'b'\n"
                    "    def run(self, name):\n"
                    "        return getattr(self, 'handle_' + name)()\n"
                ),
                "tests/test_a.py": "def test_a():\n    from handler import Handler\n    assert Handler().run('a') == 'a'\n",
                "tests/test_b.py": "def test_b():\n    from handler import Handler\n    assert Handler().run('b') in ('b', 'bb')\n",
            },
        )

    def test_editing_a_handler_a_test_did_not_run_does_not_select_that_test(
        self, dispatch_project
    ):
        run_less(dispatch_project, "-q").assert_outcomes(passed=2)
        path = dispatch_project.path / "tests/handler.py"
        path.write_text(path.read_text().replace("return 'b'", "return 'bb'"))
        result = run_less(dispatch_project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
        assert "test_b.py::test_b PASSED" in result.stdout.str()
        assert "test_a.py" not in result.stdout.str().split("selected")[-1]


class TestAResumedGeneratorIsADependency:
    """PY_RESUME: a generator primed by one test runs its code in the test
    that *resumes* it, so that test depends on the generator's block too.

    Before PY_RESUME was tracked, the resuming test held no dependency: a
    shared generator edited between runs silently skipped the test (the
    synthetic case in docs/performance-notes.md; on sqlalchemy, 186 tests
    consume class-scoped yield fixtures this way)."""

    @pytest.fixture
    def shared_project(self, pytester):
        make_project(
            pytester,
            {
                "helper.py": (
                    "def numbers():\n    n = 0\n    while True:\n        n += 1\n        yield n\n\n"
                    "G = numbers()\n"
                ),
                "tests/conftest.py": "",
                # primes the generator (its PY_START); records the block
                "tests/test_prime.py": "def test_prime():\n    import helper\n    assert next(helper.G) == 1\n",
                # only ever resumes it: previously held no 'numbers' dependency
                "tests/test_resume.py": "def test_resume():\n    import helper\n    assert next(helper.G) == 2\n",
            },
        )
        return pytester

    def test_the_resuming_test_reruns_when_the_generator_changes(self, shared_project):
        run_less(shared_project, "-q").assert_outcomes(passed=2)
        (shared_project.path / "helper.py").write_text(
            "def numbers():\n    n = 0\n    while True:\n        n += 30\n        yield n\n\n"
            "G = numbers()\n"
        )
        result = run_less(shared_project, "-v")
        # both tests run: the priming one AND the resuming one
        result.stdout.fnmatch_lines(["*selected 2/2 tests*"])

    def test_the_recorded_shard_names_the_generator_block_for_both(self, shared_project):
        from test_plugin import shard_data

        run_less(shared_project, "-q").assert_outcomes(passed=2)
        data = shard_data(shared_project.path)
        table = [tuple(b) for b in data["blocks"]]
        for nid, idx in data["tests"].items():
            names = {table[i][1] for i in idx}
            assert "numbers" in names, (nid, names)


class TestModuleLevelCodeIsTracked:
    """PY_START (+ PY_RESUME) sees code outside functions too.

    A module body, a class body and the functions *called by* module-level
    code are all code objects that start (or resume) in some window, so the
    test that triggers them holds the dependency:

    * a lazy ``import lib`` inside a test runs ``lib``'s ``<module>`` there;
    * the import-time ``X = compute(10)`` attributes ``compute`` to the
      importing test, and ``class C:`` attributes the class-body block;
    * a class body defined *inside* a test is a block of its own
      (``test_x.<locals>.D``), so editing it selects that test alone;
    * editing a module-level or class-level statement stales the module
      skeleton, which every importer holds.
    """

    @pytest.fixture
    def lib_project(self, pytester):
        make_project(
            pytester,
            {
                "lib.py": (
                    "def compute(n):\n    return n * 2\n\n"
                    "X = compute(10)\n\n"
                    "class C:\n    Y = compute(5)\n\n    def m(self):\n        return self.Y\n"
                ),
                "tests/conftest.py": "",
                "tests/test_lazy.py": "def test_lazy():\n    import lib\n    assert lib.X == 20\n",
                "tests/test_method.py": "def test_method():\n    import lib\n    assert lib.C().m() == 10\n",
                "tests/test_dyn.py": (
                    "def test_dyn():\n    class D:\n        Z = 7\n    assert D.Z == 7\n"
                ),
            },
        )
        return pytester

    def test_the_importing_test_holds_the_module_compute_and_class_blocks(
        self, lib_project
    ):
        from test_plugin import shard_data

        run_less(lib_project, "-q").assert_outcomes(passed=3)
        data = shard_data(lib_project.path)
        table = [tuple(b) for b in data["blocks"]]
        deps = {table[i][1] for i in data["tests"]["tests/test_lazy.py::test_lazy"]}
        assert {"<module>", "compute", "C"} <= deps, deps

    def test_a_module_level_statement_edit_selects_the_importers(self, lib_project):
        run_less(lib_project, "-q").assert_outcomes(passed=3)
        (lib_project.path / "lib.py").write_text(
            (lib_project.path / "lib.py").read_text().replace("X = compute(10)", "X = compute(1000)")
        )
        result = run_less(lib_project, "-v")
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])  # both lib importers

    def test_a_class_level_statement_edit_selects_the_importers(self, lib_project):
        run_less(lib_project, "-q").assert_outcomes(passed=3)
        (lib_project.path / "lib.py").write_text(
            (lib_project.path / "lib.py").read_text().replace("Y = compute(5)", "Y = compute(500)")
        )
        result = run_less(lib_project, "-v")
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])

    def test_a_method_body_edit_selects_only_the_caller(self, lib_project):
        run_less(lib_project, "-q").assert_outcomes(passed=3)
        (lib_project.path / "lib.py").write_text(
            (lib_project.path / "lib.py").read_text().replace("return self.Y", "return self.Y * 1")
        )
        result = run_less(lib_project, "-v")
        result.stdout.fnmatch_lines(["*selected 1/3 tests*"])

    def test_a_class_body_defined_inside_a_test_is_its_own_block(self, lib_project):
        run_less(lib_project, "-q").assert_outcomes(passed=3)
        path = lib_project.path / "tests" / "test_dyn.py"
        path.write_text(path.read_text().replace("Z = 7", 'Z = int("7")'))
        result = run_less(lib_project, "-v")
        result.stdout.fnmatch_lines(["*selected 1/3 tests*"])  # test_dyn alone


class TestPluginsLoadedBeforeArming:
    """pytest imports its plugins -- entry points, ``-p``, ``PYTEST_PLUGINS``
    -- before the earliest hook a plugin can use, so the module-level code of
    a repo-resident plugin has already run when the monitor arms: no event
    can see the past.  At arming, already-imported repo modules get their
    skeletons added as shared import-time dependencies (the same treatment
    conftests imported before a *late* activation get); functions called in
    a window still fire PY_START and stay block-precise."""

    @pytest.fixture
    def plugin_project(self, pytester):
        make_project(
            pytester,
            {
                "pytest.ini": "[pytest]\naddopts = -p plug_gap\n",
                "plug_gap.py": (
                    "CONST = 1\n\nimport pytest as _pytest\n\n"
                    "@_pytest.fixture\ndef val():\n    return CONST\n"
                ),
                "tests/conftest.py": "",
                # uses the fixture; never imports plug_gap by name
                "tests/test_use.py": "def test_use(val):\n    assert val == 1\n",
                # imports it by name (the bytecode import scan also covers this shape)
                "tests/test_imports.py": "def test_imports():\n    import plug_gap\n    assert plug_gap.CONST == 1\n",
                "tests/test_other.py": "def test_other():\n    assert 1 + 1 == 2\n",
            },
        )
        return pytester

    def test_a_module_level_edit_selects_the_fixture_user_too(self, plugin_project):
        run_less(plugin_project, "-q").assert_outcomes(passed=3)
        path = plugin_project.path / "plug_gap.py"
        path.write_text(path.read_text().replace("CONST = 1", "CONST = 30"))
        result = run_less(plugin_project, "-v")
        # the import scan already covered test_imports; without the sweep
        # test_use (which depends on CONST only through the fixture) was
        # silently deselected while its outcome changed
        result.stdout.fnmatch_lines(["*selected 3/3 tests*"])

    def test_a_function_body_edit_stays_block_precise(self, plugin_project):
        run_less(plugin_project, "-q").assert_outcomes(passed=3)
        path = plugin_project.path / "plug_gap.py"
        path.write_text(path.read_text().replace("return CONST", "return CONST + 0"))
        result = run_less(plugin_project, "-v")
        result.stdout.fnmatch_lines(["*selected 1/3 tests*"])

    def test_the_skeleton_is_a_shared_import_dep(self, plugin_project):
        from test_plugin import shard_data

        run_less(plugin_project, "-q").assert_outcomes(passed=3)
        data = shard_data(plugin_project.path)
        table = [tuple(b) for b in data["blocks"]]
        deps = {(table[i][0], table[i][1]) for i in data["import_deps"]}
        assert ("plug_gap.py", "<module>") in deps


class TestFixtureClosureConsumers:
    """A >function-scoped fixture's setup runs once, in the FIRST requesting
    test's window; middle consumers receive the cached value without
    executing it.  Runtime events cannot see their dependency, so the
    closure pytest knows statically (``item.fixturenames``) is recorded for
    every requesting test -- the same over-approximation the import scan
    makes for ``import`` bytecode."""

    @pytest.fixture
    def fixture_project(self, pytester):
        make_project(
            pytester,
            {
                "conftest.py": (
                    "import pytest\n\n"
                    '@pytest.fixture(scope="class")\n'
                    "def config():\n"
                    '    return {"v": 1}\n\n\n'
                    "def unused_helper():\n    return 2\n"
                ),
                "tests/conftest.py": "",
                "tests/test_suite.py": (
                    "class TestA:\n"
                    '    def test_first(self, config):\n        assert config["v"] == 1\n\n'
                    '    def test_middle(self, config):\n        assert config["v"] == 1\n\n'
                    '    def test_last(self, config):\n        assert config["v"] == 1\n'
                ),
            },
        )
        return pytester

    def test_a_fixture_body_edit_selects_every_consumer(self, fixture_project):
        run_less(fixture_project, "-q").assert_outcomes(passed=3)
        path = fixture_project.path / "conftest.py"
        path.write_text(path.read_text().replace('return {"v": 1}', 'return {"v": 30}'))
        result = run_less(fixture_project, "-v")
        # before the closure rule this selected only test_first (1/3) while
        # all three tests failed
        result.stdout.fnmatch_lines(["*selected 3/3 tests*"])
        result.assert_outcomes(failed=3)

    def test_every_consumer_holds_the_fixture_block(self, fixture_project):
        from test_plugin import shard_data

        run_less(fixture_project, "-q").assert_outcomes(passed=3)
        data = shard_data(fixture_project.path)
        table = [tuple(b) for b in data["blocks"]]
        for nid, idx in data["tests"].items():
            names = {table[i][1] for i in idx}
            assert "config" in names, (nid, names)

    def test_a_function_body_edit_of_an_uncalled_helper_selects_nothing(
        self, fixture_project
    ):
        # precision control: the closure adds the *consumed* fixture block,
        # not the whole file
        run_less(fixture_project, "-q").assert_outcomes(passed=3)
        path = fixture_project.path / "conftest.py"
        path.write_text(path.read_text().replace("return 2", "return 3"))
        result = run_less(fixture_project, "-q")
        assert result.ret == 5  # nothing selected: no test depends on it
        result.stdout.fnmatch_lines(["*selected 0/3 tests*"])

    def test_a_function_scoped_fixture_needs_no_closure_entry(self, pytester):
        # its code runs in every consumer's own window, where PY_START
        # already records it -- the closure only widens for scope != function
        make_project(
            pytester,
            {
                "conftest.py": (
                    "import pytest\n\n"
                    "@pytest.fixture\n"
                    "def value():\n    return 1\n"
                ),
                "tests/conftest.py": "",
                "tests/test_use.py": "def test_use(value):\n    assert value == 1\n",
            },
        )
        from test_plugin import shard_data

        run_less(pytester, "-q").assert_outcomes(passed=1)
        data = shard_data(pytester.path)
        table = [tuple(b) for b in data["blocks"]]
        idx = data["tests"]["tests/test_use.py::test_use"]
        pairs = {(table[i][0], table[i][1]) for i in idx}
        assert ("conftest.py", "value") in pairs  # runtime already recorded it
        assert ("conftest.py", WILDCARD) not in pairs  # and the closure added nothing wider
        assert all(f != "conftest.py" or q == "value" for f, q in pairs)


class TestActiveConfigFile:
    """pytest parses pytest.ini / pyproject.toml / tox.ini / setup.cfg before
    the earliest hook a plugin can use, so no event ever saw the read: the
    file never became a dependency, and a config change selected nothing
    (observed: it even went through the no-change fast path).  The active
    ini is now claimed as a shared file-level dependency -- pytest itself
    says which file it parsed (``config.inifile``), so there is no list to
    maintain and no glob to guess."""

    @pytest.fixture
    def ini_project(self, pytester):
        make_project(
            pytester,
            {
                "pytest.ini": "[pytest]\n",
                "tests/conftest.py": "",
                "tests/test_warn.py": (
                    "def test_warn():\n"
                    "    import warnings\n"
                    '    warnings.warn("boom")\n'
                ),
            },
        )
        return pytester

    def test_a_config_edit_selects_everything(self, ini_project):
        run_less(ini_project, "-q").assert_outcomes(passed=1)
        (ini_project.path / "pytest.ini").write_text(
            "[pytest]\nfilterwarnings = error\n"
        )
        result = run_less(ini_project, "-v")
        result.stdout.fnmatch_lines(["*selected 1/1 tests*"])
        result.assert_outcomes(failed=1)  # the warning is now an error

    def test_the_ini_is_a_shared_import_dependency(self, ini_project):
        from test_plugin import shard_data

        run_less(ini_project, "-q").assert_outcomes(passed=1)
        data = shard_data(ini_project.path)
        table = [tuple(b) for b in data["blocks"]]
        deps = {table[i] for i in data["import_deps"]}
        assert ("pytest.ini", WILDCARD, WILDCARD) in deps


class TestWatchedFiles:
    """``less_watch``: the refined tracked-files escape hatch.

    Matched files become file-level dependencies of every test -- a
    dependency, not a cache invalidation: fingerprints watch them,
    staleness reports them, and the no-change fast path stays honest while
    they are untouched.  Patterns with a ``/`` anchor at the repository
    root; without one they match the file name at any depth."""

    @pytest.fixture
    def watch_project(self, pytester):
        make_project(
            pytester,
            {
                "pytest.ini": "[pytest]\nless_watch =\n    *.env\n    **/config/*.yml\n",
                ".env": "KEY=1\n",
                "config/app.yml": "v: 1\n",
                "lib.py": "def f():\n    return 1\n\ndef g():\n    return 2\n",
                "tests/conftest.py": "",
                "tests/test_a.py": "def test_a():\n    import lib\n    assert lib.f() == 1\n",
                "tests/test_b.py": "def test_b():\n    import lib\n    assert lib.g() == 2\n",
            },
        )
        return pytester

    def test_the_matched_files_are_shared_import_deps(self, watch_project):
        from test_plugin import shard_data

        run_less(watch_project, "-q").assert_outcomes(passed=2)
        data = shard_data(watch_project.path)
        table = [tuple(b) for b in data["blocks"]]
        deps = {table[i] for i in data["import_deps"]}
        assert (".env", WILDCARD, WILDCARD) in deps
        assert ("config/app.yml", WILDCARD, WILDCARD) in deps

    def test_an_edit_to_a_watched_file_re_runs_everything(self, watch_project):
        run_less(watch_project, "-q").assert_outcomes(passed=2)
        (watch_project.path / ".env").write_text("KEY=2\n")
        result = run_less(watch_project, "-v")
        result.stdout.fnmatch_lines(["*selected 2/2 tests*"])

    def test_an_anchored_pattern_watches_nested_directories(self, watch_project):
        run_less(watch_project, "-q").assert_outcomes(passed=2)
        (watch_project.path / "config" / "app.yml").write_text("v: 2\n")
        result = run_less(watch_project, "-v")
        result.stdout.fnmatch_lines(["*selected 2/2 tests*"])

    def test_watch_does_not_widen_ordinary_edits(self, watch_project):
        run_less(watch_project, "-q").assert_outcomes(passed=2)
        path = watch_project.path / "lib.py"
        path.write_text(path.read_text().replace("return 2", "return 2 + 1"))
        result = run_less(watch_project, "-v")
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])

    def test_an_unmatched_pattern_is_harmless(self, pytester):
        make_project(
            pytester,
            {
                "pytest.ini": "[pytest]\nless_watch = *.nomatch\n",
                "lib.py": "def f():\n    return 1\n",
                "tests/conftest.py": "",
                "tests/test_a.py": "def test_a():\n    import lib\n    assert lib.f() == 1\n",
            },
        )
        result = run_less(pytester, "-q")
        result.assert_outcomes(passed=1)

    def test_adopting_a_pattern_after_a_record_re_records_everything(self, pytester):
        # a shard recorded BEFORE the pattern: the watched file is nobody's
        # dependency in it, and the no-change fast path would deselect
        # everything forever, so the claim could never take effect.  The
        # run after adopting must record everything once.
        make_project(
            pytester,
            {
                "artifact.bin": "v1\n",
                "lib.py": "def f():\n    return 1\n",
                "tests/conftest.py": "",
                "tests/test_a.py": "def test_a():\n    import lib\n    assert lib.f() == 1\n",
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=1)
        (pytester.path / "pytest.ini").write_text("[pytest]\nless_watch = artifact.bin\n")
        result = run_less(pytester, "-v")
        result.stdout.fnmatch_lines(
            ["*selected 1/1 tests: new shared dependencies*"]
        )
        result.assert_outcomes(passed=1)
        # the claim is now an ordinary dependency of every test
        (pytester.path / "artifact.bin").write_text("v2\n")
        result = run_less(pytester, "-v")
        result.stdout.fnmatch_lines(["*selected 1/1 tests*"])


class TestLessAlwaysMarker:
    """``@pytest.mark.less_always``: the targeted escape hatch.

    The test says its outcome depends on things no rule can see (the
    environment, time, a memoized value) -- so it is selected whenever
    anything changed, exactly like a spawner, but unlike the spawner rule
    it still records dependencies (the graph keeps learning), it is
    skipped when provably nothing changed (the pristine rule), and it is
    counted in the run summary.  Removing the marker takes effect on the
    next run of the test."""
    # ^ docstring ends here; the tests follow

    @pytest.fixture
    def marked_project(self, pytester):
        make_project(
            pytester,
            {
                "lib.py": "def f():\n    return 1\n",
                "tests/conftest.py": "",
                "tests/test_plain.py": "def test_plain():\n    import lib\n    assert lib.f() == 1\n",
                "tests/test_marked.py": (
                    "import pytest\n\n"
                    "@pytest.mark.less_always\n"
                    "def test_marked():\n"
                    "    import lib\n"
                    "    assert lib.f() == 1\n"
                ),
            },
        )
        return pytester

    def test_any_edit_selects_the_marked_test(self, marked_project):
        run_less(marked_project, "-q").assert_outcomes(passed=2)
        # a comment appended to lib.py: no block is stale, nothing for
        # test_plain -- but the marked test re-runs
        path = marked_project.path / "lib.py"
        path.write_text(path.read_text() + "\n# a comment\n")
        result = run_less(marked_project, "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*selected 1/2 tests*"])
        result.stdout.fnmatch_lines(["*1 always re-run*"])

    def test_nothing_changed_skips_even_the_marked_test(self, marked_project):
        run_less(marked_project, "-q").assert_outcomes(passed=2)
        result = run_less(marked_project, "-q")
        assert result.ret == 5
        result.stdout.fnmatch_lines(["*selected 0/2 tests*"])

    def test_the_marked_test_still_records_dependencies(self, marked_project):
        from test_plugin import shard_data

        run_less(marked_project, "-q").assert_outcomes(passed=2)
        data = shard_data(marked_project.path)
        table = [tuple(b) for b in data["blocks"]]
        idx = data["tests"]["tests/test_marked.py::test_marked"]
        assert "f" in {table[i][1] for i in idx}  # deps recorded normally
        assert "tests/test_marked.py::test_marked" in data["always"]

    def test_removing_the_marker_takes_effect(self, marked_project):
        run_less(marked_project, "-q").assert_outcomes(passed=2)
        path = marked_project.path / "tests" / "test_marked.py"
        path.write_text(
            "import pytest\n\n"
            "def test_marked():\n"
            "    import lib\n"
            "    assert lib.f() == 1\n"
        )
        from test_plugin import shard_data

        run_less(marked_project, "-q").assert_outcomes(passed=1)  # the file changed
        data = shard_data(marked_project.path)
        assert data["always"] == []


class TestDynamicLibraryLoads:
    """``ctypes.CDLL`` and ``importlib.import_module`` wraps.

    dlopen is not an import: no ``IMPORT_NAME`` bytecode, no Python code
    objects, so the artifact is invisible to both the event model and the
    import scan.  The wrap records the resolved library as a file-level
    dependency.  ``import_module`` produces no ``IMPORT_NAME`` opcode (it
    is a plain function call), so the bytecode scan cannot see it either;
    the wrap records the resolved module's skeleton -- the ``sys.modules``
    gap, where only the first import runs the module's code."""

    def test_cdll_dep_resolves_paths(self, tmp_path):
        from pytest_less.monitoring import FileMonitor

        (tmp_path / "lib.so").write_bytes(b"\x00")
        monitor = FileMonitor(tmp_path, data_files=False)
        assert monitor._cdll_dep("lib.so") is None  # outside the repo (cwd)
        # a path inside the repo root
        monitor.repo_rel(str(tmp_path / "lib.so"))  # warm the cache
        assert monitor._cdll_dep(str(tmp_path / "lib.so")) == "lib.so"
        # a bare name resolves to None (not on a loader path into the repo)
        assert monitor._cdll_dep("libSystem") is None

    def test_track_cdll_records_when_active(self, tmp_path):
        from pytest_less.blocks import WILDCARD
        from pytest_less.monitoring import FileMonitor

        (tmp_path / "lib.so").write_bytes(b"\x00")
        monitor = FileMonitor(tmp_path, data_files=False)
        monitor._active = True
        monitor.phase = "test"
        monitor.current_nodeid = "t"
        monitor._track_cdll(str(tmp_path / "lib.so"))
        assert ("t", {("lib.so", WILDCARD, WILDCARD)}) in monitor.test_deps.items()
        # disabled: records nothing
        monitor._active = False
        monitor._track_cdll(str(tmp_path / "lib.so"))
        assert len(monitor.test_deps) == 1  # unchanged

    def test_import_module_on_a_cached_module_records_the_skeleton(self, pytester):
        make_project(
            pytester,
            {
                "lib.py": "CONST = 1\n\ndef f():\n    return CONST\n",
                "tests/conftest.py": "",
                "tests/test_a.py": "def test_a():\n    import lib\n    assert lib.f() == 1\n",
                "tests/test_b.py": (
                    "import importlib\n\n"
                    "def test_b():\n"
                    "    lib = importlib.import_module('lib')\n"
                    "    assert lib.CONST == 1\n"
                ),
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=2)
        # edit the module-level code: test_b holds the skeleton via the wrap
        (pytester.path / "lib.py").write_text("CONST = 30\n\ndef f():\n    return CONST\n")
        result = run_less(pytester, "-v")
        result.stdout.fnmatch_lines(["*selected 2/2 tests*"])

    def test_the_interceptors_install_and_remove(self):
        import ctypes
        import importlib
        import tempfile
        from pathlib import Path

        from pytest_less.monitoring import FileMonitor

        orig_init = ctypes.CDLL.__init__
        orig_im = importlib.import_module
        root = Path(tempfile.mkdtemp())
        monitor = FileMonitor(root, data_files=False)
        monitor.activate()
        assert ctypes.CDLL.__init__ is not orig_init
        assert importlib.import_module is not orig_im
        monitor.deactivate()
        assert ctypes.CDLL.__init__ is orig_init
        assert importlib.import_module is orig_im

    def test_a_disabled_monitor_records_nothing(self):
        import importlib
        import tempfile
        from pathlib import Path

        from pytest_less.monitoring import FileMonitor

        root = Path(tempfile.mkdtemp())
        monitor = FileMonitor(root, data_files=False)
        monitor._active = False
        importlib.import_module("json")  # stdlib: would be a no-op anyway
        assert not monitor.test_deps

    @pytest.mark.skipif(
        not __import__("shutil").which("cc"), reason="cc not available"
    )
    def test_a_ctypes_artifact_rebuild_is_caught_end_to_end(self, pytester):
        import subprocess as sp
        import sysconfig

        # ``cc -shared`` produces a loadable file named .so on every
        # platform we support (macOS included: dlopen accepts it)
        so = pytester.path / "libplain.so"
        c = pytester.path / "libplain.c"
        c.write_text("int answer(void) { return 42; }\n")
        inc = sysconfig.get_paths()["include"]
        flags = ["-shared", "-fPIC"]
        if sys.platform == "darwin":
            flags += ["-undefined", "dynamic_lookup"]

        def build():
            sp.run(
                ["cc", *flags, f"-I{inc}", str(c), "-o", str(so)],
                check=True,
            )
        build()
        make_project(
            pytester,
            {
                "tests/conftest.py": "",
                "tests/test_answer.py": (
                    "import ctypes\n"
                    "from pathlib import Path\n\n"
                    "def test_answer():\n"
                    "    so = Path(__file__).parent / '..' / 'libplain.so'\n"
                    "    lib = ctypes.CDLL(str(so))\n"
                    "    assert lib.answer() == 42\n"
                ),
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=1)
        c.write_text("int answer(void) { return 43; }\n")
        build()
        result = run_less(pytester, "-v")
        result.stdout.fnmatch_lines(["*selected 1/1 tests*"])
        result.assert_outcomes(failed=1)  # 43 != 42


class TestMemoizationWraps:
    """``functools.lru_cache`` / ``cache`` / ``cached_property``, scoped.

    A memoized function consumes the original's result without executing
    it on a cache hit, so ``PY_START`` never fires for later callers.
    The wrap patches the *decorator* at monitor activation; only repo
    functions get the tracking wrapper (checked at decoration time via
    ``co_filename``), stdlib and third-party cached functions keep the
    untouched C wrapper.  The dependency (the original function's block)
    is pre-computed at decoration time and added to the caller's bucket
    at every call, hit or miss."""

    @pytest.fixture
    def memo_project(self, pytester):
        make_project(
            pytester,
            {
                "lib.py": (
                    "import functools\n\n"
                    "@functools.lru_cache(maxsize=None)\n"
                    "def compute(x):\n"
                    "    return x * 10\n\n"
                    "@functools.cache\n"
                    "def other(x):\n"
                    "    return x + 1\n"
                ),
                "tests/conftest.py": "",
                # test_a makes the first call (cache miss: PY_START fires
                # AND the tracking wrapper records); test_b gets the cache
                # hit (only the tracking wrapper records)
                "tests/test_a.py": (
                    "def test_a():\n"
                    "    import lib\n"
                    "    assert lib.compute(1) == 10\n"
                ),
                "tests/test_b.py": (
                    "def test_b():\n"
                    "    import lib\n"
                    "    assert lib.compute(1) == 10\n"  # cache hit
                    "    assert lib.other(1) == 2\n"      # first call
                ),
                "tests/test_c.py": "def test_c():\n    assert 1 + 1 == 2\n",
            },
        )
        return pytester

    def test_every_caller_holds_the_function_block(self, memo_project):
        from test_plugin import shard_data

        run_less(memo_project, "-q").assert_outcomes(passed=3)
        data = shard_data(memo_project.path)
        table = [tuple(b) for b in data["blocks"]]
        for nid in ("tests/test_a.py::test_a", "tests/test_b.py::test_b"):
            deps = {table[i][1] for i in data["tests"][nid]}
            assert "compute" in deps, (nid, deps)

    def test_an_edit_selects_every_caller(self, memo_project):
        run_less(memo_project, "-q").assert_outcomes(passed=3)
        path = memo_project.path / "lib.py"
        path.write_text(path.read_text().replace("x * 10", "x * 100"))
        result = run_less(memo_project, "-v")
        # both the cache-miss caller and the cache-hit caller are selected
        result.stdout.fnmatch_lines(["*selected 2/3 tests*"])

    def test_the_wrapper_is_removed_on_deactivate(self):
        import functools
        import tempfile
        from pathlib import Path

        from pytest_less.monitoring import FileMonitor

        orig_lru = functools.lru_cache
        orig_cache = functools.cache
        orig_cp = functools.cached_property
        root = Path(tempfile.mkdtemp())
        monitor = FileMonitor(root, data_files=False)
        monitor.activate()
        assert functools.lru_cache is not orig_lru
        assert functools.cache is not orig_cache
        assert functools.cached_property is not orig_cp
        monitor.deactivate()
        assert functools.lru_cache is orig_lru
        assert functools.cache is orig_cache
        assert functools.cached_property is orig_cp

    def test_stdlib_cached_functions_are_untouched(self):
        """The scoping check: a stdlib function keeps the C wrapper."""
        import functools
        import tempfile
        from pathlib import Path

        from pytest_less.monitoring import FileMonitor

        root = Path(tempfile.mkdtemp())
        (root / "conftest.py").touch()
        monitor = FileMonitor(root, data_files=False)
        monitor.activate()
        try:
            # decorate a non-repo function: must get the real C wrapper
            @functools.cache
            def stdlib_fn(x):
                return x

            assert type(stdlib_fn).__name__ == "_lru_cache_wrapper"  # the C type
            assert not hasattr(stdlib_fn, "cache_info") or callable(stdlib_fn.cache_info)
            # the monitor was active but no dep was recorded (not repo code)
            assert not monitor.test_deps
        finally:
            monitor.deactivate()

    def test_a_repo_function_gets_the_tracking_wrapper(self):
        import functools
        import tempfile
        from pathlib import Path

        from pytest_less.monitoring import FileMonitor

        root = Path(tempfile.mkdtemp())
        (root / "lib.py").write_text("X = 1\n")
        monitor = FileMonitor(root, data_files=False)
        monitor.activate()
        try:
            # a function in the root (which IS the repo)
            # compile in the repo so _is_repo_code sees it as repo
            code = compile(
                "@functools.lru_cache\ndef f(x):\n    return x\n",
                str(root / "lib2.py"), "exec",
            )
            ns = {"__name__": "m", "functools": functools}
            exec(code, ns)  # noqa: S102 - test helper
            f = ns["f"]
            # the tracking wrapper has __wrapped__ pointing to the original
            assert f.__wrapped__.__name__ == "f"
            assert hasattr(f, "cache_info")
        finally:
            monitor.deactivate()

    def test_cached_property_on_a_shared_instance(self, pytester):
        """The third first-consumer-wins member, now closed: every consumer
        of a cached_property holds the original function's block."""
        make_project(
            pytester,
            {
                "lib.py": (
                    "import functools\n\n"
                    "class Shared:\n"
                    "    @functools.cached_property\n"
                    "    def config(self):\n"
                    "        return 10\n"
                ),
                "conftest.py": (
                    "import pytest\n"
                    "import lib\n\n"
                    '@pytest.fixture(scope="session")\n'
                    "def shared():\n"
                    "    return lib.Shared()\n"
                ),
                "tests/conftest.py": "",
                "tests/test_a.py": "def test_a(shared):\n    assert shared.config == 10\n",
                "tests/test_b.py": "def test_b(shared):\n    assert shared.config == 10\n",
            },
        )
        run_less(pytester, "-q").assert_outcomes(passed=2)
        path = pytester.path / "lib.py"
        path.write_text(path.read_text().replace("return 10", "return 100"))
        result = run_less(pytester, "-v")
        result.stdout.fnmatch_lines(["*selected 2/2 tests*"])
