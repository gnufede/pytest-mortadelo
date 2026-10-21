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

import pytest
from test_plugin import commit, git, run_less


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
