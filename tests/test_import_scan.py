"""The import scan finds IMPORT_NAME / IMPORT_FROM without disassembling.

Every first-start analysis scans the code object for imports (a conservative
over-approximation: an import in a branch that did not run still counts).  The
reference implementation decoded the whole code object with ``dis`` -- ~1 ms for
a module-level block, 0.43 s of a 0.69 s warm analysis on the sqlalchemy
corpus.  The byte scanner finds the two opcodes directly (cost proportional to
the number of imports) and defers to ``dis`` whenever the surrounding bytecode
is not the plain shape, so it may only ever be faster, never different.
"""

import sys
import sysconfig
import types
from pathlib import Path

import pytest

from pytest_less import monitoring
from pytest_less.monitoring import _imports_via_bytes, _imports_via_dis

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 13), reason="the byte scan relies on the 3.13 co_code"
)


def all_codes(code):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from all_codes(const)


def compiled(source, name="m.py"):
    return compile(source, name, "exec")


class TestAgreesWithDis:
    @pytest.mark.parametrize(
        "source",
        [
            "import os\n",
            "import os.path, sys\n",
            "import a.b.c as d\n",
            "from os import path, sep\n",
            "from . import sibling\n",
            "from .. import up as renamed\n",
            "from .pkg import name\n",
            "from ...deep.mod import (x, y, z)\n",
            "def f():\n    import json\n    from collections import abc\n",
            "class C:\n    import re\n    def m(self):\n        from . import x\n",
            "try:\n    import ujson as json\nexcept ImportError:\n    import json\n",
            "if False:\n    import never_runs\n",
            "x = [__import__('os')]\n",  # no IMPORT_NAME: a call, not an import op
            "import os\nimport os\nfrom os import path\n",
            "",
        ],
    )
    def test_statement_forms(self, source):
        for code in all_codes(compiled(source)):
            fast = _imports_via_bytes(code)
            assert fast is not None
            assert fast == _imports_via_dis(code)

    def test_more_than_255_names_forces_extended_arg_on_the_import_itself(self):
        # co_names index > 255 -> the import opcodes carry EXTENDED_ARG prefixes,
        # while the level / fromlist constants stay few (0 and None are shared).
        code = compiled("".join(f"import module_{i}\n" for i in range(300)))
        assert len(code.co_names) > 255
        fast = _imports_via_bytes(code)
        assert fast is not None
        assert fast == _imports_via_dis(code)
        assert len(fast) == 300

    def test_distinct_fromlists_beyond_255_defer_to_dis(self):
        # 300 distinct ``('thing_i',)`` constants: the fromlist loads need
        # EXTENDED_ARG, which the scanner does not decode -- it must say so
        # (None) rather than guess, and dis then answers.
        code = compiled("".join(f"from pkg_{i} import thing_{i}\n" for i in range(300)))
        assert len(code.co_consts) > 255
        fast = _imports_via_bytes(code)
        assert fast is None
        assert len(_imports_via_dis(code)) == 600

    def test_more_than_255_constants_in_front_of_the_import(self):
        # constants beyond one byte before the level/fromlist loads
        pad = "".join(f"v{i} = {i + 1000}\n" for i in range(300))
        code = compiled(pad + "from . import late\nimport tail\n")
        fast = _imports_via_bytes(code)
        assert fast is None or fast == _imports_via_dis(code)  # never *different*

    def test_a_byte_that_looks_like_an_opcode_is_not_mistaken_for_one(self):
        # IMPORT_NAME's opcode value as an *argument* byte of another unit
        value = monitoring._IMPORT_NAME_OP
        code = compiled(f"x = names_{value}\n" + "\n".join(f"v{i}={i}" for i in range(value + 5)))
        assert _imports_via_bytes(code) == _imports_via_dis(code) == []


def corpus():
    root = Path(sysconfig.get_paths()["stdlib"])
    skip = {"test", "idlelib", "lib2to3", "tkinter", "turtledemo", "site-packages"}
    files = [
        p
        for p in sorted(root.rglob("*.py"))
        if not skip.intersection(p.relative_to(root).parts)
    ]
    files += sorted(Path(monitoring.__file__).parent.glob("*.py"))
    files += sorted(Path(__file__).parent.glob("*.py"))
    return files


def test_identical_on_the_stdlib_and_this_repository():
    checked = fallbacks = imports = 0
    for path in corpus():
        try:
            code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for sub in all_codes(code):
            if (
                monitoring._IMPORT_NAME_OP not in sub.co_code
                and monitoring._IMPORT_FROM_OP not in sub.co_code
            ):
                continue
            reference = _imports_via_dis(sub)
            fast = _imports_via_bytes(sub)
            if fast is None:
                fallbacks += 1
                continue
            assert fast == reference, (path, sub.co_name)
            checked += 1
            imports += len(reference)
    assert checked > 1000 and imports > 4000, (checked, imports)
    # the fast path really is the common case
    assert fallbacks / (checked + fallbacks) < 0.05, (fallbacks, checked)


def test_analysis_uses_the_byte_scan_and_falls_back_to_dis(tmp_path, monkeypatch):
    # _analyze must give the same dependencies whichever scanner answers.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("def f():\n    from . import b\n    import pkg.c\n")
    (tmp_path / "pkg" / "b.py").write_text("X = 1\n")
    (tmp_path / "pkg" / "c.py").write_text("Y = 1\n")
    code = compile((tmp_path / "pkg" / "a.py").read_text(), str(tmp_path / "pkg" / "a.py"), "exec")
    f_code = next(c for c in all_codes(code) if c.co_name == "f")

    def deps(use_bytes):
        monitor = monitoring.FileMonitor(tmp_path)
        if not use_bytes:
            monkeypatch.setattr(monitoring, "_imports_via_bytes", lambda c: None)
        return monitor._analyze(f_code)

    with_bytes = deps(True)
    monkeypatch.undo()
    with_dis = deps(False)
    assert with_bytes == with_dis
    assert {"pkg/b.py", "pkg/c.py"} <= {d[0] for d in with_bytes}
