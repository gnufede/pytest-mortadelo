"""What an import statement compiles to changed in 3.14 and 3.15; the scan must
read the same facts on every version.

The facts that matter to dependency discovery are the module name and the
relative-import *level* (``from .pkg import x`` is not the absolute ``pkg``).
These tests assert them against the source text, not against another
implementation: two implementations can agree and both be wrong.

* 3.14 loads the level with ``LOAD_SMALL_INT`` instead of ``LOAD_CONST``.
* 3.15 packs lazy/eager flags into IMPORT_NAME's low two bits, so ``co_names[arg]``
  is the wrong entry (the import used to vanish), and adds ``lazy import``.
* a load needing ``EXTENDED_ARG`` (a module with more than 255 constants) is a
  separate instruction in ``dis``, between the level load and IMPORT_NAME.
"""

import sys

import pytest

from pytest_less import monitoring
from pytest_less.monitoring import _imports_via_bytes, _imports_via_dis

RELATIVE = "from . import a\nfrom .. import b\nfrom ...pkg import c\nimport d\n"
EXPECTED = [
    ("IMPORT_NAME", "", 1),
    ("IMPORT_FROM", "a", 0),
    ("IMPORT_NAME", "", 2),
    ("IMPORT_FROM", "b", 0),
    ("IMPORT_NAME", "pkg", 3),
    ("IMPORT_FROM", "c", 0),
    ("IMPORT_NAME", "d", 0),
]


def scanned(source):
    code = compile(source, "m.py", "exec")
    via_dis = _imports_via_dis(code)
    fast = _imports_via_bytes(code) if monitoring._FAST_IMPORT_PRECHECK else None
    return via_dis, fast


@pytest.mark.parametrize("padding", [0, 40, 300], ids=["plain", "some-consts", "extended-arg"])
def test_relative_levels_are_read_from_the_source(padding):
    # distinct values >= 256 are real constants (not small ints): beyond 255 of
    # them, the level / fromlist loads below need an EXTENDED_ARG prefix
    pad = "".join(f"v{i} = {i + 1000}\n" for i in range(padding))
    via_dis, fast = scanned(pad + RELATIVE)
    assert via_dis == EXPECTED
    assert fast is None or fast == EXPECTED  # deferring to dis is fine; differing is not


def test_an_import_inside_a_function_keeps_its_level():
    source = "def f():\n    from ..up import x\n    import y\n"
    code = compile(source, "m.py", "exec")
    inner = next(c for c in code.co_consts if hasattr(c, "co_code"))
    assert _imports_via_dis(inner) == [
        ("IMPORT_NAME", "up", 2),
        ("IMPORT_FROM", "x", 0),
        ("IMPORT_NAME", "y", 0),
    ]


def test_many_names_do_not_move_the_name_lookup():
    # IMPORT_NAME's name index above 255 (EXTENDED_ARG on the import itself),
    # on a version where the operand also carries flags in its low bits
    source = "".join(f"import module_{i}\n" for i in range(300))
    via_dis, fast = scanned(source)
    assert [n for _, n, _ in via_dis] == [f"module_{i}" for i in range(300)]
    assert fast is None or fast == via_dis


@pytest.mark.skipif(sys.version_info < (3, 15), reason="PEP 810 lazy imports are 3.15+")
def test_a_lazy_import_is_still_an_import():
    # Over-approximating is the safe direction: record it as an import.
    via_dis, fast = scanned("lazy import json\nlazy from os import path\nimport sys\n")
    names = [(op, n) for op, n, _ in via_dis]
    assert ("IMPORT_NAME", "json") in names
    assert ("IMPORT_NAME", "os") in names and ("IMPORT_FROM", "path") in names
    assert ("IMPORT_NAME", "sys") in names
    assert fast is None or fast == via_dis
