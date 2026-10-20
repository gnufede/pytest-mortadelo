"""Block hashes from compiled code objects.

``compile()`` is a C call -- cheaper than ``ast.parse`` alone, which has to
build Python objects -- and a code object already *is* a tree of blocks: one per
``def``, one per class body, one for the module.  Hashing the 220 files / 8.8 MB
a sqlalchemy record run touches takes 0.7 s this way (0.53 s of it the compiler);
walking and dumping an AST took 2.4 s.

What is hashed is the code compiled **from the file on disk**, never the code
object that is running: pytest rewrites asserts in test modules (and other
import hooks transform code), so the live bytecode cannot be reproduced from
the source at selection time, but ``compile(source)`` is reproducible on both
sides.

What the hash is and is not sensitive to (each point is tested):

* Moving code, blank lines and comments change nothing: line numbers are never
  hashed.  CPython 3.13 injects ``__firstlineno__`` (the class's own line!) and
  ``__static_attributes__`` (attributes assigned in *method bodies*) into every
  class body; both are masked, or a class skeleton would change whenever a line
  was added above it or a method gained ``self.x = 1``.
* Anything the compiler discards cannot matter: ``1 + 0``, dead branches,
  bare string statements, annotations of local variables.  Equal bytecode is
  equal behavior.
* **Docstrings are behavior**: editing one changes the hash of its block (and,
  for a class or module docstring, of the skeleton every importer depends on).
  Ignoring prose needed a per-block docstring hash and a "reads ``__doc__``"
  scan to stay safe, and on a codebase that reads docstrings itself (sqlalchemy)
  it saved nothing; this prefers to select a few extra tests.
* A class / module hash is a *skeleton*: ``def`` bodies are replaced by their
  signature, everything else (class-level statements, decorators, defaults,
  annotations, lambdas, comprehensions) is included.  Decorators, defaults
  and annotations are evaluated by the *parent*, so an edit to one changes the
  parent's block rather than the function's own; the parent always runs when
  the function object is created.
* The constant table is not part of a class / module identity: inserting a
  constant (the injected ``__firstlineno__``, colliding with an equal int
  elsewhere) renumbers every later ``LOAD_CONST``.  Those bodies are hashed by
  the *value* each ``LOAD_CONST`` loads.  A function's own table is a function
  of its own body, so its raw bytes are stable.

Anything not modelled raises :class:`Unsupported` (PEP 695 generics / ``type``
statements, interpreters other than 3.12 / 3.13) and the file becomes a
file-level dependency: any change to it re-runs the tests that executed it.
Hashes carry a ``c<minor>`` prefix, so two Python versions can never be mistaken
for each other: a mismatch is simply "changed".
"""

from __future__ import annotations

import hashlib
import opcode
import re
import sys
import types
import warnings

from .blocks import MODULE_BLOCK, FileBlocks

SUPPORTED = sys.version_info[:2] in {(3, 12), (3, 13)}
TAG = f"c{sys.version_info[1]}"  # bytecode differs between minor versions

_CO_OPTIMIZED = 0x1
_CO_VARARGS = 0x4
_CO_VARKEYWORDS = 0x8
_CO_ASYNC = 0x80 | 0x200  # CO_COROUTINE | CO_ASYNC_GENERATOR
# Structural flags only.  The high bits are the ``__future__`` features in
# effect for the *file* (CO_FUTURE_ANNOTATIONS, ...): they leak into every
# function of a file that toggles one, while their real effect is in the
# parent's bytecode.
_STRUCTURAL_FLAGS = 0xFFFF

_OP = opcode.opmap
_EXTENDED_ARG = _OP["EXTENDED_ARG"]
_STORE_NAME = _OP["STORE_NAME"]
_LOAD_CONST = _OP["LOAD_CONST"]
# (opcode.hasconst lists pseudo-instructions too, numbered >= 256: they never
# appear in co_code, and formatting them into a pattern would match byte 0x10.)
_CONST_OPS = sorted(set(opcode.hasconst) & {o for o in _OP.values() if o < 256})
_CONST_OP_RE = re.compile(b"[" + b"".join(b"\\x%02x" % o for o in _CONST_OPS) + b"]")

# Stored by the compiler itself in every 3.13 class body, derived from things
# that have their own blocks (the class's line, its methods' bodies).
_MASKED_STORES = frozenset({"__firstlineno__", "__static_attributes__"})

_TYPE_STMT = re.compile(rb"^[ \t]*type[ \t]+[A-Za-z_]\w*[ \t]*(\[|=)", re.MULTILINE)


class Unsupported(Exception):
    """The file uses something this hasher does not model: the file is then a file-level dependency."""


def _key(c: object) -> str:
    """A stable, unambiguous text form of a constant (no hash-order leaks)."""
    t = type(c)
    if t is tuple:
        return "(" + ",".join([_key(x) for x in c]) + ")"
    if t is frozenset:
        return "{" + ",".join(sorted([_key(x) for x in c])) + "}"
    if t in (str, int, float, bool, bytes, complex, type(None), type(Ellipsis)):
        return repr(c)
    return t.__name__ + repr(c)


def _digest(parts: list[bytes]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p)
        h.update(b"\x00")
    return TAG + h.hexdigest()


class _Analysis:
    """One pass over a compiled module: a digest per block."""

    def __init__(self) -> None:
        self._memo: dict[tuple[int, bool], str] = {}
        # (qualname, first line, digest) of every def and class body
        self.defs: list[tuple[str, int, str]] = []

    # -- classification -------------------------------------------------
    @staticmethod
    def _is_def(code: types.CodeType) -> bool:
        return bool(code.co_flags & _CO_OPTIMIZED) and not code.co_name.startswith("<")

    @staticmethod
    def _is_scope(code: types.CodeType) -> bool:
        """A class body or the module: not optimized, runs at definition."""
        return not (code.co_flags & _CO_OPTIMIZED)

    @staticmethod
    def _stub(code: types.CodeType) -> str:
        """What a class / module skeleton keeps of a nested ``def``: its
        signature (names, arity, kinds, async-ness).  The body lives in the
        function's own block; decorators, defaults and annotations are in the
        *parent's* bytecode."""
        n = (
            code.co_argcount
            + code.co_kwonlyargcount
            + bool(code.co_flags & _CO_VARARGS)
            + bool(code.co_flags & _CO_VARKEYWORDS)
        )
        return _digest(
            [
                b"stub",
                repr(
                    (
                        code.co_name,
                        bool(code.co_flags & _CO_ASYNC),
                        code.co_argcount,
                        code.co_posonlyargcount,
                        code.co_kwonlyargcount,
                        bool(code.co_flags & _CO_VARARGS),
                        bool(code.co_flags & _CO_VARKEYWORDS),
                        code.co_varnames[:n],
                    )
                ).encode(),
            ]
        )

    @staticmethod
    def _header(code: types.CodeType) -> bytes:
        return repr(
            (
                code.co_name,
                code.co_flags & _STRUCTURAL_FLAGS,
                code.co_argcount,
                code.co_posonlyargcount,
                code.co_kwonlyargcount,
                code.co_names,
                code.co_varnames,
                code.co_freevars,
                code.co_cellvars,
            )
        ).encode()

    # -- digests --------------------------------------------------------
    def digest(self, code: types.CodeType, skeleton: bool) -> str:
        """Digest of *code*.  With *skeleton*, ``def`` bodies nested directly
        in class / module bodies are replaced by their signature stub."""
        scope = self._is_scope(code)
        memo_key = (id(code), skeleton if scope else False)
        got = self._memo.get(memo_key)
        if got is None:
            got = self._scope(code, skeleton) if scope else self._function(code)
            self._memo[memo_key] = got
        return got

    def _function(self, code: types.CodeType) -> str:
        """A ``def`` / lambda / comprehension: bytecode + constants.  The
        constant table of a function is a function of its own body only, so
        its raw bytes are stable under unrelated edits."""
        keys = [
            "C" + self.digest(c, False) if type(c) is types.CodeType else _key(c)
            for c in code.co_consts
        ]
        return _digest(
            [
                code.co_code,
                code.co_exceptiontable,
                self._header(code),
                "\x01".join(keys).encode("utf-8", "surrogatepass"),
            ]
        )

    def _scope(self, code: types.CodeType, skeleton: bool) -> str:
        """A class body or module, hashed by the *value* each ``LOAD_CONST``
        loads (the injected ``__firstlineno__`` shifts the constant table)."""
        raw = code.co_code
        consts = code.co_consts
        names = code.co_names
        canon = bytearray(raw)
        keys: list[str] = []
        masks = any(n in _MASKED_STORES for n in names)
        for m in _CONST_OP_RE.finditer(raw):
            at = m.start()
            if at & 1:
                continue  # an argument byte that happens to equal an opcode
            arg = raw[at + 1]
            shift = 8
            j = at - 2
            while j >= 0 and raw[j] == _EXTENDED_ARG:
                arg |= raw[j + 1] << shift
                canon[j + 1] = 0
                shift += 8
                j -= 2
            canon[at + 1] = 0
            const = consts[arg]
            if masks and raw[at] == _LOAD_CONST:
                # LOAD_CONST value; [EXTENDED_ARG...]; STORE_NAME <name>
                store = at + 2
                hi = 0
                while store < len(raw) and raw[store] == _EXTENDED_ARG:
                    hi = (hi << 8) | raw[store + 1]
                    store += 2
                if (
                    store < len(raw)
                    and raw[store] == _STORE_NAME
                    and names[(hi << 8) | raw[store + 1]] in _MASKED_STORES
                ):
                    keys.append("<masked>")
                    continue
            if type(const) is types.CodeType:
                if skeleton and self._is_def(const):
                    keys.append("D" + self._stub(const))
                else:
                    keys.append("C" + self.digest(const, skeleton))
            else:
                keys.append(_key(const))
        return _digest(
            [
                bytes(canon),
                code.co_exceptiontable,
                self._header(code),
                "\x01".join(keys).encode("utf-8", "surrogatepass"),
            ]
        )

    # -- block collection ---------------------------------------------------
    def collect(self, code: types.CodeType) -> None:
        """Record every ``def`` and class body below *code* as a block."""
        for c in code.co_consts:
            if type(c) is not types.CodeType:
                continue
            if c.co_name.startswith("<generic parameters"):
                # PEP 695: the wrapper scope creates the function, so a
                # skeleton would have to stub it from inside the wrapper.
                raise Unsupported("PEP 695 generics")
            if self._is_def(c):
                self.defs.append((c.co_qualname, c.co_firstlineno, self.digest(c, False)))
            elif self._is_scope(c):
                self.defs.append((c.co_qualname, c.co_firstlineno, self.digest(c, True)))
            self.collect(c)


def compile_blocks(source: bytes, *, filename: str = "<src>") -> FileBlocks | None:
    """Block table of *source* from its compiled code; None if it does not
    compile.  Raises :class:`Unsupported` for constructs this hasher does not
    model (the caller then treats the file as a file-level dependency)."""
    if not SUPPORTED:
        raise Unsupported(f"python {sys.version_info[:2]}")
    if _TYPE_STMT.search(source):
        raise Unsupported("type statement")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            top = compile(source, filename, "exec", dont_inherit=True, optimize=0)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    work = _Analysis()
    work.collect(top)
    module_digest = work.digest(top, True)

    # A qualified name defined more than once with *different* content
    # (``@overload`` stubs, property setters, conditional definitions) gets one
    # id per definition, ``qname~<hash>``, resolved by source line.
    distinct: dict[str, set[str]] = {}
    for qname, _, content in work.defs:
        distinct.setdefault(qname, set()).add(content)
    blocks: dict[str, str] = {}
    spans: list[tuple[int, str]] = []
    for qname, start, content in work.defs:
        block_id = qname if len(distinct[qname]) == 1 else f"{qname}~{content}"
        blocks[block_id] = content
        spans.append((start, block_id))
    blocks[MODULE_BLOCK] = module_digest
    return FileBlocks(blocks, spans, None, source)
