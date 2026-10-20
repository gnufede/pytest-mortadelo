"""Test helper: the block table of a source string."""

from pytest_less.blocks import FileBlocks
from pytest_less.codeblocks import compile_blocks


def table_of(source: str) -> FileBlocks | None:
    """What the plugin would record for a file holding *source* (None if it
    does not compile)."""
    return compile_blocks(source.encode())
