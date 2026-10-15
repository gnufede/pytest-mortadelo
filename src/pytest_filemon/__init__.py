"""pytest-filemon: file-level test impact analysis (TIA) for pytest.

Records which repository files each test depends on using ``sys.monitoring``
(PY_START events plus a conservative import-bytecode scan), stores the data as
JSON shards in dedicated git refs, and uses ``git diff`` to select only the
tests affected by changed files.
"""

__version__ = "0.1.0"
