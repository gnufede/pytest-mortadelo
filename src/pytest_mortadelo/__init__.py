"""pytest-mortadelo: block-level test impact analysis (TIA) for pytest.

Records which code blocks each test depends on using ``sys.monitoring``
(PY_START/PY_RESUME events plus a conservative import-bytecode scan), stores
the data as JSON under ``.pytest_mortadelo/``, and uses file fingerprints to
select only the tests affected by changed files.  The file-level sibling is
``pytest-filemon``.
"""

__version__ = "0.1.0"
