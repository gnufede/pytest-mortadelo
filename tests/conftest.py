import os

import pytest

pytest_plugins = ["pytester"]

# Hermetic git: the tests create throwaway repos and commit in them many
# times.  User-level config (global hooks such as secret scanners, SSH/GPG
# commit signing) must not apply -- it can turn a 1-minute suite into 10+.
# Env-based config overrides global/system config and is inherited by every
# subprocess (pytester runs, the plugin's own git calls).
_HERMETIC_GIT = (
    ("core.hooksPath", os.devnull),
    ("commit.gpgsign", "false"),
    ("tag.gpgsign", "false"),
)
os.environ["GIT_CONFIG_COUNT"] = str(len(_HERMETIC_GIT))
for _i, (_key, _value) in enumerate(_HERMETIC_GIT):
    os.environ[f"GIT_CONFIG_KEY_{_i}"] = _key
    os.environ[f"GIT_CONFIG_VALUE_{_i}"] = _value

# Variables that change what an inner ``pytest --less`` run does.  CI's dogfood
# job sets PYTEST_LESS_SYNC so the *outer* session records and pushes its shard;
# without this, every inner pytester run inherited it too and fetched/pushed
# against a remote of its own.  Cleared per test, never at import time: the
# outer plugin reads them while it configures, before any test runs.
_PLUGIN_ENV = ("PYTEST_LESS", "PYTEST_LESS_SYNC", "PYTEST_LESS_PARSE_CACHE")


@pytest.fixture(autouse=True)
def _inner_runs_do_not_inherit_the_outer_plugin_env(monkeypatch):
    for name in _PLUGIN_ENV:
        monkeypatch.delenv(name, raising=False)

