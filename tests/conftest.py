import os

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
