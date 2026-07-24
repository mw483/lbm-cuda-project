# config.py
import sys

# Master Control Switch: Set to "LAB" or "TSUBAME"
TARGET_ENV = "LAB"

from .config_base import BASE_PARAMS

if TARGET_ENV == "LAB":
    from . import config_local as env
elif TARGET_ENV == "TSUBAME":
    from . import config_tsubame as env
else:
    print(f"Error: Unknown TARGET_ENV string configuration parameter: {TARGET_ENV}")
    sys.exit(1)

# Expose the correct execution variables to the template scripts
PARTICLE_SOURCES = env.PARTICLE_SOURCES
PARTICLE_OUTPUT  = env.PARTICLE_OUTPUT

def deep_merge(dict_base, dict_extend):
    for key, val in dict_extend.items():
        if key in dict_base and isinstance(dict_base[key], dict) and isinstance(val, dict):
            deep_merge(dict_base[key], val)
        else:
            dict_base[key] = val

# Combine dictionaries
PARAMS = BASE_PARAMS.copy()
deep_merge(PARAMS, env.ENV_PARAMS)