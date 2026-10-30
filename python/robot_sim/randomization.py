"""Load portable DR profiles. Rust validates and applies them at episode reset."""
import copy
import json
from pathlib import Path
from importlib.resources import files
import tomllib


def load_profile(profile=None):
    if profile is None: return {}
    if isinstance(profile,dict): return copy.deepcopy(profile)
    if str(profile) in ('none','light','moderate'):
        return tomllib.loads(files('robot_sim').joinpath('profiles',f'{profile}.toml').read_text())
    path=Path(profile)
    return json.loads(path.read_text()) if path.suffix=='.json' else tomllib.loads(path.read_text())


def load_parameters(parameters=None):
    if parameters is None: return None
    if isinstance(parameters,dict): return copy.deepcopy(parameters)
    return json.loads(Path(parameters).read_text())
