"""
src/utils/config.py
-------------------
Utility for loading and validating YAML study config files.
"""

import pathlib
from typing import Any

import yaml


def load_config(config_path: str | pathlib.Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config as a Python dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as fh:
        config = yaml.safe_load(fh)

    return config or {}


def get_required(config: dict, key: str) -> Any:
    """Retrieve a required key from the config, raising an error if absent.

    Args:
        config: The config dictionary.
        key: Dot-notation or simple key to look up.

    Returns:
        The value associated with the key.

    Raises:
        KeyError: If the key is not present.
    """
    if key not in config:
        raise KeyError(f"Required config key '{key}' is missing.")
    return config[key]
