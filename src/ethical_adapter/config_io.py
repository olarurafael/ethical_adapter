# src/ethical_adapter/config_io.py
import yaml
import os
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file into a Python dict.
    If the path is relative, resolve it relative to the project root.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML config file {path} is empty or invalid.")

    # sanitize numeric types
    for key in ["lr", "alpha", "dropout", "rank", "batch_size", "epochs", "max_length", "save_every"]:
        if key in cfg and isinstance(cfg[key], str):
            try:
                # convert scientific notation or numeric strings
                cfg[key] = float(cfg[key]) if "." in cfg[key] or "e" in cfg[key].lower() else int(cfg[key])
            except ValueError:
                pass  # leave as string if truly non-numeric
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """
    Ensure directories like 'data_dir' and 'runs_dir' exist.
    """
    for key in ["data_dir", "runs_dir"]:
        if key in cfg:
            os.makedirs(cfg[key], exist_ok=True)
