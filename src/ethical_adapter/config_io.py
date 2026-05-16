import os
from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a Python dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML config file {path} is empty or invalid.")

    for key in [
        "lr",
        "alpha",
        "dropout",
        "rank",
        "batch_size",
        "epochs",
        "max_length",
        "save_every",
    ]:
        if key in cfg and isinstance(cfg[key], str):
            try:
                cfg[key] = (
                    float(cfg[key])
                    if "." in cfg[key] or "e" in cfg[key].lower()
                    else int(cfg[key])
                )
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
