# src/ethical_adapter/run_utils.py
import json
import logging
import os
from datetime import datetime
from typing import Dict, Tuple


def make_run_dir(base_dir: str = "./runs") -> str:
    """
    Create a unique run directory like ./runs/DATE_TIME/
    and return its path. 
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(run_dir: str, cfg: Dict) -> str:
    """
    Save the configuration dict to <run_dir>/config.json for reproducibility.
    """
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return path


def get_logger(run_dir: str, name: str = "train") -> logging.Logger:
    """
    Create a logger that writes to both stdout and <run_dir>/train.log.
    - INFO level to console
    - DEBUG level to file (more verbose)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # master level

    # avoid duplicate handlers if called twice
    if logger.handlers:
        return logger

    # format
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # file handler (DEBUG+)
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # be quiet from overly chatty libs unless critical
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logger


def log_trainable_params(logger: logging.Logger, model) -> Tuple[int, int]:
    """
    Compute and log total params and trainable params.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(f"Params — total: {total:,} | trainable: {trainable:,} ({pct:.4f}%)")
    return total, trainable
