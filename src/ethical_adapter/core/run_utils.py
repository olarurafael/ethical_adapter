# src/ethical_adapter/run_utils.py
import json
import logging
import os
from datetime import datetime
from typing import Tuple
from pathlib import Path

from safetensors.torch import save_file


def log_trainable_params(logger: logging.Logger, model) -> Tuple[int, int]:
    """
    Compute and log total params and trainable params.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(
        "Params — total: %s | trainable: %s (%.4f%%)",
        f"{total:,}",
        f"{trainable:,}",
        pct,
    )
    return total, trainable


def setup_run(config):
    """Create a timestamped run directory, save config, and set up a basic file logger."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config["runs_dir"], timestamp)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(run_dir, "train.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.info("New training run started in: %s", run_dir)
    return run_dir, logger


def _adapter_gate_state_dict(model):
    """Return only the small trainable steering state, not base-model weights."""
    keep = {}
    for name, tensor in model.state_dict().items():
        if ".adapter." in name or ".gate_controller." in name:
            keep[name] = tensor.detach().cpu().contiguous()
    return keep


def save_checkpoint(model, tokenizer, run_dir, epoch, logger, best=False):
    """Save model checkpoints for the given epoch."""
    name = f"epoch_{epoch}" if not best else "best"
    ckpt_dir = os.path.join(run_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info("Saved checkpoint to %s", ckpt_dir)


def save_adapter_checkpoint(model, tokenizer, run_dir, epoch, logger, best=False):
    """Save only adapter/gate weights for the given epoch."""
    name = f"epoch_{epoch}" if not best else "best"
    ckpt_dir = os.path.join(run_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    steering_state = _adapter_gate_state_dict(model)
    if not steering_state:
        raise RuntimeError("No adapter or gate weights found to save.")

    save_file(steering_state, os.path.join(ckpt_dir, "model.safetensors"))
    tokenizer.save_pretrained(ckpt_dir)

    manifest = {
        "checkpoint_type": "adapter_gate_only",
        "num_tensors": len(steering_state),
        "tensor_names": sorted(steering_state),
    }
    with open(os.path.join(ckpt_dir, "checkpoint_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Saved adapter/gate checkpoint to %s (%s tensors)",
        ckpt_dir,
        len(steering_state),
    )


def save_training_checkpoint(
    model,
    tokenizer,
    run_dir,
    epoch,
    logger,
    best=False,
    adapter_only=False,
):
    if adapter_only:
        return save_adapter_checkpoint(
            model, tokenizer, run_dir, epoch, logger, best=best
        )
    return save_checkpoint(model, tokenizer, run_dir, epoch, logger, best=best)


def get_latest_best_checkpoint(runs_dir="./runs"):
    """
    Return the path to the most recent run's 'best' checkpoint, if any.
    Looks for ./runs/YYYY-MM-DD_HH-MM-SS/best/
    """
    runs = sorted(Path(runs_dir).glob("*"), key=os.path.getmtime, reverse=True)
    for run in runs:
        best_dir = run / "best"
        if best_dir.exists():
            return str(best_dir)
    return None
