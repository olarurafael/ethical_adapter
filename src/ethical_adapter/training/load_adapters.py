# src/ethical_adapter/load_adapters.py

import os
import logging
from safetensors.torch import load_file


def load_adapters_from_checkpoint(
    model, checkpoint_dir, logger: logging.Logger | None = None
):
    """
    Load adapter weights from a safetensors checkpoint (runs/.../best).
    """
    logger = logger or logging.getLogger(__name__)

    import json

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")

    state = {}

    if os.path.exists(index_path):
        logger.info("Detected sharded checkpoint")

        with open(index_path, "r") as f:
            index = json.load(f)

        shard_files = sorted(
            set(os.path.join(checkpoint_dir, f) for f in index["weight_map"].values())
        )

        for shard in shard_files:
            logger.info(f"Loading shard: {shard}")
            shard_state = load_file(shard, device="cpu")
            state.update(shard_state)

    else:
        adapter_path = os.path.join(checkpoint_dir, "model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"[ERROR] No model.safetensors or index found in {checkpoint_dir}"
            )

        logger.info(f"Loading adapter weights from {adapter_path}")
        state = load_file(adapter_path, device="cpu")

    # keep only adapter-related weights
    adapter_state = {k: v for k, v in state.items() if "adapter" in k}

    # detect model device + dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # move adapter modules to correct device/dtype
    for module in model.modules():
        if hasattr(module, "adapter") and module.adapter is not None:
            module.adapter.to(device=device, dtype=dtype)

    # move the state tensors to correct device and dtype, then load
    adapter_state = {
        k: v.to(device=device, dtype=dtype) for k, v in adapter_state.items()
    }
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)

    if missing:
        logger.warning("Missing keys (%s): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%s): %s", len(unexpected), unexpected[:5])

    logger.info("Adapter weights loaded successfully.")
    return model
