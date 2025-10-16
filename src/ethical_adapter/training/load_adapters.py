# src/ethical_adapter/load_adapters.py

import os
import logging
from safetensors.torch import load_file

def load_adapters_from_checkpoint(model, 
                                  checkpoint_dir, 
                                  logger: logging.Logger | None = None):
    """
    Load adapter weights from a safetensors checkpoint (runs/.../best).
    """
    logger = logger or logging.getLogger(__name__)

    adapter_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"[ERROR] No model.safetensors found in {checkpoint_dir}")

    logger.info(f" Loading adapter weights from {adapter_path}")

    # load tensors
    state = load_file(adapter_path, device="cpu")

    # keep only adapter-related weights
    adapter_state = {k: v for k, v in state.items() if "adapter" in k}

    # detect model device + dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # move adapter modules to correct device/dtype
    for name, module in model.named_modules():
        if "adapter" in name:
            module.to(device=device, dtype=dtype)

    # move the state tensors and load
    adapter_state = {k: v.to(device) for k, v in adapter_state.items()}
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)

    if missing:
        logger.warning("Missing keys (%s): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%s): %s", len(unexpected), unexpected[:5])

    logger.info("Adapter weights loaded successfully.")
    return model
