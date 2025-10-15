import torch
import os

def load_adapters_from_checkpoint(model, checkpoint_dir, logger=None):
    """
    Load adapter weights directly from an existing training run checkpoint.
    Expects that inject_adapters() has already been called.
    """
    log = logger.info if logger else print

    adapter_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"[ERROR] No model.safetensors found in {checkpoint_dir}")

    log(f"[INFO] Loading adapter weights from {adapter_path}")

    state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    # keep only adapter weights
    adapter_state = {k: v for k, v in state.items() if "adapter" in k}
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)

    if missing:
        log(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        log(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    log("[INFO] Adapter weights loaded successfully.")
    return model
