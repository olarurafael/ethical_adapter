# src/ethical_adapter/utils.py
import torch
import torch.nn as nn

def count_parameters(model: nn.Module):
    """
    Return a summary dict with total, trainable, and adapter parameters.
    Assumes adapter modules are named 'adapter' or contain 'adapter' in their path.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    adapter = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "adapter" in name.lower():
            adapter += param.numel()

    pct = 100.0 * adapter / total
    return {
        "total_params": total,
        "trainable_params": trainable,
        "adapter_params": adapter,
        "adapter_pct_total": pct,
    }


def print_param_summary(model: nn.Module):
    """Pretty print parameter stats."""
    stats = count_parameters(model)
    print("Parameter Summary")
    print("-----------------")
    for k, v in stats.items():
        print(f"{k:20s}: {v:,}")
