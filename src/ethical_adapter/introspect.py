# src/ethical_adapter/introspect.py
import torch
import torch.nn as nn
from .adapter import ParallelLinear


def list_adapters(model: nn.Module):
    """
    Return a list of tuples (name, module, param_count) for all ParallelLinear adapters.
    """
    adapters = []
    for name, module in model.named_modules():
        if isinstance(module, ParallelLinear):
            count = sum(p.numel() for p in module.adapter.parameters())
            adapters.append((name, module, count))
    return adapters


def print_adapter_summary(model: nn.Module):
    """Print a concise report of all adapters inside the model."""
    adapters = list_adapters(model)
    if not adapters:
        print("No adapters found.")
        return

    print(f"{'Layer':60s} | {'Params':>10s} | {'In':>6s} | {'Out':>6s} | {'Rank':>4s}")
    print("-" * 95)
    for name, module, count in adapters:
        print(f"{name:60s} | {count:10,d} | {module.in_features:6d} | {module.out_features:6d} | {module.adapter.rank:4d}")
    print("-" * 95)
    total = sum(c for _, _, c in adapters)
    print(f"Total adapter parameters: {total:,}")


def print_layer_map(model: nn.Module, pattern: str = "decoder.layers"):
    """
    Print a numbered list of all layers matching a pattern.
    Helps to identify which indices correspond to early/mid/late positions.
    Example pattern: 'decoder.layers' or 'transformer.h'
    """
    matches = []
    for name, module in model.named_modules():
        if pattern in name:
            matches.append(name)

    if not matches:
        print(f"No layers found matching pattern '{pattern}'.")
        return

    print(f"\nLayer map for pattern '{pattern}':")
    print("-" * 60)
    for i, name in enumerate(matches):
        print(f"{i:3d}: {name}")
    print("-" * 60)
    print(f"Total matched layers: {len(matches)}")
