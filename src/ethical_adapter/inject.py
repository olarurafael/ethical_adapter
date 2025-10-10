import torch.nn as nn
from types import SimpleNamespace
from .adapter import ParallelLinear
from .config import AdapterConfig


def get_submodule(model: nn.Module, dotted_path: str):
    """
    Traverse a model by a dotted path (e.g., 'layers.16.mlp.down_proj')
    and return (parent_module, attribute_name, target_module).
    """
    parts = dotted_path.split(".")
    current = model
    for name in parts[:-1]:
        current = getattr(current, name)
    parent = current
    attr_name = parts[-1]
    target = getattr(parent, attr_name)
    return parent, attr_name, target


def inject_adapters(model: nn.Module, config: AdapterConfig) -> nn.Module:
    """
    Replace selected nn.Linear modules in 'model' with ParallelLinear adapters.
    The base weights are frozen; adapters are newly created and trainable.
    """
    injected = []
    for path in config.target_modules:
        parent, name, target = get_submodule(model, path)

        if not isinstance(target, nn.Linear):
            raise TypeError(f"Target module '{path}' is not an nn.Linear.")

        # Replace the original linear with our wrapped version
        wrapped = ParallelLinear(
            base_linear=target,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, name, wrapped)
        injected.append(path)

    # Return the model and metadata (for logging/debug)
    return SimpleNamespace(model=model, injected_layers=injected)
