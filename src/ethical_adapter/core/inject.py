# src/ethical_adapter/inject.py
from dataclasses import dataclass

import torch.nn as nn

from .adapter import GatedAdapter
from .config import AdapterConfig
from .gate import GateCache, GateController, GatePredictor


@dataclass
class InjectedAdapters:
    model: nn.Module
    injected_layers: list[str]
    gate_controller: GateController | None
    gate_cache: GateCache | None


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


def inject_adapters(model: nn.Module, config: AdapterConfig) -> InjectedAdapters:
    """
    Replace selected nn.Linear modules in 'model' with gated adapters.
    The base weights are frozen; adapters are newly created and trainable.
    """
    gate_cache = None
    gate_ctrl = None

    if config.gate.enabled:
        gate_cache = GateCache()
        parent, name, target = get_submodule(model, config.gate.source_module)
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            input_dim = model.config.hidden_size
        else:
            raise TypeError(
                "Model must expose `config.hidden_size` for gate injection."
            )

        gate_ctrl = GateController(
            input_dim=input_dim,
            temperature=config.gate.temperature,
            dropout=config.gate.dropout,
            pooling=config.gate.pooling,
        )
        # Move gate controller to same device as model
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        gate_ctrl = gate_ctrl.to(device=model_device, dtype=model_dtype)
        wrapped_source = GatePredictor(
            base_module=target,
            gate_controller=gate_ctrl,
            gate_cache=gate_cache,
        )
        setattr(parent, name, wrapped_source)
        model.gate_cache = gate_cache

    injected = []
    for path in config.target_modules:
        parent, name, target = get_submodule(model, path)

        if not isinstance(target, nn.Linear):
            raise TypeError(f"Target module '{path}' is not an nn.Linear.")

        wrapped = GatedAdapter(
            base_linear=target,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            gate_cache=gate_cache,
            gate_hard_threshold=config.gate.hard_threshold if gate_ctrl else None,
        )
        setattr(parent, name, wrapped)
        injected.append(path)

    return InjectedAdapters(
        model=model,
        injected_layers=injected,
        gate_controller=gate_ctrl,
        gate_cache=gate_cache,
    )
