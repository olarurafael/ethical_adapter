# src/ethical_adapter/inject.py
import torch
import torch.nn as nn
from types import SimpleNamespace
from .adapter import ParallelLinear
from .config import AdapterConfig
from .gate import GateController, GatedSourceWrapper


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
    gate_store = {}
    gate_ctrl = None

    if config.gate.enabled:
        parent, name, target = get_submodule(model, config.gate.source_module)
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            input_dim = model.config.hidden_size
        else:
            raise TypeError(
                "Model does not expose `config.hidden_size`; "
                "please specify gate input_dim manually."
            )

        gate_ctrl = GateController(
            input_dim=input_dim,
            hidden_dim=config.gate.hidden_size,
            num_gates=config.gate.num_gates,
            activation=config.gate.activation,
            pooling=config.gate.pooling,
        )
        # Move gate controller to same device as model
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        gate_ctrl = gate_ctrl.to(device=model_device, dtype=model_dtype)
        wrapped_source = GatedSourceWrapper(
            base_module=target,
            gate_controller=gate_ctrl,
            gate_store=gate_store,
            store_key="logits",
        )
        setattr(parent, name, wrapped_source)
        model_device = next(model.parameters()).device
        gate_store["logits"] = torch.zeros(
            1, config.gate.num_gates, device=model_device
        )
        model.gate_store = gate_store

    injected = []
    for idx, path in enumerate(config.target_modules):
        parent, name, target = get_submodule(model, path)

        if not isinstance(target, nn.Linear):
            raise TypeError(f"Target module '{path}' is not an nn.Linear.")

        if gate_ctrl:
            num_gates = config.gate.num_gates
            gate_index = idx % num_gates  # <-- universal logic
        else:
            gate_index = None

        wrapped = ParallelLinear(
            base_linear=target,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            gate_index=gate_index if gate_ctrl else None,
            gate_store=gate_store if gate_ctrl else None,
        )
        setattr(parent, name, wrapped)
        injected.append(path)

    # Return the model and metadata (for logging/debug)
    return SimpleNamespace(
        model=model,
        injected_layers=injected,
        gate_controller=gate_ctrl,
        gate_store=gate_store,
    )
