# src/ethical_adapter/training/optim_utils.py
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from ethical_adapter.core.adapter import GatedAdapter


def prepare_model_for_adapter_training(model):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze adapters
    for module in model.modules():
        if isinstance(module, GatedAdapter):
            if hasattr(module, "adapter") and module.adapter is not None:
                for p in module.adapter.parameters():
                    p.requires_grad = True


def prepare_model_for_gate_training(model, gate_controller):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    if gate_controller is None:
        raise ValueError("gate_controller is required for gate training.")

    gate_controller.requires_grad_(True)


def get_adapter_optimizer(model, lr, weight_decay=0.01, betas=(0.9, 0.999)):
    adapter_params = []
    for module in model.modules():
        if isinstance(module, GatedAdapter):
            if hasattr(module, "adapter") and module.adapter is not None:
                for p in module.adapter.parameters():
                    if p.requires_grad:
                        adapter_params.append(p)
    return AdamW(
        [
            {
                "params": adapter_params,
                "weight_decay": weight_decay,
                "lr": lr,
                "betas": betas,
            }
        ],
    )


def get_gate_optimizer(gate_controller, lr, weight_decay=0.01, betas=(0.9, 0.999)):
    params = [p for p in gate_controller.parameters() if p.requires_grad]
    return AdamW(
        [{"params": params, "weight_decay": weight_decay, "lr": lr, "betas": betas}],
    )


def get_scheduler(optimizer, warmup_steps, total_steps):
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
