# src/ethical_adapter/training/alignguard_utils.py
import torch
from ethical_adapter.core.adapter import ParallelLinear


def iter_parallel_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, ParallelLinear):
            yield name, module


def compute_delta_w(pl: ParallelLinear) -> torch.Tensor:
    """
    ΔW = scaling * (B @ A)
    A.weight: (r, in)
    B.weight: (out, r)
    returns: (out, in)
    """
    A = pl.adapter.A.weight
    B = pl.adapter.B.weight
    return pl.adapter.scaling * (B @ A)


def alignguard_loss(
    model,
    fisher_dict: dict,
    task_curv_dict: dict | None,
    lambda_align: float,
    lambda_task: float,
    lambda_riem: float,
    lambda_geo: float,
) -> torch.Tensor:
    """
    Computes AlignGuard-style loss terms over ParallelLinear modules.
    fisher_dict: key -> tensor (out, in)  (alignment Fisher diag mask/weights)
    task_curv_dict: key -> tensor (out, in) (EMA proxy; optional)
    Keys are f"{module_name}.adapter"
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    loss_align = torch.zeros((), device=device, dtype=dtype)
    loss_task = torch.zeros((), device=device, dtype=dtype)
    loss_riem = torch.zeros((), device=device, dtype=dtype)
    loss_geo = torch.zeros((), device=device, dtype=dtype)

    for name, pl in iter_parallel_linear(model):
        key = f"{name}.adapter"
        if key not in fisher_dict:
            continue

        dW = compute_delta_w(pl)  # (out, in)
        Fdiag = fisher_dict[key].to(device=dW.device, dtype=dW.dtype)

        # Alignment subspace projector (coordinate-mask approximation)
        M = (Fdiag > 0).to(dW.dtype)

        dW_A = M * dW
        dW_T = (1.0 - M) * dW

        # (1) Alignment preservation: Fisher-weighted norm on alignment component
        loss_align = loss_align + torch.sum(Fdiag * dW_A * dW_A)

        # (2) Task stability: optional curvature-weighted norm on task component
        if task_curv_dict is not None and (key in task_curv_dict) and (lambda_task > 0):
            Hdiag = task_curv_dict[key].to(device=dW.device, dtype=dW.dtype)
            loss_task = loss_task + torch.sum(Hdiag * dW_T * dW_T)

        # (3) Riemannian collision: coordinate overlap
        loss_riem = loss_riem + torch.sum(torch.abs(dW_A * dW_T))

        # (4) Geodesic collision: squared cosine similarity
        a = dW_A.flatten()
        t = dW_T.flatten()
        denom = (a.norm() * t.norm() + 1e-8)
        cos2 = (torch.dot(a, t) / denom) ** 2
        loss_geo = loss_geo + cos2

    return (
        lambda_align * loss_align
        + lambda_task * loss_task
        + lambda_riem * loss_riem
        + lambda_geo * loss_geo
    )


@torch.no_grad()
def init_task_curvature(model) -> dict:
    """
    Initializes task curvature buffers (EMA proxy) per ParallelLinear as ΔW-shaped zeros on CPU.
    """
    curv = {}
    for name, pl in iter_parallel_linear(model):
        key = f"{name}.adapter"
        dW = compute_delta_w(pl)
        curv[key] = torch.zeros_like(dW, device="cpu", dtype=torch.float32)
    return curv


@torch.no_grad()
def update_task_curvature(model, task_curv: dict, beta: float):
    """
    Updates EMA curvature proxy using current ΔW magnitude: curv <- beta*curv + (1-beta)*(ΔW^2)
    Stored on CPU to save VRAM.
    """
    for name, pl in iter_parallel_linear(model):
        key = f"{name}.adapter"
        if key not in task_curv:
            continue
        dW = compute_delta_w(pl).detach().float().cpu()
        task_curv[key].mul_(beta).add_((1.0 - beta) * (dW * dW))
