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

def project_delta_w(dW: torch.Tensor, proj_entry: dict):
    """
    Coordinate projector using stored top-m indices.
    """

    idx = proj_entry["idx"].to(dW.device)

    flat = dW.flatten()

    flat_A = torch.zeros_like(flat)
    flat_A[idx] = flat[idx]

    flat_T = flat - flat_A

    dW_A = flat_A.view_as(dW)
    dW_T = flat_T.view_as(dW)

    return dW_A, dW_T



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
        proj_entry = fisher_dict[key]  # <-- fisher entry is a dict now

        # projection into alignment/task subspaces
        dW_A, dW_T = project_delta_w(dW, proj_entry)

        # diagonal Fisher weights (optional)
        Fdiag = proj_entry.get("Fdiag", None)
        if Fdiag is not None:
            Fdiag = Fdiag.to(device=dW.device, dtype=dW.dtype)
        else:
            Fdiag = torch.ones_like(dW)


        # (1) Alignment preservation: Fisher-weighted norm on alignment component
        loss_align = loss_align + torch.sum(Fdiag * (dW_A ** 2))


        # (2) Task stability: optional curvature-weighted norm on task component
        if task_curv_dict is not None and (key in task_curv_dict) and (lambda_task > 0):
            Hdiag = task_curv_dict[key].to(device=dW.device, dtype=dW.dtype)
            loss_task = loss_task + torch.sum(Hdiag * dW_T * dW_T)

        # (3) Riemannian collision: coordinate overlap
        # smooth overlap weighting η
        beta = 10.0
        tau = 0.01

        mag = torch.abs(dW_A + dW_T)
        eta = 1.0 + beta * torch.sigmoid(mag - tau)

        loss_riem = loss_riem + torch.sum(eta * dW_A * dW_T)


        # (4) Geodesic collision: squared cosine similarity
        dot = torch.sum(Fdiag * dW_A * dW_T)

        normA = torch.sqrt(torch.sum(Fdiag * (dW_A ** 2)) + 1e-8)
        normT = torch.sqrt(torch.sum(Fdiag * (dW_T ** 2)) + 1e-8)

        cos2 = (dot / (normA * normT + 1e-8)) ** 2
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



def build_alignment_mask(Fdiag: torch.Tensor, frac: float = 0.1) -> torch.Tensor:
    """
    Builds top-fraction Fisher mask.
    Keeps highest Fisher entries as alignment-critical.
    """
    flat = Fdiag.flatten()
    k = max(1, int(frac * flat.numel()))

    topk_vals = torch.topk(flat, k, largest=True).values
    threshold = topk_vals.min()

    mask = (Fdiag >= threshold).to(Fdiag.dtype)
    return mask

@torch.no_grad()
def update_task_curvature(model, task_curv: dict, beta: float):
    """
    EMA of squared gradients mapped into ΔW space.
    """
    for name, pl in iter_parallel_linear(model):
        key = f"{name}.adapter"
        if key not in task_curv:
            continue

        A = pl.adapter.A.weight
        B = pl.adapter.B.weight

        if A.grad is None or B.grad is None:
            continue

        gW = pl.adapter.scaling * (
            B.grad @ A.detach() + B.detach() @ A.grad
        )

        gW = gW.detach().float().cpu()

        task_curv[key].mul_(beta).add_((1.0 - beta) * (gW ** 2))
