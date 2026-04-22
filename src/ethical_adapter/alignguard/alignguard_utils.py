# src/ethical_adapter/training/alignguard_utils.py
import torch
from ethical_adapter.core.adapter import ParallelLinear


def iter_parallel_linear(model):
    for name, module in model.named_modules():
        if isinstance(module, ParallelLinear):
            yield name, module


def compute_delta_w(pl: ParallelLinear) -> torch.Tensor:
    return pl.adapter.delta_weight()


def set_capture_delta_grad(model, enabled: bool):
    for _, pl in iter_parallel_linear(model):
        pl.capture_delta_grad = enabled
        if not enabled:
            pl._last_delta_w = None


def get_last_delta_grad(pl: ParallelLinear) -> torch.Tensor | None:
    dW = getattr(pl, "_last_delta_w", None)
    if dW is None:
        return None
    return dW.grad


def project_delta_w(dW: torch.Tensor, proj_entry: dict):
    """
    Blockwise projector:
      vec(ΔW)_A = concat_b U_b (U_b^T vec(ΔW)_b)
      vec(ΔW)_T = vec(ΔW) - vec(ΔW)_A
    """
    flat = dW.flatten()
    if proj_entry.get("type") != "block_subspace":
        raise ValueError(f"Expected block_subspace projector, got {proj_entry.get('type')}")

    flat_A = torch.zeros_like(flat)

    for (s, e), blk in zip(proj_entry["ranges"], proj_entry["blocks"]):
        U = blk["U"].to(device=dW.device, dtype=dW.dtype)          # (block_dim, r)
        x = flat[s:e]
        flat_A[s:e] = U @ (U.T @ x)

    flat_T = flat - flat_A
    return flat_A.view_as(dW), flat_T.view_as(dW)


def fisher_quadratic_from_block_subspace(dW: torch.Tensor, proj_entry: dict) -> torch.Tensor:
    """
    Approximate vec(dW)^T F vec(dW) with blockwise low-rank Fisher:
      sum_b x_b^T U_b diag(evals_b) U_b^T x_b
    """
    flat = dW.flatten()
    out = torch.zeros((), device=dW.device, dtype=dW.dtype)

    for (s, e), blk in zip(proj_entry["ranges"], proj_entry["blocks"]):
        U = blk["U"].to(device=dW.device, dtype=dW.dtype)
        evals = blk["evals"].to(device=dW.device, dtype=dW.dtype)
        coeff = U.T @ flat[s:e]
        out = out + torch.sum(evals * coeff.pow(2))

    return out


def alignguard_loss(
    model,
    fisher_dict: dict,
    task_curv_dict: dict | None,
    lambda_align: float,
    lambda_task: float,
    lambda_riem: float,
    lambda_geo: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    loss_align = torch.zeros((), device=device, dtype=dtype)
    loss_task = torch.zeros((), device=device, dtype=dtype)
    loss_riem = torch.zeros((), device=device, dtype=dtype)
    loss_geo = torch.zeros((), device=device, dtype=dtype)

    beta = 5.0
    tau = 0.01
    eps = 1e-8

    for name, pl in iter_parallel_linear(model):
        key = f"{name}.adapter"
        if key not in fisher_dict:
            continue

        dW = compute_delta_w(pl)
        proj_entry = fisher_dict[key]

        dW_A, dW_T = project_delta_w(dW, proj_entry)

        loss_align = loss_align + fisher_quadratic_from_block_subspace(dW_A, proj_entry)

        if lambda_task > 0:
            if task_curv_dict is not None and key in task_curv_dict:
                Hdiag = task_curv_dict[key].to(device=dW.device, dtype=dW.dtype)
            else:
                Hdiag = torch.ones_like(dW_T)
            loss_task = loss_task + torch.sum(Hdiag * dW_T.pow(2))

        mag = torch.abs(dW_A + dW_T)
        eta = 1.0 + beta * torch.sigmoid(mag - tau)
        loss_riem = loss_riem + torch.sum(eta * dW_A * dW_T)

        dot = torch.sum(dW_A * dW_T)
        normA = torch.sqrt(torch.sum(dW_A.pow(2)) + eps)
        normT = torch.sqrt(torch.sum(dW_T.pow(2)) + eps)
        loss_geo = loss_geo + (dot / (normA * normT + eps)).pow(2)

    return (
        lambda_align * loss_align
        + lambda_task * loss_task
        + lambda_riem * loss_riem
        + lambda_geo * loss_geo
    )


@torch.no_grad()
def init_task_curvature_identity(model) -> dict:
    curv = {}
    for name, pl in iter_parallel_linear(model):
        key = f"{name}.adapter"
        dW = compute_delta_w(pl)
        curv[key] = torch.ones_like(dW, device="cpu", dtype=torch.float32)
    return curv