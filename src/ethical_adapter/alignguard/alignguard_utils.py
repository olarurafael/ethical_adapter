import torch
from ethical_adapter.core.adapter import GatedAdapter


def iter_gated_adapters(model):
    for name, module in model.named_modules():
        if isinstance(module, GatedAdapter):
            yield name, module


def compute_delta_w(adapter_module: GatedAdapter) -> torch.Tensor:
    return adapter_module.adapter.delta_weight()


def set_capture_delta_grad(model, enabled: bool):
    for _, adapter_module in iter_gated_adapters(model):
        adapter_module.capture_delta_grad = enabled
        if not enabled:
            adapter_module._last_delta_w = None


def get_last_delta_grad(adapter_module: GatedAdapter) -> torch.Tensor | None:
    delta_weight = getattr(adapter_module, "_last_delta_w", None)
    if delta_weight is None:
        return None
    return delta_weight.grad


def project_delta_w(delta_weight: torch.Tensor, proj_entry: dict):
    """
    Blockwise projector:
      vec(ΔW)_A = concat_b U_b (U_b^T vec(ΔW)_b)
      vec(ΔW)_T = vec(ΔW) - vec(ΔW)_A
    """
    flat = delta_weight.flatten()
    if proj_entry.get("type") != "block_subspace":
        raise ValueError(
            f"Expected block_subspace projector, got {proj_entry.get('type')}"
        )

    flat_A = torch.zeros_like(flat)

    for (s, e), blk in zip(proj_entry["ranges"], proj_entry["blocks"]):
        U = blk["U"].to(device=delta_weight.device, dtype=delta_weight.dtype)
        x = flat[s:e]
        flat_A[s:e] = U @ (U.T @ x)

    flat_T = flat - flat_A
    return flat_A.view_as(delta_weight), flat_T.view_as(delta_weight)


def fisher_quadratic_from_block_subspace(
    delta_weight: torch.Tensor,
    proj_entry: dict,
) -> torch.Tensor:
    """
    Approximate vec(dW)^T F vec(dW) with blockwise low-rank Fisher:
      sum_b x_b^T U_b diag(evals_b) U_b^T x_b
    """
    flat = delta_weight.flatten()
    out = torch.zeros((), device=delta_weight.device, dtype=delta_weight.dtype)

    for (s, e), blk in zip(proj_entry["ranges"], proj_entry["blocks"]):
        U = blk["U"].to(device=delta_weight.device, dtype=delta_weight.dtype)
        evals = blk["evals"].to(device=delta_weight.device, dtype=delta_weight.dtype)
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

    for name, adapter_module in iter_gated_adapters(model):
        key = f"{name}.adapter"
        if key not in fisher_dict:
            continue

        delta_weight = compute_delta_w(adapter_module)
        proj_entry = fisher_dict[key]

        alignment_component, task_component = project_delta_w(delta_weight, proj_entry)

        loss_align = loss_align + fisher_quadratic_from_block_subspace(
            alignment_component,
            proj_entry,
        )

        if lambda_task > 0:
            if task_curv_dict is not None and key in task_curv_dict:
                task_curvature_diag = task_curv_dict[key].to(
                    device=delta_weight.device,
                    dtype=delta_weight.dtype,
                )
            else:
                task_curvature_diag = torch.ones_like(task_component)
            loss_task = loss_task + torch.sum(
                task_curvature_diag * task_component.pow(2)
            )

        mag = torch.abs(alignment_component + task_component)
        eta = 1.0 + beta * torch.sigmoid(mag - tau)
        loss_riem = loss_riem + torch.sum(eta * alignment_component * task_component)

        dot = torch.sum(alignment_component * task_component)
        alignment_norm = torch.sqrt(torch.sum(alignment_component.pow(2)) + eps)
        task_norm = torch.sqrt(torch.sum(task_component.pow(2)) + eps)
        loss_geo = loss_geo + (dot / (alignment_norm * task_norm + eps)).pow(2)

    return (
        lambda_align * loss_align
        + lambda_task * loss_task
        + lambda_riem * loss_riem
        + lambda_geo * loss_geo
    )


@torch.no_grad()
def init_task_curvature_identity(model) -> dict:
    curv = {}
    for name, adapter_module in iter_gated_adapters(model):
        key = f"{name}.adapter"
        delta_weight = compute_delta_w(adapter_module)
        curv[key] = torch.ones_like(delta_weight, device="cpu", dtype=torch.float32)
    return curv
