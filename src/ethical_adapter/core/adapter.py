# src/ethical_adapter/core/adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRAAdapter(nn.Module):
    """
    A minimal low-rank adapter with effective update:
        ΔW = scaling * (B @ A)
    where:
        A.weight: (rank, in_features)
        B.weight: (out_features, rank)

    Forward is implemented directly through ΔW so we can retain the exact
    dL/d(ΔW) used in the model graph.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # A: in -> r, B: r -> out
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # LoRA initialization: very small A, zero B so we start as a near no-op.
        nn.init.normal_(self.A.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.B.weight)

    def delta_weight(self) -> torch.Tensor:
        """
        Returns the effective LoRA weight update ΔW with shape:
            (out_features, in_features)
        """
        return self.scaling * (self.B.weight @ self.A.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_drop = self.dropout(x)
        dW = self.delta_weight()
        return F.linear(x_drop, dW, bias=None)

class ParallelLinear(nn.Module):
    """
    Wraps a frozen nn.Linear (the base map) with a parallel LoRA adapter.
    Forward: base(x) + gate * adapter(x), where gate is scalar (default 1).
    """

    debug_counter = 0
    DEBUG_EVERY = 1500

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        gate_index: int = None,
        gate_store: Optional[dict] = None,
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("ParallelLinear expects an nn.Linear as base_linear")

        # Copy the base linear so parameters/shape persist
        self.base = base_linear
        # Freeze base weights (we're steering, not fine-tuning)
        for p in self.base.parameters():
            p.requires_grad = False

        self.adapter = LoRAAdapter(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        adapter_dtype = self.base.weight.dtype
        self.adapter.A.weight.data = self.adapter.A.weight.data.to(adapter_dtype)
        self.adapter.B.weight.data = self.adapter.B.weight.data.to(adapter_dtype)

        self.gate_index = gate_index
        self.gate_store = gate_store

        self.force_gate_open = False
        self.force_gate_closed = False

        # AlignGuard hooks
        self.capture_delta_grad = False
        self._last_delta_w = None

    @property
    def in_features(self):  # convenience for debugging
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def _resolve_gate(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        if self.force_gate_open:
            return torch.ones(batch_size, device=x.device, dtype=x.dtype)
        if self.force_gate_closed:
            return torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        # Get gate logits from gate_store
        if self.gate_store is None:
            return torch.ones(batch_size, device=x.device, dtype=x.dtype)

        logits = self.gate_store.get("logits", None)
        if logits is None:
            return torch.ones(batch_size, device=x.device, dtype=x.dtype)
        

        assert self.gate_index is not None, "gate_index not set!"
        assert self.gate_index < logits.size(1), f"gate_index {self.gate_index} >= num_gates {logits.size(1)}"

        # Select this adapter’s gate index
        gate_logits = logits[:, self.gate_index]  # shape (B,)
        return gate_logits.to(x.device, x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # resolve gate logits for this layer

        gate_logits = self._resolve_gate(x)  # shape (batch,)
        gate_values = torch.sigmoid(gate_logits)

        # Expand to (batch, 1, 1, ...) to match adapter_out dims
        gate_values_expanded = gate_values.view(-1, *([1] * (x.dim() - 1)))
        base_out = self.base(x)

        # Compute the exact ΔW used in forward and optionally retain its grad
        x_drop = self.adapter.dropout(x)
        delta_w = self.adapter.delta_weight()
        
        if self.capture_delta_grad and delta_w.requires_grad:
            self._last_delta_w = delta_w
            self._last_delta_w.retain_grad()
        else:
            self._last_delta_w = None

        adapter_raw = F.linear(x_drop, delta_w, bias=None)
        adapter_out = gate_values_expanded * adapter_raw

        self.last_gate_logits = gate_logits.detach().float()
        self.last_gate_values = gate_values.detach().float()

        # debug print
        ParallelLinear.debug_counter += 1
        if ParallelLinear.debug_counter % ParallelLinear.DEBUG_EVERY == 0:
            with torch.no_grad():
                print("\n[ParallelLinear DEBUG] step:", ParallelLinear.debug_counter)
                print(
                    "  base_out:     std={:.4f}  mean={:.4f}  max={:.4f}".format(
                        base_out.std().item(),
                        base_out.mean().item(),
                        base_out.abs().max().item(),
                    )
                )
                print(
                    "  adapter_raw:  std={:.4f}  mean={:.4f}  max={:.4f}".format(
                        adapter_raw.std().item(),
                        adapter_raw.mean().item(),
                        adapter_raw.abs().max().item(),
                    )
                )
                print(
                    "  adapter_out:  std={:.4f}  (raw * gate)".format(
                        adapter_out.std().item()
                    )
                )
                print(
                    "  gate:         mean={:.4f}  min={:.4f}  max={:.4f}".format(
                        gate_values.mean().item(),
                        gate_values.min().item(),
                        gate_values.max().item(),
                    )
                )
                print(
                    "  ratio (adapter/base std): {:.4f}".format(
                        adapter_raw.std().item() / (base_out.std().item() + 1e-8)
                    )
                )
                print("-----------------------------------------")

        return base_out + adapter_out
