# src/ethical_adapter/core/adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional


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
    Forward: base(x) + gate * adapter(x), where gate is one scalar per example.

    adapter_mode controls the adapter path explicitly:
      - "on":   adapter is fully applied (gate = 1.0)
      - "off":  adapter is disabled (gate = 0.0)
      - "gate": adapter is controlled by gate_store["logits"]
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        gate_store: Optional[dict] = None,
        gate_hard_threshold: Optional[float] = None,
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("ParallelLinear expects an nn.Linear as base_linear")
        if gate_hard_threshold is not None and not (0.0 < gate_hard_threshold < 1.0):
            raise ValueError("gate_hard_threshold must be in (0, 1)")

        self.base = base_linear
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

        self.gate_store = gate_store
        self.gate_hard_threshold = gate_hard_threshold
        self.adapter_mode: Literal["on", "off", "gate"] = (
            "gate" if gate_store is not None else "on"
        )

        self.capture_delta_grad = False
        self._last_delta_w = None

    @property
    def in_features(self):  # convenience for debugging
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def set_adapter_mode(self, mode: Literal["on", "off", "gate"]) -> None:
        if mode not in {"on", "off", "gate"}:
            raise ValueError("adapter mode must be one of: on, off, gate")
        if mode == "gate" and self.gate_store is None:
            raise ValueError("adapter mode 'gate' requires a gate_store")
        self.adapter_mode = mode

    def _resolve_gate_values(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.size(0)

        if self.adapter_mode == "on":
            return torch.ones(batch_size, device=x.device, dtype=x.dtype), None
        if self.adapter_mode == "off":
            return torch.zeros(batch_size, device=x.device, dtype=x.dtype), None

        if self.gate_store is None:
            raise RuntimeError("adapter mode 'gate' requires a gate_store")

        logits = self.gate_store.get("logits", None)
        if logits is None:
            raise RuntimeError("adapter mode 'gate' requires gate_store['logits']")

        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if logits.dim() != 1:
            raise ValueError(
                "gate_store['logits'] must have shape (batch,) or (batch, 1); "
                f"got {tuple(logits.shape)}"
            )
        if logits.size(0) != batch_size:
            raise ValueError(
                "gate logits batch size does not match adapter input batch size: "
                f"{logits.size(0)} != {batch_size}"
            )

        gate_logits = logits.to(device=x.device, dtype=x.dtype)
        gate_values = torch.sigmoid(gate_logits)
        if self.gate_hard_threshold is not None:
            gate_values = (gate_values >= self.gate_hard_threshold).to(dtype=x.dtype)
        return gate_values, gate_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_values, gate_logits = self._resolve_gate_values(x)

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

        self.last_gate_logits = (
            None if gate_logits is None else gate_logits.detach().float()
        )
        self.last_gate_values = gate_values.detach().float()

        return base_out + adapter_out
