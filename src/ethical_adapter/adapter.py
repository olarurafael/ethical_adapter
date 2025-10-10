import torch
import torch.nn as nn
import math

class LoRAAdapter(nn.Module):
    """
    A minimal low-rank adapter: ΔW = B @ A.
    Implemented as two linear layers with no bias: A: in->r, B: r->out.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # A: in -> r, B: r -> out
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # LoRA initialization: small A, zero B so we start as a no-op.
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional dropout on the adapter path to regularize
        z = self.A(self.dropout(x))
        z = self.B(z)
        return z * self.scaling

class ParallelLinear(nn.Module):
    """
    Wraps a frozen nn.Linear (the base map) with a parallel LoRA adapter.
    Forward: base(x) + adapter(x)
    """
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
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
            dropout=dropout
        )

    @property
    def in_features(self):  # convenience for debugging
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter(x)
