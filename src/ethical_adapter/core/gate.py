# src/ethical_adapter/core/gate.py
import torch
import torch.nn as nn


class GateController(nn.Module):
    """
    Produces one scalar gate logit per example.

    The controller sees hidden states from a selected source module, pools over
    sequence positions, and predicts a logit with a linear readout. ParallelLinear
    converts the logit to a gate value with sigmoid, or to a hard 0/1 gate when
    configured.
    """

    def __init__(
        self,
        input_dim: int,
        temperature: float = 1.0,
        dropout: float = 0.0,
        pooling: str = "mean",  # how to collapse (batch, seq, dim) -> (batch, dim)
    ):
        super().__init__()
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if pooling not in {"mean", "cls", "max", "logsumexp"}:
            raise ValueError("pooling must be one of: mean, cls, max, logsumexp")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature
        self.pooling = pooling
        self.last_gate_logits = None

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, input_dim)

        Returns:
            gate logits: (batch,)
        """
        if hidden_states.dim() == 2:
            pooled = hidden_states
        elif self.pooling == "mean":
            pooled = hidden_states.mean(dim=1)
        elif self.pooling == "max":
            pooled = hidden_states.max(dim=1).values
        elif self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "logsumexp":
            # soft-max pooling, more sensitive than mean, less brittle than max
            pooled = torch.logsumexp(hidden_states * 5.0, dim=1) / 5.0
        else:
            raise RuntimeError(f"Unsupported pooling mode: {self.pooling}")

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(-1) / self.temperature

        self.last_gate_logits = logits

        return logits


class GatedSourceWrapper(nn.Module):
    """
    Wraps a chosen module so it:
      1. runs the original computation
      2. computes one scalar gate logit via GateController
      3. stores those gates for downstream adapters
    """

    def __init__(
        self,
        base_module: nn.Module,
        gate_controller: GateController,
        gate_store: dict,
        store_key: str,
    ):
        super().__init__()
        self.base = base_module
        self.gate_controller = gate_controller
        self.gate_store = gate_store
        self.store_key = store_key

    def __getattr__(self, name):
        # Preserve the wrapped module's interface so transformer internals
        # can still access attributes such as attention_type.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)

    def forward(self, *args, **kwargs):
        # Run the original module first
        output = self.base(*args, **kwargs)

        # Determine hidden states to feed the gate controller
        if isinstance(output, torch.Tensor):
            hidden = output
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            hidden = output[0]
        else:
            raise TypeError("GatedSourceWrapper expects tensor or tuple output.")


        logits = self.gate_controller(hidden)
        self.gate_store[self.store_key] = logits

        # Return the ORIGINAL module output (not the logits!)
        return output
