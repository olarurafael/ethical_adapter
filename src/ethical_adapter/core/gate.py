# src/ethical_adapter/core/gate.py
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "none": lambda x: x,
}


class GateController(nn.Module):
    """
    Produces a fixed number of scalar gates (e.g., one per adapter) in [0,1].
    Feed it a pooled hidden representation and it returns shape (batch, num_gates).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_gates: int = 1,
        activation: str = "silu",
        temperature: float = 1.0,
        pooling: str = "mean",  # how to collapse (batch, seq, dim) -> (batch, dim)
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(ACTIVATIONS.keys())}."
            )
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_gates <= 0:
            raise ValueError("num_gates must be > 0")
        if pooling not in {"mean", "cls", "max", 'none', "logsumexp"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        # self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        # self.fc2 = nn.Linear(hidden_dim, num_gates, bias=False)
        self.fc = nn.Linear(input_dim,num_gates,bias=True)
        self.act_name = activation
        self.temperature = temperature
        self.pooling = pooling
        self.last_gates = None
        # self.norm = nn.LayerNorm(input_dim)
        self.norm = nn.LayerNorm(input_dim)
        # self.dropout = nn.Dropout(0.1)

        #nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.fc.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, input_dim)

        Returns:
            gates: (batch, num_gates) in (0,1)
        """
        if self.pooling == "mean":
            pooled = hidden_states.mean(dim=1)
        elif self.pooling == "max":
            pooled = hidden_states.max(dim=1).values
        elif self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "logsumexp":
            # soft-max pooling, more sensitive than mean, less brittle than max
            pooled = torch.logsumexp(hidden_states * 5.0, dim=1) / 5.0

        elif self.pooling == "topk":
            k = min(4, hidden_states.size(1))
            pooled = hidden_states.topk(k, dim=1).values.mean(dim=1)
        else:  # "none"
            pooled = hidden_states  # (batch, seq, dim)

        pooled = self.norm(pooled)

        # h = self.fc1(pooled)
        # h = ACTIVATIONS[self.act_name](h)
        # # h = self.dropout(h)

        # logits = self.fc2(h) / self.temperature
        logits = self.fc(pooled) / self.temperature

        self.last_gate_logits = logits

        return logits


class GatedSourceWrapper(nn.Module):
    """
    Wraps a chosen module so it:
      1. runs the original computation
      2. computes N scalar gates via GateController
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


        # Compute gate logits
        logits = self.gate_controller(hidden)

        # Store them for downstream adapters
        self.gate_store["logits"] = logits

        # Return the ORIGINAL module output (not the logits!)
        return output
