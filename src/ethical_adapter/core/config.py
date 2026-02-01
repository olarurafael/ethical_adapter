# src/ethical_adapter/config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class GateConfig:
    enabled: bool = False
    source_module: str = ""
    hidden_size: int = 256
    activation: str = "gelu"
    num_gates: int = 1
    temperature: float = 1.0
    dropout: float = 0.0
    pooling: str = "mean"  # how to reduce (batch, seq, dim) -> (batch, dim)

    def __post_init__(self):
        if self.enabled and not self.source_module:
            raise ValueError("source_module must be set when gate.enabled=True")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.num_gates <= 0:
            raise ValueError("num_gates must be > 0")
        if self.activation not in {"relu", "gelu", "silu", "none"}:
            raise ValueError("activation must be one of: relu, gelu, silu")
        if self.pooling not in {"mean", "cls", "max", "logsumexp"}:
            raise ValueError("pooling must be 'mean' or 'cls'")


@dataclass
class AdapterConfig:
    rank: int = 8  # r in LoRA
    alpha: float = 16.0  # scale of the low-rank update
    dropout: float = 0.0  # optional dropout on adapter path
    target_modules: List[str] = None  # modules to apply adapters to
    gate: GateConfig = field(default_factory=GateConfig)

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = []
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
