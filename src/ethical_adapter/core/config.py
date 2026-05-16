from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateConfig:
    enabled: bool = False
    source_module: str = ""
    temperature: float = 1.0
    dropout: float = 0.0
    pooling: str = "mean"  # how to reduce (batch, seq, dim) -> (batch, dim)
    hard_threshold: float | None = None

    def __post_init__(self):
        if self.enabled and not self.source_module:
            raise ValueError("source_module must be set when gate.enabled=True")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.pooling not in {"mean", "cls", "max", "logsumexp"}:
            raise ValueError("pooling must be one of: mean, cls, max, logsumexp")
        if self.hard_threshold is not None and not (0.0 < self.hard_threshold < 1.0):
            raise ValueError("hard_threshold must be in (0, 1)")


@dataclass
class AdapterConfig:
    rank: int = 8  # r in LoRA
    alpha: float = 16.0  # scale of the low-rank update
    dropout: float = 0.0  # optional dropout on adapter path
    target_modules: list[str] = field(default_factory=list)
    gate: GateConfig = field(default_factory=GateConfig)

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
