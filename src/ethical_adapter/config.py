from dataclasses import dataclass
from typing import List

@dataclass
class AdapterConfig:
    rank: int = 8              # r in LoRA
    alpha: float = 16.0        # scale of the low-rank update
    dropout: float = 0.0       # optional dropout on adapter path
    target_modules: List[str] = None  # modules to apply adapters to

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = []
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
