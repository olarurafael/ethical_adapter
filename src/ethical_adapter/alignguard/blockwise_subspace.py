# src/ethical_adapter/alignguard/blockwise_subspace.py
import math
import torch


def split_ranges(total_dim: int, block_size: int):
    ranges = []
    start = 0
    while start < total_dim:
        end = min(start + block_size, total_dim)
        ranges.append((start, end))
        start = end
    return ranges


class OjaBlock:
    """
    Streaming top-subspace estimator for one block of a flattened ΔW vector.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        eta0: float = 0.5,
        orth_every: int = 10,
        init_samples: int | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = dim
        self.rank = min(rank, dim)
        self.eta0 = eta0
        self.orth_every = orth_every
        self.init_samples = init_samples or self.rank
        self.device = device
        self.dtype = dtype

        self.U = None
        self.evals = None
        self.num_updates = 0
        self._buffer = []

    def _maybe_init(self):
        if self.U is not None:
            return
        if len(self._buffer) < self.init_samples:
            return

        G = torch.stack(self._buffer, dim=1)  # (dim, k)
        Q, _ = torch.linalg.qr(G, mode="reduced")

        if Q.size(1) < self.rank:
            pad = self.rank - Q.size(1)
            extra = torch.randn(self.dim, pad, dtype=self.dtype, device=self.device)
            if Q.numel() > 0:
                extra = extra - Q @ (Q.T @ extra)
            extra, _ = torch.linalg.qr(extra, mode="reduced")
            Q = torch.cat([Q, extra[:, :pad]], dim=1)

        self.U = Q[:, :self.rank].contiguous()
        self.evals = torch.zeros(self.rank, dtype=self.dtype, device=self.device)
        self._buffer = []

    @torch.no_grad()
    def update(self, g_block: torch.Tensor):
        g = g_block.to(device=self.device, dtype=self.dtype).flatten()
        gnorm = torch.norm(g)
        if not torch.isfinite(gnorm) or gnorm.item() == 0.0:
            return

        if self.U is None:
            self._buffer.append(g / (gnorm + 1e-12))
            self._maybe_init()
            return

        self.num_updates += 1
        eta = self.eta0 / math.sqrt(max(self.num_updates, 1))

        y = self.U.T @ g
        resid = g - self.U @ y

        self.U.add_(eta * resid.unsqueeze(1) * y.unsqueeze(0))

        if self.num_updates % self.orth_every == 0:
            self.U, _ = torch.linalg.qr(self.U, mode="reduced")

        n = float(self.num_updates)
        self.evals.mul_((n - 1.0) / n).add_(y.pow(2) / n)

    @torch.no_grad()
    def finalize(self):
        if self.U is None:
            raise RuntimeError("Block estimator never initialized; increase init_samples or minibatches.")
        self.U, _ = torch.linalg.qr(self.U, mode="reduced")
        evals = torch.clamp(self.evals, min=0.0)
        return {
            "U": self.U.cpu().contiguous(),
            "evals": evals.cpu().contiguous(),
        }


class BlockwiseOjaEstimator:
    """
    Blockwise subspace estimator for vec(ΔW).
    """

    def __init__(
        self,
        total_dim: int,
        block_size: int,
        rank_per_block: int,
        eta0: float = 0.5,
        orth_every: int = 10,
        init_samples: int | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.total_dim = total_dim
        self.block_size = block_size
        self.rank_per_block = rank_per_block
        self.ranges = split_ranges(total_dim, block_size)

        self.blocks = []
        for s, e in self.ranges:
            self.blocks.append(
                OjaBlock(
                    dim=e - s,
                    rank=rank_per_block,
                    eta0=eta0,
                    orth_every=orth_every,
                    init_samples=init_samples,
                    device=device,
                    dtype=dtype,
                )
            )

    @torch.no_grad()
    def update(self, g_flat: torch.Tensor):
        g_flat = g_flat.flatten()
        for (s, e), blk in zip(self.ranges, self.blocks):
            blk.update(g_flat[s:e])

    @torch.no_grad()
    def finalize(self):
        return {
            "type": "block_subspace",
            "total_dim": self.total_dim,
            "block_size": self.block_size,
            "ranges": self.ranges,
            "blocks": [blk.finalize() for blk in self.blocks],
        }