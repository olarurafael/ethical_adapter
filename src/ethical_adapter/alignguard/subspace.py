# src/ethical_adapter/alignguard/subspace.py
import torch


class OjaSubspaceEstimator:
    """
    Streaming estimator for the top Fisher subspace in ΔW-space.

    Tracks a basis U in R^{d x m} using Oja-style updates on gradient samples g_t,
    where empirical Fisher is approximated by E[g_t g_t^T].

    This is a practical low-rank approximation to the paper's Fisher eigenspace
    without forming the full d x d matrix.
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
        self.rank = rank
        self.eta0 = eta0
        self.orth_every = orth_every
        self.init_samples = init_samples or rank
        self.device = device
        self.dtype = dtype

        self.U = None                  # (d, m)
        self.evals = None              # (m,)
        self.num_updates = 0
        self._buffer = []

    def _maybe_init(self):
        if self.U is not None:
            return

        if len(self._buffer) < self.init_samples:
            return

        G = torch.stack(self._buffer, dim=1)  # (d, k)
        Q, _ = torch.linalg.qr(G, mode="reduced")

        if Q.size(1) < self.rank:
            pad = self.rank - Q.size(1)
            extra = torch.randn(self.dim, pad, dtype=self.dtype, device=self.device)
            extra = extra - Q @ (Q.T @ extra)
            extra, _ = torch.linalg.qr(extra, mode="reduced")
            Q = torch.cat([Q, extra[:, :pad]], dim=1)

        self.U = Q[:, : self.rank].contiguous()
        self.evals = torch.zeros(self.rank, dtype=self.dtype, device=self.device)
        self._buffer = []

    @torch.no_grad()
    def update(self, g: torch.Tensor):
        """
        g: flattened gradient vector in R^d (CPU float32 recommended)
        """
        g = g.to(device=self.device, dtype=self.dtype).flatten()
        gnorm = torch.norm(g)
        if not torch.isfinite(gnorm) or gnorm.item() == 0.0:
            return

        if self.U is None:
            self._buffer.append(g / (gnorm + 1e-12))
            self._maybe_init()
            return

        self.num_updates += 1
        eta = self.eta0 / (self.num_updates ** 0.5)

        y = self.U.T @ g                       # (m,)
        resid = g - self.U @ y                # (d,)

        # Oja update
        self.U.add_(eta * resid.unsqueeze(1) * y.unsqueeze(0))

        # Periodic re-orthonormalization
        if self.num_updates % self.orth_every == 0:
            self.U, _ = torch.linalg.qr(self.U, mode="reduced")

        # Running estimate of subspace eigenvalues / captured Fisher energy
        n = float(self.num_updates)
        self.evals.mul_((n - 1.0) / n).add_((y.pow(2)) / n)

    @torch.no_grad()
    def finalize(self):
        if self.U is None:
            raise RuntimeError("Subspace estimator never initialized; not enough samples.")

        self.U, _ = torch.linalg.qr(self.U, mode="reduced")
        evals = torch.clamp(self.evals, min=0.0)

        return {
            "type": "subspace",
            "U": self.U.cpu().contiguous(),        # (d, m)
            "evals": evals.cpu().contiguous(),     # (m,)
        }