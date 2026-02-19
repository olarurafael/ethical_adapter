# ethical_adapter/training/alignguard.py
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class AlignGuard:
    """
    Minimal AlignGuard-style gradient filter operating on trainable params only
    (i.e. your LoRA adapters + optionally gate).
    """

    def __init__(self, model, fisher_diag, topk=1024, retain_frac=0.0):
        self.model = model
        self.fisher_diag = fisher_diag
        self.topk = topk
        self.retain_frac = retain_frac  # 0.0 = fully protect alignment directions

        # cache trainable params
        self.params = [p for p in model.parameters() if p.requires_grad]

        # precompute top-k mask
        idx = torch.argsort(self.fisher_diag, descending=True)
        self.align_idx = idx[:topk]

    @torch.no_grad()
    def filter_grads(self):
        grads = []
        for p in self.params:
            if p.grad is None:
                grads.append(torch.zeros_like(p).flatten())
            else:
                grads.append(p.grad.flatten())

        grad_vec = torch.cat(grads)

        mask = torch.ones_like(grad_vec)
        mask[self.align_idx] = self.retain_frac  # suppress alignment-sensitive dims

        filtered = grad_vec * mask

        offset = 0
        for p in self.params:
            numel = p.numel()
            grad_slice = filtered[offset: offset + numel].view_as(p)

            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)

            offset += numel

