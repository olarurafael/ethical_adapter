# src/ethical_adapter/alignguard/compute_alignment_fisher.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from ethical_adapter.core.config import AdapterConfig, GateConfig
from ethical_adapter.core.inject import inject_adapters
from ethical_adapter.training.data import (
    build_phase_dataset,
    tokenize_dataset,
)
from ethical_adapter.core.adapter import ParallelLinear
from ethical_adapter.alignguard.alignguard_utils import (
    iter_parallel_linear,
    compute_delta_w,
)

# ----------------------------
# CONFIG (keep simple)
# ----------------------------

DEVICE = "cuda"
DTYPE = torch.bfloat16
BATCH_SIZE = 1              # keep 1 for stable Fisher
MAX_BATCHES = 1000          # set to e.g. 500 to cap runtime
OUT_PATH = "alignment_fisher.pt"


# ----------------------------
# MAIN
# ----------------------------

def main(cfg):
    torch.manual_seed(0)

    # ---- tokenizer ----
    tokenizer_name = cfg.get("tokenizer_name", cfg.get("local_path"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- model ----
    model_name = cfg.get("model_name", cfg.get("local_path"))
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto",
    )

    # ---- inject adapters (same as training) ----
    adapter_cfg = AdapterConfig(
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        gate=GateConfig(**cfg.get("gate", {})),
    )

    injected = inject_adapters(base_model, adapter_cfg)
    model = injected.model
    model.eval()

    # Force adapters ON, gates irrelevant here
    for m in model.modules():
        if isinstance(m, ParallelLinear):
            m.force_gate_open = True
            m.force_gate_closed = False

    model.to(DEVICE)

    # ---- alignment dataset ----
    logger = DummyLogger()
    align_ds = build_phase_dataset(cfg, logger=logger, phase="adapters")
    align_ds = tokenize_dataset(align_ds, tokenizer, cfg)

    loader = DataLoader(
        align_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---- init Fisher buffers (ΔW-shaped) ----
    fisher = {}
    for name, pl in iter_parallel_linear(model):
        key = f"{name}.adapter"
        dW = compute_delta_w(pl)
        fisher[key] = torch.zeros_like(dW, device="cpu", dtype=torch.float32)

    # ---- accumulate squared gradients ----
    step = 0
    for batch in loader:
        if MAX_BATCHES is not None and step >= MAX_BATCHES:
            break

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)

        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()

        # Accumulate Fisher in ΔW space
        for name, pl in iter_parallel_linear(model):
            key = f"{name}.adapter"

            # d(ΔW)/dθ via chain rule: we approximate Fisher on ΔW directly
            dW = compute_delta_w(pl)
            grad_sq = (dW.grad if dW.requires_grad else None)

            # ΔW.grad is not populated directly; instead use A/B grads
            A = pl.adapter.A.weight
            B = pl.adapter.B.weight

            if A.grad is None or B.grad is None:
                continue

            # Approximate Fisher in ΔW coordinates:
            # F ≈ E[ (∂L/∂ΔW)^2 ] ≈ outer(B.grad @ A + B @ A.grad)
            gW = pl.adapter.scaling * (
                B.grad @ A.detach() + B.detach() @ A.grad
            )

            fisher[key] += (gW.detach().float().cpu() ** 2)

        step += 1
        if step % 50 == 0:
            print(f"[Fisher] processed {step} batches")

    # ---- normalize ----
    for k in fisher:
        fisher[k] /= max(step, 1)


    # ---- build projector dictionaries ----
    proj = {}

    TOP_M = cfg.get("alignguard_topm", 1024)

    for key, Fdiag in fisher.items():

        flat = Fdiag.flatten()
        d = flat.numel()

        m = min(TOP_M, d)

        # indices of largest Fisher entries
        idx = torch.topk(flat, k=m, largest=True).indices.cpu()

        proj[key] = {
            "idx": idx,               # (m,)
            "shape": Fdiag.shape,     # (out, in)
            "Fdiag": Fdiag.clone()
        }

    torch.save(proj, OUT_PATH)
    print(f"\nSaved alignment Fisher to: {OUT_PATH}")



# ----------------------------
# ENTRY POINT
# ----------------------------

class DummyLogger:
    def info(self, *args, **kwargs):
        print("[INFO]", *args)

    def warning(self, *args, **kwargs):
        print("[WARN]", *args)

    def error(self, *args, **kwargs):
        print("[ERROR]", *args)


if __name__ == "__main__":
    import argparse
    from ethical_adapter.config_io import load_yaml_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    OUT_PATH = args.out

    main(cfg)
