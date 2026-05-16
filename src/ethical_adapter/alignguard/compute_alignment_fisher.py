import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from ethical_adapter.core.config import AdapterConfig, GateConfig
from ethical_adapter.core.inject import inject_adapters
from ethical_adapter.training.data import (
    build_alignment_dataset,
    prepare_alignment_dataset,
)
from ethical_adapter.core.adapter import GatedAdapter
from ethical_adapter.alignguard.alignguard_utils import (
    iter_gated_adapters,
    compute_delta_w,
    set_capture_delta_grad,
    get_last_delta_grad,
)
from ethical_adapter.alignguard.blockwise_subspace import BlockwiseOjaEstimator


DEVICE = "cuda"
DTYPE = torch.bfloat16
BATCH_SIZE = 1
MAX_BATCHES = 1000
OUT_PATH = "alignment_fisher.pt"


def main(cfg):
    seed = int(cfg.get("fisher_seed", cfg.get("alignment_seed", 0)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer_name = cfg.get("tokenizer_name", cfg.get("local_path"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_name = cfg.get("model_name", cfg.get("local_path"))
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto",
    )

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

    for m in model.modules():
        if isinstance(m, GatedAdapter):
            m.set_adapter_mode("on")

    model.to(DEVICE)

    logger = DummyLogger()
    align_ds = build_alignment_dataset(cfg, logger=logger)
    align_ds = prepare_alignment_dataset(align_ds, tokenizer, cfg)

    batch_size = int(cfg.get("alignment_fisher_batch_size", BATCH_SIZE))
    max_batches = cfg.get("alignment_fisher_max_batches", MAX_BATCHES)
    if max_batches is not None:
        max_batches = int(max_batches)

    loader = DataLoader(
        align_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    block_size = int(cfg.get("alignguard_block_size", 262144))
    rank_per_block = int(cfg.get("alignguard_rank_per_block", 4))
    eta0 = float(cfg.get("alignguard_oja_eta0", 0.5))
    orth_every = int(cfg.get("alignguard_oja_orth_every", 10))
    init_samples = int(cfg.get("alignguard_oja_init_samples", rank_per_block))

    estimators = {}
    for name, adapter_module in iter_gated_adapters(model):
        key = f"{name}.adapter"
        delta_weight = compute_delta_w(adapter_module)
        estimators[key] = BlockwiseOjaEstimator(
            total_dim=delta_weight.numel(),
            block_size=block_size,
            rank_per_block=rank_per_block,
            eta0=eta0,
            orth_every=orth_every,
            init_samples=init_samples,
            device="cpu",
            dtype=torch.float32,
        )

    set_capture_delta_grad(model, True)

    step = 0
    for batch in loader:
        if max_batches is not None and step >= max_batches:
            break

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        outputs.loss.backward()

        for name, adapter_module in iter_gated_adapters(model):
            key = f"{name}.adapter"
            delta_grad = get_last_delta_grad(adapter_module)
            if delta_grad is not None:
                estimators[key].update(delta_grad.detach().float().cpu().flatten())

        step += 1
        if step % 25 == 0:
            print(f"[AlignGuard Fisher] processed {step} batches")

    set_capture_delta_grad(model, False)

    model.zero_grad(set_to_none=True)
    proj = {k: est.finalize() for k, est in estimators.items()}
    torch.save(proj, OUT_PATH)
    print(f"Saved blockwise alignment projector to: {OUT_PATH}")


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
