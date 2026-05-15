# src/ethical_adapter/alignguard/train_alignguard_adapters.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from ethical_adapter.core.config import AdapterConfig, GateConfig
from ethical_adapter.core.inject import inject_adapters
from ethical_adapter.core.run_utils import (
    setup_run,
    save_training_checkpoint,
)
from ethical_adapter.core.utils import print_param_summary
from ethical_adapter.training.early_stop_manager import EarlyStopManager
from ethical_adapter.training.load_adapters import load_adapters_from_checkpoint
from ethical_adapter.training.data import (
    build_task_dataset,
    build_alignment_dataset,
    prepare_task_dataset,
    prepare_alignment_dataset,
)
from ethical_adapter.core.adapter import GatedAdapter
from ethical_adapter.training.optim_utils import (
    get_adapter_optimizer,
    prepare_model_for_adapter_training,
    get_scheduler,
)
from ethical_adapter.alignguard.alignguard_utils import (
    alignguard_loss,
    init_task_curvature_identity,
    iter_gated_adapters,
    set_capture_delta_grad,
    get_last_delta_grad,
    compute_delta_w,
)
from ethical_adapter.alignguard.blockwise_subspace import BlockwiseOjaEstimator


# eval step for adapter training
# @torch.no_grad()
# def eval_step(model, loader):
#     model.eval()
#     total_loss = 0.0
#     count = 0

#     for batch in loader:
#         batch = {k: v.to(model.device) for k, v in batch.items()}
#         outputs = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             labels=batch["labels"],
#         )
#         total_loss += outputs.loss.item()
#         count += 1

#     return total_loss / max(count, 1)


@torch.no_grad()
def eval_step(model, loader):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch["labels"] if "labels" in batch else batch["input_ids"]

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        total_loss += outputs.loss.item()
        count += 1

    return total_loss / max(count, 1)

    
@torch.no_grad()
def _cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def recompute_alignment_projector(
    model,
    align_batch_iter,
    num_batches: int,
    block_size: int,
    rank_per_block: int,
    eta0: float,
    orth_every: int,
    init_samples: int,
    previous_projector: dict | None = None,
    logger=None,
):
    """
    Recompute blockwise Fisher projector from fresh alignment minibatches.
    """
    was_training = model.training
    model.eval()

    estimators = {}
    for name, pl in iter_gated_adapters(model):
        key = f"{name}.adapter"
        dW = compute_delta_w(pl)
        estimators[key] = BlockwiseOjaEstimator(
            total_dim=dW.numel(),
            block_size=block_size,
            rank_per_block=rank_per_block,
            eta0=eta0,
            orth_every=orth_every,
            init_samples=init_samples,
            device="cpu",
            dtype=torch.float32,
        )

    set_capture_delta_grad(model, True)

    valid_batches = 0

    for _ in range(num_batches):
        batch = next(align_batch_iter)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        model.zero_grad(set_to_none=True)
        outputs = model(**batch, labels=batch["input_ids"])
        if not torch.isfinite(outputs.loss):
            if logger is not None:
                logger.warning("Skipping projector refresh minibatch with non-finite loss.")
            continue
        outputs.loss.backward()

        saw_valid_grad = False

        for name, pl in iter_gated_adapters(model):
            key = f"{name}.adapter"
            gW = get_last_delta_grad(pl)
            if gW is not None:
                flat_grad = gW.detach().float().cpu().flatten()
                if torch.isfinite(flat_grad).all():
                    estimators[key].update(flat_grad)
                    saw_valid_grad = True

        if saw_valid_grad:
            valid_batches += 1

    set_capture_delta_grad(model, False)
    model.zero_grad(set_to_none=True)

    if was_training:
        model.train()

    uninitialized = [k for k, est in estimators.items() if not est.is_initialized()]
    if uninitialized:
        if logger is not None:
            logger.warning(
                "AlignGuard projector refresh skipped: %s/%s valid minibatches, %s uninitialized estimators. "
                "Keeping previous projector.",
                valid_batches,
                num_batches,
                len(uninitialized),
            )
        if previous_projector is not None:
            return previous_projector
        raise RuntimeError(
            "Projector refresh failed with no usable fallback; "
            f"valid_batches={valid_batches}, uninitialized_estimators={len(uninitialized)}"
        )

    if logger is not None:
        logger.info(
            "AlignGuard projector refresh succeeded with %s/%s valid minibatches.",
            valid_batches,
            num_batches,
        )
    return {k: est.finalize() for k, est in estimators.items()}

# adapter training main function
def main(config):
    # setup logging & run directory
    run_dir, logger = setup_run(config)

    # Early stopping
    es_cfg = config.get("early_stop", {})
    early_stop = EarlyStopManager(
        run_dir=run_dir,
        enabled=es_cfg.get("enabled", False),
        patience=es_cfg.get("patience", 1),
        min_delta=es_cfg.get("min_delta", 0.0),
    )

    # load tokenizer and base model
    tokenizer_name = config.get("tokenizer_name", config.get("local_path"))
    model_name = config.get("model_name", config.get("local_path"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # inject adapters and gate controller
    adapter_cfg = AdapterConfig(
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config["dropout"],
        target_modules=config["target_modules"],
        gate=GateConfig(**config.get("gate", {})),
    )

    injected = inject_adapters(base_model, adapter_cfg)
    model = injected.model
    # gate_controller = injected.gate_controller

    # Freeze or unfreeze parameters
    prepare_model_for_adapter_training(model)
    for m in model.modules():
        if isinstance(m, GatedAdapter):
            m.set_adapter_mode("on")

    model.to(next(model.parameters()).device)

    # Warm-start adapters ONLY if explicitly provided in config
    load_dir = config.get("load_adapters_from", None)

    if load_dir:
        logger.info(f"Loading adapters from checkpoint: {load_dir}")
        model = load_adapters_from_checkpoint(model, load_dir, logger)
    else:
        logger.info("No adapter checkpoint specified; training from scratch.")

        # ---- AlignGuard: load alignment Fisher (optional) ----
    fisher_path = config.get("alignment_fisher_path", None)
    alignment_fisher = None
    ag_cfg = config.get("alignguard", {})

    if fisher_path:
        logger.info(f"Loading AlignGuard Fisher from: {fisher_path}")
        alignment_fisher = torch.load(fisher_path, map_location="cpu")

    use_alignguard = alignment_fisher is not None

    refresh_every = int(ag_cfg.get("refresh_every", 1000))
    refresh_batches = int(ag_cfg.get("refresh_batches", 64))
    block_size = int(config.get("alignguard_block_size", 262144))
    rank_per_block = int(config.get("alignguard_rank_per_block", 4))
    oja_eta0 = float(config.get("alignguard_oja_eta0", 0.5))
    oja_orth_every = int(config.get("alignguard_oja_orth_every", 10))
    oja_init_samples = int(config.get("alignguard_oja_init_samples", rank_per_block))

    # Optional task-curvature proxy (EMA); init only if AlignGuard is on
    task_curvature = None
    if use_alignguard and float(ag_cfg.get("lambda_task", 0.0)) > 0.0:
        task_curvature = init_task_curvature_identity(model)
        logger.info(f"Initialized task regularizer H=I for {len(task_curvature)} layers")
    if use_alignguard:
        logger.info(f"AlignGuard enabled. Fisher entries: {len(alignment_fisher)}")
    else:
        logger.info("AlignGuard disabled (no alignment_fisher_path set).")
        print("WARNING: You are training with AlignGuard objectives disabled. Set 'alignment_fisher_path' in")


    full_task_ds = build_task_dataset(config, logger)
    splits = full_task_ds.train_test_split(test_size=0.1, seed=42)

    train_ds, collator = prepare_task_dataset(splits["train"], tokenizer, config)
    val_ds, _ = prepare_task_dataset(splits["test"], tokenizer, config)


    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=int(config.get("num_workers", 0)),
    )

    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=collator, **loader_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, collate_fn=collator, **loader_kwargs)

    # Separate alignment-preservation loader for projector refresh
    align_full_ds = build_alignment_dataset(config, logger)
    align_ds = prepare_alignment_dataset(align_full_ds, tokenizer, config)

    align_loader = DataLoader(
        align_ds,
        shuffle=True,
        collate_fn=None,
        **loader_kwargs,
    )
    align_batch_iter = _cycle_loader(align_loader)
   
    # optimizer + scheduler
    optimizer = get_adapter_optimizer(
        model, config["lr"], config.get("weight_decay", 0.01)
    )
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = config.get("warmup_steps", 100)

    scheduler = get_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )
    grad_accum = int(config.get("gradient_accumulation_steps", 1))

    best_val = float("inf")
    save_adapter_only = bool(config.get("save_adapter_only", False))

    # training loop
    global_step = 0
    use_amp = True

    for epoch in range(config["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']} starting")
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        total_loss = 0.0

        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            if global_step % grad_accum == 0:
                optimizer.zero_grad()


            if (
                use_alignguard
                and global_step > 0
                and global_step % refresh_every == 0
            ):
                logger.info(
                    f"Recomputing AlignGuard projector at step {global_step} "
                    f"from {refresh_batches} alignment minibatches"
                )
                alignment_fisher = recompute_alignment_projector(
                    model=model,
                    align_batch_iter=align_batch_iter,
                    num_batches=refresh_batches,
                    block_size=block_size,
                    rank_per_block=rank_per_block,
                    eta0=oja_eta0,
                    orth_every=oja_orth_every,
                    init_samples=oja_init_samples,
                    previous_projector=alignment_fisher,
                    logger=logger,
                )

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                labels = batch["labels"] if "labels" in batch else batch["input_ids"]

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                )
                loss = outputs.loss / grad_accum
                # ---- AlignGuard-LoRA objective (adapter-space, ΔW-based) ----
                if use_alignguard:
                    ag_term = alignguard_loss(
                        model=model,
                        fisher_dict=alignment_fisher,
                        task_curv_dict=task_curvature,
                        lambda_align=float(ag_cfg.get("lambda_align", 1.0)),
                        lambda_task=float(ag_cfg.get("lambda_task", 0.0)),
                        lambda_riem=float(ag_cfg.get("lambda_riem", 0.01)),
                        lambda_geo=float(ag_cfg.get("lambda_geo", 0.01)),
                    )
                    loss = loss + (ag_term / grad_accum)

            if not torch.isfinite(loss):
                logger.warning(
                    "Skipping non-finite training loss at global_step=%s (epoch=%s).",
                    global_step,
                    epoch + 1,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * grad_accum
            global_step += 1
            pbar.set_postfix({"loss": f"{(loss.item() * grad_accum):.4f}"})

        avg_train = total_loss / len(train_loader)
        val_loss = eval_step(model, val_loader)

        logger.info(
            f"Epoch {epoch + 1} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f}"
        )
        print_param_summary(model)

        # Save checkpoints
        if (epoch + 1) % config["save_every"] == 0:
            save_training_checkpoint(
                model,
                tokenizer,
                run_dir,
                epoch + 1,
                logger,
                adapter_only=save_adapter_only,
            )

        if val_loss < best_val:
            best_val = val_loss
            save_training_checkpoint(
                model,
                tokenizer,
                run_dir,
                "best",
                logger,
                best=True,
                adapter_only=save_adapter_only,
            )

        # early stopping
        if early_stop.should_stop(val_loss, epoch + 1):
            reason = early_stop.reason
            logger.info("Early stopping triggered (%s).", reason)
            save_training_checkpoint(
                model,
                tokenizer,
                run_dir,
                "early_stop",
                logger,
                adapter_only=save_adapter_only,
            )
            break

    logger.info(f"Training complete. Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    import argparse
    from ethical_adapter.config_io import load_yaml_config

    print("Starting training...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    main(cfg)
