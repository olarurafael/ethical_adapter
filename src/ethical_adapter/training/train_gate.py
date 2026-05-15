# src/ethical_adapter/train_gate.py
import torch
import torch.nn.functional as F
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
    build_gate_dataset,
    load_gate_dataset,
    tokenize_text_dataset,
)
from ethical_adapter.core.adapter import ParallelLinear
from ethical_adapter.training.optim_utils import (
    prepare_model_for_gate_training,
    get_gate_optimizer,
    get_scheduler,
)


def gate_loss(gate_logits, labels):
    target = labels.to(gate_logits.device, gate_logits.dtype).unsqueeze(1)
    return F.binary_cross_entropy_with_logits(gate_logits, target)



@torch.no_grad()
def eval_step(model, loader):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}

        _ = model(**batch)

        gate_logits = model.gate_store["logits"]

        loss = gate_loss(gate_logits, batch["label"])

        total_loss += loss.item()

    return total_loss / len(loader)


# gate training main function
def main(config):
    # Setup logging & run directory
    run_dir, logger = setup_run(config)

    # Early stopping
    es_cfg = config.get("early_stop", {})
    early_stop = EarlyStopManager(
        run_dir=run_dir,
        enabled=es_cfg.get("enabled", False),
        patience=es_cfg.get("patience", 1),
        min_delta=es_cfg.get("min_delta", 0.0),
    )

    # --------------------------------------------------------
    # Tokenizer & Base Model
    # --------------------------------------------------------
    tokenizer_name = config.get("tokenizer_name", config.get("local_path"))
    model_name = config.get("model_name", config.get("local_path"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to("cuda")
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    # inject adapters and gate controller.
    adapter_cfg = AdapterConfig(
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config["dropout"],
        target_modules=config["target_modules"],
        gate=GateConfig(**config.get("gate", {})),
    )

    injected = inject_adapters(base_model, adapter_cfg)
    model = injected.model
    gate_controller = injected.gate_controller

    # configure params for gate toxicity training.
    prepare_model_for_gate_training(model, gate_controller)

    # make sure adapters are not forced to be open or closed.
    for m in model.modules():
        if isinstance(m, ParallelLinear):
            m.force_gate_open = False
            m.force_gate_closed = False

    model.to(next(model.parameters()).device)

    # Load pretrained adapters — REQUIRED for gate training
    load_dir = config.get("load_adapters_from", None)

    if load_dir:
        logger.info(f"Loading adapters from checkpoint: {load_dir}")
        model = load_adapters_from_checkpoint(model, load_dir, logger)
    else:
        logger.warning("No adapter checkpoint specified; training from scratch.")

    # --------------------------------------------------------
    # Load datasets for gate_toxicity phase
    # --------------------------------------------------------
    full_ds = build_gate_dataset(config, logger)
    gate_val_cfg = config.get("datasets", {}).get("gate_validation", [])
    if gate_val_cfg:
        logger.info("Using explicit gate validation dataset.")
        train_raw = full_ds
        val_raw = load_gate_dataset(config, gate_val_cfg)
    else:
        logger.info("No explicit gate validation dataset; using internal 90/10 split.")
        splits = full_ds.train_test_split(
            test_size=float(config.get("validation_split", 0.1)),
            seed=int(config.get("dataset_seed", 42)),
        )
        train_raw = splits["train"]
        val_raw = splits["test"]

    logger.info("Gate train rows: %d | validation rows: %d", len(train_raw), len(val_raw))

    train_ds = tokenize_text_dataset(train_raw, tokenizer, config)
    val_ds = tokenize_text_dataset(val_raw, tokenizer, config)

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=int(config.get("num_workers", 0)),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # --------------------------------------------------------
    # Optimizer (gate params only)
    # --------------------------------------------------------
    optimizer = get_gate_optimizer(
        gate_controller,
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_scheduler(
        optimizer, warmup_steps=config.get("warmup_steps", 100), total_steps=total_steps
    )
    grad_accum = int(config.get("gradient_accumulation_steps", 1))

    best_val = float("inf")
    save_adapter_only = bool(config.get("save_adapter_only", False))

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    global_step = 0
    use_amp = True


    torch.cuda.empty_cache()
    for epoch in range(config["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']} starting")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        total_loss = 0.0
        model.train()

        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            if global_step % grad_accum == 0:
                optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _ = model(**batch)

                gate_logits = model.gate_store["logits"]

                loss = gate_loss(gate_logits, batch["label"]) / grad_accum

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

        # Save best model
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


        # Early stopping
        if early_stop.should_stop(val_loss, epoch + 1):
            logger.info("Early stopping triggered (%s).", early_stop.reason)
            save_training_checkpoint(
                model,
                tokenizer,
                run_dir,
                "early_stop",
                logger,
                adapter_only=save_adapter_only,
            )
            break

        
        # Periodic save
        if (epoch + 1) % config["save_every"] == 0:
            save_training_checkpoint(
                model,
                tokenizer,
                run_dir,
                epoch + 1,
                logger,
                adapter_only=save_adapter_only,
            )

    logger.info(f"Gate training complete. Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    import argparse
    from ethical_adapter.config_io import load_yaml_config

    print("Starting gate training...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    main(cfg)
