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
    tokenize_text_dataset,
    tokenize_supervised_dataset,
)
from ethical_adapter.core.adapter import GatedAdapter
from ethical_adapter.training.optim_utils import (
    get_adapter_optimizer,
    prepare_model_for_adapter_training,
    get_scheduler,
)
from ethical_adapter.training.data import SupervisedCollator


def _seed_training(config, logger) -> int:
    seed = int(config.get("training_seed", config.get("seed", 42)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Training seed set to %d", seed)
    return seed


@torch.no_grad()
def eval_step(model, loader):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        total_loss += outputs.loss.item()
        count += 1

    return total_loss / max(count, 1)


def main(config):
    run_dir, logger = setup_run(config)
    training_seed = _seed_training(config, logger)
    dataset_seed = int(config.get("dataset_seed", 42))

    es_cfg = config.get("early_stop", {})
    early_stop = EarlyStopManager(
        run_dir=run_dir,
        enabled=es_cfg.get("enabled", False),
        patience=es_cfg.get("patience", 1),
        min_delta=es_cfg.get("min_delta", 0.0),
    )

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

    adapter_cfg = AdapterConfig(
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config["dropout"],
        target_modules=config["target_modules"],
        gate=GateConfig(**config.get("gate", {})),
    )

    injected = inject_adapters(base_model, adapter_cfg)
    model = injected.model

    prepare_model_for_adapter_training(model)
    for m in model.modules():
        if isinstance(m, GatedAdapter):
            m.set_adapter_mode("on")

    model.to(next(model.parameters()).device)

    load_dir = config.get("load_adapters_from", None)

    if load_dir:
        logger.info(f"Loading adapters from checkpoint: {load_dir}")
        model = load_adapters_from_checkpoint(model, load_dir, logger)
    else:
        logger.info("No adapter checkpoint specified; training from scratch.")

    full_ds = build_task_dataset(config, logger)

    splits = full_ds.train_test_split(test_size=0.1, seed=dataset_seed)

    if "prompt" in full_ds.column_names:
        collator = SupervisedCollator(tokenizer)
        train_ds = tokenize_supervised_dataset(splits["train"], tokenizer, config)
        val_ds = tokenize_supervised_dataset(splits["test"], tokenizer, config)
    else:
        collator = None
        train_ds = tokenize_text_dataset(splits["train"], tokenizer, config)
        val_ds = tokenize_text_dataset(splits["test"], tokenizer, config)

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=int(config.get("num_workers", 0)),
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(training_seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        collate_fn=collator,
        **loader_kwargs,
    )

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

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss / grad_accum

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    main(cfg)
