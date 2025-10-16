# src/ethical_adapter/train_adapters.py
import torch
import logging
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from ethical_adapter.config import AdapterConfig
from ethical_adapter.inject import inject_adapters
from ethical_adapter.utils import print_param_summary
from ethical_adapter.run_utils import setup_run, save_checkpoint
from ethical_adapter.early_stop_manager import EarlyStopManager
from ethical_adapter.load_adapters import load_adapters_from_checkpoint
from ethical_adapter.run_utils import get_latest_best_checkpoint


# ------------------------------------------------------------
# core utilities
# ------------------------------------------------------------
def freeze_model(model):
    """Freeze all model parameters to disable gradient updates."""
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_optimizer(model, lr=5e-4):
    """Return an AdamW optimizer over trainable model parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)


def train_step(model, inputs, optimizer, scaler=None, accum_steps=1, step_idx=0):
    model.train()
    # only zero the grads at the start of an accumulation window
    if step_idx % accum_steps == 0:
        optimizer.zero_grad()

    with torch.amp.autocast("cuda", enabled=scaler is not None):
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss / accum_steps

    if scaler:
        scaler.scale(loss).backward()
        # only step when we’ve accumulated enough micro-batches
        if (step_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        if (step_idx + 1) % accum_steps == 0:
            optimizer.step()

    return loss.item() * accum_steps  # return the original (unscaled) loss


@torch.no_grad()
def eval_step(model, loader):
    """Evaluate model loss over a validation loader."""
    model.eval()
    total_loss = 0
    count = 0
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        total_loss += outputs.loss.item()
        count += 1
    return total_loss / max(count, 1)



# ------------------------------------------------------------
# dataset helpers
# ------------------------------------------------------------
def example_to_text(ex, dcfg):
    # use dataset-specific text field if defined
    tf = dcfg.get("text_field")
    if tf and tf in ex and isinstance(ex[tf], str):
        return ex[tf]

    # for HH-RLHF datasets: prefer concatenating prompt + chosen response
    if "prompt" in ex and "chosen" in ex and isinstance(ex["prompt"], str) and isinstance(ex["chosen"], str):
        return ex["prompt"] + "\n" + ex["chosen"]

    # common fallbacks (different datasets use different names)
    for k in ("text", "comment_text", "content"):
        if k in ex and isinstance(ex[k], str):
            return ex[k]

    # fallback: join all string fields if none match
    pieces = [str(v) for v in ex.values() if isinstance(v, str)]
    return "\n".join(pieces) if pieces else ""


def load_and_merge_datasets(config):
    merged = None
    for dcfg in config["datasets"]:
        ds = load_dataset(
            dcfg["name"],
            dcfg.get("config"),
            split=dcfg["split"],
            cache_dir=config["data_dir"],
        )
        # map every row to a single 'text' column
        ds = ds.map(
            lambda ex: {"text": example_to_text(ex, dcfg)},
            remove_columns=ds.column_names,
            num_proc=config["num_proc"] if "num_proc" in config else 1, 
        )
        merged = ds if merged is None else concatenate_datasets([merged, ds])

    merged = merged.shuffle(seed=42)
    
    if "max_train_samples" in config and config["max_train_samples"]:
        limit = config["max_train_samples"]
        limit = min(limit, len(merged))
        logging.info("Using only %d samples out of %d for training.", limit, len(merged))
        merged = merged.select(range(limit))

    return merged


# ------------------------------------------------------------
# main training function
# ------------------------------------------------------------
def main(config):
    """Main training entrypoint for adapter fine-tuning."""

    # setup run directory + logger
    run_dir, logger = setup_run(config)

    # early stopping setup
    es_cfg = config.get("early_stop", {})
    early_stop = EarlyStopManager(
        run_dir=run_dir,
        enabled=es_cfg.get("enabled", False),
        patience=es_cfg.get("patience", 1),
        min_delta=es_cfg.get("min_delta", 0.0),
    )


    tokenizer = AutoTokenizer.from_pretrained(config["local_path"])
    # ensure pad_token is set (common for decoder-only models)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config["local_path"],
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model = freeze_model(model)

    # inject adapters
    cfg = AdapterConfig(
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config["dropout"],
        target_modules=config["target_modules"],
    )
    res = inject_adapters(model, cfg)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model = res.model.to(device=device, dtype=dtype)


    # --- Optional warm start ---
    load_dir = config.get("load_adapters_from")
    if not load_dir:
        load_dir = get_latest_best_checkpoint(config.get("runs_dir", "./runs"))
        if load_dir:
            logger.info("Auto-selected latest best checkpoint: %s", load_dir)

        else:
            logger.info("No previous 'best' checkpoint found — starting fresh.")

    if load_dir:
        model = load_adapters_from_checkpoint(model, load_dir, logger)


    # --------------------------------------------------------
    # dataset mixture (train+val split AFTER merge)
    # --------------------------------------------------------
    logger.info("Loading and merging datasets from cache directory: %s", config["data_dir"])    
    full_ds = load_and_merge_datasets(config)

    # 90/10 split for validation
    splits = full_ds.train_test_split(test_size=0.1, seed=42)
    logger.info("Split dataset into %d training and %d validation samples.", len(splits["train"]), len(splits["test"]))
    train_ds, val_ds = splits["train"], splits["test"]

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=config["max_length"],
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=config["num_proc"] if "num_proc" in config else 1)
    val_ds   = val_ds.map(tokenize_fn,   batched=True, remove_columns=["text"], num_proc=config["num_proc"] if "num_proc" in config else 1)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=int(config.get("num_workers", 0)),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # optimizer (adapters only)
    optimizer = get_optimizer(model, lr=config["lr"])
    scaler = None  # bf16 doesn’t need it

    # gradient accumulation (from YAML)
    grad_accum = int(config.get("gradient_accumulation_steps", 1))

    best_val = float("inf")

    # --------------------------------------------------------
    # training loop
    # --------------------------------------------------------
    global_step = 0
    for epoch in range(config["epochs"]):
        logger.info("Epoch %d/%d starting", epoch + 1, config["epochs"])
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0

        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = train_step(
                model,
                batch,
                optimizer,
                scaler,
                accum_steps=grad_accum,
                step_idx=global_step,
            )
            total_loss += loss
            global_step += 1
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_train = total_loss / len(train_loader)
        val_loss = eval_step(model, val_loader)
        logger.info("Epoch %d | train_loss=%.4f | val_loss=%.4f", epoch + 1, avg_train, val_loss)

        print_param_summary(model)

        # save checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            save_checkpoint(model, tokenizer, run_dir, epoch + 1, logger)

        # best model logic
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, tokenizer, run_dir, "best", logger, best=True)

        # early stopping check
        if early_stop.update(val_loss, epoch + 1):
            logger.info(
                "Metric early stop triggered at epoch %d (best was epoch %d with val_loss=%.4f)",
                epoch + 1, early_stop.best_epoch, early_stop.best_val,
            )

            save_checkpoint(model, tokenizer, run_dir, "early_stop", logger)
            break

        if early_stop.manual_stop_requested():
            logger.info("Manual early stop requested — saving checkpoint and exiting.")
            save_checkpoint(model, tokenizer, run_dir, epoch + 1, logger)
            break 

    logger.info("Training completed successfully.")
    logger.info("Best validation loss: %.4f", best_val)




# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from ethical_adapter.config_io import load_yaml_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
   
    main(cfg)
