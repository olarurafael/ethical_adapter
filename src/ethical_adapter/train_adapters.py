# src/ethical_adapter/train_adapters.py
import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from ethical_adapter.config import AdapterConfig
from ethical_adapter.inject import inject_adapters
from ethical_adapter.utils import print_param_summary
from ethical_adapter.run_utils import setup_run, log_info, save_checkpoint  # new helpers


# ------------------------------------------------------------
# core utilities
# ------------------------------------------------------------
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_optimizer(model, lr=5e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)


def train_step(model, inputs, optimizer, scaler=None):
    model.train()
    optimizer.zero_grad()

    with torch.amp.autocast("cuda", enabled=scaler is not None):
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def eval_step(model, loader):
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
# main training function
# ------------------------------------------------------------
def main(config):
    # setup run directory + logger
    run_dir, logger = setup_run(config)
    log = lambda msg: log_info(msg, logger)

    tokenizer = AutoTokenizer.from_pretrained(config["local_path"])
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

    # --------------------------------------------------------
    # dataset (train + validation split)
    # --------------------------------------------------------
    log(f"Loading dataset {config['dataset_name']} ({config['dataset_split']})...")
    merged = None
    for dcfg in config["datasets"]:
        ds = load_dataset(
            dcfg["name"],
            dcfg["config"],
            split=dcfg["split"],
            cache_dir=config["data_dir"],
        )
        merged = ds if merged is None else concatenate_datasets([merged, ds])

    # simple 90/10 split for validation
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = ds["train"], ds["test"]

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=config["max_length"],
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    # optimizer (adapters only)
    optimizer = get_optimizer(model, lr=config["lr"])
    scaler = None  # bf16 doesn’t need it

    best_val = float("inf")

    # --------------------------------------------------------
    # training loop
    # --------------------------------------------------------
    for epoch in range(config["epochs"]):
        log(f"Epoch {epoch+1}/{config['epochs']} starting")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_loss = 0

        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, scaler)
            total_loss += loss
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_train = total_loss / len(train_loader)
        val_loss = eval_step(model, val_loader)
        log(f"Epoch {epoch+1} train_loss={avg_train:.4f} val_loss={val_loss:.4f}")

        print_param_summary(model)

        # save checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            save_checkpoint(model, tokenizer, run_dir, epoch + 1, log)

        # best model logic
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, tokenizer, run_dir, "best", log, best=True)

    log("Training completed successfully.")


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
