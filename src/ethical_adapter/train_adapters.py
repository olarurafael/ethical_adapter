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
    # honor explicit text_field if provided and present
    tf = dcfg.get("text_field")
    if tf and tf in ex and isinstance(ex[tf], str):
        return ex[tf]

    # Anthropic HH-RLHF: prefer prompt + chosen (positive response)
    if "prompt" in ex and "chosen" in ex and isinstance(ex["prompt"], str) and isinstance(ex["chosen"], str):
        return ex["prompt"] + "\n" + ex["chosen"]

    # common fallbacks (different datasets use different names)
    for k in ("text", "comment_text", "content"):
        if k in ex and isinstance(ex[k], str):
            return ex[k]

    # last resort: join all string-like fields
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

    # big mixture -> shuffle for interleaving
    merged = merged.shuffle(seed=42)
    if "max_train_samples" in config and config["max_train_samples"]:
        limit = config["max_train_samples"]
        limit = min(limit, len(merged))
        print(f"[INFO] Using only {limit} samples out of {len(merged)} for training.")
        merged = merged.select(range(limit))

    return merged


# ------------------------------------------------------------
# main training function
# ------------------------------------------------------------
def main(config):
    # setup run directory + logger
    run_dir, logger = setup_run(config)
    log = lambda msg: log_info(msg, logger)

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

    # --------------------------------------------------------
    # dataset mixture (train+val split AFTER merge)
    # --------------------------------------------------------
    log(f"Loading multi-dataset mixture from cache: {config['data_dir']}")
    full_ds = load_and_merge_datasets(config)

    # 90/10 split for validation
    splits = full_ds.train_test_split(test_size=0.1, seed=42)
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
        log(f"Epoch {epoch+1}/{config['epochs']} starting")
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
