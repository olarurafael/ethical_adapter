# src/ethical_adapter/train_adapters.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
from ethical_adapter.config_io import load_yaml_config, ensure_dirs
from ethical_adapter.config import AdapterConfig
from ethical_adapter.inject import inject_adapters
from tqdm import tqdm
import os
from ethical_adapter.run_utils import make_run_dir, save_config, get_logger, log_trainable_params



def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_optimizer(model, lr=5e-4):
    # only trainable params (adapters)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr)


def train_step(model, inputs, optimizer, scaler=None):
    model.train()
    optimizer.zero_grad()

    with torch.amp.autocast('cuda', enabled=scaler is not None):
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


def main():
    # --- load YAML config ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    ensure_dirs(cfg)
    # create a timestamped run folder and set up logging
    run_dir = make_run_dir(cfg.get("runs_dir", "./runs"))
    save_config(run_dir, cfg)
    logger = get_logger(run_dir)
    logger.info(f"New training run started in: {run_dir}")


    local_path = cfg["local_path"]

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )


    # freeze base weights
    model = freeze_model(model)

    # inject adapters
    adapter_cfg = AdapterConfig(
    rank=cfg["rank"],
    alpha=cfg["alpha"],
    dropout=cfg["dropout"],
    target_modules=cfg["target_modules"],
)

    res = inject_adapters(model, adapter_cfg)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model = res.model.to(device=device, dtype=dtype)

    # load dataset (1% slice for debug)
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split=cfg["dataset_split"])

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=cfg["max_length"],
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)


    # optimizer (adapters only)
    optimizer = get_optimizer(model, lr=cfg["lr"])
    scaler = None  # bf16 doesn’t need it

    for epoch in range(cfg["epochs"]):
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']} starting")
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=100)

        epoch_losses = []

        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, scaler)
            epoch_losses.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        log_trainable_params(logger, model)

        # save checkpoint each epoch
        save_path = os.path.join(run_dir, f"epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    # save adapter weights only
    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items() if "adapter" in k},
        "adapter_weights.pt",
    )


if __name__ == "__main__":
    main()
