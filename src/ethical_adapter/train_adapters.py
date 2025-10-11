# src/ethical_adapter/train_adapters.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from ethical_adapter.config import AdapterConfig
from ethical_adapter.inject import inject_adapters
from tqdm import tqdm


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
    local_path = "./models/OLMo-2-0425-1B-full"

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # freeze base weights
    model = freeze_model(model)

    # inject adapters
    cfg = AdapterConfig(
        rank=8,
        alpha=16.0,
        dropout=0.05,
        target_modules=[
            "model.layers.8.mlp.down_proj",
            "model.layers.15.mlp.down_proj",
        ],
    )
    res = inject_adapters(model, cfg)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model = res.model.to(device=device, dtype=dtype)

    # load dataset (1% slice for debug)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = DataLoader(ds, batch_size=2, shuffle=True)

    # optimizer (adapters only)
    optimizer = get_optimizer(model, lr=5e-4)
    scaler = None  # bf16 doesn’t need it

    # training loop
    for epoch in range(1):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, scaler)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

    # save adapter weights only
    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items() if "adapter" in k},
        "adapter_weights.pt",
    )


if __name__ == "__main__":
    main()
