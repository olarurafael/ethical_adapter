# src/ethical_adapter/metrics.py
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import logging


# ------------------------------------------------------------
# 1. Perplexity (fluency cost)
# ------------------------------------------------------------
@torch.no_grad()
def compute_perplexity(model, tokenizer,
                       dataset_name="wikitext",
                       dataset_config="wikitext-2-raw-v1",
                       split="test",
                       max_samples=100,
                       max_length=512):
    """
    Estimate model fluency using perplexity on a small reference dataset.
    """
    
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(min(max_samples, len(ds))))

    total_loss = 0.0
    count = 0

    for ex in tqdm(ds, desc="Evaluating perplexity"):
        text = ex["text"].strip()
        if not text:
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ).to(model.device)

        # Create labels once and mask out padding tokens
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100  # ignore pad tokens in loss
        
        # Run the model with mixed precision if applicable
        with torch.amp.autocast("cuda", enabled=(model.dtype in (torch.bfloat16, torch.float16))):
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.detach()

        total_loss += loss.item()
        count += 1

    # Handle empty sample case gracefully
    if count == 0:
        return float("nan")

    mean_loss = total_loss / count
    ppl = float(torch.exp(torch.tensor(mean_loss)))
    return ppl



# ------------------------------------------------------------
# 2. Representation similarity
# ------------------------------------------------------------
@torch.no_grad()
def cosine_similarity_between_models(model_a, model_b, tokenizer, text="Moral reasoning requires empathy.", layer_name="model.layers.15.mlp.down_proj"):
    """
    Compute average cosine similarity between hidden states of two models
    on the same input (using the last hidden state or a named layer if available).
    """
    inputs = tokenizer(text, return_tensors="pt").to(model_a.device)

    # get activations
    def get_repr(model):
        outputs = model(**inputs, output_hidden_states=True)
        # use last hidden state for simplicity
        hidden = outputs.hidden_states[-1]
        return hidden.mean(dim=1).squeeze(0)  # (hidden_size,)

    a = get_repr(model_a)
    b = get_repr(model_b)

    sim = F.cosine_similarity(a, b, dim=0).item()
    return sim


# ------------------------------------------------------------
# 3. Pretty print / comparison helper
# ------------------------------------------------------------
def evaluate_models(vanilla, adapter, tokenizer):
    """
    Run a quick evaluation summary: PPL delta + cosine shift.
    """
    logging.info("Evaluating fluency (perplexity)...")
    ppl_vanilla = compute_perplexity(vanilla, tokenizer)
    ppl_adapter = compute_perplexity(adapter, tokenizer)
    logging.info("Vanilla PPL: %.2f", ppl_vanilla)
    logging.info("Adapter PPL: %.2f", ppl_adapter)
    logging.info("Δ Perplexity: %+0.2f", ppl_adapter - ppl_vanilla)

    logging.info("Evaluating representation similarity...")
    sim = cosine_similarity_between_models(vanilla, adapter, tokenizer)
    logging.info("Cosine similarity between models: %.4f", sim)
