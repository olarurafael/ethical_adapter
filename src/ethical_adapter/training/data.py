# src/ethical_adapter/training/data.py
import logging
import torch
import random
import os
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import torch

from typing import Dict, List, Any

from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
    load_from_disk,
)

from ethical_adapter.training.glue_format import FORMATTERS

@dataclass
class SupervisedCollator:
    tokenizer: any

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def smart_load_dataset(name, config_name=None, split=None, **kwargs):
    """
    Automatically loads either:
    - a local dataset saved with save_to_disk() using load_from_disk, or
    - a standard HF dataset via load_dataset().
    """
    # If local directory containing dataset_info.json → load_from_disk
    if os.path.isdir(name) and os.path.exists(os.path.join(name, "dataset_info.json")):
        ds = load_from_disk(name)
        return ds

    # Otherwise fall back to the normal HF loader
    return load_dataset(name, config_name, split=split, **kwargs)


def get_text_field(ex: Dict[str, Any], dcfg: Dict[str, Any]) -> str:
    # custom field override
    tf = dcfg.get("text_field")
    if tf and tf in ex and isinstance(ex[tf], str):
        return ex[tf]

    # HH-RLHF style (prompt + chosen)
    if "prompt" in ex and "chosen" in ex:
        p, c = ex["prompt"], ex["chosen"]
        if isinstance(p, str) and isinstance(c, str):
            return p + "\n" + c

    # common fields
    for key in ("text", "content", "comment_text", "prompt"):
        if key in ex and isinstance(ex[key], str):
            return ex[key]

    # fallback: concatenate all string fields
    pieces = [v for v in ex.values() if isinstance(v, str)]
    return "\n".join(pieces)


def _load_and_merge(
    config: Dict[str, Any], subset_cfg: List[Dict[str, Any]]
) -> Dataset:
    merged = None
    for dcfg in subset_cfg:
        ds = smart_load_dataset(
            dcfg["name"],
            dcfg.get("config"),
            split=dcfg.get("split"),
            cache_dir=config["data_dir"],
        )

        # KEEP ORIGINAL COLUMNS + TEXT
        ds = ds.map(
            lambda ex: {"text": get_text_field(ex, dcfg)},
            num_proc=config.get("num_proc", 1),
        )

        merged = ds if merged is None else concatenate_datasets([merged, ds])

    merged = merged.shuffle(seed=42)

    if config.get("max_train_samples"):
        limit = min(config["max_train_samples"], len(merged))
        logging.info(
            "Using only %d samples out of %d for training.", limit, len(merged)
        )
        merged = merged.select(range(limit))

    return merged


def extract_toxicity_field(ex, field):
    val = ex.get(field)

    # RTP-style nested structure
    if isinstance(val, dict):
        tox = val.get("toxicity")
        if isinstance(tox, (int, float)):
            return float(tox)
        return None

    # Flat structure
    if isinstance(val, (int, float)):
        return float(val)

    return None


def load_alignment_phase(config, datasets_cfg):
    rows = []
    max_len = config.get("max_train_samples")

    for dcfg in datasets_cfg:
        ds = smart_load_dataset(
            dcfg["name"],
            dcfg.get("config"),
            split=dcfg.get("split", "train"),
            cache_dir=config["data_dir"],
            trust_remote_code=True,
        )

        text_field = dcfg.get("text_field", "text")
        tox_field = dcfg.get("toxicity_field", "toxicity")

        for ex in ds:
            # ---- Handle RealToxicityPrompts nested samples ----
            val = ex.get(text_field)
            if isinstance(val, dict):
                text = val.get("text")
                tox = extract_toxicity_field(val, "toxicity")
                if isinstance(text, str) and tox is not None:
                    rows.append({"text": text, "toxicity": tox})
                continue

            # ---- Standard flat datasets ----
            if text_field not in ex or tox_field not in ex:
                continue

            text = ex[text_field]
            tox = extract_toxicity_field(ex, tox_field)
            if isinstance(text, str) and tox is not None:
                rows.append({"text": text, "toxicity": tox})

    random.shuffle(rows)
    if max_len:
        rows = rows[:max_len]

    return Dataset.from_list(rows)


def build_phase_dataset(config, logger, phase):
    all_cfg = config["datasets"]

    def by_role(role):
        return [d for d in all_cfg if d.get("role") == role]

    logger.info(f"Building dataset for phase: {phase}")

    # ---- Phase 1: normal text datasets ----
    if phase == "adapters":
        task_cfg = by_role("task")

        # if datasets specify task_type → supervised GLUE mode
        if any("task_type" in d for d in task_cfg):
            return load_supervised_task_dataset(config, task_cfg)

        used = task_cfg if task_cfg else all_cfg
        return _load_and_merge(config, used)
    
    if phase == "alignment":
        align_cfg = by_role("alignment")
        if not align_cfg:
            raise ValueError("No datasets with role='alignment' configured.")
        return _load_and_merge(config, align_cfg)

    # ---- Phase 2: toxicity datasets ----
    if phase == "gate_toxicity":
        align_cfg = by_role("alignment")
        return load_alignment_phase(config, align_cfg)

    raise ValueError(f"Unknown phase: {phase}")


def tokenize_supervised_dataset(ds, tokenizer, config):
    max_length = config["max_length"]

    def tok_fn(ex):
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["answer"]},
        ]

        # Render chat text first (string)
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Now tokenize normally → returns lists of ints
        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        prompt_enc = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = full_enc["input_ids"]
        labels = input_ids.copy()

        cut = min(len(prompt_enc["input_ids"]), len(labels))
        for i in range(cut):
            labels[i] = -100

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    new_ds = ds.map(tok_fn, remove_columns=ds.column_names)
    new_ds.set_format(type="torch")
    return new_ds

    max_length = config["max_length"]

    def tok_fn(ex):
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["answer"]},
        ]

        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )

        input_ids = full_ids[:max_length]
        labels = input_ids.copy()

        cut = min(len(prompt_ids), len(labels))
        for i in range(cut):
            labels[i] = -100

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    new_ds = ds.map(tok_fn, remove_columns=ds.column_names)
    new_ds.set_format(type="torch")
    return new_ds


def tokenize_dataset(ds, tokenizer, config):
    # Automatically infer numeric fields (toxicity, score, labels, etc.)
    numeric_cols = [
        col
        for col in ds.column_names
        if col != "text"
        and all(isinstance(x, (int, float)) or x is None for x in ds[col])
    ]

    def tok_fn(batch):
        enc = tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=config["max_length"],
        )

        out = {k: v for k, v in enc.items()}

        # Preserve all numeric fields as float32 tensors
        for col in numeric_cols:
            out[col] = torch.tensor(batch[col], dtype=torch.float32)

        return out

    new_ds = ds.map(
        tok_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=config.get("num_proc", 1),
    )

    new_ds.set_format(type="torch")
    return new_ds

def load_supervised_task_dataset(config, datasets_cfg):
    rows = []
    max_len = config.get("max_train_samples")

    for dcfg in datasets_cfg:
        name = dcfg["name"]
        task = dcfg.get("task_type")

        if task not in FORMATTERS:
            raise ValueError(f"Unknown task_type: {task}")

        formatter = FORMATTERS[task]

        ds = smart_load_dataset(
            name,
            dcfg.get("config"),
            split=dcfg.get("split", "train"),
            cache_dir=config["data_dir"],
        )

        for ex in ds:
            # skip missing labels (e.g. MNLI test)
            if "label" not in ex or ex["label"] == -1:
                continue

            prompt, answer = formatter(ex)

            rows.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                }
            )

    random.shuffle(rows)

    if max_len:
        rows = rows[:max_len]

    return Dataset.from_list(rows)
