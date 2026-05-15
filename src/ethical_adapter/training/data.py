import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
)

from ethical_adapter.training.glue_format import FORMATTERS


# -------------------------
# Collators
# -------------------------

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


# -------------------------
# Dataset loading helpers
# -------------------------

def smart_load_dataset(name, config_name=None, split=None, **kwargs):
    """
    Loads:
      - a local JSON/JSONL file via datasets json loader
      - otherwise a normal HF dataset
    """
    if Path(name).exists():
        return load_dataset(
            "json",
            data_files=name,
            split=split,
        )

    return load_dataset(name, config_name, split=split, **kwargs)


def get_text_field(ex: Dict[str, Any], dcfg: Dict[str, Any]) -> str:
    tf = dcfg.get("text_field")
    if tf and tf in ex and isinstance(ex[tf], str):
        return ex[tf]

    if "prompt" in ex and "chosen" in ex:
        p, c = ex["prompt"], ex["chosen"]
        if isinstance(p, str) and isinstance(c, str):
            return p + "\n" + c

    for key in ("text", "content", "comment_text", "prompt"):
        if key in ex and isinstance(ex[key], str):
            return ex[key]

    pieces = [v for v in ex.values() if isinstance(v, str)]
    return "\n".join(pieces)


def _apply_max_samples(ds: Dataset, config: Dict[str, Any]) -> Dataset:
    if config.get("max_train_samples"):
        limit = min(config["max_train_samples"], len(ds))
        logging.info("Using only %d samples out of %d.", limit, len(ds))
        ds = ds.select(range(limit))
    return ds


def _get_dataset_seed(config: Dict[str, Any]) -> int:
    return int(config.get("dataset_seed", 42))


def _write_dataset_jsonl(ds: Dataset, path: str) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return digest


def _maybe_load_frozen_task_dataset(config: Dict[str, Any], logger) -> Dataset | None:
    frozen_path = config.get("frozen_task_dataset_path")
    if not frozen_path:
        return None

    frozen = Path(frozen_path)
    if not frozen.exists():
        return None

    logger.info("Loading frozen task dataset from %s", frozen)
    ds = smart_load_dataset(str(frozen), split="train")

    required = {"prompt", "answer"}
    if not required.issubset(ds.column_names):
        raise ValueError(
            f"Frozen task dataset {frozen} must contain columns {sorted(required)}; "
            f"got {ds.column_names}"
        )

    logger.info("Loaded %d frozen task rows.", len(ds))
    return ds


def _maybe_export_task_dataset(ds: Dataset, config: Dict[str, Any], logger) -> None:
    export_path = config.get("frozen_task_dataset_path") or config.get("export_task_dataset_path")
    if not export_path:
        return

    digest = _write_dataset_jsonl(ds, export_path)
    logger.info(
        "Exported task dataset to %s (%d rows, sha256=%s)",
        export_path,
        len(ds),
        digest,
    )


def _load_text_datasets(config: Dict[str, Any], subset_cfg: List[Dict[str, Any]]) -> Dataset:
    merged = None

    for dcfg in subset_cfg:
        ds = smart_load_dataset(
            dcfg["name"],
            dcfg.get("config"),
            split=dcfg.get("split", "train"),
            cache_dir=config["data_dir"],
        )

        ds = ds.map(
            lambda ex: {"text": get_text_field(ex, dcfg)},
            num_proc=config.get("num_proc", 1),
        )

        merged = ds if merged is None else concatenate_datasets([merged, ds])

    merged = merged.shuffle(seed=_get_dataset_seed(config))
    merged = _apply_max_samples(merged, config)
    return merged


# -------------------------
# Task dataset builders
# -------------------------

def load_supervised_task_dataset(config, datasets_cfg):
    rows = []

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
            if "label" not in ex or ex["label"] == -1:
                continue

            prompt, answer = formatter(ex)
            rows.append({"prompt": prompt, "answer": answer})

    rng = random.Random(_get_dataset_seed(config))
    rng.shuffle(rows)

    if config.get("max_train_samples"):
        rows = rows[: config["max_train_samples"]]

    return Dataset.from_list(rows)


def build_task_dataset(config, logger):
    task_cfg = config.get("datasets", {}).get("task", [])
    if not task_cfg:
        raise ValueError("No task datasets configured under datasets.task")

    frozen = _maybe_load_frozen_task_dataset(config, logger)
    if frozen is not None:
        return frozen

    logger.info("Building task dataset")

    # supervised classification / reasoning tasks
    if any("task_type" in d for d in task_cfg):
        ds = load_supervised_task_dataset(config, task_cfg)
        _maybe_export_task_dataset(ds, config, logger)
        return ds

    # fallback: plain text LM-style task dataset
    ds = _load_text_datasets(config, task_cfg)
    _maybe_export_task_dataset(ds, config, logger)
    return ds


# -------------------------
# Alignment dataset builders
# -------------------------

def build_alignment_dataset(config, logger):
    align_cfg = config.get("datasets", {}).get("alignment", [])
    if not align_cfg:
        raise ValueError("No alignment datasets configured under datasets.alignment")

    logger.info("Building alignment dataset")
    return _load_text_datasets(config, align_cfg)


# -------------------------
# Gate dataset builders
# -------------------------

def load_gate_dataset(config, datasets_cfg) -> Dataset:
    datasets = []

    for dcfg in datasets_cfg:
        source = dcfg.get("data_files", dcfg.get("name"))
        ds = smart_load_dataset(
            source,
            dcfg.get("config"),
            split=dcfg.get("split", "train"),
            cache_dir=config["data_dir"],
        )

        text_field = dcfg.get("text_field", "text")
        label_field = dcfg.get("label_field", "label")

        ds = ds.filter(
            lambda x: isinstance(x.get(text_field), str)
            and isinstance(x.get(label_field), (int, float))
        )

        ds = ds.map(
            lambda x: {
                "text": x[text_field],
                "label": float(x[label_field]),
            },
            remove_columns=ds.column_names,
        )

        datasets.append(ds)

    merged = concatenate_datasets(datasets).shuffle(seed=42)
    merged = _apply_max_samples(merged, config)
    return merged


def build_gate_dataset(config, logger):
    gate_cfg = config.get("datasets", {}).get("gate", [])
    if not gate_cfg:
        raise ValueError("No gate datasets configured under datasets.gate")

    logger.info("Building gate dataset")
    return load_gate_dataset(config, gate_cfg)


# -------------------------
# Tokenization
# -------------------------

def tokenize_supervised_dataset(ds, tokenizer, config):
    max_length = config["max_length"]

    def tok_fn(ex):
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["answer"]},
        ]

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


def tokenize_text_dataset(ds, tokenizer, config):
    """
    For plain text datasets:
      - returns input_ids + attention_mask
      - preserves label field if it exists (for gate training)
      - training code can use labels = input_ids when needed
    """
    def tok_fn(batch):
        enc = tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=config["max_length"],
        )
        output = {k: v for k, v in enc.items()}
        
        # Preserve label field if it exists (for gate training)
        if "label" in batch:
            output["label"] = torch.tensor(batch["label"], dtype=torch.float32)
        
        return output

    # Keep label column if it exists
    cols_to_remove = [c for c in ds.column_names if c != "label"]
    
    new_ds = ds.map(
        tok_fn,
        batched=True,
        remove_columns=cols_to_remove,
        num_proc=config.get("num_proc", 1),
    )
    new_ds.set_format(type="torch")
    return new_ds


def prepare_task_dataset(ds, tokenizer, config):
    """
    Returns tokenized task dataset + collator.
    """
    if "prompt" in ds.column_names and "answer" in ds.column_names:
        collator = SupervisedCollator(tokenizer)
        tok_ds = tokenize_supervised_dataset(ds, tokenizer, config)
    else:
        collator = None
        tok_ds = tokenize_text_dataset(ds, tokenizer, config)

    return tok_ds, collator


def prepare_alignment_dataset(ds, tokenizer, config):
    """
    Alignment datasets are plain text.
    """
    return tokenize_text_dataset(ds, tokenizer, config)
