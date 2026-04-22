# # src/ethical_adapter/training/data.py
# import logging
# import torch
# import random
# import os
# from dataclasses import dataclass
# from torch.nn.utils.rnn import pad_sequence
# import torch

# from typing import Dict, List, Any

# from datasets import (
#     load_dataset,
#     concatenate_datasets,
#     Dataset,
#     load_from_disk,
# )

# from ethical_adapter.training.glue_format import FORMATTERS

# @dataclass
# class SupervisedCollator:
#     tokenizer: any

#     def __call__(self, features):
#         input_ids = [f["input_ids"] for f in features]
#         attention_mask = [f["attention_mask"] for f in features]
#         labels = [f["labels"] for f in features]

#         input_ids = pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id,
#         )

#         attention_mask = pad_sequence(
#             attention_mask,
#             batch_first=True,
#             padding_value=0,
#         )

#         labels = pad_sequence(
#             labels,
#             batch_first=True,
#             padding_value=-100,
#         )

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels,
#         }
# from pathlib import Path

# def smart_load_dataset(name, config_name=None, split=None, **kwargs):
#     """
#     Automatically loads either:
#     - a local dataset saved with save_to_disk() using load_from_disk, or
#     - a standard HF dataset via load_dataset().
#     """
#     # # If local directory containing dataset_info.json → load_from_disk
#     # if os.path.isdir(name) and os.path.exists(os.path.join(name, "dataset_info.json")):
#     #     ds = load_from_disk(name)
#     #     return ds

#     # # Otherwise fall back to the normal HF loader
#     # return load_dataset(name, config_name, split=split, **kwargs)

#     # If it's a file path, treat it as JSON
#     if Path(name).exists():
#         return load_dataset(
#             "json",
#             data_files=name,
#             split=split,
#         )

#     return load_dataset(name, config_name, split=split, **kwargs)


# def get_text_field(ex: Dict[str, Any], dcfg: Dict[str, Any]) -> str:
#     # custom field override
#     tf = dcfg.get("text_field")
#     if tf and tf in ex and isinstance(ex[tf], str):
#         return ex[tf]

#     # HH-RLHF style (prompt + chosen)
#     if "prompt" in ex and "chosen" in ex:
#         p, c = ex["prompt"], ex["chosen"]
#         if isinstance(p, str) and isinstance(c, str):
#             return p + "\n" + c

#     # common fields
#     for key in ("text", "content", "comment_text", "prompt"):
#         if key in ex and isinstance(ex[key], str):
#             return ex[key]

#     # fallback: concatenate all string fields
#     pieces = [v for v in ex.values() if isinstance(v, str)]
#     return "\n".join(pieces)


# def _load_and_merge(
#     config: Dict[str, Any], subset_cfg: List[Dict[str, Any]]
# ) -> Dataset:
#     merged = None
#     for dcfg in subset_cfg:
#         ds = smart_load_dataset(
#             dcfg["name"],
#             dcfg.get("config"),
#             split=dcfg.get("split"),
#             cache_dir=config["data_dir"],
#         )

#         # KEEP ORIGINAL COLUMNS + TEXT
#         ds = ds.map(
#             lambda ex: {"text": get_text_field(ex, dcfg)},
#             num_proc=config.get("num_proc", 1),
#         )

#         merged = ds if merged is None else concatenate_datasets([merged, ds])

#     merged = merged.shuffle(seed=42)

#     if config.get("max_train_samples"):
#         limit = min(config["max_train_samples"], len(merged))
#         logging.info(
#             "Using only %d samples out of %d for training.", limit, len(merged)
#         )
#         merged = merged.select(range(limit))

#     return merged


# def extract_toxicity_field(ex, field):
#     val = ex.get(field)

#     # RTP-style nested structure
#     if isinstance(val, dict):
#         tox = val.get("toxicity")
#         if isinstance(tox, (int, float)):
#             return float(tox)
#         return None

#     # Flat structure
#     if isinstance(val, (int, float)):
#         return float(val)

#     return None


# def load_alignment_phase(config, datasets_cfg):
#     rows = []
#     max_len = config.get("max_train_samples")

#     for dcfg in datasets_cfg:
#         ds = smart_load_dataset(
#             dcfg["name"],
#             dcfg.get("config"),
#             split=dcfg.get("split", "train"),
#             cache_dir=config["data_dir"],
#             trust_remote_code=True,
#         )

#         text_field = dcfg.get("text_field", "text")
#         tox_field = dcfg.get("toxicity_field", "toxicity")

#         for ex in ds:
#             # ---- Handle RealToxicityPrompts nested samples ----
#             val = ex.get(text_field)
#             if isinstance(val, dict):
#                 text = val.get("text")
#                 tox = extract_toxicity_field(val, "toxicity")
#                 if isinstance(text, str) and tox is not None:
#                     rows.append({"text": text, "toxicity": tox})
#                 continue

#             # ---- Standard flat datasets ----
#             if text_field not in ex or tox_field not in ex:
#                 continue

#             text = ex[text_field]
#             tox = extract_toxicity_field(ex, tox_field)
#             if isinstance(text, str) and tox is not None:
#                 rows.append({"text": text, "toxicity": tox})

#     random.shuffle(rows)
#     if max_len:
#         rows = rows[:max_len]

#     return Dataset.from_list(rows)


# def build_phase_dataset(config, logger, phase):
#     all_cfg = config["datasets"]

#     def by_role(role):
#         return [d for d in all_cfg if d.get("role") == role]

#     logger.info(f"Building dataset for phase: {phase}")

#     # ---- Phase 1: normal text datasets ----
#     if phase == "adapters":
#         task_cfg = by_role("task")

#         # if datasets specify task_type → supervised GLUE mode
#         if any("task_type" in d for d in task_cfg):
#             return load_supervised_task_dataset(config, task_cfg)

#         used = task_cfg if task_cfg else all_cfg
#         return _load_and_merge(config, used)
    
#     if phase == "gate_alignment":
#         align_cfg = by_role("alignment")
#         if not align_cfg:
#             raise ValueError("No datasets with role='alignment' configured.")
#         return load_gate_alignment_dataset(config, align_cfg)

#     raise ValueError(f"Unknown phase: {phase}")


# def tokenize_supervised_dataset(ds, tokenizer, config):
#     max_length = config["max_length"]

#     def tok_fn(ex):
#         messages = [
#             {"role": "user", "content": ex["prompt"]},
#             {"role": "assistant", "content": ex["answer"]},
#         ]

#         # Render chat text first (string)
#         full_text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=False,
#         )

#         prompt_text = tokenizer.apply_chat_template(
#             [{"role": "user", "content": ex["prompt"]}],
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         # Now tokenize normally → returns lists of ints
#         full_enc = tokenizer(
#             full_text,
#             truncation=True,
#             max_length=max_length,
#             padding=False,
#         )

#         prompt_enc = tokenizer(
#             prompt_text,
#             truncation=True,
#             max_length=max_length,
#             padding=False,
#         )

#         input_ids = full_enc["input_ids"]
#         labels = input_ids.copy()

#         cut = min(len(prompt_enc["input_ids"]), len(labels))
#         for i in range(cut):
#             labels[i] = -100

#         attention_mask = [1] * len(input_ids)

#         return {
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#             "labels": torch.tensor(labels, dtype=torch.long),
#         }

#     new_ds = ds.map(tok_fn, remove_columns=ds.column_names)
#     new_ds.set_format(type="torch")
#     return new_ds

#     max_length = config["max_length"]

#     def tok_fn(ex):
#         messages = [
#             {"role": "user", "content": ex["prompt"]},
#             {"role": "assistant", "content": ex["answer"]},
#         ]

#         full_ids = tokenizer.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=False,
#         )

#         prompt_ids = tokenizer.apply_chat_template(
#             [{"role": "user", "content": ex["prompt"]}],
#             tokenize=True,
#             add_generation_prompt=True,
#         )

#         input_ids = full_ids[:max_length]
#         labels = input_ids.copy()

#         cut = min(len(prompt_ids), len(labels))
#         for i in range(cut):
#             labels[i] = -100

#         attention_mask = [1] * len(input_ids)

#         return {
#             "input_ids": torch.tensor(input_ids),
#             "attention_mask": torch.tensor(attention_mask),
#             "labels": torch.tensor(labels),
#         }

#     new_ds = ds.map(tok_fn, remove_columns=ds.column_names)
#     new_ds.set_format(type="torch")
#     return new_ds


# def tokenize_dataset(ds, tokenizer, config):
#     # Automatically infer numeric fields (toxicity, score, labels, etc.)
#     numeric_cols = [
#         col
#         for col in ds.column_names
#         if col != "text"
#         and all(isinstance(x, (int, float)) or x is None for x in ds[col])
#     ]

#     def tok_fn(batch):
#         enc = tokenizer(
#             batch["text"],
#             return_tensors="pt",
#             truncation=True,
#             padding="max_length",
#             max_length=config["max_length"],
#         )

#         out = {k: v for k, v in enc.items()}

#         # Preserve all numeric fields as float32 tensors
#         for col in numeric_cols:
#             out[col] = torch.tensor(batch[col], dtype=torch.float32)

#         return out

#     new_ds = ds.map(
#         tok_fn,
#         batched=True,
#         remove_columns=ds.column_names,
#         num_proc=config.get("num_proc", 1),
#     )

#     new_ds.set_format(type="torch")
#     return new_ds

# def load_supervised_task_dataset(config, datasets_cfg):
#     rows = []
#     max_len = config.get("max_train_samples")

#     for dcfg in datasets_cfg:
#         name = dcfg["name"]
#         task = dcfg.get("task_type")

#         if task not in FORMATTERS:
#             raise ValueError(f"Unknown task_type: {task}")

#         formatter = FORMATTERS[task]

#         ds = smart_load_dataset(
#             name,
#             dcfg.get("config"),
#             split=dcfg.get("split", "train"),
#             cache_dir=config["data_dir"],
#         )

#         for ex in ds:
#             # skip missing labels (e.g. MNLI test)
#             if "label" not in ex or ex["label"] == -1:
#                 continue

#             prompt, answer = formatter(ex)

#             rows.append(
#                 {
#                     "prompt": prompt,
#                     "answer": answer,
#                 }
#             )

#     random.shuffle(rows)

#     if max_len:
#         rows = rows[:max_len]

#     return Dataset.from_list(rows)


# def load_gate_alignment_dataset(config, datasets_cfg) -> Dataset:
#     """
#     Load datasets for alignment gate training.

#     Expected fields:
#         text: str
#         label: float (0 = misaligned, 1 = aligned)

#     Returns:
#         Dataset with:
#             text
#             label
#     """
#     datasets = []

#     for dcfg in datasets_cfg:
#         ds = smart_load_dataset(
#             dcfg["data_files"],
#             dcfg.get("config"),
#             split=dcfg.get("split", "train"),
#             cache_dir=config["data_dir"],
#         )

#         text_field = dcfg.get("text_field", "text")
#         label_field = dcfg.get("label_field", "label")

#         ds = ds.filter(
#             lambda x: isinstance(x.get(text_field), str)
#             and isinstance(x.get(label_field), (int, float))
#         )

#         ds = ds.map(
#             lambda x: {
#                 "text": x[text_field],
#                 "label": float(x[label_field]),
#             },
#             remove_columns=ds.column_names,
#         )

#         datasets.append(ds)

#     merged = concatenate_datasets(datasets).shuffle(seed=42)

#     if config.get("max_train_samples"):
#         merged = merged.select(range(config["max_train_samples"]))

#     return merged
# #What I would change conceptually

# # Instead of this:

# # build_phase_dataset(config, logger, phase="adapters")
# # build_phase_dataset(config, logger, phase="alignment")
# # build_phase_dataset(config, logger, phase="gate_alignment")

# # use these:

# # build_task_dataset(config, logger)
# # build_alignment_dataset(config, logger)
# # build_gate_dataset(config, logger)

# # That makes the call sites much clearer and removes the weird legacy branch logic.

import logging
import os
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

    merged = merged.shuffle(seed=42)
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

    random.shuffle(rows)

    if config.get("max_train_samples"):
        rows = rows[: config["max_train_samples"]]

    return Dataset.from_list(rows)


def build_task_dataset(config, logger):
    task_cfg = config.get("datasets", {}).get("task", [])
    if not task_cfg:
        raise ValueError("No task datasets configured under datasets.task")

    logger.info("Building task dataset")

    # supervised classification / reasoning tasks
    if any("task_type" in d for d in task_cfg):
        return load_supervised_task_dataset(config, task_cfg)

    # fallback: plain text LM-style task dataset
    return _load_text_datasets(config, task_cfg)


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