"""Dataset construction and tokenization for adapter/gate training.

Public entry points:
  - build_task_dataset: task data for adapter training
  - build_alignment_dataset: plain-text alignment data for AlignGuard
  - build_gate_dataset / load_gate_dataset: labeled gate training data
  - prepare_task_dataset / prepare_alignment_dataset: tokenization helpers
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.nn.utils.rnn import pad_sequence

from ethical_adapter.task_formatting import FORMATTERS


PROMPT_ANSWER_COLUMNS = {"prompt", "answer"}


@dataclass
class SupervisedCollator:
    tokenizer: Any

    def __call__(self, features):
        input_ids = pad_sequence(
            [feature["input_ids"] for feature in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [feature["attention_mask"] for feature in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_dataset_source(name, config_name=None, split=None, **kwargs):
    """Load either a local JSON/JSONL file or a Hugging Face dataset."""
    if Path(name).exists():
        return load_dataset("json", data_files=name, split=split)

    return load_dataset(name, config_name, split=split, **kwargs)


def extract_training_text(
    example: dict[str, Any], dataset_config: dict[str, Any]
) -> str:
    configured_field = dataset_config.get("text_field")
    if (
        configured_field
        and configured_field in example
        and isinstance(example[configured_field], str)
    ):
        return example[configured_field]

    if "prompt" in example and "chosen" in example:
        prompt = example["prompt"]
        chosen = example["chosen"]
        if isinstance(prompt, str) and isinstance(chosen, str):
            return prompt + "\n" + chosen

    for field in ("text", "content", "comment_text", "prompt"):
        if field in example and isinstance(example[field], str):
            return example[field]

    return "\n".join(value for value in example.values() if isinstance(value, str))


def _dataset_seed(config: dict[str, Any]) -> int:
    return int(config.get("dataset_seed", 42))


def _limit_rows(dataset: Dataset, config: dict[str, Any]) -> Dataset:
    max_samples = config.get("max_train_samples")
    if not max_samples:
        return dataset

    limit = min(max_samples, len(dataset))
    logging.info("Using only %d samples out of %d.", limit, len(dataset))
    return dataset.select(range(limit))


def _write_jsonl_dataset(dataset: Dataset, path: str) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def _load_prebuilt_task_dataset(config: dict[str, Any], logger) -> Dataset | None:
    dataset_path = config.get("frozen_task_dataset_path")
    if not dataset_path:
        return None

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return None

    logger.info("Loading prebuilt task dataset from %s", dataset_path)
    dataset = load_dataset_source(str(dataset_path), split="train")
    if not PROMPT_ANSWER_COLUMNS.issubset(dataset.column_names):
        raise ValueError(
            f"Prebuilt task dataset {dataset_path} must contain columns "
            f"{sorted(PROMPT_ANSWER_COLUMNS)}; got {dataset.column_names}"
        )

    logger.info("Loaded %d prebuilt task rows.", len(dataset))
    return dataset


def _export_task_dataset_if_requested(
    dataset: Dataset,
    config: dict[str, Any],
    logger,
) -> None:
    export_path = config.get("frozen_task_dataset_path") or config.get(
        "export_task_dataset_path"
    )
    if not export_path:
        return

    digest = _write_jsonl_dataset(dataset, export_path)
    logger.info(
        "Exported task dataset to %s (%d rows, sha256=%s)",
        export_path,
        len(dataset),
        digest,
    )


def _build_plain_text_dataset(
    config: dict[str, Any],
    dataset_configs: list[dict[str, Any]],
) -> Dataset:
    datasets = []

    for dataset_config in dataset_configs:
        dataset = load_dataset_source(
            dataset_config["name"],
            dataset_config.get("config"),
            split=dataset_config.get("split", "train"),
            cache_dir=config["data_dir"],
        )
        dataset = dataset.map(
            lambda example: {"text": extract_training_text(example, dataset_config)},
            num_proc=config.get("num_proc", 1),
        )
        datasets.append(dataset)

    merged = concatenate_datasets(datasets)
    merged = merged.shuffle(seed=_dataset_seed(config))
    return _limit_rows(merged, config)


def _build_prompt_answer_task_dataset(
    config: dict[str, Any],
    dataset_configs: list[dict[str, Any]],
) -> Dataset:
    rows = []

    for dataset_config in dataset_configs:
        task = dataset_config.get("task_type")
        if task not in FORMATTERS:
            raise ValueError(f"Unknown task_type: {task}")

        dataset = load_dataset_source(
            dataset_config["name"],
            dataset_config.get("config"),
            split=dataset_config.get("split", "train"),
            cache_dir=config["data_dir"],
        )
        formatter = FORMATTERS[task]

        for example in dataset:
            if "label" not in example or example["label"] == -1:
                continue

            prompt, answer = formatter(example)
            rows.append({"prompt": prompt, "answer": answer})

    random.Random(_dataset_seed(config)).shuffle(rows)
    max_samples = config.get("max_train_samples")
    if max_samples:
        rows = rows[:max_samples]

    return Dataset.from_list(rows)


def build_task_dataset(config, logger):
    task_configs = config.get("datasets", {}).get("task", [])
    if not task_configs:
        raise ValueError("No task datasets configured under datasets.task")

    prebuilt_dataset = _load_prebuilt_task_dataset(config, logger)
    if prebuilt_dataset is not None:
        return prebuilt_dataset

    logger.info("Building task dataset")
    if any("task_type" in dataset_config for dataset_config in task_configs):
        dataset = _build_prompt_answer_task_dataset(config, task_configs)
    else:
        dataset = _build_plain_text_dataset(config, task_configs)

    _export_task_dataset_if_requested(dataset, config, logger)
    return dataset


def build_alignment_dataset(config, logger):
    alignment_configs = config.get("datasets", {}).get("alignment", [])
    if not alignment_configs:
        raise ValueError("No alignment datasets configured under datasets.alignment")

    logger.info("Building alignment dataset")
    return _build_plain_text_dataset(config, alignment_configs)


def load_gate_dataset(config, dataset_configs) -> Dataset:
    datasets = []

    for dataset_config in dataset_configs:
        source = dataset_config.get("data_files", dataset_config.get("name"))
        text_field = dataset_config.get("text_field", "text")
        label_field = dataset_config.get("label_field", "label")

        dataset = load_dataset_source(
            source,
            dataset_config.get("config"),
            split=dataset_config.get("split", "train"),
            cache_dir=config["data_dir"],
        )
        dataset = dataset.filter(
            lambda row: isinstance(row.get(text_field), str)
            and isinstance(row.get(label_field), (int, float))
        )
        dataset = dataset.map(
            lambda row: {
                "text": row[text_field],
                "label": float(row[label_field]),
            },
            remove_columns=dataset.column_names,
        )
        datasets.append(dataset)

    merged = concatenate_datasets(datasets).shuffle(seed=_dataset_seed(config))
    return _limit_rows(merged, config)


def build_gate_dataset(config, logger):
    gate_configs = config.get("datasets", {}).get("gate", [])
    if not gate_configs:
        raise ValueError("No gate datasets configured under datasets.gate")

    logger.info("Building gate dataset")
    return load_gate_dataset(config, gate_configs)


def tokenize_supervised_dataset(dataset, tokenizer, config):
    max_length = config["max_length"]

    def tokenize_example(example):
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["answer"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = full_tokens["input_ids"]
        labels = input_ids.copy()
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    tokenized = dataset.map(tokenize_example, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")
    return tokenized


def tokenize_text_dataset(dataset, tokenizer, config):
    def tokenize_batch(batch):
        tokens = tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=config["max_length"],
        )
        output = dict(tokens)
        if "label" in batch:
            output["label"] = torch.tensor(batch["label"], dtype=torch.float32)
        return output

    columns_to_remove = [name for name in dataset.column_names if name != "label"]
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
        num_proc=config.get("num_proc", 1),
    )
    tokenized.set_format(type="torch")
    return tokenized


def prepare_task_dataset(dataset, tokenizer, config):
    if PROMPT_ANSWER_COLUMNS.issubset(dataset.column_names):
        tokenized = tokenize_supervised_dataset(dataset, tokenizer, config)
        return tokenized, SupervisedCollator(tokenizer)

    return tokenize_text_dataset(dataset, tokenizer, config), None


def prepare_alignment_dataset(dataset, tokenizer, config):
    return tokenize_text_dataset(dataset, tokenizer, config)
