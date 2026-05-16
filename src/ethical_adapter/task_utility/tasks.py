from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from datasets import load_dataset


Formatter = Callable[[dict], tuple[str, str]]


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset_name: str
    config_name: str | None
    split: str
    choices: tuple[str, ...]
    formatter: Formatter


def format_boolq(ex: dict) -> tuple[str, str]:
    prompt = (
        "Read the passage and answer the question with yes or no.\n\n"
        f"Passage: {ex['passage']}\n"
        f"Question: {ex['question']}\n\n"
        "Answer:"
    )
    return prompt, ("yes" if bool(ex["label"]) else "no")


def format_sst2(ex: dict) -> tuple[str, str]:
    prompt = (
        "Classify the sentiment of the sentence as positive or negative.\n\n"
        f"Sentence: {ex['sentence']}\n\n"
        "Answer:"
    )
    return prompt, ("positive" if int(ex["label"]) == 1 else "negative")


def format_qqp(ex: dict) -> tuple[str, str]:
    prompt = (
        "Are the following two questions duplicates? Answer yes or no.\n\n"
        f"Question 1: {ex['question1']}\n"
        f"Question 2: {ex['question2']}\n\n"
        "Answer:"
    )
    return prompt, ("yes" if int(ex["label"]) == 1 else "no")


def format_qnli(ex: dict) -> tuple[str, str]:
    prompt = (
        "Does the sentence answer the question? Answer yes or no.\n\n"
        f"Question: {ex['question']}\n"
        f"Sentence: {ex['sentence']}\n\n"
        "Answer:"
    )
    return prompt, ("yes" if int(ex["label"]) == 0 else "no")


def format_mrpc(ex: dict) -> tuple[str, str]:
    prompt = (
        "Do the following two sentences have the same meaning? Answer yes or no.\n\n"
        f"Sentence 1: {ex['sentence1']}\n"
        f"Sentence 2: {ex['sentence2']}\n\n"
        "Answer:"
    )
    return prompt, ("yes" if int(ex["label"]) == 1 else "no")


def format_mnli(ex: dict) -> tuple[str, str]:
    label_map = {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    }
    prompt = (
        "Determine the relationship between the premise and hypothesis. "
        "Answer entailment, neutral, or contradiction.\n\n"
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n\n"
        "Answer:"
    )
    return prompt, label_map[int(ex["label"])]


def format_wic(ex: dict) -> tuple[str, str]:
    prompt = (
        "Does the word have the same meaning in both sentences? "
        "Answer yes or no.\n\n"
        f"Word: {ex['word']}\n"
        f"Sentence 1: {ex['sentence1']}\n"
        f"Sentence 2: {ex['sentence2']}\n\n"
        "Answer:"
    )
    return prompt, ("yes" if int(ex["label"]) == 1 else "no")


def format_multirc(ex: dict) -> tuple[str, str]:
    prompt = (
        "Given the paragraph, decide whether the candidate answer is correct. "
        "Answer yes or no.\n\n"
        f"Paragraph: {ex['paragraph']}\n"
        f"Question: {ex['question']}\n"
        f"Candidate answer: {ex['answer']}\n\n"
        "Answer:"
    )
    return prompt, ("yes" if int(ex["label"]) == 1 else "no")


TASK_SPECS: dict[str, TaskSpec] = {
    "boolq": TaskSpec(
        task="boolq",
        dataset_name="super_glue",
        config_name="boolq",
        split="validation",
        choices=("yes", "no"),
        formatter=format_boolq,
    ),
    "sst2": TaskSpec(
        task="sst2",
        dataset_name="glue",
        config_name="sst2",
        split="validation",
        choices=("negative", "positive"),
        formatter=format_sst2,
    ),
    "qqp": TaskSpec(
        task="qqp",
        dataset_name="glue",
        config_name="qqp",
        split="validation",
        choices=("no", "yes"),
        formatter=format_qqp,
    ),
    "qnli": TaskSpec(
        task="qnli",
        dataset_name="glue",
        config_name="qnli",
        split="validation",
        choices=("no", "yes"),
        formatter=format_qnli,
    ),
    "mrpc": TaskSpec(
        task="mrpc",
        dataset_name="glue",
        config_name="mrpc",
        split="validation",
        choices=("no", "yes"),
        formatter=format_mrpc,
    ),
    "mnli": TaskSpec(
        task="mnli",
        dataset_name="glue",
        config_name="mnli",
        split="validation_matched",
        choices=("entailment", "neutral", "contradiction"),
        formatter=format_mnli,
    ),
    "wic": TaskSpec(
        task="wic",
        dataset_name="super_glue",
        config_name="wic",
        split="validation",
        choices=("no", "yes"),
        formatter=format_wic,
    ),
    "multirc": TaskSpec(
        task="multirc",
        dataset_name="super_glue",
        config_name="multirc",
        split="validation",
        choices=("no", "yes"),
        formatter=format_multirc,
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_SPECS))
        raise ValueError(
            f"Unsupported task '{task}'. Supported tasks: {supported}."
        ) from exc


def load_task_dataframe(
    task: str,
    limit: int | None = None,
    seed: int = 0,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    spec = get_task_spec(task)
    ds = load_dataset(
        spec.dataset_name,
        spec.config_name,
        split=spec.split,
        cache_dir=cache_dir,
    )
    ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    for idx, ex in enumerate(ds):
        if "label" in ex and int(ex["label"]) == -1:
            continue
        prompt, gold_answer = spec.formatter(ex)
        rows.append(
            {
                "id": f"{task}_{idx:05d}",
                "prompt": prompt,
                "gold_answer": gold_answer,
            }
        )

    return pd.DataFrame(rows)
