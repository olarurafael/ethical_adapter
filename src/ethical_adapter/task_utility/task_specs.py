from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset

from ethical_adapter.task_formatting import FORMATTERS, TASK_CHOICES, PromptFormatter


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset_name: str
    config_name: str | None
    split: str
    choices: tuple[str, ...]
    formatter: PromptFormatter


def _spec(
    task: str,
    dataset_name: str,
    config_name: str | None,
    split: str,
) -> TaskSpec:
    return TaskSpec(
        task=task,
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        choices=TASK_CHOICES[task],
        formatter=FORMATTERS[task],
    )


TASK_SPECS: dict[str, TaskSpec] = {
    "boolq": _spec("boolq", "super_glue", "boolq", "validation"),
    "sst2": _spec("sst2", "glue", "sst2", "validation"),
    "qqp": _spec("qqp", "glue", "qqp", "validation"),
    "qnli": _spec("qnli", "glue", "qnli", "validation"),
    "mrpc": _spec("mrpc", "glue", "mrpc", "validation"),
    "mnli": _spec("mnli", "glue", "mnli", "validation_matched"),
    "wic": _spec("wic", "super_glue", "wic", "validation"),
    "multirc": _spec("multirc", "super_glue", "multirc", "validation"),
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
