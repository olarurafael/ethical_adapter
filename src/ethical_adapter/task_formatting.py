from __future__ import annotations

from typing import Any, Callable


PromptFormatter = Callable[[dict[str, Any]], tuple[str, str]]


# Canonical labels. Keep these stable because training and eval both depend on
# exact answer strings.
LABELS: dict[str, dict[int, str]] = {
    "boolq": {0: "no", 1: "yes"},
    "sst2": {0: "negative", 1: "positive"},
    "qqp": {0: "no", 1: "yes"},
    "qnli": {0: "yes", 1: "no"},
    "mrpc": {0: "no", 1: "yes"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "wic": {0: "no", 1: "yes"},
    "multirc": {0: "no", 1: "yes"},
}


TASK_CHOICES: dict[str, tuple[str, ...]] = {
    "boolq": ("yes", "no"),
    "sst2": ("negative", "positive"),
    "qqp": ("no", "yes"),
    "qnli": ("no", "yes"),
    "mrpc": ("no", "yes"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "wic": ("no", "yes"),
    "multirc": ("no", "yes"),
}


def format_boolq(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Read the passage and answer the question with yes or no.\n\n"
        f"Passage: {example['passage']}\n"
        f"Question: {example['question']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["boolq"][int(example["label"])]


def format_sst2(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Classify the sentiment of the sentence as positive or negative.\n\n"
        f"Sentence: {example['sentence']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["sst2"][int(example["label"])]


def format_qqp(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Are the following two questions duplicates? Answer yes or no.\n\n"
        f"Question 1: {example['question1']}\n"
        f"Question 2: {example['question2']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["qqp"][int(example["label"])]


def format_qnli(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Does the sentence answer the question? Answer yes or no.\n\n"
        f"Question: {example['question']}\n"
        f"Sentence: {example['sentence']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["qnli"][int(example["label"])]


def format_mrpc(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Do the following two sentences have the same meaning? Answer yes or no.\n\n"
        f"Sentence 1: {example['sentence1']}\n"
        f"Sentence 2: {example['sentence2']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["mrpc"][int(example["label"])]


def format_mnli(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Determine the relationship between the premise and hypothesis. "
        "Answer entailment, neutral, or contradiction.\n\n"
        f"Premise: {example['premise']}\n"
        f"Hypothesis: {example['hypothesis']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["mnli"][int(example["label"])]


def format_wic(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Does the word have the same meaning in both sentences? "
        "Answer yes or no.\n\n"
        f"Word: {example['word']}\n"
        f"Sentence 1: {example['sentence1']}\n"
        f"Sentence 2: {example['sentence2']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["wic"][int(example["label"])]


def format_multirc(example: dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Given the paragraph, decide whether the candidate answer is correct. "
        "Answer yes or no.\n\n"
        f"Paragraph: {example['paragraph']}\n"
        f"Question: {example['question']}\n"
        f"Candidate answer: {example['answer']}\n\n"
        "Answer:"
    )
    return prompt, LABELS["multirc"][int(example["label"])]


FORMATTERS: dict[str, PromptFormatter] = {
    "boolq": format_boolq,
    "sst2": format_sst2,
    "qqp": format_qqp,
    "qnli": format_qnli,
    "mrpc": format_mrpc,
    "mnli": format_mnli,
    "wic": format_wic,
    "multirc": format_multirc,
}
