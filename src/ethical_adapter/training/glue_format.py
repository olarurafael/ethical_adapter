from typing import Dict, Tuple

# Canonical labels — keep these stable for evaluation
LABELS = {
    "sst2": {0: "negative", 1: "positive"},
    "qqp": {0: "no", 1: "yes"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "wic": {0: "no", 1: "yes"},
    "multirc": {0: "no", 1: "yes"},
}


def format_sst2(ex: Dict) -> Tuple[str, str]:
    sent = ex["sentence"]
    y = LABELS["sst2"][int(ex["label"])]

    prompt = (
        "Classify the sentiment of the sentence as positive or negative.\n\n"
        f"Sentence: {sent}\n\nAnswer:"
    )
    return prompt, y


def format_qqp(ex: Dict) -> Tuple[str, str]:
    q1 = ex["question1"]
    q2 = ex["question2"]
    y = LABELS["qqp"][int(ex["label"])]

    prompt = (
        "Are the following two questions duplicates? Answer yes or no.\n\n"
        f"Question 1: {q1}\n"
        f"Question 2: {q2}\n\n"
        "Answer:"
    )
    return prompt, y


def format_mnli(ex: Dict) -> Tuple[str, str]:
    prem = ex["premise"]
    hyp = ex["hypothesis"]
    y = LABELS["mnli"][int(ex["label"])]
    

    prompt = (
        "Determine the relationship between the premise and hypothesis. "
        "Answer entailment, neutral, or contradiction.\n\n"
        f"Premise: {prem}\n"
        f"Hypothesis: {hyp}\n\n"
        "Answer:"
    )
    return prompt, y


def format_wic(ex: Dict) -> Tuple[str, str]:
    word = ex["word"]
    s1 = ex["sentence1"]
    s2 = ex["sentence2"]
    y = LABELS["wic"][int(ex["label"])]

    prompt = (
        "Does the word have the same meaning in both sentences? "
        "Answer yes or no.\n\n"
        f"Word: {word}\n"
        f"Sentence 1: {s1}\n"
        f"Sentence 2: {s2}\n\n"
        "Answer:"
    )
    return prompt, y


def format_multirc(ex: Dict) -> Tuple[str, str]:
    para = ex["paragraph"]
    question = ex["question"]
    answer = ex["answer"]
    y = LABELS["multirc"][int(ex["label"])]

    prompt = (
        "Given the paragraph, decide whether the candidate answer is correct. "
        "Answer yes or no.\n\n"
        f"Paragraph: {para}\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n\n"
        "Answer:"
    )
    return prompt, y


FORMATTERS = {
    "sst2": format_sst2,
    "qqp": format_qqp,
    "mnli": format_mnli,
    "wic": format_wic,
    "multirc": format_multirc,
}
