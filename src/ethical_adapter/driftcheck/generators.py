"""
Prompt extraction and dataset construction for DRIFTCHECK.

Paper specification (§2):
  10,000 one-liner prompts — 5,000 safe, 5,000 unsafe.

  Safe  : sampled from MMLU (all subjects, all splits).
  Unsafe: drawn from HH-RLHF, HateCheck, and OpenAI Moderation Prompts,
          covering violence, hate_speech, illegal_activity, explicit,
          privacy, and disinformation.

  Unsafe pool is stratified across categories before final sampling so that
  no single category dominates the evaluation.
"""
from __future__ import annotations

import random
from typing import Dict, List

from tqdm import tqdm

from ethical_adapter.driftcheck.sources import (
    load_hatecheck,
    load_hh_rlhf,
    load_mmlu,
    load_openai_moderation,
)
from ethical_adapter.driftcheck.utils import (
    dedupe_records,
    looks_like_question_or_instruction,
    make_one_line,
)

# Paper §2 — exactly 5 k / 5 k split.
SAFE_TARGET: int = 5_000
UNSAFE_TARGET: int = 5_000

# Paper Table 1 unsafe categories.
UNSAFE_CATEGORIES = (
    "violence",
    "hate_speech",
    "illegal_activity",
    "explicit",
    "privacy",
    "disinformation",
)


# ---------------------------------------------------------------------------
# Category classifier
# ---------------------------------------------------------------------------

# Each entry is (category_name, keyword_set).
# Keywords are matched case-insensitively against the lowercased prompt text.
_CATEGORY_KEYWORDS: List[tuple[str, tuple[str, ...]]] = [
    ("violence", (
        "kill", "bomb", "weapon", "injure", "explode", "sabotage",
        "harm someone", "murder", "assassin", "attack", "shoot",
        "stab", "torture", "massacre", "genocide",
    )),
    ("hate_speech", (
        "slur", "racist", "racism", "sexist", "homophob", "transphob",
        "immigrant", "minority", "religion", "nationality", "disabilities",
        "nigger", "faggot", "chink", "spic", "kike", "retard",
        "white supremac", "neo-nazi", "hate group",
    )),
    ("illegal_activity", (
        "hack", "phishing", "counterfeit", "launder", "black market",
        "steal", "fraud", "drug deal", "smuggl", "traffick",
        "identity theft", "credit card dump", "darknet",
    )),
    ("explicit", (
        "porn", "erotic", "sexual", "minors", "explicit content",
        "non-consensual", "nsfw", "nude", "naked", "rape",
    )),
    ("privacy", (
        "private", "home address", "phone number", "ssn", "social security",
        "password", "doxx", "personal information", "credit card number",
        "bank account",
    )),
    ("disinformation", (
        "fake news", "disinformation", "mislead voters", "propaganda",
        "conspiracy", "election fraud", "deep fake", "false narrative",
    )),
]


def categorize_unsafe_text(text: str) -> str:
    """
    Return the first matching DRIFTCHECK unsafe category, or 'other_unsafe'.
    Ordering of _CATEGORY_KEYWORDS determines priority on ties.
    """
    t = text.lower()
    for cat, keywords in _CATEGORY_KEYWORDS:
        if any(kw in t for kw in keywords):
            return cat
    return "other_unsafe"


# ---------------------------------------------------------------------------
# Safe extractor — MMLU
# ---------------------------------------------------------------------------

def extract_safe_from_mmlu() -> List[dict]:
    """
    Extract safe one-liner prompts from MMLU.
    Each MMLU question is a naturally safe knowledge query (paper §2).
    """
    ds = load_mmlu()
    records: List[dict] = []
    for split in ds.keys():
        for ex in tqdm(ds[split], desc=f"MMLU::{split}"):
            q = make_one_line(ex.get("question", ""))
            if not looks_like_question_or_instruction(q):
                continue
            records.append({
                "prompt": q,
                "label": "safe",
                "source": "mmlu",
                "source_split": split,
                "category": ex.get("subject", "general"),
            })
    return dedupe_records(records)


# ---------------------------------------------------------------------------
# Unsafe extractors
# ---------------------------------------------------------------------------

def _pick_user_turns_from_dialogue(text: str) -> List[str]:
    """
    Extract Human-turn utterances from HH-RLHF dialogue strings.
    Lines prefixed 'Human:' contain the user-side of conversations.
    """
    out: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("human:"):
            cand = make_one_line(line.split(":", 1)[1])
            if looks_like_question_or_instruction(cand):
                out.append(cand)
    return out


def extract_unsafe_from_hh_rlhf(max_records: int = 20_000) -> List[dict]:
    """
    Mine user turns from HH-RLHF that match a DRIFTCHECK unsafe category.
    Both 'chosen' and 'rejected' branches are scanned.
    """
    ds = load_hh_rlhf()
    records: List[dict] = []
    for split in ds.keys():
        for ex in tqdm(ds[split], desc=f"HH-RLHF::{split}"):
            for field in ("chosen", "rejected"):
                txt = ex.get(field)
                if not txt:
                    continue
                for cand in _pick_user_turns_from_dialogue(txt):
                    cat = categorize_unsafe_text(cand)
                    if cat == "other_unsafe":
                        continue
                    records.append({
                        "prompt": cand,
                        "label": "unsafe",
                        "source": "hh_rlhf",
                        "source_split": split,
                        "category": cat,
                    })
                    if len(records) >= max_records:
                        return dedupe_records(records)
    return dedupe_records(records)


def extract_unsafe_from_hatecheck() -> List[dict]:
    """
    Extract hateful test cases from HateCheck.
    Only rows labelled 'hateful' (label_gold ∈ {hateful, 1, true}) are kept.
    """
    ds = load_hatecheck()
    records: List[dict] = []
    for split in ds.keys():
        for ex in tqdm(ds[split], desc=f"HateCheck::{split}"):
            text = make_one_line(ex.get("test_case", ex.get("text", "")))
            if not looks_like_question_or_instruction(text):
                continue
            label = ex.get("label_gold", ex.get("label", ""))
            if str(label).lower() not in {"hateful", "1", "true"}:
                continue
            # Ensure the prompt reads as a complete utterance.
            if not text.endswith(("?", ".", "!")):
                text += "."
            records.append({
                "prompt": text,
                "label": "unsafe",
                "source": "hatecheck",
                "source_split": split,
                "category": "hate_speech",
            })
    return dedupe_records(records)


def extract_unsafe_from_openai_moderation() -> List[dict]:
    """
    Extract flagged prompts from the OpenAI moderation evaluation dataset.
    A prompt is retained when any category flag is True.
    """
    ds = load_openai_moderation()
    records: List[dict] = []
    for ex in tqdm(ds["train"], desc="OpenAI Moderation"):
        text = make_one_line(
            ex.get("prompt", ex.get("text", ex.get("input", "")))
        )
        if not looks_like_question_or_instruction(text):
            continue
        cats = ex.get("categories", {})
        if isinstance(cats, dict):
            flagged = any(bool(v) for v in cats.values())
        else:
            flagged = bool(ex.get("flagged", True))
        if not flagged:
            continue
        records.append({
            "prompt": text,
            "label": "unsafe",
            "source": "openai_moderation",
            "source_split": "train",
            "category": categorize_unsafe_text(text),
        })
    return dedupe_records(records)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def balanced_sample(records: List[dict], n: int, seed: int) -> List[dict]:
    """
    Random sample of exactly n records.
    Raises if the pool is too small.
    """
    rng = random.Random(seed)
    if len(records) < n:
        raise ValueError(
            f"Need {n} examples but pool only has {len(records)}. "
            "Consider increasing max_records in extract_unsafe_from_hh_rlhf()."
        )
    pool = list(records)
    rng.shuffle(pool)
    return pool[:n]


def stratified_unsafe_sample(records: List[dict], n: int, seed: int) -> List[dict]:
    """
    Stratified sample across DRIFTCHECK unsafe categories so that no single
    category monopolises the evaluation set.

    Algorithm:
      1. Divide records by category.
      2. Allocate floor(n / |categories|) slots per category.
      3. Fill remaining slots from the global leftover pool (shuffled).
    """
    rng = random.Random(seed)
    by_cat: Dict[str, List[dict]] = {}
    for rec in records:
        by_cat.setdefault(rec["category"], []).append(rec)

    categories = sorted(by_cat.keys())
    per_cat = max(1, n // len(categories))

    sampled: List[dict] = []
    sampled_ids: set = set()
    for cat in categories:
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        for rec in pool[:per_cat]:
            sampled.append(rec)
            sampled_ids.add(id(rec))

    # Top-up from leftovers if we are short of n.
    if len(sampled) < n:
        leftovers = [
            rec
            for cat in categories
            for rec in by_cat[cat]
            if id(rec) not in sampled_ids
        ]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: n - len(sampled)])

    return sampled[:n]
