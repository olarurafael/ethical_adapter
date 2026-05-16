"""
Shared text utilities for DRIFTCHECK.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable, List


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Unicode-normalize curly quotes, NBSP, and collapse whitespace."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_one_line(text: str) -> str:
    """Collapse a multi-line string into a single normalized line."""
    text = normalize_text(text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def looks_like_question_or_instruction(text: str) -> bool:
    """
    Minimal surface heuristic: must be >= 12 chars and >= 3 tokens.
    The paper uses one-liner prompts; this rejects degenerate leftovers.
    """
    if len(text) < 12:
        return False
    if len(text.split()) < 3:
        return False
    return True


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


def dedupe_records(records: Iterable[dict], text_key: str = "prompt") -> List[dict]:
    out: List[dict] = []
    seen: set = set()
    for rec in records:
        h = text_hash(rec[text_key])
        if h in seen:
            continue
        seen.add(h)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
