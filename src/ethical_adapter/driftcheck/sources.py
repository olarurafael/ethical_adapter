"""
Dataset loaders for DRIFTCHECK source corpora.

Paper (§2):
  - Safe prompts  → MMLU (Hendrycks et al., 2021)
  - Unsafe prompts → Anthropic HH-RLHF, OpenAI Moderation Prompts, HateCheck
"""
from __future__ import annotations

from datasets import load_dataset


def load_mmlu():
    """MMLU 'all' config via the cais HF mirror."""
    return load_dataset("cais/mmlu", "all")


def load_hh_rlhf():
    """Anthropic HH-RLHF (helpful + harmless dialogue pairs)."""
    return load_dataset("Anthropic/hh-rlhf")


def load_hatecheck():
    """HateCheck functional test cases for hate-speech detection."""
    return load_dataset("Paul/hatecheck")


def load_openai_moderation():
    """
    OpenAI moderation evaluation dataset (samples-1680.jsonl).
    Official release: github.com/openai/moderation-api-release
    """
    return load_dataset("mmathys/openai-moderation-api-evaluation")
