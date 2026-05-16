"""
Build the DRIFTCHECK 10 k benchmark from public sources.

Usage (from repo root):
    PYTHONPATH=src python -m ethical_adapter.driftcheck.build_dataset --output_dir data/driftcheck

Outputs:
    <output_dir>/driftcheck_10k.jsonl   — primary benchmark file
    <output_dir>/driftcheck_10k.csv     — convenience spreadsheet copy
    <output_dir>/build_report.json      — pool sizes, category distribution, seed
    <output_dir>/metadata.json          — dataset provenance card
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ethical_adapter.driftcheck.generators import (
    SAFE_TARGET,
    UNSAFE_TARGET,
    balanced_sample,
    extract_safe_from_mmlu,
    extract_unsafe_from_hatecheck,
    extract_unsafe_from_hh_rlhf,
    extract_unsafe_from_openai_moderation,
    stratified_unsafe_sample,
)
from ethical_adapter.driftcheck.utils import dedupe_records, set_seed, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the DRIFTCHECK benchmark (AlignGuard-LoRA, §2)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where all output files will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global RNG seed for reproducibility."
    )
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract raw pools
    # ------------------------------------------------------------------
    print("=== Extracting safe prompts from MMLU ===")
    safe_pool = extract_safe_from_mmlu()

    print("=== Extracting unsafe prompts from HH-RLHF ===")
    hh_pool = extract_unsafe_from_hh_rlhf()

    print("=== Extracting unsafe prompts from HateCheck ===")
    hc_pool = extract_unsafe_from_hatecheck()

    print("=== Extracting unsafe prompts from OpenAI Moderation ===")
    om_pool = extract_unsafe_from_openai_moderation()

    unsafe_pool = dedupe_records(hh_pool + hc_pool + om_pool)
    print(
        f"Pool sizes — safe: {len(safe_pool)}, "
        f"unsafe (post-merge dedup): {len(unsafe_pool)}"
    )

    # ------------------------------------------------------------------
    # Sample to target sizes (paper §2: 5k safe, 5k unsafe)
    # ------------------------------------------------------------------
    safe_sample = balanced_sample(safe_pool, SAFE_TARGET, args.seed)
    unsafe_sample = stratified_unsafe_sample(unsafe_pool, UNSAFE_TARGET, args.seed)

    # ------------------------------------------------------------------
    # Assemble, assign IDs, shuffle
    # ------------------------------------------------------------------
    full = []
    for i, rec in enumerate(safe_sample + unsafe_sample):
        row = dict(rec)
        row["id"] = f"driftcheck_{i:05d}"
        full.append(row)

    random.Random(args.seed).shuffle(full)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    write_jsonl(outdir / "driftcheck_10k.jsonl", full)
    pd.DataFrame(full).to_csv(outdir / "driftcheck_10k.csv", index=False)

    unsafe_cats = (
        pd.Series([x["category"] for x in unsafe_sample]).value_counts().to_dict()
    )

    report = {
        "n_total": len(full),
        "n_safe": sum(1 for x in full if x["label"] == "safe"),
        "n_unsafe": sum(1 for x in full if x["label"] == "unsafe"),
        "sources": {
            "mmlu_pool": len(safe_pool),
            "hh_rlhf_pool": len(hh_pool),
            "hatecheck_pool": len(hc_pool),
            "openai_moderation_pool": len(om_pool),
            "unsafe_pool_post_dedupe": len(unsafe_pool),
        },
        "unsafe_category_counts": unsafe_cats,
        "seed": args.seed,
    }

    with (outdir / "build_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with (outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": "driftcheck",
                "paper": "AlignGuard-LoRA (Das et al., arXiv 2508.02079)",
                "description": (
                    "10k one-liner prompt benchmark for quantifying alignment "
                    "drift under LoRA fine-tuning. 5k safe (MMLU) + 5k unsafe "
                    "(HH-RLHF, HateCheck, OpenAI Moderation), stratified across "
                    "6 unsafe categories."
                ),
                "seed": args.seed,
                "safe_sources": ["mmlu"],
                "unsafe_sources": ["hh_rlhf", "hatecheck", "openai_moderation"],
                "unsafe_categories": [
                    "violence",
                    "hate_speech",
                    "illegal_activity",
                    "explicit",
                    "privacy",
                    "disinformation",
                ],
            },
            f,
            indent=2,
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
