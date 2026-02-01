# ethical-adapter-llm

**A comparative study of alignment–utility trade-offs in parameter-efficient adapters for small language models.**

---

## Scope

This repository accompanies a dissertation investigating the trade-offs between:

- **AlignGuard-style LoRA alignment training**, and
- **Simple learned gating mechanisms combined with task adapters**

with respect to:

- **Task utility** (e.g. SST-2 sentiment classification)
- **Alignment behavior** (e.g. toxicity / refusal signals)

The focus is **measuring what is gained or lost** when different lightweight alignment strategies are applied to small language models.

---

## Research question

> For smaller LLMs, how does AlignGuard-style alignment training compare to a simple gate + LoRA approach in terms of preserving task performance while improving alignment?

More concretely:

- Does alignment-aware LoRA introduce unnecessary task degradation?
- Can a learned gate recover utility while still modulating behavior?
- Are there measurable differences that justify the added complexity of AlignGuard-style training?

---

## Methodology (high level)

All approaches are evaluated under identical constraints:

- Same base model
- Same task dataset
- Same adapter rank and parameter budget
- Same evaluation pipeline

The following variants are compared:

1. **Base model** (no adapters)
2. **Task-only LoRA adapters**
3. **Task LoRA + learned alignment gate**
4. **AlignGuard-style alignment-aware LoRA adapters**

Task utility and alignment metrics are evaluated separately to avoid conflating effects.

---

## Design constraints

This work is intentionally limited to:

- ~1B parameter open-source models
- Single consumer GPU (RTX 4060 Ti, 16GB VRAM)
- No distributed training or RLHF

These constraints are part of the study, not a limitation to be hidden.

---

## Repository structure

```text
src/ethical_adapter/    Core adapter, gate, and training logic
runs/
  ├── adapters/         Task, AlignGuard, and LoRA checkpoints
  └── gates/            Learned gate checkpoints
scripts/                Training, evaluation, and probing scripts
data/                   Task and alignment datasets (HF format)
analysis/               Plots, logs, and evaluation summaries
