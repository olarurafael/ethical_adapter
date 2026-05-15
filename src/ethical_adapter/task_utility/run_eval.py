from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ethical_adapter.config_io import load_yaml_config
from ethical_adapter.driftcheck.scorers import HFGenerator
from ethical_adapter.task_utility.tasks import get_task_spec, load_task_dataframe


def _looks_like_adapter_config(cfg: dict) -> bool:
    return all(k in cfg for k in ("rank", "alpha", "dropout", "target_modules"))


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    if cfg_path.suffix in {".yaml", ".yml"}:
        return load_yaml_config(str(cfg_path))
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_config_from_checkpoint(checkpoint_dir: str | None) -> dict | None:
    if not checkpoint_dir:
        return None

    ckpt = Path(checkpoint_dir)
    candidates = [ckpt / "config.json", ckpt.parent / "config.json"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if _looks_like_adapter_config(cfg):
            return cfg
    return None


def _resolve_model_and_adapter_config(args) -> tuple[str, dict | None]:
    cfg = None
    if args.config:
        cfg = _load_config(args.config)
    elif args.gate_checkpoint or args.adapter_checkpoint:
        cfg = _infer_config_from_checkpoint(args.gate_checkpoint)
        if cfg is None:
            cfg = _infer_config_from_checkpoint(args.adapter_checkpoint)

    if cfg is not None:
        model_name = cfg.get("local_path") or cfg.get("model_name")
        if not model_name:
            raise ValueError(
                "Resolved config is missing both 'local_path' and 'model_name'."
            )
        return model_name, cfg

    if not args.model_name:
        raise ValueError(
            "Provide --model_name for base eval, or pass --config / "
            "--adapter_checkpoint for injected adapter evaluation."
        )
    return args.model_name, None


def _render_prompt_and_choice(tokenizer, prompt: str, choice: str) -> tuple[str, str]:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": choice},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text

    sep = "" if prompt.endswith((" ", "\n", "\t")) else " "
    prompt_text = prompt
    full_text = f"{prompt}{sep}{choice}"
    return prompt_text, full_text


def _score_choices(
    generator: HFGenerator,
    prompt: str,
    choices: tuple[str, ...],
    max_length: int,
) -> dict[str, float]:
    tokenizer = generator.tokenizer
    model = generator.model

    rendered = [_render_prompt_and_choice(tokenizer, prompt, choice) for choice in choices]
    prompt_texts = [x[0] for x in rendered]
    full_texts = [x[1] for x in rendered]

    prompt_lens = [
        len(
            tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]
        )
        for prompt_text in prompt_texts
    ]

    enc = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        logits = model(**enc).logits[:, :-1, :]
        target_ids = enc["input_ids"][:, 1:]
        target_mask = enc["attention_mask"][:, 1:].bool()
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)

    scores: dict[str, float] = {}
    for idx, choice in enumerate(choices):
        answer_mask = target_mask[idx].clone()
        answer_start = max(prompt_lens[idx] - 1, 0)
        answer_mask[:answer_start] = False
        answer_token_count = int(answer_mask.sum().item())
        if answer_token_count == 0:
            scores[choice] = float("-inf")
            continue
        mean_logprob = (
            token_logprobs[idx][answer_mask].sum() / answer_token_count
        ).item()
        scores[choice] = float(mean_logprob)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate task utility for one supervised task."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["boolq", "mnli", "mrpc", "multirc", "qnli", "qqp", "sst2", "wic"],
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="HuggingFace model id or local path (for plain base-model eval).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Training/eval config (yaml/json) used for adapter injection.",
    )
    parser.add_argument(
        "--adapter_checkpoint", type=str, default=None,
        help="Checkpoint dir with adapter weights to load into injected modules.",
    )
    parser.add_argument(
        "--gate_checkpoint", type=str, default=None,
        help="Optional checkpoint dir with gate weights.",
    )
    parser.add_argument(
        "--adapter_mode", type=str, default="auto",
        choices=["auto", "on", "off", "gate"],
        help="Injected adapter execution mode.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for predictions.jsonl and metrics.json.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of evaluation examples.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Shuffle seed before applying --limit.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./data",
        help="Dataset cache dir passed to HuggingFace datasets.",
    )
    parser.add_argument(
        "--score_max_length", type=int, default=1024,
        help="Max tokenized length used for prompt+answer scoring.",
    )
    parser.add_argument(
        "--preflight_prompts", type=int, default=16,
        help="How many task prompts to use for adapter-effect preflight.",
    )
    parser.add_argument(
        "--preflight_max_new_tokens", type=int, default=16,
        help="Max tokens for preflight generations.",
    )
    parser.add_argument(
        "--preflight_min_change_rate", type=float, default=0.05,
        help="Minimum required output change rate vs baseline mode.",
    )
    parser.add_argument(
        "--allow_preflight_fail", action="store_true",
        help="If set, do not fail the run when adapter-effect preflight fails.",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_name, adapter_config = _resolve_model_and_adapter_config(args)
    spec = get_task_spec(args.task)
    df = load_task_dataframe(
        task=args.task,
        limit=args.limit,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded {len(df)} examples for task '{args.task}' from {spec.dataset_name}/{spec.config_name}.")

    generator = HFGenerator(
        model_name,
        adapter_config=adapter_config,
        adapter_checkpoint=args.adapter_checkpoint,
        gate_checkpoint=args.gate_checkpoint,
        adapter_mode=args.adapter_mode,
    )

    preflight: dict | None = None
    if adapter_config is not None and args.adapter_mode in {"on", "gate"}:
        preflight_prompts = df["prompt"].head(args.preflight_prompts).tolist()
        preflight = generator.preflight_adapter_effect(
            prompts=preflight_prompts,
            target_mode=args.adapter_mode,
            baseline_mode="off",
            max_new_tokens=args.preflight_max_new_tokens,
        )
        if preflight.get("status") == "ok":
            threshold = args.preflight_min_change_rate
            passed = preflight["change_rate"] >= threshold
            preflight["min_change_rate"] = threshold
            preflight["pass"] = bool(passed)
            print("\n=== Preflight Adapter Effect ===")
            print(json.dumps(preflight, indent=2))
            if not passed and not args.allow_preflight_fail:
                raise RuntimeError(
                    "Adapter preflight failed: change_rate "
                    f"{preflight['change_rate']:.4f} < "
                    f"min_change_rate {threshold:.4f}. "
                    "Use --allow_preflight_fail to continue anyway."
                )
        else:
            preflight["pass"] = False
            print("\n=== Preflight Adapter Effect (skipped) ===")
            print(json.dumps(preflight, indent=2))
            if not args.allow_preflight_fail:
                raise RuntimeError(
                    "Adapter preflight skipped and run is configured to fail "
                    "on preflight issues. Use --allow_preflight_fail to continue."
                )

    prediction_rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Scoring"):
        choice_scores = _score_choices(
            generator=generator,
            prompt=row.prompt,
            choices=spec.choices,
            max_length=args.score_max_length,
        )
        pred_answer = max(choice_scores, key=choice_scores.get)
        prediction_rows.append(
            {
                "id": row.id,
                "prompt": row.prompt,
                "gold_answer": row.gold_answer,
                "pred_answer": pred_answer,
                "correct": bool(pred_answer == row.gold_answer),
                "choice_scores": choice_scores,
            }
        )

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_json(
        outdir / "predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    accuracy = float(pred_df["correct"].mean()) if len(pred_df) else 0.0
    per_label_accuracy = {}
    for label in spec.choices:
        sub = pred_df[pred_df["gold_answer"] == label]
        if len(sub) == 0:
            continue
        per_label_accuracy[label] = {
            "accuracy": float(sub["correct"].mean()),
            "n": int(len(sub)),
        }

    metrics = {
        "task": args.task,
        "dataset_name": spec.dataset_name,
        "dataset_config": spec.config_name,
        "dataset_split": spec.split,
        "metric_name": "accuracy",
        "accuracy": accuracy,
        "n_total": int(len(pred_df)),
        "choices": list(spec.choices),
        "score_method": "mean_logprob_of_gold_completion_tokens",
        "model": model_name,
        "adapter_checkpoint": args.adapter_checkpoint,
        "gate_checkpoint": args.gate_checkpoint,
        "adapter_mode": args.adapter_mode,
        "preflight": preflight,
        "per_label_accuracy": per_label_accuracy,
    }

    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Task Utility Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
