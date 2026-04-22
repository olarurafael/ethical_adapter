"""
Run three DRIFTCHECK eval conditions sequentially with live logs:
  1) alignguard_off
  2) alignguard_on
  3) gate_mode

Usage:
  PYTHONPATH=src python -m ethical_adapter.driftcheck.script_to_run_all_3 \
    --dataset data/driftcheck/driftcheck_10k.jsonl \
    --output_root results/driftcheck_all3 \
    --alignguard_checkpoint runs/adapters/qwen25_3b_alignguard/2026-03-06_20-19-34/best \
    --gate_checkpoint runs/gates/qwen25_3b/2026-02-25_12-37-59/best
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run_with_live_output(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        return process.wait()


def _build_common_args(args, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "ethical_adapter.driftcheck.run_eval",
        "--dataset",
        args.dataset,
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(args.batch_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--preflight_prompts",
        str(args.preflight_prompts),
        "--preflight_max_new_tokens",
        str(args.preflight_max_new_tokens),
        "--preflight_min_change_rate",
        str(args.preflight_min_change_rate),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.config:
        cmd.extend(["--config", args.config])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 3 driftcheck eval modes with live streamed logs."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--alignguard_checkpoint", type=str, required=True)
    parser.add_argument("--gate_checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preflight_prompts", type=int, default=16)
    parser.add_argument("--preflight_max_new_tokens", type=int, default=48)
    parser.add_argument("--preflight_min_change_rate", type=float, default=0.05)
    parser.add_argument(
        "--allow_preflight_fail",
        action="store_true",
        help="Continue even if preflight says adapter effect is weak.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, list[str]]] = []

    off_dir = output_root / "alignguard_off"
    off_cmd = _build_common_args(args, off_dir)
    off_cmd.extend(
        [
            "--adapter_checkpoint",
            args.alignguard_checkpoint,
            "--adapter_mode",
            "off",
        ]
    )
    runs.append(("alignguard_off", off_cmd))

    on_dir = output_root / "alignguard_on"
    on_cmd = _build_common_args(args, on_dir)
    on_cmd.extend(
        [
            "--adapter_checkpoint",
            args.alignguard_checkpoint,
            "--adapter_mode",
            "on",
        ]
    )
    if args.allow_preflight_fail:
        on_cmd.append("--allow_preflight_fail")
    runs.append(("alignguard_on", on_cmd))

    gate_dir = output_root / "gate_mode"
    gate_cmd = _build_common_args(args, gate_dir)
    gate_cmd.extend(
        [
            "--adapter_checkpoint",
            args.gate_checkpoint,
            "--gate_checkpoint",
            args.gate_checkpoint,
            "--adapter_mode",
            "gate",
        ]
    )
    if args.allow_preflight_fail:
        gate_cmd.append("--allow_preflight_fail")
    runs.append(("gate_mode", gate_cmd))

    print("=== Starting 3-run driftcheck sequence ===")
    for name, cmd in runs:
        print(f"\n=== [{name}] command ===")
        print(" ".join(cmd))
        log_path = output_root / f"{name}.log"
        rc = _run_with_live_output(cmd, log_path=log_path)
        print(f"=== [{name}] exit code: {rc} | log: {log_path} ===")
        if rc != 0:
            raise SystemExit(rc)

    print("\n=== All 3 runs completed successfully ===")
    print(f"Results root: {output_root}")


if __name__ == "__main__":
    main()
