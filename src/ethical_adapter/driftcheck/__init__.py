# driftcheck — DRIFTCHECK benchmark (AlignGuard-LoRA, Das et al. 2025)
# Place this folder under src/ and run with src/ on PYTHONPATH:
#   PYTHONPATH=src python -m ethical_adapter.driftcheck.build_dataset --output_dir data/driftcheck
#   PYTHONPATH=src python -m ethical_adapter.driftcheck.run_eval --dataset data/driftcheck/driftcheck_10k.jsonl \
#       --model_name meta-llama/Llama-3-8B-Instruct --output_dir results/
