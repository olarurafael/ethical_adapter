"""Task utility evaluation harness.

Evaluate one supervised task at a time under three adapter execution modes:

1. adapter off
2. adapter on
3. gate mode

Typical usage:

    PYTHONPATH=src python -m ethical_adapter.task_utility.run_eval \
        --task boolq \
        --adapter_checkpoint runs/adapters/qwen25_3b_alignguard/boolq/.../best \
        --gate_checkpoint runs/gates/qwen25_3b_alignguard/2026-03-11_16-33-23/best \
        --adapter_mode gate \
        --output_dir results/task_utility_boolq_gate

Or run all three modes sequentially:

    PYTHONPATH=src python -m ethical_adapter.task_utility.script_to_run_all_3 \
        --task boolq \
        --adapter_checkpoint runs/adapters/qwen25_3b_alignguard/boolq/.../best \
        --gate_checkpoint runs/gates/qwen25_3b_alignguard/2026-03-11_16-33-23/best \
        --output_root results/task_utility_boolq_all3
"""

