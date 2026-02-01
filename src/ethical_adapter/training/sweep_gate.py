import itertools
import subprocess
import time
import os
from pathlib import Path

HIDDEN_SIZES = [512] #512 is the size for only 1 fc, not fc1 or fc2.
ACTIVATIONS = ["relu"] # only relu works
POOLINGS = ["logsumexp"]  # max pooling NEVER works
TEMPERATURES = [1] # 0.7 helps output much more confident numbers than 1.
EPOCHS = [8] # early stop stops longer intervention
BATCH_SIZES = [8] # 8 and 2 seems to be a local minima.
GRAD_ACCUMS = [2] # local minima
# LEARNING_RATES = [3e-3]
LEARNING_RATES = [5e-3]





BASE_TRAIN = Path("configs/sweeps/base_gate_train.yaml").read_text()
BASE_EVAL  = Path("configs/sweeps/base_gate_eval.yaml").read_text()

RUNS_ROOT = Path("runs/sweeps/qwen25_3b/")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

def write_config(text, path, **kwargs):
    for k, v in kwargs.items():
        text = text.replace(f"__{k}__", str(v))
    path.write_text(text)

def latest_best(run_root):
    candidates = sorted(run_root.glob("*/best"), key=lambda p: p.parent.name)
    return candidates[-1]

for (
    hidden_size,
    activation,
    pooling,
    temperature,
    epochs,
    batch_size,
    grad_accum,
    lr,
) in itertools.product(
    HIDDEN_SIZES,
    ACTIVATIONS,
    POOLINGS,
    TEMPERATURES,
    EPOCHS,
    BATCH_SIZES,
    GRAD_ACCUMS,
    LEARNING_RATES,
):


    tag = (
        f"hs{hidden_size}_"
        f"{activation}_"
        f"{pooling}_"
        f"t{temperature}_"
        f"e{epochs}_"
        f"bs{batch_size}_"
        f"ga{grad_accum}_"
        f"lr{lr}"
    )



    run_dir = RUNS_ROOT / tag
    run_dir.mkdir(exist_ok=True)

    train_cfg = run_dir / "train.yaml"
    eval_cfg  = run_dir / "eval.yaml"

    write_config(
        BASE_TRAIN,
        train_cfg,
        HIDDEN_SIZE=hidden_size,
        ACTIVATION=activation,
        POOLING=pooling,
        TEMPERATURE=temperature,
        EPOCHS=epochs,
        BATCH_SIZE=batch_size,
        GRAD_ACCUM=grad_accum,
        LEARNING_RATE=lr,
    )




    print(f"\n=== TRAIN {tag} ===")

    with open(run_dir / "train.log", "w") as log:
        subprocess.run(
            [
                "python",
                "src/ethical_adapter/training/train_gate.py",
                "--config",
                str(train_cfg),
            ],
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    # discover latest trained gate
    gate_root = Path("runs/gates/qwen25_3b")
    best_dir = latest_best(gate_root)

    write_config(
        BASE_EVAL,
        eval_cfg,
        HIDDEN_SIZE=hidden_size,
        ACTIVATION=activation,
        POOLING=pooling,
        TEMPERATURE=temperature,
        RUNS_DIR=best_dir,
        EPOCHS=epochs,
        BATCH_SIZE=batch_size,
        GRAD_ACCUM=grad_accum,
        LEARNING_RATE=lr,
    )



    print(f"\n=== EVAL {tag} ===")

    with open(run_dir / "eval.log", "w") as log:
        subprocess.run(
            [
                "python",
                "scripts/random_bullshit_i_dont_wanna_see/new_eval_gate.py",
                "--config",
                str(eval_cfg),
            ],
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

