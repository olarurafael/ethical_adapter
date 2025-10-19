# /src/ethical_adapter/moral_eval/eval_moral_foundations.py
import os
import json
import logging
import argparse
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ethical_adapter.config_io import load_yaml_config
from ethical_adapter.moral_eval.plot_utils import plot_moral_deltas



@torch.no_grad()
def classify_moral_foundations(texts, model, tokenizer, device):
    all_probs = []
    for text in tqdm(texts, desc="Scoring moral foundations"):
        encodings = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512,
        ).to(device)
        logits = model(**encodings).logits
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        all_probs.append(probs)
    return torch.from_numpy(np.stack(all_probs))


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load paired outputs
    with open(cfg["input_file"]) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logging.info(f"Loaded {len(df)} samples for moral alignment eval.")

    # check input format
    if not {"vanilla", "adapter"}.issubset(df.columns):
        raise ValueError("Input JSON must contain both 'vanilla' and 'adapter' fields.")

    # load classifier
    model_name = cfg["moral_classifier"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    # load label names
    label_names = cfg.get("label_names")
    if not label_names:
        raise ValueError("Config file must include a 'label_names' list.")
    
    if model.config.num_labels != len(label_names):
        raise ValueError(
            f"Model expects {model.config.num_labels} labels, "
            f"but config provides {len(label_names)}."
        )
    logging.info(f"Loaded classifier '{model_name}' with {len(label_names)} labels.")

    # classify both vanilla and adapter outputs
    probs_v = classify_moral_foundations(df["vanilla"], model, tokenizer, device)
    probs_a = classify_moral_foundations(df["adapter"], model, tokenizer, device)

   
    # convert to DataFrame
    for i, label in enumerate(label_names):
        df[f"{label}_vanilla"] = probs_v[:, i]
        df[f"{label}_adapter"] = probs_a[:, i]

    # compute mean scores
    mean_v = probs_v.mean().item()
    mean_a = probs_a.mean().item()
    delta = mean_a - mean_v

    summary = (
        f"=== Moral Alignment Summary ===\n"
        f"Classifier: {model_name}\n"
        f"Mean moral prob (vanilla): {mean_v:.4f}\n"
        f"Mean moral prob (adapter): {mean_a:.4f}\n"
        f"Δ moral alignment (adapter - vanilla): {delta:+.4f}\n"
    )

    # per-foundation deltas
    deltas = {}
    summary += "\nPer-foundation Δ moral scores:\n"
    for i, label in enumerate(label_names):
        diff = float(df[f"{label}_adapter"].mean() - df[f"{label}_vanilla"].mean())
        deltas[label] = diff
        summary += f"  {label:20s}: {diff:+.4f}\n"

    print(summary)
    logging.info(summary)

    # make sure output dirs exist
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_summary"]), exist_ok=True)

    # save results
    df.to_csv(cfg["output_csv"], index=False)
    with open(cfg["output_summary"], "w") as f:
        f.write(summary)

    # plot and save deltas
    plot_path = cfg.get(
        "output_plot", 
        f"./eval/moral_eval/moral_foundations_delta_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True) 
    plot_moral_deltas(deltas, plot_path)
    
    logging.info(f"Saved moral foundations delta plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    cfg = load_yaml_config(args.config)

    log_file = f"./eval/logs/moral_eval/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), 
                  logging.StreamHandler()],
    )
    
    main(cfg)
