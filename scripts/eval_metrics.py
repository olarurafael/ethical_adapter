# scripts/eval_metrics.py
import os
import json
import logging
from datetime import datetime
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ethical_adapter.evaluation.metrics import compute_perplexity, cosine_similarity_between_models
from ethical_adapter.training.load_adapters import load_adapters_from_checkpoint
from ethical_adapter.core.inject import inject_adapters
from ethical_adapter.core.config import AdapterConfig
from ethical_adapter.config_io import load_yaml_config
from ethical_adapter.evaluation.metrics_cka import compute_layerwise_CKA, compute_layerwise_deltas
from ethical_adapter.evaluation.plot_utils import plot_cka_and_delta


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
# simple helper for visually separating sections in logs
def log_section(title: str):
    logging.info("=" * 80)
    logging.info(title)
    logging.info("=" * 80)


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def main(cfg):
    # Initialize tokenizer and base/adapter models
    tokenizer = AutoTokenizer.from_pretrained(cfg["local_path"])
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    log_section("Loading Base Model")
    vanilla = AutoModelForCausalLM.from_pretrained(
        cfg["local_path"],
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # Inject adapter layers and load their trained weights
    log_section("Loading Adapter Model")
    adapter_model = AutoModelForCausalLM.from_pretrained(
        cfg["local_path"],
        device_map="auto",
        dtype=torch.bfloat16,
    )

    adapter_cfg = AdapterConfig(
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
    )
    inject_adapters(adapter_model, adapter_cfg)
    adapter_model = load_adapters_from_checkpoint(adapter_model, cfg["load_adapters_from"])


    # sanity check: are adapter weights nonzero?
    for name, p in adapter_model.named_parameters():
        if "adapter.B.weight" in name:
            print(name, p.abs().mean().item())

    # sanity check forward pass
    text = "Ethical reasoning requires understanding perspectives."
    inputs = tokenizer(text, return_tensors="pt").to(vanilla.device)
    with torch.no_grad():
        out_v = vanilla(**inputs, output_hidden_states=True).hidden_states[-1].mean().item()
        out_a = adapter_model(**inputs, output_hidden_states=True).hidden_states[-1].mean().item()
    logging.debug("Mean hidden state: vanilla=%.6f, adapter=%.6f", out_v, out_a)

    # perplexity evaluation: measuring fluency cost
    log_section("Evaluating Perplexity (Fluency Cost)")
    ppl_vanilla = compute_perplexity(vanilla, tokenizer, max_samples=50)
    ppl_adapter = compute_perplexity(adapter_model, tokenizer, max_samples=50)
    delta = ppl_adapter - ppl_vanilla
    logging.info("Vanilla PPL: %.2f", ppl_vanilla)
    logging.info("Adapter PPL: %.2f", ppl_adapter)
    logging.info("Δ Perplexity: %+0.2f", delta)


    # representation similarity: compute embedding-level similarity
    log_section("Evaluating Representation Similarity (Cosine)")
    sim = cosine_similarity_between_models(
        vanilla, adapter_model, tokenizer,
        text="Ethical reasoning requires understanding perspectives."
    )
    logging.info("Cosine similarity: %.4f", sim)

    # layerwise CKA similarity + deltas
    log_section("Evaluating Layerwise CKA Similarity and Δ Hidden Activations")
    texts = [
        "Ethics guide our actions in complex societies.",
        "Justice and fairness are cornerstones of moral reasoning.",
        "AI systems should balance efficiency and safety.",
        "Privacy must be respected even in data-driven economies.",
        "Empathy plays a role in ethical decision-making.",
        "Freedom entails responsibility for consequences."
    ]

    # Compute metrics
    cka_scores = compute_layerwise_CKA(vanilla, adapter_model, tokenizer, texts)
    delta_scores = compute_layerwise_deltas(vanilla, adapter_model, tokenizer, texts)

    # Print summaries
    logging.info("Layerwise CKA similarity:")
    for l, v in cka_scores.items():
        logging.info("Layer %02d: %.4f", l, v)

    logging.info("Layerwise Δ hidden activations:")
    for l, v in delta_scores.items():
        logging.info("Layer %02d: %.6f", l, v)

    # Plot
    adapter_layers = [8, 15]
    plot_cka_and_delta(cka_scores, delta_scores,
                       adapter_layers, output_path="./eval/cka_delta_plot.png")


    # save results
    log_section("Saving Results")
    os.makedirs("./eval/logs/eval_metrics", exist_ok=True)
    results_path = os.path.join("./eval", "metrics_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "vanilla_ppl": ppl_vanilla,
            "adapter_ppl": ppl_adapter,
            "delta_ppl": delta,
            "cosine_similarity": sim,
            "cka_scores": cka_scores,
            "delta_scores": delta_scores,
        }, f, indent=2)
    logging.info(f"Results saved to {results_path}")


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":


    log_file = f"./eval/logs/eval_metrics/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    main(cfg)



