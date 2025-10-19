# src/ethical_adapter/eval_sampling.py
import json
from pathlib import Path
import argparse
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from ethical_adapter.config_io import load_yaml_config
from ethical_adapter.core.inject import inject_adapters
from ethical_adapter.core.config import AdapterConfig
from ethical_adapter.training.load_adapters import load_adapters_from_checkpoint


@torch.inference_mode()
def generate_responses(model, tokenizer, prompts, gen_cfg, device="cuda"):
    """
    Generate text continuations for a list of prompts.
    """
    outputs = []
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 100),
            temperature=gen_cfg.get("temperature", 0.7),
            top_k=gen_cfg.get("top_k", 50),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=True,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"] if "tokenizer_name" in cfg else cfg["local_path"])
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # load base model
    logging.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"] if "model_name" in cfg else cfg["local_path"], dtype=torch.bfloat16, device_map="auto"
    )

    # create copy of base model and load adapters
    logging.info("Preparing adapter model...")
    steer_model = AutoModelForCausalLM.from_pretrained(
         cfg["model_name"] if "model_name" in cfg else cfg["local_path"], dtype=torch.bfloat16, device_map="auto"
    )

    adapter_cfg = AdapterConfig(
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
    )
    inject_adapters(steer_model, adapter_cfg)
    steer_model = load_adapters_from_checkpoint(steer_model, cfg["adapter_checkpoint"])
    steer_model.eval()
    base_model.eval()

    # prompts
    if isinstance(cfg["prompts"], str):
        with open(cfg["prompts"]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = cfg["prompts"]

    # generate
    logging.info("Generating baseline outputs...")
    base_outs = generate_responses(base_model, tokenizer, prompts, cfg["generation"], device=device)
    logging.info("Generating adapter-steered outputs...")
    steer_outs = generate_responses(steer_model, tokenizer, prompts, cfg["generation"], device=device)

    # save results
    results = []
    for p, b, s in zip(prompts, base_outs, steer_outs):
        results.append({"prompt": p, "vanilla": b, "adapter": s})

    out_path = Path(cfg.get("output_file", "eval_outputs.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info("Saved comparison outputs to %s", out_path)


if __name__ == "__main__":

    log_file = log_file = f"./eval/logs/eval_sampling/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    
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
