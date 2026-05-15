import argparse

from transformers import AutoConfig, AutoModelForCausalLM


def _module_depth(name: str) -> int:
    if not name:
        return 0
    return name.count(".") + 1


def _print_matching_modules(model, pattern: str | None, max_depth: int | None) -> None:
    matches = []
    for name, module in model.named_modules():
        if pattern and pattern not in name:
            continue
        if max_depth is not None and _module_depth(name) > max_depth:
            continue
        matches.append((name, module.__class__.__name__))

    if not matches:
        print("No matching modules found.")
        return

    print(f"{'Module':80s} | {'Type'}")
    print("-" * 110)
    for name, cls_name in matches:
        label = name or "<root>"
        print(f"{label:80s} | {cls_name}")
    print("-" * 110)
    print(f"Matched modules: {len(matches)}")


def _print_layer_summary(model) -> None:
    layers = []
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and name.count(".") == 2:
            layers.append((name, module.__class__.__name__))

    if not layers:
        print("No decoder layers found under model.layers.")
        return

    print(f"{'Index':>5s} | {'Layer':40s} | {'Type'}")
    print("-" * 80)
    for idx, (name, cls_name) in enumerate(layers):
        print(f"{idx:5d} | {name:40s} | {cls_name}")
    print("-" * 80)
    print(f"Decoder layers: {len(layers)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Local model path or HF model id.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Only print module names containing this substring.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Only print module names up to this dotted depth.",
    )
    parser.add_argument(
        "--layers_only",
        action="store_true",
        help="Print only the top-level decoder layer map under model.layers.",
    )
    parser.add_argument(
        "--from_config_only",
        action="store_true",
        help="Build the model from config without loading weights.",
    )
    args = parser.parse_args()

    if args.from_config_only:
        cfg = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="cpu",
        )

    print(f"Loaded model: {args.model}")
    print(f"Model class: {model.__class__.__name__}")
    if hasattr(model, "config"):
        cfg = model.config
        print(f"Hidden size: {getattr(cfg, 'hidden_size', 'n/a')}")
        print(f"Hidden layers: {getattr(cfg, 'num_hidden_layers', 'n/a')}")
        print(f"Intermediate size: {getattr(cfg, 'intermediate_size', 'n/a')}")
    print()

    if args.layers_only:
        _print_layer_summary(model)
        return

    _print_matching_modules(model, args.pattern, args.max_depth)


if __name__ == "__main__":
    main()
