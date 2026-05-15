from pathlib import Path

from ethical_adapter.config_io import load_yaml_config
from ethical_adapter.training.data import build_task_dataset


class _StdoutLogger:
    def info(self, msg, *args):
        print(msg % args if args else msg)


def main(config_path: str) -> None:
    cfg = load_yaml_config(config_path)
    target = cfg.get("frozen_task_dataset_path") or cfg.get("export_task_dataset_path")
    if not target:
        raise ValueError(
            "Config must define frozen_task_dataset_path or export_task_dataset_path."
        )

    logger = _StdoutLogger()
    ds = build_task_dataset(cfg, logger)
    print(f"Materialized {len(ds)} task rows.")
    print(f"Dataset file: {Path(target)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
