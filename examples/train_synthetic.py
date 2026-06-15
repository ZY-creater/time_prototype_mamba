from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from time_prototype_mamba.training.train import load_config, run_training


def _resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Time-Prototype Mamba on the synthetic dataset.")
    parser.add_argument("--config", default="configs/tpm_synthetic.yaml")
    args = parser.parse_args()
    config_path = _resolve_input_path(args.config)
    summary = run_training(load_config(config_path))
    print(summary)


if __name__ == "__main__":
    main()
