from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from time_prototype_mamba.training.evaluate import evaluate_checkpoint


def _resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TPM checkpoint.")
    parser.add_argument("--config", default="configs/tpm_synthetic.yaml")
    parser.add_argument("--checkpoint", default="outputs/synthetic_smoke/checkpoints/best.pt")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config_path = _resolve_input_path(args.config)
    checkpoint_path = _resolve_input_path(args.checkpoint)
    payload = evaluate_checkpoint(config_path, checkpoint_path, split=args.split, output_path=args.out)
    print(payload["metrics"])


if __name__ == "__main__":
    main()
