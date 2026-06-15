from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from time_prototype_mamba.training.train import build_dataloaders, build_model, evaluate, load_config, resolve_project_root
from time_prototype_mamba.utils.reproducibility import set_reproducibility


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "val",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 0))
    generator = set_reproducibility({**dict(config.get("reproducibility", {})), "seed": seed}, seed=seed)
    train_loader, val_loader = build_dataloaders(config, generator)
    loader = train_loader if split == "train" else val_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model_config = checkpoint.get("config", config)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, loader, device, dict(config.get("training", {})))
    payload = {
        "split": split,
        "checkpoint": str(Path(checkpoint_path)),
        "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
        "predictions": metrics.get("predictions", []),
    }
    if output_path is None:
        root = resolve_project_root(config)
        output_path = root / str(config.get("training", {}).get("output_dir", "outputs/run")) / f"{split}_evaluation.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def config_from_checkpoint(checkpoint_path: str | Path, out_path: str | Path) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise KeyError("checkpoint does not contain a config mapping.")
    Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

