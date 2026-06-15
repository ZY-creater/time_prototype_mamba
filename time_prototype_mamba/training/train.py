from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from time_prototype_mamba.data.dataset import LongitudinalVolumeDataset, make_collate_fn
from time_prototype_mamba.models import TimePrototypeMamba
from time_prototype_mamba.training.losses import temporal_contrastive_loss, temporal_gap_smoothness_loss
from time_prototype_mamba.training.metrics import binary_classification_metrics
from time_prototype_mamba.utils.reproducibility import seed_worker, set_reproducibility


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("configuration file must contain a mapping.")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve_project_root(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    root_value = Path(str(config.get("project_root", ".")))
    if root_value.is_absolute():
        return root_value
    return (config_path.parent.parent / root_value).resolve()


def build_dataloaders(config: dict[str, Any], generator: torch.Generator) -> tuple[DataLoader, DataLoader]:
    root = resolve_project_root(config)
    data_cfg = dict(config.get("data", {}))
    train_manifest = root / str(data_cfg["train_manifest"])
    val_manifest = root / str(data_cfg["val_manifest"])
    max_timepoints = data_cfg.get("max_timepoints")
    train_ds = LongitudinalVolumeDataset(train_manifest, project_root=train_manifest.parent, max_timepoints=max_timepoints)
    val_ds = LongitudinalVolumeDataset(val_manifest, project_root=val_manifest.parent, max_timepoints=max_timepoints)
    collate = make_collate_fn(
        spatial_size_multiple=int(data_cfg.get("spatial_size_multiple", 1)),
        fixed_spatial_shape_zyx=data_cfg.get("fixed_spatial_shape_zyx"),
    )
    train_cfg = dict(config.get("training", {}))
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(train_cfg.get("num_workers", 0))
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(train_cfg.get("persistent_workers", False))
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, generator=generator, **common)
    return train_loader, val_loader


def build_model(config: dict[str, Any]) -> TimePrototypeMamba:
    model_cfg = deepcopy(dict(config.get("model", {})))
    model_cfg.pop("time_delta_source", None)
    return TimePrototypeMamba(**model_cfg)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _forward_with_losses(
    model: TimePrototypeMamba,
    batch: dict[str, Any],
    train_cfg: dict[str, Any],
    epoch: int,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
        out = model(
            batch["ct"],
            batch["cbct"],
            batch["cbct_valid_mask"],
            batch["slic"],
            cbct_days_from_ct=batch["cbct_days_from_ct"],
            cbct_days_from_prev_cbct=batch["cbct_days_from_prev_cbct"],
            return_aux_loss=True,
        )
        logits = out["logits"]
        bce = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        warmup_epochs = max(1, int(train_cfg.get("warmup_epochs", 1)))
        warmup = min(1.0, float(epoch) / float(warmup_epochs))
        cluster = out["cluster_loss"]
        diversity = out["diversity_loss"]
        contrast = temporal_contrastive_loss(
            out["proto_trajs"],
            batch["cbct_valid_mask"],
            labels=batch["labels"],
            tau=float(train_cfg.get("tau_contrast", 0.1)),
            negative_class_weight=float(train_cfg.get("contrast_negative_class_weight", 0.5)),
            positive_class_weight=float(train_cfg.get("contrast_positive_class_weight", 1.5)),
        )
        smooth = temporal_gap_smoothness_loss(
            out["proto_trajs"],
            batch["cbct_valid_mask"],
            batch["cbct_days_from_prev_cbct"],
            tau_days=float(train_cfg.get("temporal_smooth_tau_days", 7.0)),
        )
        loss = bce
        loss = loss + float(train_cfg.get("lambda_cluster", 0.0)) * cluster
        loss = loss + float(train_cfg.get("lambda_diversity", 0.0)) * diversity
        loss = loss + warmup * float(train_cfg.get("lambda_contrast", 0.0)) * contrast
        loss = loss + float(train_cfg.get("lambda_temporal_smooth", 0.0)) * smooth
    parts = {
        "bce": float(bce.detach().float().item()),
        "cluster": float(cluster.detach().float().item()),
        "diversity": float(diversity.detach().float().item()),
        "contrast": float(contrast.detach().float().item()),
        "temporal_smooth": float(smooth.detach().float().item()),
    }
    return loss, parts, logits


@torch.no_grad()
def evaluate(
    model: TimePrototypeMamba,
    loader: DataLoader,
    device: torch.device,
    train_cfg: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    labels: list[float] = []
    probs: list[float] = []
    losses: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        loss, _, logits = _forward_with_losses(model, batch, train_cfg, epoch=10**9, amp_enabled=False)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation loss.")
        losses.append(float(loss.detach().cpu().item()))
        labels.extend([float(v) for v in batch["labels"].detach().cpu().tolist()])
        probs.extend([float(v) for v in torch.sigmoid(logits.detach().float()).cpu().tolist()])
        case_ids.extend([str(v) for v in batch.get("case_id", [])])
    metrics = binary_classification_metrics(labels, probs)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    metrics["predictions"] = [
        {"case_id": cid, "label": int(label), "probability": float(prob)}
        for cid, label, prob in zip(case_ids, labels, probs)
    ]
    return metrics


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    repro_cfg = dict(config.get("reproducibility", {}))
    repro_cfg["seed"] = seed
    generator = set_reproducibility(repro_cfg, seed=seed)
    train_loader, val_loader = build_dataloaders(config, generator)
    train_cfg = dict(config.get("training", {}))
    root = resolve_project_root(config)
    output_dir = root / str(train_cfg.get("output_dir", "outputs/run"))
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = deepcopy(config)
    resolved_config["project_root"] = str(root)
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(_json_safe(resolved_config), sort_keys=False), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg.get("epochs", 5))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(train_cfg.get("amp_dtype", "bfloat16")).lower() == "bfloat16" else torch.float16
    accumulate = max(1, int(train_cfg.get("accumulate_grad_batches", 1)))
    fail_on_nonfinite = bool(train_cfg.get("fail_on_nonfinite", True))
    eval_every = max(1, int(train_cfg.get("eval_every", 1)))
    save_eval = bool(train_cfg.get("save_eval_checkpoints", False))

    best_auc = -math.inf
    best_epoch = 0
    best_path = checkpoints_dir / "best.pt"
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        loss_parts: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            loss, parts, _ = _forward_with_losses(model, batch, train_cfg, epoch, amp_enabled, amp_dtype)
            if fail_on_nonfinite and not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}, step {step}.")
            (loss / accumulate).backward()
            if (step % accumulate == 0) or (step == len(train_loader)):
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach().float().cpu().item()))
            for key, value in parts.items():
                loss_parts.setdefault(key, []).append(value)

        should_eval = (epoch % eval_every == 0) or (epoch == epochs)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(sum(epoch_losses) / max(1, len(epoch_losses))),
        }
        for key, values in loss_parts.items():
            row[f"train_{key}"] = float(sum(values) / max(1, len(values)))
        if should_eval:
            val_metrics = evaluate(model, val_loader, device, train_cfg)
            row.update({f"val_{k}": v for k, v in val_metrics.items() if k != "predictions"})
            auc = val_metrics.get("auc")
            if auc is not None and math.isfinite(float(auc)) and float(auc) >= best_auc:
                best_auc = float(auc)
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": resolved_config,
                        "val_metrics": {k: v for k, v in val_metrics.items() if k != "predictions"},
                    },
                    best_path,
                )
            if save_eval:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": resolved_config,
                        "val_metrics": {k: v for k, v in val_metrics.items() if k != "predictions"},
                    },
                    checkpoints_dir / f"epoch_{epoch:03d}.pt",
                )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
        print(json.dumps(_json_safe(row), sort_keys=True))

    final_path = checkpoints_dir / "final.pt"
    torch.save(
        {
            "epoch": epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": resolved_config,
        },
        final_path,
    )
    summary = {
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "final_checkpoint": str(final_path),
        "best_epoch": best_epoch,
        "best_val_auc": best_auc if math.isfinite(best_auc) else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary
