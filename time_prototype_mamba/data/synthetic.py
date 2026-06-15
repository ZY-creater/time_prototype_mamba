from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _make_slic(shape: tuple[int, int, int]) -> np.ndarray:
    d, h, w = shape
    slic = np.zeros((1, d, h, w), dtype=np.float32)
    label = 1
    z0, z1 = d // 4, d - d // 4
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    for zi in np.array_split(np.arange(z0, z1), 2):
        for yi in np.array_split(np.arange(y0, y1), 3):
            for xi in np.array_split(np.arange(x0, x1), 3):
                slic[0, zi[:, None, None], yi[None, :, None], xi[None, None, :]] = label
                label += 1
    return slic


def _smooth_blob(shape: tuple[int, int, int], center: tuple[float, float, float], radius: float) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, shape[0], dtype=np.float32),
        np.linspace(-1, 1, shape[1], dtype=np.float32),
        np.linspace(-1, 1, shape[2], dtype=np.float32),
        indexing="ij",
    )
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return np.exp(-dist2 / max(radius, 1e-3)).astype(np.float32)


def create_synthetic_dataset(
    out_dir: str | Path,
    num_samples: int = 24,
    shape_zyx: tuple[int, int, int] = (16, 24, 24),
    max_timepoints: int = 4,
    val_fraction: float = 0.25,
    seed: int = 7,
) -> dict[str, Path]:
    """Create a tiny label-correlated longitudinal dataset for smoke tests."""
    out_dir = Path(out_dir)
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    slic = _make_slic(shape_zyx)
    base_blob = _smooth_blob(shape_zyx, center=(0.0, 0.0, 0.0), radius=0.35)[None]
    side_blob = _smooth_blob(shape_zyx, center=(0.15, -0.25, 0.2), radius=0.18)[None]

    records: list[dict[str, object]] = []
    for idx in range(int(num_samples)):
        label = idx % 2
        case_id = f"synthetic_{idx:03d}"
        num_steps = int(rng.integers(max(2, max_timepoints - 1), max_timepoints + 1))
        days_from_ct = np.cumsum(rng.integers(2, 7, size=num_steps)).astype(np.float32)
        days_from_prev = np.diff(np.concatenate([[0.0], days_from_ct])).astype(np.float32)
        noise = rng.normal(0, 0.18, size=(1, *shape_zyx)).astype(np.float32)
        ct = noise + 0.8 * base_blob + 0.12 * side_blob
        cbct = []
        for t in range(num_steps):
            trend = (t + 1) / max(1, num_steps)
            response_signal = (0.32 * trend if label == 1 else -0.24 * trend)
            visit = ct + response_signal * side_blob
            visit += rng.normal(0, 0.10, size=(1, *shape_zyx)).astype(np.float32)
            cbct.append(visit.astype(np.float32))
        cbct_arr = np.stack(cbct, axis=0)

        ct_file = arrays_dir / f"{case_id}_ct.npz"
        cbct_file = arrays_dir / f"{case_id}_cbct.npz"
        slic_file = arrays_dir / f"{case_id}_slic.npz"
        np.savez_compressed(ct_file, ct=ct.astype(np.float32))
        np.savez_compressed(cbct_file, cbct=cbct_arr.astype(np.float32))
        np.savez_compressed(slic_file, slic=slic.astype(np.float32))
        records.append(
            {
                "case_id": case_id,
                "label": int(label),
                "ct": str(ct_file.relative_to(out_dir)),
                "cbct": str(cbct_file.relative_to(out_dir)),
                "slic": str(slic_file.relative_to(out_dir)),
                "cbct_days_from_ct": [float(v) for v in days_from_ct],
                "cbct_days_from_prev_cbct": [float(v) for v in days_from_prev],
            }
        )

    positives = [item for item in records if int(item["label"]) == 1]
    negatives = [item for item in records if int(item["label"]) == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    num_val = max(2, int(round(len(records) * float(val_fraction))))
    num_val_pos = max(1, min(len(positives) - 1, num_val // 2))
    num_val_neg = max(1, min(len(negatives) - 1, num_val - num_val_pos))
    val_records = positives[:num_val_pos] + negatives[:num_val_neg]
    train_records = positives[num_val_pos:] + negatives[num_val_neg:]
    rng.shuffle(val_records)
    rng.shuffle(train_records)
    train_manifest = out_dir / "train_manifest.json"
    val_manifest = out_dir / "val_manifest.json"
    all_manifest = out_dir / "manifest.json"
    train_manifest.write_text(json.dumps(train_records, indent=2), encoding="utf-8")
    val_manifest.write_text(json.dumps(val_records, indent=2), encoding="utf-8")
    all_manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return {
        "out_dir": out_dir,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "manifest": all_manifest,
    }
