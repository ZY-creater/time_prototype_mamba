from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _as_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_npz_array(path: Path, key: str) -> torch.Tensor:
    with np.load(str(path)) as payload:
        if key not in payload:
            raise KeyError(f"{path} does not contain npz key {key!r}.")
        arr = np.asarray(payload[key], dtype=np.float32)
    return torch.from_numpy(arr).contiguous()


def _time_values(values: Any, length: int) -> torch.Tensor:
    if values is None:
        out = torch.zeros((0,), dtype=torch.float32)
    elif isinstance(values, torch.Tensor):
        out = values.detach().to(dtype=torch.float32).flatten()
    else:
        out = torch.as_tensor([0.0 if item is None else float(item) for item in values], dtype=torch.float32).flatten()
    if out.numel() < length:
        out = torch.cat([out, torch.zeros(length - out.numel(), dtype=torch.float32)])
    return out[:length]


def _normalize_fixed_shape(value: Any) -> tuple[int, int, int] | None:
    if value is None or value is False or value == "":
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.replace("x", ",").replace("X", ",").split(",") if v.strip()]
    else:
        items = list(value)
    if len(items) != 3:
        raise ValueError("fixed_spatial_shape_zyx must contain exactly three values.")
    out = tuple(int(v) for v in items)
    if any(v <= 0 for v in out):
        raise ValueError("fixed_spatial_shape_zyx values must be positive.")
    return out


def _round_up(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return ((int(value) + multiple - 1) // multiple) * multiple


class LongitudinalVolumeDataset(Dataset):
    """Manifest-backed dataset for longitudinal CT/CBCT volumes.

    Manifest records are intentionally simple and public-data friendly:

    ``case_id``
        Case identifier used for logging and prediction export.
    ``label``
        Binary label, where 1 is the positive class used by the training loss.
    ``ct`` / ``cbct`` / ``slic``
        Relative or absolute paths to npz files. Expected array keys are
        ``ct`` with shape [1, D, H, W], ``cbct`` with shape [T, 1, D, H, W],
        and ``slic`` with shape [1, D, H, W].
    ``cbct_days_from_ct`` / ``cbct_days_from_prev_cbct``
        Real acquisition timing arrays with length T.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path | None = None,
        max_timepoints: int | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root) if project_root is not None else self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise TypeError("manifest must be a JSON list of records.")
        self.records: list[dict[str, Any]] = records
        self.max_timepoints = None if max_timepoints in {None, 0} else int(max_timepoints)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        case_id = str(record.get("case_id", f"case_{index:04d}"))
        ct = _read_npz_array(_as_path(self.project_root, record["ct"]), "ct")
        cbct = _read_npz_array(_as_path(self.project_root, record["cbct"]), "cbct")
        slic = _read_npz_array(_as_path(self.project_root, record["slic"]), "slic")

        if ct.ndim != 4 or ct.shape[0] != 1:
            raise ValueError(f"{case_id}: ct must have shape [1, D, H, W].")
        if cbct.ndim != 5 or cbct.shape[1] != 1:
            raise ValueError(f"{case_id}: cbct must have shape [T, 1, D, H, W].")
        if slic.ndim != 4 or slic.shape[0] != 1:
            raise ValueError(f"{case_id}: slic must have shape [1, D, H, W].")

        if self.max_timepoints is not None:
            cbct = cbct[: self.max_timepoints]
        num_steps = int(cbct.shape[0])

        days_from_ct = _time_values(record.get("cbct_days_from_ct", []), num_steps)
        days_from_prev = _time_values(record.get("cbct_days_from_prev_cbct", []), num_steps)
        return {
            "case_id": case_id,
            "label": float(record.get("label", 0)),
            "ct": ct,
            "cbct": cbct,
            "slic": slic,
            "cbct_days_from_ct": days_from_ct,
            "cbct_days_from_prev_cbct": days_from_prev,
        }


def longitudinal_collate_fn(
    batch: list[dict[str, Any]],
    spatial_size_multiple: int = 1,
    fixed_spatial_shape_zyx: Any | None = None,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch is not supported.")
    fixed_shape = _normalize_fixed_shape(fixed_spatial_shape_zyx)
    max_t = max(int(item["cbct"].shape[0]) for item in batch)
    max_d = max(int(item["ct"].shape[1]) for item in batch)
    max_h = max(int(item["ct"].shape[2]) for item in batch)
    max_w = max(int(item["ct"].shape[3]) for item in batch)
    for item in batch:
        _, _, d, h, w = item["cbct"].shape
        max_d, max_h, max_w = max(max_d, int(d)), max(max_h, int(h)), max(max_w, int(w))

    if fixed_shape is None:
        max_d = _round_up(max_d, spatial_size_multiple)
        max_h = _round_up(max_h, spatial_size_multiple)
        max_w = _round_up(max_w, spatial_size_multiple)
    else:
        if (max_d > fixed_shape[0]) or (max_h > fixed_shape[1]) or (max_w > fixed_shape[2]):
            raise ValueError(f"batch shape {(max_d, max_h, max_w)} exceeds fixed shape {fixed_shape}.")
        max_d, max_h, max_w = fixed_shape

    bsz = len(batch)
    ct = torch.zeros((bsz, 1, max_d, max_h, max_w), dtype=torch.float32)
    cbct = torch.zeros((bsz, max_t, 1, max_d, max_h, max_w), dtype=torch.float32)
    slic = torch.zeros((bsz, 1, max_d, max_h, max_w), dtype=torch.float32)
    valid = torch.zeros((bsz, max_t), dtype=torch.bool)
    labels = torch.zeros((bsz,), dtype=torch.float32)
    days_from_ct = torch.zeros((bsz, max_t), dtype=torch.float32)
    days_from_prev = torch.zeros((bsz, max_t), dtype=torch.float32)
    case_ids: list[str] = []

    for i, item in enumerate(batch):
        c = item["ct"]
        s = item["slic"]
        x = item["cbct"]
        _, d, h, w = c.shape
        ct[i, :, :d, :h, :w] = c
        slic[i, :, :d, :h, :w] = s
        t = int(x.shape[0])
        cbct[i, :t, :, : x.shape[2], : x.shape[3], : x.shape[4]] = x
        valid[i, :t] = True
        labels[i] = float(item["label"])
        days_from_ct[i, :t] = _time_values(item.get("cbct_days_from_ct"), t)
        days_from_prev[i, :t] = _time_values(item.get("cbct_days_from_prev_cbct"), t)
        case_ids.append(str(item.get("case_id", i)))

    return {
        "ct": ct,
        "cbct": cbct,
        "slic": slic,
        "cbct_valid_mask": valid,
        "cbct_days_from_ct": days_from_ct,
        "cbct_days_from_prev_cbct": days_from_prev,
        "labels": labels,
        "case_id": case_ids,
    }


def make_collate_fn(
    spatial_size_multiple: int = 1,
    fixed_spatial_shape_zyx: Any | None = None,
):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return longitudinal_collate_fn(
            batch,
            spatial_size_multiple=spatial_size_multiple,
            fixed_spatial_shape_zyx=fixed_spatial_shape_zyx,
        )

    return _collate


def resize_batch_to_shape(batch: dict[str, torch.Tensor], shape_zyx: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    """Optional helper for notebooks that need a quick spatial resize."""
    batch = dict(batch)
    batch["ct"] = F.interpolate(batch["ct"], size=shape_zyx, mode="trilinear", align_corners=False)
    bsz, steps = batch["cbct"].shape[:2]
    flat = batch["cbct"].reshape(bsz * steps, *batch["cbct"].shape[2:])
    flat = F.interpolate(flat, size=shape_zyx, mode="trilinear", align_corners=False)
    batch["cbct"] = flat.reshape(bsz, steps, 1, *shape_zyx)
    batch["slic"] = F.interpolate(batch["slic"], size=shape_zyx, mode="nearest")
    return batch
