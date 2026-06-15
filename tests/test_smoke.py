from __future__ import annotations

from pathlib import Path

import torch
import pytest
from torch.utils.data import DataLoader

from time_prototype_mamba.data.dataset import LongitudinalVolumeDataset, make_collate_fn
from time_prototype_mamba.data.synthetic import create_synthetic_dataset
from time_prototype_mamba.models import TimePrototypeMamba


def test_synthetic_forward(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Mamba smoke test requires a CUDA device supported by mamba-ssm.")
    paths = create_synthetic_dataset(tmp_path / "synthetic", num_samples=6, shape_zyx=(12, 16, 16), max_timepoints=3)
    ds = LongitudinalVolumeDataset(paths["train_manifest"], project_root=paths["out_dir"], max_timepoints=3)
    loader = DataLoader(
        ds,
        batch_size=2,
        collate_fn=make_collate_fn(fixed_spatial_shape_zyx=[12, 16, 16]),
    )
    batch = next(iter(loader))
    model = TimePrototypeMamba(
        encoder_backend="monai_unet",
        unet_features=[4, 8],
        num_prototypes=4,
        classifier_max_timepoints=3,
        mlp_hidden=16,
        classifier_depth=1,
        temporal_backend="mamba",
        strict_mamba=True,
    ).cuda()
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    out = model(
        batch["ct"],
        batch["cbct"],
        batch["cbct_valid_mask"],
        batch["slic"],
        cbct_days_from_ct=batch["cbct_days_from_ct"],
        cbct_days_from_prev_cbct=batch["cbct_days_from_prev_cbct"],
        return_aux_loss=True,
    )
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
