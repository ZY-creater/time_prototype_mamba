# Time-Prototype Mamba

> Manuscript status: under review.

![Time-Prototype Mamba overview](assets/tpm_hero.webp)

**Time-Prototype Mamba (TPM)** is a compact research codebase for longitudinal volumetric imaging. It turns planning CT and serial follow-up scans into **prototype trajectories**, injects real acquisition timing, and uses a Mamba sequence block to predict a patient-level binary outcome.

This repository contains the public implementation of TPM, including the core model, a manifest-based dataset interface, training/evaluation scripts, and a generated smoke-test dataset.

## Why TPM?

Longitudinal radiotherapy imaging is not just a stack of scans. It is a timed trajectory: anatomical subregions evolve at different rates, scans are acquired at irregular intervals, and the clinically useful signal may be local rather than global.

TPM models that trajectory in four steps:

1. **Shared 3D encoder** extracts volumetric features from planning CT and serial scans.
2. **Subregion-to-prototype pooling** compresses variable SLIC-style regions into a fixed set of learnable phenotype prototypes.
3. **Dual-clock time embedding** injects both days from the reference scan and days from the previous scan.
4. **Mamba temporal modeling** encodes prototype trajectories before patient-level classification.

The same forward pass can also return interpretability tensors: prototype assignments, time gates, temporal tokens, and region-time attention weights.

## Repository Layout

```text
time-prototype-mamba/
  configs/
    tpm_synthetic.yaml       # small runnable smoke-test config
    tpm_full_template.yaml   # template for user-provided datasets
  examples/
    make_synthetic_dataset.py
    train_synthetic.py
    evaluate_checkpoint.py
  time_prototype_mamba/
    data/                    # manifest dataset and synthetic generator
    models/                  # encoder, prototype, temporal, TPM model
    training/                # losses, metrics, train/evaluate helpers
    utils/                   # reproducibility utilities
  tests/
    test_smoke.py
```

## Installation

TPM targets Python 3.10-3.12 with a CUDA-enabled PyTorch stack compatible with Mamba-SSM. The smoke tests were verified with Python 3.12, PyTorch 2.9.1, MONAI 1.5.2, and Mamba-SSM 2.3.2.

Create your environment from this folder:

```bash
uv sync --extra dev
```

The install uses PyTorch, MONAI, Mamba-SSM, NumPy, PyYAML, and scikit-learn.

The synthetic smoke config and the full template both use the paper-aligned Mamba backend:

```yaml
temporal_backend: mamba
strict_mamba: true
```

## Quickstart

Generate a tiny synthetic longitudinal dataset:

```bash
uv run python examples/make_synthetic_dataset.py --out data/synthetic --num-samples 24
```

Train the smoke model:

```bash
uv run python examples/train_synthetic.py --config configs/tpm_synthetic.yaml
```

Evaluate the best checkpoint:

```bash
uv run python examples/evaluate_checkpoint.py \
  --config configs/tpm_synthetic.yaml \
  --checkpoint outputs/synthetic_smoke/checkpoints/best.pt \
  --split val
```

Expected outputs:

```text
outputs/synthetic_smoke/
  config_resolved.yaml
  metrics.jsonl
  summary.json
  checkpoints/
    best.pt
    final.pt
```

## Data Format

TPM uses a JSON manifest plus `.npz` arrays. A minimal record looks like this:

```json
{
  "case_id": "case_0001",
  "label": 1,
  "ct": "arrays/case_0001_ct.npz",
  "cbct": "arrays/case_0001_cbct.npz",
  "slic": "arrays/case_0001_slic.npz",
  "cbct_days_from_ct": [3, 7, 12, 18],
  "cbct_days_from_prev_cbct": [3, 4, 5, 6]
}
```

Expected array keys and shapes:

| File | Key | Shape |
|---|---|---|
| planning/reference CT | `ct` | `[1, D, H, W]` |
| longitudinal scans | `cbct` | `[T, 1, D, H, W]` |
| subregion labels | `slic` | `[1, D, H, W]` |

Labels are binary and use `1` as the positive class for `BCEWithLogitsLoss`. If your study uses the opposite convention, remap labels before writing the manifest.

## Data Availability

This repository includes generated synthetic data for software testing and example execution. The data that support the findings of the accompanying study are not publicly available because they contain sensitive patient information. Deidentified data may be made available by the corresponding authors upon reasonable request and subject to approval by the participating institutions and ethics committees after peer review.

## Training Objective

The public training loop uses:

```text
L = L_BCE
  + lambda_cluster * L_cluster
  + lambda_diversity * L_diversity
  + lambda_contrast * L_temporal_contrast
  + lambda_temporal_smooth * L_gap_smoothness
```

`L_cluster` and `L_diversity` regularize the prototype bank. `L_temporal_contrast` encourages the same prototype to remain comparable between adjacent valid visits and supports class-dependent weights matching the manuscript configuration. `L_gap_smoothness` penalizes short-interval prototype jumps more strongly than long-interval changes.

## Reproducibility Notes

The utilities seed Python, NumPy, PyTorch CPU/CUDA, DataLoader workers, cuDNN flags, TF32 flags, and optional cuBLAS workspace configuration. Saved checkpoints and prediction files can be re-evaluated directly.

GPU sequence kernels can still differ across hardware and dependency versions. For paper-grade experiments, keep the resolved config, manifests, metrics log, checkpoint hashes, hardware information, and repeated-run spread instead of relying on `seed` alone.

## Programmatic Use

```python
from time_prototype_mamba import TimePrototypeMamba

model = TimePrototypeMamba(
    unet_features=[8, 16, 32],
    num_prototypes=24,
    classifier_max_timepoints=8,
    temporal_backend="mamba",
    strict_mamba=True,
)
```

During inference or visualization:

```python
out = model(
    ct=batch["ct"],
    cbct=batch["cbct"],
    cbct_valid_mask=batch["cbct_valid_mask"],
    slic=batch["slic"],
    cbct_days_from_ct=batch["cbct_days_from_ct"],
    cbct_days_from_prev_cbct=batch["cbct_days_from_prev_cbct"],
    return_interpretability=True,
)

logits = out["logits"]
attention = out["region_time_attention"]
prototype_tokens = out["temporal_tokens"]
```

## Citation

The accompanying manuscript is currently under review. Citation details will be updated after publication.
