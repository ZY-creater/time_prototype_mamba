from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_reproducibility(config: dict[str, Any] | None = None, seed: int | None = None) -> torch.Generator:
    """Set common random sources and return a DataLoader generator."""
    cfg = dict(config or {})
    if seed is None:
        seed = int(cfg.get("seed", 0))
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    if cfg.get("cublas_workspace_config"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(cfg["cublas_workspace_config"]))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(cfg.get("cudnn_deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", False))
    if cfg.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(cfg["float32_matmul_precision"]))
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(
            True,
            warn_only=bool(cfg.get("deterministic_warn_only", True)),
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

