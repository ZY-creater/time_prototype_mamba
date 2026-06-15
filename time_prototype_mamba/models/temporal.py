from __future__ import annotations

from typing import Any

import torch
from torch import nn


try:  # pragma: no cover - availability depends on the local CUDA stack.
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover
    Mamba = None  # type: ignore


class TemporalSequenceEncoder(nn.Module):
    """Mamba temporal encoder for prototype trajectories."""

    def __init__(
        self,
        embed_dim: int,
        backend: str = "mamba",
        strict_mamba: bool = True,
        mamba_use_fast_path: bool = True,
        state_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        backend = str(backend).strip().lower()
        if backend != "mamba":
            raise ValueError("TPM uses temporal_backend='mamba'.")
        if Mamba is None:
            raise ImportError("mamba-ssm is required for Time-Prototype Mamba.")
        if not strict_mamba:
            raise ValueError("strict_mamba must remain true for the public TPM implementation.")
        kwargs: dict[str, Any] = {
            "d_model": self.embed_dim,
            "use_fast_path": bool(mamba_use_fast_path),
        }
        if state_dim is not None:
            kwargs["d_state"] = int(state_dim)
        self.backend = "mamba"
        self.sequence = Mamba(**kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Encode [B, K, T, C] prototype trajectories."""
        bsz, num_proto, num_steps, channels = z.shape
        x = z.reshape(bsz * num_proto, num_steps, channels)
        y = self.sequence(x)
        return y.reshape(bsz, num_proto, num_steps, channels)
