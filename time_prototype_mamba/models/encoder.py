from __future__ import annotations

import torch
from torch import nn


class VolumeEncoder(nn.Module):
    """3D encoder for planning and longitudinal volumetric scans.

    TPM uses MONAI UNet as the shared encoder for planning and longitudinal
    volumetric scans.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: tuple[int, ...] = (8, 16, 32),
        dropout: float = 0.0,
        backend: str = "monai_unet",
    ) -> None:
        super().__init__()
        backend = str(backend).strip().lower().replace("-", "_")
        self.backend = backend
        self.out_channels = int(features[-1])

        if backend != "monai_unet":
            raise ValueError("TPM uses encoder_backend='monai_unet'.")

        try:
            from monai.networks.nets import UNet
        except ImportError as exc:  # pragma: no cover
            raise ImportError("MONAI is required for Time-Prototype Mamba.") from exc

        strides = (2,) * max(0, len(features) - 1)
        self.net = UNet(
            spatial_dims=3,
            in_channels=int(in_channels),
            out_channels=self.out_channels,
            channels=tuple(int(v) for v in features),
            strides=strides,
            num_res_units=2,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
