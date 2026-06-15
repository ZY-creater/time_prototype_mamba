from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import VolumeEncoder
from .prototype import PrototypeSubregionModule
from .temporal import TemporalSequenceEncoder


class DualClockTimeEmbedding(nn.Module):
    """Embed absolute and interval scan timing into prototype trajectories."""

    def __init__(
        self,
        embed_dim: int,
        num_prototypes: int,
        scale: float = 30.0,
        init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.scale = float(scale)
        hidden = max(8, min(64, int(embed_dim)))
        self.time_net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(embed_dim)),
        )
        self.prototype_time_bias = nn.Parameter(torch.zeros(int(num_prototypes), int(embed_dim)))
        self.gate_net = nn.Sequential(
            nn.Linear(2 * int(embed_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    @staticmethod
    def _valid_day_values(days: torch.Tensor) -> torch.Tensor:
        out = days.to(dtype=torch.float32)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

    def forward(
        self,
        proto_trajs: torch.Tensor,
        days_from_origin: torch.Tensor,
        days_from_previous: torch.Tensor,
        valid_mask: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, num_proto, num_steps, channels = proto_trajs.shape
        abs_days = self._valid_day_values(days_from_origin).to(device=proto_trajs.device)
        gap_days = self._valid_day_values(days_from_previous).to(device=proto_trajs.device)
        valid = valid_mask.to(device=proto_trajs.device, dtype=proto_trajs.dtype)
        if abs_days.shape != (bsz, num_steps) or gap_days.shape != (bsz, num_steps):
            raise ValueError("time tensors must have shape [B, T].")

        first_flag = torch.zeros((bsz, num_steps), device=proto_trajs.device, dtype=proto_trajs.dtype)
        if num_steps > 0:
            first_flag[:, 0] = 1.0
        scale = max(self.scale, 1e-6)
        log_scale = torch.log1p(abs_days.new_tensor(scale))
        features = torch.stack(
            [
                abs_days / scale,
                gap_days / scale,
                torch.log1p(abs_days) / log_scale,
                torch.log1p(gap_days) / log_scale,
                first_flag,
            ],
            dim=-1,
        )
        time_embed = self.time_net(features) * valid.unsqueeze(-1)
        time_embed = time_embed.unsqueeze(1) + self.prototype_time_bias[None, :, None, :]
        gate = torch.sigmoid(self.gate_net(torch.cat([proto_trajs, time_embed.expand_as(proto_trajs)], dim=-1)))
        delta = self.residual_scale.to(dtype=proto_trajs.dtype) * gate * time_embed
        fused = (proto_trajs + delta) * valid[:, None, :, None]
        if not return_details:
            return fused
        return fused, {
            "time_features": features,
            "time_gate": gate.squeeze(-1),
            "time_delta": delta,
            "time_residual_scale": self.residual_scale.detach().reshape(1),
        }


class RegionTimeAttentionPooling(nn.Module):
    """CT-conditioned attention over prototype-time tokens."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.token_norm = nn.LayerNorm(int(embed_dim))
        self.ct_norm = nn.LayerNorm(int(embed_dim))
        self.score = nn.Sequential(
            nn.Linear(2 * int(embed_dim), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        ct_features: torch.Tensor,
        valid_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bsz, num_proto, num_steps, channels = tokens.shape
        valid = valid_mask.to(device=tokens.device, dtype=torch.bool)
        z = self.token_norm(tokens)
        ct_context = self.ct_norm(ct_features).view(bsz, 1, 1, channels).expand_as(z)
        scores = self.score(torch.cat([z, ct_context], dim=-1)).squeeze(-1)
        token_valid = valid[:, None, :].expand(bsz, num_proto, num_steps)
        scores = scores.masked_fill(~token_valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.reshape(bsz, num_proto * num_steps), dim=-1).reshape(bsz, num_proto, num_steps)
        attention = attention * token_valid.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        pooled = (tokens * attention.unsqueeze(-1)).sum(dim=(1, 2))
        if return_attention:
            return pooled, attention
        return pooled


def _make_classifier(in_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    depth = max(1, int(depth))
    for _ in range(depth):
        layers.extend(
            [
                nn.Linear(current, int(hidden_dim)),
                nn.LayerNorm(int(hidden_dim)),
                nn.GELU(),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(p=float(dropout)))
        current = int(hidden_dim)
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


class TimePrototypeMamba(nn.Module):
    """Time-Prototype Mamba for longitudinal volumetric response modeling.

    Inputs are planning CT, serial CBCT volumes, SLIC-style subregion labels,
    a valid-timepoint mask, and real acquisition timing. The model builds
    prototype trajectories from subregion features and encodes them with Mamba
    before patient-level binary classification.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_backend: str = "monai_unet",
        unet_features: tuple[int, ...] | list[int] = (8, 16, 32),
        unet_dropout: float = 0.0,
        num_prototypes: int = 24,
        pooling_mode: str = "flatten_tokens",
        classifier_depth: int = 3,
        classifier_max_timepoints: int = 8,
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.15,
        use_time_delta: bool = True,
        time_embedding_mode: str = "dual_clock",
        time_delta_scale: float = 30.0,
        time_residual_init: float = 0.05,
        mask_padding_before_temporal: bool = True,
        ct_encoder_chunk_size: int = 0,
        cbct_encoder_chunk_size: int = 0,
        encode_valid_timepoints_only: bool = True,
        temporal_backend: str = "mamba",
        strict_mamba: bool = True,
        mamba_use_fast_path: bool = True,
    ) -> None:
        super().__init__()
        features = tuple(int(v) for v in unet_features)
        embed_dim = int(features[-1])
        self.num_prototypes = int(num_prototypes)
        self.embed_dim = embed_dim
        self.pooling_mode = str(pooling_mode).strip().lower().replace("-", "_")
        self.classifier_max_timepoints = int(classifier_max_timepoints)
        self.use_time_delta = bool(use_time_delta)
        self.time_embedding_mode = str(time_embedding_mode).strip().lower().replace("-", "_")
        self.mask_padding_before_temporal = bool(mask_padding_before_temporal)
        self.ct_encoder_chunk_size = max(0, int(ct_encoder_chunk_size))
        self.cbct_encoder_chunk_size = max(0, int(cbct_encoder_chunk_size))
        self.encode_valid_timepoints_only = bool(encode_valid_timepoints_only)

        if self.pooling_mode not in {"mean", "region_time_attention", "flatten_tokens"}:
            raise ValueError("pooling_mode must be 'mean', 'region_time_attention', or 'flatten_tokens'.")
        if self.time_embedding_mode not in {"none", "dual_clock"}:
            raise ValueError("time_embedding_mode must be 'dual_clock' or 'none'.")
        if self.pooling_mode == "flatten_tokens" and self.classifier_max_timepoints <= 0:
            raise ValueError("flatten_tokens pooling requires classifier_max_timepoints > 0.")

        self.encoder = VolumeEncoder(
            in_channels=int(in_channels),
            features=features,
            dropout=float(unet_dropout),
            backend=encoder_backend,
        )
        self.prototype_module = PrototypeSubregionModule(embed_dim, self.num_prototypes)
        self.time_embedding = (
            DualClockTimeEmbedding(embed_dim, self.num_prototypes, time_delta_scale, time_residual_init)
            if self.use_time_delta and self.time_embedding_mode == "dual_clock"
            else None
        )
        self.temporal_encoder = TemporalSequenceEncoder(
            embed_dim,
            backend=temporal_backend,
            strict_mamba=bool(strict_mamba),
            mamba_use_fast_path=bool(mamba_use_fast_path),
        )
        self.region_time_pool = (
            RegionTimeAttentionPooling(embed_dim, int(mlp_hidden))
            if self.pooling_mode == "region_time_attention"
            else None
        )
        self.flatten_token_norm = nn.LayerNorm(embed_dim) if self.pooling_mode == "flatten_tokens" else None
        self.flatten_ct_norm = nn.LayerNorm(embed_dim) if self.pooling_mode == "flatten_tokens" else None

        if self.pooling_mode == "flatten_tokens":
            classifier_in = self.num_prototypes * self.classifier_max_timepoints * embed_dim
            classifier_in += embed_dim + self.classifier_max_timepoints
        else:
            classifier_in = 2 * embed_dim
        self.classifier = _make_classifier(classifier_in, int(mlp_hidden), int(classifier_depth), float(mlp_dropout))

    def _encode_volumes(self, x: torch.Tensor, chunk_size: int = 0) -> torch.Tensor:
        if chunk_size > 0 and x.shape[0] > chunk_size:
            return torch.cat([self.encoder(chunk) for chunk in x.split(chunk_size, dim=0)], dim=0)
        return self.encoder(x)

    def _encode_cbct(self, cbct: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_steps, channels, depth, height, width = cbct.shape
        flat = cbct.reshape(bsz * num_steps, channels, depth, height, width)
        if self.encode_valid_timepoints_only:
            valid_flat = valid_mask.reshape(bsz * num_steps).to(device=cbct.device, dtype=torch.bool)
        else:
            valid_flat = torch.ones((bsz * num_steps,), device=cbct.device, dtype=torch.bool)
        valid_indices = torch.nonzero(valid_flat, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("At least one valid longitudinal scan is required.")
        valid_volumes = flat.index_select(0, valid_indices)
        encoded = self._encode_volumes(valid_volumes, self.cbct_encoder_chunk_size)
        out = encoded.new_zeros((bsz * num_steps, *encoded.shape[1:]))
        out = out.index_copy(0, valid_indices, encoded)
        return out.reshape(bsz, num_steps, *encoded.shape[1:])

    @staticmethod
    def _subregion_mean_features(feat: torch.Tensor, slic_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = feat.shape[0]
        feat_flat = feat.reshape(channels, -1).transpose(0, 1)
        valid = slic_flat > 0
        if not bool(valid.any()):
            return feat_flat.mean(dim=0, keepdim=True), torch.zeros((1,), device=feat.device, dtype=torch.long)
        labels = slic_flat[valid].long()
        values = feat_flat[valid]
        unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        sums = values.new_zeros((unique_labels.numel(), channels))
        sums.index_add_(0, inverse, values)
        counts = torch.bincount(inverse, minlength=unique_labels.numel()).to(device=feat.device, dtype=values.dtype)
        return sums / counts.unsqueeze(1).clamp(min=1.0), unique_labels

    def _prototype_trajectories(
        self,
        cbct_features: torch.Tensor,
        slic: torch.Tensor,
        valid_mask: torch.Tensor,
        return_details: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[dict[str, torch.Tensor]]]]:
        bsz, num_steps, channels, feat_d, feat_h, feat_w = cbct_features.shape
        slic_vol = slic[:, 0]
        proto_by_case = []
        cluster_total = cbct_features.new_tensor(0.0)
        diversity_total = cbct_features.new_tensor(0.0)
        loss_count = 0
        assignments: list[list[dict[str, torch.Tensor]]] = []

        for b in range(bsz):
            per_time = []
            per_assignment: list[dict[str, torch.Tensor]] = []
            slic_b = slic_vol[b]
            if tuple(slic_b.shape) != (feat_d, feat_h, feat_w):
                slic_b = F.interpolate(
                    slic_b[None, None].float(),
                    size=(feat_d, feat_h, feat_w),
                    mode="nearest",
                )[0, 0]
            slic_flat = slic_b.reshape(-1).long()
            for t in range(num_steps):
                if not bool(valid_mask[b, t].detach().item()):
                    per_time.append(cbct_features.new_zeros((self.num_prototypes, self.embed_dim)))
                    continue
                subregion_features, subregion_labels = self._subregion_mean_features(cbct_features[b, t], slic_flat)
                if return_details:
                    proto, assignment, bank, cluster, diversity = self.prototype_module(
                        subregion_features,
                        return_details=True,
                    )
                    cluster_total = cluster_total + cluster
                    diversity_total = diversity_total + diversity
                    loss_count += 1
                    per_assignment.append(
                        {
                            "subregion_labels": subregion_labels.detach(),
                            "assignment": assignment.detach(),
                            "prototype_bank": bank.detach(),
                        }
                    )
                else:
                    proto = self.prototype_module(subregion_features)
                per_time.append(proto)
            proto_by_case.append(torch.stack(per_time, dim=0).transpose(0, 1))
            assignments.append(per_assignment)

        proto_trajs = torch.stack(proto_by_case, dim=0)
        if loss_count > 0:
            cluster_total = cluster_total / loss_count
            diversity_total = diversity_total / loss_count
        return proto_trajs, cluster_total, diversity_total, assignments

    def _flatten_features(
        self,
        tokens: torch.Tensor,
        ct_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.flatten_token_norm is not None
        assert self.flatten_ct_norm is not None
        bsz, num_proto, num_steps, channels = tokens.shape
        max_steps = self.classifier_max_timepoints
        if num_steps > max_steps:
            raise ValueError(f"num_timepoints={num_steps} exceeds classifier_max_timepoints={max_steps}.")
        valid = valid_mask.to(device=tokens.device, dtype=tokens.dtype)
        z = self.flatten_token_norm(tokens) * valid[:, None, :, None]
        if num_steps < max_steps:
            pad_tokens = z.new_zeros((bsz, num_proto, max_steps - num_steps, channels))
            pad_valid = valid.new_zeros((bsz, max_steps - num_steps))
            z = torch.cat([z, pad_tokens], dim=2)
            valid = torch.cat([valid, pad_valid], dim=1)
        flat_tokens = z.reshape(bsz, num_proto * max_steps * channels)
        features = torch.cat([flat_tokens, self.flatten_ct_norm(ct_features), valid], dim=-1)
        token_valid = valid[:, None, :].expand(bsz, num_proto, max_steps)
        attention = token_valid / token_valid.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        return features, attention

    def forward(
        self,
        ct: torch.Tensor,
        cbct: torch.Tensor,
        cbct_valid_mask: torch.Tensor,
        slic: torch.Tensor,
        cbct_days_from_ct: torch.Tensor | None = None,
        cbct_days_from_prev_cbct: torch.Tensor | None = None,
        return_aux_loss: bool = False,
        return_interpretability: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        bsz, num_steps, _, depth, height, width = cbct.shape
        if ct.shape != (bsz, 1, depth, height, width):
            raise ValueError("ct must have shape [B, 1, D, H, W] matching cbct.")
        if slic.shape != (bsz, 1, depth, height, width):
            raise ValueError("slic must have shape [B, 1, D, H, W] matching cbct.")

        ct_features = self._encode_volumes(ct, self.ct_encoder_chunk_size).mean(dim=(2, 3, 4))
        cbct_features = self._encode_cbct(cbct, cbct_valid_mask)
        need_details = return_aux_loss or return_interpretability
        proto_trajs, cluster_loss, diversity_loss, assignments = self._prototype_trajectories(
            cbct_features,
            slic,
            cbct_valid_mask,
            return_details=need_details,
        )

        time_details: dict[str, torch.Tensor] = {}
        if self.time_embedding is not None:
            if cbct_days_from_ct is None or cbct_days_from_prev_cbct is None:
                raise ValueError("dual-clock time embedding requires cbct_days_from_ct and cbct_days_from_prev_cbct.")
            proto_with_time, time_details = self.time_embedding(
                proto_trajs,
                cbct_days_from_ct,
                cbct_days_from_prev_cbct,
                cbct_valid_mask,
                return_details=True,
            )
        else:
            proto_with_time = proto_trajs

        if self.mask_padding_before_temporal:
            proto_with_time = proto_with_time * cbct_valid_mask.to(proto_with_time.dtype)[:, None, :, None]
        temporal_tokens = self.temporal_encoder(proto_with_time)
        if self.mask_padding_before_temporal:
            temporal_tokens = temporal_tokens * cbct_valid_mask.to(temporal_tokens.dtype)[:, None, :, None]

        if self.pooling_mode == "flatten_tokens":
            classifier_features, attention = self._flatten_features(temporal_tokens, ct_features, cbct_valid_mask)
            pooled_cbct = classifier_features
        elif self.pooling_mode == "region_time_attention":
            assert self.region_time_pool is not None
            pooled_cbct, attention = self.region_time_pool(
                temporal_tokens,
                ct_features,
                cbct_valid_mask,
                return_attention=True,
            )
            classifier_features = torch.cat([pooled_cbct, ct_features], dim=-1)
        else:
            token_mean = temporal_tokens.mean(dim=1)
            valid = cbct_valid_mask.to(dtype=token_mean.dtype).unsqueeze(-1)
            pooled_cbct = (token_mean * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            classifier_features = torch.cat([pooled_cbct, ct_features], dim=-1)
            token_valid = cbct_valid_mask.to(temporal_tokens.dtype)[:, None, :].expand(
                bsz,
                temporal_tokens.shape[1],
                num_steps,
            )
            attention = token_valid / token_valid.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)

        logits = self.classifier(classifier_features).squeeze(-1)
        if not need_details:
            return logits
        return {
            "logits": logits,
            "proto_trajs": proto_with_time,
            "raw_proto_trajs": proto_trajs,
            "temporal_tokens": temporal_tokens,
            "cluster_loss": cluster_loss,
            "diversity_loss": diversity_loss,
            "ct_features": ct_features,
            "pooled_cbct": pooled_cbct,
            "classifier_features": classifier_features,
            "region_time_attention": attention,
            "prototype_assignment": assignments,
            **time_details,
        }
