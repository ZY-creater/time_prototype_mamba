from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_contrastive_loss(
    proto_trajs: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
    tau: float = 0.1,
    negative_class_weight: float = 0.5,
    positive_class_weight: float = 1.5,
) -> torch.Tensor:
    """InfoNCE-style consistency for the same prototype across adjacent visits.

    If labels are provided, the loss supports class-dependent weights. The
    public default assumes label 1 is the positive class used by the binary
    classifier. Set both weights to 1.0 to disable class-dependent weighting.
    """
    bsz, num_proto, _, _ = proto_trajs.shape
    total = proto_trajs.new_tensor(0.0)
    count = 0
    tau = max(float(tau), 1e-6)
    for b in range(bsz):
        weight = proto_trajs.new_tensor(1.0)
        if labels is not None:
            label = int(labels[b].detach().item())
            weight = proto_trajs.new_tensor(
                float(positive_class_weight if label == 1 else negative_class_weight)
            )
        valid_steps = torch.where(valid_mask[b])[0]
        if valid_steps.numel() < 2:
            continue
        for i in range(valid_steps.numel() - 1):
            t0 = int(valid_steps[i].item())
            t1 = int(valid_steps[i + 1].item())
            z0 = F.normalize(proto_trajs[b, :, t0, :], dim=-1)
            z1 = F.normalize(proto_trajs[b, :, t1, :], dim=-1)
            sim = (z0 @ z1.t()) / tau
            target = torch.arange(num_proto, device=proto_trajs.device)
            total = total + weight * F.cross_entropy(sim, target)
            count += 1
    return total / max(1, count)


def temporal_gap_smoothness_loss(
    proto_trajs: torch.Tensor,
    valid_mask: torch.Tensor,
    gap_days: torch.Tensor | None,
    tau_days: float = 7.0,
) -> torch.Tensor:
    """Penalize short-interval phenotype jumps more than long-interval jumps."""
    if gap_days is None:
        return proto_trajs.new_tensor(0.0)
    total = proto_trajs.new_tensor(0.0)
    count = 0
    tau = max(float(tau_days), 1e-6)
    for b in range(proto_trajs.shape[0]):
        valid_steps = torch.where(valid_mask[b])[0]
        if valid_steps.numel() < 2:
            continue
        for i in range(valid_steps.numel() - 1):
            t0 = int(valid_steps[i].item())
            t1 = int(valid_steps[i + 1].item())
            z0 = F.normalize(proto_trajs[b, :, t0, :], dim=-1)
            z1 = F.normalize(proto_trajs[b, :, t1, :], dim=-1)
            gap = gap_days[b, t1].to(device=proto_trajs.device, dtype=proto_trajs.dtype)
            gap = torch.nan_to_num(gap, nan=0.0, posinf=tau * 10.0, neginf=0.0).clamp(min=0.0)
            total = total + torch.exp(-gap / tau) * ((z1 - z0) ** 2).mean()
            count += 1
    return total / max(1, count)
