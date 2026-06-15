from __future__ import annotations

import torch
from torch import nn


class PrototypeSubregionModule(nn.Module):
    """Learnable prototype bank over SLIC-style subregion features."""

    def __init__(self, feat_dim: int, num_prototypes: int = 24) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_prototypes = int(num_prototypes)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.feat_dim) * 0.02)

    def forward(
        self,
        subregion_features: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate variable-count subregions into a fixed prototype set.

        Args:
            subregion_features: Tensor of shape [S, C].
            return_details: Return soft assignments and regularization losses.

        Returns:
            Prototype embeddings with shape [K, C]. If ``return_details=True``,
            also returns assignment probabilities, the prototype bank, cluster
            loss, and diversity loss.
        """
        eps = 1e-6
        logits = subregion_features @ self.prototypes.t()
        assignment = torch.softmax(logits, dim=-1)
        numerator = assignment.t() @ subregion_features
        denominator = assignment.sum(dim=0).unsqueeze(1).clamp(min=eps)
        proto_embed = numerator / denominator

        if not return_details:
            return proto_embed

        diff = subregion_features.unsqueeze(1) - self.prototypes.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1)
        cluster_loss = (assignment * dist2).sum() / assignment.sum().clamp(min=eps)

        p = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True).clamp(min=eps)
        sim = p @ p.t()
        eye = torch.eye(self.num_prototypes, device=sim.device, dtype=sim.dtype)
        off_diag = sim * (1.0 - eye)
        diversity_loss = (off_diag * off_diag).sum() / max(1, self.num_prototypes * (self.num_prototypes - 1))

        return proto_embed, assignment, self.prototypes, cluster_loss, diversity_loss

