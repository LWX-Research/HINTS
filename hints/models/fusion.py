from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quaternion import QuaternionLinear, qsvd_reconstruction


class StructuralQuaternionFusion(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.q_linear = nn.Sequential(
            QuaternionLinear(feature_dim, 256),
            QuaternionLinear(256, feature_dim),
        )
        self.linear_q = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        query = self.linear_q(torch.mean(features, dim=-1)).float()
        projected = self.q_linear(features.unsqueeze(1)).float()
        projected = qsvd_reconstruction(projected).squeeze(1)
        attention = torch.einsum("bd,bdn->bn", query, projected) / (query.size(-1) ** 0.5)
        attention = F.softmax(attention, dim=-1)
        return torch.einsum("bn,bdn->bd", attention, projected).to(query.dtype)
