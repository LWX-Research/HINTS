from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..settings import Settings


class PrototypeDisentangler(nn.Module):
    def __init__(self, settings: Settings, modalities: list[str], feature_dim: int):
        super().__init__()
        self.settings = settings
        self.modalities = modalities
        self.feature_dim = feature_dim

        self.prototypes = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(settings.num_prototypes, feature_dim)) for name in [*modalities, "share"]}
        )
        self.mlp_dict = nn.ModuleDict({name: self._build_pool_head() for name in [*modalities, "share"]})

    def _build_pool_head(self):
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 1),
        )

    def update_prototypes(self, features, modality: str, momentum: float = 0.9):
        combined = torch.cat([self.prototypes["share"], self.prototypes[modality]], dim=0)
        query = combined.unsqueeze(0).expand(features.size(0), -1, -1)
        attention_scores = torch.bmm(query, features.transpose(1, 2)) / (query.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        updated = torch.einsum("bnm,bmd->bnd", attention_weights, features)
        updated = query + F.layer_norm(updated, updated.shape[-1:])

        with torch.no_grad():
            updated_shared = updated[:, : self.settings.num_prototypes, :]
            updated_modality = updated[:, self.settings.num_prototypes :, :]
            if self.training:
                self.prototypes["share"] = self.prototypes["share"] * momentum + (1 - momentum) * updated_shared.mean(0)
                self.prototypes[modality] = self.prototypes[modality] * momentum + (1 - momentum) * updated_modality.mean(0)

        shared_weights = F.softmax(self.mlp_dict["share"](updated_shared).squeeze(-1), dim=-1)
        shared_feature = torch.einsum("bn,bnd->bd", shared_weights, updated_shared)

        modality_weights = F.softmax(self.mlp_dict[modality](updated_modality).squeeze(-1), dim=-1)
        modality_feature = torch.einsum("bn,bnd->bd", modality_weights, updated_modality)
        return shared_feature, modality_feature

    def forward(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        updated = [self.update_prototypes(encoded[modality], modality) for modality in self.modalities]
        shared = torch.stack([item[0] for item in updated], dim=-1)
        distinct = torch.stack([item[1] for item in updated], dim=-1)
        return shared, distinct
