from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..settings import Settings
from .disentangle import PrototypeDisentangler
from .encoder import SubregionEncoder
from .fusion import StructuralQuaternionFusion
from .head import SurvivalPredictionHead


class MultiModalGCN(nn.Module):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.gcn_dim = settings.hidden_dim
        self.dim = settings.in_channels + self.gcn_dim
        self.nodes_num = settings.num_nodes
        self.modalities = ["t1", "t1ce", "t2"]

        self.encoder = SubregionEncoder(settings, self.modalities)
        self.disentangler = PrototypeDisentangler(settings, self.modalities, feature_dim=self.dim)
        self.fusion = StructuralQuaternionFusion(feature_dim=self.dim * 2)
        self.prediction_head = SurvivalPredictionHead(input_dim=self.dim * 2)
        self.slice_attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, 1),
        )
        self.last_slice_attention = None

    def forward(self, data):
        encoded = self.encoder(data)
        shared, distinct = self.disentangler(encoded)
        features = torch.cat([shared, distinct], dim=1)
        quaternion_features = torch.cat([torch.zeros_like(features[:, :, 0].unsqueeze(-1)), features], dim=-1)
        fused = self.fusion(quaternion_features)
        if hasattr(data, "slice_to_patient"):
            fused = self.aggregate_slices(fused, data.slice_to_patient, data.num_patients)
        prediction = self.prediction_head(fused)
        return prediction, shared, distinct

    def aggregate_slices(self, slice_features: torch.Tensor, slice_to_patient: torch.Tensor, num_patients: int) -> torch.Tensor:
        attention_logits = self.slice_attention(slice_features).squeeze(-1)
        patient_embeddings = []
        attention_weights = torch.zeros_like(attention_logits)
        for patient_index in range(num_patients):
            mask = slice_to_patient == patient_index
            patient_logits = attention_logits[mask]
            patient_weights = F.softmax(patient_logits, dim=0)
            attention_weights[mask] = patient_weights
            patient_embeddings.append((patient_weights.unsqueeze(-1) * slice_features[mask]).sum(dim=0))
        self.last_slice_attention = attention_weights
        return torch.stack(patient_embeddings, dim=0)
