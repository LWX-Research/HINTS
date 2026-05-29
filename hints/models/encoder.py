from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ..settings import Settings


class SubregionEncoder(nn.Module):
    def __init__(self, settings: Settings, modalities: list[str]):
        super().__init__()
        self.settings = settings
        self.modalities = modalities
        self.gcn_dim = settings.hidden_dim
        self.nodes_num = settings.num_nodes

        self.modal_encoders = nn.ModuleDict(
            {
                modal: nn.ModuleList(
                    [
                        GCNConv(settings.in_channels, self.gcn_dim // 2),
                        GCNConv(self.gcn_dim // 2, self.gcn_dim),
                    ]
                )
                for modal in self.modalities
            }
        )
        self.bn_layers = nn.ModuleDict(
            {
                modal: nn.ModuleDict({"1": nn.BatchNorm1d(self.gcn_dim // 2), "2": nn.BatchNorm1d(self.gcn_dim)})
                for modal in self.modalities
            }
        )

    def encode_modality(self, x: torch.Tensor, edge_index: torch.Tensor, modality: str) -> torch.Tensor:
        x = self.modal_encoders[modality][0](x, edge_index)
        x = F.relu(self.bn_layers[modality]["1"](x))
        x = self.modal_encoders[modality][1](x, edge_index)
        x = self.bn_layers[modality]["2"](x)
        batch_graphs = x.size(0) // self.nodes_num
        if batch_graphs * self.nodes_num != x.size(0):
            raise ValueError(f"Unexpected node count for modality {modality}: {x.size(0)}")
        return F.relu(x.view(batch_graphs, self.nodes_num, -1))

    def forward(self, data) -> dict[str, torch.Tensor]:
        encoded = {}
        for modality in self.modalities:
            encoded_feat = self.encode_modality(getattr(data, f"{modality}_x"), getattr(data, f"{modality}_edge_index"), modality)
            original_view = getattr(data, f"{modality}_x").view(-1, self.nodes_num, self.settings.in_channels)
            encoded[modality] = torch.cat([original_view, encoded_feat], dim=2)
        return encoded
