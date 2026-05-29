from __future__ import annotations

import torch.nn as nn


class SurvivalPredictionHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.layers(x)
