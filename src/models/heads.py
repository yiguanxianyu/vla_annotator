from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class KHead(nn.Module):
    def __init__(self, in_dim: int, k_max: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, k_max + 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: [B, D]
        return self.mlp(pooled)


class StartEndHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.start = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 1),
        )
        self.end = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 1),
        )

    def forward(self, frame_feats: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        # frame_feats: [B, T, D]
        B, T, D = frame_feats.shape
        x = frame_feats.reshape(B * T, D)
        s = self.start(x).reshape(B, T)
        e = self.end(x).reshape(B, T)
        if valid_mask is not None:
            # mask out padding positions
            s = s.masked_fill(valid_mask <= 0, 0.0)
            e = e.masked_fill(valid_mask <= 0, 0.0)
        return s, e

