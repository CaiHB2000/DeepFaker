from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityEstimator(nn.Module):
    """
    Simple event–modality reliability network r_ϕ(k, e) -> (0,1).

    Inputs:
      - modality_ids: LongTensor (B,) with values {0=text,1=img,2=fuse}
      - event_ids: LongTensor (B,), -1 for unknown
      - extra_feats: FloatTensor (B, D) optional

    Output:
      - pi: FloatTensor (B,) in (0,1)
    """

    def __init__(
        self,
        num_modalities: int = 3,
        num_events: int = 5000,
        event_dim: int = 32,
        hidden_dim: int = 64,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.event_emb = nn.Embedding(num_events, event_dim)
        self.mod_emb = nn.Embedding(num_modalities, event_dim)
        in_dim = event_dim * 2 + extra_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        modality_ids: torch.Tensor,
        event_ids: torch.Tensor,
        extra_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = modality_ids.device
        event_ids_clamped = event_ids.clamp(min=0)
        event_vec = self.event_emb(event_ids_clamped)
        mod_vec = self.mod_emb(modality_ids.clamp(min=0))
        feats = [event_vec, mod_vec]
        if extra_feats is not None:
            feats.append(extra_feats)
        x = torch.cat(feats, dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logits)
