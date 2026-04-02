"""
Multimodal Fusion Transformer (Section 3.3.4)
==============================================
Cross-modal transformer with attention pooling and pain intensity
regression head (Section 3.3.5).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Learns adaptive weights over multimodal tokens (Eq. 44–45).
    Provides modality importance scores for interpretability.
    """

    def __init__(self, d_model: int, n_modalities: int = 4):
        super().__init__()
        self.W_pool = nn.Linear(d_model, d_model // 2)
        self.w      = nn.Linear(d_model // 2, 1, bias=False)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        H : (B, M, D)   — M multimodal tokens
        returns:
          z_fused : (B, D)
          alphas  : (B, M)  — modality importance scores
        """
        scores = self.w(torch.tanh(self.W_pool(H))).squeeze(-1)  # (B, M)
        alphas = torch.softmax(scores, dim=-1)                    # (B, M)
        z      = (alphas.unsqueeze(-1) * H).sum(dim=1)            # (B, D)
        return z, alphas


class FusionTransformer(nn.Module):
    """
    12-layer cross-modal transformer (Eq. 39–45).

    Modalities: visual (v), physiological (p), contextual (c), action units (a).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d_model    = cfg.get("d_model", 512)
        n_layers   = cfg.get("n_layers", 12)
        n_heads    = cfg.get("n_heads", 12)
        d_ff       = cfg.get("d_ff", 2048)
        dropout    = cfg.get("dropout", 0.1)
        n_mod      = 4     # visual, physio, context, AU

        # Per-modality linear projections to d_model (Eq. 35–38)
        self.proj_v = nn.Linear(512, d_model)   # visual
        self.proj_p = nn.Linear(512, d_model)   # physio
        self.proj_c = nn.Linear(256, d_model)   # context
        self.proj_a = nn.Linear(6,   d_model)   # AU

        # Learnable modality-type embeddings (Eq. 39)
        self.mod_embed = nn.Embedding(n_mod, d_model)

        # Transformer encoder stack (Eq. 40–43)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Attention pooling (Eq. 44–45)
        self.attn_pool = AttentionPooling(d_model, n_mod)

    def forward(
        self,
        hv: torch.Tensor,   # (B, 512)
        hp: torch.Tensor,   # (B, 512)
        hc: torch.Tensor,   # (B, 256)
        ha: torch.Tensor,   # (B, 6)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_fused : (B, d_model)
          alphas  : (B, 4)        modality importance scores
        """
        B = hv.size(0)
        device = hv.device

        # Project to d_model (Eq. 35–38)
        hv_ = self.proj_v(hv)   # (B, d_model)
        hp_ = self.proj_p(hp)
        hc_ = self.proj_c(hc)
        ha_ = self.proj_a(ha)

        # Stack + add modality embeddings (Eq. 39)
        ids = torch.arange(4, device=device)
        mod = self.mod_embed(ids).unsqueeze(0)          # (1, 4, d_model)
        H   = torch.stack([hv_, hp_, hc_, ha_], dim=1)  # (B, 4, d_model)
        H   = H + mod

        # Transformer (Eq. 40–43)
        H = self.transformer(H)                         # (B, 4, d_model)

        # Attention pooling (Eq. 44–45)
        z_fused, alphas = self.attn_pool(H)             # (B, D), (B, 4)
        return z_fused, alphas


class PainRegressionHead(nn.Module):
    """
    3-layer MLP with residual connections → continuous pain ∈ [0, 10] (Eq. 46–48).

    Also provides binary and category classification heads for multi-task
    training (Eq. 49).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        dims    = cfg.get("hidden_dims", [256, 128])
        drop    = cfg.get("dropout", 0.3)
        d_in    = cfg.get("d_model", 512)
        scale   = cfg.get("output_scale", 10.0)
        self.scale = scale

        # Layer 1: 512 → 256  (with residual projection)
        self.fc1 = nn.Linear(d_in, dims[0])
        self.res1 = nn.Linear(d_in, dims[0])
        self.drop1 = nn.Dropout(drop)

        # Layer 2: 256 → 128  (with residual projection)
        self.fc2 = nn.Linear(dims[0], dims[1])
        self.res2 = nn.Linear(dims[0], dims[1])
        self.drop2 = nn.Dropout(drop)

        # Output: 128 → 1, bounded [0, scale]
        self.out = nn.Linear(dims[1], 1)

        # Auxiliary heads (Eq. 49)
        self.binary_head   = nn.Linear(dims[0], 1)   # pain present/absent
        self.category_head = nn.Linear(dims[0], 4)   # no/mild/moderate/severe
        self.au_head       = nn.Linear(dims[0], 6)   # AU detection

    def forward(self, z: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """z: (B, d_model) → dict of predictions."""
        # Eq. 46
        z1 = F.relu(self.fc1(z) + self.res1(z))
        z1 = self.drop1(z1)

        # Eq. 47
        z2 = F.relu(self.fc2(z1) + self.res2(z1))
        z2 = self.drop2(z2)

        # Eq. 48 — sigmoid to bound [0, scale]
        pain = torch.sigmoid(self.out(z2)).squeeze(-1) * self.scale

        return {
            "pain":     pain,                                         # (B,)
            "binary":   self.binary_head(z1).squeeze(-1),            # (B,) logit
            "category": self.category_head(z1),                      # (B, 4) logit
            "au":       torch.sigmoid(self.au_head(z1)),             # (B, 6)
        }
