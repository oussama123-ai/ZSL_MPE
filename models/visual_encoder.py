"""
Visual Encoder (Section 3.3.1)
===============================
  - ViT-Base/16 for spatial frame features
  - Lightweight temporal transformer for sequence aggregation
  - Multi-label AU detection head for interpretability
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    """3-layer transformer encoder for temporal aggregation across video frames."""

    def __init__(self, d_model: int = 768, n_layers: int = 3,
                 n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos_embed = nn.Embedding(256, d_model)   # up to 256 frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        return self.encoder(x)


class VisualEncoder(nn.Module):
    """
    ViT-Base/16 + temporal transformer → 512-dim video representation.

    Also outputs AU detection head for interpretability (Eq. 23).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        vit_dim     = cfg.get("embed_dim", 768)
        output_dim  = cfg.get("output_dim", 512)
        n_au        = cfg.get("num_au", 6)
        pretrained  = cfg.get("pretrained", True)

        # Vision Transformer backbone
        self.vit = self._load_vit(pretrained)
        self.vit_dim = vit_dim

        # Temporal transformer (Eq. 21)
        self.temporal = TemporalTransformer(
            d_model=vit_dim,
            n_layers=cfg.get("temporal_transformer_layers", 3),
            n_heads=cfg.get("temporal_heads", 8),
        )

        # Projection to shared embedding dim (Eq. 22)
        self.proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, output_dim),
        )

        # Action Unit detection head (Eq. 23)
        self.au_head = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_au),
            nn.Sigmoid(),
        )

    @staticmethod
    def _load_vit(pretrained: bool) -> nn.Module:
        try:
            import timm
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,          # remove classification head
            )
            return model
        except ImportError:
            pass

        # Fallback: minimal ViT stub
        return _ViTStub(embed_dim=768, patch_size=16, img_size=224)

    def forward(self, video: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        video : (B, T, 3, H, W)
        returns:
          hv_final : (B, output_dim)
          au_preds : (B, n_au)
        """
        B, T, C, H, W = video.shape

        # Extract per-frame features via ViT
        frames = video.view(B * T, C, H, W)
        hv_frames = self.vit(frames)        # (B*T, vit_dim)
        hv_frames = hv_frames.view(B, T, self.vit_dim)

        # AU prediction from averaged frame features
        au_preds = self.au_head(hv_frames.mean(dim=1))   # (B, n_au)

        # Temporal aggregation (Eq. 21–22)
        hv_temp   = self.temporal(hv_frames)              # (B, T, vit_dim)
        hv_pooled = hv_temp.mean(dim=1)                   # (B, vit_dim)
        hv_final  = self.proj(hv_pooled)                  # (B, output_dim)

        return hv_final, au_preds


# ---------------------------------------------------------------------------
# Minimal ViT stub (used when timm is unavailable)
# ---------------------------------------------------------------------------

class _ViTStub(nn.Module):
    """Minimal ViT: patch embedding + 4 transformer layers."""

    def __init__(self, embed_dim: int = 768, patch_size: int = 16,
                 img_size: int = 224):
        super().__init__()
        n_patches    = (img_size // patch_size) ** 2
        patch_dim    = 3 * patch_size * patch_size
        self.embed   = nn.Linear(patch_dim, embed_dim)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        layer        = nn.TransformerEncoderLayer(
            embed_dim, nhead=12, dim_feedforward=3072,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm    = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, embed_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)              # (B, 3, nh, nw, p, p)
        x = x.contiguous().view(B, C, -1, p * p)            # (B, 3, N, p²)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)  # (B, N, patch_dim)
        x = self.embed(x)

        cls = self.cls_tok.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_emb
        x   = self.norm(self.encoder(x))
        return x[:, 0]   # CLS token
