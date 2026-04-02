"""
Physiological Signal Encoder (Section 3.3.2)
=============================================
Signal-specific 1D CNN encoders for HRV, EDA, and tremor.
Self-attention multi-signal fusion → 512-dim physiological representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DEncoder(nn.Module):
    """
    4-layer 1D CNN for a single physiological signal.
    Captures multi-scale temporal patterns (Eq. 24–26).
    """

    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        output_dim: int = 128,
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()
        assert len(channels) == len(kernels) == 3

        layers = []
        c_in = in_channels
        for c_out, k in zip(channels, kernels):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c_in = c_out

        layers.append(nn.AdaptiveAvgPool1d(1))   # global avg pool → (B, C, 1)
        self.net   = nn.Sequential(*layers)
        self.proj  = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) → (B, output_dim)"""
        h = self.net(x).squeeze(-1)    # (B, C)
        return self.proj(h)             # (B, output_dim)


class PhysioEncoder(nn.Module):
    """
    Three signal-specific 1D CNN encoders + self-attention fusion.
    Outputs fused_physio_dim-dimensional physiological representation.
    (Eq. 27–29)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        out_dim  = cfg.get("output_dim", 128)           # per-signal dim
        fused    = cfg.get("fused_physio_dim", 512)
        dropout  = cfg.get("dropout", 0.3)
        n_heads  = cfg.get("self_attention_heads", 4)

        hrv_ch = cfg.get("hrv_channels", [16, 32, 64])
        hrv_k  = cfg.get("hrv_kernels",  [31, 15, 7])
        scr_ch = cfg.get("scr_channels", [16, 32, 64])
        scr_k  = cfg.get("scr_kernels",  [31, 15, 7])
        trm_ch = cfg.get("tremor_channels", [16, 32, 64])
        trm_k  = cfg.get("tremor_kernels",  [31, 15, 7])

        self.hrv_enc    = Conv1DEncoder(hrv_ch, hrv_k, out_dim, dropout)
        self.scr_enc    = Conv1DEncoder(scr_ch, scr_k, out_dim, dropout)
        self.tremor_enc = Conv1DEncoder(trm_ch, trm_k, out_dim, dropout)

        # Self-attention over 3 signals (Eq. 28)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )

        # Final projection (Eq. 29)
        self.proj = nn.Linear(3 * out_dim, fused)
        self.norm = nn.LayerNorm(3 * out_dim)

    def forward(
        self,
        hrv: torch.Tensor,      # (B, 1, L_hrv)
        eda: torch.Tensor,      # (B, 1, L_eda)
        tremor: torch.Tensor,   # (B, 1, L_trem)
    ) -> torch.Tensor:
        """Returns (B, fused_physio_dim)."""
        h_hrv    = self.hrv_enc(hrv)       # (B, out_dim)
        h_scr    = self.scr_enc(eda)       # (B, out_dim)
        h_tremor = self.tremor_enc(tremor) # (B, out_dim)

        # Stack as sequence: (B, 3, out_dim)
        hp = torch.stack([h_hrv, h_scr, h_tremor], dim=1)

        # Self-attention (Eq. 28)
        hp_attn, _ = self.self_attn(hp, hp, hp)  # (B, 3, out_dim)

        # Flatten and project (Eq. 29)
        hp_flat = hp_attn.flatten(1)              # (B, 3*out_dim)
        hp_flat = self.norm(hp_flat)
        return self.proj(hp_flat)                 # (B, fused_physio_dim)
