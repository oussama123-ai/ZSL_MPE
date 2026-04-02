"""
Context Encoder (Section 3.3.3)
================================
Demographic embeddings (age, ethnicity, sex) + clinical context
encoding → 256-dim contextual representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Encodes demographic and clinical context (Eq. 30–34).

    Inputs
    ------
    age       : (B,)          long — age group index [0, 3]
    ethnicity : (B,)          long — ethnicity index  [0, 4]
    sex       : (B,)          long — sex index        [0, 1]
    v_setting : (B, 5)        float — multi-hot clinical setting
    v_type    : (B, 5)        float — multi-hot pain type

    Output
    ------
    hc : (B, output_dim)
    """

    N_AGE_GROUPS  = 4
    N_ETHNICITIES = 5
    N_SEX         = 2

    def __init__(self, cfg: dict):
        super().__init__()
        age_dim  = cfg.get("age_embed_dim", 32)
        eth_dim  = cfg.get("ethnicity_embed_dim", 32)
        sex_dim  = cfg.get("sex_embed_dim", 16)
        ctx_dim  = cfg.get("clinical_embed_dim", 64)
        out_dim  = cfg.get("output_dim", 256)
        ctx_in   = cfg.get("clinical_input_dim", 10)   # 5 settings + 5 types

        # Lookup tables (Eq. 30–32)
        self.age_embed = nn.Embedding(self.N_AGE_GROUPS,  age_dim)
        self.eth_embed = nn.Embedding(self.N_ETHNICITIES,  eth_dim)
        self.sex_embed = nn.Embedding(self.N_SEX,          sex_dim)

        # Clinical context MLP (Eq. 33)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_in, ctx_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim * 2, ctx_dim),
            nn.ReLU(inplace=True),
        )

        # Combined projection (Eq. 34)
        combined = age_dim + eth_dim + sex_dim + ctx_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(combined),
            nn.Linear(combined, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        age: torch.Tensor,          # (B,)
        ethnicity: torch.Tensor,    # (B,)
        sex: torch.Tensor,          # (B,)
        v_setting: torch.Tensor,    # (B, 5)
        v_type: torch.Tensor,       # (B, 5)
    ) -> torch.Tensor:
        e_age = self.age_embed(age)           # (B, age_dim)
        e_eth = self.eth_embed(ethnicity)     # (B, eth_dim)
        e_sex = self.sex_embed(sex)           # (B, sex_dim)

        v_ctx = torch.cat([v_setting, v_type], dim=-1)   # (B, 10)
        e_ctx = self.ctx_mlp(v_ctx)                       # (B, ctx_dim)

        combined = torch.cat([e_age, e_eth, e_sex, e_ctx], dim=-1)
        return self.proj(combined)                        # (B, output_dim)
