"""
Pain Estimator — Full Model Wrapper
=====================================
Combines all sub-modules into one nn.Module for easy use.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.visual_encoder   import VisualEncoder
from models.physio_encoder   import PhysioEncoder
from models.context_encoder  import ContextEncoder
from models.fusion_transformer import FusionTransformer, PainRegressionHead


class PainEstimator(nn.Module):
    """
    Full zero-shot multimodal pain estimator.

    Architecture
    ------------
    video  ──► VisualEncoder  ──► hv (512)  ──┐
    hrv  ─┐                                    │
    eda  ─┼─► PhysioEncoder  ──► hp (512)  ──┤
    trem ─┘                                    ├──► FusionTransformer ──► PainHead
    age  ─┐                                    │
    eth  ─┤                                    │
    sex  ─┼─► ContextEncoder ──► hc (256)  ──┤
    set  ─┤                                    │
    type ─┘                                    │
    au   ──────────────────────► ha (6)    ──┘
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.visual_enc  = VisualEncoder(cfg["model"]["visual"])
        self.physio_enc  = PhysioEncoder(cfg["model"]["physio"])
        self.context_enc = ContextEncoder(cfg["model"]["context"])
        self.fusion      = FusionTransformer(cfg["model"]["fusion"])
        self.head        = PainRegressionHead({
            **cfg["model"]["regression"],
            "d_model": cfg["model"]["fusion"]["d_model"],
        })

    def encode(self, batch: Dict[str, torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (z_fused, alphas) where z_fused is the domain-alignable feature.
        """
        hv, au_preds = self.visual_enc(batch["video"])
        hp = self.physio_enc(batch["hrv"], batch["eda"], batch["tremor"])
        hc = self.context_enc(
            batch["age"], batch["ethnicity"], batch["sex"],
            batch["clinical_setting"], batch["pain_type"],
        )
        ha = batch.get("au", au_preds)   # use provided AUs or predicted AUs

        z_fused, alphas = self.fusion(hv, hp, hc, ha)
        return z_fused, alphas

    def forward(self, batch: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """Full forward pass returning all predictions."""
        z_fused, alphas = self.encode(batch)
        preds = self.head(z_fused)
        preds["modality_weights"] = alphas
        return preds

    def predict_pain(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convenience: return pain scores only."""
        with torch.no_grad():
            return self.forward(batch)["pain"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
