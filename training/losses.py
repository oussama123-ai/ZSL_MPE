"""
Training Losses (Section 3.3.5, 3.4.4)
========================================
  L_total = L_pain + λ1·L_contrast + λ2·L_domain + λ3·L_consist
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PainLoss(nn.Module):
    """
    Multi-task pain estimation loss (Eq. 49).

    L_total_pain = L_MSE + 0.2·L_BCE(binary) + 0.1·L_CE(category) + 0.1·L_AU
    """

    BINARY_THRESHOLD = 2.0 / 10.0   # normalised (pain labels in [0, 1])

    def __init__(self, cfg: dict):
        super().__init__()
        aux = cfg.get("training", {}).get("aux_loss", {})
        self.w_binary   = aux.get("binary_weight",   0.2)
        self.w_category = aux.get("category_weight", 0.1)
        self.w_au       = aux.get("au_weight",       0.1)
        self.bin_thresh = aux.get("binary_threshold", 2.0) / 10.0

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        pain_gt: torch.Tensor,      # (B,) normalised [0, 1]
        au_gt: Optional[torch.Tensor] = None,   # (B, 6)
    ) -> Dict[str, torch.Tensor]:
        # Main regression loss
        pain_pred = preds["pain"] / 10.0     # normalise predicted pain to [0, 1]
        l_mse = F.mse_loss(pain_pred, pain_gt)

        # Binary pain classification (pain ≥ threshold)
        bin_gt   = (pain_gt >= self.bin_thresh).float()
        l_binary = F.binary_cross_entropy_with_logits(preds["binary"], bin_gt)

        # 4-class severity classification
        cat_gt = torch.clamp((pain_gt * 10 / 2.5).long(), 0, 3)   # 4 bins
        l_cat  = F.cross_entropy(preds["category"], cat_gt)

        # AU multi-label classification
        l_au = torch.tensor(0.0, device=pain_gt.device)
        if au_gt is not None:
            l_au = F.binary_cross_entropy(preds["au"], au_gt.clamp(0, 1))

        total = (l_mse
                 + self.w_binary   * l_binary
                 + self.w_category * l_cat
                 + self.w_au       * l_au)

        return {
            "pain_total":  total,
            "mse":         l_mse,
            "binary":      l_binary,
            "category":    l_cat,
            "au":          l_au,
        }


class TotalTrainingLoss(nn.Module):
    """
    Combined loss for all three training stages (Eq. 59).

    L = L_pain + λ1·L_contrast + λ2·L_domain + λ3·L_consist
    """

    def __init__(self, cfg: dict):
        super().__init__()
        align = cfg.get("alignment", {})
        self.lambda1 = align.get("lambda1", 0.5)
        self.lambda2 = align.get("lambda2", 0.3)
        self.lambda3 = align.get("lambda3", 0.2)
        self.pain_loss = PainLoss(cfg)

    def forward(
        self,
        preds_synth: Dict[str, torch.Tensor],
        pain_gt: torch.Tensor,
        au_gt: Optional[torch.Tensor] = None,
        l_contrast: Optional[torch.Tensor] = None,
        l_domain: Optional[torch.Tensor] = None,
        l_consist: Optional[torch.Tensor] = None,
        stage: int = 1,
    ) -> Dict[str, torch.Tensor]:
        losses = self.pain_loss(preds_synth, pain_gt, au_gt)
        total  = losses["pain_total"]

        if stage >= 2 and l_contrast is not None:
            total = total + self.lambda1 * l_contrast
        if stage >= 2 and l_domain is not None:
            total = total + self.lambda2 * l_domain
        if stage >= 2 and l_consist is not None:
            total = total + self.lambda3 * l_consist

        losses.update({
            "total":       total,
            "l_contrast":  l_contrast if l_contrast is not None else torch.tensor(0.0),
            "l_domain":    l_domain   if l_domain   is not None else torch.tensor(0.0),
            "l_consist":   l_consist  if l_consist  is not None else torch.tensor(0.0),
        })
        return losses
