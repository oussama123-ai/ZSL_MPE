"""
Domain Alignment (Section 3.4)
================================
  - Gradient Reversal Layer (GRL)  — adversarial alignment (3.4.2)
  - Domain Discriminator
  - Momentum encoder for MoCo-style contrastive learning (3.4.1)
  - Contrastive loss (supervised + unsupervised)
  - Adversarial domain loss
  - Consistency regularization loss (3.4.3)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Section 3.4.2, Eq. 54)
# ---------------------------------------------------------------------------

class _GradReversalFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradReversalFn.apply(x, lambda_)


# ---------------------------------------------------------------------------
# Domain Discriminator (Section 3.4.2, Eq. 53)
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """3-layer MLP: predicts P(synthetic) ∈ [0, 1]."""

    def __init__(self, cfg: dict):
        super().__init__()
        in_dim   = cfg.get("input_dim",   512)
        h_dims   = cfg.get("hidden_dims", [256, 128])

        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(h_dims[0], h_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(h_dims[1], 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """z: (B, D) → (B,) probability of being synthetic."""
        z_rev = grad_reverse(z, lambda_)
        return self.net(z_rev).squeeze(-1)


# ---------------------------------------------------------------------------
# Progressive GRL schedule (Section 3.4.2, Eq. 56)
# ---------------------------------------------------------------------------

def grl_lambda(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """λ(e) grows from 0 to 1 over training."""
    p = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# ---------------------------------------------------------------------------
# Momentum Encoder for MoCo (Section 3.4.1)
# ---------------------------------------------------------------------------

class MomentumEncoder(nn.Module):
    """
    Exponential moving average copy of the main feature encoder.
    Used as key encoder in MoCo-style contrastive learning (Eq. 51).
    """

    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.momentum = momentum
        self.encoder  = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """θ_ema ← m·θ_ema + (1−m)·θ"""
        m = self.momentum
        for p_ema, p_online in zip(self.encoder.parameters(),
                                   online_encoder.parameters()):
            p_ema.data.mul_(m).add_(p_online.data, alpha=1.0 - m)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


# ---------------------------------------------------------------------------
# FIFO Negative Queue for MoCo (Section 3.4.1)
# ---------------------------------------------------------------------------

class NegativeQueue:
    """Circular FIFO queue storing K negative keys of dimension D."""

    def __init__(self, K: int = 16_384, D: int = 512, device: str = "cpu"):
        self.K      = K
        self.queue  = F.normalize(torch.randn(K, D, device=device), dim=1)
        self.ptr    = 0

    @torch.no_grad()
    def enqueue_dequeue(self, keys: torch.Tensor) -> None:
        """keys: (B, D), L2-normalized."""
        B = keys.size(0)
        end = (self.ptr + B) % self.K
        if end > self.ptr:
            self.queue[self.ptr:end] = keys.detach()
        else:
            self.queue[self.ptr:] = keys.detach()[:self.K - self.ptr]
            self.queue[:end]      = keys.detach()[self.K - self.ptr:]
        self.ptr = end

    def get_negatives(self) -> torch.Tensor:
        return self.queue.clone()


# ---------------------------------------------------------------------------
# Contrastive Losses (Section 3.4.1)
# ---------------------------------------------------------------------------

def supervised_contrastive_loss(
    z: torch.Tensor,           # (B, D) L2-normalized
    pain_labels: torch.Tensor, # (B,)   continuous
    tau: float = 0.07,
    pain_threshold: float = 2.0,
) -> torch.Tensor:
    """
    Supervised contrastive loss on synthetic samples (Eq. 50).
    Positives: samples with |p_i − p_j| < pain_threshold.
    """
    B = z.size(0)
    sim = torch.mm(z, z.T) / tau                      # (B, B)
    # Mask self-similarity
    mask_self = torch.eye(B, dtype=torch.bool, device=z.device)

    # Positive mask: close pain intensities
    pain_diff = (pain_labels.unsqueeze(0) - pain_labels.unsqueeze(1)).abs()
    pos_mask  = (pain_diff < pain_threshold) & ~mask_self  # (B, B)

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # For each anchor, compute log-softmax over all others
    sim_exp   = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
    sim_exp   = sim_exp.masked_fill(mask_self, 0)
    denom     = sim_exp.sum(dim=1, keepdim=True).clamp(min=1e-8)   # (B, 1)

    log_prob  = sim - torch.log(denom)   # (B, B)

    # Average over positives per anchor
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    loss  = -(log_prob * pos_mask).sum(dim=1) / n_pos
    return loss.mean()


def moco_contrastive_loss(
    q: torch.Tensor,           # (B, D) query (L2-normalized)
    k: torch.Tensor,           # (B, D) key   (L2-normalized)
    queue: torch.Tensor,       # (K, D) negative keys
    tau: float = 0.07,
) -> torch.Tensor:
    """
    MoCo-style unsupervised contrastive loss on real samples (Eq. 51).
    """
    B = q.size(0)

    # Positive logits: (B, 1)
    l_pos = (q * k).sum(dim=1, keepdim=True) / tau

    # Negative logits: (B, K)
    l_neg = torch.mm(q, queue.T) / tau

    logits = torch.cat([l_pos, l_neg], dim=1)   # (B, 1+K)
    labels = torch.zeros(B, dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Adversarial Domain Loss (Section 3.4.2, Eq. 55)
# ---------------------------------------------------------------------------

def adversarial_domain_loss(
    d_preds_synthetic: torch.Tensor,   # (B_s,) predicted P(synthetic)
    d_preds_real: torch.Tensor,        # (B_r,) predicted P(synthetic)
) -> torch.Tensor:
    """Binary cross-entropy: synthetic=1, real=0."""
    ones  = torch.ones_like(d_preds_synthetic)
    zeros = torch.zeros_like(d_preds_real)
    loss_s = F.binary_cross_entropy(d_preds_synthetic.clamp(1e-6, 1 - 1e-6), ones)
    loss_r = F.binary_cross_entropy(d_preds_real.clamp(1e-6, 1 - 1e-6), zeros)
    return loss_s + loss_r


# ---------------------------------------------------------------------------
# Consistency Regularization (Section 3.4.3, Eq. 57)
# ---------------------------------------------------------------------------

def consistency_loss(
    pred_original: torch.Tensor,       # (B,) raw pain predictions [0, 10]
    pred_augmented: torch.Tensor,      # (B,) raw pain predictions [0, 10]
) -> torch.Tensor:
    """MSE consistency between original and strongly augmented predictions."""
    return F.mse_loss(pred_original, pred_augmented)


# ---------------------------------------------------------------------------
# Domain Alignment Manager
# ---------------------------------------------------------------------------

class DomainAligner(nn.Module):
    """
    Combines all three domain alignment mechanisms.
    Maintains the MoCo queue and momentum encoder update.
    """

    def __init__(self, cfg: dict, feature_encoder: nn.Module, device: str = "cpu"):
        super().__init__()
        align_cfg   = cfg.get("alignment", {})
        cont_cfg    = align_cfg.get("contrastive", {})
        self.tau    = cont_cfg.get("temperature", 0.07)
        self.queue  = NegativeQueue(
            K=cont_cfg.get("queue_size", 16_384),
            D=512,
            device=device,
        )
        self.momentum_enc = MomentumEncoder(
            feature_encoder,
            momentum=cont_cfg.get("momentum", 0.999),
        )
        self.discriminator = DomainDiscriminator(cfg.get("model", {}).get("discriminator", {}))

        self.lambda1 = align_cfg.get("lambda1", 0.5)
        self.lambda2 = align_cfg.get("lambda2", 0.3)
        self.lambda3 = align_cfg.get("lambda3", 0.2)

    def contrastive_loss_synthetic(
        self,
        z_synth: torch.Tensor,
        pain_labels: torch.Tensor,
    ) -> torch.Tensor:
        z_norm = F.normalize(z_synth, dim=1)
        return supervised_contrastive_loss(z_norm, pain_labels, self.tau)

    def contrastive_loss_real(
        self,
        z_real_q: torch.Tensor,      # online encoder features
        z_real_k: torch.Tensor,      # momentum encoder features
    ) -> torch.Tensor:
        q = F.normalize(z_real_q, dim=1)
        k = F.normalize(z_real_k, dim=1)
        neg = self.queue.get_negatives().to(q.device)
        loss = moco_contrastive_loss(q, k, neg, self.tau)
        self.queue.enqueue_dequeue(k)
        return loss

    def adversarial_loss(
        self,
        z_synth: torch.Tensor,
        z_real: torch.Tensor,
        lambda_grl: float,
    ) -> torch.Tensor:
        d_s = self.discriminator(z_synth, lambda_grl)
        d_r = self.discriminator(z_real,  lambda_grl)
        return adversarial_domain_loss(d_s, d_r)

    def get_discriminator_accuracy(
        self,
        z_synth: torch.Tensor,
        z_real: torch.Tensor,
    ) -> float:
        """Diagnostic: should converge to ~50% (domain confusion)."""
        with torch.no_grad():
            d_s = self.discriminator(z_synth, 0.0).round()
            d_r = self.discriminator(z_real,  0.0).round()
            acc = torch.cat([d_s, 1.0 - d_r]).mean().item()
        return float(acc)
