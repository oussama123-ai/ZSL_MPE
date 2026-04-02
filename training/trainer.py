"""
Three-Stage Trainer (Section 3.5, Algorithm 1)
================================================
Stage 1: Synthetic pre-training      (epochs 1–100)
Stage 2: Domain alignment            (epochs 101–150)
Stage 3: Consistency refinement      (epochs 151–170)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.pain_estimator   import PainEstimator
from models.domain_alignment import DomainAligner, grl_lambda
from training.losses         import TotalTrainingLoss
from training.scheduler      import WarmupCosineScheduler
from data.augmentations      import VideoAugmentor, PhysioAugmentor
from utils.checkpoint        import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates three-stage training for zero-shot pain estimation.

    Usage
    -----
    >>> trainer = Trainer(cfg, model, synth_loader, real_loader)
    >>> trainer.train()
    """

    def __init__(
        self,
        cfg: dict,
        model: PainEstimator,
        synth_loader: DataLoader,
        real_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        output_dir: str = "checkpoints",
    ):
        self.cfg        = cfg
        self.model      = model.to(device)
        self.device     = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.synth_loader = synth_loader
        self.real_loader  = real_loader
        self.val_loader   = val_loader

        train_cfg = cfg.get("training", {})
        s1, s2, s3 = (train_cfg.get("stage1", {}),
                      train_cfg.get("stage2", {}),
                      train_cfg.get("stage3", {}))
        self.s1_epochs = s1.get("epochs", 100)
        self.s2_epochs = s2.get("epochs", 50)
        self.s3_epochs = s3.get("epochs", 20)
        self.total_epochs = self.s1_epochs + self.s2_epochs + self.s3_epochs

        # Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=s1.get("lr_max", 1e-4),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )
        self.grad_clip = opt_cfg.get("grad_clip_norm", 1.0)

        # LR scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=s1.get("warmup_epochs", 5),
            total_epochs=self.total_epochs,
            lr_max=s1.get("lr_max", 1e-4),
            lr_min=s1.get("lr_min", 1e-6),
        )

        # Loss
        self.criterion = TotalTrainingLoss(cfg)

        # Domain aligner (includes discriminator + momentum encoder)
        self.aligner = DomainAligner(cfg, model, device).to(device)

        # Separate optimizer for discriminator
        disc_params = list(self.aligner.discriminator.parameters())
        self.disc_optimizer = torch.optim.AdamW(disc_params, lr=1e-4, weight_decay=1e-4)

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", True) and device != "cpu"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # Strong augmentor for consistency stage
        self.strong_video_aug  = VideoAugmentor(cfg, strong=True)
        self.strong_physio_aug = PhysioAugmentor(cfg)

        # State
        self.epoch       = 0
        self.best_val_mae = float("inf")

    # -----------------------------------------------------------------------
    # Public entry
    # -----------------------------------------------------------------------

    def train(self) -> None:
        logger.info("Starting three-stage training. Total epochs: %d", self.total_epochs)

        # Stage 1
        logger.info("=== Stage 1: Synthetic Pre-training (epochs 1–%d) ===", self.s1_epochs)
        for e in range(1, self.s1_epochs + 1):
            self.epoch = e
            metrics = self._train_epoch_stage1()
            self._log_epoch(e, "S1", metrics)
            self.scheduler.step()
            if self.val_loader:
                val_mae = self._validate()
                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    save_checkpoint(self.model, self.optimizer, e, val_mae,
                                    self.output_dir / "best_model.pt")

        # Stage 2
        logger.info("=== Stage 2: Domain Alignment (epochs %d–%d) ===",
                    self.s1_epochs + 1, self.s1_epochs + self.s2_epochs)
        real_iter = iter(self.real_loader)
        for e in range(1, self.s2_epochs + 1):
            self.epoch = self.s1_epochs + e
            lambda_grl = grl_lambda(e, self.s2_epochs)
            metrics, real_iter = self._train_epoch_stage2(lambda_grl, real_iter)
            self._log_epoch(self.epoch, "S2", metrics)
            self.scheduler.step()
            if self.val_loader:
                val_mae = self._validate()
                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    save_checkpoint(self.model, self.optimizer, self.epoch, val_mae,
                                    self.output_dir / "best_model.pt")

        # Stage 3
        logger.info("=== Stage 3: Consistency Refinement (epochs %d–%d) ===",
                    self.s1_epochs + self.s2_epochs + 1, self.total_epochs)
        real_iter = iter(self.real_loader)
        for e in range(1, self.s3_epochs + 1):
            self.epoch = self.s1_epochs + self.s2_epochs + e
            metrics, real_iter = self._train_epoch_stage3(real_iter)
            self._log_epoch(self.epoch, "S3", metrics)
            self.scheduler.step()

        logger.info("Training complete. Best val MAE: %.4f", self.best_val_mae)
        save_checkpoint(self.model, self.optimizer, self.total_epochs,
                        self.best_val_mae, self.output_dir / "final_model.pt")

    # -----------------------------------------------------------------------
    # Stage 1: Supervised on synthetic (Eq. 60)
    # -----------------------------------------------------------------------

    def _train_epoch_stage1(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.synth_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                preds = self.model(batch)
                loss_dict = self.criterion(
                    preds,
                    pain_gt=batch["pain"],
                    au_gt=batch.get("au"),
                    stage=1,
                )
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        return {"loss": total_loss / max(n_batches, 1)}

    # -----------------------------------------------------------------------
    # Stage 2: Domain alignment (Eq. 61)
    # -----------------------------------------------------------------------

    def _train_epoch_stage2(
        self, lambda_grl: float, real_iter
    ) -> tuple[Dict[str, float], object]:
        self.model.train()
        metrics = {"loss": 0.0, "l_contrast": 0.0, "l_domain": 0.0,
                   "disc_acc": 0.0}
        n = 0

        for synth_batch in self.synth_loader:
            # Attempt to get real batch
            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter  = iter(self.real_loader)
                real_batch = next(real_iter)

            synth_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in synth_batch.items()}
            real_batch  = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in real_batch.items()}

            self.optimizer.zero_grad()
            self.disc_optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Encode both domains
                z_synth, _ = self.model.encode(synth_batch)
                z_real,  _ = self.model.encode(real_batch)

                # Supervised pain loss on synthetic
                preds_synth = self.model(synth_batch)
                pain_losses = self.criterion.pain_loss(
                    preds_synth, synth_batch["pain"], synth_batch.get("au"))

                # Contrastive loss (supervised on synth + MoCo on real)
                z_real_k = self.aligner.momentum_enc.encoder.encode(real_batch)[0] \
                           if hasattr(self.aligner.momentum_enc.encoder, "encode") \
                           else z_real.detach()
                l_cont_s = self.aligner.contrastive_loss_synthetic(
                    z_synth, synth_batch["pain"] * 10)
                l_cont_r = self.aligner.contrastive_loss_real(
                    F.normalize(z_real, dim=1),
                    F.normalize(z_real_k, dim=1),
                )
                l_contrast = l_cont_s + l_cont_r

                # Adversarial alignment
                l_domain = self.aligner.adversarial_loss(z_synth, z_real, lambda_grl)

                # Total (no consistency in stage 2 by default)
                loss_dict = self.criterion(
                    preds_synth, synth_batch["pain"], synth_batch.get("au"),
                    l_contrast=l_contrast, l_domain=l_domain, stage=2,
                )
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.step(self.disc_optimizer)
            self.scaler.update()

            # Momentum encoder update
            self.aligner.momentum_enc.update(self.model)

            disc_acc = self.aligner.get_discriminator_accuracy(
                z_synth.detach(), z_real.detach())

            metrics["loss"]       += loss.item()
            metrics["l_contrast"] += l_contrast.item()
            metrics["l_domain"]   += l_domain.item()
            metrics["disc_acc"]   += disc_acc
            n += 1

        return {k: v / max(n, 1) for k, v in metrics.items()}, real_iter

    # -----------------------------------------------------------------------
    # Stage 3: Consistency refinement (Eq. 62)
    # -----------------------------------------------------------------------

    def _train_epoch_stage3(self, real_iter) -> tuple[Dict[str, float], object]:
        self.model.train()
        metrics = {"loss": 0.0, "l_consist": 0.0}
        n = 0

        for synth_batch in self.synth_loader:
            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter  = iter(self.real_loader)
                real_batch = next(real_iter)

            synth_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in synth_batch.items()}
            real_batch  = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in real_batch.items()}

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                # Pain loss on synthetic
                preds_synth = self.model(synth_batch)
                pain_losses = self.criterion.pain_loss(
                    preds_synth, synth_batch["pain"], synth_batch.get("au"))

                # Consistency on real (original vs augmented predictions)
                pred_orig = self.model(real_batch)["pain"]
                # We rely on the DataLoader's strong augmentor for the 2nd view
                pred_aug  = pred_orig.detach()   # placeholder: same batch
                l_consist = F.mse_loss(pred_orig, pred_aug)

                loss_dict = self.criterion(
                    preds_synth, synth_batch["pain"], synth_batch.get("au"),
                    l_consist=l_consist, stage=3,
                )
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics["loss"]     += loss.item()
            metrics["l_consist"] += l_consist.item()
            n += 1

        return {k: v / max(n, 1) for k, v in metrics.items()}, real_iter

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_mae, n = 0.0, 0

        for batch in self.val_loader:
            batch  = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            preds  = self.model(batch)
            pain_pred = preds["pain"]
            pain_gt   = batch["pain"] * 10   # scale back
            mae = (pain_pred - pain_gt).abs().mean().item()
            total_mae += mae
            n += 1

        return total_mae / max(n, 1)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    @staticmethod
    def _log_epoch(epoch: int, stage: str, metrics: Dict[str, float]) -> None:
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info("[Epoch %3d | %s] %s", epoch, stage, parts)
