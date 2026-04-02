"""LR Scheduler — linear warmup + cosine annealing (Section 3.5.4, Eq. 63)."""

from __future__ import annotations
import math
import torch


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 lr_max: float = 1e-4, lr_min: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.lr_max        = lr_max
        self.lr_min        = lr_min
        self._epoch        = 0

    def step(self) -> None:
        self._epoch += 1
        lr = self._get_lr(self._epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _get_lr(self, e: int) -> float:
        if e < self.warmup_epochs:
            return self.lr_max * e / max(self.warmup_epochs, 1)
        t = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.lr_min + (self.lr_max - self.lr_min) * cos
