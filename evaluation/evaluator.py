"""
Evaluator — Full Evaluation Pipeline
======================================
Runs inference on a benchmark dataset and computes all reported metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.pain_estimator import PainEstimator
from evaluation.metrics import evaluate, print_report

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Zero-shot evaluation on UNBC-McMaster, BioVid, or Neonatal benchmarks.
    No pain labels are accessed during training — only at eval time.
    """

    def __init__(
        self,
        model: PainEstimator,
        device: str = "cuda",
        pain_threshold: float = 4.0,
        bootstrap_iters: int = 10_000,
    ):
        self.model     = model
        self.device    = device
        self.threshold = pain_threshold
        self.n_boot    = bootstrap_iters

    @torch.no_grad()
    def run(self, loader: DataLoader, dataset_name: str = "Unknown"
            ) -> Dict[str, object]:
        """
        Run inference on all batches and compute evaluation metrics.

        Returns a dict of metric → value.
        """
        self.model.eval()
        all_pred, all_gt = [], []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pain_pred = self.model.predict_pain(batch).cpu().numpy()
            pain_gt   = (batch["pain"].cpu().numpy() * 10)   # de-normalise

            all_pred.append(pain_pred)
            all_gt.append(pain_gt)

        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_gt)

        report = evaluate(y_true, y_pred, self.threshold, self.n_boot)
        print_report(report, title=f"Evaluation — {dataset_name}")
        return report

    def cross_dataset_evaluation(
        self,
        loaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict[str, object]]:
        """Evaluate on multiple datasets and summarise cross-dataset results."""
        results = {}
        for name, loader in loaders.items():
            results[name] = self.run(loader, name)
        return results
