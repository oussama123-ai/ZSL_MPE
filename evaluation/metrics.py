"""
Evaluation Metrics (Section 4.5)
==================================
MAE, RMSE, PCC, ICC-2,1, AUC-ROC, sensitivity, specificity,
clinical accuracy, bootstrap CI, Wilcoxon test, fairness metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Optional sklearn / scipy imports
try:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ---------------------------------------------------------------------------
# Core regression metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_cc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def icc_2_1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Intraclass Correlation Coefficient (ICC-2,1) — two-way mixed."""
    n    = len(y_true)
    data = np.stack([y_true, y_pred], axis=1)
    grand_mean = data.mean()
    ss_r  = 2 * np.sum((data.mean(axis=1) - grand_mean) ** 2)
    ss_c  = n * np.sum((data.mean(axis=0) - grand_mean) ** 2)
    ss_e  = np.sum((data - data.mean(axis=0)) ** 2) - ss_r
    ms_r  = ss_r / (n - 1)
    ms_e  = ss_e / (n - 1)
    ms_c  = ss_c
    icc   = (ms_r - ms_e) / (ms_r + ms_e + 2 * ms_c / n)
    return float(np.clip(icc, -1, 1))


def percent_within(y_true: np.ndarray, y_pred: np.ndarray,
                   tolerance: float = 1.0) -> float:
    """% predictions within ±tolerance pain units (clinical acceptability)."""
    return float(np.mean(np.abs(y_true - y_pred) <= tolerance) * 100)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 4.0,
) -> Dict[str, float]:
    """Sensitivity, specificity, accuracy for pain ≥ threshold."""
    gt   = (y_true >= threshold).astype(int)
    pred = (y_pred >= threshold).astype(int)

    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    acc  = (tp + tn) / max(len(gt), 1)

    auc = 0.5
    if _SKLEARN:
        try:
            auc = roc_auc_score(gt, y_pred)
        except Exception:
            pass

    return {"sensitivity": sens, "specificity": spec,
            "accuracy": acc, "auc": auc}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_iter: int = 10_000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Returns (point_estimate, lower_ci, upper_ci).
    """
    point = metric_fn(y_true, y_pred)
    rng   = np.random.default_rng(0)
    n     = len(y_true)
    samples = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        samples.append(metric_fn(y_true[idx], y_pred[idx]))
    samples = np.array(samples)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    result = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size with pooled standard deviation."""
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / max(pooled_std, 1e-8))


# ---------------------------------------------------------------------------
# Fairness metrics (Section 5.3)
# ---------------------------------------------------------------------------

def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    threshold: float = 4.0,
) -> float:
    """
    Maximum pairwise |Δ sensitivity| or |Δ specificity| across groups.
    """
    gt   = y_true >= threshold
    pred = y_pred >= threshold
    group_ids = np.unique(groups)
    sens_list, spec_list = [], []

    for g in group_ids:
        mask = groups == g
        tp   = np.sum(pred[mask] & gt[mask])
        tn   = np.sum(~pred[mask] & ~gt[mask])
        fp   = np.sum(pred[mask] & ~gt[mask])
        fn   = np.sum(~pred[mask] & gt[mask])
        sens_list.append(tp / max(tp + fn, 1))
        spec_list.append(tn / max(tn + fp, 1))

    sens_diffs = [abs(s1 - s2) for i, s1 in enumerate(sens_list)
                  for s2 in sens_list[i + 1:]]
    spec_diffs = [abs(s1 - s2) for i, s1 in enumerate(spec_list)
                  for s2 in spec_list[i + 1:]]
    max_eod = max(max(sens_diffs, default=0.0), max(spec_diffs, default=0.0))
    return float(max_eod)


def demographic_parity_difference(
    y_pred: np.ndarray,
    groups: np.ndarray,
    threshold: float = 4.0,
) -> float:
    """Max |P(ŷ ≥ threshold | G=g1) − P(ŷ ≥ threshold | G=g2)| over pairs."""
    pred    = y_pred >= threshold
    g_ids   = np.unique(groups)
    rates   = [np.mean(pred[groups == g]) for g in g_ids]
    diffs   = [abs(r1 - r2) for i, r1 in enumerate(rates) for r2 in rates[i + 1:]]
    max_dpd = max(diffs) if diffs else 0.0
    return float(max_dpd)


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 4.0,
    bootstrap_iters: int = 10_000,
) -> Dict[str, object]:
    """
    Compute the full set of metrics reported in the paper.
    """
    report: Dict[str, object] = {}

    # Regression
    mae_val, mae_lo, mae_hi = bootstrap_ci(y_true, y_pred, mae, bootstrap_iters)
    rms, rms_lo, rms_hi     = bootstrap_ci(y_true, y_pred, rmse, bootstrap_iters)
    pcc_val = pearson_cc(y_true, y_pred)
    icc_val = icc_2_1(y_true, y_pred)
    pct_acc = percent_within(y_true, y_pred)

    report.update({
        "MAE":         mae_val,
        "MAE_95CI":    [mae_lo, mae_hi],
        "RMSE":        rms,
        "RMSE_95CI":   [rms_lo, rms_hi],
        "PCC":         pcc_val,
        "ICC":         icc_val,
        "Pct_within1": pct_acc,
    })

    # Binary classification
    bin_met = binary_metrics(y_true, y_pred, threshold)
    report.update({
        "Sensitivity": bin_met["sensitivity"],
        "Specificity": bin_met["specificity"],
        "Accuracy":    bin_met["accuracy"],
        "AUC":         bin_met["auc"],
    })

    return report


def print_report(report: Dict[str, object], title: str = "Evaluation") -> None:
    logger.info("=" * 50)
    logger.info(title)
    logger.info("=" * 50)
    for k, v in report.items():
        if isinstance(v, float):
            logger.info("  %-20s %.4f", k, v)
        else:
            logger.info("  %-20s %s", k, v)
