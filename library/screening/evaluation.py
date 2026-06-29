"""
Evaluation metrics for the screening model.

All evaluation is performed on the incident-only test fold to assess
prospective validity — prevalent cases are excluded from evaluation even
if they appear in the test set.

Metrics:
  - ROC-AUC: overall discrimination
  - PR-AUC (average precision): discrimination under class imbalance
  - Brier score: calibration / probabilistic accuracy
  - Calibration slope: logistic regression of y_true on log-odds(y_pred)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Metrics for a single outer CV fold."""
    fold: int
    paradigm: str
    n_cases: int
    n_controls: int
    roc_auc: float
    pr_auc: float
    brier_score: float
    calibration_slope: float
    best_params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "fold": self.fold,
            "paradigm": self.paradigm,
            "n_cases": self.n_cases,
            "n_controls": self.n_controls,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "calibration_slope": self.calibration_slope,
            **{f"param_{k}": v for k, v in self.best_params.items()},
        }


def evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold: int,
    paradigm: str,
    best_params: Optional[Dict] = None,
) -> FoldMetrics:
    """
    Compute evaluation metrics for a single test fold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (1 = incident PD case, 0 = control).
        Must contain only incident cases and controls — prevalent cases
        must be excluded by the caller before invoking this function.
    y_prob : np.ndarray
        Predicted probabilities for the positive class (incident PD).
    fold : int
        Outer fold index (0-based), for logging and results tracing.
    paradigm : str
        Paradigm name for labelling.
    best_params : dict, optional
        Best hyperparameters from inner CV (logged but not used in metrics).

    Returns
    -------
    FoldMetrics
    """
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    if n_pos == 0:
        logger.error("Fold %d: no incident cases in test set — skipping metrics.", fold)
        return FoldMetrics(
            fold=fold, paradigm=paradigm, n_cases=0, n_controls=n_neg,
            roc_auc=np.nan, pr_auc=np.nan, brier_score=np.nan,
            calibration_slope=np.nan, best_params=best_params or {},
        )

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    cal_slope = _calibration_slope(y_true, y_prob)

    logger.debug(
        "Fold %d [%s]: ROC-AUC=%.3f, PR-AUC=%.3f, Brier=%.4f, CalSlope=%.3f",
        fold, paradigm, roc_auc, pr_auc, brier, cal_slope,
    )
    return FoldMetrics(
        fold=fold, paradigm=paradigm, n_cases=n_pos, n_controls=n_neg,
        roc_auc=roc_auc, pr_auc=pr_auc, brier_score=brier,
        calibration_slope=cal_slope,
        best_params=best_params or {},
    )


def _calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Estimate calibration slope via logistic regression of y_true on log-odds(y_pred).

    A well-calibrated model yields slope ≈ 1.0.
    Values < 1 indicate over-confident predictions; > 1 indicate under-confidence.

    Clipping y_prob avoids log(0) / log(1) instability.
    """
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))

    try:
        lr = LogisticRegression(fit_intercept=True, max_iter=200)
        lr.fit(log_odds.reshape(-1, 1), y_true)
        slope = float(lr.coef_[0, 0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Calibration slope computation failed: %s", exc)
        slope = np.nan
    return slope


def aggregate_metrics(fold_metrics: List[FoldMetrics]) -> pd.DataFrame:
    """
    Aggregate fold-level metrics to mean ± SD across outer CV folds.

    Parameters
    ----------
    fold_metrics : list[FoldMetrics]

    Returns
    -------
    pd.DataFrame
        One row per metric (roc_auc, pr_auc, brier_score, calibration_slope)
        with columns: metric, mean, std, min, max.
    """
    records = [m.to_dict() for m in fold_metrics]
    df = pd.DataFrame(records)

    metric_cols = ["roc_auc", "pr_auc", "brier_score", "calibration_slope"]
    rows = []
    for col in metric_cols:
        vals = df[col].dropna()
        rows.append({
            "metric": col,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_folds": int(vals.count()),
        })
    return pd.DataFrame(rows)


def compile_results_table(
    all_metrics: Dict[str, List[FoldMetrics]],
) -> pd.DataFrame:
    """
    Build a wide-format summary table comparing all paradigms.

    Parameters
    ----------
    all_metrics : dict
        ``{paradigm_name: [FoldMetrics, ...]}`` for all paradigms.

    Returns
    -------
    pd.DataFrame
        Rows = paradigms, columns = metric_mean / metric_std pairs.
    """
    rows = []
    for paradigm_name, folds in all_metrics.items():
        agg = aggregate_metrics(folds)
        row = {"paradigm": paradigm_name}
        for _, agg_row in agg.iterrows():
            m = agg_row["metric"]
            row[f"{m}_mean"] = round(agg_row["mean"], 4)
            row[f"{m}_std"] = round(agg_row["std"], 4)
        rows.append(row)
    return pd.DataFrame(rows)
