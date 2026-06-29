"""
metrics.py
==========

Cohort statistics and per-fold classification metrics.

Loss / metric rationale
-----------------------
At ~0.7% prevalence, ROC-AUC over-states discrimination because the negative
class dominates. We therefore:

* Optimise hyperparameters on **average precision** (PR-AUC) — the area
  under the precision-recall curve. PR-AUC is sensitive to ranking of the
  rare positive class.
* Select the operating threshold via **Youden's J statistic**
  (``J = sensitivity + specificity - 1``), evaluated on the same fold.
* Report a full battery of metrics for downstream comparison.

The training loss is binary cross-entropy with positive-class reweighting
(``class_weight='balanced'`` or ``scale_pos_weight = n_neg/n_pos``). This is
the maximum-likelihood loss for Bernoulli outcomes and is consistent with
the ``predict_proba`` interface used by every model.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# --- Result dataclasses ------------------------------------------------------

@dataclass(frozen=True)
class MetricsResult:
    """Per-fold classification metrics."""

    auc_roc: float
    auc_pr: float
    f1: float
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    brier: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    n: int
    n_pos: int

    def to_dict(self) -> dict[str, Any]:
        """Return as plain dict (JSON-serialisable)."""
        return asdict(self)


@dataclass(frozen=True)
class CohortStats:
    """Cohort-level descriptive statistics."""

    n_subjects: int
    n_cases: int
    n_controls: int
    prevalence: float
    feature_stats: pd.DataFrame  # mean / sd / median / iqr / missing per feature
    n_incident: int | None = None   # incident cases only (None if not available)
    n_prevalent: int | None = None  # prevalent cases only (None if not available)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n_subjects": self.n_subjects,
            "n_cases": self.n_cases,
            "n_controls": self.n_controls,
            "prevalence": self.prevalence,
            "feature_stats": self.feature_stats.to_dict(orient="index"),
        }
        if self.n_incident is not None:
            d["n_incident"] = self.n_incident
        if self.n_prevalent is not None:
            d["n_prevalent"] = self.n_prevalent
        return d


# --- Threshold selection -----------------------------------------------------

def youden_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Return the threshold that maximises Youden's J = sens + spec - 1.

    Falls back to 0.5 if the ROC curve is degenerate.
    """
    fpr, tpr, thresh = roc_curve(y_true, y_pred_proba)
    if len(thresh) == 0:
        return 0.5
    j = tpr - fpr
    idx = int(np.argmax(j))
    t = float(thresh[idx])
    # roc_curve can return inf as the first threshold; clamp it.
    if not np.isfinite(t):
        t = 0.5
    return t


# --- Metric computation ------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray,
    threshold: float | None = None,
) -> MetricsResult:
    """
    Compute the full metric battery for a single fold.

    Parameters
    ----------
    y_true :
        Ground-truth binary labels.
    y_pred_proba :
        Positive-class probabilities.
    threshold :
        Classification threshold. If ``None``, Youden's J is used.

    Returns
    -------
    MetricsResult
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    if threshold is None:
        threshold = youden_threshold(y_true, y_pred_proba)

    y_pred = (y_pred_proba >= threshold).astype(int)

    # Guard against single-class folds (degenerate AUCs).
    if len(np.unique(y_true)) < 2:
        auc_roc = float("nan")
        auc_pr = float("nan")
    else:
        auc_roc = float(roc_auc_score(y_true, y_pred_proba))
        auc_pr = float(average_precision_score(y_true, y_pred_proba))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return MetricsResult(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        sensitivity=float(sens),
        specificity=float(spec),
        ppv=float(ppv),
        npv=float(npv),
        brier=float(brier_score_loss(y_true, y_pred_proba)),
        threshold=float(threshold),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
    )


# --- Cohort stats ------------------------------------------------------------

def compute_cohort_stats(
    X: pd.DataFrame,
    y: pd.Series,
    y_incident: pd.Series | None = None,
    y_prevalent: pd.Series | None = None,
) -> CohortStats:
    """
    First-order descriptive statistics for the feature matrix and outcome.

    The ``feature_stats`` table reports per-column ``mean``, ``sd``, ``median``,
    ``iqr`` and ``missing_pct``. Computed on the *raw* (pre-imputation)
    matrix because we want missingness to be visible.

    Parameters
    ----------
    X :
        Feature matrix (pre-imputation).
    y :
        Binary outcome vector (0/1).
    y_incident :
        Binary indicator for incident cases only. If provided, written to
        ``cohort_stats.json`` as ``n_incident``.
    y_prevalent :
        Binary indicator for prevalent cases only. If provided, written to
        ``cohort_stats.json`` as ``n_prevalent``.
    """
    y = y.astype(int)
    n_total = int(len(y))
    n_pos = int(y.sum())
    n_neg = n_total - n_pos

    n_incident = int(y_incident.astype(int).sum()) if y_incident is not None else None
    n_prevalent = int(y_prevalent.astype(int).sum()) if y_prevalent is not None else None

    rows = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile([0.25, 0.75])
            rows.append({
                "feature": col,
                "mean": float(s.mean()),
                "sd": float(s.std()),
                "median": float(s.median()),
                "iqr": float(q3 - q1),
                "missing_pct": float(s.isna().mean() * 100.0),
            })
        else:
            rows.append({
                "feature": col,
                "mean": float("nan"),
                "sd": float("nan"),
                "median": float("nan"),
                "iqr": float("nan"),
                "missing_pct": float(s.isna().mean() * 100.0),
            })
    feat_df = pd.DataFrame(rows).set_index("feature")

    return CohortStats(
        n_subjects=n_total,
        n_cases=n_pos,
        n_controls=n_neg,
        prevalence=(n_pos / n_total) if n_total > 0 else 0.0,
        feature_stats=feat_df,
        n_incident=n_incident,
        n_prevalent=n_prevalent,
    )
