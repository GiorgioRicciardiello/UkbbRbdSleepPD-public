"""
distribution.py
===============

Feature and outcome distribution reporting for the ml_cross_sectional pipeline.

Functions
---------
feature_distribution_by_class
    Per-feature mean / SD / missing% stratified by case vs. control, plus
    standardised mean difference (SMD).

fold_composition_table
    Per outer fold: n_train_cases, n_train_controls, n_test_cases, n_test_controls.

oof_distribution_summary
    Out-of-fold (OOF) aggregated statistics: OOF prevalence, mean predicted
    probability by class, calibration slope.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from ..training import NestedCVResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature distribution stratified by class
# ---------------------------------------------------------------------------

def feature_distribution_by_class(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Compute feature statistics stratified by case (y=1) and control (y=0).

    For each numeric feature reports: mean, SD, median, IQR, missing_pct for
    each class, plus the standardised mean difference (SMD):
        SMD = (mean_cases − mean_controls) / pooled_SD

    Assumptions
    -----------
    * ``X`` and ``y`` are aligned (same index order).
    * ``y`` is binary int {0, 1}.
    * Non-numeric columns receive NaN for all statistics.

    Parameters
    ----------
    X :
        Raw (pre-imputation) feature matrix.
    y :
        Binary outcome series.

    Returns
    -------
    pd.DataFrame
        Multi-level columns: (class, stat) where class in {"cases", "controls"}
        and stat in {"mean", "sd", "median", "iqr", "missing_pct"}, plus a
        top-level "smd" column.
        Index = feature names.
    """
    y = y.astype(int)
    mask_cases = y == 1
    mask_ctrl = y == 0

    X_cases = X.loc[mask_cases]
    X_ctrl = X.loc[mask_ctrl]

    rows = []
    for col in X.columns:
        row: dict = {"feature": col}
        for label, Xg in [("cases", X_cases), ("controls", X_ctrl)]:
            s = Xg[col]
            if pd.api.types.is_numeric_dtype(s):
                q1, q3 = s.quantile([0.25, 0.75])
                row[f"{label}_mean"] = float(s.mean())
                row[f"{label}_sd"] = float(s.std(ddof=1))
                row[f"{label}_median"] = float(s.median())
                row[f"{label}_iqr"] = float(q3 - q1)
                row[f"{label}_missing_pct"] = float(s.isna().mean() * 100.0)
            else:
                for stat in ("mean", "sd", "median", "iqr", "missing_pct"):
                    row[f"{label}_{stat}"] = float("nan")

        # Standardised mean difference.
        m_c = row.get("cases_mean", float("nan"))
        m_k = row.get("controls_mean", float("nan"))
        sd_c = row.get("cases_sd", float("nan"))
        sd_k = row.get("controls_sd", float("nan"))
        if all(np.isfinite(v) for v in [m_c, m_k, sd_c, sd_k]):
            pooled = np.sqrt((sd_c**2 + sd_k**2) / 2.0)
            row["smd"] = float((m_c - m_k) / pooled) if pooled > 0 else float("nan")
        else:
            row["smd"] = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("feature")
    return df


# ---------------------------------------------------------------------------
# Fold composition table
# ---------------------------------------------------------------------------

def fold_composition_table(results: "NestedCVResult") -> pd.DataFrame:
    """
    Summarise per-fold cohort composition from a NestedCVResult.

    Parameters
    ----------
    results :
        NestedCVResult returned by either NestedCVTrainer or P1CombinedTrainer.

    Returns
    -------
    pd.DataFrame
        Columns: fold, n_train_cases, n_train_controls, n_test_cases,
        n_test_controls, train_prevalence, test_prevalence.
    """
    rows = []
    for fr in results.folds:
        n_tr = fr.n_train_cases + fr.n_train_controls
        n_te = fr.n_test_cases + fr.n_test_controls
        rows.append({
            "fold": fr.fold,
            "n_train_cases": fr.n_train_cases,
            "n_train_controls": fr.n_train_controls,
            "n_train_total": n_tr,
            "train_prevalence": fr.n_train_cases / n_tr if n_tr > 0 else float("nan"),
            "n_test_cases": fr.n_test_cases,
            "n_test_controls": fr.n_test_controls,
            "n_test_total": n_te,
            "test_prevalence": fr.n_test_cases / n_te if n_te > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Out-of-fold (OOF) distribution summary
# ---------------------------------------------------------------------------

def oof_distribution_summary(results: "NestedCVResult") -> pd.DataFrame:
    """
    Aggregate all out-of-fold predictions into a calibration summary.

    Parameters
    ----------
    results :
        NestedCVResult with ``val_proba`` and ``val_true`` per fold.

    Returns
    -------
    pd.DataFrame
        One row, columns:
        oof_n, oof_n_pos, oof_prevalence,
        oof_mean_proba_cases, oof_mean_proba_controls,
        oof_calibration_slope, oof_calibration_intercept.

    Notes
    -----
    Calibration slope is estimated by a logistic regression of y_true on
    log-odds(y_pred_clipped). Slope ≈ 1.0 = well calibrated;
    < 1 = overconfident; > 1 = underconfident.
    """
    all_true = np.concatenate([fr.val_true for fr in results.folds])
    all_proba = np.concatenate([fr.val_proba for fr in results.folds])

    n = len(all_true)
    n_pos = int(all_true.sum())
    prev = n_pos / n if n > 0 else float("nan")

    mask_pos = all_true == 1
    mask_neg = all_true == 0
    mean_pos = float(all_proba[mask_pos].mean()) if mask_pos.any() else float("nan")
    mean_neg = float(all_proba[mask_neg].mean()) if mask_neg.any() else float("nan")

    # Calibration slope via logistic regression on log-odds.
    eps = 1e-6
    clipped = np.clip(all_proba, eps, 1 - eps)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
        lr.fit(log_odds, all_true)
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
    except Exception as exc:
        logger.warning("Calibration slope failed: %s", exc)
        slope, intercept = float("nan"), float("nan")

    return pd.DataFrame([{
        "oof_n": n,
        "oof_n_pos": n_pos,
        "oof_prevalence": prev,
        "oof_mean_proba_cases": mean_pos,
        "oof_mean_proba_controls": mean_neg,
        "oof_calibration_slope": slope,
        "oof_calibration_intercept": intercept,
    }])
