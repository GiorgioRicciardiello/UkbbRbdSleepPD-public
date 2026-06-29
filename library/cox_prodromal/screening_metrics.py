"""
Screening performance metrics: PPV, NPV, sensitivity, specificity.

Computes diagnostic accuracy at percentile thresholds of the RBD
probability score, using cumulative incidence within a fixed time
horizon as the binary "truth" label.

Wilson score 95% CIs are used for all proportions (recommended for
small numerators; Brown et al., Am Statistician, 2001).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from library.cox_prodromal.cox_config import (
    SCREENING_PERCENTILES,
    SCREENING_TIME_HORIZONS,
)


def _wilson_ci(
    count: int,
    nobs: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion.

    Parameters
    ----------
    count : int
        Number of successes.
    nobs : int
        Number of observations.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the CI.
    """
    if nobs == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def compute_screening_metrics(
    df: pd.DataFrame,
    rbd_prob_col: str,
    time_col: str,
    event_col: str,
    percentiles: Optional[List[float]] = None,
    time_horizons: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compute PPV, NPV, sensitivity, specificity at percentile thresholds.

    For each (percentile, time_horizon) pair:
    1. Define "screen positive" as rbd_prob >= percentile threshold.
    2. Define "true positive" as event within time_horizon years.
    3. Compute 2x2 table and derived metrics with Wilson CIs.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with columns for RBD probability, time, event.
    rbd_prob_col : str
        Continuous RBD probability column.
    time_col : str
        Follow-up time in years.
    event_col : str
        Binary event indicator (0/1).
    percentiles : list[float], optional
        Percentile thresholds (default from config: 90, 95, 99).
    time_horizons : list[float], optional
        Time horizons in years (default from config: 5, 10).

    Returns
    -------
    pd.DataFrame
        One row per (percentile, time_horizon) combination.
    """
    percentiles = percentiles or SCREENING_PERCENTILES
    time_horizons = time_horizons or SCREENING_TIME_HORIZONS

    cols = [rbd_prob_col, time_col, event_col]
    df_cc = df[cols].dropna().copy()
    if df_cc.empty:
        return pd.DataFrame()

    rows: List[Dict] = []

    for pct in percentiles:
        threshold = float(np.percentile(df_cc[rbd_prob_col], pct))

        for t_horizon in time_horizons:
            # Binary truth: event within time horizon
            # Subjects censored before t_horizon with no event are excluded
            # (we cannot determine their status).
            has_event_in_window = (df_cc[event_col] == 1) & (df_cc[time_col] <= t_horizon)
            censored_after = df_cc[time_col] > t_horizon
            no_event = censored_after | ((df_cc[event_col] == 0) & (df_cc[time_col] <= t_horizon))
            # Keep only evaluable subjects: those with event in window
            # OR those followed past the horizon (true negatives)
            evaluable = has_event_in_window | censored_after
            df_eval = df_cc[evaluable].copy()

            if df_eval.empty:
                continue

            truth = ((df_eval[event_col] == 1) & (df_eval[time_col] <= t_horizon)).astype(int)
            screen_pos = (df_eval[rbd_prob_col] >= threshold).astype(int)

            tp = int(((screen_pos == 1) & (truth == 1)).sum())
            fp = int(((screen_pos == 1) & (truth == 0)).sum())
            fn = int(((screen_pos == 0) & (truth == 1)).sum())
            tn = int(((screen_pos == 0) & (truth == 0)).sum())

            n_pos = tp + fn  # total true positives
            n_neg = tn + fp  # total true negatives
            n_screen_pos = tp + fp
            n_screen_neg = tn + fn

            sens = tp / n_pos if n_pos > 0 else np.nan
            spec = tn / n_neg if n_neg > 0 else np.nan
            ppv = tp / n_screen_pos if n_screen_pos > 0 else np.nan
            npv = tn / n_screen_neg if n_screen_neg > 0 else np.nan

            sens_lo, sens_hi = _wilson_ci(tp, n_pos)
            spec_lo, spec_hi = _wilson_ci(tn, n_neg)
            ppv_lo, ppv_hi = _wilson_ci(tp, n_screen_pos)
            npv_lo, npv_hi = _wilson_ci(tn, n_screen_neg)

            rows.append({
                "percentile": pct,
                "threshold_value": round(threshold, 6),
                "time_horizon_years": t_horizon,
                "N_evaluable": len(df_eval),
                "incidence_rate": round(n_pos / len(df_eval), 6) if len(df_eval) > 0 else np.nan,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "sensitivity": round(sens, 4),
                "sensitivity_lci": round(sens_lo, 4),
                "sensitivity_uci": round(sens_hi, 4),
                "specificity": round(spec, 4),
                "specificity_lci": round(spec_lo, 4),
                "specificity_uci": round(spec_hi, 4),
                "PPV": round(ppv, 4),
                "PPV_lci": round(ppv_lo, 4),
                "PPV_uci": round(ppv_hi, 4),
                "NPV": round(npv, 4),
                "NPV_lci": round(npv_lo, 4),
                "NPV_uci": round(npv_hi, 4),
            })

    return pd.DataFrame(rows)
