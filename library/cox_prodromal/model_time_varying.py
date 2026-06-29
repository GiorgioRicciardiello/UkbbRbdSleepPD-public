"""
Time-covariate interaction sensitivity for PH violators.

For covariates where the Schoenfeld residuals test rejects PH (p < 0.05),
refit the model with a covariate x log(time) interaction term.  If the
interaction is non-significant, the PH violation is inconsequential.
If significant, report time-varying HRs at clinically relevant time points.

This is the standard parametric approach recommended by Therneau & Grambsch
(2000, §6.3) when full time-varying coefficients are not practical.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from library.cox_prodromal.cox_config import RIDGE_PENALIZER, MIN_EVENTS_FOR_MODEL


# Time points at which to report the time-varying HR
_REPORT_TIMES: List[float] = [2.0, 5.0, 10.0]


def fit_time_interaction_sensitivity(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    ph_violators: List[str],
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
    report_times: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Refit Cox with covariate x log(time) interaction for PH violators.

    For each violating covariate, fits a model that includes all original
    covariates plus ``covariate * log(time + epsilon)`` as an additional
    predictor.  The time-varying HR at time t is:

        HR(t) = exp(beta + beta_interaction * log(t))

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    ph_violators : list[str]
        Covariate names that failed the PH test.
    covariates : list[str]
        All model covariates (ph_violators should be a subset).
    penalizer : float
        Ridge penalizer.
    report_times : list[float], optional
        Time points for time-varying HR (default: 2, 5, 10 years).

    Returns
    -------
    pd.DataFrame
        One row per (violator, report_time) with columns:
        covariate, hr_original, coef_main, coef_interaction,
        interaction_p, hr_at_t, t_years.
    """
    report_times = report_times or _REPORT_TIMES
    eps = 0.01  # avoid log(0)

    cols = [time_col, event_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return pd.DataFrame()

    # Fit original model (no time interaction) for reference HRs
    X_orig = df_mod.reset_index(drop=True)
    cph_orig = CoxPHFitter(penalizer=penalizer)
    try:
        cph_orig.fit(X_orig, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"Original Cox fit failed: {exc}")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for violator in ph_violators:
        if violator not in df_mod.columns:
            continue

        # Create interaction term: violator * log(time)
        int_col = f"{violator}__x__logtime"
        df_int = df_mod.copy()
        df_int[int_col] = df_int[violator] * np.log(df_int[time_col] + eps)

        X_int = df_int[
            [time_col, event_col] + covariates + [int_col]
        ].reset_index(drop=True)

        cph_int = CoxPHFitter(penalizer=penalizer)
        try:
            cph_int.fit(X_int, duration_col=time_col, event_col=event_col, robust=False)
        except Exception:
            continue

        # Original HR (from model without interaction)
        hr_orig = np.nan
        if violator in cph_orig.summary.index:
            hr_orig = float(cph_orig.summary.loc[violator, "exp(coef)"])

        # Interaction model coefficients
        coef_main = float(cph_int.summary.loc[violator, "coef"])
        coef_int = float(cph_int.summary.loc[int_col, "coef"])
        int_p = float(cph_int.summary.loc[int_col, "p"])
        int_se = float(cph_int.summary.loc[int_col, "se(coef)"])

        for t in report_times:
            # HR(t) = exp(beta_main + beta_interaction * log(t))
            log_t = np.log(t + eps)
            hr_at_t = np.exp(coef_main + coef_int * log_t)

            rows.append({
                "covariate": violator,
                "hr_original": round(hr_orig, 4),
                "coef_main": round(coef_main, 6),
                "coef_interaction": round(coef_int, 6),
                "se_interaction": round(int_se, 6),
                "interaction_p": round(int_p, 4),
                "t_years": t,
                "hr_at_t": round(hr_at_t, 4),
                "interpretation": (
                    "PH violation inconsequential"
                    if int_p >= 0.05
                    else f"HR varies with time (p={int_p:.3f})"
                ),
            })

    return pd.DataFrame(rows)
