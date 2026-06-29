"""
Model diagnostics: PH assumption testing and FDR correction.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def run_ph_test(
    cph: CoxPHFitter,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Schoenfeld residuals test for proportional hazards assumption.

    Uses rank-transformed time. A p-value < 0.05 flags a violation.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox model.
    training_data : pd.DataFrame
        The data used to fit ``cph`` (must include duration and event cols).

    Returns
    -------
    pd.DataFrame
        Columns: ph_stat, ph_p, ph_violation.  Index = covariate names.
        Empty DataFrame if the test fails.
    """
    try:
        result = proportional_hazard_test(
            cph, training_data, time_transform="rank"
        )
        ph = result.summary[["test_statistic", "p"]].copy()
        ph.columns = ["ph_stat", "ph_p"]
        ph["ph_violation"] = ph["ph_p"] < 0.05
        return ph
    except Exception as exc:
        warnings.warn(f"PH test failed: {exc}")
        return pd.DataFrame()


def apply_fdr(
    p_series: pd.Series,
    method: str = "fdr_bh",
) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_series : pd.Series
        Raw p-values (may contain NaN).
    method : str
        Correction method passed to ``multipletests`` (default BH).

    Returns
    -------
    pd.Series
        Adjusted p-values aligned to the input index.
    """
    if not STATSMODELS_AVAILABLE:
        return pd.Series(np.nan, index=p_series.index)
    valid = p_series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=p_series.index)
    _, p_adj, _, _ = multipletests(valid.values, method=method)
    out = pd.Series(np.nan, index=p_series.index)
    out.loc[valid.index] = p_adj
    return out


def extract_model_fit_metrics(
    cph: CoxPHFitter,
    n_events: int,
) -> dict:
    """
    Extract model fit statistics from a fitted CoxPHFitter.

    Computes AIC (partial likelihood), BIC using the Volinsky & Raftery (2000)
    convention for Cox models (n = number of uncensored events), log partial
    likelihood, and the likelihood ratio test versus the null (intercept-only)
    model.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox model.
    n_events : int
        Number of uncensored events (used as n in BIC formula).

    Returns
    -------
    dict
        Keys: AIC, BIC, log_likelihood, n_params, LRT_stat, LRT_p, LRT_df.
    """
    k = len(cph.params_)
    ll = cph.log_likelihood_
    # AIC_partial_ is the correct attribute for semi-parametric Cox models
    # (lifelines >= 0.27). AIC_ raises an error for non-parametric baselines.
    aic = cph.AIC_partial_

    # BIC for Cox partial likelihood: k * ln(n_events) - 2 * LL
    # Volinsky & Raftery (2000): use number of events, not subjects.
    bic = k * np.log(max(n_events, 1)) - 2.0 * ll

    # Likelihood ratio test vs null (no covariates)
    try:
        lrt = cph.log_likelihood_ratio_test()
        lrt_stat = float(lrt.test_statistic)
        lrt_p = float(lrt.p_value)
        lrt_df = int(lrt.degrees_of_freedom)
    except Exception:
        lrt_stat = np.nan
        lrt_p = np.nan
        lrt_df = 0

    return {
        "AIC": round(float(aic), 2),
        "BIC": round(float(bic), 2),
        "log_likelihood": round(float(ll), 2),
        "n_params": k,
        "LRT_stat": round(lrt_stat, 3),
        "LRT_p": lrt_p,
        "LRT_df": lrt_df,
    }


def summarize_ph_violations(
    ph_results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Summarize PH violations across all models.

    Parameters
    ----------
    ph_results : pd.DataFrame
        Aggregated PH test results with columns:
        outcome, prodromal_var, covariate, ph_stat, ph_p, ph_violation.
    alpha : float
        Significance threshold for violation.

    Returns
    -------
    pd.DataFrame
        Counts of violations by covariate, sorted descending.
    """
    if ph_results.empty or "ph_violation" not in ph_results.columns:
        return pd.DataFrame(columns=["covariate", "n_violations", "n_tests"])

    grp = ph_results.groupby("covariate").agg(
        n_violations=("ph_violation", "sum"),
        n_tests=("ph_violation", "count"),
    ).reset_index()
    grp["violation_rate"] = (grp["n_violations"] / grp["n_tests"]).round(3)
    return grp.sort_values("n_violations", ascending=False).reset_index(drop=True)
