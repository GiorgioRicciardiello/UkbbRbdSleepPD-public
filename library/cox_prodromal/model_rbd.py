"""
Model 0 -- RBD-Only Cox.

h(t) = h0(t) exp(beta_R R + beta_X X)

Quantifies the independent association of actigraphy-derived RBD risk
with incident outcome, adjusted for baseline confounders.

Includes:
- Categorical (binary/tertile) Cox
- Continuous per-SD Cox
- Threshold stability analysis across percentile cutoffs
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from library.cox_prodromal.categorical_ref import pick_reference_category
from library.cox_prodromal.cox_config import (
    MIN_EVENTS_FOR_MODEL,
    RIDGE_PENALIZER,
    THRESHOLD_STABILITY_PERCENTILES,
)
from library.cox_prodromal.diagnostics import extract_model_fit_metrics, run_ph_test


def fit_rbd_only_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model 0 (categorical): Cox PH with RBD risk group as exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_col : str
        Categorical RBD risk group column (e.g. Low/High or Low/Mid/High).
    covariates : list[str]
        Adjustment covariates.
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: summary, c_index, c_index_null, c_index_incremental,
        ph_df, N, events.
    """
    cols = [time_col, event_col, rbd_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    # Null model
    X_null = df_mod[[time_col, event_col] + covariates].reset_index(drop=True)

    # Dummy-encode RBD group (reference: lowest-risk category)
    dum = pd.get_dummies(df_mod[rbd_col], prefix="rbd", drop_first=False)
    rbd_ref = pick_reference_category(dum.columns.tolist())
    dum = dum.drop(columns=[rbd_ref])
    X = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         dum.reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"RBD-only Cox failed: {exc}")
        return None

    c_null = np.nan
    try:
        cph_null = CoxPHFitter(penalizer=penalizer)
        cph_null.fit(
            X_null, duration_col=time_col, event_col=event_col, robust=False
        )
        c_null = cph_null.concordance_index_
    except Exception:
        pass

    ph_df = run_ph_test(cph, X)

    summary = cph.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)
    summary["N"] = len(X)
    summary["events"] = int(X[event_col].sum())

    n_ev = int(X[event_col].sum())
    fit_metrics = extract_model_fit_metrics(cph, n_ev)

    fit_null = {}
    try:
        fit_null = extract_model_fit_metrics(cph_null, n_ev)
    except Exception:
        pass

    return {
        "summary": summary,
        "c_index": cph.concordance_index_,
        "c_index_null": c_null,
        "c_index_incremental": cph.concordance_index_ - c_null,
        "ph_df": ph_df,
        "N": len(X),
        "events": n_ev,
        **fit_metrics,
        "AIC_null": fit_null.get("AIC", np.nan),
        "BIC_null": fit_null.get("BIC", np.nan),
        "log_likelihood_null": fit_null.get("log_likelihood", np.nan),
    }


def fit_rbd_continuous_per_sd(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_prob_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model 0 (continuous): HR per 1-SD increase in RBD probability.

    Standardizes ``rbd_prob_col`` to mean=0, sd=1 before fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    rbd_prob_col : str
        Continuous RBD probability column.

    Returns
    -------
    dict or None
        Keys: hr_per_sd, hr_lci, hr_uci, p, c_index, c_index_null,
        c_index_incremental, N, events, rbd_mean, rbd_sd.
    """
    cols = [time_col, event_col, rbd_prob_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    rbd_mean = float(df_mod[rbd_prob_col].mean())
    rbd_sd = float(df_mod[rbd_prob_col].std())
    if rbd_sd < 1e-10:
        return None

    df_mod["rbd_z"] = (df_mod[rbd_prob_col] - rbd_mean) / rbd_sd

    X = df_mod[
        [time_col, event_col, "rbd_z"] + covariates
    ].reset_index(drop=True)

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"RBD continuous Cox failed: {exc}")
        return None

    # Null model
    c_null = np.nan
    try:
        X_null = df_mod[[time_col, event_col] + covariates].reset_index(drop=True)
        cph_null = CoxPHFitter(penalizer=penalizer)
        cph_null.fit(
            X_null, duration_col=time_col, event_col=event_col, robust=False
        )
        c_null = cph_null.concordance_index_
    except Exception:
        pass

    n_ev = int(X[event_col].sum())
    fit_metrics = extract_model_fit_metrics(cph, n_ev)

    fit_null = {}
    try:
        fit_null = extract_model_fit_metrics(cph_null, n_ev)
    except Exception:
        pass

    row = cph.summary.loc["rbd_z"]
    return {
        "hr_per_sd": float(row["exp(coef)"]),
        "hr_lci": float(row["exp(coef) lower 95%"]),
        "hr_uci": float(row["exp(coef) upper 95%"]),
        "p": float(row["p"]),
        "coef": float(row["coef"]),
        "se_coef": float(row["se(coef)"]),
        "se_hr": float(row["exp(coef)"]) * float(row["se(coef)"]),
        "z": float(row["z"]),
        "c_index": cph.concordance_index_,
        "c_index_null": c_null,
        "c_index_incremental": cph.concordance_index_ - c_null,
        "N": len(X),
        "events": n_ev,
        "rbd_mean": rbd_mean,
        "rbd_sd": rbd_sd,
        **fit_metrics,
        "AIC_null": fit_null.get("AIC", np.nan),
        "BIC_null": fit_null.get("BIC", np.nan),
        "log_likelihood_null": fit_null.get("log_likelihood", np.nan),
    }


def fit_rbd_threshold_stability(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_prob_col: str,
    covariates: List[str],
    percentiles: Optional[List[float]] = None,
    penalizer: float = RIDGE_PENALIZER,
) -> pd.DataFrame:
    """
    Threshold stability: repeat binary RBD Cox at multiple percentile cutoffs.

    For each percentile p, subjects with ``rbd_prob >= p-th percentile``
    are labelled High, others Low. Reports HR for High vs Low.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    rbd_prob_col : str
        Continuous RBD probability column.
    percentiles : list[float], optional
        Percentile cutoffs (default: [5, 10, 15]).
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    pd.DataFrame
        Columns: percentile, threshold_value, hr, lci, uci, p,
        n_high, n_low, events.  Empty if no valid results.
    """
    percentiles = percentiles or THRESHOLD_STABILITY_PERCENTILES

    cols = [time_col, event_col, rbd_prob_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return pd.DataFrame()

    rows = []
    for pct in percentiles:
        threshold = float(np.percentile(df_mod[rbd_prob_col], 100 - pct))
        df_mod["rbd_high"] = (df_mod[rbd_prob_col] >= threshold).astype(int)
        n_high = int(df_mod["rbd_high"].sum())
        n_low = len(df_mod) - n_high

        if n_high < 5 or n_low < 5:
            continue

        X = df_mod[
            [time_col, event_col, "rbd_high"] + covariates
        ].reset_index(drop=True)

        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
        except Exception:
            continue

        row = cph.summary.loc["rbd_high"]
        rows.append({
            "percentile": pct,
            "threshold_value": round(threshold, 6),
            "hr": round(float(row["exp(coef)"]), 4),
            "lci": round(float(row["exp(coef) lower 95%"]), 4),
            "uci": round(float(row["exp(coef) upper 95%"]), 4),
            "p": float(row["p"]),
            "n_high": n_high,
            "n_low": n_low,
            "events": int(X[event_col].sum()),
        })

    return pd.DataFrame(rows)
