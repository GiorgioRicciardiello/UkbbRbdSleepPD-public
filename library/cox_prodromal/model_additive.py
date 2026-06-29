"""
Model 2 -- Additive Combined Cox (no interaction).

h(t) = h0(t) exp(beta_R R + beta_P P + beta_X X)

Tests whether RBD and the prodromal marker each retain an independent
association with the outcome when both are included simultaneously.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from library.cox_prodromal.categorical_ref import pick_reference_category
from library.cox_prodromal.cox_config import MIN_EVENTS_FOR_MODEL, RIDGE_PENALIZER
from library.cox_prodromal.diagnostics import extract_model_fit_metrics, run_ph_test


def fit_additive_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    prod_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model 2: Cox PH with RBD and prodromal as separate main effects.

    No interaction term is included. Comparison of the resulting HRs
    with Model 0 and Model 1 reveals attenuation (confounding or
    shared pathway effects).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``time_col``, ``event_col``, ``rbd_col``,
        ``prod_col`` and all ``covariates``.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_col : str
        Categorical RBD risk group column.
    prod_col : str
        Categorical prodromal group column.
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
    cols = [time_col, event_col, rbd_col, prod_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    # Dummy-encode both exposures
    rbd_dum = pd.get_dummies(df_mod[rbd_col], prefix="rbd", drop_first=False)
    prod_dum = pd.get_dummies(df_mod[prod_col], prefix="prod", drop_first=False)

    rbd_ref = pick_reference_category(rbd_dum.columns.tolist())
    prod_ref = pick_reference_category(prod_dum.columns.tolist())
    rbd_dum = rbd_dum.drop(columns=[rbd_ref])
    prod_dum = prod_dum.drop(columns=[prod_ref])

    X = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         rbd_dum.reset_index(drop=True),
         prod_dum.reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    # Null model (covariates only)
    X_null = df_mod[
        [time_col, event_col] + covariates
    ].reset_index(drop=True)

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"Additive Cox failed: {exc}")
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
    summary["rbd_ref"] = rbd_ref
    summary["prod_ref"] = prod_ref
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
