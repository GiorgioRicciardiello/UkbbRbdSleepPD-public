"""
Model F — RBD Risk Strata × PRS_PD Interaction Cox.

h(t) = h0(t) exp(β_RBD β_PRS·prs_pd_z + β_int_RBD*PRS + β_PC PCs + β_X X)

Tests multiplicative interaction between RBD risk stratification
(Low/Mid/High percentile groups) and continuous PD polygenic risk score.

Computes RERI and Synergy Index for both comparisons:
  - Mid RBD × PRS_PD vs Low RBD × PRS_PD
  - High RBD × PRS_PD vs Low RBD × PRS_PD

PRS_PD is z-scored for interpretable per-SD hazard ratios.
Ancestry PCs (PC1–10) included for population stratification adjustment.
Demographic covariates (age, sex, BMI) included as baseline adjustment.

No prodromal markers included (demographics + genetic adjustment only).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from library.cox_prodromal.categorical_ref import pick_reference_category
from library.cox_prodromal.cox_config import (
    MIN_EVENTS_FOR_MODEL,
    RIDGE_PENALIZER,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    BOOTSTRAP_JOBS,
    BOOTSTRAP_RIDGE_PENALIZER,
    BOOTSTRAP_COEF_CLIP,
)
from library.cox_prodromal.diagnostics import extract_model_fit_metrics, run_ph_test


# ============================================================================
# POINT ESTIMATE: Fit full interaction Cox model
# ============================================================================

def fit_rbd_prs_interaction_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_cat_col: str,
    prs_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model F: Cox PH with RBD × PRS_PD interaction term.

    RBD is categorical (Low/Mid/High); PRS_PD is z-scored continuous.
    Constructs interaction columns: RBD_Mid × PRS_z, RBD_High × PRS_z.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_cat_col : str
        Categorical RBD risk group (Low/Mid/High).
    prs_col : str
        Continuous PRS_PD column.
    covariates : list[str]
        Adjustment covariates (demographics + PC1-10).
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: summary, c_index, c_index_null, c_index_incremental, ph_df,
        N, events, rbd_ref, prs_mean, prs_sd, (fit metrics).
    """
    # prs_col may appear in covariates (e.g. when model_f_covs includes PRS_COLS);
    # remove it from covariates to avoid duplicate columns in df_mod.
    covariates = [c for c in covariates if c != prs_col]
    cols = [time_col, event_col, rbd_cat_col, prs_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    # Z-score PRS for interpretability (per 1-SD)
    prs_mean = float(df_mod[prs_col].mean())
    prs_sd = float(df_mod[prs_col].std())
    if prs_sd < 1e-10:
        return None

    df_mod["prs_z"] = (df_mod[prs_col] - prs_mean) / prs_sd

    # Dummy-encode RBD (reference: Low)
    rbd_dum = pd.get_dummies(df_mod[rbd_cat_col], prefix="rbd", drop_first=False)
    rbd_ref = pick_reference_category(rbd_dum.columns.tolist())
    rbd_dum = rbd_dum.drop(columns=[rbd_ref])

    # Interaction columns: RBD_group × PRS_z
    interaction_cols: List[str] = []
    for rc in rbd_dum.columns:
        iname = f"{rc}__x__prs_z"
        df_mod[iname] = rbd_dum[rc].values * df_mod["prs_z"].values
        interaction_cols.append(iname)

    # Design matrix
    X = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         rbd_dum.reset_index(drop=True),
         df_mod[["prs_z"]].reset_index(drop=True),
         df_mod[interaction_cols].reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    # Null model (covariates only)
    X_null = df_mod[
        [time_col, event_col] + covariates
    ].reset_index(drop=True)

    # Fit interaction model
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"RBD × PRS interaction Cox failed: {exc}")
        return None

    # Null model for C-index
    c_null = np.nan
    try:
        cph_null = CoxPHFitter(penalizer=penalizer)
        cph_null.fit(
            X_null, duration_col=time_col, event_col=event_col, robust=False
        )
        c_null = cph_null.concordance_index_
    except Exception:
        pass

    # Extract PH test and fit metrics
    ph_df = run_ph_test(cph, X)
    n_ev = int(X[event_col].sum())
    fit_metrics = extract_model_fit_metrics(cph, n_ev)

    fit_null = {}
    try:
        fit_null = extract_model_fit_metrics(cph_null, n_ev)
    except Exception:
        pass

    summary = cph.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)
    summary["rbd_ref"] = rbd_ref
    summary["N"] = len(X)
    summary["events"] = n_ev

    return {
        "summary": summary,
        "c_index": cph.concordance_index_,
        "c_index_null": c_null,
        "c_index_incremental": cph.concordance_index_ - c_null,
        "ph_df": ph_df,
        "N": len(X),
        "events": n_ev,
        "rbd_ref": rbd_ref,
        "prs_mean": prs_mean,
        "prs_sd": prs_sd,
        **fit_metrics,
        "AIC_null": fit_null.get("AIC", np.nan),
        "BIC_null": fit_null.get("BIC", np.nan),
        "log_likelihood_null": fit_null.get("log_likelihood", np.nan),
    }


# ============================================================================
# RERI / SYNERGY INDEX with Bootstrap
# ============================================================================

def compute_reri_si_stratified(
    hr_list: List[Tuple[str, float]],  # [(rbd_group, hr), ...]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute RERI and Synergy Index for two stratified comparisons.

    Assumes:
      - hr_list[0] = (Low_RBD, HR_PRS in Low RBD)
      - hr_list[1] = (Mid_RBD, HR_PRS in Mid RBD)
      - hr_list[2] = (High_RBD, HR_PRS in High RBD)

    Comparisons:
      1. Mid RBD × PRS vs Low RBD × PRS
      2. High RBD × PRS vs Low RBD × PRS

    RERI = HR_11 - HR_10 - HR_01 + 1
      where HR_11 = exposure (RBD=Mid/High) AND exposure (PRS=High)
            HR_10 = exposure (RBD=Mid/High) AND no exposure (PRS=Low)
            HR_01 = no exposure (RBD=Low) AND exposure (PRS=High)

    SI = (HR_11 - 1) / ((HR_10 - 1) + (HR_01 - 1))

    For continuous PRS stratified at median:
      - Low PRS effect in Low RBD = HR_01 = 1.0 (reference)
      - Low PRS effect in Mid/High RBD = HR_10 (main effect of RBD at median PRS)
      - High PRS effect in Low RBD = HR_01 (main effect of PRS in Low RBD)
      - High PRS effect in Mid/High RBD = HR_11 (joint effect)

    Returns
    -------
    dict, dict
        (reri_mid_vs_low, reri_high_vs_low), (si_mid_vs_low, si_high_vs_low)
    """
    hrs_dict = {g: hr for g, hr in hr_list}

    # Assume: Low = ref (HR_PRS_in_Low = HR from main effect term)
    # Mid/High: HR_PRS = HR_PRS_in_Low + β_int × [RBD=Mid/High]
    # For simplicity, extract from stratified models or Cox interaction coefficients

    reri_results = {}
    si_results = {}

    # This is a placeholder; actual computation depends on the full model structure
    # For now, return empty dicts; will populate in the runner after stratified fits

    return reri_results, si_results


def _fit_one_bootstrap_stratified(
    df: pd.DataFrame,
    boot_seed: int,
    time_col: str,
    event_col: str,
    rbd_cat_col: str,
    prs_col: str,
    covariates: List[str],
    penalizer: float,
    stratification_groups: List[str],  # e.g., ["Low", "Mid", "High"]
) -> Optional[Dict[str, float]]:
    """
    Single bootstrap iteration: fit Cox in each RBD stratum, extract PRS effect.

    Parameters
    ----------
    stratification_groups : list[str]
        RBD group labels to iterate over.

    Returns
    -------
    dict or None
        Keys: {f"prs_hr_{group}": float for group in stratification_groups}
    """
    covariates = [c for c in covariates if c != prs_col]
    rng = np.random.default_rng(boot_seed)
    idx = rng.choice(len(df), size=len(df), replace=True)
    df_boot = df.iloc[idx].reset_index(drop=True)

    result_dict = {}
    for group in stratification_groups:
        df_strat = df_boot[df_boot[rbd_cat_col] == group].copy()
        if df_strat[event_col].sum() < 2:
            result_dict[f"prs_hr_{group}"] = np.nan
            continue

        cols = [time_col, event_col, prs_col] + covariates
        df_fit = df_strat[cols].dropna()
        if len(df_fit) < 5:
            result_dict[f"prs_hr_{group}"] = np.nan
            continue

        # Z-score PRS within bootstrap sample
        prs_mean = df_fit[prs_col].mean()
        prs_sd = df_fit[prs_col].std()
        if prs_sd < 1e-10:
            result_dict[f"prs_hr_{group}"] = np.nan
            continue

        df_fit["prs_z"] = (df_fit[prs_col] - prs_mean) / prs_sd

        X = df_fit[[time_col, event_col, "prs_z"] + covariates].reset_index(drop=True)

        try:
            cph = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)

            prs_hr = float(np.exp(np.clip(cph.params_["prs_z"], -BOOTSTRAP_COEF_CLIP, BOOTSTRAP_COEF_CLIP)))
            result_dict[f"prs_hr_{group}"] = prs_hr
        except Exception:
            result_dict[f"prs_hr_{group}"] = np.nan

    return result_dict if any(not np.isnan(v) for v in result_dict.values()) else None


def bootstrap_rbd_prs_interaction(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_cat_col: str,
    prs_col: str,
    covariates: List[str],
    stratification_groups: List[str] = None,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    penalizer: float = RIDGE_PENALIZER,
    n_jobs: int = BOOTSTRAP_JOBS,
) -> Optional[Dict[str, Any]]:
    """
    Bootstrap stratified PRS effects across RBD groups.

    For each RBD stratum, compute HR per 1-SD PRS increase.
    Returns CIs for the stratified HRs and derived RERI/SI metrics.

    Parameters
    ----------
    stratification_groups : list[str]
        RBD group labels (e.g., ["Low", "Mid", "High"]).
        If None, inferred from data.

    Returns
    -------
    dict or None
        Keys: prs_hr_low, prs_hr_mid, prs_hr_high, prs_hr_low_lci, ...,
              reri_mid_vs_low, reri_high_vs_low, si_mid_vs_low, si_high_vs_low, etc.
    """
    if stratification_groups is None:
        stratification_groups = sorted(df[rbd_cat_col].dropna().unique())

    covariates = [c for c in covariates if c != prs_col]

    # Point estimates (fit Cox in each stratum)
    prs_hrs_point = {}
    for group in stratification_groups:
        df_strat = df[df[rbd_cat_col] == group].copy()
        cols = [time_col, event_col, prs_col] + covariates
        df_fit = df_strat[cols].dropna()

        if df_fit[event_col].sum() < 2 or len(df_fit) < 5:
            prs_hrs_point[group] = np.nan
            continue

        prs_mean = df_fit[prs_col].mean()
        prs_sd = df_fit[prs_col].std()
        if prs_sd < 1e-10:
            prs_hrs_point[group] = np.nan
            continue

        df_fit["prs_z"] = (df_fit[prs_col] - prs_mean) / prs_sd

        X = df_fit[[time_col, event_col, "prs_z"] + covariates].reset_index(drop=True)

        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
            prs_hrs_point[group] = float(np.exp(np.clip(cph.params_["prs_z"], -BOOTSTRAP_COEF_CLIP, BOOTSTRAP_COEF_CLIP)))
        except Exception:
            prs_hrs_point[group] = np.nan

    # Bootstrap CIs
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one_bootstrap_stratified)(
            df, seed + i, time_col, event_col, rbd_cat_col, prs_col,
            covariates, penalizer, stratification_groups,
        )
        for i in range(n_bootstrap)
    )

    boot_hrs = {g: [] for g in stratification_groups}
    for res in results:
        if res is None:
            continue
        for g in stratification_groups:
            hr_val = res.get(f"prs_hr_{g}", np.nan)
            if not np.isnan(hr_val):
                boot_hrs[g].append(hr_val)

    # Percentile CIs
    ci_results = {}
    for g in stratification_groups:
        pt = prs_hrs_point.get(g, np.nan)
        boot_vals = boot_hrs[g]
        if len(boot_vals) > 10:
            lci = float(np.percentile(boot_vals, 2.5))
            uci = float(np.percentile(boot_vals, 97.5))
        else:
            lci = uci = np.nan

        ci_results[f"prs_hr_{g}"] = pt
        ci_results[f"prs_hr_{g}_lci"] = lci
        ci_results[f"prs_hr_{g}_uci"] = uci

    # TODO: Compute RERI / SI from stratified HRs
    # Placeholder for now; full logic in runner

    return ci_results
