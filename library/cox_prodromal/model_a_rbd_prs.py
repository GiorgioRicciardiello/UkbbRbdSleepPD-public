"""
Model A — RBD Score with Polygenic Risk Scores and Ancestry PCs.

h(t) = h0(t) exp(beta_R R + beta_PRS PRS + beta_PC1..10 PC1..10 + beta_X X)

Tests whether actigraphy-derived RBD predicts disease independently of
genetic predisposition (PRS) and population stratification (PC1-PC10).

Sub-models per outcome:
  - continuous: z-scored RBD probability, HR per 1-SD increase
  - categorical: RBD risk group (Low/High or Low/Mid/High), HRs by group
  - interaction: rbd_z × prs_score_pd product term tests biological synergy
  - discrimination: nested C-statistic comparison (RBD-only, additive, interaction)

Rationale
---------
PRS columns control for genetic confounding: if individuals with high
actigraphy-RBD scores simply carry more PD/RBD risk alleles, the RBD-disease
association is confounded by shared genetic architecture. Model A isolates
the non-genetic (phenotypic/behavioural) component of the actigraphy signal.

PC1-PC10 are included ONLY here because they address population stratification,
a bias specific to genetic variables. Prodromal models (cognitive/autonomic
markers) do not include PCs because ancestry is not a proposed confounder of
non-genetic exposures in this European-ancestry UKBB cohort.

Interaction model (MA_rbd_x_prs_pd)
-------------------------------------
h(t) = h0(t) exp(β_R·rbd_z + β_PD·prs_pd + β_int·(rbd_z × prs_pd)
                 + β_RBD·prs_rbd + β_PC·PCs + β_X·X)

β_int > 0: subjects who are both high-actigraphy-RBD and high-PD-PRS carry
multiplicative excess risk beyond the additive contributions.  This is the
direct test of biological coherence between the actigraphy score and genetic
PD liability.

Note: the interaction term is constructed as a column product (rbd_z * prs_pd)
before model fitting because lifelines does not support formula syntax.  The
product term is mean-centred to reduce multicollinearity with main effects.
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
    PRS_COLS,
    RIDGE_PENALIZER,
)
from library.cox_prodromal.diagnostics import extract_model_fit_metrics, run_ph_test

# Primary interaction: RBD score × PD PRS
_INTERACTION_PRS_COL: str = "prs_score_pd"
_INTERACTION_TERM_COL: str = "rbd_z_x_prs_pd"


def fit_model_a_rbd_prs_continuous(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_prob_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model A (continuous): HR per 1-SD increase in z-scored RBD score.

    Adjusted for age, sex, BMI, smoking, alcohol + PRS_PD + PRS_RBD + PC1-PC10.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_prob_col : str
        Continuous RBD probability column (e.g. 'abk_rbd_score_mean', range ~-10 to -15).
    covariates : list[str]
        Adjustment covariates, including PRS + PCs.
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: summary, c_index, c_index_null, c_index_incremental, ph_df,
        N, events, (fit metrics), rbd_mean, rbd_sd, rbd_type='continuous_z'.
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

    X = df_mod[[time_col, event_col, "rbd_z"] + covariates].reset_index(
        drop=True
    )

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"Model A continuous Cox failed: {exc}")
        return None

    # Null model
    c_null = np.nan
    try:
        X_null = df_mod[[time_col, event_col] + covariates].reset_index(
            drop=True
        )
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

    # Full summary (all covariates)
    summary = cph.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)
    summary["N"] = len(X)
    summary["events"] = n_ev

    return {
        "summary": summary,
        "c_index": cph.concordance_index_,
        "c_index_null": c_null,
        "c_index_incremental": cph.concordance_index_ - c_null,
        "ph_df": run_ph_test(cph, X),
        "N": len(X),
        "events": n_ev,
        "rbd_type": "continuous_z",
        "rbd_mean": rbd_mean,
        "rbd_sd": rbd_sd,
        **fit_metrics,
        "AIC_null": fit_null.get("AIC", np.nan),
        "BIC_null": fit_null.get("BIC", np.nan),
        "log_likelihood_null": fit_null.get("log_likelihood", np.nan),
    }


def fit_model_a_rbd_prs_categorical(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_cat_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model A (categorical): Cox with RBD risk group as exposure + PRS + PCs.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_cat_col : str
        Categorical RBD risk group column (e.g. Low/High or Low/Mid/High).
    covariates : list[str]
        Adjustment covariates, including PRS + PCs.
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: summary, c_index, c_index_null, c_index_incremental, ph_df,
        N, events, (fit metrics), rbd_ref, rbd_type='categorical'.
    """
    cols = [time_col, event_col, rbd_cat_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    # Null model
    X_null = df_mod[[time_col, event_col] + covariates].reset_index(drop=True)

    # Dummy-encode RBD group (reference: lowest-risk category)
    dum = pd.get_dummies(df_mod[rbd_cat_col], prefix="rbd", drop_first=False)
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
        warnings.warn(f"Model A categorical Cox failed: {exc}")
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

    n_ev = int(X[event_col].sum())
    fit_metrics = extract_model_fit_metrics(cph, n_ev)

    fit_null = {}
    try:
        fit_null = extract_model_fit_metrics(cph_null, n_ev)
    except Exception:
        pass

    summary = cph.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)
    summary["N"] = len(X)
    summary["events"] = n_ev

    return {
        "summary": summary,
        "c_index": cph.concordance_index_,
        "c_index_null": c_null,
        "c_index_incremental": cph.concordance_index_ - c_null,
        "ph_df": run_ph_test(cph, X),
        "N": len(X),
        "events": n_ev,
        "rbd_type": "categorical",
        "rbd_ref": rbd_ref,
        **fit_metrics,
        "AIC_null": fit_null.get("AIC", np.nan),
        "BIC_null": fit_null.get("BIC", np.nan),
        "log_likelihood_null": fit_null.get("log_likelihood", np.nan),
    }


def fit_model_a_rbd_prs_interaction(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_prob_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """Model A — interaction: rbd_z × prs_score_pd product term.

    Tests whether subjects who are both high-actigraphy-RBD and high-PD-PRS
    carry multiplicative excess risk beyond the additive contributions.

    Design matrix
    -------------
    rbd_z  (z-scored RBD probability)
    prs_score_pd  (main effect, already in covariates)
    rbd_z_x_prs_pd  (mean-centred product term)
    all remaining covariates as main effects

    The product term is mean-centred (after z-scoring rbd_z) to reduce
    collinearity with the main effects.  This does not change the
    interpretation of β_int but stabilises the optimiser.

    N is explicitly reported because the PRS-complete subset is smaller
    than the full analytical cohort.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset (PRS-complete rows only).
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_prob_col : str
        Continuous RBD probability column.
    covariates : list[str]
        Adjustment covariates including PRS + PCs (same as Model A additive).
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Same keys as fit_model_a_rbd_prs_continuous, plus:
        rbd_type='interaction', interaction_term=_INTERACTION_TERM_COL,
        interaction_prs=_INTERACTION_PRS_COL.
    """
    if _INTERACTION_PRS_COL not in covariates:
        warnings.warn(
            f"Interaction PRS column '{_INTERACTION_PRS_COL}' not in covariates; "
            "skipping interaction model."
        )
        return None

    cols = [time_col, event_col, rbd_prob_col] + covariates
    df_mod = df[[c for c in cols if c in df.columns]].dropna().copy()

    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    rbd_mean = float(df_mod[rbd_prob_col].mean())
    rbd_sd = float(df_mod[rbd_prob_col].std())
    if rbd_sd < 1e-10:
        return None

    df_mod = df_mod.copy()
    df_mod["rbd_z"] = (df_mod[rbd_prob_col] - rbd_mean) / rbd_sd

    # Mean-centre the product to reduce multicollinearity with main effects.
    product = df_mod["rbd_z"] * df_mod[_INTERACTION_PRS_COL]
    df_mod[_INTERACTION_TERM_COL] = product - product.mean()

    feature_cols = ["rbd_z", _INTERACTION_TERM_COL] + covariates
    X = df_mod[[time_col, event_col] + feature_cols].reset_index(drop=True)

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"Model A interaction Cox failed: {exc}")
        return None

    # Additive null (same covariates, no interaction term) for LRT comparison.
    additive_cols = ["rbd_z"] + covariates
    X_add = df_mod[[time_col, event_col] + additive_cols].reset_index(drop=True)
    c_additive = np.nan
    fit_additive: dict = {}
    try:
        cph_add = CoxPHFitter(penalizer=penalizer)
        cph_add.fit(X_add, duration_col=time_col, event_col=event_col, robust=False)
        c_additive = cph_add.concordance_index_
        fit_additive = extract_model_fit_metrics(cph_add, int(X_add[event_col].sum()))
    except Exception:
        pass

    n_ev = int(X[event_col].sum())
    fit_metrics = extract_model_fit_metrics(cph, n_ev)

    # LRT: interaction vs. additive (1 extra df for the product term)
    lrt_int_stat = np.nan
    lrt_int_p = np.nan
    try:
        import scipy.stats as _scipy_stats
        ll_int = cph.log_likelihood_
        ll_add = fit_additive.get("log_likelihood", np.nan)
        if pd.notna(ll_int) and pd.notna(ll_add):
            lrt_int_stat = float(2 * (ll_int - ll_add))
            lrt_int_p = float(1.0 - _scipy_stats.chi2.cdf(lrt_int_stat, df=1))
    except Exception:
        pass

    summary = cph.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)
    summary["N"] = len(X)
    summary["events"] = n_ev

    return {
        "summary": summary,
        "c_index": cph.concordance_index_,
        "c_index_additive": c_additive,
        "c_index_incremental_vs_additive": cph.concordance_index_ - c_additive,
        "ph_df": run_ph_test(cph, X),
        "N": len(X),
        "events": n_ev,
        "rbd_type": "interaction",
        "interaction_term": _INTERACTION_TERM_COL,
        "interaction_prs": _INTERACTION_PRS_COL,
        "lrt_interaction_stat": lrt_int_stat,
        "lrt_interaction_p": lrt_int_p,
        **fit_metrics,
        "AIC_additive": fit_additive.get("AIC", np.nan),
        "BIC_additive": fit_additive.get("BIC", np.nan),
        "log_likelihood_additive": fit_additive.get("log_likelihood", np.nan),
        "delta_AIC_vs_additive": (
            round(fit_metrics.get("AIC", np.nan) - fit_additive.get("AIC", np.nan), 2)
            if pd.notna(fit_metrics.get("AIC")) and pd.notna(fit_additive.get("AIC"))
            else np.nan
        ),
    }


def fit_model_a_discrimination_comparison(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_prob_col: str,
    base_covariates: List[str],
    model_a_covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """Compare C-statistics across three nested Model A variants.

    Nested hierarchy (each adds one component to the previous):
      M0  — rbd_z + base_covariates          (no PRS)
      M1  — rbd_z + base_covariates + PRS+PCs (additive)
      M2  — M1 + rbd_z × prs_pd              (interaction)

    The delta-C values (M1 vs M0, M2 vs M1, M2 vs M0) quantify the
    discriminative contribution of genetics beyond the actigraphy signal.

    Note: formal bootstrap delta-C tests are intentionally omitted here
    (they are already run in the main discrimination module).  This function
    only provides point estimates for the interaction model comparison table.

    Parameters
    ----------
    df : pd.DataFrame
        PRS-complete survival dataset.
    time_col, event_col, rbd_prob_col : str
        Standard column names.
    base_covariates : list[str]
        Age, sex, BMI, smoking, alcohol (no PRS, no PCs).
    model_a_covariates : list[str]
        Full Model A covariates (base + PRS + PCs).
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: N, events,
        c_M0, c_M1, c_M2,
        delta_c_M1_vs_M0, delta_c_M2_vs_M1, delta_c_M2_vs_M0,
        AIC_M0, AIC_M1, AIC_M2, BIC_M0, BIC_M1, BIC_M2.
    """
    all_cols = (
        [time_col, event_col, rbd_prob_col]
        + model_a_covariates
    )
    df_mod = df[[c for c in all_cols if c in df.columns]].dropna().copy()

    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    rbd_sd = float(df_mod[rbd_prob_col].std())
    if rbd_sd < 1e-10:
        return None

    df_mod = df_mod.copy()
    df_mod["rbd_z"] = (df_mod[rbd_prob_col] - df_mod[rbd_prob_col].mean()) / rbd_sd

    # Product term (mean-centred)
    product = df_mod["rbd_z"] * df_mod[_INTERACTION_PRS_COL]
    df_mod[_INTERACTION_TERM_COL] = product - product.mean()

    n_ev = int(df_mod[event_col].sum())
    results: dict = {"N": len(df_mod), "events": n_ev}

    def _fit(feature_cols: List[str]) -> tuple:
        X = df_mod[[time_col, event_col] + feature_cols].reset_index(drop=True)
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
            fm = extract_model_fit_metrics(cph, n_ev)
            return cph.concordance_index_, fm
        except Exception:
            return np.nan, {}

    avail_base = [c for c in base_covariates if c in df_mod.columns]
    avail_model_a = [c for c in model_a_covariates if c in df_mod.columns]

    c_m0, fm0 = _fit(["rbd_z"] + avail_base)
    c_m1, fm1 = _fit(["rbd_z"] + avail_model_a)
    c_m2, fm2 = _fit(["rbd_z", _INTERACTION_TERM_COL] + avail_model_a)

    results.update({
        "c_M0_rbd_base": round(c_m0, 4) if pd.notna(c_m0) else np.nan,
        "c_M1_rbd_prs_additive": round(c_m1, 4) if pd.notna(c_m1) else np.nan,
        "c_M2_rbd_prs_interaction": round(c_m2, 4) if pd.notna(c_m2) else np.nan,
        "delta_c_M1_vs_M0": (
            round(c_m1 - c_m0, 4)
            if pd.notna(c_m1) and pd.notna(c_m0) else np.nan
        ),
        "delta_c_M2_vs_M1": (
            round(c_m2 - c_m1, 4)
            if pd.notna(c_m2) and pd.notna(c_m1) else np.nan
        ),
        "delta_c_M2_vs_M0": (
            round(c_m2 - c_m0, 4)
            if pd.notna(c_m2) and pd.notna(c_m0) else np.nan
        ),
        "AIC_M0": fm0.get("AIC", np.nan),
        "BIC_M0": fm0.get("BIC", np.nan),
        "AIC_M1": fm1.get("AIC", np.nan),
        "BIC_M1": fm1.get("BIC", np.nan),
        "AIC_M2": fm2.get("AIC", np.nan),
        "BIC_M2": fm2.get("BIC", np.nan),
        "delta_AIC_M1_vs_M0": (
            round(fm1.get("AIC", np.nan) - fm0.get("AIC", np.nan), 2)
            if pd.notna(fm1.get("AIC")) and pd.notna(fm0.get("AIC")) else np.nan
        ),
        "delta_AIC_M2_vs_M1": (
            round(fm2.get("AIC", np.nan) - fm1.get("AIC", np.nan), 2)
            if pd.notna(fm2.get("AIC")) and pd.notna(fm1.get("AIC")) else np.nan
        ),
    })
    return results
