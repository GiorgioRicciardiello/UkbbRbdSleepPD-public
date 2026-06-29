"""
Model G — RBD Score × GBA Carrier Interaction.

h(t) = h0(t) exp(beta_R R + beta_GBA GBA + beta_int (R x GBA) + beta_X X)

Tests whether GBA carrier status modifies the actigraphy-RBD → PD hazard.

Sub-models:
  - continuous : z-scored RBD + GBA binary + RBD×GBA product term (mean-centred).
  - categorical: RBD Low/Mid/High dummies + GBA binary + interaction dummies;
                 RERI computed for additive interaction at High-RBD × GBA+.

Design notes
------------
- GBA is binary (0/1); no z-scoring or ancestry PC adjustment needed.
- Carrier prevalence ~1-2% in EUR → cells may be sparse; N and events are
  reported per cell and a warning is issued if any cell has <10 events.
- Restricted to PRIMARY_OUTCOME only (GBA variants are PD-specific).
- No bootstrap for GBA: with <2% carriers, stratum-level bootstrapping is
  unreliable. RERI and LRT provide the interaction inference.
- Mean-centred product term (rbd_z × gba_carrier) reduces multicollinearity
  between main effects and the interaction term (same convention as Model A).
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

# Minimum events per cell before issuing a power warning
_MIN_CELL_EVENTS: int = 10


# ── Continuous (z-scored RBD) × GBA ───────────────────────────────────────────

def fit_model_g_rbd_gba_continuous(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_prob_col: str,
    gba_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model G (continuous): HR per 1-SD increase in z-scored RBD, adjusted for GBA.

    Fits two nested Cox models to obtain the LRT for the RBD × GBA interaction:
      - Additive: rbd_z + gba_carrier + covariates
      - Interaction: rbd_z + gba_carrier + rbd_z×gba (mean-centred) + covariates

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset; must contain time_col, event_col, rbd_prob_col,
        gba_col, and all covariates.
    time_col : str
        Duration column (years).
    event_col : str
        Binary event indicator (0/1).
    rbd_prob_col : str
        Continuous RBD probability column (e.g. 'abk_rbd_score_mean').
    gba_col : str
        Binary GBA carrier column (0/1 integers).
    covariates : list[str]
        Base adjustment covariates (age, sex, BMI, smoking, alcohol).
    penalizer : float
        Ridge penalizer for CoxPHFitter.

    Returns
    -------
    dict or None
        Keys: summary, c_index, c_index_null, c_index_incremental, ph_df,
        N, events, rbd_type, rbd_mean, rbd_sd,
        lrt_interaction_stat, lrt_interaction_p, (fit metrics).
        None if insufficient events.
    """
    cols = [time_col, event_col, rbd_prob_col, gba_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    n_carriers = int(df_mod[gba_col].sum())
    if n_carriers == 0:
        warnings.warn("Model G continuous: no GBA carriers in complete-case sample.")
        return None

    rbd_mean = float(df_mod[rbd_prob_col].mean())
    rbd_sd = float(df_mod[rbd_prob_col].std())
    if rbd_sd < 1e-10:
        return None

    df_mod = df_mod.copy()
    df_mod["rbd_z"] = (df_mod[rbd_prob_col] - rbd_mean) / rbd_sd

    # Mean-centred interaction term (reduces multicollinearity with main effects)
    product = df_mod["rbd_z"] * df_mod[gba_col]
    df_mod["rbd_z_x_gba"] = product - float(product.mean())

    # ── Interaction model ──────────────────────────────────────────────────────
    feat_int = ["rbd_z", gba_col, "rbd_z_x_gba"] + covariates
    X_int = df_mod[[time_col, event_col] + feat_int].reset_index(drop=True)

    cph_int = CoxPHFitter(penalizer=penalizer)
    try:
        cph_int.fit(X_int, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"Model G continuous (interaction) Cox failed: {exc}")
        return None

    # ── Additive model (for LRT) ───────────────────────────────────────────────
    feat_add = ["rbd_z", gba_col] + covariates
    X_add = df_mod[[time_col, event_col] + feat_add].reset_index(drop=True)

    cph_add = CoxPHFitter(penalizer=penalizer)
    lrt_stat = np.nan
    lrt_p = np.nan
    try:
        cph_add.fit(X_add, duration_col=time_col, event_col=event_col, robust=False)
        lr = max(-2.0 * (cph_add.log_likelihood_ - cph_int.log_likelihood_), 0.0)
        lrt_stat = lr
        import scipy.stats as scipy_stats
        lrt_p = float(scipy_stats.chi2.sf(lr, df=1))
    except Exception as exc:
        warnings.warn(f"Model G continuous (additive LRT) failed: {exc}")

    # ── Null model (covariates only, for incremental C) ───────────────────────
    c_null = np.nan
    try:
        X_null = df_mod[[time_col, event_col] + covariates].reset_index(drop=True)
        cph_null = CoxPHFitter(penalizer=penalizer)
        cph_null.fit(X_null, duration_col=time_col, event_col=event_col, robust=False)
        c_null = cph_null.concordance_index_
    except Exception:
        pass

    n_ev = int(X_int[event_col].sum())
    fit_metrics = extract_model_fit_metrics(cph_int, n_ev)

    summary = cph_int.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)
    summary["N"] = len(X_int)
    summary["events"] = n_ev

    print(
        f"  [Model G continuous] N={len(X_int):,}, events={n_ev}, "
        f"GBA carriers={n_carriers} "
        f"LRT interaction: stat={lrt_stat:.3f}, p={lrt_p:.4f}"
    )

    return {
        "summary": summary,
        "c_index": cph_int.concordance_index_,
        "c_index_null": c_null,
        "c_index_incremental": cph_int.concordance_index_ - c_null,
        "ph_df": run_ph_test(cph_int, X_int),
        "N": len(X_int),
        "events": n_ev,
        "n_gba_carriers": n_carriers,
        "rbd_type": "continuous_z",
        "rbd_mean": rbd_mean,
        "rbd_sd": rbd_sd,
        "lrt_interaction_stat": round(lrt_stat, 4) if not np.isnan(lrt_stat) else np.nan,
        "lrt_interaction_p": lrt_p,
        **fit_metrics,
    }


# ── Categorical RBD × GBA ─────────────────────────────────────────────────────

def _compute_reri(
    hr_rbd_only: float,
    hr_gba_only: float,
    hr_both: float,
) -> float:
    """
    RERI = HR(RBD+, GBA+) - HR(RBD+, GBA-) - HR(RBD-, GBA+) + 1.

    Reference cell: RBD_Low, GBA=0.

    Parameters
    ----------
    hr_rbd_only : float
        HR for High-RBD, GBA=0 (vs Low-RBD, GBA=0).
    hr_gba_only : float
        HR for Low-RBD, GBA=1 (vs Low-RBD, GBA=0).
    hr_both : float
        HR for High-RBD, GBA=1 (vs Low-RBD, GBA=0).
        Estimated as exp(coef_RBD_High + coef_GBA + coef_RBD_High×GBA).

    Returns
    -------
    float
        RERI value. Positive values indicate super-additive (synergistic) interaction.
    """
    return hr_both - hr_rbd_only - hr_gba_only + 1.0


def fit_model_g_rbd_gba_categorical(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_cat_col: str,
    gba_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Model G (categorical): Cox with RBD Low/Mid/High × GBA carrier interaction.

    Reports:
    - Per-stratum N and events (6 cells: 3 RBD × 2 GBA strata).
    - Main effects (RBD Mid, RBD High, GBA carrier) + interaction HRs.
    - RERI for High-RBD × GBA+ vs Low-RBD × GBA- reference.
    - LRT for interaction (2 df: Mid×GBA and High×GBA terms).

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col : str
        Duration column (years).
    event_col : str
        Binary event indicator.
    rbd_cat_col : str
        Categorical RBD column (Low / Mid / High).
    gba_col : str
        Binary GBA carrier column (0/1).
    covariates : list[str]
        Base adjustment covariates.
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: summary, c_index, c_index_null, c_index_incremental, ph_df,
        N, events, cell_counts, reri, lrt_interaction_stat, lrt_interaction_p,
        rbd_ref, (fit metrics).
        None if insufficient events.
    """
    cols = [time_col, event_col, rbd_cat_col, gba_col] + covariates
    df_mod = df[cols].dropna().copy()
    df_mod[rbd_cat_col] = df_mod[rbd_cat_col].astype(str)
    df_mod[gba_col] = df_mod[gba_col].astype(int)

    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    n_carriers = int(df_mod[gba_col].sum())
    if n_carriers == 0:
        warnings.warn("Model G categorical: no GBA carriers in complete-case sample.")
        return None

    # ── Cell counts ────────────────────────────────────────────────────────────
    cell_counts: List[Dict[str, Any]] = []
    low_power_cells: List[str] = []
    for rbd_grp in sorted(df_mod[rbd_cat_col].unique()):
        for gba_val in [0, 1]:
            mask = (df_mod[rbd_cat_col] == rbd_grp) & (df_mod[gba_col] == gba_val)
            cell_n = int(mask.sum())
            cell_ev = int(df_mod.loc[mask, event_col].sum())
            label = f"RBD={rbd_grp}, GBA={'carrier' if gba_val else 'non-carrier'}"
            cell_counts.append({"rbd_group": rbd_grp, "gba": gba_val,
                                 "N": cell_n, "events": cell_ev})
            if cell_ev < _MIN_CELL_EVENTS:
                low_power_cells.append(f"{label} (n_events={cell_ev})")

    if low_power_cells:
        warnings.warn(
            f"Model G categorical: low event count (<{_MIN_CELL_EVENTS}) in cells: "
            + "; ".join(low_power_cells)
        )

    # ── Dummy-encode RBD groups ────────────────────────────────────────────────
    dum = pd.get_dummies(df_mod[rbd_cat_col], prefix="rbd", drop_first=False)
    rbd_ref = pick_reference_category(dum.columns.tolist())
    dum = dum.drop(columns=[rbd_ref])
    rbd_non_ref_cols = list(dum.columns)

    # ── Interaction dummies: RBD_group × GBA ──────────────────────────────────
    int_cols: List[str] = []
    for rc in rbd_non_ref_cols:
        iname = f"{rc}__x__gba"
        dum[iname] = dum[rc].astype(int) * df_mod[gba_col].values
        int_cols.append(iname)

    # ── Interaction model ──────────────────────────────────────────────────────
    feat_int = rbd_non_ref_cols + [gba_col] + int_cols + covariates
    X_int = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         dum[rbd_non_ref_cols + int_cols].reset_index(drop=True),
         df_mod[[gba_col]].reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    cph_int = CoxPHFitter(penalizer=penalizer)
    try:
        cph_int.fit(X_int, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"Model G categorical (interaction) Cox failed: {exc}")
        return None

    # ── Additive model (for LRT, df=2) ────────────────────────────────────────
    X_add = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         dum[rbd_non_ref_cols].reset_index(drop=True),
         df_mod[[gba_col]].reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    lrt_stat = np.nan
    lrt_p = np.nan
    try:
        cph_add = CoxPHFitter(penalizer=penalizer)
        cph_add.fit(X_add, duration_col=time_col, event_col=event_col, robust=False)
        lr = max(-2.0 * (cph_add.log_likelihood_ - cph_int.log_likelihood_), 0.0)
        lrt_stat = lr
        import scipy.stats as scipy_stats
        lrt_p = float(scipy_stats.chi2.sf(lr, df=len(int_cols)))
    except Exception as exc:
        warnings.warn(f"Model G categorical (additive LRT) failed: {exc}")

    # ── Null model ─────────────────────────────────────────────────────────────
    c_null = np.nan
    try:
        X_null = df_mod[[time_col, event_col] + covariates].reset_index(drop=True)
        cph_null = CoxPHFitter(penalizer=penalizer)
        cph_null.fit(X_null, duration_col=time_col, event_col=event_col, robust=False)
        c_null = cph_null.concordance_index_
    except Exception:
        pass

    # ── RERI: High-RBD + GBA+ vs Low-RBD + GBA- ───────────────────────────────
    reri = np.nan
    try:
        params = cph_int.params_
        # Identify High-RBD dummy column
        high_col = next((c for c in rbd_non_ref_cols if "High" in c or "high" in c), None)
        if high_col is not None:
            coef_high_rbd = float(params.get(high_col, np.nan))
            coef_gba = float(params.get(gba_col, np.nan))
            int_col_high = f"{high_col}__x__gba"
            coef_int_high = float(params.get(int_col_high, np.nan))
            if not any(np.isnan(v) for v in [coef_high_rbd, coef_gba, coef_int_high]):
                hr_rbd_only = np.exp(coef_high_rbd)
                hr_gba_only = np.exp(coef_gba)
                hr_both = np.exp(coef_high_rbd + coef_gba + coef_int_high)
                reri = _compute_reri(hr_rbd_only, hr_gba_only, hr_both)
    except Exception as exc:
        warnings.warn(f"Model G categorical: RERI computation failed: {exc}")

    n_ev = int(X_int[event_col].sum())
    fit_metrics = extract_model_fit_metrics(cph_int, n_ev)

    summary = cph_int.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)
    summary["N"] = len(X_int)
    summary["events"] = n_ev

    print(
        f"  [Model G categorical] N={len(X_int):,}, events={n_ev}, "
        f"GBA carriers={n_carriers}, "
        f"RERI={reri:.4f} "
        f"LRT interaction (df={len(int_cols)}): stat={lrt_stat:.3f}, p={lrt_p:.4f}"
    )
    for cc in cell_counts:
        print(
            f"    Cell RBD={cc['rbd_group']}, GBA={'carrier' if cc['gba'] else 'non-carrier'}: "
            f"N={cc['N']:,}, events={cc['events']}"
        )

    return {
        "summary": summary,
        "c_index": cph_int.concordance_index_,
        "c_index_null": c_null,
        "c_index_incremental": cph_int.concordance_index_ - c_null,
        "ph_df": run_ph_test(cph_int, X_int),
        "N": len(X_int),
        "events": n_ev,
        "n_gba_carriers": n_carriers,
        "rbd_ref": rbd_ref,
        "rbd_type": "categorical",
        "cell_counts": cell_counts,
        "reri": reri,
        "lrt_interaction_stat": round(lrt_stat, 4) if not np.isnan(lrt_stat) else np.nan,
        "lrt_interaction_p": lrt_p,
        **fit_metrics,
    }
