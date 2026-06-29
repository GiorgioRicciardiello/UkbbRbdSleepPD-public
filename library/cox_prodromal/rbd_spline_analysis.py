"""
RBD spline dose-response analysis for the Cox prodromal pipeline.

Provides a self-contained sub-analysis for PRIMARY_OUTCOME that:
  1. Fits a natural cubic spline (df=4) Cox model on continuous RBD score.
  2. Computes two formal LRT tests:
       P_overall       — spline vs. covariate-only null (df=n_df)
       P_non-linearity — spline vs. linear-RBD model   (df=n_df - 1)
  3. Generates the HR curve (delta-method CIs) relative to the cohort median.
  4. Produces a two-panel publication figure:
       Panel A — spline HR curve + categorical HR overlays + threshold lines
       Panel B — kernel density of RBD scores (cases vs. non-cases)
  5. Saves figure, LRT summary table, HR curve CSV, and model data CSV.

Entry point for the pipeline:
    run_rbd_spline_analysis(
        df_risk, thresholds, extended_covariates,
        rbd_only_rows, path_report, path_results,
    )

Design notes
------------
- `df_risk` is the already-loaded, neuro-excluded, subject-level DataFrame
  from `load_prodromal_dataset()` — no redundant data loading.
- Prevalent-case exclusion: subjects with NaN in `{outcome}__surv_days`
  are excluded, consistent with `build_survival_dataset_for_outcome()`.
- Categorical HRs in the figure are extracted dynamically from `rbd_only_rows`
  (the M0 results from the pipeline), not hardcoded.
- Reference = analytical-cohort median (post-exclusion), consistent with the
  Cox estimation sample.
- Spline basis evaluation on new data uses `patsy.build_design_matrices(design_info, ...)`
  to preserve knot positions from the training data (avoids patsy single-point failure).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from lifelines import CoxPHFitter
from patsy import build_design_matrices
from patsy import dmatrix as patsy_dmatrix
from scipy.stats import gaussian_kde

from library.cox_prodromal.cox_config import (
    PRIMARY_METHOD,
    PRIMARY_OUTCOME,
    RIDGE_PENALIZER,
    RISK_PALETTE,
    SPLINE_DF,
)
from library.cox_prodromal.utils import save_table
from library.column_registry import col_prevalent


# ── Constants ─────────────────────────────────────────────────────────────────

N_GRID: int = 300  # evaluation points for the HR curve


# ── Cohort preparation ────────────────────────────────────────────────────────

def build_spline_survival_dataset(
    df_risk: pd.DataFrame,
    outcome: str,
    covariates: List[str],
) -> Optional[pd.DataFrame]:
    """
    Construct the analytical cohort for the spline analysis.

    Uses the already-loaded, neuro-excluded, subject-level `df_risk` from
    `load_prodromal_dataset()`.  Prevalent-case exclusion is applied by
    dropping subjects whose `{outcome}__surv_days` is NaN — matching the
    logic in `build_survival_dataset_for_outcome()`.

    Parameters
    ----------
    df_risk : pd.DataFrame
        Subject-level, neuro-excluded dataset.  Must contain `rbd_prob`,
        `{outcome}__surv_days`, `{outcome}__incident`, and all `covariates`.
    outcome : str
        Outcome prefix (e.g. 'outcome_1a_pd_only').
    covariates : list[str]
        Adjustment covariate column names.

    Returns
    -------
    pd.DataFrame or None
        Columns: eid, rbd_prob, time (years), event, *covariates.
        None if required survival columns are absent.
    """
    surv_col     = f"{outcome}__surv_days"
    event_col    = f"{outcome}__incident"
    prevalent_col = col_prevalent(outcome)   # {outcome}__prevalent

    missing = [c for c in (surv_col, event_col, prevalent_col) if c not in df_risk.columns]
    if missing:
        warnings.warn(
            f"Spline analysis: missing columns {missing}. Skipping."
        )
        return None

    # Mirror build_survival_dataset_for_outcome: incident cases + controls only,
    # prevalent cases explicitly excluded, non-NaN survival time required.
    incident_mask  = df_risk[event_col].fillna(False).astype(bool)
    control_mask   = (
        df_risk["control"].fillna(False).astype(bool)
        if "control" in df_risk.columns
        else pd.Series(False, index=df_risk.index)
    )
    prevalent_mask = df_risk[prevalent_col].fillna(False).astype(bool)
    surv_mask      = df_risk[surv_col].notna()

    df_subj = df_risk[
        (incident_mask | control_mask) & ~prevalent_mask & surv_mask
    ].copy()

    df_subj["time"]  = df_subj[surv_col] / 365.25
    df_subj["event"] = df_subj[event_col].astype(int)

    keep_cols = ["eid", "rbd_prob", "time", "event"] + covariates
    df_subj = (
        df_subj[[c for c in keep_cols if c in df_subj.columns]]
        .dropna()
    )

    print(
        f"  Spline cohort ({outcome}): "
        f"{len(df_subj):,} subjects, {int(df_subj['event'].sum())} events"
    )
    return df_subj


# ── Categorical HR extraction ─────────────────────────────────────────────────

def extract_categorical_hrs(
    rbd_only_rows: List[Dict[str, Any]],
    outcome: str,
    method: str = PRIMARY_METHOD,
    model: str = "M0_rbd_only",
) -> Dict[str, Tuple[float, float, float]]:
    """
    Extract point-estimate and 95 % CI for each risk stratum from `rbd_only_rows`.

    Looks for rows matching `outcome`, `method`, `model` and identifies
    'Intermediate' (Mid) and 'High' groups from the covariate name.

    Parameters
    ----------
    rbd_only_rows : list[dict]
        Rows accumulated by `_flatten_summary` for Model 0 (RBD-only Cox).
    outcome : str
        Outcome key to filter.
    method : str
        RBD stratification method key (e.g. 'percentile_3g').
    model : str
        Model label (e.g. 'M0_rbd_only').

    Returns
    -------
    dict
        {'Intermediate': (HR, LCI, UCI), 'High': (HR, LCI, UCI)}.
        Empty dict if no matching rows are found.
    """
    result: Dict[str, Tuple[float, float, float]] = {}
    for row in rbd_only_rows:
        if row.get("outcome") != outcome:
            continue
        if row.get("method") != method:
            continue
        if row.get("model") != model:
            continue
        cov = str(row.get("covariate", ""))
        if "Mid" in cov or "Intermediate" in cov:
            result["Intermediate"] = (
                float(row["HR"]), float(row["HR_lower"]), float(row["HR_upper"])
            )
        elif "High" in cov:
            result["High"] = (
                float(row["HR"]), float(row["HR_lower"]), float(row["HR_upper"])
            )
    return result


# ── Spline Cox model ──────────────────────────────────────────────────────────

def fit_spline_model(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    covariates: List[str],
    n_df: int = SPLINE_DF,
    penalizer: float = RIDGE_PENALIZER,
) -> Tuple[CoxPHFitter, List[str], pd.DataFrame, object]:
    """
    Fit a natural cubic spline (NCS) Cox model for the RBD dose-response.

    Patsy `cr()` is used to construct the NCS basis.  The resulting
    `design_info` object is returned alongside the model so that the same
    knot positions can be applied when evaluating the basis on new data
    (grid points, reference point) via `patsy.build_design_matrices()`.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col, event_col : str
        Survival time (years) and binary event indicator.
    rbd_col : str
        Continuous RBD score column.
    covariates : list[str]
        Adjustment covariate column names.
    n_df : int
        Spline degrees of freedom (default 4).
    penalizer : float
        Ridge penalizer for `CoxPHFitter`.

    Returns
    -------
    cph_s : CoxPHFitter
        Fitted model.
    spline_cols : list[str]
        Names of the spline basis columns (used to index coefficients).
    df_mod : pd.DataFrame
        Complete-case data used for fitting.
    design_info : patsy.DesignInfo
        Stores knot positions — pass to `build_design_matrices()` for
        evaluation on new `x` arrays.
    """
    cols = [time_col, event_col, rbd_col] + covariates
    df_mod = df[cols].dropna().copy()
    df_mod[rbd_col] = pd.to_numeric(df_mod[rbd_col], errors="coerce")
    df_mod = df_mod.dropna()

    spline_dm = patsy_dmatrix(
        f"cr(x, df={n_df}) - 1",
        {"x": df_mod[rbd_col].values},
        return_type="dataframe",
    )
    design_info = spline_dm.design_info
    spline_dm.index = df_mod.index
    spline_cols = [f"_rbd_s{i}" for i in range(spline_dm.shape[1])]
    spline_dm.columns = spline_cols

    X_spline = pd.concat(
        [
            df_mod[[time_col, event_col]].reset_index(drop=True),
            spline_dm.reset_index(drop=True),
            df_mod[covariates].reset_index(drop=True),
        ],
        axis=1,
    )

    cph_s = CoxPHFitter(penalizer=penalizer)
    cph_s.fit(X_spline, duration_col=time_col, event_col=event_col, robust=False)

    return cph_s, spline_cols, df_mod, design_info


# ── Likelihood-ratio tests ────────────────────────────────────────────────────

def compute_likelihood_ratio_tests(
    df_mod: pd.DataFrame,
    cph_spline: CoxPHFitter,
    time_col: str,
    event_col: str,
    rbd_col: str,
    spline_cols: List[str],
    covariates: List[str],
    n_df: int = SPLINE_DF,
    penalizer: float = RIDGE_PENALIZER,
) -> Dict[str, float]:
    """
    Two likelihood-ratio tests (LRT) against the fitted spline model.

    P_overall (χ², df=n_df):
        H0: RBD has no association with the hazard.
        Model: spline Cox vs. covariate-only null.

    P_non-linearity (χ², df=n_df-1):
        H0: The dose-response is linear in RBD.
        Model: spline Cox vs. linear-RBD Cox.

    Both tests use the partial log-likelihood from the fitted models.
    With ridge penalization, the LRT is approximate — standard practice
    for penalized spline Cox models (Wood 2017; Royston & Sauerbrei 2007).

    Parameters
    ----------
    df_mod : pd.DataFrame
        Complete-case data used for `cph_spline` (same rows required).
    cph_spline : CoxPHFitter
        Already-fitted spline model.
    time_col, event_col : str
        Survival columns.
    rbd_col : str
        Continuous RBD column.
    spline_cols : list[str]
        Spline basis column names (not used directly but kept for signature
        consistency with `compute_hr_curve()`).
    covariates : list[str]
        Adjustment covariate columns.
    n_df : int
        Spline degrees of freedom.
    penalizer : float
        Ridge penalizer (must match the spline model).

    Returns
    -------
    dict
        lr_overall_stat, lr_overall_p,
        lr_nonlinear_stat, lr_nonlinear_p,
        ll_spline, ll_linear, ll_null,
        c_index_spline, N, events.
    """
    X_null = pd.concat(
        [
            df_mod[[time_col, event_col]].reset_index(drop=True),
            df_mod[covariates].reset_index(drop=True),
        ],
        axis=1,
    )
    cph_null = CoxPHFitter(penalizer=penalizer)
    cph_null.fit(X_null, duration_col=time_col, event_col=event_col, robust=False)
    ll_null = cph_null.log_likelihood_

    X_lin = pd.concat(
        [
            df_mod[[time_col, event_col, rbd_col]].reset_index(drop=True),
            df_mod[covariates].reset_index(drop=True),
        ],
        axis=1,
    )
    cph_lin = CoxPHFitter(penalizer=penalizer)
    cph_lin.fit(X_lin, duration_col=time_col, event_col=event_col, robust=False)
    ll_linear = cph_lin.log_likelihood_

    ll_spline = cph_spline.log_likelihood_

    lr_overall_stat   = max(-2.0 * (ll_null   - ll_spline), 0.0)
    lr_nonlinear_stat = max(-2.0 * (ll_linear - ll_spline), 0.0)

    return {
        "lr_overall_stat":   lr_overall_stat,
        "lr_overall_p":      float(scipy_stats.chi2.sf(lr_overall_stat,   df=n_df)),
        "lr_nonlinear_stat": lr_nonlinear_stat,
        "lr_nonlinear_p":    float(scipy_stats.chi2.sf(lr_nonlinear_stat, df=n_df - 1)),
        "ll_spline":         ll_spline,
        "ll_linear":         ll_linear,
        "ll_null":           ll_null,
        "c_index_spline":    round(cph_spline.concordance_index_, 4),
        "N":                 len(df_mod),
        "events":            int(df_mod[event_col].sum()),
    }


# ── HR curve (delta method) ───────────────────────────────────────────────────

def compute_hr_curve(
    cph_spline: CoxPHFitter,
    spline_cols: List[str],
    rbd_col: str,
    df_mod: pd.DataFrame,
    reference: float,
    design_info: object,
    n_grid: int = N_GRID,
) -> pd.DataFrame:
    """
    Compute the HR curve relative to *reference* using the delta method.

    HR(x) = exp[ f(x) - f(reference) ]

    where f(x) = Σ_j β_j · s_j(x) is the spline linear predictor.

    Variance of log HR(x):
        Var[ f(x) - f(ref) ] = (s(x) - s(ref))^T · V_β · (s(x) - s(ref))

    V_β is the spline coefficient covariance matrix extracted from
    `cph_spline.variance_matrix_`.

    Spline bases for the evaluation grid and the reference point are computed
    via `patsy.build_design_matrices(design_info, ...)`, which reuses the
    knot positions stored in `design_info` (from `fit_spline_model()`).
    This avoids the patsy failure mode where a single evaluation point
    cannot define its own knots.

    Parameters
    ----------
    cph_spline : CoxPHFitter
        Fitted spline model.
    spline_cols : list[str]
        Spline basis column names.
    rbd_col : str
        RBD column name (for range extraction from df_mod).
    df_mod : pd.DataFrame
        Modelling data (only used for min/max range extraction).
    reference : float
        Reference RBD value (HR ≡ 1.0 at this point).
    design_info : patsy.DesignInfo
        Stored from `fit_spline_model()`.  Encodes knot positions.
    n_grid : int
        Number of evaluation points.

    Returns
    -------
    pd.DataFrame
        Columns: rbd_prob, HR, HR_LCI, HR_UCI.
        Empty DataFrame on failure.
    """
    grid = np.linspace(
        float(df_mod[rbd_col].min()), float(df_mod[rbd_col].max()), n_grid
    )

    try:
        grid_dm_raw = build_design_matrices([design_info], {"x": grid})[0]
        grid_dm = pd.DataFrame(np.asarray(grid_dm_raw), columns=spline_cols)

        ref_dm_raw = build_design_matrices([design_info], {"x": np.array([reference])})[0]
        ref_dm = pd.DataFrame(np.asarray(ref_dm_raw), columns=spline_cols)
    except Exception as exc:
        warnings.warn(f"Spline basis evaluation failed: {exc}")
        return pd.DataFrame()

    betas = np.array([
        cph_spline.params_[c] for c in spline_cols
        if c in cph_spline.params_.index
    ])
    if len(betas) != len(spline_cols):
        warnings.warn("Spline coefficient extraction failed — column mismatch.")
        return pd.DataFrame()

    log_hr = grid_dm.values @ betas - (ref_dm.values @ betas)[0]

    try:
        spline_idx = [
            list(cph_spline.params_.index).index(c) for c in spline_cols
        ]
        V = cph_spline.variance_matrix_.iloc[spline_idx, spline_idx].values
        diff = grid_dm.values - ref_dm.values
        se_log_hr = np.sqrt(np.diag(diff @ V @ diff.T))
    except Exception as exc:
        warnings.warn(f"Delta-method variance failed: {exc}")
        se_log_hr = np.full(len(grid), np.nan)

    return pd.DataFrame({
        "rbd_prob": grid,
        "HR":       np.exp(log_hr),
        "HR_LCI":   np.exp(log_hr - 1.96 * se_log_hr),
        "HR_UCI":   np.exp(log_hr + 1.96 * se_log_hr),
    })


# ── Plotting ──────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    """Format a p-value for figure annotation."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def plot_rbd_spline_figure(
    hr_curve: pd.DataFrame,
    df_subj: pd.DataFrame,
    lrt: Dict[str, float],
    thresholds: dict,
    reference: float,
    cat_hrs: Dict[str, Tuple[float, float, float]],
    out_path: Path,
    dpi: int = 300,
) -> None:
    """
    Generate and save the two-panel RBD spline figure.

    Panel A — HR curve with 95 % CI ribbon, categorical HR overlays
              (horizontal bands with CI shading), stratum background,
              threshold lines, reference marker, and LRT annotation box.
    Panel B — Kernel density of RBD scores: incident cases (solid) vs.
              non-cases (dashed), with threshold lines matching Panel A.

    Parameters
    ----------
    hr_curve : pd.DataFrame
        Output of `compute_hr_curve()`. Columns: rbd_prob, HR, HR_LCI, HR_UCI.
    df_subj : pd.DataFrame
        Analytical cohort. Requires 'rbd_prob' and 'event'.
    lrt : dict
        Output of `compute_likelihood_ratio_tests()`.
    thresholds : dict
        Risk threshold dict from risk_collection.json.
    reference : float
        Median RBD score (analytical cohort post-exclusion).
    cat_hrs : dict
        {'Intermediate': (HR, LCI, UCI), 'High': (HR, LCI, UCI)}.
        Categorical overlays are omitted for absent keys.
    out_path : Path
        Output PNG path.
    dpi : int
        Figure resolution.
    """
    thr_3g = thresholds["percentile_3g"]["rbd_only_distribution"]["all"]
    p90: float = thr_3g["p90"]
    p99: float = thr_3g["p99"]

    p_min = float(df_subj["rbd_prob"].min())
    p_max = float(df_subj["rbd_prob"].max())

    col_low  = RISK_PALETTE.rbd_low   # blue
    col_mid  = RISK_PALETTE.rbd_mid   # salmon
    col_high = RISK_PALETTE.rbd_high  # red

    fig, axes = plt.subplots(
        2, 1, figsize=(7.5, 6.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.06)
    ax_hr, ax_den = axes

    # ── Panel A — HR curve ───────────────────────────────────────────────────
    ax_hr.axvspan(p_min, p90,  alpha=0.06, color=col_low,  zorder=0)
    ax_hr.axvspan(p90,  p99,   alpha=0.06, color=col_mid,  zorder=0)
    ax_hr.axvspan(p99,  p_max, alpha=0.06, color=col_high, zorder=0)

    for x_thr, col in [(p90, col_mid), (p99, col_high)]:
        ax_hr.axvline(x_thr, color=col, linestyle="--", linewidth=1.2,
                      alpha=0.8, zorder=1)

    ax_hr.axvline(reference, color="grey", linestyle=":", linewidth=0.9,
                  alpha=0.7, zorder=1)
    ax_hr.axhline(1.0, color=col_low, linestyle="-", linewidth=0.8, alpha=0.6)

    # Categorical HR overlays
    if cat_hrs:
        cat_cfg = [
            ("Intermediate", p90, p99,   col_mid),
            ("High",         p99, p_max, col_high),
        ]
        for grp, x_lo, x_hi, col in cat_cfg:
            if grp not in cat_hrs:
                continue
            hr, lci, uci = cat_hrs[grp]
            ax_hr.hlines(hr, x_lo, x_hi, colors=col, linewidths=1.4,
                         linestyles="--", alpha=0.9, zorder=2)
            ax_hr.fill_between([x_lo, x_hi], lci, uci,
                                color=col, alpha=0.12, zorder=1)

    # Spline curve coloured by stratum region
    def _plot_segment(x_lo: float, x_hi: float, col: str, zorder: int = 4) -> None:
        mask = (hr_curve["rbd_prob"] >= x_lo) & (hr_curve["rbd_prob"] <= x_hi)
        seg  = hr_curve[mask]
        if seg.empty:
            return
        ax_hr.fill_between(seg["rbd_prob"], seg["HR_LCI"], seg["HR_UCI"],
                            color=col, alpha=0.25, zorder=zorder - 1)
        ax_hr.plot(seg["rbd_prob"], seg["HR"],
                   color=col, linewidth=2.0, zorder=zorder)

    _plot_segment(p_min, p90,  col_low)
    _plot_segment(p90,  p99,   col_mid)
    _plot_segment(p99,  p_max, col_high)

    ax_hr.scatter([reference], [1.0], s=40, color="grey", zorder=5,
                  label=f"Reference (median={reference:.2f})")

    ann_text = (
        f"P overall = {_fmt_p(lrt['lr_overall_p'])}\n"
        f"P non-linearity = {_fmt_p(lrt['lr_nonlinear_p'])}"
    )
    ax_hr.text(
        0.97, 0.97, ann_text, transform=ax_hr.transAxes,
        fontsize=8, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.85),
    )

    for lbl, x_c, col in [
        ("Low",          (p_min + p90) / 2,  col_low),
        ("Intermediate", (p90  + p99)  / 2,  col_mid),
        ("High",         (p99  + p_max) / 2, col_high),
    ]:
        ax_hr.text(x_c, ax_hr.get_ylim()[1] * 0.92, lbl,
                   fontsize=7, color=col, ha="center", va="top", style="italic")

    ax_hr.text(p90, ax_hr.get_ylim()[0] * 1.02,
               f" p90={p90:.2f}", fontsize=7, color=col_mid, va="bottom")
    ax_hr.text(p99, ax_hr.get_ylim()[0] * 1.02,
               f" p99={p99:.2f}", fontsize=7, color=col_high, va="bottom")

    ax_hr.set_ylabel("Hazard Ratio (vs. median)", fontsize=10)
    ax_hr.set_ylim(bottom=0)
    ax_hr.axhline(1.0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)

    legend_handles: List[mpatches.Patch] = [
        mpatches.Patch(color=col_low, label="Low (<p90): HR=1.0 [ref]"),
    ]
    for grp, col in [("Intermediate", col_mid), ("High", col_high)]:
        if grp in cat_hrs:
            hr, lci, uci = cat_hrs[grp]
            rng = "p90–p99" if grp == "Intermediate" else ">p99"
            legend_handles.append(
                mpatches.Patch(
                    color=col,
                    label=f"{grp} ({rng}): HR={hr:.2f} ({lci:.2f}–{uci:.2f})",
                )
            )
        else:
            rng = "p90–p99" if grp == "Intermediate" else ">p99"
            legend_handles.append(mpatches.Patch(color=col, label=f"{grp} ({rng})"))

    ax_hr.legend(handles=legend_handles, fontsize=7.5, loc="upper left",
                 framealpha=0.85, edgecolor="grey")
    ax_hr.set_title(
        "RBD Dose–Response: Continuous Spline vs. Categorical Cox Estimates\n"
        "Outcome: incident Parkinson's disease (outcome 1a)",
        fontsize=10, pad=8,
    )

    # ── Panel B — Kernel density ─────────────────────────────────────────────
    rbd_cases    = df_subj.loc[df_subj["event"] == 1, "rbd_prob"].values
    rbd_noncases = df_subj.loc[df_subj["event"] == 0, "rbd_prob"].values
    x_eval = np.linspace(p_min, p_max, 500)

    def _kde(x: np.ndarray) -> np.ndarray:
        try:
            return gaussian_kde(x, bw_method="silverman")(x_eval)
        except Exception:
            return np.zeros_like(x_eval)

    ax_den.fill_between(x_eval, _kde(rbd_noncases), alpha=0.25, color="steelblue",
                        label=f"Non-cases (n={len(rbd_noncases):,})")
    ax_den.plot(x_eval, _kde(rbd_noncases), color="steelblue",
                linewidth=1.4, linestyle="--")
    ax_den.fill_between(x_eval, _kde(rbd_cases), alpha=0.35, color=col_high,
                        label=f"Cases (n={len(rbd_cases):,})")
    ax_den.plot(x_eval, _kde(rbd_cases), color=col_high, linewidth=1.6)

    for x_thr, col in [(p90, col_mid), (p99, col_high)]:
        ax_den.axvline(x_thr, color=col, linestyle="--", linewidth=1.2, alpha=0.8)
    ax_den.axvline(reference, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)

    ax_den.axvspan(p_min, p90,  alpha=0.06, color=col_low)
    ax_den.axvspan(p90,  p99,   alpha=0.06, color=col_mid)
    ax_den.axvspan(p99,  p_max, alpha=0.06, color=col_high)

    ax_den.set_xlabel("RBD score (mean across nights)", fontsize=10)
    ax_den.set_ylabel("Density", fontsize=9)
    ax_den.legend(fontsize=7.5, loc="upper right", framealpha=0.85, edgecolor="grey")
    ax_den.set_xlim(p_min, p_max)
    ax_den.set_ylim(bottom=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_path}")


# ── Output saving ─────────────────────────────────────────────────────────────

def save_spline_outputs(
    hr_curve: pd.DataFrame,
    df_subj: pd.DataFrame,
    lrt: Dict[str, float],
    outcome: str,
    reference: float,
    path_report: Path,
    path_results: Path,
) -> None:
    """
    Persist all spline analysis artefacts.

    Files written
    -------------
    path_report/
      table_9c_rbd_spline_lrt.csv   — LRT summary (P_overall, P_non-linearity)
    path_results/
      rbd_spline_hr_curve.csv       — HR curve (rbd_prob, HR, HR_LCI, HR_UCI)
      rbd_spline_model_data.csv     — Subject-level data used for fitting

    Parameters
    ----------
    hr_curve : pd.DataFrame
        HR curve from `compute_hr_curve()`.
    df_subj : pd.DataFrame
        Analytical cohort used for fitting.
    lrt : dict
        LRT statistics from `compute_likelihood_ratio_tests()`.
    outcome : str
        Outcome key.
    reference : float
        Median reference value.
    path_report, path_results : Path
        Output directories.
    """
    df_lrt = pd.DataFrame([{
        "outcome":           outcome,
        "reference_rbd":     round(reference, 4),
        "c_index_spline":    lrt.get("c_index_spline", np.nan),
        "N":                 lrt.get("N", np.nan),
        "events":            lrt.get("events", np.nan),
        "ll_null":           round(lrt["ll_null"], 3),
        "ll_linear":         round(lrt["ll_linear"], 3),
        "ll_spline":         round(lrt["ll_spline"], 3),
        "lr_overall_stat":   round(lrt["lr_overall_stat"], 3),
        "lr_overall_p":      lrt["lr_overall_p"],
        "lr_nonlinear_stat": round(lrt["lr_nonlinear_stat"], 3),
        "lr_nonlinear_p":    lrt["lr_nonlinear_p"],
    }])
    save_table(df_lrt, path_report / "table_9c_rbd_spline_lrt.csv")

    hr_out = hr_curve.copy()
    hr_out.insert(0, "outcome", outcome)
    hr_out.insert(1, "reference_rbd", reference)
    save_table(hr_out, path_results / "rbd_spline_hr_curve.csv")

    save_table(df_subj, path_results / "rbd_spline_model_data.csv")

    print(
        f"  LRT table  : {path_report / 'table_9c_rbd_spline_lrt.csv'}\n"
        f"  HR curve   : {path_results / 'rbd_spline_hr_curve.csv'}\n"
        f"  Model data : {path_results / 'rbd_spline_model_data.csv'}"
    )


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_rbd_spline_analysis(
    df_risk: pd.DataFrame,
    thresholds: dict,
    extended_covariates: List[str],
    rbd_only_rows: List[Dict[str, Any]],
    path_report: Path,
    path_results: Path,
    outcome: str = PRIMARY_OUTCOME,
    n_df: int = SPLINE_DF,
    penalizer: float = RIDGE_PENALIZER,
    n_grid: int = N_GRID,
    figure_dpi: int = 300,
) -> None:
    """
    Full RBD spline dose-response analysis, integrated into the pipeline.

    Runs in the main process after the parallel outcome loop completes.
    Uses the already-loaded `df_risk` (no redundant data loading).

    Steps:
      1. Build analytical cohort (prevalent exclusion).
      2. Set reference = cohort median.
      3. Fit NCS Cox model (df=4).
      4. Compute P_overall and P_non-linearity (LRT).
      5. Compute HR curve (delta method).
      6. Extract categorical HRs from pipeline M0 results.
      7. Generate two-panel figure.
      8. Save table_9c_rbd_spline_lrt.csv, rbd_spline_hr_curve.csv,
         rbd_spline_model_data.csv, figure_rbd_spline.png.

    Parameters
    ----------
    df_risk : pd.DataFrame
        Subject-level, neuro-excluded dataset from `load_prodromal_dataset()`.
    thresholds : dict
        Risk threshold dict from risk_collection.json.
    extended_covariates : list[str]
        Active covariate column names.
    rbd_only_rows : list[dict]
        M0 (RBD-only Cox) rows from `_flatten_summary`; used to extract
        categorical HRs for the figure overlay.
    path_report : Path
        Report output directory (tables + figure).
    path_results : Path
        Full results directory (supplementary data).
    outcome : str
        Outcome to analyse (default: PRIMARY_OUTCOME).
    n_df : int
        Spline degrees of freedom.
    penalizer : float
        Ridge penalizer.
    n_grid : int
        HR curve grid points.
    figure_dpi : int
        Figure output resolution.
    """
    matplotlib.use("Agg")

    print(f"\n  RBD spline dose-response → {outcome}")

    # 1. Analytical cohort
    df_subj = build_spline_survival_dataset(df_risk, outcome, extended_covariates)
    if df_subj is None or df_subj.empty:
        warnings.warn("RBD spline: empty cohort — skipping.")
        return

    # 2. Reference = cohort median
    reference = float(df_subj["rbd_prob"].median())
    print(f"  Reference (median RBD): {reference:.4f}")

    # 3. Fit spline model
    try:
        cph_spline, spline_cols, df_mod, design_info = fit_spline_model(
            df=df_subj,
            time_col="time",
            event_col="event",
            rbd_col="rbd_prob",
            covariates=extended_covariates,
            n_df=n_df,
            penalizer=penalizer,
        )
    except Exception as exc:
        warnings.warn(f"RBD spline fit failed: {exc}")
        return

    print(
        f"  C-index: {cph_spline.concordance_index_:.4f}  "
        f"Log-L: {cph_spline.log_likelihood_:.2f}"
    )

    # 4. LRT tests
    lrt = compute_likelihood_ratio_tests(
        df_mod=df_mod,
        cph_spline=cph_spline,
        time_col="time",
        event_col="event",
        rbd_col="rbd_prob",
        spline_cols=spline_cols,
        covariates=extended_covariates,
        n_df=n_df,
        penalizer=penalizer,
    )
    print(
        f"  P_overall      (df={n_df}):   "
        f"LR={lrt['lr_overall_stat']:.2f}, {_fmt_p(lrt['lr_overall_p'])}\n"
        f"  P_non-linearity (df={n_df - 1}): "
        f"LR={lrt['lr_nonlinear_stat']:.2f}, {_fmt_p(lrt['lr_nonlinear_p'])}"
    )

    # 5. HR curve
    hr_curve = compute_hr_curve(
        cph_spline=cph_spline,
        spline_cols=spline_cols,
        rbd_col="rbd_prob",
        df_mod=df_mod,
        reference=reference,
        design_info=design_info,
        n_grid=n_grid,
    )
    if hr_curve.empty:
        warnings.warn("RBD spline: HR curve failed — skipping figure.")
        return

    ref_row = hr_curve.iloc[(hr_curve["rbd_prob"] - reference).abs().argmin()]
    print(f"  HR at reference: {ref_row['HR']:.6f} [should be ~1.0]")

    # 6. Categorical HRs from pipeline M0 results
    cat_hrs = extract_categorical_hrs(rbd_only_rows, outcome=outcome)
    if not cat_hrs:
        warnings.warn(
            "No categorical HRs found in rbd_only_rows — "
            "figure overlays will be omitted."
        )

    # 7. Figure
    plot_rbd_spline_figure(
        hr_curve=hr_curve,
        df_subj=df_subj,
        lrt=lrt,
        thresholds=thresholds,
        reference=reference,
        cat_hrs=cat_hrs,
        out_path=path_report / "figure_rbd_spline.png",
        dpi=figure_dpi,
    )

    # 8. Tables and data
    save_spline_outputs(
        hr_curve=hr_curve,
        df_subj=df_subj,
        lrt=lrt,
        outcome=outcome,
        reference=reference,
        path_report=path_report,
        path_results=path_results,
    )
