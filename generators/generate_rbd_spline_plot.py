"""
Generate publication figure: RBD dose-response spline analysis.

Two-panel figure for outcome_1a_pd_only (incident PD):

  Panel A — HR curve (natural cubic spline Cox, df=4)
    - Reference = analytical-cohort median RBD score
    - 95% CI ribbon via delta method
    - Categorical HR overlays: High (HR=3.76) and Intermediate (HR=1.89)
      as horizontal bands within their respective strata
    - Vertical threshold lines at p90 and p99 (Low/Mid/High boundaries)
    - Formal test statistics reported in annotation box:
        P_overall (LRT spline vs null, df=4)
        P_non-linearity (LRT spline vs linear, df=3)

  Panel B — Event-weighted distribution
    - Kernel density of continuous RBD scores: cases (solid) vs. non-cases (dashed)
    - Threshold lines matching Panel A

Data loading uses get_clean_risk_data() (neuro_exclude==0 applied automatically).
Prevalent cases are removed via surv_days.notna() to match the primary analysis.
Reference point is computed on the analytical cohort (post-exclusion), consistent
with Cox model estimation sample.

Output:
  results/cox_prodromal_abk_03_19_2026_14_17_29/report/figure_rbd_spline.png
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["svg.fonttype"] = "none"
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from lifelines import CoxPHFitter
from patsy import dmatrix as patsy_dmatrix, build_design_matrices
from scipy.stats import gaussian_kde

from library.risk.risk_helpers import get_clean_risk_data, make_subject_level
from library.cox_prodromal.plotting import _save_ghost_copy
from library.cox_prodromal.cox_config import (
    BASE_COVARIATES,
    ALCOHOL_CANDIDATES,
    SMOKING_CANDIDATES,
    RIDGE_PENALIZER,
    RISK_PALETTE,
)

# ── Constants ──────────────────────────────────────────────────────────────────

PRIMARY_OUTCOME: str = "outcome_1a_pd_only"
SPLINE_DF: int = 4
OUT_DIR: Path = Path(
    "results/cox_prodromal_abk_03_19_2026_14_17_29/report"
)
OUT_PATH: Path = OUT_DIR / "figure_rbd_spline.png"
N_GRID: int = 300  # grid points for HR curve

# Categorical HRs (percentile_3g, M0_rbd_only) — from rbd_only_cox.xlsx.
# Reference group = Low (< p90), fixed at HR=1.
_CAT_HRS: Dict[str, Tuple[float, float, float]] = {
    "Intermediate": (1.8870, 1.6436, 2.1665),
    "High":         (3.7598, 2.4329, 5.8104),
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def _select_covariate(
    df: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    """Return the first candidate column present in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_analytical_cohort(
    outcome: str = PRIMARY_OUTCOME,
) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """
    Load, filter, and construct the analytical cohort for *outcome*.

    Applies, in order:
      1. Neurological exclusion (via get_clean_risk_data, neuro_exclude==0)
      2. Prevalent-case exclusion (surv_days.notna())
      3. Collapse to subject level

    Parameters
    ----------
    outcome : str
        Outcome prefix (e.g. 'outcome_1a_pd_only').

    Returns
    -------
    df_subj : pd.DataFrame
        Subject-level analytical cohort with columns:
          rbd_prob, time (years), event, covariates.
    thresholds : dict
        Risk threshold dictionary (from risk_collection.json).
    covariates : list[str]
        Covariate column names used in all Cox models.
    """
    print("Loading data via get_clean_risk_data …")
    thresholds, df_night = get_clean_risk_data(
        file_name="ehr_diag_pd_rbd_only_all"
    )

    # Collapse to subject level; renames prob_mean → rbd_prob
    df_subj = make_subject_level(df_night, id_col="eid", prob_col="prob_mean")
    print(f"  Subject-level (post neuro-exclusion): {len(df_subj):,}")

    # ── Build survival columns ─────────────────────────────────────────────
    surv_col = f"{outcome}__surv_days"
    event_col = f"{outcome}__incident"

    if surv_col not in df_subj.columns or event_col not in df_subj.columns:
        raise KeyError(
            f"Expected columns '{surv_col}' and '{event_col}' not found. "
            f"Available: {[c for c in df_subj.columns if outcome in c]}"
        )

    # Convert days → years; remove prevalent cases (surv_days is NaN for them)
    df_subj = df_subj[df_subj[surv_col].notna()].copy()
    df_subj["time"] = df_subj[surv_col] / 365.25
    df_subj["event"] = df_subj[event_col].astype(int)
    print(
        f"  Analytical cohort: {len(df_subj):,} subjects, "
        f"{df_subj['event'].sum()} events"
    )

    # ── Covariates ─────────────────────────────────────────────────────────
    covariates: List[str] = list(BASE_COVARIATES)

    smoking = _select_covariate(df_subj, SMOKING_CANDIDATES)
    if smoking:
        # Harmonise to a single 'cov_smoking' column for clean model matrices
        df_subj = df_subj.rename(columns={smoking: "cov_smoking"})
        covariates.append("cov_smoking")

    alcohol = _select_covariate(df_subj, ALCOHOL_CANDIDATES)
    if alcohol:
        df_subj = df_subj.rename(columns={alcohol: "cov_alcohol"})
        covariates.append("cov_alcohol")

    # Drop rows with any missing covariate or RBD score
    keep_cols = ["eid", "rbd_prob", "time", "event"] + covariates
    df_subj = df_subj[[c for c in keep_cols if c in df_subj.columns]].dropna()
    print(
        f"  After covariate dropna: {len(df_subj):,} subjects, "
        f"{df_subj['event'].sum()} events"
    )

    return df_subj, thresholds, covariates


# ── Spline fitting ─────────────────────────────────────────────────────────────

def _build_spline_matrix(
    x: np.ndarray, n_df: int, spline_cols: List[str]
) -> pd.DataFrame:
    """
    Build natural cubic spline basis matrix for a 1-D array *x*.

    The basis is constructed via patsy cr() with the same knot positions
    implied by the training data used in fit_spline_model().

    Parameters
    ----------
    x : np.ndarray
        Values to evaluate the spline basis at.
    n_df : int
        Spline degrees of freedom (passed to cr()).
    spline_cols : list[str]
        Column names to assign (must match length of basis).

    Returns
    -------
    pd.DataFrame
        Shape (len(x), n_df).
    """
    dm = patsy_dmatrix(
        f"cr(x, df={n_df}) - 1",
        {"x": x},
        return_type="dataframe",
    )
    dm.columns = spline_cols
    return dm


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
    Fit natural cubic spline Cox model for the RBD dose-response.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col, event_col : str
        Survival time (years) and event indicator columns.
    rbd_col : str
        Continuous RBD probability column.
    covariates : list[str]
        Adjustment covariates.
    n_df : int
        Spline degrees of freedom.
    penalizer : float
        Ridge penalizer for CoxPHFitter.

    Returns
    -------
    cph_s : CoxPHFitter
        Fitted spline Cox model.
    spline_cols : list[str]
        Names of the spline basis columns.
    df_mod : pd.DataFrame
        Complete-case dataset used for fitting (needed for knot positions).
    design_info : patsy.DesignInfo
        Patsy design info capturing knot positions — required to evaluate
        the spline basis on new data (including single reference points)
        without re-fitting knots from scratch.
    """
    cols = [time_col, event_col, rbd_col] + covariates
    df_mod = df[cols].dropna().copy()
    df_mod[rbd_col] = pd.to_numeric(df_mod[rbd_col], errors="coerce")
    df_mod = df_mod.dropna()

    # Build spline basis (knots anchored to the analytical cohort).
    # design_info captures the knot positions for later evaluation on new data.
    spline_dm = patsy_dmatrix(
        f"cr(x, df={n_df}) - 1",
        {"x": df_mod[rbd_col].values},
        return_type="dataframe",
    )
    design_info = spline_dm.design_info  # stores knot positions
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
    cph_s.fit(X_spline, duration_col=time_col, event_col=event_col, robust=True)

    return cph_s, spline_cols, df_mod, design_info


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
    Compute two likelihood-ratio tests against the fitted spline model.

    Test 1 — P_overall (df=n_df):
        H0: β_spline = 0, i.e. RBD has no association with hazard.
        Compares spline model vs. covariate-only null model.

    Test 2 — P_non-linearity (df=n_df-1):
        H0: the dose-response is linear in RBD.
        Compares spline model vs. linear-RBD model.

    Both tests use the log-likelihood from the fitted CoxPH models.
    With ridge penalization the LRT is approximate; this is standard
    practice for spline Cox (Wood 2017, Royston & Sauerbrei 2007).

    Parameters
    ----------
    df_mod : pd.DataFrame
        Complete-case data (same rows used for cph_spline).
    cph_spline : CoxPHFitter
        Already-fitted spline model.
    time_col, event_col : str
        Survival columns.
    rbd_col : str
        Continuous RBD column name.
    spline_cols : list[str]
        Spline basis column names.
    covariates : list[str]
        Adjustment covariates.
    n_df : int
        Spline df (used to infer degrees of freedom for LRT).
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict with keys:
        lr_overall_stat, lr_overall_p,
        lr_nonlinear_stat, lr_nonlinear_p,
        ll_spline, ll_linear, ll_null.
    """
    # ── Null model: covariates only ─────────────────────────────────────────
    X_null = pd.concat(
        [
            df_mod[[time_col, event_col]].reset_index(drop=True),
            df_mod[covariates].reset_index(drop=True),
        ],
        axis=1,
    )
    cph_null = CoxPHFitter(penalizer=penalizer)
    cph_null.fit(
        X_null, duration_col=time_col, event_col=event_col, robust=True
    )
    ll_null = cph_null.log_likelihood_

    # ── Linear model: linear RBD term ───────────────────────────────────────
    X_lin = pd.concat(
        [
            df_mod[[time_col, event_col, rbd_col]].reset_index(drop=True),
            df_mod[covariates].reset_index(drop=True),
        ],
        axis=1,
    )
    cph_lin = CoxPHFitter(penalizer=penalizer)
    cph_lin.fit(
        X_lin, duration_col=time_col, event_col=event_col, robust=True
    )
    ll_linear = cph_lin.log_likelihood_

    ll_spline = cph_spline.log_likelihood_

    # ── LRT statistics ─────────────────────────────────────────────────────
    lr_overall_stat = max(-2.0 * (ll_null - ll_spline), 0.0)
    lr_nonlinear_stat = max(-2.0 * (ll_linear - ll_spline), 0.0)

    lr_overall_p = float(scipy_stats.chi2.sf(lr_overall_stat, df=n_df))
    lr_nonlinear_p = float(scipy_stats.chi2.sf(lr_nonlinear_stat, df=n_df - 1))

    return {
        "lr_overall_stat": lr_overall_stat,
        "lr_overall_p": lr_overall_p,
        "lr_nonlinear_stat": lr_nonlinear_stat,
        "lr_nonlinear_p": lr_nonlinear_p,
        "ll_spline": ll_spline,
        "ll_linear": ll_linear,
        "ll_null": ll_null,
    }


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
    Compute HR curve relative to *reference* using the delta method.

    HR(x) = exp[f(x) - f(reference)]

    where f(x) is the linear predictor contribution of the spline basis:
      f(x) = Σ_j β_j · s_j(x)

    Variance of log HR(x):
      Var[f(x) - f(ref)] = (s(x) - s(ref))^T · V_β · (s(x) - s(ref))

    where V_β is the covariance matrix of the spline coefficients (extracted
    from the fitted model's variance_matrix_).

    The spline basis for both the grid and the reference point is computed
    using the *same* knot positions as the training model, via patsy
    build_design_matrices(design_info, ...). This avoids the issue where
    patsy cannot infer knots from a single evaluation point.

    Parameters
    ----------
    cph_spline : CoxPHFitter
        Fitted spline Cox model.
    spline_cols : list[str]
        Spline basis column names.
    rbd_col : str
        Original RBD column (for range extraction).
    df_mod : pd.DataFrame
        Modelling data (used for range extraction only).
    reference : float
        RBD value used as reference (HR=1 by definition).
    design_info : patsy.DesignInfo
        Stored from fit_spline_model(); encodes knot positions.
    n_grid : int
        Number of evaluation points.

    Returns
    -------
    pd.DataFrame
        Columns: rbd_prob, HR, HR_LCI, HR_UCI.
    """
    p_min = float(df_mod[rbd_col].min())
    p_max = float(df_mod[rbd_col].max())
    grid = np.linspace(p_min, p_max, n_grid)

    # Evaluate spline basis using stored design_info (correct knot positions)
    try:
        grid_dm_raw = build_design_matrices(
            [design_info], {"x": grid}
        )[0]
        grid_dm = pd.DataFrame(
            np.asarray(grid_dm_raw), columns=spline_cols
        )

        ref_dm_raw = build_design_matrices(
            [design_info], {"x": np.array([reference])}
        )[0]
        ref_dm = pd.DataFrame(
            np.asarray(ref_dm_raw), columns=spline_cols
        )
    except Exception as exc:
        warnings.warn(f"Spline basis evaluation failed: {exc}")
        return pd.DataFrame()

    # Spline coefficients
    betas = np.array([
        cph_spline.params_[c] for c in spline_cols
        if c in cph_spline.params_.index
    ])
    if len(betas) != len(spline_cols):
        warnings.warn("Coefficient extraction failed — spline column mismatch.")
        return pd.DataFrame()

    log_hr = grid_dm.values @ betas - (ref_dm.values @ betas)[0]

    # Delta-method variance
    try:
        spline_idx = [
            list(cph_spline.params_.index).index(c)
            for c in spline_cols
        ]
        V = cph_spline.variance_matrix_.iloc[spline_idx, spline_idx].values
        diff = grid_dm.values - ref_dm.values  # shape (n_grid, n_df)
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


# ── Plotting ───────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    """Format a p-value for figure annotation."""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def plot_rbd_spline_figure(
    hr_curve: pd.DataFrame,
    df_subj: pd.DataFrame,
    lrt: Dict[str, float],
    thresholds: dict,
    reference: float,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """
    Generate and save the two-panel RBD spline figure.

    Panel A: HR curve with CI ribbon, categorical HR overlays,
             stratum shading, and threshold lines.
    Panel B: Kernel density of RBD scores for cases vs. non-cases,
             with threshold lines matching Panel A.

    Parameters
    ----------
    hr_curve : pd.DataFrame
        Output of compute_hr_curve(). Columns: rbd_prob, HR, HR_LCI, HR_UCI.
    df_subj : pd.DataFrame
        Analytical cohort (subject-level). Must have 'rbd_prob' and 'event'.
    lrt : dict
        Output of compute_likelihood_ratio_tests().
    thresholds : dict
        Risk threshold dictionary from risk_collection.json.
    reference : float
        Reference RBD value (median of analytical cohort).
    out_path : Path
        Output file path.
    dpi : int
        Output resolution.
    """
    # ── Extract threshold boundaries ────────────────────────────────────────
    thr_3g = thresholds["percentile_3g"]["rbd_only_distribution"]["all"]
    p90: float = thr_3g["p90"]   # Low/Intermediate boundary
    p99: float = thr_3g["p99"]   # Intermediate/High boundary

    p_min = float(df_subj["rbd_prob"].min())
    p_max = float(df_subj["rbd_prob"].max())

    # ── Stratum colours ─────────────────────────────────────────────────────
    col_low  = RISK_PALETTE.rbd_low   # blue
    col_mid  = RISK_PALETTE.rbd_mid   # salmon
    col_high = RISK_PALETTE.rbd_high  # red

    # ── Figure layout ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.5, 6.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.06)

    ax_hr, ax_den = axes

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel A — HR curve
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Background stratum shading ──────────────────────────────────────────
    ax_hr.axvspan(p_min, p90,  alpha=0.06, color=col_low,  zorder=0)
    ax_hr.axvspan(p90,  p99,   alpha=0.06, color=col_mid,  zorder=0)
    ax_hr.axvspan(p99,  p_max, alpha=0.06, color=col_high, zorder=0)

    # ── Threshold vertical lines ────────────────────────────────────────────
    for x_thr, col in [(p90, col_mid), (p99, col_high)]:
        ax_hr.axvline(
            x_thr, color=col, linestyle="--", linewidth=1.2, alpha=0.8, zorder=1
        )

    # ── Reference vertical line ─────────────────────────────────────────────
    ax_hr.axvline(
        reference, color="grey", linestyle=":", linewidth=0.9, alpha=0.7, zorder=1
    )

    # ── Categorical HR overlays ─────────────────────────────────────────────
    # Draw horizontal bands (HR ± 95% CI) for Mid and High strata
    # Reference Low: HR=1.0 line
    ax_hr.axhline(1.0, color=col_low, linestyle="-", linewidth=0.8, alpha=0.6)

    cat_cfg = [
        ("Intermediate", p90, p99,    col_mid),
        ("High",         p99, p_max,  col_high),
    ]
    for grp, x_lo, x_hi, col in cat_cfg:
        hr, lci, uci = _CAT_HRS[grp]
        # Horizontal line at point estimate
        ax_hr.hlines(
            hr, x_lo, x_hi,
            colors=col, linewidths=1.4, linestyles="--", alpha=0.9, zorder=2
        )
        # CI band
        ax_hr.fill_between(
            [x_lo, x_hi], lci, uci,
            color=col, alpha=0.12, zorder=1
        )

    # ── Spline HR curve ─────────────────────────────────────────────────────
    # Colour the spline line by stratum region
    def _plot_segment(
        x_lo: float, x_hi: float, col: str, zorder: int = 4
    ) -> None:
        mask = (hr_curve["rbd_prob"] >= x_lo) & (hr_curve["rbd_prob"] <= x_hi)
        seg = hr_curve[mask]
        if seg.empty:
            return
        ax_hr.fill_between(
            seg["rbd_prob"], seg["HR_LCI"], seg["HR_UCI"],
            color=col, alpha=0.25, zorder=zorder - 1
        )
        ax_hr.plot(
            seg["rbd_prob"], seg["HR"],
            color=col, linewidth=2.0, zorder=zorder
        )

    # Low stratum
    _plot_segment(p_min, p90,  col_low)
    # Mid stratum
    _plot_segment(p90,  p99,   col_mid)
    # High stratum
    _plot_segment(p99,  p_max, col_high)

    # ── Reference point marker ──────────────────────────────────────────────
    ax_hr.scatter(
        [reference], [1.0],
        s=40, color="grey", zorder=5, label=f"Reference (median={reference:.2f})"
    )

    # ── LRT annotation box ──────────────────────────────────────────────────
    ann_text = (
        f"P overall = {_fmt_p(lrt['lr_overall_p'])}\n"
        f"P non-linearity = {_fmt_p(lrt['lr_nonlinear_p'])}"
    )
    ax_hr.text(
        0.97, 0.97, ann_text,
        transform=ax_hr.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="grey", alpha=0.85),
    )

    # ── Stratum labels ──────────────────────────────────────────────────────
    y_lbl = ax_hr.get_ylim()[1] * 0.95 if ax_hr.get_ylim()[1] > 1 else 2.5
    for lbl, x_c, col in [
        ("Low",          (p_min + p90) / 2,  col_low),
        ("Intermediate", (p90  + p99)  / 2,  col_mid),
        ("High",         (p99  + p_max) / 2, col_high),
    ]:
        ax_hr.text(
            x_c, ax_hr.get_ylim()[1] * 0.92,
            lbl, fontsize=7, color=col,
            ha="center", va="top", style="italic",
        )

    # ── Threshold annotations ───────────────────────────────────────────────
    ax_hr.text(
        p90, ax_hr.get_ylim()[0] * 1.02,
        f" p90={p90:.2f}", fontsize=7, color=col_mid, va="bottom"
    )
    ax_hr.text(
        p99, ax_hr.get_ylim()[0] * 1.02,
        f" p99={p99:.2f}", fontsize=7, color=col_high, va="bottom"
    )

    ax_hr.set_ylabel("Hazard Ratio (vs. median)", fontsize=10)
    ax_hr.set_ylim(bottom=0)
    ax_hr.axhline(1.0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=col_low,  label=f"Low (<p90): HR=1.0 [ref]"),
        mpatches.Patch(color=col_mid,  label=f"Intermediate (p90–p99): HR={_CAT_HRS['Intermediate'][0]:.2f} "
                                              f"({_CAT_HRS['Intermediate'][1]:.2f}–{_CAT_HRS['Intermediate'][2]:.2f})"),
        mpatches.Patch(color=col_high, label=f"High (>p99): HR={_CAT_HRS['High'][0]:.2f} "
                                             f"({_CAT_HRS['High'][1]:.2f}–{_CAT_HRS['High'][2]:.2f})"),
    ]
    ax_hr.legend(
        handles=legend_handles, fontsize=7.5, loc="upper left",
        framealpha=0.85, edgecolor="grey"
    )

    ax_hr.set_title(
        "RBD Dose–Response: Continuous Spline vs. Categorical Cox Estimates\n"
        "Outcome: incident Parkinson's disease (outcome 1a)",
        fontsize=10, pad=8
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Panel B — Distribution: cases vs. non-cases
    # ═══════════════════════════════════════════════════════════════════════════

    rbd_cases    = df_subj.loc[df_subj["event"] == 1, "rbd_prob"].values
    rbd_noncases = df_subj.loc[df_subj["event"] == 0, "rbd_prob"].values

    x_eval = np.linspace(p_min, p_max, 500)

    def _kde(x: np.ndarray) -> np.ndarray:
        """Gaussian KDE evaluated on x_eval."""
        try:
            kde = gaussian_kde(x, bw_method="silverman")
            return kde(x_eval)
        except Exception:
            return np.zeros_like(x_eval)

    kde_cases    = _kde(rbd_cases)
    kde_noncases = _kde(rbd_noncases)

    # Normalize to unit area for visual comparability (probability density)
    ax_den.fill_between(
        x_eval, kde_noncases,
        alpha=0.25, color="steelblue", label=f"Non-cases (n={len(rbd_noncases):,})"
    )
    ax_den.plot(x_eval, kde_noncases, color="steelblue", linewidth=1.4,
                linestyle="--")

    ax_den.fill_between(
        x_eval, kde_cases,
        alpha=0.35, color=col_high, label=f"Cases (n={len(rbd_cases):,})"
    )
    ax_den.plot(x_eval, kde_cases, color=col_high, linewidth=1.6)

    # Threshold lines matching Panel A
    for x_thr, col in [(p90, col_mid), (p99, col_high)]:
        ax_den.axvline(
            x_thr, color=col, linestyle="--", linewidth=1.2, alpha=0.8
        )

    ax_den.axvline(
        reference, color="grey", linestyle=":", linewidth=0.9, alpha=0.7
    )

    # Stratum background (same as Panel A)
    ax_den.axvspan(p_min, p90,  alpha=0.06, color=col_low)
    ax_den.axvspan(p90,  p99,   alpha=0.06, color=col_mid)
    ax_den.axvspan(p99,  p_max, alpha=0.06, color=col_high)

    ax_den.set_xlabel("RBD score (mean across nights)", fontsize=10)
    ax_den.set_ylabel("Density", fontsize=9)
    ax_den.legend(fontsize=7.5, loc="upper right", framealpha=0.85,
                  edgecolor="grey")
    ax_den.set_xlim(p_min, p_max)
    ax_den.set_ylim(bottom=0)

    # ── Save ────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    _save_ghost_copy(fig, out_path.with_name(out_path.stem + "_ghost"))
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")
    print(f"Ghost saved:  {out_path.with_name(out_path.stem + '_ghost')}.[png,svg]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """End-to-end generation of the RBD spline dose-response figure."""

    # 1. Load data
    df_subj, thresholds, covariates = load_analytical_cohort(
        outcome=PRIMARY_OUTCOME
    )

    # 2. Reference = analytical-cohort median (post-exclusion)
    reference = float(df_subj["rbd_prob"].median())
    print(f"\nReference point (median RBD): {reference:.4f}")

    # 3. Fit spline Cox model
    print("\nFitting spline Cox model …")
    cph_spline, spline_cols, df_mod, design_info = fit_spline_model(
        df=df_subj,
        time_col="time",
        event_col="event",
        rbd_col="rbd_prob",
        covariates=covariates,
        n_df=SPLINE_DF,
        penalizer=RIDGE_PENALIZER,
    )
    print(
        f"  C-index (spline): {cph_spline.concordance_index_:.4f}  "
        f"| Log-L: {cph_spline.log_likelihood_:.2f}"
    )

    # 4. Likelihood-ratio tests
    print("\nComputing likelihood-ratio tests …")
    lrt = compute_likelihood_ratio_tests(
        df_mod=df_mod,
        cph_spline=cph_spline,
        time_col="time",
        event_col="event",
        rbd_col="rbd_prob",
        spline_cols=spline_cols,
        covariates=covariates,
        n_df=SPLINE_DF,
        penalizer=RIDGE_PENALIZER,
    )
    print(
        f"  P_overall      (df={SPLINE_DF}): LR={lrt['lr_overall_stat']:.2f}, "
        f"{_fmt_p(lrt['lr_overall_p'])}"
    )
    print(
        f"  P_non-linearity (df={SPLINE_DF-1}): LR={lrt['lr_nonlinear_stat']:.2f}, "
        f"{_fmt_p(lrt['lr_nonlinear_p'])}"
    )

    # 5. HR curve with median as reference
    print("\nComputing HR curve (reference = median) …")
    hr_curve = compute_hr_curve(
        cph_spline=cph_spline,
        spline_cols=spline_cols,
        rbd_col="rbd_prob",
        df_mod=df_mod,
        reference=reference,
        design_info=design_info,
        n_grid=N_GRID,
    )

    if hr_curve.empty:
        raise RuntimeError("HR curve computation failed — empty DataFrame.")

    # Quick sanity check: HR at reference should be ~1.0
    ref_row = hr_curve.iloc[(hr_curve["rbd_prob"] - reference).abs().argmin()]
    print(f"  HR at reference ({reference:.4f}): {ref_row['HR']:.6f} [should be 1.0]")

    # 6. Generate figure
    print("\nGenerating figure …")
    plot_rbd_spline_figure(
        hr_curve=hr_curve,
        df_subj=df_subj,
        lrt=lrt,
        thresholds=thresholds,
        reference=reference,
        out_path=OUT_PATH,
        dpi=300,
    )


if __name__ == "__main__":
    main()
