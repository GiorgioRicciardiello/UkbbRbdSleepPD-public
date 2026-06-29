"""
Baron & Kenny mediation analysis with product-of-coefficients bootstrap.

For each prodromal variable P and outcome = outcome_1a_pd_only:

  Step 1 (c-path):  Cox: outcome ~ P + covariates
  Step 2 (a-path):  OLS: rbd_score_z ~ P + covariates
  Step 3 (joint):   Cox: outcome ~ P + rbd_score_z + covariates
  Step 4:           Indirect = beta_a * beta_b; PM% = indirect / beta_c * 100
  Step 5:           Bootstrap 95% CI (B=1000)
  Step 5.5:         Supplementary categorical 3g b-path
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from tqdm import tqdm

from library.cox_prodromal.cox_config import (
    BOOTSTRAP_JOBS,
    BOOTSTRAP_RIDGE_PENALIZER,
    RIDGE_PENALIZER,
)
from library.cox_prodromal.diagnostics import apply_fdr, extract_model_fit_metrics


@dataclass(frozen=True)
class MediationStepResult:
    """Point estimates for all Baron & Kenny paths."""
    variable: str
    label: str
    # c-path (total effect)
    beta_c: float
    hr_c: float
    hr_c_lower: float
    hr_c_upper: float
    p_c: float
    c_index_c: float
    # a-path (prodromal -> RBD)
    beta_a: float
    se_a: float
    p_a: float
    r2_a: float
    # b-path (RBD -> outcome, adjusted for prodromal)
    beta_b: float
    hr_b: float
    hr_b_lower: float
    hr_b_upper: float
    p_b: float
    # c'-path (direct effect, adjusted for RBD)
    beta_cprime: float
    hr_cprime: float
    hr_cprime_lower: float
    hr_cprime_upper: float
    p_cprime: float
    c_index_joint: float
    # Indirect effect
    beta_indirect: float
    hr_indirect: float
    pm_pct: float
    inconsistent_mediation: bool
    n: int
    events: int


@dataclass(frozen=True)
class MediationBootstrapResult:
    """Bootstrap confidence intervals for indirect effect and PM%."""
    variable: str
    label: str
    hr_indirect: float
    hr_indirect_lci: float
    hr_indirect_uci: float
    pm_pct: float
    pm_pct_lci: float
    pm_pct_uci: float
    n_converged: int
    n_bootstrap: int
    n_discarded: int


@dataclass(frozen=True)
class Supplementary3gResult:
    """Supplementary: categorical 3-group b-path."""
    variable: str
    label: str
    hr_intermediate_vs_low: float
    hr_intermediate_lci: float
    hr_intermediate_uci: float
    p_intermediate: float
    hr_high_vs_low: float
    hr_high_lci: float
    hr_high_uci: float
    p_high: float
    beta_cprime_3g: float
    hr_cprime_3g: float
    n: int
    events: int


@dataclass(frozen=True)
class MediationModelPerf:
    """Model performance for each Cox model in the mediation chain."""
    variable: str
    label: str
    model: str
    c_index: float
    aic: float
    bic: float
    log_likelihood: float
    lrt_stat: float
    lrt_p: float
    n_params: int


def _make_sksurv_y(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Build structured survival array for scikit-survival.

    Parameters
    ----------
    event : np.ndarray
        Binary event indicator (0/1).
    time : np.ndarray
        Follow-up duration.

    Returns
    -------
    np.ndarray
        Structured array with dtype [('event', bool), ('time', float)].
    """
    y = np.empty(len(event), dtype=[("event", bool), ("time", float)])
    y["event"] = event.astype(bool)
    y["time"] = time.astype(float)
    return y


def _one_mediation_boot(
    boot_seed: int,
    y_surv: np.ndarray,
    X_a: np.ndarray,
    y_rbd: np.ndarray,
    X_joint: np.ndarray,
    X_c: np.ndarray,
    prod_idx_joint: int,
    rbd_idx_joint: int,
    prod_idx_c: int,
    penalizer: float,
    min_events: int = 3,
) -> Optional[Tuple[float, float]]:
    """Single bootstrap iteration for mediation indirect effect and PM%.

    Uses scikit-survival CoxPHSurvivalAnalysis (~3-5x faster than lifelines)
    and numpy least-squares for the OLS a-path. Accepts only numpy arrays
    to avoid DataFrame pickling overhead in loky worker processes.

    Parameters
    ----------
    boot_seed : int
        Seed for this bootstrap draw (seed + b for reproducibility).
    y_surv : np.ndarray
        Structured survival array with ('event', 'time') dtype.
    X_a : np.ndarray
        OLS design matrix for a-path: [const, prod, *covariates].
    y_rbd : np.ndarray
        RBD z-score outcome for a-path OLS.
    X_joint : np.ndarray
        Feature matrix for joint Cox: [prod, rbd, *covariates].
    X_c : np.ndarray
        Feature matrix for c-path Cox: [prod, *covariates].
    prod_idx_joint : int
        Column index of prodromal variable in X_joint (always 0).
    rbd_idx_joint : int
        Column index of RBD mediator in X_joint (always 1).
    prod_idx_c : int
        Column index of prodromal variable in X_c (always 0).
    penalizer : float
        L2 regularization alpha for CoxPHSurvivalAnalysis.
    min_events : int
        Minimum events required in bootstrap sample (default: 3).

    Returns
    -------
    tuple[float, float] or None
        (indirect_effect, pm_pct) or None if resample failed.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    rng = np.random.default_rng(boot_seed)
    n = len(y_surv)
    idx = rng.choice(n, size=n, replace=True)

    y_b = y_surv[idx]
    if y_b["event"].sum() < min_events:
        return None

    X_a_b = X_a[idx]
    y_rbd_b = y_rbd[idx]
    X_joint_b = X_joint[idx]
    X_c_b = X_c[idx]

    try:
        # a-path: numpy least squares (no intercept — X_a already includes const)
        coef_a, _, _, _ = np.linalg.lstsq(X_a_b, y_rbd_b, rcond=None)
        beta_a_b = float(coef_a[1])  # index 1 = prod_col (index 0 is const)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # joint Cox -> beta_b (RBD mediator coefficient)
            cph_joint = CoxPHSurvivalAnalysis(alpha=penalizer, ties="breslow")
            cph_joint.fit(X_joint_b, y_b)
            beta_b_b = float(cph_joint.coef_[rbd_idx_joint])

            # c-path Cox -> beta_c (prodromal total effect coefficient)
            cph_c = CoxPHSurvivalAnalysis(alpha=penalizer, ties="breslow")
            cph_c.fit(X_c_b, y_b)
            beta_c_b = float(cph_c.coef_[prod_idx_c])

        indirect_b = beta_a_b * beta_b_b
        pm_b = (indirect_b / beta_c_b * 100.0) if abs(beta_c_b) > 1e-10 else np.nan
        return indirect_b, pm_b

    except Exception:
        return None


def _fit_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    predictors: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[CoxPHFitter]:
    """
    Fit a CoxPHFitter with standard settings.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain time_col, event_col, and all predictors.
    time_col, event_col : str
        Duration and event indicator columns.
    predictors : list[str]
        Covariate column names.
    penalizer : float
        Ridge penalty.

    Returns
    -------
    CoxPHFitter or None
    """
    cols = [time_col, event_col] + predictors
    df_fit = df[cols].dropna().copy()
    # Ensure all columns are float64 for lifelines
    for c in df_fit.columns:
        df_fit[c] = pd.to_numeric(df_fit[c], errors="coerce")
    df_fit = df_fit.dropna()
    if df_fit.empty or df_fit[event_col].sum() < 5:
        return None
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df_fit, duration_col=time_col, event_col=event_col, robust=True)
        return cph
    except Exception as exc:
        warnings.warn(f"Cox fit failed: {exc}")
        return None


def _fit_ols(
    df: pd.DataFrame,
    y_col: str,
    predictors: List[str],
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Fit OLS with HC3 robust SEs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain y_col and all predictors.
    y_col : str
        Outcome column.
    predictors : list[str]
        Feature columns.

    Returns
    -------
    RegressionResults or None
    """
    cols = [y_col] + predictors
    df_cc = df[cols].dropna()
    if len(df_cc) < 30:
        return None
    y = df_cc[y_col].to_numpy(dtype=float)
    X = sm.add_constant(df_cc[predictors].to_numpy(dtype=float))
    try:
        return sm.OLS(y, X).fit(cov_type="HC3")
    except Exception as exc:
        warnings.warn(f"OLS fit failed: {exc}")
        return None


def _extract_cox_coef(
    cph: CoxPHFitter,
    var_name: str,
) -> Optional[dict]:
    """
    Extract coefficient, HR, CI, p for a named variable from fitted Cox.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted model.
    var_name : str
        Variable name in the model.

    Returns
    -------
    dict or None
        {beta, hr, hr_lower, hr_upper, p}
    """
    summary = cph.summary
    if var_name not in summary.index:
        return None
    row = summary.loc[var_name]
    return {
        "beta": float(row["coef"]),
        "hr": float(row["exp(coef)"]),
        "hr_lower": float(row["exp(coef) lower 95%"]),
        "hr_upper": float(row["exp(coef) upper 95%"]),
        "p": float(row["p"]),
    }


def fit_mediation_steps(
    df: pd.DataFrame,
    prod_col: str,
    prod_label: str,
    covariates: List[str],
    time_col: str = "time",
    event_col: str = "event",
    rbd_col: str = "rbd_score_z",
) -> Optional[Tuple[MediationStepResult, List[MediationModelPerf]]]:
    """
    Fit all four Baron & Kenny steps for a single prodromal variable.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with rbd_score_z, prodromal variables, covariates.
    prod_col : str
        Prodromal predictor column.
    prod_label : str
        Human-readable label.
    covariates : list[str]
        Adjustment covariates.
    time_col, event_col : str
        Survival columns.
    rbd_col : str
        Continuous z-scored RBD mediator.

    Returns
    -------
    tuple[MediationStepResult, list[MediationModelPerf]] or None
        Point estimates and model performance, or None if any step fails.
    """
    # Complete-case across all variables
    all_cols = [time_col, event_col, prod_col, rbd_col] + covariates
    df_cc = df[[c for c in all_cols if c in df.columns]].dropna()
    n = len(df_cc)
    events = int(df_cc[event_col].sum())
    if n < 50 or events < 10:
        return None

    perf_rows: List[MediationModelPerf] = []

    # ── Step 1: c-path (total effect) ─────────────────────────────────
    cph_c = _fit_cox(df_cc, time_col, event_col, [prod_col] + covariates)
    if cph_c is None:
        return None
    c_coef = _extract_cox_coef(cph_c, prod_col)
    if c_coef is None:
        return None
    c_index_c = float(cph_c.concordance_index_)
    c_fit = extract_model_fit_metrics(cph_c, events)
    perf_rows.append(MediationModelPerf(
        variable=prod_col, label=prod_label, model="c_path_total",
        c_index=round(c_index_c, 4),
        aic=c_fit["AIC"], bic=c_fit["BIC"],
        log_likelihood=c_fit["log_likelihood"],
        lrt_stat=c_fit["LRT_stat"], lrt_p=c_fit["LRT_p"],
        n_params=c_fit["n_params"],
    ))

    # ── Step 2: a-path (prodromal -> RBD) ─────────────────────────────
    ols_a = _fit_ols(df_cc, rbd_col, [prod_col] + covariates)
    if ols_a is None:
        return None
    # prod_col is the first predictor (index 1 after constant)
    beta_a = float(ols_a.params[1])
    se_a = float(ols_a.bse[1])
    p_a = float(ols_a.pvalues[1])
    r2_a = float(ols_a.rsquared)

    # ── Step 3: joint model (c'-path + b-path) ────────────────────────
    cph_joint = _fit_cox(
        df_cc, time_col, event_col, [prod_col, rbd_col] + covariates,
    )
    if cph_joint is None:
        return None

    cprime_coef = _extract_cox_coef(cph_joint, prod_col)
    b_coef = _extract_cox_coef(cph_joint, rbd_col)
    if cprime_coef is None or b_coef is None:
        return None
    c_index_joint = float(cph_joint.concordance_index_)
    joint_fit = extract_model_fit_metrics(cph_joint, events)
    perf_rows.append(MediationModelPerf(
        variable=prod_col, label=prod_label, model="joint_cprime_b",
        c_index=round(c_index_joint, 4),
        aic=joint_fit["AIC"], bic=joint_fit["BIC"],
        log_likelihood=joint_fit["log_likelihood"],
        lrt_stat=joint_fit["LRT_stat"], lrt_p=joint_fit["LRT_p"],
        n_params=joint_fit["n_params"],
    ))

    # ── Step 4: indirect effect ───────────────────────────────────────
    beta_b = b_coef["beta"]
    beta_c = c_coef["beta"]
    beta_cprime = cprime_coef["beta"]
    beta_indirect = beta_a * beta_b
    hr_indirect = float(np.exp(beta_indirect))
    pm_pct = (beta_indirect / beta_c * 100.0) if abs(beta_c) > 1e-10 else np.nan
    inconsistent = bool(pd.notna(pm_pct) and (pm_pct < 0 or pm_pct > 100))

    step_result = MediationStepResult(
        variable=prod_col, label=prod_label,
        beta_c=round(beta_c, 6), hr_c=round(c_coef["hr"], 4),
        hr_c_lower=round(c_coef["hr_lower"], 4),
        hr_c_upper=round(c_coef["hr_upper"], 4),
        p_c=c_coef["p"], c_index_c=round(c_index_c, 4),
        beta_a=round(beta_a, 6), se_a=round(se_a, 6),
        p_a=p_a, r2_a=round(r2_a, 6),
        beta_b=round(beta_b, 6), hr_b=round(b_coef["hr"], 4),
        hr_b_lower=round(b_coef["hr_lower"], 4),
        hr_b_upper=round(b_coef["hr_upper"], 4),
        p_b=b_coef["p"],
        beta_cprime=round(beta_cprime, 6),
        hr_cprime=round(cprime_coef["hr"], 4),
        hr_cprime_lower=round(cprime_coef["hr_lower"], 4),
        hr_cprime_upper=round(cprime_coef["hr_upper"], 4),
        p_cprime=cprime_coef["p"],
        c_index_joint=round(c_index_joint, 4),
        beta_indirect=round(beta_indirect, 6),
        hr_indirect=round(hr_indirect, 4),
        pm_pct=round(pm_pct, 2) if pd.notna(pm_pct) else np.nan,
        inconsistent_mediation=inconsistent,
        n=n, events=events,
    )

    return step_result, perf_rows


def bootstrap_mediation(
    df: pd.DataFrame,
    prod_col: str,
    prod_label: str,
    covariates: List[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    time_col: str = "time",
    event_col: str = "event",
    rbd_col: str = "rbd_score_z",
    n_jobs: int = BOOTSTRAP_JOBS,
) -> Optional[MediationBootstrapResult]:
    """
    Bootstrap 95% CI for indirect effect and PM% via product of coefficients.

    Parallelised using joblib (loky backend). Numpy arrays are extracted
    once before dispatch to avoid DataFrame pickling overhead; sksurv's
    CoxPHSurvivalAnalysis is used for ~3-5x faster bootstrap Cox fits
    compared with lifelines. Progress is tracked via tqdm.

    Per resample:
      1. Draw n subjects with replacement (seed + b for reproducibility)
      2. Refit a-path OLS (numpy lstsq) -> beta_a
      3. Refit joint Cox (sksurv) -> beta_b
      4. Refit c-path Cox (sksurv) -> beta_c
      5. Compute indirect = beta_a * beta_b; PM = indirect / beta_c

    Report percentile [2.5th, 97.5th] for HR_indirect and PM%.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    prod_col, prod_label : str
        Prodromal variable and label.
    covariates : list[str]
        Adjustment covariates.
    n_bootstrap : int
        Number of resamples (default: 1000).
    seed : int
        Base random seed; iteration i uses seed + i.
    time_col, event_col : str
        Survival columns.
    rbd_col : str
        Continuous z-scored RBD mediator.
    n_jobs : int
        Number of parallel workers (default: BOOTSTRAP_JOBS from config).

    Returns
    -------
    MediationBootstrapResult or None
    """
    all_cols = [time_col, event_col, prod_col, rbd_col] + covariates
    df_cc = df[[c for c in all_cols if c in df.columns]].dropna()
    n = len(df_cc)
    if n < 50 or df_cc[event_col].sum() < 10:
        return None

    # ── Pre-extract numpy arrays once — avoids DataFrame pickling per worker ──
    event_arr = df_cc[event_col].to_numpy(dtype=float)
    time_arr = df_cc[time_col].to_numpy(dtype=float)
    y_surv = _make_sksurv_y(event_arr, time_arr)

    y_rbd = df_cc[rbd_col].to_numpy(dtype=float)
    prod_arr = df_cc[prod_col].to_numpy(dtype=float)
    cov_arr = df_cc[covariates].to_numpy(dtype=float)
    const_col = np.ones((n, 1))

    # X_a: [const, prod, *covariates] — OLS a-path (prod is index 1 after const)
    X_a = np.hstack([const_col, prod_arr[:, None], cov_arr])

    # X_joint: [prod, rbd, *covariates] — joint Cox (prod=0, rbd=1)
    X_joint = np.hstack([prod_arr[:, None], y_rbd[:, None], cov_arr])
    prod_idx_joint, rbd_idx_joint = 0, 1

    # X_c: [prod, *covariates] — c-path Cox (prod=0)
    X_c = np.hstack([prod_arr[:, None], cov_arr])
    prod_idx_c = 0

    # ── Parallel bootstrap with tqdm completion tracking ─────────────────────
    parallel = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=0,
        return_as="generator",
    )
    boot_gen = parallel(
        delayed(_one_mediation_boot)(
            seed + b,
            y_surv, X_a, y_rbd, X_joint, X_c,
            prod_idx_joint, rbd_idx_joint, prod_idx_c,
            BOOTSTRAP_RIDGE_PENALIZER,
        )
        for b in range(n_bootstrap)
    )
    raw_results: List[Optional[Tuple[float, float]]] = list(
        tqdm(boot_gen, total=n_bootstrap,
             desc=f"  Bootstrap [{prod_label[:35]}]",
             leave=False, ncols=90)
    )

    # ── Aggregate results ─────────────────────────────────────────────────────
    indirect_boot: List[float] = []
    pm_boot: List[float] = []
    n_discarded = sum(1 for r in raw_results if r is None)

    for r in raw_results:
        if r is None:
            continue
        indirect_b, pm_b = r
        indirect_boot.append(indirect_b)
        if pd.notna(pm_b):
            pm_boot.append(pm_b)

    n_converged = len(indirect_boot)
    if n_converged < max(10, n_bootstrap * 0.1):
        warnings.warn(
            f"Bootstrap for {prod_label}: only {n_converged}/{n_bootstrap} converged"
        )
        if n_converged < 10:
            return None

    if n_discarded > n_bootstrap * 0.1:
        warnings.warn(
            f"Bootstrap for {prod_label}: {n_discarded}/{n_bootstrap} "
            f"({n_discarded / n_bootstrap * 100:.1f}%) discarded"
        )

    indirect_arr = np.array(indirect_boot)
    hr_indirect_arr = np.exp(indirect_arr)
    hr_indirect_med = float(np.median(hr_indirect_arr))
    hr_lci = float(np.percentile(hr_indirect_arr, 2.5))
    hr_uci = float(np.percentile(hr_indirect_arr, 97.5))

    pm_arr = np.array(pm_boot) if pm_boot else np.array([np.nan])
    pm_med = float(np.median(pm_arr))
    pm_lci = float(np.percentile(pm_arr, 2.5)) if len(pm_boot) >= 10 else np.nan
    pm_uci = float(np.percentile(pm_arr, 97.5)) if len(pm_boot) >= 10 else np.nan

    return MediationBootstrapResult(
        variable=prod_col, label=prod_label,
        hr_indirect=round(hr_indirect_med, 4),
        hr_indirect_lci=round(hr_lci, 4),
        hr_indirect_uci=round(hr_uci, 4),
        pm_pct=round(pm_med, 2),
        pm_pct_lci=round(pm_lci, 2),
        pm_pct_uci=round(pm_uci, 2),
        n_converged=n_converged,
        n_bootstrap=n_bootstrap,
        n_discarded=n_discarded,
    )


def fit_supplementary_3g_bpath(
    df: pd.DataFrame,
    prod_col: str,
    prod_label: str,
    covariates: List[str],
    time_col: str = "time",
    event_col: str = "event",
    rbd_3g_col: str = "rbd_group_3g",
) -> Optional[Supplementary3gResult]:
    """
    Supplementary: categorical 3-group b-path Cox model.

    Cox: outcome ~ P + rbd_group_3g_dummies + covariates

    Not used for product method. Reports HR for Intermediate vs Low and
    High vs Low (RBD categorical terms) and attenuation of P coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with rbd_group_3g column.
    prod_col, prod_label : str
        Prodromal variable and label.
    covariates : list[str]
        Adjustment covariates.

    Returns
    -------
    Supplementary3gResult or None
    """
    all_cols = [time_col, event_col, prod_col, rbd_3g_col] + covariates
    df_cc = df[[c for c in all_cols if c in df.columns]].dropna()
    if len(df_cc) < 50 or df_cc[event_col].sum() < 10:
        return None

    # Dummy-encode 3g with Low as reference
    dummies = pd.get_dummies(df_cc[rbd_3g_col], prefix="rbd3g", drop_first=False)
    ref_col = "rbd3g_Low"
    dummy_cols = [c for c in dummies.columns if c != ref_col]
    if len(dummy_cols) < 2:
        return None

    df_fit = df_cc[[time_col, event_col, prod_col] + covariates].copy()
    for dc in dummy_cols:
        df_fit[dc] = dummies[dc].values

    cph = _fit_cox(df_fit, time_col, event_col, [prod_col] + dummy_cols + covariates)
    if cph is None:
        return None

    # Extract RBD group HRs
    summary = cph.summary
    int_col = [c for c in dummy_cols if "intermediate" in c.lower() or "Intermediate" in c]
    high_col = [c for c in dummy_cols if "high" in c.lower() or "High" in c]

    if not int_col or not high_col:
        return None

    int_name = int_col[0]
    high_name = high_col[0]

    int_coef = _extract_cox_coef(cph, int_name)
    high_coef = _extract_cox_coef(cph, high_name)
    cprime_coef = _extract_cox_coef(cph, prod_col)
    if int_coef is None or high_coef is None or cprime_coef is None:
        return None

    return Supplementary3gResult(
        variable=prod_col, label=prod_label,
        hr_intermediate_vs_low=round(int_coef["hr"], 4),
        hr_intermediate_lci=round(int_coef["hr_lower"], 4),
        hr_intermediate_uci=round(int_coef["hr_upper"], 4),
        p_intermediate=int_coef["p"],
        hr_high_vs_low=round(high_coef["hr"], 4),
        hr_high_lci=round(high_coef["hr_lower"], 4),
        hr_high_uci=round(high_coef["hr_upper"], 4),
        p_high=high_coef["p"],
        beta_cprime_3g=round(cprime_coef["beta"], 6),
        hr_cprime_3g=round(cprime_coef["hr"], 4),
        n=len(df_cc),
        events=int(df_cc[event_col].sum()),
    )


def run_mediation_analysis(
    df: pd.DataFrame,
    active_vars: Dict[str, str],
    covariates: List[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    n_jobs: int = BOOTSTRAP_JOBS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run full Baron & Kenny mediation for all active prodromal variables.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with z-scored prodromals and RBD encodings.
    active_vars : dict
        {column_name: label}.
    covariates : list[str]
        Adjustment covariates.
    n_bootstrap : int
        Bootstrap resamples.
    seed : int
        Random seed.
    n_jobs : int
        Parallel workers for bootstrap (default: BOOTSTRAP_JOBS from config).

    Returns
    -------
    tuple of 5 DataFrames
        (steps_df, indirect_df, perf_df, supplementary_3g_df, feasibility_df)
    """
    step_rows: List[dict] = []
    boot_rows: List[dict] = []
    perf_rows: List[dict] = []
    supp_rows: List[dict] = []
    feas_rows: List[dict] = []

    for i, (prod_col, prod_label) in enumerate(active_vars.items()):
        print(f"\n  [{i + 1}/{len(active_vars)}] Mediation: {prod_label}")

        # Feasibility check
        all_cols = ["time", "event", prod_col, "rbd_score_z"] + covariates
        df_cc = df[[c for c in all_cols if c in df.columns]].dropna()
        n_cc = len(df_cc)
        events_cc = int(df_cc["event"].sum()) if "event" in df_cc.columns else 0
        feas_rows.append({
            "variable": prod_col, "label": prod_label,
            "n_complete_case": n_cc, "events": events_cc,
            "feasible": n_cc >= 50 and events_cc >= 10,
        })

        if n_cc < 50 or events_cc < 10:
            print(f"    SKIP: N={n_cc}, events={events_cc}")
            continue

        # Steps 1-4: point estimates
        result = fit_mediation_steps(df, prod_col, prod_label, covariates)
        if result is None:
            print(f"    Point estimates failed")
            continue

        step_result, model_perfs = result
        step_rows.append(step_result.__dict__)
        for mp in model_perfs:
            perf_rows.append(mp.__dict__)

        print(f"    c-path HR={step_result.hr_c:.3f} (p={step_result.p_c:.4f}), "
              f"a-path beta={step_result.beta_a:.4f} (p={step_result.p_a:.4f}), "
              f"PM={step_result.pm_pct:.1f}%"
              f"{'  [INCONSISTENT]' if step_result.inconsistent_mediation else ''}")

        # Step 5: bootstrap CI
        var_seed = seed + i * 100
        boot_result = bootstrap_mediation(
            df, prod_col, prod_label, covariates,
            n_bootstrap=n_bootstrap, seed=var_seed, n_jobs=n_jobs,
        )
        if boot_result is not None:
            boot_rows.append(boot_result.__dict__)
            print(f"    Bootstrap HR_indirect={boot_result.hr_indirect:.3f} "
                  f"[{boot_result.hr_indirect_lci:.3f}, {boot_result.hr_indirect_uci:.3f}], "
                  f"PM={boot_result.pm_pct:.1f}% "
                  f"[{boot_result.pm_pct_lci:.1f}, {boot_result.pm_pct_uci:.1f}], "
                  f"converged={boot_result.n_converged}/{boot_result.n_bootstrap}")

        # Step 5.5: supplementary 3g b-path
        supp_result = fit_supplementary_3g_bpath(df, prod_col, prod_label, covariates)
        if supp_result is not None:
            supp_rows.append(supp_result.__dict__)

    # Build DataFrames
    df_steps = pd.DataFrame(step_rows)
    df_boot = pd.DataFrame(boot_rows)
    df_perf = pd.DataFrame(perf_rows)
    df_supp = pd.DataFrame(supp_rows)
    df_feas = pd.DataFrame(feas_rows)

    # FDR on a-path and c-path p-values
    if not df_steps.empty:
        df_steps["p_a_fdr"] = apply_fdr(df_steps["p_a"]).values
        df_steps["p_c_fdr"] = apply_fdr(df_steps["p_c"]).values

    return df_steps, df_boot, df_perf, df_supp, df_feas
