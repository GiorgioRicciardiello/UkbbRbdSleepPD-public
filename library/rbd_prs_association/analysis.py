"""
Statistical analyses for the RBD–PRS biological strength study.

Analysis chain (run for full cohort AND High-risk subgroup)
────────────────────────────────────────────────────────────
1. Spearman correlation with permutation-based p-value and 95 % CI (Fisher Z).
2. OLS partial regression: abk_rbd_score_mean ~ prs + covariates.
   Reports standardised β, partial R², residual diagnostics.
3. GAM (pygam LinearGAM): abk_rbd_score_mean ~ s(prs) + linear(covariates).
   Reports effective df per spline, pseudo-R², non-linearity test vs. OLS.

Assumptions
───────────
- RBD score and PRS scores are continuous.  Spearman is preferred over
  Pearson because the RBD probability is bounded [0, 1] and may be
  right-skewed in the general population.
- PRS scores are z-standardised (mean=0, SD=1) at source; no re-scaling.
- Ancestry PCs are included in all regression models as fixed linear terms
  to absorb population stratification — a bias that inflates PRS associations
  in ethnically heterogeneous samples (Price et al. 2006).
- The non-linearity test compares OLS vs. GAM deviance (F-test).  If the
  GAM edf per smooth ≈ 1.0 the relationship is effectively linear.

Reproducibility: RANDOM_SEED = 42 for all permutations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from pygam import LinearGAM, s, l, te
from scipy.stats import shapiro, pearsonr
from statsmodels.stats.diagnostic import het_breuschpagan

from library.rbd_prs_association.config import (
    ADJUSTMENT_COVARIATES,
    GAM_LAMBDA_GRID,
    GAM_MAX_ITER,
    GAM_N_SPLINES,
    HIGH_RISK_LABEL,
    PERMUTATION_N,
    PRS_PD_COL,
    PRS_RBD_COL,
    RANDOM_SEED,
    RBD_SCORE_COL,
    RISK_GROUP_COL,
    RG_ORDER,
    RG_SHORT,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class SpearmanResult:
    """Spearman correlation result with CI and permutation p-value."""
    prs_col: str
    stratum: str          # "full" | risk group label
    n: int
    rho: float
    ci_lower: float
    ci_upper: float
    p_value: float        # analytical
    p_permutation: float  # permutation-based


@dataclass
class OLSResult:
    """OLS regression result for one PRS predictor."""
    prs_col: str
    stratum: str
    n: int
    beta_std: float          # standardised coefficient
    beta_raw: float          # unstandardised coefficient
    se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    partial_r2: float
    model_r2: float          # full model R²
    adj_r2: float
    residual_normality_p: float   # Shapiro-Wilk on residuals (n ≤ 5000 subsample)
    breusch_pagan_p: float        # heteroscedasticity test p-value


@dataclass
class GAMResult:
    """GAM result for one PRS predictor spline term."""
    prs_col: str
    stratum: str
    n: int
    edf: float               # effective degrees of freedom of spline
    pseudo_r2: float
    ols_r2: float            # R² of equivalent linear model (for comparison)
    nonlinearity_f: float    # F-statistic: Δdeviance / Δdf between GAM and OLS
    nonlinearity_p: float    # approximate p-value from F distribution
    gam_deviance: float
    ols_deviance: float
    best_lambda: float       # selected regularisation parameter


@dataclass
class AnalysisResults:
    """Container for all analysis outputs."""
    spearman: List[SpearmanResult] = field(default_factory=list)
    ols: List[OLSResult] = field(default_factory=list)
    gam: List[GAMResult] = field(default_factory=list)


# ── Spearman correlation ──────────────────────────────────────────────────────

def _fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """95 % CI for Spearman ρ via Fisher Z transformation.

    Rationale: Fisher Z transforms ρ to an approximately normal statistic
    with SE = 1/sqrt(n-3), enabling symmetric CI construction without
    bootstrap.  Valid for |ρ| < 1 and n > 3.
    """
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def _permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    observed_rho: float,
    n_permutations: int,
    seed: int,
) -> float:
    """Two-tailed permutation p-value for Spearman ρ.

    Permutes y (the PRS vector) while holding x fixed.  The null distribution
    is empirical, making no distributional assumptions.
    """
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(y_perm)
        rho_perm, _ = stats.spearmanr(x, y_perm)
        if abs(rho_perm) >= abs(observed_rho):
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)  # add 1 for continuity


def compute_spearman(
    df: pd.DataFrame,
    prs_col: str,
    stratum: str,
) -> SpearmanResult:
    """Compute Spearman ρ between RBD score and one PRS column.

    Parameters
    ----------
    df : pd.DataFrame
        Subset for this stratum (already filtered).
    prs_col : str
        Column name of the PRS predictor.
    stratum : str
        Label for this analysis stratum.
    """
    valid = df[[RBD_SCORE_COL, prs_col]].dropna()
    n = len(valid)
    x = valid[RBD_SCORE_COL].to_numpy(dtype=float)
    y = valid[prs_col].to_numpy(dtype=float)

    rho, p_val = stats.spearmanr(x, y)
    ci_lo, ci_hi = _fisher_z_ci(rho, n)
    p_perm = _permutation_p(x, y, rho, PERMUTATION_N, RANDOM_SEED)

    return SpearmanResult(
        prs_col=prs_col,
        stratum=stratum,
        n=n,
        rho=float(rho),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value=float(p_val),
        p_permutation=p_perm,
    )


# ── OLS partial regression ────────────────────────────────────────────────────

def _partial_r2(model_full: sm.regression.linear_model.RegressionResultsWrapper,
                df_model: pd.DataFrame,
                prs_col: str,
                covariates: List[str]) -> float:
    """Compute partial R² for prs_col via the semi-partial approach.

    partial R² = (SS_res_reduced - SS_res_full) / SS_res_reduced
    where the reduced model excludes prs_col.
    """
    reduced_cols = [c for c in covariates if c != prs_col]
    X_reduced = sm.add_constant(df_model[reduced_cols].to_numpy(dtype=float))
    y = df_model[RBD_SCORE_COL].to_numpy(dtype=float)
    model_reduced = sm.OLS(y, X_reduced).fit()
    ss_res_full = model_full.ssr
    ss_res_reduced = model_reduced.ssr
    if ss_res_reduced == 0:
        return 0.0
    return float((ss_res_reduced - ss_res_full) / ss_res_reduced)


def compute_ols(
    df: pd.DataFrame,
    prs_col: str,
    stratum: str,
    active_covariates: List[str],
) -> OLSResult:
    """OLS regression of RBD score on PRS + adjustment covariates.

    Model: abk_rbd_score_mean = α + β_prs * prs + β_cov * covariates + ε

    Standardised β is computed by z-scoring both outcome and predictor
    prior to fitting (on the analysis subset), so it represents the SD
    change in RBD score per 1-SD change in PRS.
    """
    all_cols = [RBD_SCORE_COL, prs_col] + active_covariates
    available = [c for c in all_cols if c in df.columns]
    valid = df[available].dropna()
    n = len(valid)

    y_raw = valid[RBD_SCORE_COL].to_numpy(dtype=float)
    x_raw = valid[prs_col].to_numpy(dtype=float)
    cov_cols = [c for c in active_covariates if c in valid.columns]

    # Build design matrix (raw scale for beta_raw)
    X_raw = np.column_stack([x_raw] + [valid[c].to_numpy(dtype=float) for c in cov_cols])
    X_raw_const = sm.add_constant(X_raw)
    model_raw = sm.OLS(y_raw, X_raw_const).fit()

    beta_raw = float(model_raw.params[1])
    se_raw = float(model_raw.bse[1])
    t_stat = float(model_raw.tvalues[1])
    p_val = float(model_raw.pvalues[1])
    ci_lo = float(model_raw.conf_int()[0][1])
    ci_hi = float(model_raw.conf_int()[1][1])

    # Standardised β: z-score y and prs_col only (covariates remain on original scale)
    y_std = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-12)
    x_std = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-12)
    X_std = np.column_stack([x_std] + [valid[c].to_numpy(dtype=float) for c in cov_cols])
    model_std = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    beta_std = float(model_std.params[1])

    # Partial R²
    df_model = valid[[RBD_SCORE_COL, prs_col] + cov_cols].copy()
    partial_r2 = _partial_r2(model_raw, df_model, prs_col, [prs_col] + cov_cols)

    # Residual diagnostics
    residuals = model_raw.resid
    # Shapiro-Wilk on subsample (max 5000) — test breaks for very large n
    subsample = residuals if len(residuals) <= 5000 else residuals[
        np.random.default_rng(RANDOM_SEED).choice(len(residuals), 5000, replace=False)
    ]
    _, sw_p = shapiro(subsample)

    # Breusch-Pagan heteroscedasticity
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model_raw.model.exog)

    return OLSResult(
        prs_col=prs_col,
        stratum=stratum,
        n=n,
        beta_std=beta_std,
        beta_raw=beta_raw,
        se=se_raw,
        t_stat=t_stat,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        partial_r2=partial_r2,
        model_r2=float(model_raw.rsquared),
        adj_r2=float(model_raw.rsquared_adj),
        residual_normality_p=float(sw_p),
        breusch_pagan_p=float(bp_p),
    )


# ── GAM non-linearity analysis ────────────────────────────────────────────────

def _build_gam_X(
    valid: pd.DataFrame,
    prs_col: str,
    cov_cols: List[str],
) -> np.ndarray:
    """Stack PRS column first, then covariates, for pygam term indexing."""
    parts = [valid[prs_col].to_numpy(dtype=float).reshape(-1, 1)]
    for c in cov_cols:
        parts.append(valid[c].to_numpy(dtype=float).reshape(-1, 1))
    return np.hstack(parts)


def compute_gam(
    df: pd.DataFrame,
    prs_col: str,
    stratum: str,
    active_covariates: List[str],
) -> GAMResult:
    """Fit a GAM with a spline on prs_col and linear terms for covariates.

    Model: E[abk_rbd_score_mean] = f(prs) + β_cov * covariates
    where f(.) is a smoothing spline with n_splines basis functions.

    Non-linearity test (F-test)
    ────────────────────────────
    H0: the GAM smooth reduces to a linear term (edf ≈ 1).
    F = (Δdeviance / Δdf) / (GAM_deviance / (n - GAM_df))
    Δdeviance = OLS_deviance - GAM_deviance
    Δdf = GAM_edf - 1  (extra df used by the spline vs. linear)
    Under H0 this is approximately F(Δdf, n - GAM_df).

    Lambda selection via GCV (pygam default).  The grid GAM_LAMBDA_GRID is
    passed to gridsearch() to ensure a reproducible candidate set.
    """
    all_cols = [RBD_SCORE_COL, prs_col] + active_covariates
    available = [c for c in all_cols if c in df.columns]
    valid = df[available].dropna()
    n = len(valid)
    y = valid[RBD_SCORE_COL].to_numpy(dtype=float)
    cov_cols = [c for c in active_covariates if c in valid.columns]

    X = _build_gam_X(valid, prs_col, cov_cols)
    n_covs = len(cov_cols)

    # GAM: spline on index 0 (prs), linear on remaining indices
    terms = s(0, n_splines=GAM_N_SPLINES, spline_order=3)
    for i in range(1, 1 + n_covs):
        terms = terms + l(i)

    gam = LinearGAM(terms, max_iter=GAM_MAX_ITER)
    # pygam gridsearch expects lam[term_index] = [candidate_values].
    # Term 0 (PRS spline) is searched over GAM_LAMBDA_GRID;
    # linear covariate terms are fixed at 0.6 (no shrinkage needed).
    lams = [GAM_LAMBDA_GRID] + [[0.6]] * n_covs
    gam.gridsearch(X, y, lam=lams, progress=False)

    # Effective degrees of freedom of the spline term
    edf = float(gam.statistics_["edof_per_coef"][0]) if "edof_per_coef" in gam.statistics_ else float(gam.statistics_["edof"])

    gam_deviance = float(gam.statistics_["deviance"])
    pseudo_r2 = float(gam.statistics_["pseudo_r2"]["explained_deviance"])
    best_lambda = float(gam.lam[0][0]) if hasattr(gam, "lam") else float("nan")

    # OLS equivalent (linear term for prs instead of spline)
    X_ols_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_ols_const).fit()
    ols_deviance = float(ols_model.ssr)
    ols_r2 = float(ols_model.rsquared)

    # Non-linearity F-test
    delta_dev = ols_deviance - gam_deviance
    delta_df = max(edf - 1.0, 1e-6)
    gam_dof = max(n - edf - n_covs - 1, 1)
    f_stat = (delta_dev / delta_df) / (gam_deviance / gam_dof)
    p_nonlin = float(1.0 - stats.f.cdf(f_stat, delta_df, gam_dof))

    return GAMResult(
        prs_col=prs_col,
        stratum=stratum,
        n=n,
        edf=edf,
        pseudo_r2=pseudo_r2,
        ols_r2=ols_r2,
        nonlinearity_f=float(f_stat),
        nonlinearity_p=p_nonlin,
        gam_deviance=gam_deviance,
        ols_deviance=ols_deviance,
        best_lambda=best_lambda,
    )


# ── Descriptive summary ───────────────────────────────────────────────────────

def compute_descriptives(
    df: pd.DataFrame,
    active_covariates: List[str],
) -> pd.DataFrame:
    """Compute mean ± SD for key variables stratified by risk group and case/control.

    Returns a tidy DataFrame suitable for Table 1.
    """
    key_cols = [RBD_SCORE_COL, PRS_PD_COL, PRS_RBD_COL] + [
        c for c in active_covariates[:3] if c in df.columns  # age, sex, BMI
    ]

    rows = []
    # Overall
    for col in key_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "variable": col,
            "stratum": "Overall",
            "n": len(vals),
            "mean": vals.mean(),
            "sd": vals.std(),
            "median": vals.median(),
            "q25": vals.quantile(0.25),
            "q75": vals.quantile(0.75),
        })

    # By risk group
    for rg in RG_ORDER:
        sub = df.loc[df[RISK_GROUP_COL] == rg]
        label = RG_SHORT.get(rg, rg)
        for col in key_cols:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            rows.append({
                "variable": col,
                "stratum": f"RG={label}",
                "n": len(vals),
                "mean": vals.mean(),
                "sd": vals.std(),
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
            })

    # By case/control (incident PD outcome)
    if "case_control" in df.columns:
        for cc in ["case", "control"]:
            sub = df.loc[df["case_control"] == cc]
            for col in key_cols:
                if col not in df.columns:
                    continue
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                rows.append({
                    "variable": col,
                    "stratum": f"CC={cc}",
                    "n": len(vals),
                    "mean": vals.mean(),
                    "sd": vals.std(),
                    "median": vals.median(),
                    "q25": vals.quantile(0.25),
                    "q75": vals.quantile(0.75),
                })

    return pd.DataFrame(rows)


# ── Main analysis orchestrator ────────────────────────────────────────────────

def run_all_analyses(
    df: pd.DataFrame,
    active_covariates: List[str],
) -> AnalysisResults:
    """Run the full statistical chain on the full cohort and High-risk subgroup.

    For each stratum × PRS combination:
    1. Spearman ρ + permutation p
    2. OLS partial regression
    3. GAM non-linearity test

    Parameters
    ----------
    df : pd.DataFrame
        Analytical dataset (one row per subject, all required columns present).
    active_covariates : list[str]
        Covariates to include in regression models.

    Returns
    -------
    AnalysisResults
        Container with lists of SpearmanResult, OLSResult, GAMResult.
    """
    results = AnalysisResults()

    strata: Dict[str, pd.DataFrame] = {
        "Full cohort": df,
        f"High-risk ({HIGH_RISK_LABEL})": df.loc[df[RISK_GROUP_COL] == HIGH_RISK_LABEL],
    }
    # Also run per risk group for stratified Spearman
    for rg in RG_ORDER:
        strata[RG_SHORT.get(rg, rg)] = df.loc[df[RISK_GROUP_COL] == rg]

    prs_cols = [PRS_PD_COL, PRS_RBD_COL]

    for stratum_label, sub in strata.items():
        n = len(sub)
        logger.info("Stratum '%s': N=%d", stratum_label, n)
        if n < 30:
            logger.warning("Stratum '%s' too small (N=%d), skipping.", stratum_label, n)
            continue

        for prs_col in prs_cols:
            if prs_col not in sub.columns:
                logger.warning("Column %s missing in stratum '%s'", prs_col, stratum_label)
                continue

            logger.info("  Spearman: %s × %s", RBD_SCORE_COL, prs_col)
            results.spearman.append(compute_spearman(sub, prs_col, stratum_label))

            # OLS and GAM only for full cohort and high-risk (not per-RG to limit runtime)
            if stratum_label in ("Full cohort", f"High-risk ({HIGH_RISK_LABEL})"):
                logger.info("  OLS: %s", prs_col)
                results.ols.append(compute_ols(sub, prs_col, stratum_label, active_covariates))

                logger.info("  GAM: %s", prs_col)
                results.gam.append(compute_gam(sub, prs_col, stratum_label, active_covariates))

    return results
