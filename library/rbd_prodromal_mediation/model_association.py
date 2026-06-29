"""
Models 1a, 1b, 1b-3g: Association between prodromal markers and RBD.

Model 1a  — OLS: rbd_score_z ~ P_z + age + sex + BMI
Model 1b  — Logistic: rbd_high_binary ~ P_z + age + sex + BMI
Model 1b-3g — Multinomial logit: rbd_group_3g ~ P_z + age + sex + BMI
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from library.cox_prodromal.diagnostics import apply_fdr


@dataclass(frozen=True)
class OLSResult:
    """Results from Model 1a (OLS)."""
    variable: str
    label: str
    beta: float
    se: float
    ci_lower: float
    ci_upper: float
    p: float
    partial_r2: float
    model_r2: float
    model_r2_adj: float
    f_stat: float
    rmse: float
    n: int


@dataclass(frozen=True)
class LogisticResult:
    """Results from Model 1b (Logistic)."""
    variable: str
    label: str
    or_val: float
    or_lower: float
    or_upper: float
    p: float
    pseudo_r2_mcfadden: float
    pseudo_r2_nagelkerke: float
    auc_roc: float
    lrt_stat: float
    lrt_p: float
    n: int
    n_high: int


@dataclass(frozen=True)
class MultinomialResult:
    """Results from Model 1b-3g (Multinomial logit)."""
    variable: str
    label: str
    contrast: str
    or_val: float
    or_lower: float
    or_upper: float
    p: float
    pseudo_r2: float
    lrt_stat: float
    lrt_p: float
    n: int


def fit_ols_association(
    df: pd.DataFrame,
    prod_col: str,
    prod_label: str,
    covariates: List[str],
    rbd_col: str = "rbd_score_z",
) -> Optional[OLSResult]:
    """
    Model 1a: OLS regression of RBD z-score on a single prodromal marker.

    rbd_score_z ~ P + age + sex + BMI

    Uses HC3 heteroskedasticity-robust standard errors.

    Parameters
    ----------
    df : pd.DataFrame
        Analytic cohort with z-scored variables.
    prod_col : str
        Prodromal predictor column.
    prod_label : str
        Human-readable label.
    covariates : list[str]
        Adjustment covariates (age, sex, BMI, ...).
    rbd_col : str
        RBD outcome column (default: rbd_score_z).

    Returns
    -------
    OLSResult or None
        Regression results, or None if model fails.
    """
    cols = [rbd_col, prod_col] + covariates
    df_cc = df[cols].dropna()
    if len(df_cc) < 30:
        print(f"  Skipping {prod_label} (too few data points)")
        return None

    y = df_cc[rbd_col].to_numpy(dtype=float)
    X = sm.add_constant(
        df_cc[[prod_col] + covariates].to_numpy(dtype=float)
    )
    feature_names = ["const", prod_col] + covariates

    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception as exc:
        warnings.warn(f"OLS failed for {prod_label}: {exc}")
        return None

    idx = feature_names.index(prod_col)
    beta = float(model.params[idx])
    se = float(model.bse[idx])
    ci = model.conf_int(alpha=0.05)
    ci_lower = float(ci[idx, 0])
    ci_upper = float(ci[idx, 1])
    p_val = float(model.pvalues[idx])

    # Partial R^2: SSR_reduced - SSR_full / SSR_reduced
    X_reduced = np.delete(X, idx, axis=1)
    try:
        model_reduced = sm.OLS(y, X_reduced).fit()
        ssr_reduced = float(model_reduced.ssr)
        ssr_full = float(model.ssr)
        partial_r2 = (ssr_reduced - ssr_full) / ssr_reduced
    except Exception:
        partial_r2 = np.nan

    residuals = model.resid
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return OLSResult(
        variable=prod_col, label=prod_label,
        beta=round(beta, 6), se=round(se, 6),
        ci_lower=round(ci_lower, 6), ci_upper=round(ci_upper, 6),
        p=p_val, partial_r2=round(partial_r2, 6),
        model_r2=round(float(model.rsquared), 6),
        model_r2_adj=round(float(model.rsquared_adj), 6),
        f_stat=round(float(model.fvalue), 4),
        rmse=round(rmse, 6),
        n=len(df_cc),
    )


def fit_logistic_association(
    df: pd.DataFrame,
    prod_col: str,
    prod_label: str,
    covariates: List[str],
    rbd_col: str = "rbd_high_binary",
) -> Optional[LogisticResult]:
    """
    Model 1b: Logistic regression of binary high-RBD on prodromal marker.

    rbd_high_binary ~ P + age + sex + BMI

    Parameters
    ----------
    df : pd.DataFrame
        Analytic cohort.
    prod_col : str
        Prodromal predictor column.
    prod_label : str
        Human-readable label.
    covariates : list[str]
        Adjustment covariates.
    rbd_col : str
        Binary RBD outcome column.

    Returns
    -------
    LogisticResult or None
    """
    cols = [rbd_col, prod_col] + covariates
    df_cc = df[cols].dropna()
    n_high = int(df_cc[rbd_col].sum())
    if n_high < 10 or len(df_cc) < 30:
        return None

    y = df_cc[rbd_col].to_numpy(dtype=float)
    X = sm.add_constant(
        df_cc[[prod_col] + covariates].to_numpy(dtype=float)
    )
    feature_names = ["const", prod_col] + covariates

    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception as exc:
        warnings.warn(f"Logistic failed for {prod_label}: {exc}")
        return None

    idx = feature_names.index(prod_col)
    beta = float(model.params[idx])
    se = float(model.bse[idx])
    ci = model.conf_int(alpha=0.05)
    or_val = float(np.exp(beta))
    or_lower = float(np.exp(ci[idx, 0]))
    or_upper = float(np.exp(ci[idx, 1]))
    p_val = float(model.pvalues[idx])

    # McFadden pseudo-R2
    pseudo_r2_mcf = float(model.prsquared)

    # Nagelkerke pseudo-R2
    null_model = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0)
    ll_null = float(null_model.llf)
    ll_full = float(model.llf)
    n = len(y)
    cox_snell = 1.0 - np.exp(-2.0 / n * (ll_full - ll_null))
    cox_snell_max = 1.0 - np.exp(2.0 / n * ll_null)
    nagelkerke = cox_snell / cox_snell_max if cox_snell_max > 0 else np.nan

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score
        pred_prob = model.predict(X)
        auc = float(roc_auc_score(y, pred_prob))
    except Exception:
        auc = np.nan

    # LRT vs null
    lrt_stat = float(-2.0 * (ll_null - ll_full))
    from scipy.stats import chi2
    lrt_df = len(model.params) - 1
    lrt_p = float(chi2.sf(lrt_stat, lrt_df))

    return LogisticResult(
        variable=prod_col, label=prod_label,
        or_val=round(or_val, 4), or_lower=round(or_lower, 4),
        or_upper=round(or_upper, 4),
        p=p_val,
        pseudo_r2_mcfadden=round(pseudo_r2_mcf, 6),
        pseudo_r2_nagelkerke=round(nagelkerke, 6),
        auc_roc=round(auc, 4),
        lrt_stat=round(lrt_stat, 3), lrt_p=lrt_p,
        n=n, n_high=n_high,
    )


def fit_multinomial_association(
    df: pd.DataFrame,
    prod_col: str,
    prod_label: str,
    covariates: List[str],
    rbd_col: str = "rbd_group_3g",
) -> Optional[List[MultinomialResult]]:
    """
    Model 1b-3g: Multinomial logit of 3-group RBD on prodromal marker.

    rbd_group_3g (Low / Intermediate / High) ~ P + age + sex + BMI
    Reference = Low group.

    Parameters
    ----------
    df : pd.DataFrame
        Analytic cohort.
    prod_col : str
        Prodromal predictor column.
    prod_label : str
        Human-readable label.
    covariates : list[str]
        Adjustment covariates.
    rbd_col : str
        3-group RBD categorical column.

    Returns
    -------
    list[MultinomialResult] or None
        Two results (Intermediate vs Low, High vs Low), or None if fails.
    """
    cols = [rbd_col, prod_col] + covariates
    df_cc = df[cols].dropna()
    if len(df_cc) < 50:
        return None

    # Encode outcome: Low=0, Intermediate=1, High=2
    cat_map = {"Low": 0, "Intermediate": 1, "High": 2}
    y_raw = df_cc[rbd_col].map(cat_map)
    if y_raw.isna().any():
        df_cc = df_cc[y_raw.notna()]
        y_raw = y_raw.dropna()
    y = y_raw.astype(int).values

    X = sm.add_constant(
        df_cc[[prod_col] + covariates].to_numpy(dtype=float)
    )
    feature_names = ["const", prod_col] + covariates

    try:
        model = sm.MNLogit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:
        warnings.warn(f"MNLogit failed for {prod_label}: {exc}")
        return None

    idx = feature_names.index(prod_col)

    # LRT vs null
    null_model = sm.MNLogit(y, np.ones((len(y), 1))).fit(disp=0, maxiter=200)
    lrt_stat = float(-2.0 * (null_model.llf - model.llf))
    from scipy.stats import chi2
    lrt_df = model.df_model
    lrt_p = float(chi2.sf(lrt_stat, lrt_df))

    pseudo_r2 = float(model.prsquared)

    results = []
    contrast_names = ["Intermediate vs Low", "High vs Low"]
    # MNLogit params shape: (n_features, n_categories-1)
    # Column 0 = category 1 (Intermediate), column 1 = category 2 (High)
    for j, contrast in enumerate(contrast_names):
        beta = float(model.params[idx, j])
        se = float(model.bse[idx, j])
        # Use Wald CI from beta ± 1.96 * se — avoids MNLogit conf_int indexing complexity
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se

        or_val = float(np.exp(beta))
        or_lower = float(np.exp(ci_lower))
        or_upper = float(np.exp(ci_upper))
        p_val = float(model.pvalues[idx, j])

        results.append(MultinomialResult(
            variable=prod_col, label=prod_label,
            contrast=contrast,
            or_val=round(or_val, 4),
            or_lower=round(or_lower, 4),
            or_upper=round(or_upper, 4),
            p=p_val,
            pseudo_r2=round(pseudo_r2, 6),
            lrt_stat=round(lrt_stat, 3), lrt_p=lrt_p,
            n=len(df_cc),
        ))

    return results


def run_association_models(
    df: pd.DataFrame,
    active_vars: Dict[str, str],
    covariates: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all three association models for every active prodromal variable.

    Parameters
    ----------
    df : pd.DataFrame
        Analytic cohort with z-scored variables and RBD encodings.
    active_vars : dict
        {column_name: label} for prodromal variables to test.
    covariates : list[str]
        Adjustment covariates.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_ols, df_logistic, df_multinomial) with FDR-corrected p-values.
    """
    ols_rows: List[dict] = []
    logistic_rows: List[dict] = []
    multinomial_rows: List[dict] = []

    for prod_col, prod_label in active_vars.items():
        print(f"  Association models: {prod_label}")

        # Model 1a: OLS
        ols_res = fit_ols_association(df, prod_col, prod_label, covariates)
        if ols_res is not None:
            ols_rows.append(ols_res.__dict__)

        # Model 1b: Logistic
        logit_res = fit_logistic_association(df, prod_col, prod_label, covariates)
        if logit_res is not None:
            logistic_rows.append(logit_res.__dict__)

        # Model 1b-3g: Multinomial
        mn_res = fit_multinomial_association(df, prod_col, prod_label, covariates)
        if mn_res is not None:
            for r in mn_res:
                multinomial_rows.append(r.__dict__)

    # Build DataFrames and apply FDR
    df_ols = pd.DataFrame(ols_rows)
    if not df_ols.empty and "p" in df_ols.columns:
        df_ols["p_fdr"] = apply_fdr(df_ols["p"]).values

    df_logistic = pd.DataFrame(logistic_rows)
    if not df_logistic.empty and "p" in df_logistic.columns:
        df_logistic["p_fdr"] = apply_fdr(df_logistic["p"]).values

    df_mn = pd.DataFrame(multinomial_rows)
    if not df_mn.empty and "p" in df_mn.columns:
        df_mn["p_fdr"] = apply_fdr(df_mn["p"]).values

    return df_ols, df_logistic, df_mn
