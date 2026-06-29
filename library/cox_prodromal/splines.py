"""
Spline Cox models for continuous prodromal markers and RBD probability.

Uses natural cubic splines (patsy cr()) with likelihood-ratio test
for non-linearity assessment.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from library.cox_prodromal.cox_config import (
    MIN_EVENTS_FOR_MODEL,
    RIDGE_PENALIZER,
    SPLINE_DF,
)

try:
    from patsy import dmatrix as patsy_dmatrix
    PATSY_AVAILABLE = True
except ImportError:
    PATSY_AVAILABLE = False

try:
    import scipy.stats as _scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def fit_spline_cox_prodromal(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    prod_col: str,
    covariates: List[str],
    n_df: int = SPLINE_DF,
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Spline Cox for a continuous prodromal marker.

    Fits a natural cubic spline basis (patsy cr()) and compares to a
    linear-term model via likelihood-ratio test (LRT).

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    prod_col : str
        Continuous prodromal variable.
    n_df : int
        Degrees of freedom for the spline basis (default 4).
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: c_index_spline, c_index_linear, lr_stat, lr_p, N, events.
    """
    if not (PATSY_AVAILABLE and SCIPY_AVAILABLE):
        return None

    cols = [time_col, event_col, prod_col] + covariates
    df_mod = df[cols].dropna().copy()
    df_mod[prod_col] = pd.to_numeric(df_mod[prod_col], errors="coerce")
    df_mod = df_mod.dropna()

    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None
    if df_mod[prod_col].nunique() < 5:
        return None

    # Build natural-cubic-spline basis
    try:
        spline_dm = patsy_dmatrix(
            f"cr(x, df={n_df}) - 1",
            {"x": df_mod[prod_col].values},
            return_type="dataframe",
        )
        spline_dm.index = df_mod.index
        spline_dm.columns = [f"_s{i}" for i in range(spline_dm.shape[1])]
    except Exception as exc:
        warnings.warn(f"Spline basis failed: {exc}")
        return None

    X_spline = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         spline_dm.reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    cph_s = CoxPHFitter(penalizer=penalizer)
    try:
        cph_s.fit(
            X_spline, duration_col=time_col, event_col=event_col, robust=False
        )
    except Exception as exc:
        warnings.warn(f"Spline Cox fit failed: {exc}")
        return None

    # Linear model for LRT comparison
    X_lin = pd.concat(
        [df_mod[[time_col, event_col, prod_col]].reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )
    c_linear = lr_stat = lr_p = np.nan
    try:
        cph_l = CoxPHFitter(penalizer=penalizer)
        cph_l.fit(
            X_lin, duration_col=time_col, event_col=event_col, robust=False
        )
        c_linear = cph_l.concordance_index_
        lr_stat = max(
            -2 * (cph_l.log_likelihood_ - cph_s.log_likelihood_), 0.0
        )
        lr_df = n_df - 1
        lr_p = float(_scipy_stats.chi2.sf(lr_stat, df=lr_df))
    except Exception:
        pass

    return {
        "c_index_spline": cph_s.concordance_index_,
        "c_index_linear": c_linear,
        "lr_stat": lr_stat,
        "lr_p": lr_p,
        "N": len(X_spline),
        "events": int(X_spline[event_col].sum()),
    }


def fit_spline_cox_rbd(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_prob_col: str,
    covariates: List[str],
    n_df: int = 4,
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Restricted cubic spline Cox for continuous RBD probability.

    Characterizes the dose-response relationship between RBD probability
    and hazard. Reports non-linearity LRT, C-index, and the HR curve.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with ``rbd_prob_col``.
    rbd_prob_col : str
        Continuous RBD probability column.
    n_df : int
        Spline degrees of freedom (default 4).
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: c_index_spline, c_index_linear, lr_stat, lr_p,
        hr_curve (DataFrame), N, events.
    """
    if not (PATSY_AVAILABLE and SCIPY_AVAILABLE):
        return None

    cols = [time_col, event_col, rbd_prob_col] + covariates
    df_mod = df[cols].dropna().copy()
    df_mod[rbd_prob_col] = pd.to_numeric(
        df_mod[rbd_prob_col], errors="coerce"
    )
    df_mod = df_mod.dropna()

    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None
    if df_mod[rbd_prob_col].nunique() < 10:
        return None

    # Spline basis
    try:
        spline_dm = patsy_dmatrix(
            f"cr(x, df={n_df}) - 1",
            {"x": df_mod[rbd_prob_col].values},
            return_type="dataframe",
        )
        spline_dm.index = df_mod.index
        spline_cols = [f"_rbd_s{i}" for i in range(spline_dm.shape[1])]
        spline_dm.columns = spline_cols
    except Exception as exc:
        warnings.warn(f"RBD spline basis failed: {exc}")
        return None

    X_spline = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         spline_dm.reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    cph_s = CoxPHFitter(penalizer=penalizer)
    try:
        cph_s.fit(
            X_spline, duration_col=time_col, event_col=event_col, robust=False
        )
    except Exception as exc:
        warnings.warn(f"RBD spline Cox failed: {exc}")
        return None

    # Linear model for LRT
    X_lin = pd.concat(
        [df_mod[[time_col, event_col, rbd_prob_col]].reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )
    c_linear = lr_stat = lr_p = np.nan
    try:
        cph_l = CoxPHFitter(penalizer=penalizer)
        cph_l.fit(
            X_lin, duration_col=time_col, event_col=event_col, robust=False
        )
        c_linear = cph_l.concordance_index_
        lr_stat = max(
            -2 * (cph_l.log_likelihood_ - cph_s.log_likelihood_), 0.0
        )
        lr_df = n_df - 1
        lr_p = float(_scipy_stats.chi2.sf(lr_stat, df=lr_df))
    except Exception:
        pass

    # Build HR curve on a grid
    hr_curve = _predict_rbd_hr_curve(
        cph_s, spline_cols, rbd_prob_col, df_mod, covariates
    )

    return {
        "c_index_spline": cph_s.concordance_index_,
        "c_index_linear": c_linear,
        "lr_stat": lr_stat,
        "lr_p": lr_p,
        "hr_curve": hr_curve,
        "N": len(X_spline),
        "events": int(X_spline[event_col].sum()),
    }


def _predict_rbd_hr_curve(
    cph: CoxPHFitter,
    spline_cols: List[str],
    rbd_prob_col: str,
    df_ref: pd.DataFrame,
    covariates: List[str],
    n_grid: int = 200,
) -> pd.DataFrame:
    """
    Predict HR curve across the RBD probability range.

    Reference point = minimum observed RBD probability.
    HR(p) = exp(f(p) - f(p_min)) where f() is the spline function.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted spline Cox model.
    spline_cols : list[str]
        Column names for the spline basis terms.
    rbd_prob_col : str
        Name of the original RBD probability column.
    df_ref : pd.DataFrame
        Data used for computing the spline basis on the grid.
    covariates : list[str]
        Covariate columns (held at median).
    n_grid : int
        Number of grid points.

    Returns
    -------
    pd.DataFrame
        Columns: rbd_prob, HR, HR_LCI, HR_UCI.
    """
    if not PATSY_AVAILABLE:
        return pd.DataFrame()

    p_min = float(df_ref[rbd_prob_col].min())
    p_max = float(df_ref[rbd_prob_col].max())
    grid = np.linspace(p_min, p_max, n_grid)

    n_df = len(spline_cols)
    try:
        grid_dm = patsy_dmatrix(
            f"cr(x, df={n_df}) - 1",
            {"x": grid},
            return_type="dataframe",
        )
        grid_dm.columns = spline_cols
    except Exception:
        return pd.DataFrame()

    # Reference point spline basis
    try:
        ref_dm = patsy_dmatrix(
            f"cr(x, df={n_df}) - 1",
            {"x": np.array([p_min])},
            return_type="dataframe",
        )
        ref_dm.columns = spline_cols
    except Exception:
        return pd.DataFrame()

    # Extract spline coefficients from the model
    betas = np.array([
        cph.params_[col] for col in spline_cols if col in cph.params_.index
    ])
    if len(betas) != len(spline_cols):
        return pd.DataFrame()

    # Log-HR relative to reference
    log_hr_grid = grid_dm.values @ betas
    log_hr_ref = ref_dm.values @ betas
    log_hr = log_hr_grid - log_hr_ref[0]

    # Variance via delta method
    try:
        var_matrix = cph.variance_matrix_
        spline_idx = [
            list(cph.params_.index).index(c)
            for c in spline_cols if c in cph.params_.index
        ]
        V = var_matrix.iloc[spline_idx, spline_idx].values
        diff = grid_dm.values - ref_dm.values
        se_log_hr = np.sqrt(np.diag(diff @ V @ diff.T))
    except Exception:
        se_log_hr = np.full(len(grid), np.nan)

    return pd.DataFrame({
        "rbd_prob": grid,
        "HR": np.exp(log_hr),
        "HR_LCI": np.exp(log_hr - 1.96 * se_log_hr),
        "HR_UCI": np.exp(log_hr + 1.96 * se_log_hr),
    })
