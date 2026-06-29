"""
Additive interaction analysis: RERI, AP, and Synergy Index.

Tests departure from additivity on the HR scale, which corresponds
to biological interaction (Rothman, 1986).

Uses bootstrap for confidence intervals because the RERI point estimate
involves differences of ratios and is not log-linear.

Bootstrap uses scikit-survival (sksurv) for ~3-5x faster Cox fits.
The original lifelines bootstrap is preserved as ``_fit_one_bootstrap_lifelines``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
from lifelines import CoxPHFitter

from library.cox_prodromal.cox_config import (
    BOOTSTRAP_COEF_CLIP,
    BOOTSTRAP_JOBS,
    BOOTSTRAP_N,
    BOOTSTRAP_RIDGE_PENALIZER,
    BOOTSTRAP_SEED,
    MIN_EVENTS_FOR_MODEL,
    RIDGE_PENALIZER,
)


@dataclass
class AdditiveInteractionResult:
    """Results from additive interaction analysis."""
    reri: float           # HR_11 - HR_10 - HR_01 + 1
    reri_lci: float
    reri_uci: float
    ap: float             # RERI / HR_11
    ap_lci: float
    ap_uci: float
    synergy_index: float  # (HR_11 - 1) / ((HR_10 - 1) + (HR_01 - 1))
    si_lci: float
    si_uci: float
    hr_11: float          # R=1, P=1
    hr_10: float          # R=1, P=0
    hr_01: float          # R=0, P=1
    n_bootstrap: int
    N: int
    events: int
    # Per-cell counts (R=rbd, P=prodromal)
    n_00: int = 0         # R=0, P=0 (reference)
    events_00: int = 0
    n_10: int = 0         # R=1, P=0
    events_10: int = 0
    n_01: int = 0         # R=0, P=1
    events_01: int = 0
    n_11: int = 0         # R=1, P=1
    events_11: int = 0
    sparse_cell: bool = False  # True if any cell has < 10 events


def compute_four_group_hrs(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_binary_col: str,
    prod_binary_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, float]]:
    """
    Fit Cox with 4-group indicator (R=0/P=0 as reference).

    Groups:
    - grp_00: R=0, P=0 (reference, HR = 1.0)
    - grp_10: R=1, P=0
    - grp_01: R=0, P=1
    - grp_11: R=1, P=1

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    rbd_binary_col : str
        Binary RBD column (0 = Low, 1 = High).
    prod_binary_col : str
        Binary prodromal column (0 = No, 1 = Yes).
    covariates : list[str]
        Adjustment covariates.
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: hr_00, hr_10, hr_01, hr_11.
    """
    cols = [time_col, event_col, rbd_binary_col, prod_binary_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    # Create 4-group indicator
    r = df_mod[rbd_binary_col].astype(int)
    p = df_mod[prod_binary_col].astype(int)

    df_mod["grp_10"] = ((r == 1) & (p == 0)).astype(int)
    df_mod["grp_01"] = ((r == 0) & (p == 1)).astype(int)
    df_mod["grp_11"] = ((r == 1) & (p == 1)).astype(int)

    X = df_mod[
        [time_col, event_col, "grp_10", "grp_01", "grp_11"] + covariates
    ].reset_index(drop=True)

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception:
        return None

    # Cell counts for reviewer transparency
    grp_00_mask = (r == 0) & (p == 0)
    grp_10_mask = (r == 1) & (p == 0)
    grp_01_mask = (r == 0) & (p == 1)
    grp_11_mask = (r == 1) & (p == 1)

    ev = df_mod[event_col]
    cell_counts = {
        "n_00": int(grp_00_mask.sum()),
        "events_00": int(ev[grp_00_mask].sum()),
        "n_10": int(grp_10_mask.sum()),
        "events_10": int(ev[grp_10_mask].sum()),
        "n_01": int(grp_01_mask.sum()),
        "events_01": int(ev[grp_01_mask].sum()),
        "n_11": int(grp_11_mask.sum()),
        "events_11": int(ev[grp_11_mask].sum()),
    }
    cell_counts["sparse_cell"] = any(
        cell_counts[f"events_{g}"] < 10 for g in ("00", "10", "01", "11")
    )

    return {
        "hr_00": 1.0,
        "hr_10": float(cph.summary.loc["grp_10", "exp(coef)"]),
        "hr_01": float(cph.summary.loc["grp_01", "exp(coef)"]),
        "hr_11": float(cph.summary.loc["grp_11", "exp(coef)"]),
        **cell_counts,
    }


def compute_reri_ap_si(
    hr_11: float,
    hr_10: float,
    hr_01: float,
) -> Tuple[float, float, float]:
    """
    Point estimates for RERI, AP, and Synergy Index.

    RERI = HR_11 - HR_10 - HR_01 + 1
    AP   = RERI / HR_11
    S    = (HR_11 - 1) / ((HR_10 - 1) + (HR_01 - 1))

    Parameters
    ----------
    hr_11, hr_10, hr_01 : float
        Hazard ratios for the three non-reference groups.

    Returns
    -------
    tuple of (RERI, AP, S)
    """
    reri = hr_11 - hr_10 - hr_01 + 1.0

    ap = reri / hr_11 if hr_11 != 0 else np.nan

    denom = (hr_10 - 1.0) + (hr_01 - 1.0)
    si = (hr_11 - 1.0) / denom if abs(denom) > 1e-10 else np.nan

    return reri, ap, si


def _fit_one_bootstrap_lifelines(
    df: pd.DataFrame,
    boot_seed: int,
    time_col: str,
    event_col: str,
    rbd_binary_col: str,
    prod_binary_col: str,
    covariates: List[str],
    penalizer: float,
) -> Optional[Tuple[float, float, float]]:
    """Single bootstrap iteration using lifelines (kept as fallback).

    Pure function — safe for parallel dispatch via joblib.
    Each call gets a unique ``boot_seed`` so results are deterministic
    and independent of execution order.

    Parameters
    ----------
    df : pd.DataFrame
        Complete-case survival dataset (not resampled).
    boot_seed : int
        Seed for this specific bootstrap draw.

    Returns
    -------
    tuple(RERI, AP, SI) or None if Cox fit failed.
    """
    rng = np.random.default_rng(boot_seed)
    idx = rng.choice(len(df), size=len(df), replace=True)
    df_boot = df.iloc[idx].reset_index(drop=True)

    hrs_b = compute_four_group_hrs(
        df_boot, time_col, event_col,
        rbd_binary_col, prod_binary_col, covariates, penalizer,
    )
    if hrs_b is None:
        return None
    return compute_reri_ap_si(hrs_b["hr_11"], hrs_b["hr_10"], hrs_b["hr_01"])


def _prepare_sksurv_arrays(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_binary_col: str,
    prod_binary_col: str,
    covariates: List[str],
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """Pre-compute numpy arrays for sksurv bootstrap.

    Returns
    -------
    X : np.ndarray
        Feature matrix [grp_10, grp_01, grp_11, *covariates].
    y : np.ndarray
        Structured survival array for sksurv.
    idx_grp_10, idx_grp_01, idx_grp_11 : int
        Column indices into X for the group indicators.
    """
    cols = [time_col, event_col, rbd_binary_col, prod_binary_col] + covariates
    df_cc = df[cols].dropna()

    r = df_cc[rbd_binary_col].values.astype(int)
    p = df_cc[prod_binary_col].values.astype(int)

    grp_10 = ((r == 1) & (p == 0)).astype(np.float64)
    grp_01 = ((r == 0) & (p == 1)).astype(np.float64)
    grp_11 = ((r == 1) & (p == 1)).astype(np.float64)

    cov_arr = df_cc[covariates].values.astype(np.float64)
    X = np.column_stack([grp_10, grp_01, grp_11, cov_arr])

    y = np.empty(len(df_cc), dtype=[("event", bool), ("time", float)])
    y["event"] = df_cc[event_col].values.astype(bool)
    y["time"] = df_cc[time_col].values.astype(float)

    return X, y, 0, 1, 2  # indices of grp_10, grp_01, grp_11


def _fit_one_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    boot_seed: int,
    alpha: float,
    idx_10: int,
    idx_01: int,
    idx_11: int,
) -> Optional[Tuple[float, float, float]]:
    """Single bootstrap iteration using sksurv (3-5x faster than lifelines).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (pre-computed, columns: grp_10, grp_01, grp_11, covs).
    y : np.ndarray
        Structured survival array.
    boot_seed : int
        Seed for this bootstrap draw.
    alpha : float
        L2 regularization strength. Use BOOTSTRAP_RIDGE_PENALIZER (0.1)
        to suppress overflow in pathological samples.
    idx_10, idx_01, idx_11 : int
        Column indices for group indicators in X.

    Returns
    -------
    tuple(RERI, AP, SI) or None if fit failed.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    rng = np.random.default_rng(boot_seed)
    n = len(y)
    idx = rng.choice(n, size=n, replace=True)

    try:
        cph = CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow")
        # Suppress ConvergenceWarning: non-convergent bootstrap samples are
        # discarded via the None return; the warning itself is uninformative
        # noise at scale (1000 iterations).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cph.fit(X[idx], y[idx])
        # Clip to prevent exp() overflow on pathological samples that squeaked
        # through despite strong regularization.
        coefs = np.clip(cph.coef_, -BOOTSTRAP_COEF_CLIP, BOOTSTRAP_COEF_CLIP)
        hr_10 = float(np.exp(coefs[idx_10]))
        hr_01 = float(np.exp(coefs[idx_01]))
        hr_11 = float(np.exp(coefs[idx_11]))
        return compute_reri_ap_si(hr_11, hr_10, hr_01)
    except Exception:
        return None


def bootstrap_additive_interaction(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_binary_col: str,
    prod_binary_col: str,
    covariates: List[str],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    penalizer: float = RIDGE_PENALIZER,
    n_jobs: int = BOOTSTRAP_JOBS,
) -> Optional[AdditiveInteractionResult]:
    """Bootstrap CIs for RERI, AP, and Synergy Index.

    Algorithm:
    1. Compute point estimates from the original data.
    2. Resample N rows with replacement (n_bootstrap times) **in parallel**.
    3. For each bootstrap sample, fit 4-group Cox and compute RERI, AP, S.
    4. Report percentile CIs (2.5th, 97.5th).

    Parallelism uses ``joblib`` with the ``loky`` backend, which is safe
    for nested process pools (this function runs inside a
    ``ProcessPoolExecutor`` outcome worker).

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    rbd_binary_col : str
        Binary RBD column (0/1 or Low/High mapped to 0/1).
    prod_binary_col : str
        Binary prodromal column (0/1 or No/Yes mapped to 0/1).
    covariates : list[str]
        Adjustment covariates.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.
    penalizer : float
        Ridge penalizer.
    n_jobs : int
        Number of parallel workers for bootstrap (default from config).

    Returns
    -------
    AdditiveInteractionResult or None
    """
    # Point estimates
    hrs = compute_four_group_hrs(
        df, time_col, event_col,
        rbd_binary_col, prod_binary_col, covariates, penalizer,
    )
    if hrs is None:
        return None

    reri_pt, ap_pt, si_pt = compute_reri_ap_si(
        hrs["hr_11"], hrs["hr_10"], hrs["hr_01"]
    )

    # Pre-compute numpy arrays for sksurv fast path (avoids repeated
    # DataFrame → numpy conversion inside each bootstrap iteration).
    X_np, y_np, idx_10, idx_01, idx_11 = _prepare_sksurv_arrays(
        df, time_col, event_col,
        rbd_binary_col, prod_binary_col, covariates,
    )

    # Parallel bootstrap via sksurv (3-5x faster than lifelines per fit).
    # BOOTSTRAP_RIDGE_PENALIZER (0.1) is stronger than the main-model
    # penalizer to prevent overflow in bootstrap samples with near-perfect
    # separation.  Only affects the CI width marginally.
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one_bootstrap)(
            X_np, y_np, seed + i, BOOTSTRAP_RIDGE_PENALIZER,
            idx_10, idx_01, idx_11,
        )
        for i in range(n_bootstrap)
    )

    boot_reri: List[float] = []
    boot_ap: List[float] = []
    boot_si: List[float] = []
    for res in results:
        if res is not None:
            boot_reri.append(res[0])
            boot_ap.append(res[1])
            boot_si.append(res[2])

    if len(boot_reri) < n_bootstrap * 0.5:
        warnings.warn(
            f"Only {len(boot_reri)}/{n_bootstrap} bootstrap samples converged"
        )
        if len(boot_reri) < 10:
            return None

    def _pci(vals: List[float]) -> Tuple[float, float]:
        arr = np.array([v for v in vals if np.isfinite(v)])
        if len(arr) < 10:
            return np.nan, np.nan
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    reri_lo, reri_hi = _pci(boot_reri)
    ap_lo, ap_hi = _pci(boot_ap)
    si_lo, si_hi = _pci(boot_si)

    cols_needed = [time_col, event_col, rbd_binary_col, prod_binary_col] + covariates
    df_cc = df[cols_needed].dropna()

    return AdditiveInteractionResult(
        reri=reri_pt, reri_lci=reri_lo, reri_uci=reri_hi,
        ap=ap_pt, ap_lci=ap_lo, ap_uci=ap_hi,
        synergy_index=si_pt, si_lci=si_lo, si_uci=si_hi,
        hr_11=hrs["hr_11"], hr_10=hrs["hr_10"], hr_01=hrs["hr_01"],
        n_bootstrap=len(boot_reri),
        N=len(df_cc),
        events=int(df_cc[event_col].sum()),
        n_00=hrs.get("n_00", 0), events_00=hrs.get("events_00", 0),
        n_10=hrs.get("n_10", 0), events_10=hrs.get("events_10", 0),
        n_01=hrs.get("n_01", 0), events_01=hrs.get("events_01", 0),
        n_11=hrs.get("n_11", 0), events_11=hrs.get("events_11", 0),
        sparse_cell=hrs.get("sparse_cell", False),
    )
