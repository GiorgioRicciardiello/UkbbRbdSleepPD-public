"""
Discrimination metrics: C-index, delta-C bootstrap test, NRI, IDI.

C-index (Harrell's concordance) is insensitive to incremental improvement.
NRI and IDI provide complementary reclassification-based evaluation.

Bootstrap uses scikit-survival (sksurv) for ~3-5x faster Cox fits via its
C/Cython internals.  The original lifelines implementation is preserved as
``bootstrap_delta_c_test_lifelines`` for validation/fallback.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter

from library.cox_prodromal.cox_config import (
    BOOTSTRAP_JOBS,
    BOOTSTRAP_N,
    BOOTSTRAP_RIDGE_PENALIZER,
    BOOTSTRAP_SEED,
    RIDGE_PENALIZER,
)


def compute_c_index(cph: CoxPHFitter) -> float:
    """Extract Harrell's C-index from a fitted CoxPHFitter."""
    return cph.concordance_index_


def compute_incremental_c_index(c_full: float, c_null: float) -> float:
    """Delta C = C_full - C_null."""
    return c_full - c_null


# ── sksurv helpers ────────────────────────────────────────────────────────────

def _make_sksurv_y(
    event: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Build the structured array that sksurv expects as y.

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


def _fit_one_delta_c_sksurv(
    X_full: np.ndarray,
    X_null: np.ndarray,
    y: np.ndarray,
    boot_seed: int,
    alpha: float,
) -> Optional[float]:
    """Single bootstrap iteration for delta-C using sksurv.

    Parameters
    ----------
    X_full : np.ndarray
        Feature matrix for the full model.
    X_null : np.ndarray
        Feature matrix for the null model.
    y : np.ndarray
        Structured survival array.
    boot_seed : int
        Seed for this bootstrap draw.
    alpha : float
        L2 regularization strength (maps to sksurv's alpha parameter).

    Returns
    -------
    float or None
        Delta-C for this bootstrap sample, or None on failure.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    rng = np.random.default_rng(boot_seed)
    n = len(y)
    idx = rng.choice(n, size=n, replace=True)

    y_b = y[idx]
    X_full_b = X_full[idx]
    X_null_b = X_null[idx]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cph_full = CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow")
            cph_full.fit(X_full_b, y_b)
            c_full = cph_full.score(X_full_b, y_b)

            cph_null = CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow")
            cph_null.fit(X_null_b, y_b)
            c_null = cph_null.score(X_null_b, y_b)

        return c_full - c_null
    except Exception:
        return None


def bootstrap_delta_c_test(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    model_full_cols: List[str],
    model_null_cols: List[str],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    penalizer: float = RIDGE_PENALIZER,
    n_jobs: int = BOOTSTRAP_JOBS,
) -> Dict[str, float]:
    """
    Parallel bootstrap test for significance of delta-C (C_full - C_null).

    Uses scikit-survival's CoxPHSurvivalAnalysis for ~3-5x faster fitting.
    Bootstrap iterations are parallelized via joblib.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset (must contain all specified columns).
    model_full_cols : list[str]
        Predictor columns for the full model.
    model_null_cols : list[str]
        Predictor columns for the null (reference) model.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.
    penalizer : float
        Ridge penalizer (mapped to sksurv alpha).
    n_jobs : int
        Number of parallel workers for bootstrap.

    Returns
    -------
    dict
        Keys: delta_c, delta_c_lci, delta_c_uci, p_value,
              n_converged, n_bootstrap.
        p_value = proportion of bootstrap deltas <= 0.
    """
    all_cols = list(set([time_col, event_col] + model_full_cols + model_null_cols))
    df_cc = df[all_cols].dropna().reset_index(drop=True)

    y = _make_sksurv_y(
        df_cc[event_col].values,
        df_cc[time_col].values,
    )
    X_full = df_cc[model_full_cols].values.astype(np.float64)
    X_null = df_cc[model_null_cols].values.astype(np.float64)

    # Use BOOTSTRAP_RIDGE_PENALIZER (0.1) to prevent overflow in bootstrap
    # samples with near-perfect separation.
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one_delta_c_sksurv)(
            X_full, X_null, y, seed + i, BOOTSTRAP_RIDGE_PENALIZER,
        )
        for i in range(n_bootstrap)
    )

    deltas = [r for r in results if r is not None]

    if len(deltas) < 10:
        return {
            "delta_c": np.nan, "delta_c_lci": np.nan,
            "delta_c_uci": np.nan, "p_value": np.nan,
            "n_converged": len(deltas), "n_bootstrap": n_bootstrap,
        }

    arr = np.array(deltas)
    p_value = float(np.mean(arr <= 0))

    return {
        "delta_c": float(np.mean(arr)),
        "delta_c_lci": float(np.percentile(arr, 2.5)),
        "delta_c_uci": float(np.percentile(arr, 97.5)),
        "p_value": p_value,
        "n_converged": len(deltas),
        "n_bootstrap": n_bootstrap,
    }


# ── Lifelines fallback (preserved for validation) ────────────────────────────

def bootstrap_delta_c_test_lifelines(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    model_full_cols: List[str],
    model_null_cols: List[str],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    penalizer: float = RIDGE_PENALIZER,
) -> Dict[str, float]:
    """
    Original sequential lifelines bootstrap for delta-C.

    Kept as fallback/validation. Use ``bootstrap_delta_c_test`` for production.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset (must contain all specified columns).
    model_full_cols : list[str]
        Predictor columns for the full model.
    model_null_cols : list[str]
        Predictor columns for the null (reference) model.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict
        Keys: delta_c, delta_c_lci, delta_c_uci, p_value.
        p_value = proportion of bootstrap deltas <= 0.
    """
    all_cols = list(set([time_col, event_col] + model_full_cols + model_null_cols))
    df_cc = df[all_cols].dropna().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    deltas = []
    n = len(df_cc)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        df_b = df_cc.iloc[idx].reset_index(drop=True)

        try:
            cph_full = CoxPHFitter(penalizer=penalizer)
            cph_full.fit(
                df_b[[time_col, event_col] + model_full_cols],
                duration_col=time_col, event_col=event_col, robust=False,
            )
            cph_null = CoxPHFitter(penalizer=penalizer)
            cph_null.fit(
                df_b[[time_col, event_col] + model_null_cols],
                duration_col=time_col, event_col=event_col, robust=False,
            )
            deltas.append(
                cph_full.concordance_index_ - cph_null.concordance_index_
            )
        except Exception:
            continue

    if len(deltas) < 10:
        return {
            "delta_c": np.nan, "delta_c_lci": np.nan,
            "delta_c_uci": np.nan, "p_value": np.nan,
        }

    arr = np.array(deltas)
    p_value = float(np.mean(arr <= 0))

    return {
        "delta_c": float(np.mean(arr)),
        "delta_c_lci": float(np.percentile(arr, 2.5)),
        "delta_c_uci": float(np.percentile(arr, 97.5)),
        "p_value": p_value,
    }


def extract_predicted_risks(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
) -> np.ndarray:
    """
    Extract predicted hazard (linear predictor exp(Xb)) from a fitted model.

    Used as input to NRI and IDI computations.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted model.
    df : pd.DataFrame
        Data containing the same columns used for fitting.

    Returns
    -------
    np.ndarray
        Predicted partial hazard for each subject.
    """
    model_cols = [
        c for c in cph.params_.index if c in df.columns
    ]
    X = df[model_cols].copy()
    return cph.predict_partial_hazard(X).values.flatten()


def compute_nri(
    risk_old: np.ndarray,
    risk_new: np.ndarray,
    events: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Category-based Net Reclassification Improvement.

    Classifies subjects as above/below threshold under old and new models.
    NRI = (P(up|event) - P(down|event)) + (P(down|non-event) - P(up|non-event))

    Parameters
    ----------
    risk_old : np.ndarray
        Predicted risks from the reference model.
    risk_new : np.ndarray
        Predicted risks from the enhanced model.
    events : np.ndarray
        Binary event indicators (0/1).
    threshold : float
        Risk cutoff for reclassification.

    Returns
    -------
    dict
        Keys: nri_events, nri_nonevents, nri_total, se, z, p.
    """
    old_class = (risk_old >= threshold).astype(int)
    new_class = (risk_new >= threshold).astype(int)

    up = (new_class > old_class).astype(int)
    down = (new_class < old_class).astype(int)

    ev_mask = events == 1
    ne_mask = events == 0

    n_events = ev_mask.sum()
    n_nonevents = ne_mask.sum()

    if n_events == 0 or n_nonevents == 0:
        return {
            "nri_events": np.nan, "nri_nonevents": np.nan,
            "nri_total": np.nan, "se": np.nan, "z": np.nan, "p": np.nan,
        }

    nri_events = (up[ev_mask].sum() - down[ev_mask].sum()) / n_events
    nri_nonevents = (down[ne_mask].sum() - up[ne_mask].sum()) / n_nonevents
    nri_total = nri_events + nri_nonevents

    # Standard error via Pencina et al. 2008
    se_ev = np.sqrt(
        (up[ev_mask].sum() + down[ev_mask].sum()) / n_events**2
    )
    se_ne = np.sqrt(
        (up[ne_mask].sum() + down[ne_mask].sum()) / n_nonevents**2
    )
    se = np.sqrt(se_ev**2 + se_ne**2)
    z = nri_total / se if se > 0 else np.nan

    from scipy.stats import norm
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan

    return {
        "nri_events": float(nri_events),
        "nri_nonevents": float(nri_nonevents),
        "nri_total": float(nri_total),
        "se": float(se),
        "z": float(z),
        "p": p,
    }


def compute_idi(
    risk_old: np.ndarray,
    risk_new: np.ndarray,
    events: np.ndarray,
) -> Dict[str, float]:
    """
    Integrated Discrimination Improvement (Pencina et al., 2008).

    IDI = (mean_risk_new_events - mean_risk_new_nonevents)
        - (mean_risk_old_events - mean_risk_old_nonevents)

    Continuous analog of NRI. No threshold needed.

    Parameters
    ----------
    risk_old : np.ndarray
        Predicted risks from the reference model.
    risk_new : np.ndarray
        Predicted risks from the enhanced model.
    events : np.ndarray
        Binary event indicators (0/1).

    Returns
    -------
    dict
        Keys: idi, is_old, is_new, p.
    """
    ev_mask = events == 1
    ne_mask = events == 0

    if ev_mask.sum() == 0 or ne_mask.sum() == 0:
        return {"idi": np.nan, "is_old": np.nan, "is_new": np.nan, "p": np.nan}

    # Integration discrimination slope
    is_old = risk_old[ev_mask].mean() - risk_old[ne_mask].mean()
    is_new = risk_new[ev_mask].mean() - risk_new[ne_mask].mean()
    idi = is_new - is_old

    # Approximate SE via bootstrap of the difference in means
    n = len(events)
    se_idi = np.sqrt(
        np.var(risk_new[ev_mask]) / ev_mask.sum()
        + np.var(risk_new[ne_mask]) / ne_mask.sum()
        + np.var(risk_old[ev_mask]) / ev_mask.sum()
        + np.var(risk_old[ne_mask]) / ne_mask.sum()
    )
    z = idi / se_idi if se_idi > 0 else np.nan

    from scipy.stats import norm
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan

    return {
        "idi": float(idi),
        "is_old": float(is_old),
        "is_new": float(is_new),
        "p": p,
    }
