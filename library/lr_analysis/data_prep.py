"""
Data preparation for the LR-MDS analysis.

Loads the production dataset, filters to incident PD cases and controls,
and applies a no-leakage z-score normalisation of the RBD score using
only the control distribution.

Also computes the UKBB-empirical age-band prior used in Option C1.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from library.cox_prodromal.data_prep import load_prodromal_dataset
from library.lr_analysis.config import (
    AGE_COL,
    COHORT_DEFINITIONS,
    CONTROL_COL,
    INCIDENT_COL,
    MDS_AGE_PRIOR,
    PRODROMAL_EXCLUDED,
    PRODROMAL_VIABLE,
    RBD_COL,
    RBD_ZSCORE_COL,
    SEX_COL,
)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalysisFrame:
    """All data needed by downstream analysis modules.

    Attributes
    ----------
    df : pd.DataFrame
        Analysis-set rows (incident PD + controls). Contains ``rbd_zscore``.
    is_case : pd.Series[bool]
        True for incident PD subjects.
    is_ctrl : pd.Series[bool]
        True for control subjects.
    zscore_mu : float
        Control-group mean of ``abk_rbd_score_mean`` (z-score reference).
    zscore_sigma : float
        Control-group SD of ``abk_rbd_score_mean`` (z-score reference).
    n_cases : int
    n_controls : int
    ukbb_age_prior : dict[int, float]
        Age-band specific incidence proportion from UKBB data.
        Keys: lower bound of 5-year band. Used as prior in Option C1.
    """

    df: pd.DataFrame
    is_case: pd.Series
    is_ctrl: pd.Series
    zscore_mu: float
    zscore_sigma: float
    n_cases: int
    n_controls: int
    ukbb_age_prior: dict[int, float]


# ── Age-band utilities ────────────────────────────────────────────────────────

_AGE_BANDS: list[int] = sorted(MDS_AGE_PRIOR.keys())


def _assign_age_band(age: float) -> int:
    """Return the lower bound of the MDS 5-year age band for a given age.

    Ages below the lowest band floor are assigned to the lowest band.
    Ages above the highest band floor are assigned to the highest band.
    """
    if np.isnan(age):
        return _AGE_BANDS[0]
    for band in reversed(_AGE_BANDS):
        if age >= band:
            return band
    return _AGE_BANDS[0]


# ── Z-score (no-leakage) ─────────────────────────────────────────────────────

def _zscore_controls_only(
    series: pd.Series,
    ctrl_mask: pd.Series,
) -> tuple[pd.Series, float, float]:
    """Z-score a series using mean and SD from controls only.

    Parameters
    ----------
    series : pd.Series
        Raw RBD score column.
    ctrl_mask : pd.Series[bool]
        True for control subjects. Cases must NOT influence the normalization.

    Returns
    -------
    z_series : pd.Series
        Z-scored values (NaN propagates).
    mu : float
        Control mean used for normalization.
    sigma : float
        Control SD used for normalization.

    Raises
    ------
    ValueError
        If control variance is zero.
    """
    ctrl_vals = series[ctrl_mask].dropna()
    mu = float(ctrl_vals.mean())
    sigma = float(ctrl_vals.std(ddof=1))
    if sigma == 0.0:
        raise ValueError(
            f"Zero variance in controls for column '{series.name}'. "
            "Cannot compute z-score."
        )
    z = (series - mu) / sigma
    return z, mu, sigma


# ── UKBB empirical prior ──────────────────────────────────────────────────────

def _compute_ukbb_empirical_prior(
    df: pd.DataFrame,
    is_case: pd.Series,
) -> dict[int, float]:
    """Estimate age-band incidence proportions from the UKBB cohort.

    For each MDS 5-year age band: proportion = n_cases_in_band / n_total_in_band.

    This is a cross-sectional approximation. It underestimates the true
    lifetime incidence because follow-up time is finite, but is internally
    consistent when used alongside empirically-derived LRs from the same
    cohort (Option C1).

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-set rows.
    is_case : pd.Series[bool]
        Incident PD indicator, aligned to df.index.

    Returns
    -------
    dict[int, float]
        {age_band_lower_bound: incidence_proportion}.
        Falls back to the overall proportion for empty bands.
    """
    age_bands = df[AGE_COL].apply(_assign_age_band)
    overall_prop = float(is_case.mean())
    prior: dict[int, float] = {}

    for band in _AGE_BANDS:
        band_mask = age_bands == band
        n_band = int(band_mask.sum())
        if n_band == 0:
            warnings.warn(
                f"Age band {band} has no subjects — using overall proportion.",
                UserWarning,
                stacklevel=2,
            )
            prior[band] = overall_prop
        else:
            n_cases_band = int(is_case[band_mask].sum())
            prior[band] = n_cases_band / n_band

    return prior


# ── Cohort filtering (complete-case by required variables) ────────────────────

def filter_cohort(
    frame: AnalysisFrame,
    cohort_name: str,
) -> tuple[AnalysisFrame, dict[str, int]]:
    """Filter analysis frame to complete cases for a specific cohort.

    Parameters
    ----------
    frame : AnalysisFrame
        Base analysis frame from build_analysis_frame().
    cohort_name : str
        Name of cohort ("mds_standard", "cognitive", "tmt", "genetic").

    Returns
    -------
    filtered_frame : AnalysisFrame
        Frame restricted to subjects with all required variables for cohort.
    cohort_stats : dict
        Statistics: n_total, n_cases, n_controls, pct_complete, n_missing.
    """
    if cohort_name not in COHORT_DEFINITIONS:
        raise ValueError(
            f"Unknown cohort '{cohort_name}'. "
            f"Must be one of {list(COHORT_DEFINITIONS.keys())}"
        )

    required_cols = COHORT_DEFINITIONS[cohort_name]
    missing_cols = [c for c in required_cols if c not in frame.df.columns]
    if missing_cols:
        raise KeyError(
            f"Cohort '{cohort_name}' requires columns {missing_cols}, "
            f"which are missing from dataset."
        )

    # Create complete-case mask
    complete_mask = frame.df[required_cols].notna().all(axis=1)
    n_complete = int(complete_mask.sum())
    n_missing = len(frame.df) - n_complete
    pct_complete = 100.0 * n_complete / len(frame.df)

    # Filter frame
    df_cohort = frame.df[complete_mask].copy().reset_index(drop=True)
    is_case_cohort = frame.is_case[complete_mask].reset_index(drop=True)
    is_ctrl_cohort = frame.is_ctrl[complete_mask].reset_index(drop=True)

    n_cases_cohort = int(is_case_cohort.sum())
    n_controls_cohort = int(is_ctrl_cohort.sum())

    cohort_stats = {
        "cohort_name": cohort_name,
        "n_total": n_complete,
        "n_cases": n_cases_cohort,
        "n_controls": n_controls_cohort,
        "pct_complete": round(pct_complete, 2),
        "n_missing": n_missing,
    }

    filtered_frame = AnalysisFrame(
        df=df_cohort,
        is_case=is_case_cohort,
        is_ctrl=is_ctrl_cohort,
        zscore_mu=frame.zscore_mu,
        zscore_sigma=frame.zscore_sigma,
        n_cases=n_cases_cohort,
        n_controls=n_controls_cohort,
        ukbb_age_prior=frame.ukbb_age_prior,
    )

    return filtered_frame, cohort_stats


# ── Main entry point ──────────────────────────────────────────────────────────

def build_analysis_frame(
    file_name: str = "ehr_diag_pd_rbd_only_all",
) -> AnalysisFrame:
    """Load production data and build the analysis-ready frame.

    Steps
    -----
    1. Load via ``load_prodromal_dataset``.
    2. Filter to incident PD cases + controls (drop all others).
    3. Validate required columns are present.
    4. Z-score ``abk_rbd_score_mean`` using control distribution only.
    5. Compute UKBB empirical age-band prior.
    6. Log excluded prodromal markers.

    Parameters
    ----------
    file_name : str
        Dataset base-name passed to ``load_prodromal_dataset``.

    Returns
    -------
    AnalysisFrame
    """
    _, df_full = load_prodromal_dataset(file_name=file_name)

    # ── Validate required columns ────────────────────────────────────────────
    required = [INCIDENT_COL, CONTROL_COL, RBD_COL, SEX_COL, AGE_COL]
    missing_cols = [c for c in required if c not in df_full.columns]
    if missing_cols:
        raise KeyError(
            f"Required columns missing from dataset: {missing_cols}"
        )

    # ── Filter to analysis set ───────────────────────────────────────────────
    is_case_full = df_full[INCIDENT_COL].fillna(False).astype(bool)
    is_ctrl_full = df_full[CONTROL_COL].fillna(False).astype(bool)
    analysis_mask = is_case_full | is_ctrl_full

    df = df_full[analysis_mask].copy().reset_index(drop=True)
    is_case = df[INCIDENT_COL].fillna(False).astype(bool)
    is_ctrl = df[CONTROL_COL].fillna(False).astype(bool)

    n_cases = int(is_case.sum())
    n_controls = int(is_ctrl.sum())
    print(
        f"[data_prep] Analysis set: {len(df):,} subjects "
        f"({n_cases} incident PD, {n_controls} controls)"
    )

    # ── Warn about excluded prodromal markers ────────────────────────────────
    for col in PRODROMAL_EXCLUDED:
        if col in df.columns and df[col].sum() == 0:
            print(f"[data_prep] EXCLUDED (all-zero): {col}")

    # ── Validate viable prodromal markers ────────────────────────────────────
    for col in PRODROMAL_VIABLE:
        if col not in df.columns:
            warnings.warn(
                f"Viable prodromal column '{col}' not found in dataset.",
                UserWarning,
                stacklevel=2,
            )
        elif df[col].isna().any():
            n_miss = int(df[col].isna().sum())
            warnings.warn(
                f"Viable prodromal column '{col}' has {n_miss} NaN values.",
                UserWarning,
                stacklevel=2,
            )

    # ── Z-score (no leakage) ─────────────────────────────────────────────────
    rbd_series = df[RBD_COL].copy()
    z_series, mu, sigma = _zscore_controls_only(rbd_series, is_ctrl)
    df = df.assign(**{RBD_ZSCORE_COL: z_series})
    print(
        f"[data_prep] RBD z-score: mu_ctrl={mu:.4f}, sigma_ctrl={sigma:.4f} "
        f"(cases mean={df.loc[is_case, RBD_ZSCORE_COL].mean():.3f} SD="
        f"{df.loc[is_case, RBD_ZSCORE_COL].std():.3f})"
    )

    # ── UKBB empirical prior ─────────────────────────────────────────────────
    ukbb_prior = _compute_ukbb_empirical_prior(df, is_case)
    print("[data_prep] UKBB empirical prior (per age band):")
    for band, p in sorted(ukbb_prior.items()):
        print(f"  Age {band}: {p*100:.3f}%")

    return AnalysisFrame(
        df=df,
        is_case=is_case,
        is_ctrl=is_ctrl,
        zscore_mu=mu,
        zscore_sigma=sigma,
        n_cases=n_cases,
        n_controls=n_controls,
        ukbb_age_prior=ukbb_prior,
    )
