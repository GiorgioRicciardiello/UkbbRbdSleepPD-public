"""
Sequential Bayesian framework — Options C1 and C2.

C2 (Hybrid): Berg/Heinzel MDS age prior + published Heinzel LRs for prodromal
markers + empirical actigraphy LR from our data.

C1 (Empirical): UKBB age-band incidence prior + empirically derived LRs for
4 viable prodromal markers from our cohort + empirical actigraphy LR.

Both produce per-subject posterior probability of prodromal PD.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from library.lr_analysis.config import (
    AGE_COL,
    FEMALE_CODE,
    HEINZEL_LRS,
    MALE_CODE,
    MDS_AGE_PRIOR,
    PRODROMAL_COL_TO_HEINZEL,
    PRODROMAL_VIABLE,
    RBD_ZSCORE_COL,
    SEX_COL,
)
from library.lr_analysis.lr_metrics import EmpiricalMarkerLR, LRResult


# ── Age-prior lookup ──────────────────────────────────────────────────────────

def _mds_prior_for_age(age: float, prior_table: dict[int, float]) -> float:
    """Linearly interpolate the age-based prior from a prior table.

    Below the lowest band: uses the lowest-band value.
    Above the highest band: uses the highest-band value.

    Parameters
    ----------
    age : float
    prior_table : dict[int, float]
        {lower_bound: probability}

    Returns
    -------
    float
        Prior probability of prodromal PD for this age.
    """
    if np.isnan(age):
        return float("nan")
    bands = sorted(prior_table.keys())
    if age <= bands[0]:
        return prior_table[bands[0]]
    if age >= bands[-1]:
        return prior_table[bands[-1]]
    for i in range(len(bands) - 1):
        lo, hi = bands[i], bands[i + 1]
        if lo <= age < hi:
            # Linear interpolation between two band midpoints
            frac = (age - lo) / (hi - lo)
            return prior_table[lo] + frac * (prior_table[hi] - prior_table[lo])
    return prior_table[bands[-1]]


# ── Single-subject Bayesian update ───────────────────────────────────────────

def _bayesian_update(
    prior_prob: float,
    lr_updates: list[float],
) -> float:
    """Apply sequential LR updates to a prior probability.

    posterior_odds = prior_odds × ∏ LR_i
    posterior_prob = posterior_odds / (1 + posterior_odds)

    Assumes conditional independence of markers (naive Bayes — same
    assumption as the MDS prodromal PD criteria).

    Parameters
    ----------
    prior_prob : float
        Prior probability of prodromal PD (0–1).
    lr_updates : list[float]
        Applicable LR values for each marker (already selected for
        marker-positive or marker-negative status).

    Returns
    -------
    float
        Posterior probability of prodromal PD.
    """
    if not np.isfinite(prior_prob) or prior_prob <= 0 or prior_prob >= 1:
        return float("nan")
    prior_odds = prior_prob / (1.0 - prior_prob)
    posterior_odds = prior_odds
    for lr in lr_updates:
        if np.isfinite(lr) and lr > 0:
            posterior_odds *= lr
    return posterior_odds / (1.0 + posterior_odds)


# ── C2: Hybrid (published MDS LRs + empirical actigraphy LR) ─────────────────

def compute_posterior_c2_hybrid(
    df: pd.DataFrame,
    actigraphy_lr_result: LRResult,
    rbd_zscore_col: str = RBD_ZSCORE_COL,
) -> pd.Series:
    """Compute per-subject posterior probability — Option C2 (Hybrid).

    Prior: MDS Berg 2015 age-based prior (``MDS_AGE_PRIOR``).
    Prodromal LRs: published Heinzel 2019 values.
    Actigraphy LR: empirical from ``actigraphy_lr_result``.

    Marker application rules:
    - Sex LR: always applied (male or female version).
    - Constipation, depression, orthostatic: applied if marker == 1 (LR+)
      or == 0 (LR-).
    - Erectile dysfunction: only applied to males.
    - Actigraphy: LR+ if rbd_zscore >= threshold, LR- otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-set rows.
    actigraphy_lr_result : LRResult
        LR+/LR- for the actigraphy z-score at the chosen threshold.
    rbd_zscore_col : str

    Returns
    -------
    pd.Series
        Per-subject posterior probability (index aligned to df).
    """
    thr = actigraphy_lr_result.threshold
    actig_lr_pos = actigraphy_lr_result.lr_pos
    actig_lr_neg = actigraphy_lr_result.lr_neg

    posteriors = np.full(len(df), float("nan"))

    for i, (idx, row) in enumerate(df.iterrows()):
        age = row.get(AGE_COL, float("nan"))
        sex = row.get(SEX_COL, float("nan"))
        prior = _mds_prior_for_age(float(age), MDS_AGE_PRIOR)

        lr_updates: list[float] = []

        # Sex LR
        if sex == MALE_CODE:
            lr_updates.append(HEINZEL_LRS["sex_male"].lr_pos)
        elif sex == FEMALE_CODE:
            lr_updates.append(HEINZEL_LRS["sex_female"].lr_pos)

        # Prodromal marker LRs (Heinzel published values)
        for col, heinzel_key in PRODROMAL_COL_TO_HEINZEL.items():
            if col not in df.columns:
                continue
            marker_val = row.get(col, float("nan"))
            if pd.isna(marker_val):
                continue
            mlr = HEINZEL_LRS.get(heinzel_key)
            if mlr is None:
                continue
            # Erectile dysfunction: males only
            if mlr.sex_restricted == "male" and sex != MALE_CODE:
                continue
            if int(marker_val) == 1:
                lr_updates.append(mlr.lr_pos)
            else:
                if mlr.lr_neg is not None:
                    lr_updates.append(mlr.lr_neg)

        # Actigraphy LR
        z = row.get(rbd_zscore_col, float("nan"))
        if pd.notna(z):
            actig_lr = actig_lr_pos if float(z) >= thr else actig_lr_neg
            if np.isfinite(actig_lr):
                lr_updates.append(actig_lr)

        posteriors[i] = _bayesian_update(prior, lr_updates)

    return pd.Series(posteriors, index=df.index, name="posterior_c2")


# ── C1: Empirical (UKBB-derived prior + empirical LRs) ───────────────────────

def compute_posterior_c1_empirical(
    df: pd.DataFrame,
    ukbb_age_prior: dict[int, float],
    empirical_marker_lrs: list[EmpiricalMarkerLR],
    actigraphy_lr_result: LRResult,
    rbd_zscore_col: str = RBD_ZSCORE_COL,
) -> pd.Series:
    """Compute per-subject posterior probability — Option C1 (Empirical).

    Prior: UKBB age-band incidence proportion (from ``data_prep``).
    Prodromal LRs: empirically computed from UKBB data (only viable markers).
    Actigraphy LR: empirical from ``actigraphy_lr_result``.

    Erectile dysfunction applied to males only (consistent with C2).

    Parameters
    ----------
    df : pd.DataFrame
    ukbb_age_prior : dict[int, float]
        Age-band incidence proportions from UKBB cohort.
    empirical_marker_lrs : list[EmpiricalMarkerLR]
        Empirically computed LRs for viable prodromal markers.
    actigraphy_lr_result : LRResult
        LR at the chosen z-score threshold.
    rbd_zscore_col : str

    Returns
    -------
    pd.Series
        Per-subject posterior probability.
    """
    from library.lr_analysis.config import PRODROMAL_LABELS

    thr = actigraphy_lr_result.threshold
    actig_lr_pos = actigraphy_lr_result.lr_pos
    actig_lr_neg = actigraphy_lr_result.lr_neg

    # Index empirical LRs by column name for fast lookup
    emp_lr_by_col = {e.col: e for e in empirical_marker_lrs}

    # Columns that are male-only (matches C2 convention)
    male_only_cols = {"prodromal_erectile_dysfunction_bl"}

    posteriors = np.full(len(df), float("nan"))

    for i, (idx, row) in enumerate(df.iterrows()):
        age = row.get(AGE_COL, float("nan"))
        sex = row.get(SEX_COL, float("nan"))
        prior = _mds_prior_for_age(float(age), ukbb_age_prior)

        lr_updates: list[float] = []

        # Prodromal marker LRs (empirical)
        for col, emp_lr in emp_lr_by_col.items():
            marker_val = row.get(col, float("nan"))
            if pd.isna(marker_val):
                continue
            if col in male_only_cols and sex != MALE_CODE:
                continue
            if int(marker_val) == 1:
                lr = emp_lr.lr_pos
            else:
                lr = emp_lr.lr_neg
            if np.isfinite(lr) and lr > 0:
                lr_updates.append(lr)

        # Actigraphy LR
        z = row.get(rbd_zscore_col, float("nan"))
        if pd.notna(z):
            actig_lr = actig_lr_pos if float(z) >= thr else actig_lr_neg
            if np.isfinite(actig_lr):
                lr_updates.append(actig_lr)

        posteriors[i] = _bayesian_update(prior, lr_updates)

    return pd.Series(posteriors, index=df.index, name="posterior_c1")


# ── Summary statistics ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PosteriorSummary:
    """Descriptive statistics of posterior probability by case/control status."""

    framework: str      # "C1_empirical" or "C2_hybrid"
    stratum: str        # "cases" or "controls"
    n: int
    mean: float
    sd: float
    median: float
    q25: float
    q75: float
    pct_above_1pct: float   # % subjects with posterior > 1%
    pct_above_5pct: float   # % subjects with posterior > 5%


def summarise_posteriors(
    posterior: pd.Series,
    is_case: pd.Series,
    framework: str,
) -> list[PosteriorSummary]:
    """Compute summary statistics for cases and controls separately.

    Parameters
    ----------
    posterior : pd.Series
        Per-subject posterior probabilities.
    is_case : pd.Series[bool]
    framework : str
        Label identifying the Bayesian framework (e.g., "C2_hybrid").

    Returns
    -------
    list[PosteriorSummary]
        Two entries: [cases, controls].
    """
    summaries = []
    for label, mask in [("cases", is_case), ("controls", ~is_case)]:
        vals = posterior[mask].dropna().values
        if len(vals) == 0:
            continue
        summaries.append(PosteriorSummary(
            framework=framework,
            stratum=label,
            n=len(vals),
            mean=float(np.mean(vals)),
            sd=float(np.std(vals, ddof=1)),
            median=float(np.median(vals)),
            q25=float(np.percentile(vals, 25)),
            q75=float(np.percentile(vals, 75)),
            pct_above_1pct=float(np.mean(vals > 0.01) * 100),
            pct_above_5pct=float(np.mean(vals > 0.05) * 100),
        ))
    return summaries
