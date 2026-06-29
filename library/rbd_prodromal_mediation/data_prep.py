"""
Data preparation for mediation analysis.

Loads the cohort, applies temporal filtering for cognitive markers,
creates RBD mediator encodings (z-scored continuous, binary high,
3-group categorical), z-scores continuous prodromals, and applies
minimum prevalence filters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from library.cox_prodromal.cox_config import (
    BASE_COVARIATES,
    PRODROMAL_BINARY_VARS,
)
from library.cox_prodromal.data_prep import (
    build_extended_covariates,
    build_survival_dataset_for_outcome,
    filter_active_variables,
    load_prodromal_dataset,
)
from library.rbd_prodromal_mediation.config import (
    ALL_MEDIATION_VARS,
    MIN_PREVALENCE_FOR_BINARY,
    PRIMARY_OUTCOME,
    RBD_3G_PERCENTILES,
    RBD_HIGH_PERCENTILE,
)


@dataclass(frozen=True)
class MediationCohort:
    """Immutable container for the prepared mediation dataset."""
    df: pd.DataFrame
    extended_covariates: List[str]
    active_vars: Dict[str, str]
    rbd_mean: float
    rbd_std: float
    zscore_params: Dict[str, Tuple[float, float]]
    temporal_filter_log: Optional[pd.DataFrame]
    n_total: int
    n_events: int


def load_mediation_cohort(
    outcome: str = PRIMARY_OUTCOME,
) -> MediationCohort:
    """
    Load, clean, and prepare the analytic cohort for mediation analysis.

    Steps:
      1. Load production cohort via load_prodromal_dataset()
      2. Build extended covariates (age, sex, BMI, smoking, alcohol)
      3. Build survival dataset for the target outcome
      4. Apply temporal filter for cognitive markers
      5. Create RBD mediator encodings (z-score, binary, 3-group)
      6. Z-score continuous prodromal markers
      7. Apply minimum prevalence filter

    Parameters
    ----------
    outcome : str
        Outcome for survival analysis (default: outcome_1a_pd_only).

    Returns
    -------
    MediationCohort
        Frozen dataclass with df, covariates, active variables, and metadata.
    """
    # 1. Load
    print("[Mediation] Loading cohort ...")
    _thresholds, df_risk = load_prodromal_dataset()
    print(f"  Loaded: {df_risk.shape[0]:,} subjects")

    # 2. Covariates
    df_risk, extended_covariates = build_extended_covariates(df_risk, list(BASE_COVARIATES))

    # 3. Survival dataset
    all_vars = dict(ALL_MEDIATION_VARS)
    active_vars = filter_active_variables(df_risk, all_vars)

    df_surv = build_survival_dataset_for_outcome(
        df_risk, outcome, active_vars, extended_covariates,
    )
    if df_surv is None or df_surv.empty:
        raise ValueError(f"No survival data for {outcome}")

    print(f"  Survival dataset: N={len(df_surv):,}, events={int(df_surv['event'].sum()):,}")

    # 4. Temporal filter for cognitive markers (visit dates)
    temporal_log = _apply_temporal_filter(df_surv, active_vars)

    # 5. RBD mediator encodings
    df_surv, rbd_mean, rbd_std = _create_rbd_encodings(df_surv)

    # 6. Z-score continuous prodromals
    df_surv, zscore_params = _zscore_continuous_prodromals(df_surv, active_vars)

    # 7. Minimum prevalence filter (post all exclusions)
    active_vars = _prevalence_filter(df_surv, active_vars)

    n_total = len(df_surv)
    n_events = int(df_surv["event"].sum())
    print(f"  Final analytic cohort: N={n_total:,}, events={n_events:,}, "
          f"active vars={len(active_vars)}")

    return MediationCohort(
        df=df_surv,
        extended_covariates=extended_covariates,
        active_vars=active_vars,
        rbd_mean=rbd_mean,
        rbd_std=rbd_std,
        zscore_params=zscore_params,
        temporal_filter_log=temporal_log,
        n_total=n_total,
        n_events=n_events,
    )


def _apply_temporal_filter(
    df: pd.DataFrame,
    active_vars: Dict[str, str],
) -> Optional[pd.DataFrame]:
    """
    Null cognitive marker values measured after actigraphy baseline.

    For each cognitive variable with a visit index suffix (_i0, _i2),
    check if the corresponding follow-up date is after wear_time_start.
    If so, set the cognitive value to NaN (measurement post-baseline
    cannot be used as a pre-exposure covariate).

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset (modified in-place for performance on large N).
    active_vars : dict
        Active variable mapping.

    Returns
    -------
    pd.DataFrame or None
        Log of temporal filtering per variable, or None if no filtering done.
    """
    if "wear_time_start" not in df.columns:
        return None

    wear_start = pd.to_datetime(df["wear_time_start"], errors="coerce")
    log_rows: List[dict] = []
    cognitive_vars = {k: v for k, v in active_vars.items() if k not in PRODROMAL_BINARY_VARS}

    for col, label in cognitive_vars.items():
        # Extract visit index from column name suffix
        visit_idx = _extract_visit_index(col)
        if visit_idx is None:
            log_rows.append({"variable": col, "label": label,
                             "visit_index": "unknown", "n_nulled": 0,
                             "n_retained": int(df[col].notna().sum()),
                             "pct_loss": 0.0})
            continue

        # Look for follow-up date column
        date_col = f"follow_up_date_i{visit_idx}"
        if date_col not in df.columns:
            log_rows.append({"variable": col, "label": label,
                             "visit_index": visit_idx, "n_nulled": 0,
                             "n_retained": int(df[col].notna().sum()),
                             "pct_loss": 0.0})
            continue

        visit_date = pd.to_datetime(df[date_col], errors="coerce")

        # ── Diagnostic: date overlap statistics (printed before nulling) ──────
        has_both = visit_date.notna() & wear_start.notna()
        n_with_dates = int(has_both.sum())
        n_before = int(df[col].notna().sum())
        if n_with_dates > 0:
            post_baseline_mask = (visit_date > wear_start) & has_both
            n_post = int(post_baseline_mask.sum())
            pct_post = round(n_post / n_with_dates * 100, 1)
            visit_min = visit_date[has_both].min().date()
            visit_max = visit_date[has_both].max().date()
            wear_min = wear_start[has_both].min().date()
            wear_max = wear_start[has_both].max().date()
            wear_median = wear_start[has_both].quantile(0.5)
            # Check: any visit dates BEFORE wear baseline (pre-baseline)
            n_pre = int(((visit_date <= wear_start) & has_both).sum())
            print(
                f"  [DATE CHECK] {col}  |  "
                f"visit {date_col}: [{visit_min} → {visit_max}]  |  "
                f"wear_time_start: [{wear_min} → {wear_max}] "
                f"(median {pd.Timestamp(wear_median).date()})  |  "
                f"post-baseline: {n_post}/{n_with_dates} ({pct_post}%)  |  "
                f"pre-baseline: {n_pre}/{n_with_dates}  |  "
                f"non-null in col: {n_before}"
            )
        else:
            print(f"  [DATE CHECK] {col}: no subjects with both dates available")

        # Null values where visit occurred after actigraphy baseline
        post_baseline = visit_date > wear_start
        df.loc[post_baseline & df[col].notna(), col] = np.nan
        n_after = int(df[col].notna().sum())
        n_nulled = n_before - n_after
        pct_loss = round(n_nulled / max(n_before, 1) * 100, 2)

        log_rows.append({
            "variable": col, "label": label,
            "visit_index": visit_idx,
            "n_nulled": n_nulled,
            "n_retained": n_after,
            "pct_loss": pct_loss,
        })
        if n_nulled > 0:
            print(f"  Temporal filter [{col}]: nulled {n_nulled} "
                  f"({pct_loss:.1f}%), retained {n_after}")

    return pd.DataFrame(log_rows) if log_rows else None


def _extract_visit_index(col_name: str) -> Optional[int]:
    """
    Extract visit index from column name suffix (_i0 -> 0, _i2 -> 2).

    Parameters
    ----------
    col_name : str
        Column name ending with _iN pattern.

    Returns
    -------
    int or None
        Visit index, or None if pattern not found.
    """
    parts = col_name.rsplit("_i", maxsplit=1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _create_rbd_encodings(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Create RBD mediator variables: z-scored continuous, binary high, 3-group.

    All thresholds computed on the post-exclusion analytic cohort.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'abk_rbd_score_mean' column.

    Returns
    -------
    tuple[pd.DataFrame, float, float]
        (updated df, rbd_mean, rbd_std)
    """
    rbd = df["abk_rbd_score_mean"].astype(float)
    rbd_mean = float(rbd.mean())
    rbd_std = float(rbd.std())

    # Z-scored continuous
    df["rbd_score_z"] = (rbd - rbd_mean) / rbd_std

    # Binary high (>= 99th percentile)
    p99 = float(np.percentile(rbd.dropna(), RBD_HIGH_PERCENTILE))
    df["rbd_high_binary"] = (rbd >= p99).astype(int)
    n_high = int(df["rbd_high_binary"].sum())
    print(f"  RBD binary high (>= p{RBD_HIGH_PERCENTILE:.0f} = {p99:.4f}): "
          f"N={n_high} ({n_high / len(df) * 100:.1f}%)")

    # 3-group categorical
    p90 = float(np.percentile(rbd.dropna(), RBD_3G_PERCENTILES[0]))
    df["rbd_group_3g"] = pd.cut(
        rbd,
        bins=[-np.inf, p90, p99, np.inf],
        labels=["Low", "Intermediate", "High"],
    )
    grp_counts = df["rbd_group_3g"].value_counts()
    print(f"  RBD 3-group: {dict(grp_counts)}")

    return df, rbd_mean, rbd_std


def _zscore_continuous_prodromals(
    df: pd.DataFrame,
    active_vars: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Z-score continuous prodromal variables in-place.

    Binary variables are left unchanged. Trail making: no sign reversal
    (positive beta = more errors = higher RBD).

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    active_vars : dict
        Active variable mapping.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (updated df, {col: (mean, std)} for reproducibility)
    """
    zscore_params: Dict[str, Tuple[float, float]] = {}

    for col in active_vars:
        if col in PRODROMAL_BINARY_VARS:
            continue
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors="coerce")
        mu = float(vals.mean())
        sigma = float(vals.std())
        if sigma < 1e-10:
            print(f"  WARNING: {col} has near-zero variance, skipping z-score")
            continue

        df[col] = (vals - mu) / sigma
        zscore_params[col] = (mu, sigma)
        print(f"  Z-scored {col}: mean={mu:.3f}, std={sigma:.3f}")

    return df, zscore_params


def _prevalence_filter(
    df: pd.DataFrame,
    active_vars: Dict[str, str],
) -> Dict[str, str]:
    """
    Remove variables failing minimum prevalence/non-null checks.

    Parameters
    ----------
    df : pd.DataFrame
        Post-temporal-filter survival dataset.
    active_vars : dict
        Current active variable mapping.

    Returns
    -------
    dict
        Filtered active variables.
    """
    filtered: Dict[str, str] = {}
    for col, label in active_vars.items():
        if col not in df.columns:
            continue
        if col in PRODROMAL_BINARY_VARS:
            n_pos = int(df[col].sum())
            if n_pos < MIN_PREVALENCE_FOR_BINARY:
                print(f"  SKIP {label}: {n_pos} positive < {MIN_PREVALENCE_FOR_BINARY}")
                continue
        else:
            n_valid = int(df[col].notna().sum())
            if n_valid < MIN_PREVALENCE_FOR_BINARY:
                print(f"  SKIP {label}: {n_valid} non-null < {MIN_PREVALENCE_FOR_BINARY}")
                continue
        filtered[col] = label
    return filtered
