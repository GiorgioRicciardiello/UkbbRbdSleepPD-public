"""
Data loading, cleaning, covariate construction, and survival dataset preparation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from config.config import config
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level
from library.cox_analysis.survival_dataset import select_survival_dataset
from library.ehr_outcomes.age_groups import create_age_groups
from library.ehr_outcomes.covariates import select_tmt_baseline  # canonical definition
from library.column_registry import (
    col_incident, col_surv_time, col_surv_event,
    AGNOSTIC_RISK_COLS,
)

from library.cox_prodromal.cox_config import (
    BASE_COVARIATES,
    GBA_COL,
    HES_GAP_COL,
    HES_GAP_THRESHOLD_YEARS,
    PRS_COLS,
    PRODROMAL_BINARY_VARS,
    MIN_PREVALENCE_FOR_BINARY,
    SMOKING_CANDIDATES,
    ALCOHOL_CANDIDATES,
)


# ── Path helpers ────────────────────────────────────────────────────────────

def insert_after_data(path: Path, folder: str) -> Path:
    """Insert a subfolder after the 'data' component in a path."""
    parts = list(path.parts)
    if "data" not in parts:
        raise ValueError(f"'data' not found in path: {path}")
    idx = parts.index("data") + 1
    return Path(*parts[:idx], folder, *parts[idx:])


# ── Data loading ────────────────────────────────────────────────────────────

def load_prodromal_dataset(
    file_name: str = "ehr_diag_pd_rbd_only_all",
) -> Tuple[dict, pd.DataFrame]:
    """
    Load and clean the production cohort dataset (ABK model).

    Reads from the canonical final directories defined in config.
    ABK outputs are promoted there by
    ``run_merge_ukbb_rbd.py::promote_abk_to_final()``.

    Parameters
    ----------
    file_name : str
        Base name of the parquet / threshold collection.

    Returns
    -------
    thresholds : dict
        Nested dict of risk thresholds by method and outcome.
    df : pd.DataFrame
        Subject-level DataFrame with ``abk_rbd_score_mean``, covariates,
        survival columns, and risk group columns.
    """
    dir_final = config["pp"]["final_dir"]
    dir_thresh = config["pp"]["thresholds"]["root"]

    thresholds, df_risk = get_clean_risk_data(
        file_name=file_name,
        thresholds_root=dir_thresh,
        final_dir=dir_final,
    )
    df_risk = make_subject_level(df_risk, id_col="eid", prob_col="abk_rbd_score_mean")
    # make_subject_level renames abk_rbd_score_mean → rbd_prob; restore the original
    # name so that runner.py column guards and carry-forward logic work correctly.
    # rbd_prob is kept for rbd_spline_analysis.py which uses that name.
    if "rbd_prob" in df_risk.columns and "abk_rbd_score_mean" not in df_risk.columns:
        df_risk["abk_rbd_score_mean"] = df_risk["rbd_prob"]

    # Normalize descriptive risk-group labels to simple Low/Mid/High.
    # Datasets built before the risk_groups.py pipeline fix carry the old
    # descriptive strings (e.g. "Low (0,90%)"). This mapping makes the
    # current run consistent without rebuilding the parquet.
    _LABEL_MAP: dict = {
        "Low (0,90%)":            "Low",
        "Intermediate (90,99%)":  "Mid",
        "High (99,100%)":         "High",
        "High (90,100%)":         "High",
    }
    from library.column_registry import AGNOSTIC_RISK_COLS
    for _col in AGNOSTIC_RISK_COLS:
        if _col in df_risk.columns:
            df_risk[_col] = df_risk[_col].map(
                lambda v, m=_LABEL_MAP: m.get(str(v), v) if pd.notna(v) else v
            )

    df_risk = create_age_groups(df=df_risk, age_col="cov_age_recruitment_21022")

    # ── TMT baseline ────────────────────────────────────────────────────────
    # tmt_*_baseline columns are produced at build time by select_tmt_baseline()
    # called from add_covariates() in library/ehr_outcomes/covariates.py.
    # Guard for datasets built before this change was deployed.
    if "tmt_ratio_baseline" not in df_risk.columns:
        df_risk = select_tmt_baseline(df_risk)
    if "cog_tmt_ratio_log_bl" not in df_risk.columns:
        # Built at dataset creation (add_covariates); fallback for pre-rebuild
        # datasets. log(ratio) ≥ 0 because ratio ≥ 1.0; NaN propagates for missing.
        df_risk["cog_tmt_ratio_log_bl"] = np.log(df_risk["tmt_ratio_baseline"])

    return thresholds, df_risk


# ── Covariate construction ──────────────────────────────────────────────────

def prepare_lifestyle_covariates(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build best-available smoking and alcohol columns from across visits.

    Prefer ``_bl`` (baseline); fall back to later visits for missing observations.
    Negative codes (e.g. -3 = prefer not to answer) are recoded to NaN.

    Returns
    -------
    df : pd.DataFrame
        Updated DataFrame with ``cov_smoking`` and/or ``cov_alcohol``.
    added : list[str]
        Names of columns with >0 non-null observations.
    """
    # cov_smoking / cov_alcohol are now built at dataset creation time by
    # prepare_lifestyle_covariates() called from add_covariates() in
    # library/ehr_outcomes/covariates.py.  If they already exist, report
    # availability and return without overwriting.
    already_built = [
        c for c in ("cov_smoking", "cov_alcohol") if c in df.columns
    ]
    if already_built:
        added: List[str] = []
        for out_col in ("cov_smoking", "cov_alcohol"):
            if out_col in df.columns:
                n_obs = int(df[out_col].notna().sum())
                pct   = n_obs / len(df) * 100
                print(
                    f"  Lifestyle covariate '{out_col}' already present: "
                    f"{n_obs:,} obs ({pct:.1f}%)"
                )
                if n_obs > 0:
                    added.append(out_col)
        return df, added

    df = df.copy()
    added = []

    for out_col, candidates in [
        ("cov_smoking", SMOKING_CANDIDATES),
        ("cov_alcohol", ALCOHOL_CANDIDATES),
    ]:
        series = pd.Series(np.nan, index=df.index, dtype=float)
        for col in candidates:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                vals = vals.where(vals >= 0, np.nan)
                series = series.fillna(vals)
        df[out_col] = series
        n_obs = int(df[out_col].notna().sum())
        pct = n_obs / len(df) * 100
        print(f"  Lifestyle covariate '{out_col}': {n_obs:,} obs ({pct:.1f}%)")
        if n_obs > 0:
            added.append(out_col)

    return df, added


def build_extended_covariates(
    df: pd.DataFrame,
    base_covariates: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Coerce base covariates to numeric and add lifestyle covariates.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with numeric covariates.
    extended : list[str]
        Full list of active covariates (base + lifestyle).
    """
    base_covariates = base_covariates or BASE_COVARIATES

    for var in base_covariates:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors="coerce")

    # Pairs status is sometimes numeric-encoded but stored as object
    pairs_col = "cog_pairs_matching_bl"
    if pairs_col in df.columns:
        df[pairs_col] = pd.to_numeric(df[pairs_col], errors="coerce")

    df, lifestyle_added = prepare_lifestyle_covariates(df)
    extended = list(base_covariates) + lifestyle_added
    print(f"  Active covariates ({len(extended)}): {extended}")
    return df, extended


# ── Discretisation ──────────────────────────────────────────────────────────

def categorize_continuous(
    series: pd.Series,
    n_quantiles: int = 3,
) -> pd.Series:
    """
    Discretize a continuous variable into quantile-based groups.

    Falls back progressively: tertiles -> median split if quantile
    computation fails (e.g. due to ties).

    Parameters
    ----------
    series : pd.Series
        Continuous numeric values.
    n_quantiles : int
        Target number of groups (3 = Low/Medium/High).

    Returns
    -------
    pd.Series
        Categorical labels.
    """
    series = pd.to_numeric(series, errors="coerce")
    if series.nunique() <= 2:
        return series.astype(str)
    labels_map = {3: ["Low", "Medium", "High"], 2: ["Low", "High"]}
    for q in range(n_quantiles, 0, -1):
        try:
            labels = labels_map.get(q, [f"Q{i+1}" for i in range(q)])
            return pd.qcut(series, q, labels=labels, duplicates="drop")
        except ValueError:
            continue
    med = series.median()
    return series.apply(lambda x: "High" if pd.notna(x) and x > med else "Low")


def discretize_prodromal(
    df: pd.DataFrame,
    prod_var: str,
    binary_vars: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """
    Create a categorical prodromal group column.

    Binary prodromal variables -> Yes/No. Variables with <=3 unique values -> as-is.
    Continuous variables -> tertile discretisation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``prod_var``.
    prod_var : str
        Column name of the prodromal variable.
    binary_vars : dict, optional
        Mapping of binary prodromal variable names to labels.

    Returns
    -------
    pd.Series
        Categorical group labels aligned to ``df.index``.
    """
    binary_vars = binary_vars or PRODROMAL_BINARY_VARS
    if prod_var in binary_vars:
        return df[prod_var].apply(lambda x: "Yes" if x == 1 else "No")
    if df[prod_var].nunique() <= 3:
        return df[prod_var].astype(str)
    return categorize_continuous(df[prod_var]).astype(str)


# ── Lag filter ──────────────────────────────────────────────────────────────

def apply_lag_filter(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    lag_years: float,
) -> pd.DataFrame:
    """
    Remove subjects whose event occurs within ``lag_years`` of baseline.

    Addresses reverse-causality bias by excluding early events that
    may reflect subclinical disease already present at actigraphy.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with ``time_col`` (years) and ``event_col`` (0/1).
    time_col : str
        Duration column.
    event_col : str
        Event indicator column.
    lag_years : float
        Exclusion window in years.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the input.
    """
    mask = (df[event_col] == 0) | (df[time_col] > lag_years)
    n_removed = int((~mask).sum())
    print(
        f"    [Lag {lag_years:.0f}y] Removed {n_removed} early-event subjects; "
        f"remaining N = {int(mask.sum()):,}"
    )
    return df[mask].copy()


# ── Availability ────────────────────────────────────────────────────────────

def build_availability_table(
    df: pd.DataFrame,
    var_dict: Dict[str, str],
    hes_derived_vars: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Report variable availability (non-missing count and percentage).

    For HES-derived binary prodromal variables, also reports the percentage
    of available subjects whose HES gap is within the configured threshold
    (``HES_GAP_THRESHOLD_YEARS``).  This column quantifies how reliably the
    "unexposed" label can be trusted for each marker.

    Parameters
    ----------
    df : pd.DataFrame
        Full cohort DataFrame.
    var_dict : dict
        Mapping of column names to human-readable labels.
    hes_derived_vars : dict, optional
        Subset of ``var_dict`` that are HES-derived (used to populate the
        ``pct_hes_active`` column).  Defaults to ``PRODROMAL_BINARY_VARS``.

    Returns
    -------
    pd.DataFrame
        Columns: variable, label, n_available, pct_available, in_dataset,
        pct_hes_active (NaN for non-HES-derived variables).
    """
    hes_derived_vars = hes_derived_vars or PRODROMAL_BINARY_VARS
    hes_gap_ok = (
        df[HES_GAP_COL] <= HES_GAP_THRESHOLD_YEARS
        if HES_GAP_COL in df.columns
        else pd.Series(False, index=df.index)
    )

    n_total = len(df)
    rows = []
    for col, label in var_dict.items():
        if col not in df.columns:
            rows.append({
                "variable": col, "label": label,
                "n_available": 0, "pct_available": 0.0,
                "in_dataset": False,
                "pct_hes_active": np.nan,
            })
            continue

        available_mask = df[col].notna()
        n_avail = int(available_mask.sum())

        # For HES-derived vars: among available subjects, what fraction have
        # adequate HES coverage (gap ≤ threshold)?  This measures the quality
        # of the "unexposed" label — a low percentage means many subjects are
        # HES-unexposed but also have sparse hospital contact, so the label
        # cannot be verified as a true negative.
        if col in hes_derived_vars and HES_GAP_COL in df.columns:
            n_hes_active = int((available_mask & hes_gap_ok).sum())
            pct_hes_active = round(n_hes_active / n_avail * 100, 1) if n_avail > 0 else np.nan
        else:
            pct_hes_active = np.nan

        rows.append({
            "variable": col, "label": label,
            "n_available": n_avail,
            "pct_available": round(n_avail / n_total * 100, 2),
            "in_dataset": True,
            "pct_hes_active": pct_hes_active,
        })
    return pd.DataFrame(rows)


def filter_active_variables(
    df: pd.DataFrame,
    all_vars: Dict[str, str],
    binary_vars: Optional[Dict[str, str]] = None,
    min_prevalence_binary: int = MIN_PREVALENCE_FOR_BINARY,
) -> Dict[str, str]:
    """
    Return prodromal variables that pass availability and prevalence checks.

    Binary prodromal variables require at least ``min_prevalence_binary``
    positive cases.  All variables require at least one non-null observation.

    Parameters
    ----------
    df : pd.DataFrame
        Full cohort.
    all_vars : dict
        Combined cognitive + binary prodromal variable mapping.
    binary_vars : dict, optional
        Binary prodromal mapping (for prevalence check).
    min_prevalence_binary : int
        Minimum positive cases for binary prodromal variables.

    Returns
    -------
    dict
        {column_name: label} for variables passing all checks.
    """
    binary_vars = binary_vars or PRODROMAL_BINARY_VARS
    active: Dict[str, str] = {}
    for col, label in all_vars.items():
        if col not in df.columns:
            continue
        if col in binary_vars and df[col].sum() < min_prevalence_binary:
            print(
                f"  SKIP {label}: {int(df[col].sum())} cases "
                f"< {min_prevalence_binary}"
            )
            continue
        if df[col].notna().sum() == 0:
            continue
        active[col] = label
    print(f"  Active prodromal variables: {len(active)}")
    return active


# ── Survival dataset builder ────────────────────────────────────────────────

def build_survival_dataset_for_outcome(
    df: pd.DataFrame,
    outcome: str,
    active_vars: Dict[str, str],
    extended_covariates: List[str],
) -> Optional[pd.DataFrame]:
    """
    Prepare a single outcome's survival DataFrame.

    Steps:
    1. Filter to incident cases + controls.
    2. Call ``select_survival_dataset`` (drops prevalent, converts to years).
    3. Carry over prodromal variables, covariates, and risk group columns.

    Parameters
    ----------
    df : pd.DataFrame
        Full cohort with all columns.
    outcome : str
        Outcome identifier (e.g. 'outcome_1a_pd_only').
    active_vars : dict
        Active prodromal variables to carry forward.
    extended_covariates : list[str]
        Covariate columns to carry forward.

    Returns
    -------
    pd.DataFrame or None
        Survival-ready DataFrame with 'time' and 'event' columns,
        or None if the outcome data is unavailable.
    """
    incident_col = col_incident(outcome)
    if incident_col not in df.columns:
        print(f"  SKIP {outcome}: missing {incident_col}")
        return None

    # Filter
    df_cohort = df[
        df[incident_col].fillna(False).astype(bool)
        | df["control"].fillna(False).astype(bool)
    ].copy()

    if df_cohort.empty:
        return None

    df_surv = select_survival_dataset(
        df_cohort,
        outcome=outcome,
        time_unit="years",
        incident_col=incident_col,
    )

    # Carry over necessary columns not already present.
    # HES_GAP_COL must be included so the sensitivity analysis in runner.py
    # can filter to the HES-active subcohort; without it the filter is silently
    # skipped because the column is absent from df_surv / df_cc.
    # PRS columns must be included for Model F (RBD × PRS interaction).
    carry_cols = (
        list(active_vars.keys())
        + extended_covariates
        + [c for c in AGNOSTIC_RISK_COLS if c in df.columns]
        + ["abk_rbd_score_mean", HES_GAP_COL]
        + [c for c in PRS_COLS if c in df.columns]
        + ([GBA_COL] if GBA_COL in df.columns else [])
    )
    extra = {
        c: df_cohort.loc[df_surv.index, c].values
        for c in carry_cols
        if c not in df_surv.columns and c in df_cohort.columns
    }
    if extra:
        df_surv = df_surv.assign(**extra)

    # Also carry competing outcome event columns for Model 4
    for c in df.columns:
        if c.endswith("_surv_event") and c not in df_surv.columns:
            if c in df_cohort.columns:
                df_surv[c] = df_cohort.loc[df_surv.index, c].values
        if c.endswith("_surv_time") and c not in df_surv.columns:
            if c in df_cohort.columns:
                df_surv[c] = df_cohort.loc[df_surv.index, c].values

    # Derive death survival columns for competing risk analysis.
    # death_surv_time (days from baseline to death) is NaN for alive subjects,
    # which encode_competing_events correctly skips.
    if "death_flag" in df_cohort.columns and "death_date" in df_cohort.columns:
        df_surv["death_surv_event"] = (
            df_cohort.loc[df_surv.index, "death_flag"].astype(int).values
        )
        wear_start = pd.to_datetime(
            df_cohort.loc[df_surv.index, "wear_time_start"], errors="coerce"
        )
        death_dt = pd.to_datetime(
            df_cohort.loc[df_surv.index, "death_date"], errors="coerce"
        )
        df_surv["death_surv_time"] = (death_dt - wear_start).dt.days.astype(float)

    df_surv = df_surv.dropna(subset=["time", "event"])
    return df_surv
