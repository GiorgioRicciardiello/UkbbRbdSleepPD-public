"""
Centralized column name registry.

Single source of truth for all derived column names in the UKBB RBD/Sleep/PD
pipeline.  All downstream code MUST use these functions instead of raw
f-strings to construct outcome-related column names.

Naming convention
-----------------
``{outcome}__{role}``

- ``__`` (double underscore) separates the outcome prefix from the column role.
- Single underscore within each segment (e.g. ``surv_days``, ``rg_pctl2``).

Survival columns (outcome-specific)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``__dx``            bool   — diagnosed with a recorded date
- ``__prevalent``     bool   — diagnosed before baseline (wear_time_start)
- ``__incident``      bool   — diagnosed during follow-up [start, censor]
- ``__competing``     bool   — died during follow-up without the outcome
- ``__tte_days``      float  — days from baseline to dx; NaN for prevalent
- ``__surv_days``     float  — survival time in days; NaN for prevalent
- ``__surv_event``    int    — 0=censored, 1=incident, 2=competing death

Risk-group columns (outcome-agnostic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``rg_pctl2``   — percentile 2-group (Low / High)
- ``rg_pctl3``   — percentile 3-group (Low / Mid / High)
- ``rg_q4``      — quartile-based (Q1–Q4)
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import pandas as pd

# ── Separator ─────────────────────────────────────────────────────────────
SEP: str = "__"

# ── Outcome identifiers ──────────────────────────────────────────────────
# Sourced from config.config.outcomes (single source of truth).
# Import deferred to avoid circular dependency; callers should prefer
# importing `outcomes` directly from config.config.
OUTCOMES: List[str] = [
    "outcome_1a_pd_only",
    "outcome_1b_pd_ad",
    "outcome_2a_vasculardementia",
    "outcome_2b_pd_vasculardementia",
    "outcome_4a_ad_only",
]

# ── Risk method -> column suffix mapping ──────────────────────────────────
METHOD_TO_RISK_SUFFIX: Dict[str, str] = {
    "percentile_2g": "rg_pctl2",
    "percentile_3g": "rg_pctl3",
    "roc":           "rg_roc",
    "pr":            "rg_pr",
    "f1":            "rg_f1",
    "surv":          "rg_surv",
    "quartile":      "rg_q4",
}

# ── Agnostic risk group columns ──────────────────────────────────────────
AGNOSTIC_RISK_COLS: List[str] = ["rg_pctl2", "rg_pctl3", "rg_q4"]


# ── Column builders ──────────────────────────────────────────────────────

def col_dx(outcome: str) -> str:
    """Boolean: has diagnosis with a recorded date."""
    return f"{outcome}{SEP}dx"


def col_prevalent(outcome: str) -> str:
    """Boolean: diagnosed before baseline (wear_time_start)."""
    return f"{outcome}{SEP}prevalent"


def col_incident(outcome: str) -> str:
    """Boolean: diagnosed during follow-up [start, censor]."""
    return f"{outcome}{SEP}incident"


def col_competing(outcome: str) -> str:
    """Boolean: died during follow-up without the outcome."""
    return f"{outcome}{SEP}competing"


def col_tte_days(outcome: str) -> str:
    """Days from baseline to diagnosis.  NaN for prevalent cases."""
    return f"{outcome}{SEP}tte_days"


def col_surv_time(outcome: str) -> str:
    """Survival time in days.  NaN for prevalent cases (excluded from analysis)."""
    return f"{outcome}{SEP}surv_days"


def col_surv_event(outcome: str) -> str:
    """Survival event indicator: 0=censored, 1=incident, 2=competing death."""
    return f"{outcome}{SEP}surv_event"


def col_outcome_date(outcome: str) -> str:
    """Earliest diagnosis date for a composite outcome."""
    return f"{outcome}_date"


def col_risk_group_agnostic(method: str) -> str:
    """Outcome-agnostic risk-group column name.

    Parameters
    ----------
    method : str
        Thresholding method key (e.g. ``'percentile_2g'``, ``'quartile'``).

    Returns
    -------
    str
        Column name, e.g. ``'rg_pctl2'``.
    """
    return METHOD_TO_RISK_SUFFIX[method]


def col_risk_group(outcome: str, method: str) -> str:
    """DEPRECATED: use col_risk_group_agnostic(method) instead.

    Risk groups are outcome-agnostic (same RBD probability thresholds
    regardless of outcome). This function is kept for backward compatibility.
    """
    warnings.warn(
        "col_risk_group() is deprecated. Use col_risk_group_agnostic(method) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    suffix = METHOD_TO_RISK_SUFFIX[method]
    return f"{outcome}{SEP}{suffix}"


# ── Parsing helpers ──────────────────────────────────────────────────────

def parse_outcome_from_col(col: str) -> str:
    """Extract the outcome prefix from a registry-generated column name.

    >>> parse_outcome_from_col('outcome_1a_pd_only__surv_days')
    'outcome_1a_pd_only'
    """
    if SEP not in col:
        raise ValueError(f"Column '{col}' does not contain separator '{SEP}'")
    return col.split(SEP)[0]


def is_risk_group_col(col: str) -> bool:
    """Check whether *col* is an agnostic risk-group column."""
    return col in AGNOSTIC_RISK_COLS


def get_risk_group_cols(columns: List[str]) -> List[str]:
    """Return all agnostic risk-group columns present in a list of column names."""
    return [c for c in columns if is_risk_group_col(c)]


# ── Legacy compatibility ─────────────────────────────────────────────────

_LEGACY_RISK_SUFFIX: Dict[str, str] = {
    "percentile_2g": "risk_group_mean_2g",
    "percentile_3g": "risk_group_mean_3g",
    "roc":           "risk_roc",
    "pr":            "risk_pr",
    "f1":            "risk_f1",
    "surv":          "risk_survival",
    "quartile":      "risk_quartiles",
}

# Old agnostic column names from run_compute_risk_group_rbd_only()
_LEGACY_AGNOSTIC: Dict[str, str] = {
    "rbd_risk_group_p90_2g":      "rg_pctl2",
    "rbd_risk_group_p90_p99_3g":  "rg_pctl3",
    "rbd_risk_group_quartiles":   "rg_q4",
}


def _build_legacy_map() -> Dict[str, str]:
    """Build old -> new column name mapping for migration."""
    m: Dict[str, str] = {}
    for oc in OUTCOMES:
        m[f"{oc}_diagnosed"]  = col_dx(oc)
        m[f"{oc}_prevalent"]  = col_prevalent(oc)
        m[f"{oc}_incident"]   = col_incident(oc)
        m[f"{oc}_competing"]  = col_competing(oc)
        m[f"{oc}_tte_days"]   = col_tte_days(oc)
        m[f"{oc}_surv_time"]  = col_surv_time(oc)
        m[f"{oc}_surv_event"] = col_surv_event(oc)
        # Per-outcome risk group columns (old f-string style)
        for method, old_suffix in _LEGACY_RISK_SUFFIX.items():
            agnostic_col = col_risk_group_agnostic(method)
            m[f"{oc}_{old_suffix}"] = agnostic_col
        # Per-outcome risk group columns (__ separator style)
        for method in METHOD_TO_RISK_SUFFIX:
            suffix = METHOD_TO_RISK_SUFFIX[method]
            agnostic_col = col_risk_group_agnostic(method)
            m[f"{oc}{SEP}{suffix}"] = agnostic_col

    # Old agnostic column names
    m.update(_LEGACY_AGNOSTIC)
    return m


LEGACY_TO_NEW: Dict[str, str] = _build_legacy_map()


def rename_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename old-style columns to the new convention.

    Returns a new DataFrame (non-destructive).  Columns not in the
    legacy map are left unchanged.
    """
    renames = {old: new for old, new in LEGACY_TO_NEW.items() if old in df.columns}
    if not renames:
        return df
    df = df.rename(columns=renames)
    # Multiple per-outcome columns may map to the same agnostic name (e.g. all 6
    # outcome__rg_pctl2 columns → rg_pctl2).  Since risk groups are outcome-agnostic
    # these duplicates are identical; keep only the first occurrence.
    return df.loc[:, ~df.columns.duplicated(keep="first")]
