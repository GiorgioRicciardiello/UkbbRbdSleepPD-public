"""
dataset.py
==========

Convert ``df_risk`` (the long survival/Cox source frame) into a flat
cross-sectional table suitable for binary classification.

Cross-sectional convention
--------------------------
* One row per subject (already enforced upstream by ``make_subject_level``).
* Outcome ``y in {0, 1}`` from ``outcome_1a_pd_only`` (boolean).
* Feature selection is configurable via ``feature_set`` parameter
  (e.g., "rbd_alone", "rbd_prs", "rbd_prodromal", "rbd_prs_prodromal").
  See ``feature_sets.py`` for available options.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from .feature_sets import FEATURE_SETS, build_feature_list
from .outcomes import DEFAULT_OUTCOME, OUTCOMES, get_outcome_config

# --- Column registry ---------------------------------------------------------

#: Default outcome used if none specified.
#: Change to another outcome name from outcomes.py (e.g., "pd_ad", "ad_only").
DEFAULT_OUTCOME_NAME: Final[str] = DEFAULT_OUTCOME

#: Wear start date column used downstream by ``features.build_time_to_event``.
WEAR_START_COL: Final[str] = "wear_time_start"

#: Demographics + genetics + RBD score that we keep as model features.
KEEP_FEATURES: Final[tuple[str, ...]] = (
    "cov_age_recruitment_21022",  # age at recruitment (years)
    "cov_sex_31",                  # 0 = female, 1 = male (UKBB encoding)
    "bmi_21001_bl",                # baseline BMI (visit i0; ~0.2% missing)
    "abk_rbd_score_mean",          # mean RBD probability across nights
    "prs_score_pd",
    "prs_score_rbd",
    "prs_pc1", "prs_pc2", "prs_pc3", "prs_pc4", "prs_pc5",
    "prs_pc6", "prs_pc7", "prs_pc8", "prs_pc9", "prs_pc10",
)

#: Explicit prodromal marker column names (baseline prevalence flags, _bl).
#: These are binary HES-derived features indicating presence/absence of prodromal
#: symptoms before the actigraphy baseline.
PRODROMAL_FEATURES: Final[tuple[str, ...]] = (
    "prodromal_constipation_bl",
    "prodromal_orthostatic_bl",
    "prodromal_depression_bl",
    "prodromal_erectile_dysfunction_bl",
    "prodromal_anosmia_bl",
    "prodromal_hyposmia_bl",
    "prodromal_anxiety_bl",
    "prodromal_dream_enactment_bl",
)


# --- Result type -------------------------------------------------------------

@dataclass(frozen=True)
class CrossSectionalFrame:
    """Container for the cross-sectional dataframe and the columns we kept."""

    df: pd.DataFrame
    feature_cols: tuple[str, ...]
    outcome_col: str
    wear_start_col: str
    event_date_col: str
    # P1 Combined auxiliary columns — empty string if absent in the source data.
    incident_col: str = ""
    prevalent_col: str = ""
    control_col: str = "control"


# --- Public API --------------------------------------------------------------

def _drop_prodromal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with all explicit prodromal columns removed."""
    drop_cols = [c for c in df.columns if c in PRODROMAL_FEATURES]
    if drop_cols:
        return df.drop(columns=drop_cols)
    return df


def convert_to_cross_sectional(
    df: pd.DataFrame,
    outcome_name: str | None = None,
    wear_start_col: str = WEAR_START_COL,
    feature_set: str | None = None,
) -> CrossSectionalFrame:
    """
    Build a cross-sectional dataframe from ``df_risk``.

    Parameters
    ----------
    df :
        The full risk dataframe returned by ``get_clean_risk_data``.
    outcome_name :
        Name of the outcome (e.g., "pd_only", "pd_ad", "ad_only").
        If ``None``, uses the default outcome from outcomes.py.
        See outcomes.list_outcomes() for available options.
    wear_start_col :
        Date column retained on the output frame for time-to-event computation.
    feature_set :
        Name of feature set config (e.g., "rbd_alone", "rbd_prs", etc.).
        If ``None``, uses ``KEEP_FEATURES`` (backward compat).

    Returns
    -------
    CrossSectionalFrame
        Wrapper holding the trimmed dataframe and the names of the kept
        feature columns. ``df`` contains exactly:
        ``feature_cols + [outcome_col, wear_start_col, event_date_col]``
        plus an ``eid`` column if present in the source.

    Notes
    -----
    * Subjects with a missing outcome value are dropped.
    * Prodromal markers are conditionally dropped based on feature_set config.
    * Date columns are coerced to ``datetime64[ns]``.
    * Outcome is specified by name (e.g., "pd_only") and resolved via outcomes.py.
    """
    # Resolve outcome configuration
    if outcome_name is None:
        outcome_name = DEFAULT_OUTCOME_NAME
    outcome_cfg = get_outcome_config(outcome_name)
    outcome_col = outcome_cfg.outcome_col
    event_date_col = outcome_cfg.event_date_col

    # Derive P1 Combined auxiliary columns from the outcome column name.
    # Naming convention: {outcome_col}__incident / {outcome_col}__prevalent.
    _incident_col = f"{outcome_col}__incident"
    _prevalent_col = f"{outcome_col}__prevalent"
    _control_col = "control"

    if outcome_col not in df.columns:
        raise KeyError(f"Outcome column not found in df: {outcome_col!r}")
    for c in (wear_start_col, event_date_col):
        if c not in df.columns:
            raise KeyError(f"Required date column missing: {c!r}")

    work = df.copy()

    # Determine feature columns based on feature_set or use default.
    if feature_set:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature_set: {feature_set!r}")
        fs_cfg = FEATURE_SETS[feature_set]
        feature_cols = build_feature_list(fs_cfg)
        # Drop any prodromal columns NOT in the feature_set's explicit prodromal list.
        # This ensures we only keep explicitly requested prodromal features.
        requested_prodromal = set(fs_cfg.get("prodromal", []))
        drop_cols = [c for c in PRODROMAL_FEATURES if c not in requested_prodromal]
        if drop_cols:
            work = work.drop(columns=[c for c in drop_cols if c in work.columns])
    else:
        # Backward compat: use KEEP_FEATURES and drop all prodromal.
        feature_cols = KEEP_FEATURES
        work = _drop_prodromal_columns(work)

    # Verify the feature columns exist; missing ones become a hard error.
    missing = [c for c in feature_cols if c not in work.columns]
    if missing:
        raise KeyError(
            f"Feature set {feature_set!r} requires columns not in source df: {missing}"
        )

    # Always include P1 auxiliary columns when present (not part of feature_cols).
    aux_cols = [c for c in [_incident_col, _prevalent_col, _control_col] if c in work.columns]
    keep = list(feature_cols) + [outcome_col, wear_start_col, event_date_col] + aux_cols
    if "eid" in work.columns:
        keep = ["eid"] + keep

    out = work.loc[:, keep].copy()

    # Cast outcome to int {0, 1} and drop NaN outcomes.
    out = out[out[outcome_col].notna()].copy()
    out[outcome_col] = out[outcome_col].astype(int)

    # Coerce date columns.
    out[wear_start_col] = pd.to_datetime(out[wear_start_col], errors="coerce")
    out[event_date_col] = pd.to_datetime(out[event_date_col], errors="coerce")

    # Record which auxiliary columns were found.
    found_incident = _incident_col if _incident_col in out.columns else ""
    found_prevalent = _prevalent_col if _prevalent_col in out.columns else ""

    return CrossSectionalFrame(
        df=out.reset_index(drop=True),
        feature_cols=feature_cols,
        outcome_col=outcome_col,
        wear_start_col=wear_start_col,
        event_date_col=event_date_col,
        incident_col=found_incident,
        prevalent_col=found_prevalent,
        control_col=_control_col if _control_col in out.columns else "",
    )
