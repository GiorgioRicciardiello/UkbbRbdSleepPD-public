"""
Data loading for screening model training.

Wraps the existing ``load_prodromal_dataset`` pipeline and builds the
ML-ready subject-level DataFrame with explicit incident/prevalent/control flags.

The DataFrame is column-slimmed to the analytical set (feature cols + label cols
+ source outcome columns) before being returned — the raw UKBB parquet has ~2k
columns, most of which are irrelevant for screening and waste memory.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from library.cox_prodromal.data_prep import load_prodromal_dataset, build_extended_covariates
from library.cox_prodromal.cox_config import BASE_COVARIATES
from library.screening.config import (
    ALL_FEATURE_COLS,
    CONTROL_COL,
    INCIDENT_COL,
    PREVALENT_COL,
    TMT_FEATURES,
    TMT_MISSING_FLAG,
)

logger = logging.getLogger(__name__)

# ── Derived label columns always kept ────────────────────────────────────────
_LABEL_COLS: List[str] = ["y_incident", "y_prevalent", "y_control", "y_case_any"]


def load_ml_dataset(
    incident_col: str = INCIDENT_COL,
    prevalent_col: str = PREVALENT_COL,
    control_col: str = CONTROL_COL,
    tmt_missing_flag: str = TMT_MISSING_FLAG,
    tmt_features: Optional[List[str]] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Load the cohort dataset and attach ML-specific label columns.

    Uses the canonical ``load_prodromal_dataset`` pipeline so that all
    upstream exclusions (train_sleep, neuro_exclude) are consistently applied.
    The returned DataFrame is slimmed to feature + label columns only.

    Parameters
    ----------
    incident_col : str
        Source column name for incident PD flag (default: ``INCIDENT_COL``).
    prevalent_col : str
        Source column name for prevalent PD flag (default: ``PREVALENT_COL``).
    control_col : str
        Source column name for control flag (default: ``CONTROL_COL``).
    tmt_missing_flag : str
        Name of the TMT missingness indicator column (default: ``TMT_MISSING_FLAG``).
    tmt_features : list[str], optional
        TMT ratio feature columns used to derive the missingness flag when absent.
        Defaults to ``TMT_FEATURES`` from config.

    Returns
    -------
    thresholds : dict
        Risk threshold collection (pass-through from upstream).
    df_ml : pd.DataFrame
        Subject-level DataFrame restricted to the analytical cohort with columns:
        - feature columns listed in ``ALL_FEATURE_COLS`` (present subset)
        - ``y_incident`` (bool): True for incident PD cases
        - ``y_prevalent`` (bool): True for prevalent PD cases
        - ``y_control``  (bool): True for controls
        - ``y_case_any`` (bool): True for incident OR prevalent PD
    """
    _tmt_features: List[str] = tmt_features if tmt_features is not None else TMT_FEATURES

    thresholds, df = load_prodromal_dataset()
    df, _ = build_extended_covariates(df, BASE_COVARIATES)

    # ── Log outcome column usage and value counts ─────────────────────────────
    for role, col in [("incident", incident_col),
                      ("prevalent", prevalent_col),
                      ("control", control_col)]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).to_dict()
            logger.info("Outcome column [%s] = '%s'  value_counts: %s", role, col, vc)
        else:
            logger.error("Outcome column [%s] = '%s' NOT FOUND in dataset.", role, col)

    # ── BMI covariate ─────────────────────────────────────────────────────────
    # cov_bmi is declared in BASE_COVARIATES but build_extended_covariates only
    # coerces existing columns — it does not create cov_bmi if absent.
    # Derive it from the primary UKBB BMI field (instance 0 preferred).
    if "cov_bmi" not in df.columns:
        # _bl = baseline visit (post-rename); legacy _i0 kept for pre-rebuild data.
        for _bmi_src in ("bmi_21001_bl", "bmi_21001_i0", "bmi_21001_i1",
                         "bmi_imp_23104_bl", "bmi_imp_23104_i0"):
            if _bmi_src in df.columns:
                df["cov_bmi"] = pd.to_numeric(df[_bmi_src], errors="coerce")
                logger.info("Derived cov_bmi from %s.", _bmi_src)
                break
        else:
            logger.warning("No BMI source column found; cov_bmi will be absent.")

    # ── TMT missingness indicator ─────────────────────────────────────────────
    # tmt_missing may already exist (built at dataset creation time).
    # If absent, derive it from the TMT ratio NaN pattern.
    if tmt_missing_flag not in df.columns:
        if _tmt_features and _tmt_features[0] in df.columns:
            df[tmt_missing_flag] = df[_tmt_features[0]].isna().astype(int)
            logger.info("Derived %s from %s NaN pattern.", tmt_missing_flag, _tmt_features[0])
        else:
            logger.warning(
                "Cannot derive %s: no TMT source column found.", tmt_missing_flag
            )
    else:
        df[tmt_missing_flag] = df[tmt_missing_flag].astype(int)

    # ── Label construction ────────────────────────────────────────────────────
    df["y_incident"] = df[incident_col].fillna(False).astype(bool)
    df["y_prevalent"] = df[prevalent_col].fillna(False).astype(bool)
    df["y_control"]  = df[control_col].fillna(False).astype(bool)
    df["y_case_any"] = df["y_incident"] | df["y_prevalent"]

    # ── Integrity check: case-control overlap ─────────────────────────────────
    overlap = (df["y_case_any"] & df["y_control"]).sum()
    if overlap > 0:
        logger.warning(
            "%d subjects flagged as both case and control — dropping control flag.", overlap
        )
        df.loc[df["y_case_any"], "y_control"] = False

    n_incident  = int(df["y_incident"].sum())
    n_prevalent = int(df["y_prevalent"].sum())
    n_controls  = int(df["y_control"].sum())
    n_total     = len(df)
    logger.info(
        "Dataset loaded: N=%d | incident=%d (%.2f%%) | prevalent=%d (%.2f%%) | controls=%d",
        n_total, n_incident, 100 * n_incident / n_total,
        n_prevalent, 100 * n_prevalent / n_total,
        n_controls,
    )

    # ── Restrict to analytical cohort ─────────────────────────────────────────
    keep_mask = df["y_incident"] | df["y_prevalent"] | df["y_control"]
    df_ml = df[keep_mask].copy()

    n_dropped = n_total - len(df_ml)
    if n_dropped > 0:
        logger.info(
            "Dropped %d subjects with no assigned role.", n_dropped
        )

    # ── Slim to required columns ──────────────────────────────────────────────
    # Keep only feature columns present in this dataset + label columns.
    # Drop the ~2k raw UKBB columns to reduce memory footprint.
    feature_cols_present = [c for c in ALL_FEATURE_COLS if c in df_ml.columns]
    keep_cols = feature_cols_present + _LABEL_COLS
    df_ml = df_ml[keep_cols].reset_index(drop=True)

    logger.info(
        "Slimmed DataFrame: %d rows × %d columns "
        "(features=%d, labels=%d, original width dropped).",
        len(df_ml), df_ml.shape[1], len(feature_cols_present), len(_LABEL_COLS),
    )

    return thresholds, df_ml


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the subset of ``ALL_FEATURE_COLS`` actually present in ``df``.

    Logs any expected columns that are missing so the caller can decide
    whether to abort or continue with a reduced feature set.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
        Present feature columns in the order defined by ``ALL_FEATURE_COLS``.
    """
    present = [c for c in ALL_FEATURE_COLS if c in df.columns]
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Feature columns absent from dataset: %s", missing)
    return present
