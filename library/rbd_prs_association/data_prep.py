"""
Data preparation for the RBD–PRS association analysis.

Pipeline
────────
1. Load prodromal dataset via canonical loader (exclusions already applied).
2. Restrict to subjects with both PRS scores AND RBD score available.
3. Build adjustment covariates (lifestyle + PCs coerced to numeric).
4. Derive case/control flag from outcome columns.
5. Log every filter step for traceability.

Assumptions
───────────
- load_prodromal_dataset() returns one row per subject (make_subject_level
  has already been applied).
- prs_score_pd and prs_score_rbd are z-scores (unit-variance); no additional
  standardisation is performed here to preserve interpretability.
- PC columns (prs_pc1–prs_pc10) may be missing for non-European subjects;
  those subjects are therefore dropped when REQUIRED_COLS are checked.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd

from library.cox_prodromal.data_prep import (
    build_extended_covariates,
    load_prodromal_dataset,
)
from library.cox_prodromal.cox_config import BASE_COVARIATES, PC_COLS
from library.rbd_prs_association.config import (
    ADJUSTMENT_COVARIATES,
    HIGH_RISK_LABEL,
    INCIDENT_COL,
    REQUIRED_COLS,
    RISK_GROUP_COL,
    RBD_SCORE_COL,
    SURV_DAYS_COL,
    SURV_EVENT_COL,
)

logger = logging.getLogger(__name__)


def _log_filter(label: str, n_before: int, df: pd.DataFrame) -> None:
    """Log subject counts before/after a filter step."""
    n_after = df["id"].nunique() if "id" in df.columns else len(df)
    logger.info("Filter [%s]: %d → %d subjects (dropped %d)", label, n_before, n_after, n_before - n_after)


def load_analysis_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """Load, filter, and annotate the RBD–PRS analytical dataset.

    Returns
    -------
    df : pd.DataFrame
        One row per subject. Columns include RBD score, PRS scores,
        adjustment covariates, risk groups, and outcome flags.
    active_covariates : list[str]
        Covariate columns that survived the numeric-coercion step
        (passed to regression models).
    """
    # ── 1. Load base dataset (exclusions already applied) ────────────────────
    logger.info("Loading prodromal dataset ...")
    thresholds, df = load_prodromal_dataset()
    n0 = df["id"].nunique() if "id" in df.columns else len(df)
    logger.info("Base cohort: %d subjects, %d columns", n0, df.shape[1])

    # ── 2. Build extended covariates (lifestyle + numeric coercion) ──────────
    logger.info("Building extended covariates ...")
    df, lifestyle_covariates = build_extended_covariates(df, BASE_COVARIATES)

    # ── 3. Coerce PC columns to numeric ─────────────────────────────────────
    # PCs may arrive as objects if the TSV join produced mixed types.
    for col in PC_COLS:
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning("PC column %s missing — will be excluded from adjustment", col)

    available_pcs = [c for c in PC_COLS if c in df.columns]
    active_covariates: List[str] = lifestyle_covariates + available_pcs

    # ── 4. Restrict to subjects with all required columns non-null ───────────
    available_required = [c for c in REQUIRED_COLS if c in df.columns]
    missing_required = set(REQUIRED_COLS) - set(available_required)
    if missing_required:
        raise ValueError(
            f"Required columns absent from dataset: {missing_required}. "
            "Ensure the PRS merge step has been run (build_ukb_dataset.py)."
        )

    n_before = df["id"].nunique() if "id" in df.columns else len(df)
    mask_complete = df[available_required].notna().all(axis=1)
    df = df.loc[mask_complete].copy()
    _log_filter("PRS + RBD score completeness", n_before, df)

    # ── 5. Drop subjects with missing adjustment covariates ─────────────────
    # Ancestry PCs are required for a valid genetic association analysis.
    # Subjects without PCs are non-European or withdrew from genetics; dropping
    # them is consistent with the PRS construction cohort definition.
    required_adj = [c for c in ADJUSTMENT_COVARIATES if c in df.columns]
    n_before = df["id"].nunique() if "id" in df.columns else len(df)
    mask_adj = df[required_adj].notna().all(axis=1)
    df = df.loc[mask_adj].copy()
    _log_filter("Adjustment covariates completeness", n_before, df)

    # ── 6. Derive case / control flag ────────────────────────────────────────
    # A subject is a "case" if they have an incident event (surv_event == 1).
    # surv_event == 2 is a competing event; these remain "control" for PD.
    if SURV_EVENT_COL in df.columns:
        df = df.copy()
        df["case_control"] = df[SURV_EVENT_COL].map(
            {0: "control", 1: "case", 2: "control"}
        ).fillna("control")
    else:
        logger.warning("Survival event column %s missing; case_control set to 'unknown'", SURV_EVENT_COL)
        df = df.copy()
        df["case_control"] = "unknown"

    # ── 7. Ensure risk group column is present ────────────────────────────────
    if RISK_GROUP_COL not in df.columns:
        raise ValueError(
            f"Risk group column '{RISK_GROUP_COL}' absent. "
            "Rerun build_ukb_dataset.py to regenerate risk groups."
        )

    # ── 8. Final summary ─────────────────────────────────────────────────────
    n_final = df["id"].nunique() if "id" in df.columns else len(df)
    n_cases = (df["case_control"] == "case").sum()
    n_high = (df[RISK_GROUP_COL] == HIGH_RISK_LABEL).sum()
    logger.info(
        "Analytical dataset ready: N=%d, cases=%d, high-RBD=%d",
        n_final, n_cases, n_high
    )

    return df, active_covariates
