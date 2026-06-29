"""
Configuration for RBD–PRS biological strength analysis.

Zero runtime logic — import-only module.

Analysis goal
─────────────
Quantify the association between the ML-derived actigraphy RBD probability
score (abk_rbd_score_mean) and genetic risk for PD / RBD (prs_score_pd,
prs_score_rbd), restricted to subjects who have BOTH scores available.

Statistical chain
─────────────────
1. Spearman ρ (+ permutation CI) — distribution-free correlation
2. OLS partial regression adjusted for age, sex, BMI, PC1-PC10
3. GAM with spline terms for each PRS — test of non-linearity
All analyses are repeated in the High-risk subgroup (rg_pctl3 == 99th–100th).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from library.cox_prodromal.cox_config import BASE_COVARIATES, PC_COLS, PRS_COLS

# ── Outcome of interest ──────────────────────────────────────────────────────

PRIMARY_OUTCOME: str = "outcome_1a_pd_only"
SURV_EVENT_COL: str = f"{PRIMARY_OUTCOME}__surv_event"
SURV_DAYS_COL: str = f"{PRIMARY_OUTCOME}__surv_days"
INCIDENT_COL: str = f"{PRIMARY_OUTCOME}__incident"

# ── Risk grouping column ─────────────────────────────────────────────────────

RISK_GROUP_COL: str = "rg_pctl3"
HIGH_RISK_LABEL: str = "High (99,100%)"
RG_ORDER: List[str] = ["Low (0,90%)", "Intermediate (90,99%)", "High (99,100%)"]
RG_SHORT: dict = {
    "Low (0,90%)": "Low",
    "Intermediate (90,99%)": "Intermediate",
    "High (99,100%)": "High",
}

# ── Key exposure / genetic columns ──────────────────────────────────────────

RBD_SCORE_COL: str = "abk_rbd_score_mean"
PRS_PD_COL: str = "prs_score_pd"
PRS_RBD_COL: str = "prs_score_rbd"

# Columns that must be non-null for a subject to be included
REQUIRED_COLS: List[str] = [RBD_SCORE_COL, PRS_PD_COL, PRS_RBD_COL]

# ── Regression covariates ────────────────────────────────────────────────────
# PRS models require ancestry PCs to control for population stratification.
# BASE_COVARIATES: age, sex, BMI (from cox_config)
# PC_COLS: PC1–PC10 (from cox_config)

ADJUSTMENT_COVARIATES: List[str] = BASE_COVARIATES + PC_COLS

# ── Permutation test ─────────────────────────────────────────────────────────

PERMUTATION_N: int = 10_000
RANDOM_SEED: int = 42

# ── GAM settings ─────────────────────────────────────────────────────────────

GAM_N_SPLINES: int = 10          # Basis functions per smooth term
GAM_MAX_ITER: int = 500          # Solver iterations
GAM_LAMBDA_GRID: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# ── Pastel color palette ──────────────────────────────────────────────────────
# Consistent across all figures in this module.

PALETTE: dict = {
    "Low (0,90%)":           "#AEC6CF",   # pastel blue
    "Intermediate (90,99%)": "#FFD1A4",   # pastel orange
    "High (99,100%)":        "#FFB3B3",   # pastel red
    "case":                  "#F4A6A6",   # pastel rose (incident case)
    "control":               "#B5D5C5",   # pastel green (control)
    "prs_pd":                "#C3B1E1",   # pastel purple
    "prs_rbd":               "#FFDAC1",   # pastel peach
    "rbd_score":             "#B5EAD7",   # pastel mint
    "regression_line":       "#555555",   # dark grey
    "ci_band":               "#CCCCCC",   # light grey
}

FIGURE_DPI: int = 300
FIGURE_STYLE: str = "whitegrid"
