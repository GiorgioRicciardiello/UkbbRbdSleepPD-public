"""
Screening model configuration.

All constants, feature definitions, and CV parameters for the training paradigm
comparison study. No runtime logic — import-only module.
"""
from __future__ import annotations

from typing import Dict, List

# ── Outcome ───────────────────────────────────────────────────────────────────
PRIMARY_OUTCOME: str = "outcome_1a_pd_only"
INCIDENT_COL: str = f"{PRIMARY_OUTCOME}__incident"
PREVALENT_COL: str = f"{PRIMARY_OUTCOME}__prevalent"
CONTROL_COL: str = "control"

# ── CV parameters ─────────────────────────────────────────────────────────────
OUTER_FOLDS: int = 10
INNER_FOLDS: int = 5
CONTROLS_PER_CASE: int = 10    # matched controls per combined case count in training
RANDOM_SEED: int = 42

# ── Feature definitions ───────────────────────────────────────────────────────

DEMO_FEATURES: List[str] = [
    "cov_age_recruitment_21022",
    "cov_sex_31",
    "cov_bmi",
    "cov_smoking",
    "cov_alcohol",
]

# Continuous RBD score (averaged nightly probability per subject)
RBD_CONTINUOUS: List[str] = ["abk_rbd_score_mean"]

# Categorical risk group (3-level percentile stratification)
# Values: 'Low (0,90%)', 'Intermediate (90,99%)', 'High (99,100%)'
RBD_CATEGORICAL: List[str] = ["rg_pctl3"]

# Binary prodromal markers derived from HES ICD-10 diagnoses
PRODROMAL_BINARY: List[str] = [
    "prodromal_constipation",
    "prodromal_depression",
    "prodromal_anxiety",
    "prodromal_orthostatic",
    "prodromal_erectile_dysfunction",
    "prodromal_dream_enactment",
    "prodromal_anosmia",
    "prodromal_hyposmia",
]

# Polygenic risk score for PD (ancestry PCs handled separately if needed)
PRS_FEATURES: List[str] = ["prs_score_pd"]

# Trail Making Test: ~50% availability.
# tmt_ratio_baseline = TMT-B duration / TMT-A duration (higher = worse executive function).
# tmt_missing is a pre-computed binary indicator (1 if subject has no valid TMT baseline).
TMT_FEATURES: List[str] = ["tmt_ratio_baseline"]
TMT_MISSING_FLAG: str = "tmt_missing"

# ── Derived: all candidate feature columns ────────────────────────────────────
ALL_FEATURE_COLS: List[str] = (
    DEMO_FEATURES
    + RBD_CONTINUOUS
    + RBD_CATEGORICAL
    + PRODROMAL_BINARY
    + PRS_FEATURES
    + TMT_FEATURES
    + [TMT_MISSING_FLAG]
)

# ── Numeric columns (imputed with median in preprocessing pipeline) ───────────
NUMERIC_FEATURES: List[str] = (
    DEMO_FEATURES
    + RBD_CONTINUOUS
    + PRS_FEATURES
    + TMT_FEATURES
)

# ── Paradigm 3: prevalent case weight ────────────────────────────────────────
# Prevalent PD cases are under medication at actigraphy time: their actigraphy
# features reflect active disease, not prodromal physiology. Lower weight limits
# their influence while retaining signal from HES-derived prodromal markers.
# This is a default; the main loop can sweep over values.
DEFAULT_PREVALENT_WEIGHT: float = 0.3

# ── XGBoost hyperparameter search space ──────────────────────────────────────
# Randomized search over this space using inner 5-fold CV scored by PR-AUC
# (average_precision), which is appropriate for severely imbalanced data.
XGB_PARAM_DISTRIBUTIONS: Dict[str, list] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.5],
}

XGB_N_ITER: int = 30   # number of random search combinations
XGB_JOBS: int = 4      # parallel jobs for inner CV (capped at available CPUs)

# ── Focal Loss Parameters ────────────────────────────────────────────────────
FOCAL_LOSS_ENABLED: bool = False   # Enable focal loss (focus on hard examples)
FOCAL_ALPHA: float = 0.25          # Class balancing factor
FOCAL_GAMMA: float = 2.0           # Focusing parameter (higher = sharper focus)
# NOTE: Focal loss requires proper integration into XGBoost objective function,
# not post-hoc retraining (which causes overfitting). Current implementation disabled.
