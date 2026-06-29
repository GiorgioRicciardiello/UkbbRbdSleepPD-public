"""
Configuration for the LR-MDS likelihood ratio analysis.

All constants live here — no magic numbers elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ── Column names ───────────────────────────────────────────────────────────────

OUTCOME: Final[str] = "outcome_1a_pd_only"
INCIDENT_COL: Final[str] = "outcome_1a_pd_only__incident"
CONTROL_COL: Final[str] = "control"
RBD_COL: Final[str] = "abk_rbd_score_mean"
RBD_ZSCORE_COL: Final[str] = "rbd_zscore"
SEX_COL: Final[str] = "cov_sex_31"        # 0 = female, 1 = male (UKBB encoding)
AGE_COL: Final[str] = "cov_age_recruitment_21022"
BMI_COL: Final[str] = "bmi_21001_bl"
ALCOHOL_COL: Final[str] = "cov_alcohol"    # Consolidated alcohol drinker status
SMOKING_COL: Final[str] = "cov_smoking"    # Consolidated smoking status
RBD_TERTILE_COL: Final[str] = "rg_pctl3"  # RBD stratification: Low/Mid/High

MALE_CODE: Final[int] = 1
FEMALE_CODE: Final[int] = 0

# Confounders for adjusted logistic regression (Phase 1 and 2)
CONFOUNDERS: Final[list[str]] = [AGE_COL, SEX_COL, BMI_COL]

# Confounders for interaction models (Phase 3)
INTERACTION_CONFOUNDERS: Final[list[str]] = [
    AGE_COL, SEX_COL, BMI_COL, ALCOHOL_COL, SMOKING_COL
]
INTERACTION_CONFOUNDERS_GENETIC: Final[list[str]] = (
    INTERACTION_CONFOUNDERS + [f"prs_pc{k}" for k in range(1, 4)]
)

# ── Cognitive variables (continuous markers) ───────────────────────────────────

COGNITIVE_VARS: Final[dict[str, str]] = {
    "cog_fluid_intelligence_bl": "Fluid Intelligence",
    "cog_react_time_bl": "Reaction Time (ms)",
    "cov_fi_questions_attempted_20128_bl": "FI Questions Attempted",
    "cog_numeric_memory_bl": "Numeric Memory",
    "cog_pairs_matching_bl": "Pairs Matching Status",
}

# ── Trail Making Test variables ────────────────────────────────────────────────

TMT_VARS: Final[dict[str, str]] = {
    "cog_tmt1_dur_bl": "TMT-A Duration (sec)",
    "cog_tmt2_dur_bl": "TMT-B Duration (sec)",
    "cog_tmt_ratio_log_bl": "TMT-B/A Ratio (log)",
}

# ── Genetic risk variables ─────────────────────────────────────────────────────

GENETIC_VARS: Final[dict[str, str]] = {
    "prs_score_pd": "PD Polygenic Risk Score",
}

# Ancestry principal components for genetic models
PC_COLS: Final[list[str]] = [f"prs_pc{k}" for k in range(1, 4)]

# ── Arm swing variables (gait actigraphy) ──────────────────────────────────────
# Derived from accelerometer raw time-series using a gait analysis model.
# j and w are accelerometer sensor axis labels (not anatomical locations).

ARM_SWING_VARS: Final[dict[str, str]] = {
    "arm_swing_amplitude_mean_j": "Arm Swing Amplitude Mean (axis j)",
    "arm_swing_amplitude_mean_w": "Arm Swing Amplitude Mean (axis w)",
    "arm_swing_amplitude_var_j": "Arm Swing Amplitude Variance (axis j)",
    "arm_swing_amplitude_var_w": "Arm Swing Amplitude Variance (axis w)",
}

# ── Cohort definitions: Required columns for each analysis universe ────────────

COHORT_DEFINITIONS: Final[dict[str, list[str]]] = {
    # Phase 1/2: Basic cohorts (age, sex, BMI; genetic adds PCs, no BMI)
    "mds_standard": [AGE_COL, SEX_COL, BMI_COL, INCIDENT_COL, CONTROL_COL],
    "cognitive": (
        list(COGNITIVE_VARS.keys())
        + [AGE_COL, SEX_COL, BMI_COL, INCIDENT_COL, CONTROL_COL]
    ),
    "tmt": (
        list(TMT_VARS.keys())
        + [AGE_COL, SEX_COL, BMI_COL, INCIDENT_COL, CONTROL_COL]
    ),
    "genetic": (
        list(GENETIC_VARS.keys())
        + PC_COLS
        + [AGE_COL, SEX_COL, INCIDENT_COL, CONTROL_COL]
    ),
    # Phase 3: Interaction cohorts (add alcohol, smoking, RBD tertile)
    "cognitive_interaction": (
        list(COGNITIVE_VARS.keys())
        + [AGE_COL, SEX_COL, BMI_COL, ALCOHOL_COL, SMOKING_COL,
           RBD_TERTILE_COL, INCIDENT_COL, CONTROL_COL]
    ),
    "tmt_interaction": (
        list(TMT_VARS.keys())
        + [AGE_COL, SEX_COL, BMI_COL, ALCOHOL_COL, SMOKING_COL,
           RBD_TERTILE_COL, INCIDENT_COL, CONTROL_COL]
    ),
    "genetic_interaction": (
        list(GENETIC_VARS.keys())
        + PC_COLS
        + [AGE_COL, SEX_COL, ALCOHOL_COL, SMOKING_COL,
           RBD_TERTILE_COL, INCIDENT_COL, CONTROL_COL]
    ),
    # Arm swing gait analysis
    "arm_swing": (
        list(ARM_SWING_VARS.keys())
        + [AGE_COL, SEX_COL, BMI_COL, ALCOHOL_COL, SMOKING_COL,
           INCIDENT_COL, CONTROL_COL]
    ),
    "arm_swing_interaction": (
        list(ARM_SWING_VARS.keys())
        + [AGE_COL, SEX_COL, BMI_COL, ALCOHOL_COL, SMOKING_COL,
           RBD_TERTILE_COL, INCIDENT_COL, CONTROL_COL]
    ),
}

# ── Prodromal columns ──────────────────────────────────────────────────────────

# Non-zero in UKBB cohort — usable for empirical LR computation
PRODROMAL_VIABLE: Final[list[str]] = [
    "prodromal_constipation_bl",
    "prodromal_depression_bl",
    "prodromal_orthostatic_bl",
    "prodromal_erectile_dysfunction_bl",
]

# All-zero in UKBB cohort (ICD10-coded, no hospital events recorded)
# Excluded from all analyses.
PRODROMAL_EXCLUDED: Final[list[str]] = [
    "prodromal_dream_enactment_bl",
    "prodromal_anosmia_bl",
    "prodromal_hyposmia_bl",
]

# Maps UKBB prodromal column → Heinzel 2019 marker name
PRODROMAL_COL_TO_HEINZEL: Final[dict[str, str]] = {
    "prodromal_constipation_bl": "constipation",
    "prodromal_depression_bl": "depression",
    "prodromal_orthostatic_bl": "orthostatic",
    "prodromal_erectile_dysfunction_bl": "erectile_dysfunction",
}

# Human-readable labels for tables and figures
PRODROMAL_LABELS: Final[dict[str, str]] = {
    "prodromal_constipation_bl": "Constipation",
    "prodromal_depression_bl": "Depression",
    "prodromal_orthostatic_bl": "Orthostatic hypotension",
    "prodromal_erectile_dysfunction_bl": "Erectile dysfunction (men only)",
}

# ── Published LRs (Heinzel et al. 2019, Table 1) ──────────────────────────────


@dataclass(frozen=True)
class MarkerLR:
    """Likelihood ratio for one prodromal/risk marker."""

    lr_pos: float
    lr_neg: float | None  # None = marker does not provide negative evidence
    label: str
    sex_restricted: str | None = None  # "male" or "female" if sex-specific


# Heinzel 2019 Table 1 entries used in C2 (hybrid) analysis.
# erectile_dysfunction is male-only per Heinzel and per UKBB variable definition.
HEINZEL_LRS: Final[dict[str, MarkerLR]] = {
    "sex_male": MarkerLR(lr_pos=1.2, lr_neg=None, label="Male sex"),
    "sex_female": MarkerLR(lr_pos=0.8, lr_neg=None, label="Female sex"),
    "constipation": MarkerLR(lr_pos=2.5, lr_neg=0.82, label="Constipation"),
    "depression": MarkerLR(lr_pos=1.6, lr_neg=0.88, label="Depression (± anxiety)"),
    "orthostatic": MarkerLR(
        lr_pos=3.2, lr_neg=0.80,
        label="Symptomatic orthostatic hypotension",
    ),
    "erectile_dysfunction": MarkerLR(
        lr_pos=3.4, lr_neg=0.87,
        label="Erectile dysfunction",
        sex_restricted="male",
    ),
}

# ── MDS age-based prior probability ───────────────────────────────────────────
# Source: Berg et al. 2015, MDS Research Criteria for Prodromal PD,
#         Supplementary Table S1; Heinzel et al. 2019.
# Keys: lower bound of 5-year age band.
# Values: prior probability of prodromal PD (proportion, 0–1).
# NOTE: verify exact values against Berg 2015 Supplementary Table S1
#       before final manuscript submission.
MDS_AGE_PRIOR: Final[dict[int, float]] = {
    50: 0.0020,
    55: 0.0030,
    60: 0.0050,
    65: 0.0080,
    70: 0.0120,
    75: 0.0170,
    80: 0.0200,
}

# ── Z-score threshold grid ────────────────────────────────────────────────────
# LR profile is computed at each threshold. Youden-optimal threshold
# is identified separately within lr_metrics.py.
ZSCORE_THRESHOLD_GRID: Final[list[float]] = [
    -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
]

# Minimum cell count for a stable LR estimate (flag if below this).
MIN_CELL_COUNT: Final[int] = 5

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_SUBDIR: Final[str] = "lr_analysis"
