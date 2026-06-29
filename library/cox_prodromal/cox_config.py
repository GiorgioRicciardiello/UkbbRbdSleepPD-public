"""
Configuration for Cox Prodromal Analysis.

All constants, DAG specification, color palette, and model hierarchy.
Zero runtime logic — import-only module.

── Model hierarchy ─────────────────────────────────────────────────────────
Documented for report generation. No runtime logic.

Model 0: h(t) = h0(t) exp(bR R + bX X)              RBD-only
Model 1: h(t) = h0(t) exp(bP P + bX X)              Prodromal-only
Model 2: h(t) = h0(t) exp(bR R + bP P + bX X)       Additive
Model 3: h(t) = h0(t) exp(bR R + bP P + bRP(RxP) + bX X)  Interaction
Model 4: Aalen-Johansen CIF + cause-specific Cox     Competing risk


"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from config.config import RBD_RISK_COLORS, RBD_RISK_COLORS_COMBINED  # noqa: E402


# ── Analysis parameters ─────────────────────────────────────────────────────

MIN_EVENTS_FOR_MODEL: int = 5
MIN_PREVALENCE_FOR_BINARY: int = 30
LAG_YEARS: float = 2.0
SPLINE_DF: int = 4
RIDGE_PENALIZER: float = 0.01
# Stronger ridge penalty for sksurv bootstrap iterations.
# Bootstrap samples can produce near-perfect separation (all events in one
# group), driving exp(x*w) to overflow and causing ConvergenceWarning (all
# n_iter=100 iterations exhausted).  alpha=1.0 makes the loss landscape
# smooth enough for fast convergence without materially biasing RERI/delta-C
# CIs — regularization is swamped by the data signal at N~86k.
# Point estimates always use RIDGE_PENALIZER=0.01 via lifelines.
BOOTSTRAP_RIDGE_PENALIZER: float = 1.0
# Hard clip applied to sksurv bootstrap coefficients before exp().
# Prevents float64 overflow (exp(>709) = inf) on bootstrap samples that still
# exhibit near-perfect separation despite strong regularization.
# exp(20) ~ 5e8: a valid but extreme HR that does not affect CI percentiles.
BOOTSTRAP_COEF_CLIP: float = 20.0
ABSOLUTE_RISK_TIMEPOINTS: List[float] = [5.0, 10.0]
PRIMARY_METHOD: str = "percentile_3g"
PRIMARY_OUTCOME: str = "outcome_1a_pd_only"
BOOTSTRAP_SEED: int = 42
BOOTSTRAP_N: int = 1000 # 1000

# ── Mediation analysis (optional stage) ───────────────────────────────────
RUN_MEDIATION: bool = True

# ── HES activity sensitivity analysis ──────────────────────────────────────
# Subjects with a gap > 4 years between their last pre-baseline HES record
# and wear_time_start have a pre-baseline window that is not well covered by
# hospital records.  The sensitivity analysis restricts to subjects with
# gap ≤ HES_GAP_THRESHOLD_YEARS to verify that HR estimates are not driven
# by misclassified unexposed subjects (false negatives in the prodromal=1 group).
# Only applies to HES-derived binary prodromal variables; cognitive (continuous)
# variables are questionnaire-based and unaffected by HES coverage.
HES_GAP_THRESHOLD_YEARS: float = 4.0
HES_GAP_COL: str = "hes_gap_pre_baseline_years"

# ── Prodromal marker definitions ────────────────────────────────────────────
# PRODROMAL_VARS: continuous cognitive markers
# PRODROMAL_BINARY_VARS: binary markers (merged HES + medication evidence)

PRODROMAL_VARS: Dict[str, str] = {
    "cog_fluid_intelligence_bl": "Fluid Intelligence",
    "cog_react_time_bl":         "Reaction Time (ms)",
    "cog_numeric_memory_bl":     "Numeric Memory",
    "cog_pairs_matching_bl":     "Pairs Matching Status",
}

# TMT sensitivity: run separately to avoid losing ~47% of cohort from the primary
# analysis. N restricted to subjects with a valid TMT assessment within ±730 days
# of wear_time_start (online_i0 2014-2015; N≈46,491 vs primary N≈88,115).
PRODROMAL_TMT_VARS: Dict[str, str] = {
    "cog_tmt_ratio_log_bl": "TMT-B/A Ratio (log)",
}

PRODROMAL_BINARY_VARS: Dict[str, str] = {
    "prodromal_constipation_bl":         "Constipation",
    "prodromal_depression_bl":           "Depression",
    "prodromal_anxiety_bl":              "Anxiety",
    "prodromal_orthostatic_bl":          "Orthostatic Hypotension",
    "prodromal_erectile_dysfunction_bl": "Erectile Dysfunction",
    "prodromal_dream_enactment_bl":      "Dream Enactment",
    "prodromal_anosmia_bl":              "Anosmia",
    "prodromal_hyposmia_bl":             "Hyposmia",
}

# ── Outcomes ────────────────────────────────────────────────────────────────

OUTCOMES: List[str] = [
    "outcome_1a_pd_only",
    # "outcome_1b_pd_ad",
    "outcome_2a_vasculardementia",
    # "outcome_2b_pd_vasculardementia",
    # "outcome_4a_ad_only",
]

METHODS: List[str] = ["percentile_3g"]

# ── Parallelism ──────────────────────────────────────────────────────────────
# Hard cap: shared system — never exceed MAX_SYSTEM_CPUS.
MAX_SYSTEM_CPUS: int = 13
_n_cpus: int = min(os.cpu_count() or 4, MAX_SYSTEM_CPUS)

# Outcome-level: one worker per outcome, capped at available CPUs.
MAX_WORKERS: int = min(len(OUTCOMES), max(1, _n_cpus - 1))

# Bootstrap-level: remaining CPUs after outcome workers finish.
# Non-primary outcomes have no bootstrap and finish quickly, freeing CPUs.
# The primary-outcome worker then uses BOOTSTRAP_JOBS for its inner loops.
BOOTSTRAP_JOBS: int = max(2, _n_cpus - MAX_WORKERS)

# ── Covariates ──────────────────────────────────────────────────────────────

BASE_COVARIATES: List[str] = [
    "cov_age_recruitment_21022",
    "cov_sex_31",
    "cov_bmi",
]

SMOKING_CANDIDATES: List[str] = [
    "cov_smoking_20116_bl",
    "cov_smoking_20116_i1",
    "cov_smoking_20116_fu",
    "cov_smoking_20116_i3",
]

ALCOHOL_CANDIDATES: List[str] = [
    "cov_alcohol_20117_bl",
    "cov_alcohol_20117_i1",
    "cov_alcohol_20117_fu",
    "cov_alcohol_20117_i3",
]

# ── Model A: PRS + ancestry PCs ──────────────────────────────────────────────
# Included ONLY in Model A (RBD + PRS models), NOT in prodromal models.
# PCs control for population stratification (a genetic-specific bias);
# prodromal variables do not need PCs unless ancestry is a direct confounder,
# which it is not after adjusting for age, sex, BMI, smoking, alcohol.
PRS_COLS: List[str] = ["prs_score_pd"] # "prs_score_rbd"]
PC_COLS: List[str] = [f"prs_pc{k}" for k in range(1, 11)]
MODEL_A_COVARIATES: List[str] = PRS_COLS + PC_COLS

# ── Model G: GBA carrier ──────────────────────────────────────────────────────
# GBA carrier is binary (0=non-carrier, 1=carrier; ~1-2% in EUR).
# No ancestry PCs needed (single-variant flag, not a polygenic score).
# Restricted to PRIMARY_OUTCOME only (GBA variants are PD-specific).
GBA_COL: str = "gba_carrier"
MODEL_G_COVARIATES: List[str] = [GBA_COL]

# ── Threshold stability cutoffs ─────────────────────────────────────────────

THRESHOLD_STABILITY_PERCENTILES: List[float] = [5.0, 10.0, 15.0]

# ── Age-stratified sensitivity analysis ───────────────────────────────────
# Two strata: ≤60 and >60 at recruitment.  Few outcomes in <50 age band,
# so a 3-group split wastes statistical power in the youngest stratum.
AGE_STRATA: List[tuple] = [(0, 60), (60, 100)]
AGE_COL: str = "cov_age_recruitment_21022"

# ── Screening metrics (PPV / NPV) ────────────────────────────────────────
# Percentile thresholds on abk_rbd_score_mean at which to compute PPV, NPV,
# sensitivity, specificity.  Time horizons for cumulative incidence.
SCREENING_PERCENTILES: List[float] = [90.0, 95.0, 99.0]
SCREENING_TIME_HORIZONS: List[float] = [5.0, 10.0]

# ── Competing risk definitions ──────────────────────────────────────────────
# Maps primary outcome -> list of competing outcome event columns.
# Death (all-cause) is a competing event for all neurological outcomes:
# subjects who die before diagnosis can no longer develop the condition.
# Death columns (death_surv_time, death_surv_event) are derived from
# death_flag / death_date in build_survival_dataset_for_outcome().

COMPETING_OUTCOMES: Dict[str, List[str]] = {
    "outcome_1a_pd_only": [
        "outcome_2a_vasculardementia",
        "outcome_4a_ad_only",
        "death",
    ],
    "outcome_4a_ad_only": [
        "outcome_1a_pd_only",
        "outcome_2a_vasculardementia",
        "death",
    ],
    "outcome_2a_vasculardementia": [
        "outcome_1a_pd_only",
        "outcome_4a_ad_only",
        "death",
    ],
}


# ── Causal DAG specification ────────────────────────────────────────────────

@dataclass(frozen=True)
class DAGNode:
    """Variable in the causal DAG."""
    name: str
    node_type: str  # "latent", "exposure", "confounder", "outcome"


@dataclass(frozen=True)
class DAGEdge:
    """Directed edge in the causal DAG."""
    source: str
    target: str


CAUSAL_DAG_NODES: List[DAGNode] = [
    DAGNode("latent_pd_pathology", "latent"),
    DAGNode("rbd_score", "exposure"),
    DAGNode("prodromal_marker", "exposure"),
    DAGNode("incident_pd", "outcome"),
    DAGNode("age", "confounder"),
    DAGNode("sex", "confounder"),
    DAGNode("bmi", "confounder"),
]

CAUSAL_DAG_EDGES: List[DAGEdge] = [
    DAGEdge("latent_pd_pathology", "rbd_score"),
    DAGEdge("latent_pd_pathology", "prodromal_marker"),
    DAGEdge("latent_pd_pathology", "incident_pd"),
    DAGEdge("age", "rbd_score"),
    DAGEdge("age", "prodromal_marker"),
    DAGEdge("age", "incident_pd"),
    DAGEdge("sex", "rbd_score"),
    DAGEdge("sex", "prodromal_marker"),
    DAGEdge("sex", "incident_pd"),
    DAGEdge("bmi", "rbd_score"),
    DAGEdge("bmi", "prodromal_marker"),
    DAGEdge("bmi", "incident_pd"),
    # No edge between rbd_score and prodromal_marker:
    # their association is fully explained by shared latent pathology.
]


# ── Harmonized color palette ────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskPalette:
    """Fixed color assignments for consistent plotting across all figures.

    Colors sourced from config.config.RBD_RISK_COLORS — do not hardcode elsewhere.
    """
    rbd_low: str = "#8ec7ea"       # sky-blue
    rbd_mid: str = "#fab46f"       # orange
    rbd_high: str = "#f78b8b"      # coral-pink
    prodromal_no: str = "#4DAF4A"  # green
    prodromal_yes: str = "#E41A1C" # red


RISK_PALETTE = RiskPalette(
    rbd_low=RBD_RISK_COLORS["Low"],
    rbd_mid=RBD_RISK_COLORS["Mid"],
    rbd_high=RBD_RISK_COLORS["High"],
)


# ── Model hierarchy ─────────────────────────────────────────────────────────
# Documented for report generation. No runtime logic.
#
# Model 0: h(t) = h0(t) exp(bR R + bX X)              RBD-only
# Model 1: h(t) = h0(t) exp(bP P + bX X)              Prodromal-only
# Model 2: h(t) = h0(t) exp(bR R + bP P + bX X)       Additive
# Model 3: h(t) = h0(t) exp(bR R + bP P + bRP(RxP) + bX X)  Interaction
# Model 4: Aalen-Johansen CIF + cause-specific Cox     Competing risk
