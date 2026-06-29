"""
Table 1 — Baseline Characteristics by RBD Risk Group
=====================================================

Produces two epidemiological baseline-characteristic tables, one per RBD
risk-thresholding method:

    percentile_2g → groups: Low / High
    percentile_3g → groups: Low / Mid / High

Dataset
-------
Reads from the canonical production paths defined in config (no mode
subdirectory).  These are populated by run_merge_ukbb_rbd.py, which runs
the ABK model and promotes its outputs to the root directories expected
here:
    data/pp/res_build_final_dataset/   ← parquet files
    data/risk_thresholds/              ← threshold JSON files

Stratification variable
-----------------------
RBD risk group based on outcome_1a_pd_only thresholds (primary PD outcome),
derived from the ABK actigraphy-based RBD probability score.

Variables included (mirrors the Cox prodromal model)
-----------------------------------------------------
  Demographics   : age at recruitment, sex, BMI
  RBD score      : ABK RBD mean probability (continuous)
  Follow-up      : PD survival time (days → years)
  Outcomes       : incident flags for all 6 outcomes
  Cognitive      : fluid intelligence, reaction time, FI questions,
                   numeric memory, pairs matching, trail making errors
  Prodromal      : constipation, depression, anxiety, orthostatic hypotension,
                   erectile dysfunction, dream enactment, anosmia, hyposmia
                   (merged HES ICD-10 + self-reported medication evidence)
  HES-only       : same 8 markers, HES ICD-10 evidence only
  Medication     : laxatives, antidepressants, anxiolytics, OH meds, PDE5i

Statistical tests (descriptive; no multiplicity correction per STROBE)
-----------------------------------------------------------------------
  Continuous : Mann-Whitney U (2 groups), Kruskal-Wallis (3 groups)
  Binary     : χ² (all expected cells ≥ 5), Fisher's exact (any cell < 5,
               2 × 2 only); χ² for larger tables regardless

Format
------
  Continuous : mean (SD); median [IQR]
  Binary     : N (%)
  Missing    : reported per variable

Output
------
  results/table_one/table1_rbd_risk_groups.xlsx  — two sheets (one per method)
  results/table_one/table1_percentile_2g.xlsx
  results/table_one/table1_percentile_3g.xlsx

Assumptions
-----------
  * cov_sex_31 == 1 → Male  (UKBB field 31 coding)
  * Survival time column {outcome}_surv_time is in days
  * Risk group columns pre-computed and stored in parquet (from run_merge_ukbb_rbd.py)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config.config import config, outcomes, outcomes_short_names
from library.column_registry import col_incident, col_prevalent, col_dx, col_surv_time, col_risk_group_agnostic, METHOD_TO_RISK_SUFFIX
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level

# ============================================================================
# CONFIGURATION
# ============================================================================

STRATIFICATION_OUTCOME: str = "outcome_1a_pd_only"  # outcome whose thresholds define risk groups
METHODS: List[str] = ["percentile_3g"]

PRODROMAL_MARKERS: List[str] = [
    "constipation", "depression", "anxiety", "orthostatic",
    "erectile_dysfunction", "dream_enactment", "anosmia", "hyposmia",
]

# ---------------------------------------------------------------------------
# Variable specification list — drives both computation and table row order.
# Each entry: (display_label, column_name, var_type)
#   var_type ∈ {"cont", "bin"}
# ---------------------------------------------------------------------------
VARIABLE_SPECS: List[Tuple[str, str, str]] = [
    # ── Demographics ─────────────────────────────────────────────────────────
    ("Age at recruitment (years)",          "cov_age_recruitment_21022",            "cont"),
    ("Male sex",                            "cov_sex_31",                           "bin"),
    ("BMI (kg/m²)",                         "cov_bmi",                              "cont"),
    # ── RBD score ────────────────────────────────────────────────────────────
    ("RBD probability score (ABK)",         "abk_rbd_score_mean",                   "cont"),
    # ── Follow-up ────────────────────────────────────────────────────────────
    ("Follow-up time (years)",              "follow_up_years",                      "cont"),
    # ── Incident events (during follow-up) ───────────────────────────────────
    ("Incident PD",                         col_incident("outcome_1a_pd_only"),          "bin"),
    ("Incident PD or AD",                   col_incident("outcome_1b_pd_ad"),            "bin"),
    ("Incident Vascular Dementia",          col_incident("outcome_2a_vasculardementia"),    "bin"),
    ("Incident PD + Vascular Dementia",     col_incident("outcome_2b_pd_vasculardementia"), "bin"),
    ("Incident AD",                         col_incident("outcome_4a_ad_only"),             "bin"),
    # ── Cognitive prodromal markers (baseline visit, _bl) ─────────────────────
    ("Fluid Intelligence Score",            "cog_fluid_intelligence_bl",            "cont"),
    ("Reaction Time (ms)",                  "cog_react_time_bl",                    "cont"),
    ("Numeric Memory (max digits)",         "cog_numeric_memory_bl",                "cont"),
    ("Pairs Matching Status",               "cog_pairs_matching_bl",                "cont"),
    ("TMT-A Duration (sec)",                 "cog_tmt1_dur_bl",                      "cont"),
    ("TMT-B Duration (sec)",                 "cog_tmt2_dur_bl",                      "cont"),
    ("TMT-B/A Ratio (log)",                  "cog_tmt_ratio_log_bl",                 "cont"),
    # ── Prodromal markers (pre-baseline HES + medication, _bl) ────────────────
    ("Constipation",                        "prodromal_constipation_bl",            "bin"),
    ("Depression",                          "prodromal_depression_bl",              "bin"),
    ("Anxiety",                             "prodromal_anxiety_bl",                 "bin"),
    ("Orthostatic Hypotension",             "prodromal_orthostatic_bl",             "bin"),
    ("Erectile Dysfunction",                "prodromal_erectile_dysfunction_bl",     "bin"),
    ("Dream Enactment",                     "prodromal_dream_enactment_bl",         "bin"),
    ("Anosmia",                             "prodromal_anosmia_bl",                 "bin"),
    ("Hyposmia",                            "prodromal_hyposmia_bl",                "bin"),
    # ── Genetics ─────────────────────────────────────────────────────────────
    ("PD PRS (z-score)",                    "prs_score_pd",                         "cont"),
    ("RBD PRS (z-score)",                   "prs_score_rbd",                        "cont"),
    ("GBA carrier",                         "gba_carrier",                          "bin"),
]

# Column name → section header (injected before the first variable of each section)
SECTION_TRIGGERS: Dict[str, str] = {
    "cov_age_recruitment_21022":        "Demographics",
    "abk_rbd_score_mean":               "RBD Score",
    "follow_up_years":                  "Follow-up",
    col_incident("outcome_1a_pd_only"):      "Incident Events (during follow-up)",
    "cog_fluid_intelligence_bl":        "Cognitive Prodromal Markers (Cox model)",
    "prodromal_constipation_bl":        "Prodromal Markers (pre-baseline HES + Medication)",
    "prs_score_pd":                     "Genetics",
}

# ---------------------------------------------------------------------------
# Supplementary Table S1 — HES-only and medication-only flags
# (moved from main Table 1 to avoid temporal-scope confusion:
#  HES flags are lifetime; prodromal markers are pre-baseline only)
# ---------------------------------------------------------------------------
SUPPLEMENTARY_VARIABLE_SPECS: List[Tuple[str, str, str]] = [
    # ── HES-only flags (lifetime, no temporal filter) ─────────────────────────
    ("Constipation (HES)",                  "constipation_hes",                     "bin"),
    ("Depression (HES)",                    "depression_hes",                       "bin"),
    ("Anxiety (HES)",                       "anxiety_hes",                          "bin"),
    ("Orthostatic Hypotension (HES)",       "Orthostatic_hes",                      "bin"),
    ("Erectile Dysfunction (HES)",          "erectile_dysfunction_hes",             "bin"),
    ("Dream Enactment (HES)",              "dream_enactment_hes",                  "bin"),
    ("Anosmia (HES)",                       "anosmia_hes",                          "bin"),
    ("Hyposmia (HES)",                      "hyposmia_hes",                         "bin"),
    # ── Medication-only flags (self-reported) ────────────────────────────────
    ("Laxatives (med)",                     "med_laxatives",                        "bin"),
    ("Antidepressants (med)",               "med_depression",                       "bin"),
    ("Anxiolytics (med)",                   "med_anxiety",                          "bin"),
    ("OH medication (med)",                 "med_orthostatic_hypotension",          "bin"),
    ("PDE5 inhibitors (med)",               "med_pde5_inhibitors",                  "bin"),
]

SUPPLEMENTARY_SECTION_TRIGGERS: Dict[str, str] = {
    "constipation_hes":  "HES-only Flags (lifetime)",
    "med_laxatives":     "Medication-only Flags (self-reported)",
}

# ---------------------------------------------------------------------------
# Condensed variable list for presentation slides (Slides 2 & 3).
# Contains only the variables shown in the proposed slides; omits rare or
# near-zero-prevalence markers (dream enactment, anosmia, hyposmia) and
# low-priority HES flags that are not highlighted in the narrative.
# ---------------------------------------------------------------------------
SLIDE_VARIABLE_SPECS: List[Tuple[str, str, str]] = [
    # ── Demographics ─────────────────────────────────────────────────────────
    ("Age at recruitment (years)",      "cov_age_recruitment_21022",            "cont"),
    ("Male sex",                        "cov_sex_31",                           "bin"),
    ("BMI (kg/m²)",                     "cov_bmi",                              "cont"),
    # ── RBD score ────────────────────────────────────────────────────────────
    ("RBD probability score (ABK)",     "abk_rbd_score_mean",                   "cont"),
    # ── Follow-up ────────────────────────────────────────────────────────────
    ("Follow-up time (years)",          "follow_up_years",                      "cont"),
    # ── Incident events ─────────────────────────────────────────────────────
    ("Incident PD",                     col_incident("outcome_1a_pd_only"),          "bin"),
    ("Incident Vascular Dementia",      col_incident("outcome_2a_vasculardementia"),    "bin"),
    ("Incident PD + Vascular Dementia", col_incident("outcome_2b_pd_vasculardementia"), "bin"),
    # ── Cognitive ────────────────────────────────────────────────────────────
    ("Reaction Time (ms)",              "cog_react_time_bl",                    "cont"),
    # ── Prodromal (pre-baseline HES + medication) ────────────────────────────
    ("Constipation",                    "prodromal_constipation_bl",            "bin"),
    ("Depression",                      "prodromal_depression_bl",              "bin"),
    ("Anxiety",                         "prodromal_anxiety_bl",                 "bin"),
    ("Orthostatic Hypotension",         "prodromal_orthostatic_bl",             "bin"),
    # ── Genetics ─────────────────────────────────────────────────────────────
    ("PD PRS (z-score)",                "prs_score_pd",                         "cont"),
    ("RBD PRS (z-score)",               "prs_score_rbd",                        "cont"),
    ("GBA carrier",                     "gba_carrier",                          "bin"),
]

SLIDE_SECTION_TRIGGERS: Dict[str, str] = {
    "cov_age_recruitment_21022":            "Demographics",
    "abk_rbd_score_mean":                   "RBD Score",
    "follow_up_years":                      "Follow-up",
    col_incident("outcome_1a_pd_only"):          "Incident Events (during follow-up)",
    "cog_react_time_bl":                    "Cognitive",
    "prodromal_constipation_bl":            "Prodromal Markers (pre-baseline)",
    "prs_score_pd":                         "Genetics",
}

# ---------------------------------------------------------------------------
# Controls-only sensitivity table (3-group RBD stratification).
#
# Purpose: show that cognitive and prodromal differences between RBD groups
# are present among subjects who NEVER develop any neurodegenerative outcome
# during follow-up, ruling out subclinical pre-diagnostic disease as a
# confound.  Requested by reviewers.
#
# Filter: control == True  (no incident event for any outcome).
# All incident outcome columns are removed from the variable spec — they
# are 0 by construction and would be uninformative.  The "Incident Events"
# section is omitted entirely.  Follow-up, demographics, cognitive markers,
# prodromal markers, and genetics are retained.
# ---------------------------------------------------------------------------

# Collect all incident-outcome column names so we can exclude them as a set.
_INCIDENT_COLS: set = {col_incident(oc) for oc in outcomes}

CONTROLS_ONLY_VARIABLE_SPECS: List[Tuple[str, str, str]] = [
    spec for spec in VARIABLE_SPECS
    if spec[1] not in _INCIDENT_COLS
]

# Drop any section trigger whose key is an incident-outcome column.
# The "Incident Events" section disappears completely; all other sections
# (Demographics, RBD Score, Follow-up, Cognitive, Prodromal, Genetics) survive.
CONTROLS_ONLY_SECTION_TRIGGERS: Dict[str, str] = {
    k: v for k, v in SECTION_TRIGGERS.items()
    if k not in _INCIDENT_COLS
}

# ---------------------------------------------------------------------------
# Controls-only table with LATEST cognitive assessments (new, additive).
#
# Purpose: cognitive performance using latest available assessment per subject
# across all UKBB follow-up instances (not just baseline visit i0).
# Includes additional cognitive variables never before in any table: SDS and
# prospective memory.
#
# Filter: control == True  (same as CONTROLS_ONLY_* above)
# Cognitive section: uses cog_*_latest columns from add_cognitive_latest_per_subject()
# Output file: tableS_controls_cognitive_latest_percentile_3g.xlsx (new file)
# ---------------------------------------------------------------------------

CONTROLS_COGNITIVE_LATEST_SPECS: List[Tuple[str, str, str]] = [
    # ── Demographics ─────────────────────────────────────────────────────────
    ("Age at recruitment (years)",          "cov_age_recruitment_21022",            "cont"),
    ("Male sex",                            "cov_sex_31",                           "bin"),
    ("BMI (kg/m²)",                         "cov_bmi",                              "cont"),
    # ── RBD score ────────────────────────────────────────────────────────────
    ("RBD probability score (ABK)",         "abk_rbd_score_mean",                   "cont"),
    # ── Follow-up ────────────────────────────────────────────────────────────
    ("Follow-up time (years)",              "follow_up_years",                      "cont"),
    # ── Cognitive performance (latest available assessment) ───────────────────
    ("Fluid Intelligence (latest)",         "cog_fluid_intelligence_latest",        "cont"),
    ("Reaction Time, ms (latest)",          "cog_react_time_latest",                "cont"),
    ("FI Questions Attempted (latest)",     "cog_fi_questions_latest",              "cont"),
    ("Numeric Memory (latest)",             "cog_numeric_memory_latest",            "cont"),
    ("Pairs Matching (latest)",             "cog_pairs_status_latest",              "cont"),
    ("SDS Correct/min (latest)",            "cog_sds_correct_per_min_latest",       "cont"),
    ("SDS Accuracy (latest)",               "cog_sds_accuracy_latest",              "cont"),
    ("Prospective Memory (latest)",         "cog_prospective_memory_latest",        "bin"),
    ("TMT-A Duration, sec (latest)",        "cog_tmt1_dur_latest",                  "cont"),
    ("TMT-B Duration, sec (latest)",        "cog_tmt2_dur_latest",                  "cont"),
    ("TMT-B/A Ratio, log (latest)",         "cog_tmt_ratio_log_latest",             "cont"),
    # ── Prodromal markers (pre-baseline HES + medication, _bl) ────────────────
    ("Constipation",                        "prodromal_constipation_bl",            "bin"),
    ("Depression",                          "prodromal_depression_bl",              "bin"),
    ("Anxiety",                             "prodromal_anxiety_bl",                 "bin"),
    ("Orthostatic Hypotension",             "prodromal_orthostatic_bl",             "bin"),
    ("Erectile Dysfunction",                "prodromal_erectile_dysfunction_bl",    "bin"),
    ("Dream Enactment",                     "prodromal_dream_enactment_bl",         "bin"),
    ("Anosmia",                             "prodromal_anosmia_bl",                 "bin"),
    ("Hyposmia",                            "prodromal_hyposmia_bl",                "bin"),
    # ── Genetics ─────────────────────────────────────────────────────────────
    ("PD PRS (z-score)",                    "prs_score_pd",                         "cont"),
    ("RBD PRS (z-score)",                   "prs_score_rbd",                        "cont"),
    ("GBA carrier",                         "gba_carrier",                          "bin"),
]

CONTROLS_COGNITIVE_SECTION_TRIGGERS: Dict[str, str] = {
    "cov_age_recruitment_21022":        "Demographics",
    "abk_rbd_score_mean":               "RBD Score",
    "follow_up_years":                  "Follow-up",
    "cog_fluid_intelligence_latest":    "Cognitive Performance (latest available assessment)",
    "prodromal_constipation_bl":        "Prodromal Markers (pre-baseline HES + Medication)",
    "prs_score_pd":                     "Genetics",
}

# ---------------------------------------------------------------------------
# Post-baseline table for controls with baseline + latest cognitive assessments.
#
# Purpose: Comprehensive controls-only Table 1 showing longitudinal cognitive
# trajectory (baseline → latest) + prodromal burden at both timepoints.
# Replaces tableS_controls_only_percentile_3g.xlsx with this richer version.
#
# Structure:
#   - Demographics, RBD, follow-up
#   - Cognitive BASELINE (i0, _bl)
#   - Cognitive LATEST (max available, _latest)
#   - Cognitive CHANGE (delta)
#   - Prodromal BASELINE (pre-baseline HES + medication, _bl)
#   - Prodromal FOLLOW-UP (post-baseline, _post)
#   - Prodromal NEW-ONSET (incident, _delta_incident)
#   - Genetics
#
# Output file: table1_postbaseline_percentile_3g_controls.xlsx
# ---------------------------------------------------------------------------
POSTBASELINE_CONTROLS_VARIABLE_SPECS: List[Tuple[str, str, str]] = [
    # ── Demographics ─────────────────────────────────────────────────────────
    ("Age at recruitment (years)",          "cov_age_recruitment_21022",            "cont"),
    ("Male sex",                            "cov_sex_31",                           "bin"),
    ("BMI (kg/m²)",                         "cov_bmi",                              "cont"),
    # ── RBD score ────────────────────────────────────────────────────────────
    ("RBD probability score (ABK)",         "abk_rbd_score_mean",                   "cont"),
    # ── Follow-up ────────────────────────────────────────────────────────────
    ("Follow-up time (years)",              "follow_up_years",                      "cont"),
    # ── Cognitive markers at BASELINE (visit i0) ────────────────────────────────
    ("Fluid Intelligence Score (baseline)", "cog_fluid_intelligence_bl",            "cont"),
    ("Reaction Time, ms (baseline)",        "cog_react_time_bl",                    "cont"),
    ("Numeric Memory (baseline)",           "cog_numeric_memory_bl",                "cont"),
    ("Pairs Matching Status (baseline)",    "cog_pairs_matching_bl",                "cont"),
    ("TMT-A Duration, sec (baseline)",      "cog_tmt1_dur_bl",                      "cont"),
    ("TMT-B Duration, sec (baseline)",      "cog_tmt2_dur_bl",                      "cont"),
    ("TMT-B/A Ratio, log (baseline)",       "cog_tmt_ratio_log_bl",                 "cont"),
    # ── Cognitive markers at LATEST (max available across all visits) ──────────
    ("Fluid Intelligence Score (latest)",   "cog_fluid_intelligence_latest",        "cont"),
    ("Reaction Time, ms (latest)",          "cog_react_time_latest",                "cont"),
    ("Numeric Memory (latest)",             "cog_numeric_memory_latest",            "cont"),
    ("Pairs Matching Status (latest)",      "cog_pairs_status_latest",              "cont"),
    ("TMT-A Duration, sec (latest)",        "cog_tmt1_dur_latest",                  "cont"),
    ("TMT-B Duration, sec (latest)",        "cog_tmt2_dur_latest",                  "cont"),
    ("TMT-B/A Ratio, log (latest)",         "cog_tmt_ratio_log_latest",             "cont"),
    # ── Cognitive change (latest − baseline) ───────────────────────────────────
    ("Fluid Intelligence (Δ latest−bl)",    "cog_fluid_intelligence_delta",         "cont"),
    ("Reaction Time (Δ latest−bl, ms)",     "cog_react_time_delta",                 "cont"),
    ("Numeric Memory (Δ latest−bl)",        "cog_numeric_memory_delta",             "cont"),
    ("Pairs Matching (Δ latest−bl)",        "cog_pairs_status_delta",               "cont"),
    ("SDS Accuracy (Δ latest−bl)",          "cog_sds_accuracy_delta",               "cont"),
    ("TMT-A Duration (Δ latest−bl, sec)",   "cog_tmt1_dur_delta",                   "cont"),
    ("TMT-B Duration (Δ latest−bl, sec)",   "cog_tmt2_dur_delta",                   "cont"),
    ("TMT-B/A Ratio (Δ latest−bl, log)",    "cog_tmt_ratio_log_delta",              "cont"),
    # ── Prodromal markers at BASELINE (pre-baseline HES + medication) ──────────
    ("Constipation (baseline)",             "prodromal_constipation_bl",            "bin"),
    ("Depression (baseline)",               "prodromal_depression_bl",              "bin"),
    ("Anxiety (baseline)",                  "prodromal_anxiety_bl",                 "bin"),
    ("Orthostatic Hypotension (baseline)",  "prodromal_orthostatic_bl",             "bin"),
    ("Erectile Dysfunction (baseline)",     "prodromal_erectile_dysfunction_bl",    "bin"),
    ("Dream Enactment (baseline)",          "prodromal_dream_enactment_bl",         "bin"),
    ("Anosmia (baseline)",                  "prodromal_anosmia_bl",                 "bin"),
    ("Hyposmia (baseline)",                 "prodromal_hyposmia_bl",                "bin"),
    ("Prodromal burden (baseline count)",   "prodromal_burden_bl",                  "cont"),
    # ── Prodromal status at FOLLOW-UP (post-baseline) ────────────────────────
    ("Constipation (follow-up)",            "prodromal_constipation_post",          "bin"),
    ("Depression (follow-up)",              "prodromal_depression_post",            "bin"),
    ("Anxiety (follow-up)",                 "prodromal_anxiety_post",               "bin"),
    ("Orthostatic Hypotension (follow-up)", "prodromal_orthostatic_post",           "bin"),
    ("Erectile Dysfunction (follow-up)",    "prodromal_erectile_dysfunction_post",  "bin"),
    ("Dream Enactment (follow-up)",         "prodromal_dream_enactment_post",       "bin"),
    ("Anosmia (follow-up)",                 "prodromal_anosmia_post",               "bin"),
    ("Hyposmia (follow-up)",                "prodromal_hyposmia_post",              "bin"),
    ("Prodromal burden (follow-up count)",  "prodromal_burden_post",                "cont"),
    # ── Prodromal NEW-ONSET during follow-up (incident) ────────────────────────
    ("Constipation (incident: new-onset)", "prodromal_constipation_delta_incident", "bin"),
    ("Depression (incident: new-onset)",    "prodromal_depression_delta_incident",   "bin"),
    ("Anxiety (incident: new-onset)",       "prodromal_anxiety_delta_incident",      "bin"),
    ("Orthostatic Hypotension (incident)", "prodromal_orthostatic_delta_incident",  "bin"),
    ("Erectile Dysfunction (incident)",    "prodromal_erectile_dysfunction_delta_incident", "bin"),
    ("Dream Enactment (incident)",          "prodromal_dream_enactment_delta_incident", "bin"),
    ("Anosmia (incident)",                  "prodromal_anosmia_delta_incident",      "bin"),
    ("Hyposmia (incident)",                 "prodromal_hyposmia_delta_incident",     "bin"),
    ("Prodromal burden change (Δ post−bl)", "prodromal_burden_delta",                "cont"),
    # ── Genetics ─────────────────────────────────────────────────────────────
    ("PD PRS (z-score)",                    "prs_score_pd",                         "cont"),
    ("RBD PRS (z-score)",                   "prs_score_rbd",                        "cont"),
    ("GBA carrier",                         "gba_carrier",                          "bin"),
]

POSTBASELINE_CONTROLS_SECTION_TRIGGERS: Dict[str, str] = {
    "cov_age_recruitment_21022":            "Demographics",
    "abk_rbd_score_mean":                   "RBD Score",
    "follow_up_years":                      "Follow-up",
    "cog_fluid_intelligence_bl":            "Cognitive Markers (Baseline, visit i0)",
    "cog_fluid_intelligence_latest":        "Cognitive Markers (Latest available assessment)",
    "cog_fluid_intelligence_delta":         "Cognitive Change (Latest − Baseline)",
    "prodromal_constipation_bl":            "Prodromal Markers (Baseline, pre-baseline)",
    "prodromal_constipation_post":          "Prodromal Status (Follow-up)",
    "prodromal_constipation_delta_incident": "Prodromal Change (New-onset during follow-up)",
    "prs_score_pd":                         "Genetics",
}

# ---------------------------------------------------------------------------
# Post-baseline temporal-window table (all participants, not controls-only).
#
# The baseline table is produced by the main VARIABLE_SPECS above (which now
# uses the _bl column convention).  This complementary spec uses the follow-up
# cognition (_fu), cognitive change (_delta), and incident post-baseline
# prodromal markers (prodromal_*_post) + prodromal_burden_post.
#
# Numeric-memory and pairs-matching have no i2 collection, so their _fu columns
# are all-NaN and render as missing — intentional.
# ---------------------------------------------------------------------------
POSTBASELINE_VARIABLE_SPECS: List[Tuple[str, str, str]] = [
    # ── Demographics ─────────────────────────────────────────────────────────
    ("Age at recruitment (years)",          "cov_age_recruitment_21022",            "cont"),
    ("Male sex",                            "cov_sex_31",                           "bin"),
    ("BMI (kg/m²)",                         "cov_bmi",                              "cont"),
    # ── RBD score ────────────────────────────────────────────────────────────
    ("RBD probability score (ABK)",         "abk_rbd_score_mean",                   "cont"),
    # ── Follow-up ────────────────────────────────────────────────────────────
    ("Follow-up time (years)",              "follow_up_years",                      "cont"),
    # ── Incident events (during follow-up) ───────────────────────────────────
    ("Incident PD",                         col_incident("outcome_1a_pd_only"),          "bin"),
    ("Incident PD or AD",                   col_incident("outcome_1b_pd_ad"),            "bin"),
    ("Incident Vascular Dementia",          col_incident("outcome_2a_vasculardementia"),    "bin"),
    ("Incident PD + Vascular Dementia",     col_incident("outcome_2b_pd_vasculardementia"), "bin"),
    ("Incident AD",                         col_incident("outcome_4a_ad_only"),             "bin"),
    # ── Cognitive markers (follow-up visit i2, _fu) ──────────────────────────
    ("Fluid Intelligence Score",            "cog_fluid_intelligence_fu",            "cont"),
    ("Reaction Time (ms)",                  "cog_react_time_fu",                    "cont"),
    ("Numeric Memory (max digits)*",        "cog_numeric_memory_fu",                "cont"),
    ("Pairs Matching Status*",              "cog_pairs_matching_fu",                "cont"),
    ("TMT-A Duration (sec)",                 "cog_tmt1_dur_fu",                      "cont"),
    ("TMT-B Duration (sec)",                 "cog_tmt2_dur_fu",                      "cont"),
    ("TMT-B/A Ratio (log)",                  "cog_tmt_ratio_log_fu",                 "cont"),
    # ── Cognitive change (follow-up − baseline, _delta) ──────────────────────
    ("Fluid Intelligence (Δ fu−bl)",        "cog_fluid_intelligence_delta",         "cont"),
    ("Reaction Time (Δ fu−bl, ms)",         "cog_react_time_delta",                 "cont"),
    ("Numeric Memory (Δ fu−bl)",            "cog_numeric_memory_delta",             "cont"),
    ("Pairs Matching (Δ fu−bl)",            "cog_pairs_status_delta",               "cont"),
    ("SDS Accuracy (Δ fu−bl)",              "cog_sds_accuracy_delta",               "cont"),
    ("TMT-A Duration (Δ fu−bl, sec)",       "cog_tmt1_dur_delta",                   "cont"),
    ("TMT-B Duration (Δ fu−bl, sec)",       "cog_tmt2_dur_delta",                   "cont"),
    ("TMT-B/A Ratio (Δ fu−bl, log)",        "cog_tmt_ratio_log_delta",              "cont"),
    # ── Prodromal markers at baseline (pre-baseline HES + medication, _bl) ────
    ("Constipation (baseline)",             "prodromal_constipation_bl",            "bin"),
    ("Depression (baseline)",               "prodromal_depression_bl",              "bin"),
    ("Anxiety (baseline)",                  "prodromal_anxiety_bl",                 "bin"),
    ("Orthostatic Hypotension (baseline)",  "prodromal_orthostatic_bl",             "bin"),
    ("Erectile Dysfunction (baseline)",     "prodromal_erectile_dysfunction_bl",    "bin"),
    ("Dream Enactment (baseline)",          "prodromal_dream_enactment_bl",         "bin"),
    ("Anosmia (baseline)",                  "prodromal_anosmia_bl",                 "bin"),
    ("Hyposmia (baseline)",                 "prodromal_hyposmia_bl",                "bin"),
    ("Prodromal burden (baseline count)",   "prodromal_burden_bl",                  "cont"),
    # ── Prodromal status at follow-up (post-baseline, _post) ─────────────────
    ("Constipation (follow-up)",            "prodromal_constipation_post",          "bin"),
    ("Depression (follow-up)",              "prodromal_depression_post",            "bin"),
    ("Anxiety (follow-up)",                 "prodromal_anxiety_post",               "bin"),
    ("Orthostatic Hypotension (follow-up)", "prodromal_orthostatic_post",           "bin"),
    ("Erectile Dysfunction (follow-up)",    "prodromal_erectile_dysfunction_post",  "bin"),
    ("Dream Enactment (follow-up)",         "prodromal_dream_enactment_post",       "bin"),
    ("Anosmia (follow-up)",                 "prodromal_anosmia_post",               "bin"),
    ("Hyposmia (follow-up)",                "prodromal_hyposmia_post",              "bin"),
    ("Prodromal burden (follow-up count)",  "prodromal_burden_post",                "cont"),
    # ── Prodromal change indicators ───────────────────────────────────────────
    ("Constipation (incident: new-onset)", "prodromal_constipation_delta_incident", "bin"),
    ("Depression (incident: new-onset)",    "prodromal_depression_delta_incident",   "bin"),
    ("Anxiety (incident: new-onset)",       "prodromal_anxiety_delta_incident",      "bin"),
    ("Orthostatic Hypotension (incident)", "prodromal_orthostatic_delta_incident",  "bin"),
    ("Erectile Dysfunction (incident)",    "prodromal_erectile_dysfunction_delta_incident", "bin"),
    ("Dream Enactment (incident)",          "prodromal_dream_enactment_delta_incident", "bin"),
    ("Anosmia (incident)",                  "prodromal_anosmia_delta_incident",      "bin"),
    ("Hyposmia (incident)",                 "prodromal_hyposmia_delta_incident",     "bin"),
    ("Prodromal burden change (Δ post−bl)", "prodromal_burden_delta",                "cont"),
    # ── Genetics ─────────────────────────────────────────────────────────────
    ("PD PRS (z-score)",                    "prs_score_pd",                         "cont"),
    ("RBD PRS (z-score)",                   "prs_score_rbd",                        "cont"),
    ("GBA carrier",                         "gba_carrier",                          "bin"),
]

POSTBASELINE_SECTION_TRIGGERS: Dict[str, str] = {
    "cov_age_recruitment_21022":        "Demographics",
    "abk_rbd_score_mean":               "RBD Score",
    "follow_up_years":                  "Follow-up",
    col_incident("outcome_1a_pd_only"): "Incident Events (during follow-up)",
    "cog_fluid_intelligence_fu":        "Cognitive Markers (follow-up visit i2)",
    "cog_fluid_intelligence_delta":     "Cognitive Change (follow-up − baseline)",
    "prodromal_constipation_bl":        "Prodromal Markers (baseline, pre-baseline)",
    "prodromal_constipation_post":      "Prodromal Status (follow-up)",
    "prodromal_constipation_incident":  "Prodromal Change (new-onset, post − baseline)",
    "prs_score_pd":                     "Genetics",
}

# ---------------------------------------------------------------------------
# Controls-only ANY-prodromal + cognitive-mean table.
#
# Purpose: for each subject show whether they EVER had each prodromal marker
# (baseline OR follow-up, OR-combination) and their mean cognitive score
# across all available UKBB instances (i0–i3).
#
# Prodromal ANY: prodromal_{marker}_any = bl | post  (binary, 0/1)
# Cognitive mean: row-wise mean of per-instance columns (skipna=True);
#   only columns present in the DataFrame are averaged.
#
# Output: table1_any_prodromal_percentile_3g_controls.xlsx
# ---------------------------------------------------------------------------

# Maps output column → ordered list of per-instance source columns.
# Uses cog_*_bl / cog_*_fu aliases for i0/i2; raw cov_* columns for i1/i3.
# Columns absent from the DataFrame are silently ignored in the mean.
_COG_MEAN_ALL_COMPUTATIONS: List[Tuple[str, List[str]]] = [
    ("cog_fluid_intelligence_mean_all", [
        "cog_fluid_intelligence_bl",
        "cov_fluid_intelligence_20016_i1",
        "cog_fluid_intelligence_fu",
        "cov_fluid_intelligence_20016_i3",
    ]),
    ("cog_react_time_mean_all", [
        "cog_react_time_bl",
        "cov_rt_mean_i1",
        "cog_react_time_fu",
        "cov_rt_mean_i3",
    ]),
    ("cog_fi_questions_mean_all", [
        "cov_fi_questions_attempted_20128_bl",
        "cov_fi_questions_attempted_20128_i1",
        "cov_fi_questions_attempted_20128_fu",
        "cov_fi_questions_attempted_20128_i3",
    ]),
    # i2 omitted: numeric_memory and pairs_matching have no i2 collection
    ("cog_numeric_memory_mean_all", [
        "cog_numeric_memory_bl",
        "cov_numeric_memory_max_20240_i1",
        "cov_numeric_memory_max_20240_i3",
    ]),
    ("cog_pairs_status_mean_all", [
        "cog_pairs_matching_bl",
        "cov_pairs_status_20244_i1",
        "cov_pairs_status_20244_i3",
    ]),
    # SDS and prospective memory: i0 and i1 only
    ("cog_sds_correct_per_min_mean_all", [
        "cov_sds_correct_per_min_i0",
        "cov_sds_correct_per_min_i1",
    ]),
    ("cog_sds_accuracy_mean_all", [
        "cov_sds_accuracy_i0",
        "cov_sds_accuracy_i1",
    ]),
    ("cog_prospective_memory_mean_all", [
        "cov_prospective_memory_6373_i0",
        "cov_prospective_memory_6373_i1",
    ]),
    # TMT: baseline (i0) and follow-up clinic visit (i2) only
    ("cog_tmt1_dur_mean_all", [
        "cog_tmt1_dur_bl",
        "cog_tmt1_dur_fu",
    ]),
    ("cog_tmt2_dur_mean_all", [
        "cog_tmt2_dur_bl",
        "cog_tmt2_dur_fu",
    ]),
    ("cog_tmt_ratio_log_mean_all", [
        "cog_tmt_ratio_log_bl",
        "cog_tmt_ratio_log_fu",
    ]),
]

ANY_PRODROMAL_COG_MEAN_VARIABLE_SPECS: List[Tuple[str, str, str]] = [
    # ── Demographics ─────────────────────────────────────────────────────────
    ("Age at recruitment (years)",              "cov_age_recruitment_21022",                "cont"),
    ("Male sex",                                "cov_sex_31",                               "bin"),
    ("BMI (kg/m²)",                             "cov_bmi",                                  "cont"),
    # ── RBD score ────────────────────────────────────────────────────────────
    ("RBD probability score (ABK)",             "abk_rbd_score_mean",                       "cont"),
    # ── Follow-up ────────────────────────────────────────────────────────────
    ("Follow-up time (years)",                  "follow_up_years",                          "cont"),
    # ── Cognitive (mean across all available instances) ───────────────────────
    ("Fluid Intelligence (mean, i0–i3)",        "cog_fluid_intelligence_mean_all",          "cont"),
    ("Reaction Time, ms (mean, i0–i3)",         "cog_react_time_mean_all",                  "cont"),
    ("FI Questions Attempted (mean, i0–i3)",    "cog_fi_questions_mean_all",                "cont"),
    ("Numeric Memory (mean, i0, i1, i3)",       "cog_numeric_memory_mean_all",              "cont"),
    ("Pairs Matching (mean, i0, i1, i3)",       "cog_pairs_status_mean_all",                "cont"),
    ("SDS Correct/min (mean, i0–i1)",           "cog_sds_correct_per_min_mean_all",         "cont"),
    ("SDS Accuracy (mean, i0–i1)",              "cog_sds_accuracy_mean_all",                "cont"),
    ("Prospective Memory (mean, i0–i1)",        "cog_prospective_memory_mean_all",          "cont"),
    ("TMT-A Duration, sec (mean, i0, i2)",      "cog_tmt1_dur_mean_all",                    "cont"),
    ("TMT-B Duration, sec (mean, i0, i2)",      "cog_tmt2_dur_mean_all",                    "cont"),
    ("TMT-B/A Ratio, log (mean, i0, i2)",       "cog_tmt_ratio_log_mean_all",               "cont"),
    # ── Prodromal markers (ANY: baseline OR follow-up) ────────────────────────
    ("Constipation (any)",                      "prodromal_constipation_any",               "bin"),
    ("Depression (any)",                        "prodromal_depression_any",                 "bin"),
    ("Anxiety (any)",                           "prodromal_anxiety_any",                    "bin"),
    ("Orthostatic Hypotension (any)",           "prodromal_orthostatic_any",                "bin"),
    ("Erectile Dysfunction (any)",              "prodromal_erectile_dysfunction_any",        "bin"),
    ("Dream Enactment (any)",                   "prodromal_dream_enactment_any",            "bin"),
    ("Anosmia (any)",                           "prodromal_anosmia_any",                    "bin"),
    ("Hyposmia (any)",                          "prodromal_hyposmia_any",                   "bin"),
    # ── Genetics ─────────────────────────────────────────────────────────────
    ("PD PRS (z-score)",                        "prs_score_pd",                             "cont"),
    ("RBD PRS (z-score)",                       "prs_score_rbd",                            "cont"),
    ("GBA carrier",                             "gba_carrier",                              "bin"),
]

ANY_PRODROMAL_COG_MEAN_SECTION_TRIGGERS: Dict[str, str] = {
    "cov_age_recruitment_21022":            "Demographics",
    "abk_rbd_score_mean":                   "RBD Score",
    "follow_up_years":                      "Follow-up",
    "cog_fluid_intelligence_mean_all":      "Cognitive Performance (mean across all available instances)",
    "prodromal_constipation_any":           "Prodromal Markers (ANY: baseline OR follow-up)",
    "prs_score_pd":                         "Genetics",
}


# ============================================================================
# DESCRIPTIVE STATISTICS HELPERS
# ============================================================================

def _cont_summary(series: pd.Series) -> Dict[str, float]:
    """Compute n, mean, SD, median, Q1, Q3 for a continuous variable.

    Parameters
    ----------
    series : raw column (coerced to float; NaN dropped internally)

    Returns
    -------
    dict with keys: n, mean, sd, median, q1, q3
    """
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan,
                "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {
        "n":      n,
        "mean":   float(x.mean()),
        "sd":     float(x.std(ddof=1)),
        "median": float(x.median()),
        "q1":     float(x.quantile(0.25)),
        "q3":     float(x.quantile(0.75)),
    }


def _bin_summary(series: pd.Series) -> Dict[str, float]:
    """Compute n, count of 1s, and percentage for a binary variable.

    Parameters
    ----------
    series : column that should contain 0/1 (coerced; NaN dropped internally)

    Returns
    -------
    dict with keys: n, count, pct
    """
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    k = int(x.sum())
    return {"n": n, "count": k, "pct": 100.0 * k / n if n > 0 else np.nan}


def _format_cont(s: Dict[str, float]) -> str:
    """Format continuous summary: 'mean (SD); median [Q1–Q3]'.

    Returns '—' when n == 0.
    """
    if s["n"] == 0 or not np.isfinite(s["mean"]):
        return "—"
    return (
        f"{s['mean']:.2f} ({s['sd']:.2f}); "
        f"{s['median']:.2f} [{s['q1']:.2f}–{s['q3']:.2f}]"
    )


def _format_bin(s: Dict[str, float]) -> str:
    """Format binary summary: 'N (%)'.

    Returns '—' when n == 0.
    """
    if s["n"] == 0:
        return "—"
    return f"{s['count']} ({s['pct']:.1f}%)"


def _format_pvalue(p: float) -> str:
    """Format p-value showing exact value.

    Rules
    -----
    * NaN / non-finite → '—'
    * otherwise        → 4 significant figures (scientific notation when < 0.001)
    """
    if not isinstance(p, (int, float)) or not np.isfinite(p):
        return "—"
    return f"{p:.4g}"


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def _test_continuous(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    groups: List[str],
) -> float:
    """Test between-group differences for a continuous variable.

    Uses Mann-Whitney U (2 groups) or Kruskal-Wallis (≥3 groups).
    Both are non-parametric and do not assume normality — appropriate for
    a descriptive epidemiological Table 1.

    Parameters
    ----------
    df        : data frame containing *col* and *group_col*
    col       : continuous variable column
    group_col : risk-group column
    groups    : ordered list of group labels to compare

    Returns
    -------
    p-value (float) or np.nan on failure
    """
    arrays = [
        pd.to_numeric(df.loc[df[group_col] == g, col], errors="coerce")
          .dropna()
          .values
        for g in groups
    ]
    # Require at least 3 observations per group for a valid test
    arrays_valid = [a for a in arrays if len(a) >= 3]
    if len(arrays_valid) < 2:
        return np.nan

    if len(arrays_valid) == 2:
        # Mann-Whitney U — two-sided, no normality assumption
        try:
            return float(stats.mannwhitneyu(*arrays_valid, alternative="two-sided").pvalue)
        except ValueError:
            return np.nan
    else:
        # Kruskal-Wallis — non-parametric analogue of one-way ANOVA
        try:
            return float(stats.kruskal(*arrays_valid).pvalue)
        except ValueError:
            # Raised when all values across groups are identical
            return np.nan


def _test_binary(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    groups: List[str],
) -> float:
    """Test between-group differences for a binary variable.

    Uses Fisher's exact test (2 × 2 with any expected cell < 5) or
    Pearson's χ² (all other cases).  For tables larger than 2 × 2,
    χ² is always used.

    Parameters
    ----------
    df        : data frame
    col       : binary variable (0/1)
    group_col : risk-group column
    groups    : ordered list of group labels

    Returns
    -------
    p-value (float) or np.nan on failure
    """
    df_cc = df[df[group_col].isin(groups)].copy()
    df_cc["_bin"] = pd.to_numeric(df_cc[col], errors="coerce")
    df_cc = df_cc.dropna(subset=["_bin", group_col])

    # Contingency table: rows = groups, columns = {0, 1}
    tab = pd.crosstab(df_cc[group_col], df_cc["_bin"])

    # Normalise column labels to integers (boolean/float columns in pandas
    # ≥2.0 use strict type matching; convert to int to avoid KeyError).
    tab.columns = [int(round(c)) for c in tab.columns]

    # Reindex to enforce (groups × [0, 1]) order and fill missing with 0
    tab = tab.reindex(index=[g for g in groups if g in tab.index])
    for val in [0, 1]:
        if val not in tab.columns:
            tab[val] = 0
    tab = tab[[0, 1]]

    if tab.shape[0] < 2:
        return np.nan

    # Fisher's exact for 2 × 2 when any observed cell is sparse or zero.
    # (Zero cells also violate chi-square expected-frequency assumptions.)
    if tab.shape == (2, 2) and ((tab < 5).any().any() or (tab == 0).any().any()):
        _, p = stats.fisher_exact(tab.values)
        return float(p)

    # χ² for all other cases; return NaN on numerical failure (e.g. zero
    # expected frequencies in larger tables with very rare events).
    try:
        _, p, _, _ = stats.chi2_contingency(tab.values, correction=False)
        return float(p)
    except ValueError:
        return np.nan


# ============================================================================
# TABLE BUILDER
# ============================================================================

def build_overview_table(
    df: pd.DataFrame,
    variable_specs: List[Tuple[str, str, str]],
) -> pd.DataFrame:
    """Build a single-column cohort overview table (Slide 1 — no stratification).

    Parameters
    ----------
    df             : subject-level DataFrame
    variable_specs : ordered list of (display_label, column_name, var_type)

    Returns
    -------
    pd.DataFrame with columns: Variable | Available N (%) | Overall (N=X)
    """
    n_total = len(df)
    overall_hdr = f"Overall (N={n_total:,})"
    rows: List[Dict] = []

    for label, col, vtype in variable_specs:
        if col not in df.columns:
            rows.append({
                "Variable":        f"  {label}",
                "Available N (%)": "column not in dataset",
                overall_hdr:       "—",
            })
            continue

        x_num = pd.to_numeric(df[col], errors="coerce")
        n_miss = int(x_num.isna().sum())
        n_avail = n_total - n_miss
        avail_str = f"{n_avail:,} ({100.0 * n_avail / n_total:.1f}%)" if n_total > 0 else "0"

        if vtype == "cont":
            fmt = _format_cont(_cont_summary(df[col]))
        else:
            fmt = _format_bin(_bin_summary(df[col]))

        rows.append({"Variable": f"  {label}", "Available N (%)": avail_str, overall_hdr: fmt})

    return pd.DataFrame(rows)


def build_table_one(
    df: pd.DataFrame,
    group_col: str,
    groups: List[str],
    variable_specs: List[Tuple[str, str, str]],
    section_triggers: Dict[str, str],
    n_prevalent_excluded: Optional[int] = None,
) -> pd.DataFrame:
    """Build a publication-quality Table 1 stratified by risk group.

    Parameters
    ----------
    df              : subject-level DataFrame (after all exclusions)
    group_col       : column name of the RBD risk group
    groups          : ordered group labels (e.g. ['Low', 'High'])
    variable_specs  : ordered list of (display_label, column_name, var_type)
    section_triggers: {column_name: section_title} mapping
    n_prevalent_excluded : optional int, number of prevalent cases excluded
                          (if provided, adds a cohort flow row at top)

    Returns
    -------
    pd.DataFrame with columns:
        Variable | Available N (%) | Overall (N=X) | <group_1> (N=X) | … | p-value

    Notes
    -----
    Continuous : mean (SD); median [IQR]
    Binary     : N (%)
    p-value    : Mann-Whitney U / Kruskal-Wallis (continuous),
                 χ² or Fisher's exact (binary)
    """
    n_total = len(df)
    group_ns: Dict[str, int] = {
        g: int((df[group_col] == g).sum()) for g in groups
    }

    # Column headers encode sample size
    overall_hdr = f"Overall\n(N={n_total:,})"
    group_hdrs: Dict[str, str] = {
        g: f"{g}\n(N={group_ns[g]:,})" for g in groups
    }

    rows: List[Dict] = []
    current_section: Optional[str] = None

    # ── Add cohort flow summary at top (if prevalent exclusion count provided) ──
    if n_prevalent_excluded is not None:
        n_total_before = n_total + n_prevalent_excluded
        rows.append({
            "Variable":        f"--- COHORT FLOW (n={n_total_before:,} baseline) ---",
            "Available N (%)": "",
            overall_hdr:       "",
            **{group_hdrs[g]: "" for g in groups},
            "p-value":         "",
        })
        rows.append({
            "Variable":        f"  Prevalent PD (excluded)",
            "Available N (%)": "",
            overall_hdr:       f"{n_prevalent_excluded:,}",
            **{group_hdrs[g]: "—" for g in groups},
            "p-value":         "—",
        })
        rows.append({
            "Variable":        f"  Analysed cohort (prevalent PD excluded)",
            "Available N (%)": "",
            overall_hdr:       f"{n_total:,}",
            **{group_hdrs[g]: f"{group_ns[g]:,}" for g in groups},
            "p-value":         "—",
        })

    for label, col, vtype in variable_specs:

        # ── Section header row ────────────────────────────────────────────
        section = section_triggers.get(col)
        if section and section != current_section:
            current_section = section
            rows.append({
                "Variable":        f"--- {section} ---",
                "Available N (%)": "",
                overall_hdr:       "",
                **{group_hdrs[g]: "" for g in groups},
                "p-value":         "",
            })

        # ── Column presence check ─────────────────────────────────────────
        if col not in df.columns:
            rows.append({
                "Variable":        f"  {label}",
                "Available N (%)": "column not in dataset",
                overall_hdr:       "—",
                **{group_hdrs[g]: "—" for g in groups},
                "p-value":         "—",
            })
            continue

        # ── Available count ───────────────────────────────────────────────
        x_num = pd.to_numeric(df[col], errors="coerce")
        n_miss = int(x_num.isna().sum())
        n_avail = n_total - n_miss
        avail_str = (
            f"{n_avail:,} ({100.0 * n_avail / n_total:.1f}%)"
            if n_total > 0 else "0"
        )

        # ── Summary and p-value ───────────────────────────────────────────
        if vtype == "cont":
            s_all = _cont_summary(df[col])
            fmt_all = _format_cont(s_all)
            group_vals = {
                group_hdrs[g]: _format_cont(
                    _cont_summary(df.loc[df[group_col] == g, col])
                )
                for g in groups
            }
            p = _test_continuous(df, col, group_col, groups)

        else:  # binary
            s_all = _bin_summary(df[col])
            fmt_all = _format_bin(s_all)
            group_vals = {
                group_hdrs[g]: _format_bin(
                    _bin_summary(df.loc[df[group_col] == g, col])
                )
                for g in groups
            }
            p = _test_binary(df, col, group_col, groups)

        rows.append({
            "Variable":        f"  {label}",
            "Available N (%)": avail_str,
            overall_hdr:       fmt_all,
            **group_vals,
            "p-value":         _format_pvalue(p),
        })

    return pd.DataFrame(rows)


# ============================================================================
# EXCEL WRITER HELPER
# ============================================================================

def _write_excel(
    path: Path,
    sheets: List[Tuple[pd.DataFrame, str]],
) -> None:
    """Write one or more DataFrames to an Excel workbook, one sheet each.

    Saves with openpyxl; handles PermissionError (file open in Excel)
    gracefully.  All Unicode characters (en-dashes in group labels, etc.)
    are preserved correctly in the xlsx binary format.

    Parameters
    ----------
    path   : output .xlsx path
    sheets : list of (DataFrame, sheet_name) pairs
    """
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for df, sheet in sheets:
                df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"  Saved: {path.name}")
    except PermissionError:
        warnings.warn(
            f"Cannot write '{path.name}' — file is open in Excel. "
            "Close it and re-run to refresh."
        )


# ============================================================================
# HELPERS
# ============================================================================

def _group_priority(lbl: str) -> int:
    """Map a risk-group label to a sort key: Low=0, Intermediate/Mid=1, High=2.

    Used to impose a consistent Low < Intermediate < High ordering regardless
    of how percentile annotations are appended to the label string.
    """
    ll = lbl.lower()
    if "low" in ll:
        return 0
    if "intermediate" in ll or "mid" in ll:
        return 1
    if "high" in ll:
        return 2
    return 3


# ============================================================================
# PREVALENT EXCLUSION TABLE (STROBE)
# ============================================================================

# Outcome labels for display in the prevalent exclusion table.
# Sourced from config.config (single source of truth).
OUTCOME_DISPLAY: List[Tuple[str, str]] = [
    (oc, outcomes_short_names[oc]) for oc in outcomes
]


def build_prevalent_exclusion_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a table of prevalent cases excluded per outcome (STROBE Item 13).

    For each composite outcome, reports the number of subjects who were
    diagnosed *before* the actigraphy baseline (prevalent) and therefore
    excluded from the survival / incident analysis.

    Parameters
    ----------
    df : subject-level DataFrame containing ``{outcome}_prevalent``,
         ``{outcome}_diagnosed``, and ``{outcome}_incident`` columns.

    Returns
    -------
    pd.DataFrame with columns: Outcome | Diagnosed | Prevalent (excluded) |
        Incident (analysed) | Prevalent %
    """
    n_total = len(df)
    rows: List[Dict] = []

    for outcome_col, display_label in OUTCOME_DISPLAY:
        diagnosed_col = col_dx(outcome_col)
        prevalent_col = col_prevalent(outcome_col)
        incident_col = col_incident(outcome_col)

        n_diagnosed = 0
        n_prevalent = 0
        n_incident = 0

        if diagnosed_col in df.columns:
            n_diagnosed = int(pd.to_numeric(
                df[diagnosed_col], errors="coerce"
            ).fillna(0).sum())
        if prevalent_col in df.columns:
            n_prevalent = int(pd.to_numeric(
                df[prevalent_col], errors="coerce"
            ).fillna(0).sum())
        if incident_col in df.columns:
            n_incident = int(pd.to_numeric(
                df[incident_col], errors="coerce"
            ).fillna(0).sum())

        pct_prevalent = (
            f"{100.0 * n_prevalent / n_diagnosed:.1f}%"
            if n_diagnosed > 0 else "—"
        )

        rows.append({
            "Outcome": display_label,
            "Total diagnosed": n_diagnosed,
            "Prevalent (excluded)": n_prevalent,
            "Incident (analysed)": n_incident,
            f"Prevalent % of diagnosed": pct_prevalent,
            "Cohort N": n_total,
        })

    return pd.DataFrame(rows)


# ============================================================================
# MAIN — PER-TABLE BUILDER FUNCTIONS
# ============================================================================

def _resolve_groups(df: pd.DataFrame, group_col: str) -> List[str]:
    """Return sorted risk-group labels present in *group_col* (Low < Mid < High)."""
    unique = [g for g in df[group_col].unique() if g not in ("nan", "None", "")]
    return sorted(unique, key=_group_priority)


def _build_main_table(
    df_subj: pd.DataFrame,
    n_prevalent_pd: int,
    path_results: Path,
) -> None:
    """Build Table 1 (main stratified baseline table) for all methods."""
    print("\n[2/6] Building main Table 1 …")
    excel_path = path_results / "table1_rbd_risk_groups.xlsx"
    tables_to_write: List[Tuple[pd.DataFrame, str, str, int]] = []

    for method in METHODS:
        group_col = col_risk_group_agnostic(method)
        if group_col not in df_subj.columns:
            warnings.warn(f"Risk group column '{group_col}' not in dataset — skipping {method}.")
            continue

        df_m = df_subj.dropna(subset=[group_col]).copy()
        df_m[group_col] = df_m[group_col].astype(str)
        groups = _resolve_groups(df_m, group_col)

        if len(groups) < 2:
            warnings.warn(f"Fewer than 2 risk groups for {method} ({groups}) — skipping.")
            continue

        n_groups_int = len(groups)
        print(f"\n  {method}: N={len(df_m):,}, groups={groups}")
        _print_group_event_counts(df_m, group_col, groups)

        table = build_table_one(
            df               = df_m,
            group_col        = group_col,
            groups           = groups,
            variable_specs   = VARIABLE_SPECS,
            section_triggers = SECTION_TRIGGERS,
            n_prevalent_excluded = n_prevalent_pd,
        )

        sheet = "2-group (Low-High)" if n_groups_int == 2 else "3-group (Low-Mid-High)"
        tables_to_write.append((table, sheet, method, n_groups_int))

        sep = "=" * 100
        print(f"\n{sep}")
        print(
            f"  TABLE 1  |  METHOD: {method.upper()}"
            f"  |  Stratified by RBD Risk Group  |  N={len(df_m):,}"
        )
        print(
            "  Continuous: mean (SD); median [IQR]"
            "   Binary: N (%)"
            f"   Test: {'Mann-Whitney U' if n_groups_int == 2 else 'Kruskal-Wallis'}"
            " / chi-sq or Fisher"
        )
        print(sep)
        tbl_display = table.rename(
            columns=lambda c: c.replace("\n", " ") if isinstance(c, str) else c
        )
        print(tbl_display.to_string(index=False))

    if not tables_to_write:
        warnings.warn("No tables produced — Excel files not written.")
        return

    _write_excel(excel_path, [(tbl, sheet) for tbl, sheet, _, _ in tables_to_write])
    for tbl, sheet, method, _ in tables_to_write:
        _write_excel(path_results / f"table1_{method}.xlsx", [(tbl, "Table 1")])

    print(f"\n[3/6] Combined workbook: {excel_path.name}")
    for _, _, method, _ in tables_to_write:
        print(f"       Per-method file : table1_{method}.xlsx")
    print(f"       Output directory: {path_results}")


def _build_supplementary_hes_table(
    df_subj: pd.DataFrame,
    path_results: Path,
) -> None:
    """Build supplementary Table S1 (HES-only and medication-only flags)."""
    print("\n[4/6] Building supplementary table (HES + medication flags) …")
    for method in METHODS:
        group_col = col_risk_group_agnostic(method)
        if group_col not in df_subj.columns:
            continue
        df_m = df_subj.dropna(subset=[group_col]).copy()
        df_m[group_col] = df_m[group_col].astype(str)
        groups = _resolve_groups(df_m, group_col)
        tbl = build_table_one(
            df               = df_m,
            group_col        = group_col,
            groups           = groups,
            variable_specs   = SUPPLEMENTARY_VARIABLE_SPECS,
            section_triggers = SUPPLEMENTARY_SECTION_TRIGGERS,
        )
        _write_excel(
            path_results / f"tableS1_hes_medication_{method}.xlsx",
            [(tbl, "Table S1 — HES + Medication")],
        )


def _build_prevalent_exclusion(
    df_subj: pd.DataFrame,
    path_results: Path,
) -> None:
    """Build Table S2 (prevalent cases excluded, STROBE Item 13)."""
    print("\n[5/6] Building prevalent exclusion table …")
    tbl = build_prevalent_exclusion_table(df_subj)
    _write_excel(path_results / "tableS2_prevalent_exclusions.xlsx",
                 [(tbl, "Prevalent Exclusions")])
    sep = "=" * 80
    print(f"\n{sep}")
    print("  TABLE S2 — PREVALENT CASES EXCLUDED (STROBE Item 13)")
    print(sep)
    print(tbl.to_string(index=False))
    print(sep)


def _build_slide_tables(
    df_subj: pd.DataFrame,
    path_results: Path,
) -> None:
    """Build presentation slide tables (overview + stratified condensed views)."""
    print("\n[6/6] Building presentation slide tables …")
    tbl_slide1 = build_overview_table(df_subj, SLIDE_VARIABLE_SPECS)
    _write_excel(path_results / "slide1_overview.xlsx", [(tbl_slide1, "Slide 1 — Overview")])

    for method in METHODS:
        group_col = col_risk_group_agnostic(method)
        if group_col not in df_subj.columns:
            warnings.warn(f"Slide table skipped for {method} — risk group column absent.")
            continue
        df_m = df_subj.dropna(subset=[group_col]).copy()
        df_m[group_col] = df_m[group_col].astype(str)
        groups = _resolve_groups(df_m, group_col)
        if len(groups) < 2:
            continue
        tbl_slide = build_table_one(
            df               = df_m,
            group_col        = group_col,
            groups           = groups,
            variable_specs   = SLIDE_VARIABLE_SPECS,
            section_triggers = SLIDE_SECTION_TRIGGERS,
        )
        slide_num  = 2 if len(groups) == 2 else 3
        slide_name = "2-group (Low-High)" if slide_num == 2 else "3-group (Low-Mid-High)"
        _write_excel(path_results / f"slide{slide_num}_{method}.xlsx",
                     [(tbl_slide, slide_name)])

    print(f"       Slide files written to: {path_results}")


def _build_controls_postbaseline_table(
    df_subj: pd.DataFrame,
    path_results: Path,
) -> None:
    """Build controls-only post-baseline Table 1 (3-group, baseline + latest + delta)."""
    print("\n[7/7] Building controls-only post-baseline table (3-group) …")
    if "control" not in df_subj.columns:
        warnings.warn("Column 'control' not found — controls-only post-baseline table skipped.")
        return

    df_ctrl = df_subj[df_subj["control"].fillna(False).astype(bool)].copy()
    n_removed = len(df_subj) - len(df_ctrl)
    print(
        f"  Controls-only cohort: {len(df_ctrl):,} subjects "
        f"({n_removed:,} cases with any incident outcome removed)"
    )

    group_col = col_risk_group_agnostic("percentile_3g")
    if group_col not in df_ctrl.columns:
        warnings.warn(f"Risk group column '{group_col}' not found — skipped.")
        return

    df_ct_m = df_ctrl.dropna(subset=[group_col]).copy()
    df_ct_m[group_col] = df_ct_m[group_col].astype(str)
    groups = _resolve_groups(df_ct_m, group_col)

    if len(groups) < 2:
        warnings.warn(f"Fewer than 2 groups in controls-only cohort ({groups}) — skipped.")
        return

    tbl = build_table_one(
        df               = df_ct_m,
        group_col        = group_col,
        groups           = groups,
        variable_specs   = POSTBASELINE_CONTROLS_VARIABLE_SPECS,
        section_triggers = POSTBASELINE_CONTROLS_SECTION_TRIGGERS,
    )

    out_path = path_results / "table1_postbaseline_percentile_3g_controls.xlsx"
    _write_excel(out_path, [(tbl, "Controls only (3-group)")])
    print(f"       Output: {out_path.name}")

    sep = "=" * 100
    print(f"\n{sep}")
    print(
        f"  TABLE — CONTROLS ONLY (POST-BASELINE)  |  percentile_3g"
        f"  |  N={len(df_ct_m):,}  |  All incident cases removed (N={n_removed:,})"
    )
    print("  Continuous: mean (SD); median [IQR]   Binary: N (%)   Test: Kruskal-Wallis / chi-sq or Fisher")
    print(sep)
    tbl_display = tbl.rename(
        columns=lambda c: c.replace("\n", " ") if isinstance(c, str) else c
    )
    try:
        print(tbl_display.to_string(index=False))
    except UnicodeEncodeError:
        print(f"(Table display skipped due to encoding; data saved to {out_path.name})")


def _build_postbaseline_all_table(
    df_subj: pd.DataFrame,
    n_prevalent_pd: int,
    path_results: Path,
) -> None:
    """Build post-baseline table for all subjects (incident outcomes + _fu cognition + _post prodromal)."""
    print("\n[8/8] Building post-baseline temporal-window table (all subjects) …")
    for method in METHODS:
        group_col = col_risk_group_agnostic(method)
        if group_col not in df_subj.columns:
            continue

        df_tw = df_subj.dropna(subset=[group_col]).copy()
        df_tw[group_col] = df_tw[group_col].astype(str)
        groups = _resolve_groups(df_tw, group_col)

        if len(groups) < 2:
            warnings.warn(f"Fewer than 2 groups for postbaseline table ({method}) — skipped.")
            continue

        tbl = build_table_one(
            df               = df_tw,
            group_col        = group_col,
            groups           = groups,
            variable_specs   = POSTBASELINE_VARIABLE_SPECS,
            section_triggers = POSTBASELINE_SECTION_TRIGGERS,
            n_prevalent_excluded = n_prevalent_pd,
        )

        out_path = path_results / f"table1_postbaseline_{method}_all.xlsx"
        _write_excel(out_path, [(tbl, "All subjects (3-group)")])
        print(f"       Output: {out_path.name}")


def _build_any_prodromal_table(
    df_subj: pd.DataFrame,
    path_results: Path,
) -> None:
    """Build controls-only table with ANY-prodromal flags and cognitive mean across all instances."""
    print("\n[9/9] Building controls-only any-prodromal + cognitive-mean table …")
    if "control" not in df_subj.columns:
        warnings.warn("Column 'control' not found — any-prodromal table skipped.")
        return

    df_any = df_subj.copy()

    # ── Prodromal ANY flags (bl OR post) ─────────────────────────────────
    for marker in PRODROMAL_MARKERS:
        col_bl_ = f"prodromal_{marker}_bl"
        col_post_ = f"prodromal_{marker}_post"
        col_any_ = f"prodromal_{marker}_any"
        if col_bl_ in df_any.columns and col_post_ in df_any.columns:
            bl_mask = pd.to_numeric(df_any[col_bl_], errors="coerce").fillna(0).astype(bool)
            post_mask = pd.to_numeric(df_any[col_post_], errors="coerce").fillna(0).astype(bool)
            df_any[col_any_] = (bl_mask | post_mask).astype(int)
        elif col_bl_ in df_any.columns:
            df_any[col_any_] = (
                pd.to_numeric(df_any[col_bl_], errors="coerce")
                .fillna(0).clip(upper=1).astype(int)
            )

    # ── Cognitive mean across all available instances ─────────────────────
    for target_col, source_cols in _COG_MEAN_ALL_COMPUTATIONS:
        avail = [c for c in source_cols if c in df_any.columns]
        if not avail:
            continue
        numeric_block = df_any[avail].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        df_any[target_col] = numeric_block.mean(axis=1, skipna=True)

    # ── Filter to controls only ───────────────────────────────────────────
    df_any_ctrl = df_any[df_any["control"].fillna(False).astype(bool)].copy()
    n_removed = len(df_any) - len(df_any_ctrl)
    print(
        f"  Controls-only cohort: {len(df_any_ctrl):,} subjects "
        f"({n_removed:,} incident cases removed)"
    )

    group_col = col_risk_group_agnostic("percentile_3g")
    if group_col not in df_any_ctrl.columns:
        warnings.warn(f"Risk group column '{group_col}' not found — any-prodromal table skipped.")
        return

    df_any_m = df_any_ctrl.dropna(subset=[group_col]).copy()
    df_any_m[group_col] = df_any_m[group_col].astype(str)
    groups = _resolve_groups(df_any_m, group_col)

    if len(groups) < 2:
        warnings.warn(f"Fewer than 2 groups ({groups}) — any-prodromal table skipped.")
        return

    tbl = build_table_one(
        df               = df_any_m,
        group_col        = group_col,
        groups           = groups,
        variable_specs   = ANY_PRODROMAL_COG_MEAN_VARIABLE_SPECS,
        section_triggers = ANY_PRODROMAL_COG_MEAN_SECTION_TRIGGERS,
    )

    out_path = path_results / "table1_any_prodromal_percentile_3g_controls.xlsx"
    _write_excel(out_path, [(tbl, "Controls — any prodromal + cog mean")])
    print(f"       Output: {out_path.name}")

    sep = "=" * 100
    print(f"\n{sep}")
    print(
        f"  TABLE — CONTROLS ONLY (ANY PRODROMAL + COG MEAN)  |  percentile_3g"
        f"  |  N={len(df_any_m):,}  |  Incident cases removed (N={n_removed:,})"
    )
    print("  Continuous: mean (SD); median [IQR]   Binary: N (%)   Test: Kruskal-Wallis / chi-sq or Fisher")
    print(sep)
    tbl_display = tbl.rename(
        columns=lambda c: c.replace("\n", " ") if isinstance(c, str) else c
    )
    try:
        print(tbl_display.to_string(index=False))
    except UnicodeEncodeError:
        print(f"(Table display skipped due to encoding; saved to {out_path.name})")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Load data, prepare subject-level dataset, and dispatch to per-table builder functions."""

    # ── 1. Paths ─────────────────────────────────────────────────────────────
    dir_final    = config["pp"]["final_dir"]
    dir_thresh   = config["pp"]["thresholds"]["root"]
    path_results = config["results"]["root"] / "table_one"
    path_results.mkdir(parents=True, exist_ok=True)
    col_irbd = "abk_rbd_score_mean"

    # ── 2. Load & prepare subject-level data ──────────────────────────────────
    print("[1/6] Loading data …")
    thresholds, df_risk = get_clean_risk_data(
        file_name="ehr_diag_pd_rbd_only_all",
        thresholds_root=dir_thresh,
        final_dir=dir_final,
    )
    df_subj = make_subject_level(df_risk, id_col="eid", prob_col=col_irbd)
    if "rbd_prob" in df_subj.columns and col_irbd not in df_subj.columns:
        df_subj[col_irbd] = df_subj["rbd_prob"]
    print(f"  Subject-level N (before cohort filter) = {len(df_subj):,}")

    # ── Prevalent exclusion ───────────────────────────────────────────────────
    surv_col      = col_surv_time(STRATIFICATION_OUTCOME)
    prevalent_col = col_prevalent(STRATIFICATION_OUTCOME)
    n_prevalent_pd = 0
    if prevalent_col in df_subj.columns:
        n_prevalent_pd = int(
            pd.to_numeric(df_subj[prevalent_col], errors="coerce").fillna(0).sum()
        )
    if surv_col in df_subj.columns:
        n_before  = len(df_subj)
        df_subj   = df_subj[df_subj[surv_col].notna()].copy()
        n_excluded = n_before - len(df_subj)
        if n_excluded > 0:
            print(f"  Prevalent exclusion (NaN surv_time): {n_before:,} -> {len(df_subj):,} "
                  f"(removed {n_excluded:,} cases)")
        if n_prevalent_pd > 0:
            print(f"  Prevalent PD count: {n_prevalent_pd:,}")

    # ── Derive follow-up years ────────────────────────────────────────────────
    if surv_col in df_subj.columns:
        df_subj["follow_up_years"] = (
            pd.to_numeric(df_subj[surv_col], errors="coerce") / 365.25
        )
    else:
        warnings.warn(f"Survival time column '{surv_col}' not found; follow_up_years omitted.")

    # ── Coerce numeric base covariates ────────────────────────────────────────
    for c in ["cov_age_recruitment_21022", "cov_bmi",
              "cov_sex_31", "cov_pairs_status_20244_i0", "cog_pairs_matching_bl"]:
        if c in df_subj.columns:
            df_subj[c] = pd.to_numeric(df_subj[c], errors="coerce")

    # ── Prodromal burden, incident flags, and cognitive change deltas ─────────
    baseline_cols = [f"prodromal_{m}_bl" for m in PRODROMAL_MARKERS
                     if f"prodromal_{m}_bl" in df_subj.columns]
    if baseline_cols and "prodromal_burden_bl" not in df_subj.columns:
        df_subj["prodromal_burden_bl"] = (
            df_subj[baseline_cols]
            .apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0).astype(int))
            .sum(axis=1)
        )

    for marker in PRODROMAL_MARKERS:
        col_bl_   = f"prodromal_{marker}_bl"
        col_post_ = f"prodromal_{marker}_post"
        col_delta_ = f"prodromal_{marker}_delta_incident"
        if col_bl_ in df_subj.columns and col_post_ in df_subj.columns:
            bl   = pd.to_numeric(df_subj[col_bl_],   errors="coerce").fillna(0).astype(bool)
            post = pd.to_numeric(df_subj[col_post_], errors="coerce").fillna(0).astype(bool)
            df_subj[col_delta_] = (~bl & post).astype(int)

    incident_cols = [f"prodromal_{m}_delta_incident" for m in PRODROMAL_MARKERS
                     if f"prodromal_{m}_delta_incident" in df_subj.columns]
    if incident_cols and "prodromal_burden_delta" not in df_subj.columns:
        df_subj["prodromal_burden_delta"] = df_subj[incident_cols].sum(axis=1)

    _cog_delta_pairs: List[Tuple[str, str, str]] = [
        ("cog_numeric_memory_latest",  "cog_numeric_memory_bl",   "cog_numeric_memory_delta"),
        ("cog_pairs_status_latest",    "cog_pairs_matching_bl",   "cog_pairs_status_delta"),
        ("cog_sds_accuracy_latest",    "cog_sds_accuracy_bl",     "cog_sds_accuracy_delta"),
        ("cog_tmt1_dur_latest",        "cog_tmt1_dur_bl",         "cog_tmt1_dur_delta"),
        ("cog_tmt2_dur_latest",        "cog_tmt2_dur_bl",         "cog_tmt2_dur_delta"),
        ("cog_tmt_ratio_log_latest",   "cog_tmt_ratio_log_bl",    "cog_tmt_ratio_log_delta"),
    ]
    for col_lat, col_bl_, col_d in _cog_delta_pairs:
        if (col_lat in df_subj.columns and col_bl_ in df_subj.columns
                and col_d not in df_subj.columns):
            df_subj[col_d] = (
                pd.to_numeric(df_subj[col_lat], errors="coerce")
                - pd.to_numeric(df_subj[col_bl_], errors="coerce")
            )

    print(f"  Analytical cohort N = {len(df_subj):,}")
    _print_availability_report(df_subj, VARIABLE_SPECS)

    # ── 3. Dispatch to per-table builder functions ────────────────────────────
    _build_main_table(df_subj, n_prevalent_pd, path_results)
    _build_supplementary_hes_table(df_subj, path_results)
    _build_prevalent_exclusion(df_subj, path_results)
    _build_slide_tables(df_subj, path_results)
    _build_controls_postbaseline_table(df_subj, path_results)
    _build_postbaseline_all_table(df_subj, n_prevalent_pd, path_results)
    _build_any_prodromal_table(df_subj, path_results)

    print("\nDone.")


# ============================================================================
# AUXILIARY REPORTING
# ============================================================================

def _print_availability_report(
    df: pd.DataFrame,
    variable_specs: List[Tuple[str, str, str]],
) -> None:
    """Print a data availability report for all analysis variables.

    Parameters
    ----------
    df             : subject-level DataFrame
    variable_specs : list of (label, column, var_type)
    """
    n = len(df)
    print(f"\n{'-' * 68}")
    print(f"  DATA AVAILABILITY  (N = {n:,})")
    print(f"{'-' * 68}")
    print(f"  {'Variable':<50} {'N avail':>8}  {'%':>6}")
    print(f"  {'-'*50} {'-'*8}  {'-'*6}")
    for label, col, _ in variable_specs:
        if col not in df.columns:
            print(f"  {label:<50} {'MISSING':>8}  {'—':>6}")
            continue
        n_avail = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        pct     = 100.0 * n_avail / n if n > 0 else 0.0
        print(f"  {label:<50} {n_avail:>8,}  {pct:>5.1f}%")
    print(f"{'-' * 68}\n")


def _print_group_event_counts(
    df: pd.DataFrame,
    group_col: str,
    groups: List[str],
) -> None:
    """Print N and incident event counts per risk group for primary outcomes.

    Parameters
    ----------
    df        : subject-level DataFrame
    group_col : risk group column
    groups    : ordered group labels
    """
    outcome_cols = [
        (col_incident("outcome_1a_pd_only"),               "Incident PD"),
        (col_incident("outcome_4a_ad_only"),                "Incident AD"),
        (col_incident("outcome_2a_vasculardementia"),       "Incident Vascular Dementia"),
        (col_incident("outcome_2b_pd_vasculardementia"),    "Incident PD + Vasc. Dementia"),
    ]
    print(f"\n  {'Group':<8}", end="")
    print(f"{'N':>8}", end="")
    for _, lbl in outcome_cols:
        if any(c in df.columns for c, _ in outcome_cols):
            print(f"  {lbl:<28}", end="")
    print()
    print(f"  {'-'*8}", end="")
    print(f"  {'-'*8}", end="")
    for col, _ in outcome_cols:
        if col in df.columns:
            print(f"  {'-'*28}", end="")
    print()

    for g in groups:
        mask   = df[group_col] == g
        n_grp  = mask.sum()
        print(f"  {g:<8}  {n_grp:>8,}", end="")
        for col, _ in outcome_cols:
            if col not in df.columns:
                continue
            n_ev = int(pd.to_numeric(df.loc[mask, col], errors="coerce").sum())
            pct  = 100.0 * n_ev / n_grp if n_grp > 0 else 0.0
            print(f"  {n_ev:>4} ({pct:4.1f}%)          ", end="")
        print()


# ============================================================================
if __name__ == "__main__":
    main()
