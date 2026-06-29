# Epidemiological Review: Analytical Changes and Bias Justification

**Project:** UKBB RBD × Prodromal Markers → Incident Neurodegeneration
**Date:** 2026-03-06
**Authors:** (research engineering record)

---

## 1. Overview

This document records the epidemiological rationale for three substantive changes made to the analytical pipeline prior to the primary analysis run. Each change targets a specific threat to internal validity. The document is written to support transparent Methods reporting in the scientific paper.

---

## 2. Change A — Restriction of Prodromal Markers to Pre-Baseline Records

### 2.1 Threat Addressed: Reverse Causation Bias

**Definition.** Reverse causation (also called *protopathic bias*) arises when the putative exposure is caused by the early, subclinical form of the outcome rather than the other way around. In a Cox survival model the consequence is upward bias in the hazard ratio: the marker appears to predict disease because it was *produced by* incipient disease, not because it precedes it.

**Mechanism here.** Prodromal markers (constipation, depression, anxiety, orthostatic hypotension, erectile dysfunction, dream enactment, anosmia/hyposmia) were derived from two sources:

1. **HES diagnostic codes** (ICD-10, p41270 / date p41280)
2. **Medication dispensing records** (p42039 / date p42040)

Both sources span the full follow-up period in the raw UKBB data. Without a temporal restriction, a subject who received an ICD-10 code for constipation *after* their actigraphy wear (i.e., after the RBD score was derived) would have been classified as `prodromal_constipation = 1`. If that diagnosis reflects early gut-brain Lewy body spread, the marker is a *consequence* of the same pathology we are trying to predict, not an independent antecedent. Fitting a Cox model on this contaminated exposure variable produces a biased HR.

### 2.2 The Fix

In `library/ehr_outcomes/covariates.py :: merge_prodromal_markers`, all HES and medication flag derivations are now restricted to records where the record date strictly precedes `wear_time_start` (the date the actigraphy device was issued, used as the analytical baseline):

```python
hes_pre_baseline: pd.Series = hes_date < baseline   # NaT → False (conservative)
hes_flag = hes_flag_raw & hes_pre_baseline
```

**Conservative handling of missing dates.** Records with `NaT` dates cannot be verified as pre-baseline and are treated as `False` (unexposed). This is conservative — it may undercount exposed subjects — but it avoids introducing bias by including temporally unverifiable records. The direction of this measurement error is non-differential with respect to outcome (the RBD score was already assigned), so it would attenuate HRs toward the null rather than inflate them.

### 2.3 Relationship to Prevalent Case Exclusion

Prevalent cases (subjects already diagnosed before actigraphy) are excluded from the survival dataset by `select_survival_dataset` via the `{outcome}_incident` flag. This is a separate, complementary exclusion. Prevalent case exclusion targets the *outcome* side of bias; the pre-baseline restriction targets the *exposure* side. Both are necessary for a temporally valid cohort design.

### 2.4 What to Report in the Paper

**Methods paragraph (suggested):**

> Prodromal markers were derived from Hospital Episode Statistics (HES) diagnostic codes (ICD-10, field p41270/p41280) and medication dispensing records (field p42039/p42040). To prevent reverse causation, only records dated strictly before each participant's actigraphy wear date were included. Records with missing dates were treated as post-baseline (unexposed). The analysis was therefore restricted to a pre-exposure window, consistent with standard prospective cohort methodology. Prevalent neurological cases (participants with a diagnosis prior to actigraphy) were additionally excluded.

**Supplementary note:** Report the counts of excluded post-baseline HES and medication records per marker. These are logged in `data_availability_report.csv` under the columns `excluded_post_baseline_hes` and `excluded_post_baseline_med`.

---

## 3. Change B — Correction of Dementia Outcome Ascertainment

### 3.1 Threat Addressed: Outcome Misclassification and Informative Censoring

**HES data limitation.** HES records in the UKBB are available to approximately March 2023. Studies using HES exclusively for outcome ascertainment therefore censor all dementia cases that occurred between March 2023 and the administrative end of follow-up (2025). This is not random censoring — subjects with later diagnoses are systematically excluded, and since later diagnoses skew toward older, less-impaired participants, selective censoring distorts survival estimates.

**First-occurrence fields.** UKBB Category 2410 provides algorithmically derived first-occurrence date fields that incorporate primary care records and death registrations in addition to HES. Relevant fields:

| Field | Condition |
|-------|-----------|
| p42018 | Dementia (all-cause) |
| p42022 | Vascular dementia |
| p42024 | Frontotemporal dementia |

These fields are updated to 2025 and extend ascertainment ~2 years beyond the HES horizon.

### 3.2 The Fix

For the composite dementia outcome (`dem`), both sources are now computed independently and the earlier date is retained:

```python
df[dx_date_col] = pd.concat(
    [fo_date.rename("fo"), hes_date.rename("hes")], axis=1
).min(axis=1)
```

**Why minimum, not maximum.** We want the *first* clinical recognition of disease, not the last. Taking the minimum ensures that if a subject appears in HES in 2021 and in the first-occurrence field in 2019 (e.g., a GP diagnosis recorded later in HES), the 2019 date is used. This is also consistent with incident case ascertainment: we exclude any subject whose first diagnosis date falls before actigraphy (prevalent case filter applied subsequently).

**DLB (Dementia with Lewy Bodies).** No dedicated first-occurrence field exists for DLB in UKBB. DLB remains HES-only. This is a data limitation acknowledged in the paper; HRs for the `outcome_3a_dlb_only` outcome are subject to outcome misclassification past March 2023.

### 3.3 What to Report in the Paper

**Methods paragraph (suggested):**

> Dementia outcomes were ascertained using two complementary sources: Hospital Episode Statistics (HES, ICD-10 codes, available to March 2023) and UKBB algorithmically derived first-occurrence date fields (Category 2410: p42018 all-cause dementia, p42022 vascular dementia, p42024 frontotemporal dementia; available to 2025). For each participant the earliest date from either source was used as the diagnosis date, maximising ascertainment sensitivity and extending follow-up beyond the HES data horizon. Dementia with Lewy bodies (DLB) could not be ascertained through first-occurrence fields and is therefore restricted to HES records; DLB results should be interpreted with awareness of potential right-truncation from 2023.

**Diagnostic audit table:** `results/logs/outcome_flags/outcome_summary.csv` — reports N diagnosed, earliest/latest diagnosis dates, and `_dx_source` tag per outcome. Include this as a supplementary table.

---

## 4. Change C — Per-Subject HES Activity Gap: Sensitivity Analysis for Exposure Misclassification

### 4.1 Threat Addressed: Non-Differential Exposure Misclassification in Binary Prodromal Markers

### 4.1.1 The Problem

Binary prodromal markers derived from HES are valid only for subjects with *adequate* HES coverage. A subject whose last HES contact occurred many years before their actigraphy wear date has a large pre-baseline window that is not monitored by hospital records. If such a subject is classified as `prodromal_constipation = 0`, this label cannot be verified: the subject may have presented with constipation at a GP or minor clinic during the unmonitored window without generating an HES record.

This is **non-differential misclassification** of the binary exposure: it is independent of future outcome status (subjects with sparse HES contact are not selectively at higher or lower risk). Under a binary Cox model, non-differential misclassification of a binary exposure attenuates the hazard ratio toward 1. The concern is therefore not inflation of associations but underestimation of true effect sizes and potential false negatives.

### 4.1.2 Why Actigraphy Timing Makes This Tractable

UKBB actigraphy was conducted in two waves:
- **Baseline:** ~2013–2015 (Assessment Centres)
- **First follow-up:** ~2019–2022 (online imaging visit)

Both waves predate the HES data horizon (~2023). This means the *post-baseline* follow-up period (from actigraphy to outcome) is fully covered by HES for outcome ascertainment. The HES coverage gap is entirely in the *pre-baseline* window (from HES data start to wear date), which is the window used to measure prodromal exposure.

The key metric is therefore: **how long before actigraphy was the subject last seen in HES?**

### 4.1.3 The Fix

`library/ehr_outcomes/covariates.py :: compute_hes_activity_gap` computes two columns per subject:

- `hes_last_pre_baseline_date` — latest date among all `p41280_a*` fields where the date precedes `wear_time_start`
- `hes_gap_pre_baseline_years` — `(wear_time_start − hes_last_pre_baseline_date)` in decimal years; `NaN` if no pre-baseline HES record exists

A gap of 4 years was chosen as the threshold for the sensitivity analysis based on:
1. Both actigraphy waves are separated from HES end (~2023) by <4 years — any pre-baseline gap >4 years is therefore not an artefact of data vintage
2. A 4-year unmonitored window is epidemiologically meaningful: all prodromal markers of interest (constipation, depression, OH, etc.) have clinical courses that would plausibly generate hospital contact within 4 years if present

### 4.2 Epidemiological Classification

| Property | Assessment |
|----------|-----------|
| Bias direction | Attenuation (toward null) |
| Differential? | No — independent of future outcome |
| Affected markers | HES-derived binary vars only (not cognitive questionnaire vars) |
| Proposed resolution | Sensitivity analysis restricted to HES-active subcohort |
| Primary analysis impact | None — primary analysis unmodified |

The sensitivity analysis is supplementary, not a replacement for the primary results. If HRs in the HES-active subcohort are larger than in the full cohort, this confirms that exposure misclassification was attenuating the primary estimates. If HRs are similar, HES coverage heterogeneity is not a material concern.

### 4.3 What to Report in the Paper

**Methods paragraph (suggested):**

> Binary prodromal markers derived from HES diagnostic records require adequate pre-baseline HES coverage for reliable classification of unexposed subjects. To assess whether heterogeneity in HES activity biased primary hazard ratio estimates, we computed the time elapsed between each participant's last pre-baseline HES record and their actigraphy wear date. In a pre-specified sensitivity analysis, participants with a gap exceeding 4 years (indicating a prolonged unmonitored pre-baseline window) were excluded, and all binary prodromal Cox models were re-estimated in this HES-active subcohort. Questionnaire-based cognitive markers were unaffected by HES coverage and were therefore not subjected to this restriction.

**Results reporting:**
- Report N excluded by HES gap restriction per marker in `table_10b_sensitivity_hes_active.csv`
- State direction of HR change (if HRs increase → misclassification was attenuating; if stable → robust)
- Conclude with whether the primary estimates are conservative or unbiased with respect to HES coverage

---

## 5. Secondary Analytical Improvements

### 5.1 Dynamic Kaplan-Meier Y-Axis Limits

**Rationale.** A fixed lower y-axis bound (e.g., 0.90) is appropriate when all outcomes have low event rates. For high-event-rate outcomes (DLB, dementia composites in older strata), a fixed 0.90 floor clips the survival curves and visually misrepresents the absolute risk over follow-up. Dynamic limits floored to the nearest 5% below the minimum terminal survival probability ensure the full range of each curve is visible without compressing resolution.

### 5.2 Full-Cohort vs. Complete-Case Kaplan-Meier Panels

**Rationale.** Complete-case analysis (subjects with non-missing prodromal covariates) introduces selection bias if missingness is associated with exposure or outcome. Presenting KM curves for the full cohort alongside complete-case curves allows readers to assess whether the complete-case sample is representative. If the full-cohort and complete-case RBD-stratified curves diverge substantially, this signals that complete-case restriction has selected a non-representative subcohort.

---

## 6. Output Files Reference

### 6.1 Primary Results Tables (for paper submission)

| File | Content | Paper Table |
|------|---------|-------------|
| `report/table_1_cohort.csv` | Cohort characteristics at actigraphy | Table 1 |
| `report/table_2_availability.csv` | Variable availability + HES activity % | Table 2 / Supplement |
| `report/table_3_ph_diagnostics.csv` | Schoenfeld residual PH tests | Supplement |
| `report/table_4a_rbd_only_pd.csv` | M0: RBD-only Cox HRs (PD primary outcome) | Table 3 |
| `report/table_4b_rbd_continuous.csv` | M0 continuous RBD score HR | Table 3 |
| `report/table_4c_threshold_stability.csv` | RBD threshold stability across percentiles | Supplement |
| `report/table_5_baseline_cox_pd.csv` | M1: Prodromal marker HRs (PD outcome, FDR-corrected) | Table 4 |
| `report/table_6a_additive_pd.csv` | M2: Additive RBD + prodromal HRs | Table 5 |
| `report/table_6b_interaction_pd.csv` | M3: Multiplicative interaction HRs | Table 5 |
| `report/table_7_additive_interaction.csv` | RERI / AP / SI additive interaction measures | Table 6 |
| `report/table_8_absolute_risks.csv` | Absolute risks at 5y / 10y by risk group | Table 7 |
| `report/table_9a_spline_cox.csv` | Restricted cubic spline dose-response (RBD) | Figure supplement |
| `report/table_9b_rbd_spline.csv` | RBD spline across outcomes | Figure supplement |
| `report/table_10_lag_sensitivity.csv` | Lag 2y sensitivity analysis | Supplement |
| `report/table_10b_sensitivity_hes_active.csv` | HES-active subcohort sensitivity (≤4y gap) | Supplement |
| `report/table_11a_c_index.csv` | Harrell C-index per model / outcome | Supplement |
| `report/table_12a_cif_vs_km.csv` | Competing risk: CIF vs KM comparison | Supplement |
| `report/table_12b_competing_cox.csv` | Cause-specific Cox HRs (competing risk) | Supplement |

### 6.2 Primary Results Figures (for paper submission)

| File pattern | Content | Paper Figure |
|-------------|---------|-------------|
| `km_full/KM_full_{method}.png` | Full-cohort RBD-stratified KM per outcome | Figure 1 |
| `{outcome}/KM_{method}_{prod_var}.png` | 4-panel KM (full + CC) per prodromal × outcome × method | Figure 2+ |
| `{outcome}/forest_*.png` | Forest plots of HRs across prodromal markers | Figure 3 |
| `{outcome}/spline_*.png` | RBD spline dose-response curves | Figure 4 |

### 6.3 Audit / Diagnostic Files (internal traceability)

| File | Content |
|------|---------|
| `results/baseline_cox_HRs.csv` | All-outcome M1 HRs (pre-FDR) |
| `results/ph_diagnostics.csv` | Full PH test results |
| `results/sensitivity_hes_active.csv` | Raw HES-active sensitivity results |
| `results/km_logrank_summary.csv` | Log-rank p-values, N and events per KM |
| `results/logs/outcome_flags/outcome_summary.csv` | Outcome ascertainment counts, date ranges, sources |
| `results/logs/covariates/log_covariates.csv` | Covariate merge log |

---

## 7. Bias Summary Table

| Bias | Type | Direction of Error | Where Addressed |
|------|------|-------------------|----------------|
| Reverse causation (prodromal markers) | Information bias | HR inflation | Pre-baseline restriction (Change A) |
| Outcome misclassification (DEM post-2023) | Information bias | HR attenuation, informative censoring | First-occurrence field merge (Change B) |
| Exposure misclassification (HES gaps) | Information bias | HR attenuation (non-differential) | HES-active sensitivity analysis (Change C) |
| Prevalent case contamination | Selection bias | HR inflation | `{outcome}_incident` filter (pre-existing) |
| Complete-case selection | Selection bias | Direction unknown | Full-cohort KM comparison (Change D) |
| Competing risks (death, other dementia) | Informative censoring | HR inflation if ignored | Cause-specific Cox + CIF (M4, pre-existing) |
| Multiple comparisons (6 outcomes × N markers) | Type I error inflation | False positive associations | FDR correction (pre-existing) |

---

## 8. Causal Assumptions and Limitations

1. **No edge between RBD score and prodromal markers** (DAG assumption). The causal DAG (`config.py`) posits that RBD score and prodromal markers share a common latent cause (PD pathology) but do not causally influence each other. This is an untestable assumption; the additive Cox models (M2) are interpreted under it. If there is an indirect path (e.g., RBD-disturbed sleep causing depression), the M2 HR for depression would be partially confounded.

2. **Actigraphy as a proxy for RBD pathology.** The RBD score is a continuous machine-learning prediction, not a polysomnography-confirmed diagnosis. Classification error is present. Under-prediction in subjects without definite RBD attenuates associations. The spline analysis (Table 9) characterises the dose-response shape and allows assessment of linearity assumptions.

3. **HES-active sensitivity analysis does not fully solve the coverage gap problem.** It quantifies robustness, not the true effect size. The gold-standard solution would be primary care data (GP records), which are available in UKBB but not yet integrated into this pipeline.

4. **DLB estimates remain conservative.** With no first-occurrence field and HES cutoff at 2023, DLB event counts are underestimated in later follow-up years. Report DLB results with explicit acknowledgment of right-truncation.
