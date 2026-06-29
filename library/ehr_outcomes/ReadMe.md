# EHR Outcomes Module — Epidemiological Documentation

**Module:** `library/ehr_outcomes/`
**Pipeline entry point:** `build_ukb_dataset.py` → `library/ehr_reader/ukb_data_processor.py`
**Last updated:** 2026-03-22

---

## Overview

This module implements the complete EHR processing pipeline for the UKBB actigraphy cohort. It transforms raw UKBB fields into analytically clean outcome flags, exclusion criteria, covariate definitions, and prodromal marker classifications. Every step is applied sequentially by `UkbDataProcessor.apply_processing_pipeline()` in the order described below.

The unit of analysis is one participant–actigraphy wave record. Participants who completed both the baseline (~2013–2015) and first follow-up actigraphy (~2019–2022) contribute one record per wave.

---

## Pipeline Execution Order

```
build_ukb_dataset.py
  └── UkbDataProcessor.apply_processing_pipeline()
        ├── Step 1  — outcome_flags.py      add_outcome_flags()
        ├── Step 2  — exclusion.py          add_neuro_exclusion()
        ├── Step 2a — exclusion.py          add_shift_worker()
        ├── Step 2b — exclusion.py          add_bad_actig_recording()
        ├── Step 2c — identify_splits.py    add_data_split_flags()
        ├── Step 3  — controls.py           add_controls()
        ├── Step 4  — covariates.py         add_covariates()
        ├── Step 4a — age_groups.py         create_age_groups()
        ├── Step 5  — medications.py        add_medication_flags()
        ├── Step 6  — covariates.py         merge_prodromal_markers()
        └── Step 7  — consort_report.py     generate_ehr_consort_flow()
```

---

## Step 1 — Outcome Ascertainment (`outcome_flags.py`)

### What

Generates disease flags, first-diagnosis dates, and the prevalent/incident split for all six analytical outcomes.

### Source fields

| Source | UKBB Field | Coverage |
|--------|-----------|---------|
| HES ICD-10 diagnoses | p41270 (codes) / p41280_a* (dates) | ~2023-03 |
| First-occurrence PD | p131022 (date) / p131023 (source) | 2025 |
| First-occurrence AD | p131036 (date) / p131037 (source) | 2025 |
| First-occurrence AD self-report | p42020 | 2025 |
| First-occurrence dementia all-cause | p42018 | 2025 |
| First-occurrence vascular dementia | p42022 | 2025 |
| First-occurrence frontotemporal dementia | p42024 | 2025 |

### ICD-10 codes used

| Disease | Codes |
|---------|-------|
| PD | G20 |
| AD | G30, G301, G308, G309 |
| Non-AD dementia (DEM) | F00, F001, F002, F009, F01, F010, F02, F023, F03 |
| DLB | G318 |

### Ascertainment strategy

**PD and AD** are ascertained from both HES and first-occurrence fields. The earlier of the two dates is used as the diagnosis date (`dx_date = min(HES date, first-occurrence date)`). This extends effective follow-up to 2025 for these conditions and prevents right-truncation at the HES horizon.

**Dementia (DEM)** is ascertained from HES ICD-10 codes merged with three first-occurrence fields (p42018, p42022, p42024). The same minimum-date strategy is applied. This was implemented because HES alone would have truncated all dementia events after March 2023, creating informative censoring in subjects with later-occurring disease.

**DLB** has no dedicated first-occurrence field in UKBB. Ascertainment is therefore restricted to HES (ICD-10 G318). This is a known data limitation: DLB event counts are underestimated in follow-up years beyond 2023, and HR estimates for the `outcome_3a_dlb_only` outcome should be interpreted with this in mind.

### Prevalent vs. incident classification

A case is **prevalent** if `dx_date < wear_time_start` (diagnosis precedes the actigraphy baseline). A case is **incident** if `dx_date >= wear_time_start`. This classification is applied per disease flag and stored as `{outcome}_prevalent` and `{outcome}_incident` boolean columns. Only incident cases enter the survival analysis.

### Censor date

The administrative censor date is `2025-11-01`. This is authoritative and is not overridden by the HES data ceiling. Subjects without an event are censored at `min(death date, censor date, last contact)`.

### Outputs (six analytical outcomes)

| Column | Definition |
|--------|-----------|
| `outcome_1a_pd_only` | PD only (no AD, no DEM) |
| `outcome_1b_pd_ad` | PD and AD comorbid |
| `outcome_2a_otherdementia` | Non-AD dementia (no PD, no AD) |
| `outcome_2b_pd_otherdementia` | PD with non-AD dementia |
| `outcome_3a_dlb_only` | DLB only (no PD) |
| `outcome_4a_ad_only` | AD only (no PD, no DEM) |

Survival columns: `{outcome}_surv_time` (days from wear start to event or censor), `{outcome}_surv_event` (0/1).

Audit log: `results/logs/outcome_flags/outcome_summary.csv`

---

## Step 2 — Neurological Exclusions (`exclusion.py`)

### What and why

Subjects with a neurological diagnosis from the exclusion list *before* their actigraphy wear date (`wear_time_start`) are flagged `neuro_exclude = True` and removed from all analyses. These participants have pre-existing neuropathology that would confound the RBD–outcome association by introducing prevalent disease at baseline or competing aetiologies for the outcomes of interest.

The exclusion is applied only to **pre-baseline diagnoses** (dx_date < wear_time_start). Post-baseline diagnoses are retained as potential incident outcomes.

### Why prevalent-only exclusion matters

Removing subjects with post-baseline neurological diagnoses would introduce survivor bias: only participants who remained disease-free would be in the risk set, and this would overestimate time-to-event in the surviving cohort. The temporally restricted exclusion avoids this by only removing those who were already affected at the time of exposure measurement.

### ICD-10 exclusion codes

Atypical parkinsonism / parkinson-plus syndromes: G21, G22, G23, G24, G25x
Neurodegenerative diseases: G10–G13, G31, G32
Demyelinating diseases: G35–G37
Epilepsy / seizures: G40, G41, R560, R568
Encephalitis / encephalopathy: G04, G05, G934, G938, G939
Narcolepsy: G474, G4740–G4742
Other neurological: additional codes per `config.neuro_exclusion_codes`

The exclusion deliberately does not include G20 (PD), G30 (AD), G318 (DLB), or F0x/F3x (dementia/depression), because these are either analytical outcomes or prodromal markers — their pre-baseline presence is handled by the prevalent/incident classification in Step 1.

Audit log: `results/logs/exclusion/exclusion_cohort_flow.csv`, `exclusion_icd_drivers.csv`

---

## Step 2a — Shift Worker Exclusion (`exclusion.py`)

### What and why

Shift work disrupts circadian rhythms and sleep architecture, directly confounding the actigraphy signal and potentially inflating RBD score estimates in participants without true REM sleep behaviour disorder. Three shift-work exposure variables are derived per assessment wave (instances i0–i3):

- `shift_any_i{n}_p826`: any shift exposure (categories 2–4 vs 1)
- `shift_high_i{n}_p826`: high-intensity (usually/always; categories 3–4)
- `night_shift_ever`, `night_shift_max`, `night_shift_high`: collapsed from field p22650 (night shifts worked per job, coding 489: 0=Never, 1=Sometimes, 2=Usually, 3=Always)

Negative codes (−1 = do not know, −3 = prefer not to answer) are recoded to NaN. These variables are retained as potential confounders and sensitivity analysis covariates; they are not used as hard exclusion criteria by default.

---

## Step 2b — Actigraphy Quality Exclusion (`exclusion.py`)

### What and why

Poor-quality accelerometer recordings cannot reliably distinguish REM from non-REM sleep and produce unreliable RBD probability scores. A composite `acc_bad_quality` flag is set if **any** of the following quality indicators are triggered:

| UKBB Field | Condition | Rationale |
|-----------|-----------|-----------|
| p90015 | = 0 | Insufficient wear time |
| p90016 | = 0 | Failed calibration |
| p90017 | = 0 | Not calibrated on own data |
| p90018 | = 1 | Daylight savings crossover |
| p90002 | ∈ {1, 2} | Unreliable device size |
| p90180 | > 0 | Non-zero recording problems |

Subjects flagged `acc_bad_quality = True` are excluded from the ML model scoring step (handled upstream in `run_merge_ukbb_rbd.py`) via the `train_sleep` and quality filters in `get_clean_risk_data()`.

---

## Step 2c — Data Split Flags (`identify_splits.py`)

### What and why

Three mutually exclusive membership flags prevent data leakage between ML model training, internal validation, and the epidemiological analysis:

| Flag | Source | Purpose |
|------|--------|---------|
| `train_sleep` | External CSV of training EIDs | Subjects used to train the RBD scoring ML model. Excluded from the risk analysis cohort in `get_clean_risk_data()` to prevent target leakage. |
| `val_pd` | External CSV of PD validation EIDs | Subjects reserved for PD model validation. |
| `val_dlb` | External CSV of DLB validation EIDs | Subjects reserved for DLB model validation. |

Subjects with `train_sleep = True` are excluded from all Cox analyses. This is a hard requirement: the RBD probability score was optimised on these subjects and including them would inflate predictive associations.

---

## Step 3 — Control Definition (`controls.py`)

### What and why

A **control** is a participant with no neurodegenerative disease diagnosis (ever, across all follow-up) and no pre-baseline neurological exclusion. The control definition is intentionally conservative: it uses the full diagnostic record rather than the incident/prevalent split used for cases.

### Logic

```
control = (
    NOT pd_flag AND
    NOT ad_flag AND
    NOT dem_flag AND
    NOT dlb_flag AND
    NOT neuro_exclude
)
```

A subject diagnosed with PD five years after actigraphy is an **incident case**, not a control. A subject who never develops any target disease across the entire follow-up period is a **control**. This creates a clean comparison group uncontaminated by undiagnosed or unrecorded disease at the time of actigraphy.

The `control` flag is used in Step 1 of the Cox survival dataset builder (`build_survival_dataset_for_outcome`) to define the risk set: each model includes incident cases plus controls.

Audit log: `results/logs/controls/control_summary.csv`

---

## Step 4 — Covariates (`covariates.py`)

### What

`add_covariates()` derives analytical covariates from the merged UKBB dataset. It operates in three sub-stages:

### 4a — Cognitive function metrics

Derived from online task responses. These are continuous variables used as prodromal markers of early neurodegeneration in the Cox models.

| Variable | Source Field | Description |
|----------|-------------|-------------|
| `cov_fluid_intelligence_20016_i0` | p20016 | Fluid intelligence score (verbal-numerical reasoning) |
| `cov_react_time_mean_20023_i0` | p20023 (derived via p404, p10147) | Mean reaction time (ms), excluding pilot rounds |
| `cov_fi_questions_attempted_20128_i0` | p20128 | Number of fluid intelligence questions attempted |
| `cov_numeric_memory_max_20240_i0` | p20240 | Maximum digit span (numeric memory) |
| `trail_making_errors_trail1_i2` | p6348 (instances 2–3) | Trail Making Test errors |
| `cov_pairs_status_20244_i0` | p20244 | Pairs Matching Test status |

### 4b — HES-based prodromal marker flags (raw)

`add_alpha_syn_covariates()` scans all HES ICD-10 code columns (p41270, with dates in p41280_a*) to derive per-subject binary flags indicating whether each prodromal marker appears in the HES record. At this stage, **no temporal restriction is applied** — all HES records are included. The pre-baseline restriction is applied in Step 6.

| Flag | ICD-10 codes |
|------|-------------|
| `constipation_hes` | K590 |
| `depression_hes` | F32, F33, F34, F38, F39 |
| `anxiety_hes` | F40, F41 |
| `Orthostatic_hes` | I951 |
| `erectile_dysfunction_hes` | N5201, N521, F5221, N529 |
| `dream_enactment_hes` | G4752 |
| `anosmia_hes` | R430 |
| `hyposmia_hes` | G520 |

### 4c — Demographic and lifestyle covariates

UKBB fields are renamed to analytical names following the pattern `cov_{label}_{field}`:

| Analytical column | Field | Description |
|------------------|-------|-------------|
| `cov_sex_31` | p31 | Biological sex |
| `cov_age_recruitment_21022` | p21022 | Age at recruitment |
| `cov_ethnicity_21000` | p21000 | Self-reported ethnicity |
| `cov_sleep_duration_1160` | p1160 | Self-reported sleep duration |
| `cov_chronotype_1180` | p1180 | Chronotype (morning/evening preference) |
| `cov_smoking_20116_i{0–3}` | p20116 | Smoking status per wave |
| `cov_alcohol_20117_i{0–3}` | p20117 | Alcohol consumption frequency per wave |
| `bmi_imp_23104_i0` | p23104 | Impedance-measured BMI |

Negative UKBB response codes (−1 = do not know, −3 = prefer not to answer) are recoded to NaN throughout.

---

## Step 4a — Age Groups (`age_groups.py`)

Three sets of age-group indicators are constructed from `cov_age_recruitment_21022`:

| Column | Groups | Purpose |
|--------|--------|---------|
| `age_group_3` / `age_group_3_cat` | <50, 50–60, >60 | Stratified KM and descriptive tables |
| `age_group_2` / `age_group_2_cat` | ≤60, >60 | Binary sensitivity stratification |
| `age_group_none` | "All ages" | Full-cohort reference label |

These are used for stratified Kaplan–Meier panels and as interaction terms in stratified Cox models.

---

## Step 5 — Medication Flags (`medications.py`)

### What and why

Self-reported medication use (UKBB field p20003, data-coding 4) provides a second ascertainment pathway for prodromal symptoms that may not generate hospital-level ICD-10 codes. Subjects who manage constipation with laxatives, depression with antidepressants, or orthostatic hypotension with vasopressors may never appear in HES with the corresponding diagnosis code, particularly if managed exclusively in primary care. Including medication evidence increases ascertainment sensitivity for all HES-derived prodromal markers.

### Drug families

The lookup is constructed by substring-matching drug names against the UKBB data dictionary (`app45551_20251118060954.dataset.data_dictionary.csv`) and codings file. Matching is performed across all instance arrays (p20003_i*_a*).

| Flag created | Drug keywords matched |
|-------------|----------------------|
| `med_laxatives` | lactulose, macrogol/movicol, senna, bisacodyl, psyllium, ispaghula, docusate |
| `med_depression` | SSRIs (fluoxetine, sertraline, paroxetine, citalopram, escitalopram, fluvoxamine), SNRIs (venlafaxine, duloxetine, desvenlafaxine), tricyclics (amitriptyline, imipramine, doxepin, nortriptyline, clomipramine, trimipramine, desipramine), MAOIs, atypical (bupropion, mirtazapine, trazodone) |
| `med_anxiety` | Benzodiazepines (diazepam, lorazepam, alprazolam, clonazepam, oxazepam, temazepam, chlordiazepoxide, bromazepam, midazolam, nitrazepam), other (buspirone, hydroxyzine) |
| `med_orthostatic_hypotension` | midodrine, fludrocortisone, droxidopa |
| `med_pde5_inhibitors` | sildenafil, tadalafil, vardenafil, avanafil |

A special exclusion rule prevents macrogol from matching topical skin preparations (ointment, cream, bath products, lauromacrogol, cetomacrogol).

Per-subject earliest medication report dates are also derived for each family.

Audit log: `results/logs/medications/`

---

## Step 6 — Prodromal Marker Merging and Pre-Baseline Restriction (`covariates.py`)

### What

`merge_prodromal_markers()` creates the eight final binary prodromal marker columns used in the Cox models by combining the HES flags (Step 4b) with the medication flags (Step 5). A subject is classified as exposed (`prodromal_{marker} = 1`) if evidence exists from **either** source before the actigraphy baseline.

| Final column | HES source | Medication source |
|-------------|-----------|------------------|
| `prodromal_constipation` | `constipation_hes` | `med_laxatives` |
| `prodromal_depression` | `depression_hes` | `med_depression` |
| `prodromal_anxiety` | `anxiety_hes` | `med_anxiety` |
| `prodromal_orthostatic` | `Orthostatic_hes` | `med_orthostatic_hypotension` |
| `prodromal_erectile_dysfunction` | `erectile_dysfunction_hes` | `med_pde5_inhibitors` |
| `prodromal_dream_enactment` | `dream_enactment_hes` | *(HES only)* |
| `prodromal_anosmia` | `anosmia_hes` | *(HES only)* |
| `prodromal_hyposmia` | `hyposmia_hes` | *(HES only)* |

### Why this merging strategy

Using HES codes alone underestimates prodromal prevalence: many patients receive a laxative prescription without a formal constipation diagnosis in secondary care. Using medications alone overestimates it: laxatives are prescribed for numerous indications. The conjunction of any source — clinical diagnosis **or** disease-specific pharmacotherapy — provides a pragmatic compromise that maximises sensitivity while maintaining construct validity.

### Pre-baseline restriction — reverse causation prevention

**This is the most epidemiologically critical step in the pipeline.** All HES and medication evidence is restricted to records dated strictly before `wear_time_start`. Records with missing dates (`NaT`) are treated as post-baseline and excluded (conservative).

**Why this restriction is necessary.** The study question is: does a pre-existing prodromal burden (at the time of actigraphy) modify the association between RBD score and incident neurodegeneration? If post-baseline HES records are included — for example, a constipation diagnosis made 18 months after actigraphy — the marker may reflect early Lewy body spread that was already progressing at the time of wear, rather than an antecedent exposure. This is **reverse causation** (protopathic bias): the marker is a consequence of early disease, not an independent predictor. Including such records inflates hazard ratios by creating a correlation between marker status and imminent disease that is spurious.

The pre-baseline restriction enforces a temporally correct prospective design: exposure is measured before the risk window opens. This is the same principle that governs prevalent case exclusion on the outcome side.

**Handling of NaT dates.** Records where the date is missing cannot be verified as pre-baseline. Treating them as post-baseline (unexposed) is conservative: it may lead to slight underestimation of prodromal prevalence, but this error is non-differential with respect to future outcome (the RBD score is already assigned), and therefore attenuates rather than inflates hazard ratios.

### HES activity gap computation

Computed within this step via `compute_hes_activity_gap()`. For each subject, the latest date among all p41280_a* columns that falls before `wear_time_start` is identified. The gap in years between this date and `wear_time_start` is stored as `hes_gap_pre_baseline_years`.

**Epidemiological purpose.** A subject whose last pre-baseline HES contact occurred many years before actigraphy has a long unmonitored window in which prodromal symptoms could have developed without generating an HES record. If such a subject is classified as `prodromal_{marker} = 0`, that label may be incorrect (false negative). This is non-differential misclassification: it is independent of future outcome status, and its effect is to attenuate hazard ratios toward the null. The `hes_gap_pre_baseline_years` column is used in the sensitivity analysis (Step 2c of the Cox pipeline) to restrict to subjects with a gap ≤ 4 years, verifying that primary estimates are not materially biased by HES coverage heterogeneity.

Audit log: `results/logs/prodromal_markers/`

---

## Summary of Bias Controls

| Bias | Where addressed | Mechanism |
|------|----------------|-----------|
| Reverse causation (prodromal markers) | Step 6 | Pre-baseline restriction of all HES and medication records |
| Prevalent case contamination | Step 1 | `{outcome}_incident` flag; survival dataset excludes prevalent cases |
| Neurological confounding at baseline | Step 2 | `neuro_exclude` flag; subjects with pre-baseline competing neuropathology removed |
| Actigraphy signal contamination | Step 2b | `acc_bad_quality` flag; unreliable recordings excluded from ML scoring |
| ML training data leakage | Step 2c | `train_sleep` flag; training subjects excluded from the analysis cohort |
| Outcome misclassification (DEM/AD right-truncation) | Step 1 | First-occurrence fields extend ascertainment to 2025 |
| Prodromal exposure misclassification (HES gaps) | Step 6 | `hes_gap_pre_baseline_years` quantifies monitoring density; sensitivity analysis at ≤4 years |

---

## Output Variables Used in Downstream Analysis

The final parquet written by `build_ukb_dataset.py` feeds into `run_merge_ukbb_rbd.py`, which merges with RBD model scores and computes risk groups. The columns written by this module that are actively used in the Cox analysis are:

**Survival structure:** `{outcome}_surv_time`, `{outcome}_surv_event`, `{outcome}_incident`, `{outcome}_prevalent`

**Risk set definition:** `control`, `neuro_exclude`, `train_sleep`

**Prodromal markers (binary):** `prodromal_constipation`, `prodromal_depression`, `prodromal_anxiety`, `prodromal_orthostatic`, `prodromal_erectile_dysfunction`, `prodromal_dream_enactment`, `prodromal_anosmia`, `prodromal_hyposmia`

**Cognitive markers (continuous):** `cov_fluid_intelligence_20016_i0`, `cov_react_time_mean_20023_i0`, `cov_fi_questions_attempted_20128_i0`, `cov_numeric_memory_max_20240_i0`, `trail_making_errors_trail1_i2`, `cov_pairs_status_20244_i0`

**Base covariates:** `cov_age_recruitment_21022`, `cov_sex_31`, `bmi_imp_23104_i0`, `cov_smoking_20116_i{0–3}`, `cov_alcohol_20117_i{0–3}`

**HES sensitivity:** `hes_gap_pre_baseline_years`, `hes_last_pre_baseline_date`

**Age stratification:** `age_group_3`, `age_group_3_cat`, `age_group_2`, `age_group_2_cat`

**Quality / split:** `wear_time_start`, `wear_time_end`, `acc_bad_quality`, `val_pd`, `val_dlb`

---

## Step 7 — CONSORT / STROBE Participant Flow Table (`consort_report.py`)

### What

`generate_ehr_consort_flow()` produces a machine-readable attrition table following STROBE Checklist Item 13 (cohort studies). It reconstructs the participant flow from flag columns that are already present in the final processed DataFrame — no additional data loading is required.

### Save path

```
data/pp/data_sheet/logs/consort/consort_flow_table.csv
data/pp/data_sheet/logs/consort/consort_flow_table.xlsx
```

These files are written automatically by `UkbDataProcessor.apply_processing_pipeline()` as the final step. If generation fails (e.g., a flag column is missing), a `UserWarning` is raised and the pipeline continues without interruption.

### Attrition steps reported

| Step | Description | Flag column(s) used |
|------|-------------|---------------------|
| Step 0 | Total UKBB participants (optional) | `n_all_ukbb` argument |
| Step 1 | Actigraphy cohort (have wear_time_start) | All rows in input `df` |
| Step 2 | After neurological exclusion | `neuro_exclude == False` |
| Step 3 | Poor actigraphy quality (informational) | `acc_bad_quality` |
| Step 4+ | Per-outcome analytical cohort (prevalent exclusion) | `{outcome}__surv_days.notna()` |

**Note on Step 3:** `acc_bad_quality` is a **flag only** — `add_bad_actig_recording()` does not drop subjects. Step 3 is reported as informational: subjects with poor-quality actigraphy are retained at subject level but excluded from ML scoring at the night level. The CONSORT table includes a note to this effect.

**Note on Step 4+:** Prevalent cases are identified via `{outcome}__surv_days.isna()` — subjects whose survival time is NaN were diagnosed before the actigraphy baseline and are excluded from the incident analysis. This mirrors the prevalent/incident split applied in Step 1 of the Cox survival dataset builder.

### Output columns

| Column | Description |
|--------|-------------|
| `step` | Sequential step identifier |
| `section` | Section label (e.g., "Exclusions", "Outcome cohorts") |
| `description` | Human-readable step description |
| `exclusion_criterion` | Flag or rule applied |
| `n_before` | N before this exclusion |
| `n_excluded` | N removed at this step |
| `pct_excluded` | Percentage excluded (2 d.p.) |
| `n_after` | N remaining after exclusion |
| `outcome` | Outcome name (populated for Step 4+, blank otherwise) |
| `n_incident_cases` | Incident event count (Step 4+ only) |
| `n_prevalent_excluded` | Prevalent cases excluded (Step 4+ only) |
| `median_follow_up_years` | Median follow-up in years (Step 4+ only) |
| `notes` | Epidemiological notes (e.g., informational flags) |

### Function signature

```python
generate_ehr_consort_flow(
    df: pd.DataFrame,
    outcomes: List[str],
    outcome_labels: Optional[Dict[str, str]] = None,
    n_all_ukbb: Optional[int] = None,
    id_col: str = "eid",
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame
```

Called from `UkbDataProcessor.apply_processing_pipeline()` with `outcomes` sourced from `config.outcomes` and `save_dir = self.out_dir_logs / "consort"`.
