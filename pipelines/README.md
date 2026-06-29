# Pipelines

Entry points for major analysis workflows. Each pipeline orchestrates data processing, analysis, and output generation for a specific research question.

---

## Stage 1: Dataset Construction

### `build_ukb_dataset.py`

**Purpose**: Extract, process, and construct the UK Biobank cohort dataset from raw CSV files.

**Input**:
- Raw UK Biobank data dictionary CSV
- Assessment center, health outcomes, biological samples, and population characteristics CSVs
- Stored in `config.tables_dir` (typically `data/raw/`)

**Output**:
- `data/pp/res_build_final_dataset/ukbb_final_cohort.parquet` — clean cohort dataset
- `data/pp/res_build_final_dataset/ukbb_final_cohort.csv` — CSV version for QC

**Transformations**:
- Map UKBB field codes to human-readable column names
- Apply inclusion/exclusion criteria
- Define outcome flags (PD, AD, dementia, DLB) based on ICD-10 codes
- Flag incident vs. prevalent cases
- Compute covariates and demographic variables
- Apply quality control filters

**Run**:
```bash
python pipelines/build_ukb_dataset.py
# or from main.py: set RUN_BUILD_EHR=True
```

**Configuration**:
- Edit `config/config.py` to adjust data paths and outcome definitions
- Edit `library/ehr_outcomes/` to modify ICD-10 mapping logic

---

## Stage 2: Actigraphy Collection

### `run_abk_collection.py`

**Purpose**: Collect and merge actigraphy-derived RBD probability scores from multiple batches.

**Input**:
- Actigraphy time-series files (CWA format, processed by external ML model)
- Stored in `data/actig_extracted_features/` with subdirectories:
  - `ActigStfRecords/` — RBD scores from sleep-time features
  - `DataOnlySleepRBD/` — sleep-only RBD scores
  - `DataRemaining/` — additional actigraphy features

**Output**:
- `data/actig_extracted_features/F_Sleep_abk_merged_rbd_pred.parquet` — subject-level RBD scores
- Columns: `eid`, `abk_rbd_score_mean`, `abk_rbd_score_sd`, `n_nights`

**Processing**:
- Aggregate nightly RBD predictions to subject level (mean, SD)
- Quality control: minimum night count thresholds
- Handle missing/invalid nights
- Merge batches from different collection rounds

**Run**:
```bash
python pipelines/run_abk_collection.py
# or from main.py: set RUN_ABK_COLLECTION=True
```

---

## Stage 3: Merge & Risk Stratification

### `run_merge_ukbb_rbd.py`

**Purpose**: Merge EHR dataset with RBD scores and gait features; compute risk groups.

**Input**:
- `data/pp/res_build_final_dataset/ukbb_final_cohort.parquet` (from Stage 1)
- `data/actig_extracted_features/F_Sleep_abk_merged_rbd_pred.parquet` (from Stage 2)
- Gait features (optional)

**Output**:
- `data/pp/res_build_final_dataset/ukbb_merged_risk_groups.parquet` — production dataset
- Columns added:
  - `rbd_prob` — normalized RBD probability (z-score)
  - `{outcome}_risk_group_mean_2g` — 2-group risk (Low/High)
  - `{outcome}_risk_group_mean_3g` — 3-group risk (Low/Mid/High)
  - Survival columns: `{outcome}_surv_time`, `{outcome}_surv_event`

**Processing**:
- Merge EHR + RBD on `eid`
- Drop prevalent cases and quality-control exclusions
- Normalize RBD scores (z-score standardization)
- Compute percentile-based risk thresholds
- Stratify into risk groups per outcome
- Compute follow-up time and event flags

**Run**:
```bash
python pipelines/run_merge_ukbb_rbd.py
# or from main.py: set RUN_MERGE=True, OVERWRITE_MERGE=True
```

**Configuration**:
- Edit `library/risk/risk_groups.py` to adjust percentile thresholds
- Edit `config/config.py` to modify outcome definitions

---

## Stage 4: Baseline Characteristics

### `run_table_one.py`

**Purpose**: Generate Table 1 (baseline characteristics) stratified by RBD risk group.

**Input**:
- `data/pp/res_build_final_dataset/ukbb_merged_risk_groups.parquet` (from Stage 3)

**Output**:
- `results/table_one/` directory:
  - `table_one_baseline.csv` — demographic and clinical variables by RBD group
  - `table_one_prodromal.csv` — prodromal marker prevalence
  - `table_one_summary.txt` — human-readable summary

**Content**:
- Demographics: age, sex, ethnicity, BMI, education
- Baseline comorbidities: hypertension, diabetes, cardiovascular disease
- Sleep questionnaire responses: dream enactment, REM density
- Prodromal markers: constipation, depression, orthostatic hypotension, erectile dysfunction

**Run**:
```bash
python pipelines/run_table_one.py
# or from main.py: set RUN_TABLE_ONE=True
```

---

## Stage 5: Cox Proportional Hazards Analysis

### `run_cox_pipeline.py`

**Purpose**: Fit Cox proportional hazards models to evaluate RBD → PD association.

**Input**:
- `data/pp/res_build_final_dataset/ukbb_merged_risk_groups.parquet`
- Cox model configuration from `library/cox_prodromal/cox_config.py`

**Output**:
- `results/cox_prodromal_abk_{TIMESTAMP}/` directory:
  - `model_results.csv` — HR, 95% CI, p-values for all models
  - `forest_plots/` — PDF/PNG forest plots for each model
  - `diagnostics/` — proportional hazards tests, C-index values
  - `tables/` — formatted results tables

**Models**:
| Model | Description | Key Result |
|-------|-------------|-----------|
| 0 | RBD only (unadjusted) | HR = 4.69 (3.21–6.87) |
| A | RBD adjusted for PD-PRS | HR = 4.42 (2.86–6.84) |
| F | RBD × PRS interaction | Interaction HR = 1.20 (p=2.7e-10) |
| G | RBD × GBA carrier (exploratory) | Requires replication |
| 4 | Competing risks (death, other diagnoses) | RBD effect robust |

**Run**:
```bash
python pipelines/run_cox_pipeline.py
# or from main.py: set RUN_COX_PIPELINE=True
```

**Configuration**:
- Edit `library/cox_prodromal/cox_config.py` to:
  - Enable/disable models
  - Adjust bootstrap replications (dev: 10, prod: 1000)
  - Set minimum event counts per model
  - Define adjustment covariates

---

## Downstream Analysis Pipelines

### `run_ml_cross_sectional.py`

**Purpose**: Machine learning cross-sectional analysis (feature importance, model comparison).

**Input**: Merged UKBB + RBD dataset

**Output**: Feature importance plots, model performance tables

---

### `run_rbd_prs_association.py`

**Purpose**: Evaluate RBD-PRS joint associations with PD.

**Input**:
- Merged dataset with RBD scores and PD-PRS
- Both continuous and categorical (risk group) RBD scores

**Output**:
- Interaction plots (RBD × PRS)
- Stratified analysis tables

---

### `run_rbd_scores_matching.py`

**Purpose**: Validation of RBD scores via propensity-score matching.

**Input**: Merged dataset

**Output**: Matched cohort analysis comparing high vs. low RBD

---

## Dependency Graph

```
[1] build_ukb_dataset ──┐
                        ├──▶ [3] run_merge_ukbb_rbd ──┬──▶ [4] run_table_one
[2] run_abk_collection ─┘                              ├──▶ [5] run_cox_pipeline
                                                       ├──▶ ML analyses
                                                       └──▶ Secondary analyses
```

---

## Execution Order

1. **Must run in order**: Stages 1 → 2 → 3
2. **Parallel OK**: Stages 1 and 2 can run simultaneously
3. **Independent**: Stages 4+ depend only on Stage 3

---

## Main Entry Point

```bash
python main.py
```

Edit toggles in `main.py` to enable/disable stages and control re-execution:
```python
RUN_BUILD_EHR: bool = True          # Stage 1
RUN_ABK_COLLECTION: bool = False    # Stage 2
RUN_MERGE: bool = True              # Stage 3
RUN_TABLE_ONE: bool = True          # Stage 4
RUN_COX_PIPELINE: bool = True       # Stage 5
RUN_SLEEP_PHENOTYPES: bool = True   # Stage 6
RUN_SLEEP_TEMPORAL: bool = True     # Stage 7

OVERWRITE_EHR: bool = False         # Re-extract from raw CSVs
OVERWRITE_MERGE: bool = True        # Regenerate even if output exists
```

---

## Troubleshooting

**Stage 1 fails**: Check that `tables_dir` in `config/config.py` points to valid UKBB data CSVs.

**Stage 2 fails**: Verify actigraphy batch files exist in `data/actig_extracted_features/`.

**Stage 3 fails**: Confirm Stages 1 and 2 completed successfully and output files exist.

**Cox models fail to fit**: Check minimum event count in `cox_config.py`; may need to reduce BOOTSTRAP_N for debugging.
