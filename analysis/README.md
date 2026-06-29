# Analysis

Secondary analysis runners that validate and extend primary Cox findings. These scripts explore:
1. Sleep phenotype stratification by RBD scores
2. Temporal stability of RBD score predictions
3. Robustness of RBD associations across follow-up periods

All analyses operate at the **subject level** (one row per `eid`).

---

## Validation Studies

### `sleep_phenotypes_rbd_scores.py`

**Purpose**: Validate that ML-derived RBD probability scores properly stratify self-reported sleep questionnaire responses.

**Core Hypothesis**: If RBD scores cannot separate dream-enactment (Yes vs. No), the ML model is not capturing RBD-related behavior.

**Input**:
- `data/pp/res_build_final_dataset/ukbb_merged_risk_groups.parquet`
- UKBB sleep questionnaire data (dream enactment, REM behavior, violent sleep)

**Output**:
- `results/sleep_phenotypes_nostf/` directory:
  - `figure_dream_enactment_by_rbd.png` — Violin/density plots
  - `figure_rem_behavior_by_rbd.png` — REM behavior stratification
  - `figure_violent_sleep_by_rbd.png` — Violent sleep stratification
  - `table_phenotype_statistics.csv` — Median RBD by phenotype group
  - `statistics_summary.txt` — Statistical test results

**Analyses**:

| Phenotype | Test Type | Expected Result |
|-----------|-----------|-----------------|
| Dream enactment (Yes/No) | Wilcoxon rank-sum | Higher RBD in Yes group |
| REM behavior (absent/present) | Wilcoxon rank-sum | Higher RBD in present group |
| Violent sleep (no/yes) | Wilcoxon rank-sum | Higher RBD in yes group |
| Dream frequency (low/high) | Spearman correlation | Positive correlation |

**Visualizations**:
- Violin plots with individual points
- Density plots by phenotype group
- Box plots with median/IQR
- Statistical test p-values and effect sizes

**Run**:
```bash
python analysis/sleep_phenotypes_rbd_scores.py
```

**Dependencies**:
- Requires merged dataset with RBD scores and sleep questionnaire responses
- Must run after `pipelines/run_merge_ukbb_rbd.py`

**Key Results Expected**:
- Significant separation (p < 0.05) between Yes/No groups for dream enactment
- Effect size (Cohen's d or rank-biserial): medium to large
- Higher median RBD in "Yes" and "present" groups

---

### `sleep_phenotypes_temporal_analysis.py`

**Purpose**: Evaluate temporal stability of RBD score predictions across multiple actigraphy assessments.

**Core Question**: Are RBD scores reliable over time, or do they drift?

**Input**:
- Longitudinal actigraphy records with multiple measurement dates per subject
- RBD scores computed for each measurement period

**Output**:
- `results/sleep_temporal_validation/` directory:
  - `figure_rbd_stability_timeline.png` — RBD score changes over time
  - `figure_rbd_correlation_assessment.png` — Correlation between assessments
  - `figure_rbd_quantile_concordance.png` — Risk group stability across time
  - `table_temporal_reliability.csv` — ICC, test-retest correlations
  - `statistics_summary.txt` — Temporal stability metrics

**Analyses**:

| Metric | Interpretation | Target |
|--------|-----------------|--------|
| Intra-class correlation (ICC) | Absolute agreement across time | ≥ 0.70 (acceptable) |
| Spearman ρ (continuous RBD) | Rank correlation across time | ≥ 0.65 (moderate) |
| Risk group concordance | % subjects staying in same group | ≥ 80% |
| Quantile agreement | % subjects changing ≤1 quintile | ≥ 85% |

**Visualizations**:
- Scatter plots: RBD at time 1 vs. time 2
- Bland-Altman plots: agreement analysis
- Timeline plot: median RBD per assessment period
- Sankey diagram: risk group transitions

**Run**:
```bash
python analysis/sleep_phenotypes_temporal_analysis.py
```

**Dependencies**:
- Requires actigraphy records with multiple dates per subject
- Input: longitudinal RBD score history
- Must run after `pipelines/run_abk_collection.py`

**Key Results Expected**:
- ICC ≥ 0.70 indicating stable predictions
- Spearman ρ ≥ 0.60 suggesting rank preservation
- ≥ 80% of subjects remain in same risk group across assessments
- No systematic drift (bias) over time

---

## Workflow for Validation

1. **Run primary analysis**: `pipelines/run_cox_pipeline.py`
   - Generates main HR estimates for RBD → PD

2. **Validate RBD score quality**: `sleep_phenotypes_rbd_scores.py`
   - Confirm RBD scores stratify known sleep phenotypes
   - If separation is weak → investigate ML model calibration

3. **Check temporal stability**: `sleep_phenotypes_temporal_analysis.py`
   - Ensure RBD scores are stable over time
   - If ICC < 0.60 → consider measurement error in HR estimates

4. **Interpret results**:
   - Strong phenotype stratification + high ICC → robust RBD scores
   - Weak phenotype stratification → model may not capture RBD well
   - Low ICC → HR estimates may be attenuated

---

## Statistical Methods

### Phenotype Stratification Tests

**Wilcoxon Rank-Sum Test** (non-parametric alternative to t-test):
- Tests for differences in distribution between Yes/No groups
- No normality assumption required
- Reports U-statistic, p-value, and rank-biserial correlation (effect size)

**Spearman Rank Correlation** (non-parametric association):
- Tests monotonic relationship between RBD and phenotype score
- ρ ranges from -1 to 1
- p-value from permutation test (robust to outliers)

### Temporal Reliability

**Intra-class Correlation (ICC[2,1])** (two-way mixed effects):
- Estimates absolute agreement between two assessments
- Interpretation: ICC < 0.50 (poor), 0.50–0.75 (moderate), 0.75–0.90 (good), > 0.90 (excellent)
- Formula: `ICC = (BMS - EMS) / (BMS + (k-1) × EMS)`
  - BMS = between-subject mean square
  - EMS = error mean square
  - k = number of assessments

**Bland-Altman Limits of Agreement**:
- Plots difference vs. mean to assess systematic bias
- 95% LoA = mean difference ± 1.96 × SD(difference)
- Interpretation: narrow LoA indicates good agreement

**Quantile Concordance**:
- % of subjects remaining in same risk quintile across assessments
- More robust to outliers than correlation-based metrics

---

## Output Interpretation

### Green Flags (Results Support RBD Score Validity)
✓ Dream enactment (Yes > No) with p < 0.001
✓ ICC ≥ 0.70 for temporal stability
✓ ≥ 85% of subjects in same risk quantile across time
✓ No systematic drift over follow-up

### Yellow Flags (Results Need Investigation)
⚠ Weak phenotype separation (0.001 < p < 0.05)
⚠ Moderate ICC (0.60–0.70)
⚠ 70–85% risk quantile concordance
⚠ Small systematic drift over time

### Red Flags (Results Question RBD Score Quality)
✗ No phenotype separation (p > 0.05)
✗ ICC < 0.60
✗ < 70% risk quantile concordance
✗ Large systematic drift; model may be recalibrating

---

## Troubleshooting

**"Data file not found"**: Confirm that:
- Merged dataset exists at `data/pp/res_build_final_dataset/`
- Actigraphy batch files exist in `data/actig_extracted_features/`

**"Empty figures / no data plotted"**: Check:
- Sleep phenotype variables exist in questionnaire data
- Non-missing rates for key columns (> 50% coverage)
- Stratification variable is not constant

**Statistical test fails to converge**: 
- Reduce sample to debugging subset (first 1000 subjects)
- Check for extreme outliers in RBD score
- Verify outcome variable has both 0 and 1 values

**Low ICC despite good phenotype separation**:
- May indicate measurement error independent of RBD quality
- Check if actigraphy recording duration differs between assessments
- Confirm dates for each assessment are correct

---

## Publications & References

**RBD Diagnostic Criteria**:
- Heinzel S, et al. Update of the diagnostic criteria for RBD in Parkinson disease. Parkinsonism Relat Disord. 2019;64:34-42.

**Phenotype Validation in Prodromal PD**:
- Schrag A, et al. Identifying prodromal Parkinson disease. Mov Disord. 2015;30(13):1657-66.

**Statistical Methods**:
- ICC: Koo TK, Li MY. A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research. J Chiropr Med. 2016;15(2):155-163.
- Bland-Altman: Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986;1(8476):307-310.
