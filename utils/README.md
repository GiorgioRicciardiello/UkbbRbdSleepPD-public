# Utils

Debugging, testing, and utility scripts that support development and validation of the analysis pipeline.

---

## Run Management

### `list_runs.py`

**Purpose**: List all timestamped analysis outputs from ML cross-sectional pipeline.

**What It Does**:
- Scans `results/ml_cross_sectional/` for timestamped run directories
- Displays run ID, timestamp, and completion status
- Shows file sizes for figures and tables
- Marks runs as [COMPLETE] or [PARTIAL]

**Output**:
```
ML CROSS-SECTIONAL PIPELINE RUNS
================================================================================
Run ID                         Date / Time          Output Directory
--------------------------------------------------------------------------------
20260327_130648               2026-03-27 13:06:48  ml_cross_sectional_20260327_130648 [COMPLETE]
  ├─ figure_feature_set_comparison.png (2.5 MB)
  ├─ figure_feature_set_supplemental.png (1.2 MB)
  └─ table_best_model_summary.csv
```

**Run**:
```bash
python utils/list_runs.py
```

**Use Case**: Quickly locate which analysis run to use for downstream processing.

---

## Cox Model Testing & Validation

### `test_model_f.py`

**Purpose**: Unit test for Model F (RBD × PRS interaction analysis).

**What It Tests**:
- Model F correctly computes interaction terms
- Hazard ratios and 95% CIs are calculated correctly
- Model fitting converges
- Interaction p-value is computed accurately

**Input**:
- Merged dataset with RBD scores and PD-PRS

**Output**:
- Console output confirming test results
- `test_model_f.log` with detailed test results

**Key Assertions**:
- Interaction HR is between 1.0 and 2.0 (expected range)
- p-value is not NaN
- 95% CIs do not include 1.0 (if significant)
- Coefficient estimates match analytical calculations

**Run**:
```bash
python utils/test_model_f.py
```

**When to Use**:
- After modifying interaction model code
- If main Cox pipeline shows unexpected Model F results
- Before committing changes to cox_prodromal module

---

### `test_cox_timing.py`

**Purpose**: Benchmark Cox model fitting performance and convergence.

**What It Does**:
- Measures time to fit each Cox model (0, A, F, G, 4)
- Tests convergence behavior (iterations to convergence)
- Profiles memory usage during model fitting
- Identifies bottlenecks in computation

**Output**:
- Timing table with model name, elapsed time, iterations
- Memory usage statistics
- Bottleneck analysis (which models are slowest)

**Example Output**:
```
Model          Time (s)   Iterations   Memory (MB)
─────────────────────────────────────────────────
Model 0        0.23       12           45
Model A        0.31       15           52
Model F        1.45       28           78
Model G        0.98       22           65
Model 4        2.10       35           95
```

**Run**:
```bash
python utils/test_cox_timing.py
```

**When to Use**:
- Optimize bootstrap replications count (`BOOTSTRAP_N`)
- Check if models are fitting abnormally fast or slow
- Identify memory constraints for large datasets

---

## Data Quality & Validation

### `test_prs_columns.py`

**Purpose**: Validate PD-PRS (Polygenic Risk Score) column presence and format.

**What It Tests**:
- PD-PRS column exists in dataset
- PD-PRS values are numeric and properly distributed
- No unexpected missing values
- PRS is on correct scale (z-score vs. raw)
- PRS correlation with outcome is within expected range

**Input**:
- Merged dataset (`data/pp/res_build_final_dataset/ukbb_merged_risk_groups.parquet`)

**Output**:
- Validation report with:
  - Column statistics (mean, SD, min, max)
  - Missing value counts
  - Distribution histogram
  - Correlation with PD outcome

**Checks Performed**:
✓ Column exists and is numeric
✓ Values are normally distributed (Shapiro-Wilk test)
✓ Missing rate < 5%
✓ Mean ≈ 0, SD ≈ 1 (if z-scored)
✓ Correlation with PD is positive (r > 0.1)

**Run**:
```bash
python utils/test_prs_columns.py
```

**When to Use**:
- After updating PRS file
- Before running Model A/F (which adjust for PRS)
- If Model A/F results look unexpected

---

## Data Sharing & Export

### `sharing_data_script.py`

**Purpose**: Prepare de-identified dataset for sharing with collaborators.

**What It Does**:
- Removes sensitive personal identifiers (names, contact info)
- Keeps analysis variables (RBD score, RBD risk group, outcomes)
- Applies access restrictions (e.g., n-reporting rules)
- Generates data dictionary for shared dataset
- Creates checksums for data integrity

**Input**:
- Full merged dataset with all variables

**Output**:
- `data/shared/` directory:
  - `ukbb_shared_dataset.parquet` — de-identified data (n-1 reporting)
  - `data_dictionary_shared.csv` — variable descriptions
  - `sharing_manifest.txt` — file listing and checksums

**De-identification Steps**:
1. Remove columns: `eid` (replace with random ID), address, contact_info
2. Round age to 5-year bins
3. Aggregate rare categories (n < 10)
4. Apply differential privacy (Laplace noise to counts)

**Run**:
```bash
python utils/sharing_data_script.py
```

**When to Use**:
- Preparing data for publication (supplementary data)
- Sharing with external collaborators
- Creating analysis-only datasets for method papers

---

## Development Utilities

### Script Dependencies & Loading Order

```
pipelines/               ← Primary analysis entry points
  ├─ build_ukb_dataset.py      (Stage 1)
  ├─ run_abk_collection.py     (Stage 2)
  ├─ run_merge_ukbb_rbd.py     (Stage 3)
  └─ run_cox_pipeline.py       (Stage 5)

generators/             ← Post-analysis figure/table generation
  └─ *_figure_*.py      (read from cox output)

analysis/               ← Validation analyses
  └─ sleep_phenotypes_*.py     (read from merged dataset)

utils/                  ← Debugging & testing
  ├─ list_runs.py      (read from results/)
  ├─ test_*.py         (read from data/)
  └─ sharing_data_script.py (read from data/)
```

---

## Quick Debug Checklist

If pipeline fails, run diagnostics in this order:

1. **Data availability**:
   ```bash
   python utils/test_prs_columns.py
   ```
   → Confirms required columns exist and are valid

2. **Model fitting**:
   ```bash
   python utils/test_cox_timing.py
   ```
   → Checks if models fit without errors

3. **Interaction model**:
   ```bash
   python utils/test_model_f.py
   ```
   → Validates RBD × PRS interaction computation

4. **ML runs**:
   ```bash
   python utils/list_runs.py
   ```
   → Confirms ml_cross_sectional outputs exist

---

## Development Tips

### Adding a New Test

1. Create `test_new_feature.py` in `utils/`
2. Follow naming convention: `test_{component}.py`
3. Include:
   - Clear docstring explaining what is tested
   - Input/output specification
   - List of assertions
   - Run instructions in comments

Example:
```python
"""
Test RBD score normalization.

Validates that z-score normalization is correct:
- Mean should be 0
- SD should be 1
- No NaN values introduced
"""

def test_rbd_zscore():
    df = load_merged_data()
    rbd_zscore = (df['abk_rbd_score_mean'] - mean) / std
    assert np.abs(rbd_zscore.mean()) < 0.01, "Mean not zero"
    assert np.abs(rbd_zscore.std() - 1.0) < 0.01, "SD not 1"
    assert not rbd_zscore.isna().any(), "NaN values present"
```

### Running Tests in Batch

```bash
for test in utils/test_*.py; do
    echo "Running $test..."
    python "$test" || echo "FAILED: $test"
done
```

---

## Troubleshooting

**"Module not found" error**: Ensure you're running from project root:
```bash
cd /path/to/UkbbRbdSleepPD
python utils/test_*.py
```

**"Data file not found"**: Check paths in `config/config.py` point to actual data locations.

**Test assertions fail**: 
- Check if data processing step upstream failed
- Review pipeline logs for errors
- Run diagnostic on raw data files

**Memory error in timing test**: Reduce dataset size for profiling:
```python
df = df.sample(n=10000, random_state=42)  # Use subset
```

---

## References

**Test-Driven Development**: Use tests to catch bugs early and document expected behavior.

**Performance Profiling**: Use timing tests to identify optimization opportunities.

**Data Validation**: Always validate data at system boundaries (file I/O, external APIs).
