# ML Cross-Sectional Pipeline

Machine learning pipeline for cross-sectional analysis of RBD and related outcomes across multiple feature sets.

## Overview

This pipeline trains multiple ML models across different feature set configurations to predict binary outcomes (PD, AD, dementia, etc.). It supports:
- **Multiple models**: Logistic Regression, Elastic Net, XGBoost, Random Forest, SVM
- **Multiple feature sets**: RBD alone, RBD + Prodromal, RBD + PRS, RBD + PRS + Prodromal
- **Parallel execution**: Process multiple feature sets concurrently using ProcessPoolExecutor
- **Comprehensive metrics**: ROC, PR, confusion matrices, calibration, feature importance (SHAP/permutation)
- **Publication-ready outputs**: Cross-feature-set comparison figures, supplemental panels, summary tables

---

## Core Modules

### Pipeline Execution

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Main pipeline orchestration; runs models for a feature set with nested CV |
| `parallel_pipeline.py` | Parallel execution wrapper; spawns workers for multiple feature sets |
| `run_ml_cross_sectional.py` | Entry point; configures and runs pipeline (serial or parallel) |

### Model Training & Validation

| Module | Purpose |
|--------|---------|
| `training.py` | Nested cross-validation (outer/inner folds); metric aggregation |
| `models/` | Model implementations (logistic, elasticnet, xgboost, random_forest, svm_model) |
| `metrics.py` | Metric computation (AUC-ROC, AUC-PR, sensitivity, specificity, F1, Brier, etc.) |

### Data & Features

| Module | Purpose |
|--------|---------|
| `dataset.py` | Data loading and preprocessing |
| `features.py` | Feature engineering and scaling |
| `feature_sets.py` | Feature set definitions (RBD alone, RBD + Prodromal, etc.) |
| `outcomes.py` | Outcome definitions |
| `matching.py` | Case-control matching logic |

### Explainability & Reporting

| Module | Purpose |
|--------|---------|
| `explainability.py` | SHAP value computation and summarization |
| `report/cross_fs_plots.py` | Cross-feature-set comparison plots (per-model) |
| `plots.py` | Generic plotting utilities |

### Results Collection & Final Report

| Module | Purpose |
|--------|---------|
| `results_collector/collector.py` | Discover and load model run artifacts from disk |
| `results_collector/final_figure.py` | Generate cross-feature-set comparison figures, supplemental panels, summary tables |
| `results_collector/runner.py` | Orchestrate final report generation |
| `results_collector/figures.py` | ROC + confusion matrix grid figures |
| `results_collector/tables.py` | Summary table formatting |
| `results_collector/plot_utils.py` | Plotting helpers (ROC, PR, confusion matrices, palettes) |

---

## Usage

### 1. Run Serial Pipeline

```bash
PYTHONPATH=. python run_ml_cross_sectional.py
```

**Configuration** (top of `run_ml_cross_sectional.py`):
```python
MODE = "abk"  # RBD score mode
USE_PARALLEL = False
N_WORKERS = 2
```

**Output Structure** (Serial Mode):
```
results/ml_cross_sectional/
├── {feature_set}/
│   ├── {model}_{YYYYMMDD_HHMMSS_UUID}/
│   │   ├── mean_metrics.csv
│   │   ├── metrics_per_fold.csv
│   │   ├── confusion_matrices.json
│   │   ├── predictions_per_fold.csv
│   │   ├── shap_summary.csv
│   │   ├── permutation_importance.csv
│   │   └── cohort_stats.json
│   └── _report/
│       ├── figure_roc_cm.png
│       ├── table_*.csv
│       └── ...
└── _final_report_{YYYYMMDD_HHMMSS_UUID}/  # One per run (parallel + serial)
    ├── figure_feature_set_comparison.png
    ├── figure_feature_set_supplemental.png
    ├── table_best_model_summary.csv
    └── table_all_models_summary.csv
```

**Output Structure** (Parallel Mode):
```
results/ml_cross_sectional/
├── rbd_alone/
│   ├── xgboost_{YYYYMMDD_HHMMSS_UUID}/
│   ├── logistic_{YYYYMMDD_HHMMSS_UUID}/
│   └── ...
├── rbd_prodromal/
│   ├── xgboost_{YYYYMMDD_HHMMSS_UUID}/
│   └── ...
├── rbd_prs/
│   └── ...
├── rbd_prs_prodromal/
│   └── ...
├── cross_fs_comparison/
│   └── {YYYYMMDD_HHMMSS_UUID}/
│       ├── xgboost/
│       │   ├── figure_roc_pr.png
│       │   ├── figure_violin.png
│       │   └── ...
│       ├── logistic/
│       └── ...
└── _final_report_{YYYYMMDD_HHMMSS_UUID}/  # Single shared run_id
    ├── figure_feature_set_comparison.png
    ├── figure_feature_set_supplemental.png
    ├── table_best_model_summary.csv
    └── table_all_models_summary.csv
```

**Key**: All models within a parallel run share the same `{YYYYMMDD_HHMMSS_UUID}` for traceability.

### 2. Run Parallel Pipeline

```bash
PYTHONPATH=. python run_ml_cross_sectional.py  # with USE_PARALLEL=True
```

**Configuration** (top of `run_ml_cross_sectional.py`):
```python
USE_PARALLEL = True
N_WORKERS = 6  # One worker per feature set
TRAINING_MODE = "p1_combined"  # "p1_combined" or "standard"
SELECTION_METRIC = "auc_roc"  # Metric for selecting best model per feature set
```

**Features**:
- Spawns `N_WORKERS` worker processes (one per feature set)
- Each worker runs all models sequentially (with internal joblib CV parallelism)
- **Single shared run_id**: Generated once before worker pool, distributed to all workers
- Expected speedup: 3-4x on 16-core machine with 4 feature sets
- Generates cross-feature-set comparison figures automatically
- Generates final report with timestamped output directory

### 3. Generate Final Report (Standalone)

```python
from library.ml_cross_sectional.results_collector.runner import run_final_report
from pathlib import Path

results_root = Path("results/ml_cross_sectional")
out_dir = run_final_report(
    results_root=results_root,
    feature_sets=("rbd_alone", "rbd_prodromal", "rbd_prs", "rbd_prs_prodromal"),
    selection_metric="f1",  # or "auc_roc", "auc_pr"
    include_pr_curve=False,  # optional: add PR curves to main figure
)
```

---

## Training Paradigms

### Standard Mode
- Stratifies CV on all cases mixed together
- No case-control matching
- Use for exploratory analysis

### P1 Combined Mode (Recommended)
- **Stratification**: CV stratifies on incident cases only
- **Training**: Trains on incident + prevalent cases (more power)
- **Evaluation**: Evaluates on incident cases only (primary target)
- **Control matching**: 1:N per-fold control matching (case-control design)
- **Advantages**: Matches epidemiological study design, reduces bias from prevalent cases
- **Use case**: Primary analysis for risk stratification

---

## Run ID Traceability

**Single Shared Run ID**: Generated once at pipeline start, distributed to all workers.

Format: `YYYYMMDD_HHMMSS_XXXXX`
- `YYYYMMDD_HHMMSS` — timestamp of pipeline start
- `XXXXX` — 5-char UUID suffix for collision prevention

Example: `20260421_085329_d6014`

**Flow**:
1. `run_ml_cross_sectional.py` calls `run_all_feature_sets_p1_parallel()`
2. `parallel_pipeline.py:run_all_feature_sets_p1_parallel()` generates run_id at line 385
3. run_id distributed to all workers (lines 400-410)
4. Each worker stores results keyed by run_id
5. Final report folder named: `_final_report_YYYYMMDD_HHMMSS_XXXXX/`

**Output structure** (example):
```
results/ml_cross_sectional/
├── rbd_alone/
│   ├── xgboost_20260421_085329_d6014/
│   ├── logistic_20260421_085329_d6014/
│   └── ...
├── rbd_prodromal/
│   ├── xgboost_20260421_085329_d6014/
│   └── ...
└── _final_report_20260421_085329_d6014/
    ├── figure_feature_set_comparison.png
    ├── figure_feature_set_supplemental.png
    ├── table_best_model_summary.csv
    └── table_all_models_summary.csv
```

---

## Final Report Generation

The final report is generated by **`results_collector/runner.py:run_final_report()`**

### Quick Start

```python
from library.ml_cross_sectional.results_collector.runner import run_final_report
from pathlib import Path

results_root = Path("results/ml_cross_sectional")
out_dir = run_final_report(
    results_root=results_root,
    feature_sets=("rbd_alone", "rbd_prodromal", "rbd_prs", "rbd_prs_prodromal"),
    selection_metric="f1",  # or "auc_roc", "auc_pr"
    include_pr_curve=False,  # optional
)
print(f"Report: {out_dir}")
```

The final report generates a timestamped directory with comparison figures and tables across feature sets.

### Output Files and Generation Functions

| File | Generated By | Source File | Description |
|------|---|---|---|
| `figure_feature_set_comparison.png` | `plot_feature_set_comparison()` | `results_collector/final_figure.py:126` | ROC curves + confusion matrices for best model per feature set |
| `figure_feature_set_supplemental.png` | `plot_supplemental_figure()` | `results_collector/final_figure.py:445` | 2×2 panel: metrics bars, SHAP importance, Brier calibration, cohort composition |
| `table_best_model_summary.csv` | `make_best_model_summary_table()` | `results_collector/final_figure.py:653` | One row per feature set; best model metrics (AUC-ROC, AUC-PR, sensitivity, specificity, F1, Brier, threshold) |
| `table_all_models_summary.csv` | `make_all_models_summary_table()` | `results_collector/final_figure.py:699` | All models across all feature sets; useful for comparing all model performance |

**Note**: All functions are called from `results_collector/runner.py:run_final_report()` (line 142), which orchestrates the entire report generation pipeline.

### Automatic Generation (Parallel Pipeline)

If you run the parallel pipeline, the final report is generated **automatically**:

```bash
PYTHONPATH=. python run_ml_cross_sectional.py  # with USE_PARALLEL=True
```

The parallel pipeline calls `run_final_report()` automatically at line 222 of `parallel_pipeline.py`:
```python
final_dir = run_final_report(
    results_root=results_root,
    feature_sets=tuple(results.keys()),
    report_timestamp=run_id,
)
```

### Generating Functions

#### `run_final_report()` (`runner.py`)
- **Entry point** for final report generation
- Loads latest models for each feature set (per model type)
- Identifies **most recent run ID** from loaded models
- Uses that ID for report directory naming (e.g., `_final_report_xgboost_20260418_190800_k9m5p`)
- Calls downstream functions to generate figures and tables

**Parameters**:
- `results_root`: Root of ML results tree (default: `results/ml_cross_sectional`)
- `feature_sets`: Feature sets to compare (default: all)
- `selection_metric`: Model selection criterion (`auc_roc`, `auc_pr`, `f1`)
- `include_pr_curve`: Add PR curves alongside ROC (default: False)
- `report_timestamp`: Explicit run ID (optional, for replay/testing)

#### `_load_best_models()` (`final_figure.py`)
- Loads latest models for each feature set
- Selects best per feature set using `selection_metric`
- Returns `{fs: (ModelRunData, fs_label)}`
- Called by `run_final_report()`

#### `plot_feature_set_comparison()` (`final_figure.py`)
- Generates main comparison figure
- **Left panel**: ROC curves (and optional PR curves) for best models
- **Right panel**: Confusion matrices arranged as grid (max 2 per row)
- Metrics bar chart overlaid

**Output structure**:
- Creates nested `GridSpec` (ROC/PR on left, CMs on right)
- PR panel is optional via `include_pr_curve` parameter
- Dynamically adjusts figure height based on number of CMs

#### `plot_supplemental_figure()` (`final_figure.py`)
- Generates 2×2 supplemental panel
- **[0,0]**: Grouped bar chart (sensitivity, specificity, F1, accuracy)
- **[0,1]**: SHAP importance (top-5 per feature set) or permutation importance fallback
- **[1,0]**: Brier calibration score with error bars
- **[1,1]**: Cohort composition (cases/controls stacked horizontal bar)

#### `make_best_model_summary_table()` (`final_figure.py`)
- One row per feature set
- Columns: Feature Set, Model, AUC-ROC, AUC-PR, Sensitivity (%), Specificity (%), PPV (%), F1 (%), Accuracy (%), Brier, TP, FP, TN, FN, N, Threshold*
- Format: `mean (sd)` for metrics

#### `make_all_models_summary_table()` (`final_figure.py`)
- One row per (feature_set, model) combination
- Same columns as best model summary
- Loads all latest models across feature sets
- Useful for comparing all models

---

## Model Run ID Scheme

Models are saved with timestamped directories:

```
{model}_{YYYYMMDD_HHMMSS}          # Original format (no UUID)
{model}_{YYYYMMDD_HHMMSS_UUID}     # With 5-char UUID suffix (collision prevention)
```

Examples:
- `xgboost_20260417_230648_k9m5p` → XGBoost run on 2026-04-17 at 23:06:48
- `logistic_20260418_121649_k9m5p` → Logistic Regression run on 2026-04-18 at 12:16:49

**Note**: Different models within a feature set may be saved at different times. The final report uses the **most recent** model timestamp for its directory name, ensuring all outputs are linked to a traceable run.

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `feature_sets.py` | Define which features to include per feature set configuration |
| `outcomes.py` | Define binary outcomes to predict |
| `config/config.py` | Global paths (data, results, intermediate outputs) |

---

## Cross-Validation Strategy

**Nested CV** (via `training.py`):
- **Outer folds** (n_outer, default=5): Model selection, final evaluation
- **Inner folds** (n_inner, default=3): Hyperparameter tuning (for models that support it)
- Metrics reported as mean ± SD across outer folds

---

## Performance & Parallelization

### Serial Execution
- One feature set at a time
- All models run sequentially
- Joblib parallelism for CV folds (n_jobs=-1 by default)
- Runtime: ~2-3 hours for 4 feature sets on 16-core machine

### Parallel Execution
- ProcessPoolExecutor with `mp_context="spawn"` (Windows compatible)
- One worker per feature set
- BLAS thread limits set per worker to prevent oversubscription
- Expected speedup: **3-4x** (coarse-grained parallelism scales well with ~100k samples)

### Recommendations
- **Serial mode**: Development, debugging, small datasets
- **Parallel mode**: Production runs, all feature sets, full dataset

---

## Troubleshooting

### ConvergenceWarning: max_iter reached (SAG/L-BFGS solver)
**Cause**: Iterative optimizer didn't fully converge before hitting `max_iter` limit.

**Impact**: Model weights may not be optimal; typically small effect on AUC for logistic regression.

**Solutions** (in priority order):
1. **Increase `max_iter`** in `models/logistic.py`:
   ```python
   LogisticRegression(max_iter=1000, solver='lbfgs')  # default ~100
   ```
2. **Verify feature scaling**: Ensure `StandardScaler` is applied in `features.py`
3. **Reduce regularization**: Increase C parameter (e.g., C=10 instead of C=1)
4. **Switch solver**: Try `'lbfgs'` or `'newton-cg'` instead of `'sag'`
5. **Check data quality**: Look for outliers, missing values in feature matrix

**Monitoring**: If only a few models warn, continue. If all models warn, investigate feature preprocessing.

### Models not found when generating final report
**Cause**: Directory naming doesn't match regex `^(.+?)_(\d{8}_\d{6})(?:_([a-zA-Z0-9]{5}))?$`

**Solution**: Verify model directories in `results/ml_cross_sectional/{feature_set}/` follow the naming scheme.

### SHAP computation fails
**Fallback**: If SHAP unavailable, uses `permutation_importance` (slower but always works)

### Memory issues with large datasets
- Reduce `n_workers` in parallel mode
- Use serial mode with joblib `n_jobs=-2` (leave one core free)

### Final report not generated
**Cause**: No valid feature sets found in results directory.

**Solution**: Ensure at least one feature set completed training before calling `run_final_report()`

---

## References

- **Pipeline**: `pipeline.py`, `parallel_pipeline.py`
- **Results collection**: `results_collector/runner.py`, `results_collector/collector.py`
- **Final figures**: `results_collector/final_figure.py`, `results_collector/figures.py`
- **Metrics**: `metrics.py`, `training.py`
- **Feature sets**: `feature_sets.py`
