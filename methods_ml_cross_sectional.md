# Methods: Cross-Sectional ML Classification Analysis

## Study Population

Participants were selected from the UK Biobank longitudinal cohort with completed actigraphy (triaxial accelerometer; CWA format). Data cleaning followed standard protocols (exclusion: training sleep = True, neuro_exclude ≠ 0). Complete case analysis was applied due to minimal missing data (<1%).

## Outcome Definition

Binary classification: Parkinson's disease (PD) cases versus controls. Case status derived from linked electronic health records (EHR).

## Features and Feature Sets

### Demographics (all models)
- Age at recruitment (UKBB field 21022)
- Biological sex (UKBB field 31)
- Body mass index (UKBB field 21001)

### RBD Scoring
- Mean nightly RBD probability from ML model applied to raw actigraphy time series (abk_rbd_score_mean)

### Feature Set Variants
1. **RBD alone**: Demographics + RBD score (baseline)
2. **RBD + Prodromal**: Above + 8 binary HES-derived prodromal markers (anosmia, anxiety, constipation, depression, dream enactment, erectile dysfunction, hyposmia, orthostatic hypotension)
3. **RBD + PRS**: Above + polygenic risk score for PD (prs_score_pd; excluding RBD-specific variants)
4. **RBD + PRS + Prodromal**: Combined
5. **RBD + TMT**: Above + log-transformed Trail Making Test ratio (TMT-B/A)

## Model Architectures

Five algorithms evaluated in parallel:
- **Logistic Regression** (L2-regularized)
- **Elastic Net** (α ∈ [0,1], λ grid search)
- **XGBoost** (tree depth, learning rate, subsample fraction)
- **Random Forest** (500 trees, max depth, min samples split)
- **Support Vector Machine** (RBF kernel; C, γ grid; skipped in full runs due to O(n²) complexity)

## Training and Hyperparameter Optimization

### Nested Cross-Validation
- **Outer loop**: 5-fold stratified CV (test-set generalization estimate)
- **Inner loop**: 3-fold stratified CV within outer training folds (hyperparameter tuning only)
- **Random seed**: Fixed at 42 for reproducibility

### Optuna Hyperparameter Search
- **Objective**: Maximize average precision (AP) on inner validation folds
- **Trials per outer fold**: 50 (production run); 5 (smoke test)
- **Search space**: Model-specific Optuna samplers (sampler = TPESampler)

### Class Imbalance Handling
- Sample weighting: positives = n_neg / n_pos; negatives = 1.0
- XGBoost: native scale_pos_weight parameter
- Linear/tree models: explicit sample_weight argument

## Training Paradigms

### Standard
- CV stratified on all cases (incident + prevalent) mixed
- No control matching between folds

### P1 Combined (Primary)
- CV stratified on incident cases only
- Training includes incident + prevalent cases
- Evaluation restricted to incident cases
- 1:N control matching per fold (matched on demographics within each outer fold)

## Missing Data and Imputation

- **Missing data mechanism**: Missingness <1% across all features
- **Strategy**: Complete case analysis (drop rows with any NaN in features)
- **Imputation disabled** in primary analysis to avoid model complexity

## Model Refitting and Interpretation

After nested CV, final model refitted on complete dataset using best hyperparameters from last outer fold:
- **SHAP values** computed on held-out validation set (last fold) to quantify feature importance and interactions
- **Permutation importance** computed as secondary importance measure
- **Cohort statistics** reported for each feature (mean, SD, missingness)

## Evaluation Metrics

Per-fold (outer) metrics:
- **ROC-AUC**: Receiver-operator characteristic area under curve
- **PR-AUC**: Precision-recall area under curve (primary selection metric for best model)
- **F1-score**: Harmonic mean of precision and recall
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate

Reported as mean ± SD across outer folds.

## Time-to-Event Handling

Three variants swept in single run:
1. **Exclude**: Time-to-event feature (log days to PD diagnosis) dropped entirely (baseline)
2. **Jittered**: Time-to-event feature retained; controls imputed with uniform draws from [Q3, max] of training-fold cases

## Model Selection Criterion

Best model per feature set selected by maximum mean PR-AUC across outer folds (secondary verification: ROC-AUC).

## Reproducibility

- Random seed documented and fixed (seed = 42)
- All hyperparameters logged at runtime
- Re-runs produce identical results
- Results persisted to disk with run ID (timestamp + UUID suffix)

## Software and Environment

- **Python**: 3.11
- **Libraries**: scikit-learn, XGBoost, Optuna, pandas, numpy
- **Parallelization**: Optional multiprocessing across feature sets (N_WORKERS = 6)
