# Screening Model — Training Paradigm Comparison

Machine learning–based PD risk screening using baseline UKBB features.
Compares training paradigms that differ in how prevalent PD cases are
incorporated, given that all prevalent cases are under dopaminergic
medication at actigraphy time (confounded motor features).

---

## Structure

```
src/screening/
├── __init__.py
├── config.py           — feature sets, CV params, XGBoost search space
├── data_loader.py      — wraps load_prodromal_dataset(), attaches ML labels
├── features.py         — sklearn ColumnTransformer (imputation + OHE)
├── matching.py         — random 1:N case-control matching within folds
├── evaluation.py       — ROC-AUC, PR-AUC, Brier, calibration slope
├── paradigms/
│   ├── __init__.py     — registry
│   ├── base.py         — BaseParadigm ABC
│   ├── p1_combined.py  — prevalent + incident, uniform weights
│   ├── p2_incident_only.py — incident only, prevalent excluded
│   ├── p3_weighted.py  — prevalent down-weighted by α (default 0.3)
│   ├── p4_subsampling.py — variable control ratio (default 1:5)
│   └── p6_prevalent_train.py — train on prevalent, evaluate on incident
└── main.py             — orchestrates full CV loop across all paradigms
```

---

## Feature Sets

| Group | Columns | Notes |
|---|---|---|
| Demographics | `cov_age_recruitment_21022`, `cov_sex_31`, `cov_bmi`, `cov_smoking`, `cov_alcohol` | Median imputed |
| RBD (continuous) | `abk_rbd_score_mean` | Nightly mean probability, subject-level |
| RBD (categorical) | `rg_pctl3` | One-hot; Low/Intermediate/High; present for all subjects |
| Prodromal (binary) | `prodromal_constipation`, `_depression`, `_anxiety`, `_orthostatic`, `_erectile_dysfunction`, `_dream_enactment`, `_anosmia`, `_hyposmia` | HES ICD-10 derived; NaN → 0 (not diagnosed) |
| PRS | `prs_score_pd` | Median imputed (missing for non-European subjects) |
| TMT | `tmt_ratio_baseline`, `tmt_missing` | ~50% availability; ratio median-imputed; `tmt_missing` is a pre-computed 0/1 flag |

---

## CV Structure

```
StratifiedKFold(outer=10, shuffle=True, seed=42)
    stratified on y_incident (1 = incident PD, 0 = all others)

    for each outer fold:
        paradigm.prepare_training_data()   →  df_selected, y_label, sample_weight
        preprocessor.fit_transform(X_train)  ← fit on training fold only
        RandomizedSearchCV(inner=5, n_iter=30, scoring='average_precision')
            .fit(X_train, y_train, sample_weight=w)
        evaluate on: incident cases + controls from test fold
                     (prevalent cases excluded from evaluation)
```

Matching ratio: **1 case : 10 controls** (random, without replacement, within each training fold).
`scale_pos_weight` is not used — imbalance handled by matching + `sample_weight`.

---

## Paradigms

| ID | File | Training positives | Weights | Purpose |
|---|---|---|---|---|
| P1 | `p1_combined.py` | Prevalent + incident | Uniform | Signal maximisation baseline |
| P2 | `p2_incident_only.py` | Incident only | Uniform | Prospective upper-bound reference |
| P3 | `p3_weighted.py` | Prevalent + incident | Incident=1.0, prevalent=α | Recommended primary paradigm |
| P4 | `p4_subsampling.py` | Prevalent + incident | Uniform | Variable control ratio test |
| P6 | `p6_prevalent_train.py` | Prevalent only | Uniform | Cross-sectional transfer exploratory |

P3 is run at α ∈ {0.1, 0.3} by default (two separate entries in `PARADIGMS` list in `main.py`).

---

## Running

```bash
cd UkbbRbdSleepPD
C:/Users/riccig01/anaconda3/envs/stats_env/python.exe -m src.screening.main
```

Results are saved to `results/screening_paradigms/`:
- `<timestamp>_paradigm_comparison_folds.csv` — per-fold ROC-AUC, PR-AUC, Brier, calibration slope
- `<timestamp>_paradigm_comparison_summary.csv` — mean ± SD across 10 folds per paradigm

---

## Adding a New Paradigm

1. Create `src/screening/paradigms/pN_name.py` implementing `BaseParadigm`.
2. Implement `name`, `description`, and `prepare_training_data()`.
   - Input: `df_fold_train` (full training fold with `y_incident`, `y_prevalent`, `y_control`)
   - Output: same DataFrame subset with `y_label` (0/1) and `sample_weight` columns added.
3. Register the class in `paradigms/__init__.py`.
4. Add an instance to `PARADIGMS` in `main.py`.
