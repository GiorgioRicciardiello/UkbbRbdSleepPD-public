"""
run_ml_cross_sectional.py
=========================

Project-root entry point for the ``library.ml_cross_sectional`` pipeline.

Edit the ``MODE`` block below to choose between a smoke test (small random
subset) and a full-cohort run, then execute from the project root::

    C:/Users/riccig01/anaconda3/envs/stats_env/python.exe run_ml_cross_sectional.py

The script loads ``df_risk`` via ``get_clean_risk_data``, converts it to a
cross-sectional binary-classification problem (outcome = ever-PD), and runs
nested stratified CV with Optuna hyperparameter optimisation for every
registered model.

Three time-to-event variants are swept in a single run:

* ``exclude``  — feature dropped entirely (baseline, no temporal signal)
* ``constant`` — controls filled with the training-fold case 95th percentile
* ``jittered`` — controls filled with uniform draws on training-fold case [Q3, max]

Outputs land under ``results/ml_cross_sectional/<tte_mode>/<model>_<timestamp>/``.
"""
from __future__ import annotations

from library.ml_cross_sectional.pipeline import (
    RunConfig,
    load_data,
    run_all_feature_sets,
    run_all_feature_sets_p1,
    run_smoke_test,
)
from library.ml_cross_sectional.parallel_pipeline import (
    run_all_feature_sets_parallel,
    run_all_feature_sets_p1_parallel,
)

# ---------------------------------------------------------------------------
# MODE: either "smoke" (200/200 random subset) or "full" (real cohort).
# ---------------------------------------------------------------------------
MODE = "full"

# ---------------------------------------------------------------------------
# TRAINING_MODE: "standard" (all cases mixed) or "p1_combined" (incident vs prevalent)
# ---------------------------------------------------------------------------
# P1 Combined is recommended:
# - Stratifies CV on incident cases only
# - Includes both incident+prevalent in training, but evaluates only on incident
# - Does 1:N control matching per fold
TRAINING_MODE = "p1_combined"

# ---------------------------------------------------------------------------
# PARALLEL: enable multi-process feature-set execution.
# ---------------------------------------------------------------------------
USE_PARALLEL = True
N_WORKERS = 6  # Number of worker processes (2-3 for 100k rows on 16-core machine)

# ---------------------------------------------------------------------------
# MODEL SELECTION: metric used to select best model per feature set.
# ---------------------------------------------------------------------------
# Options: "auc_roc" (default), "auc_pr", "youden"
SELECTION_METRIC = "auc_roc"


def main() -> None:
    """Dispatch to the selected mode."""
    if MODE == "smoke":
        # Small, fast validation run: 100 cases + 100 random controls,
        # 2 outer x 2 inner x 5 trials, tte modes = (exclude, jittered),
        # SVM skipped for speed.
        print(f"\n[SMOKE TEST MODE]")
        if USE_PARALLEL:
            print(f"[PARALLEL: {N_WORKERS} workers]")
        else:
            print(f"[SERIAL]")
        cfg = RunConfig(
            n_outer=2, n_inner=2, n_trials=5, skip_models=("svm_rbf",),
        )
        run_smoke_test(
            n_cases=100, n_controls=100, cfg=cfg,
            use_parallel=USE_PARALLEL, n_workers=N_WORKERS,
            selection_metric=SELECTION_METRIC,
        )
    elif MODE == "full":
        # Production run: full cohort, 5 outer x 3 inner x 50 trials per
        # model, tte modes = (exclude, jittered). SVM skipped: O(n^2)
        # training is prohibitive on large cohorts.
        print(f"\n[FULL RUN MODE]")
        print(f"[TRAINING MODE: {TRAINING_MODE.upper()}]")
        if USE_PARALLEL:
            print(f"[PARALLEL: {N_WORKERS} workers]")
        else:
            print(f"[SERIAL]")
        cfg = RunConfig(skip_models=("svm_rbf",))
        df = load_data(cfg.file_name)

        if TRAINING_MODE == "p1_combined":
            # P1 Combined: stratify on incident only, match controls per fold
            print("\nUsing P1 Combined training paradigm:")
            print("  - Stratifies CV on incident cases only")
            print("  - Trains on incident + prevalent, evaluates on incident only")
            print("  - Does 1:N control matching per fold")
            if USE_PARALLEL:
                print(f"\n[PARALLEL: {N_WORKERS} workers]")
                run_all_feature_sets_p1_parallel(df, cfg, n_workers=N_WORKERS, selection_metric=SELECTION_METRIC)
            else:
                run_all_feature_sets_p1(df, cfg)
        else:
            # Standard: treat all cases equally
            print("\nUsing Standard training paradigm:")
            print("  - Stratifies CV on all cases mixed together")
            print("  - No control matching")
            if USE_PARALLEL:
                run_all_feature_sets_parallel(df, cfg, n_workers=N_WORKERS, selection_metric=SELECTION_METRIC)
            else:
                run_all_feature_sets(df, cfg)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Expected 'smoke' or 'full'.")


if __name__ == "__main__":
    main()
