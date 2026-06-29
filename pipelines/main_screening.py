"""
Screening model training paradigm comparison.

Entry point for running all paradigms through the same 10-fold outer CV /
5-fold inner hyperparameter-tuning loop using XGBoost.

CV structure
------------
Outer: StratifiedKFold(n=10), stratified on y_incident (1 = incident PD, 0 = other).
Inner: StratifiedKFold(n=5) for RandomizedSearchCV, scored by average_precision
       (PR-AUC), appropriate for severe class imbalance.

Matching: within each training fold, controls are randomly sampled at 1:10
          per combined case count (paradigm-specific logic determines what
          counts as a "case").

Evaluation: always on incident cases + controls from the held-out test fold.
            Prevalent cases in the test fold are excluded from evaluation.

Results
-------
- ``results/<timestamp>_paradigm_comparison_folds.csv``: per-fold metrics
- ``results/<timestamp>_paradigm_comparison_summary.csv``: mean ± SD table
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from config.config import config
from library.screening.config import (
    CONTROLS_PER_CASE,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FOCAL_LOSS_ENABLED,
    INNER_FOLDS,
    OUTER_FOLDS,
    RANDOM_SEED,
    XGB_JOBS,
    XGB_N_ITER,
    XGB_PARAM_DISTRIBUTIONS,
    ALL_FEATURE_COLS,
)
from library.screening.data_loader import load_ml_dataset, get_feature_columns
from library.screening.evaluation import (
    FoldMetrics,
    compile_results_table,
    evaluate_fold,
)
from library.screening.features import (
    build_preprocessor,
    extract_feature_matrix,
)
from library.screening.paradigms import (
    CombinedTrainingParadigm,
    IncidentOnlyParadigm,
    WeightedTrainingParadigm,
    SubsamplingParadigm,
    PrevalentTrainParadigm,
)
from library.screening.paradigms.base import BaseParadigm
from library.screening.plot_results import run as run_plots
from library.screening.best_model_analysis import generate_full_report
from library.screening.focal_loss import compute_focal_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paradigm registry ─────────────────────────────────────────────────────────
# Add or remove paradigms here; the CV loop is paradigm-agnostic.
PARADIGMS: List[BaseParadigm] = [
    CombinedTrainingParadigm(),
    IncidentOnlyParadigm(),
    WeightedTrainingParadigm(alpha=0.3),
    WeightedTrainingParadigm(alpha=0.1),   # aggressive down-weighting
    SubsamplingParadigm(ratio=5),
    PrevalentTrainParadigm(),
]


# ── XGBoost factory ───────────────────────────────────────────────────────────

def build_xgb() -> XGBClassifier:
    """
    Build a base XGBClassifier with fixed non-tuned settings.

    ``scale_pos_weight`` is intentionally left at 1 because class imbalance
    is handled through (a) 1:10 matching and (b) ``sample_weight`` passed to
    ``fit``.  Setting both would double-count the imbalance correction.
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        use_label_encoder=False,
        scale_pos_weight=1,
        tree_method="hist",            # fast histogram-based training
        random_state=RANDOM_SEED,
        n_jobs=10,                      # inner CV manages parallelism via n_jobs
        verbosity=0,
    )


# ── CV loop ───────────────────────────────────────────────────────────────────

def run_paradigm(
    paradigm: BaseParadigm,
    df_ml: pd.DataFrame,
    feature_cols: List[str],
    outer_cv: StratifiedKFold,
    y_stratify: np.ndarray,
) -> List[FoldMetrics]:
    """
    Run one paradigm through the full outer CV loop.

    Parameters
    ----------
    paradigm : BaseParadigm
    df_ml : pd.DataFrame
        Full analytical cohort with y_incident, y_prevalent, y_control columns.
    feature_cols : list[str]
        Feature columns present in the dataset.
    outer_cv : StratifiedKFold
        Pre-built outer CV splitter.
    y_stratify : np.ndarray
        Stratification labels (1 = incident PD, 0 = all others).

    Returns
    -------
    list[FoldMetrics]
        One entry per outer fold.
    """
    logger.info("=" * 60)
    logger.info("Running paradigm: %s", paradigm.name)
    logger.info("  %s", paradigm.description)

    fold_metrics: List[FoldMetrics] = []
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(df_ml, y_stratify)
    ):
        logger.info("  Fold %d/%d", fold_idx + 1, OUTER_FOLDS)

        df_fold_train = df_ml.iloc[train_idx].copy()
        df_fold_test = df_ml.iloc[test_idx].copy()

        # ── 1. Paradigm prepares training data ────────────────────────────────
        # Fold-specific seed: deterministic but varied across folds
        fold_rng = np.random.default_rng(RANDOM_SEED + fold_idx)
        try:
            df_selected = paradigm.prepare_training_data(
                df_fold_train, CONTROLS_PER_CASE, fold_rng
            )
        except RuntimeError as exc:
            logger.error("  Fold %d: paradigm failed — %s. Skipping.", fold_idx, exc)
            continue

        # ── 2. Build feature matrices (raw; preprocessing fitted on train) ────
        X_train_raw = extract_feature_matrix(df_selected, feature_cols)
        y_train = df_selected["y_label"].values.astype(int)
        sample_weights = df_selected["sample_weight"].values.astype(float)

        # ── 3. Fit preprocessing pipeline on training fold ────────────────────
        preprocessor = build_preprocessor(feature_cols)
        X_train = preprocessor.fit_transform(X_train_raw)

        # ── 4. Build test set: incident cases + controls only ─────────────────
        test_eval_mask = df_fold_test["y_incident"] | df_fold_test["y_control"]
        df_test_eval = df_fold_test[test_eval_mask]

        if df_test_eval["y_incident"].sum() == 0:
            logger.warning("  Fold %d: no incident cases in test set — skipping.", fold_idx)
            continue

        X_test_raw = extract_feature_matrix(df_test_eval, feature_cols)
        X_test = preprocessor.transform(X_test_raw)
        y_test = df_test_eval["y_incident"].astype(int).values

        # ── 5. Inner CV hyperparameter tuning ─────────────────────────────────
        # Inner stratification uses y_train labels to maintain positive fraction
        # in each inner fold.
        xgb = build_xgb()
        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=XGB_PARAM_DISTRIBUTIONS,
            n_iter=XGB_N_ITER,
            cv=inner_cv,
            scoring="average_precision",
            n_jobs=XGB_JOBS,
            random_state=RANDOM_SEED,
            refit=True,
        )
        # sample_weight is sliced per inner fold by sklearn (requires numpy array)
        search.fit(X_train, y_train, sample_weight=sample_weights)

        best_model = search.best_estimator_
        logger.info(
            "    Best inner CV PR-AUC=%.4f | params=%s",
            search.best_score_,
            search.best_params_,
        )

        # ── Focal Loss (optional) ───────────────────────────────────────────────
        # NOTE: Post-hoc focal loss retraining causes overfitting (retraining on same
        # data with new objective after inner CV has selected hyperparams for standard loss).
        # Correct approach requires integrating focal loss into XGBoost's objective function
        # DURING inner CV training. Disabled for now.
        if FOCAL_LOSS_ENABLED:
            logger.warning(
                "    Focal loss currently disabled (requires XGBoost objective integration)"
            )

        # ── 6. Evaluate on incident test fold ─────────────────────────────────
        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = evaluate_fold(
            y_true=y_test,
            y_prob=y_prob,
            fold=fold_idx,
            paradigm=paradigm.name,
            best_params=search.best_params_,
        )
        fold_metrics.append(metrics)

        logger.info(
            "    Test → ROC-AUC=%.3f | PR-AUC=%.3f | Brier=%.4f | CalSlope=%.3f",
            metrics.roc_auc, metrics.pr_auc, metrics.brier_score,
            metrics.calibration_slope,
        )

    return fold_metrics


# ── Results persistence ───────────────────────────────────────────────────────

def save_results(
    all_metrics: Dict[str, List[FoldMetrics]],
    out_dir: Path,
    timestamp: str,
) -> None:
    """
    Save per-fold and summary results to CSV.

    Parameters
    ----------
    all_metrics : dict
        ``{paradigm_name: [FoldMetrics, ...]}``
    out_dir : Path
        Output directory (created if absent).
    timestamp : str
        Timestamp string prefix for file names.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold detail
    all_rows = []
    for paradigm_name, folds in all_metrics.items():
        all_rows.extend([m.to_dict() for m in folds])
    df_folds = pd.DataFrame(all_rows)
    folds_path = out_dir / f"{timestamp}_paradigm_comparison_folds.csv"
    df_folds.to_csv(folds_path, index=False)
    logger.info("Saved per-fold results → %s", folds_path)

    # Summary mean ± SD
    df_summary = compile_results_table(all_metrics)
    summary_path = out_dir / f"{timestamp}_paradigm_comparison_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info("Saved summary results → %s", summary_path)

    # Print summary to console
    print("\n" + "=" * 70)
    print("PARADIGM COMPARISON SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70)


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """
    Run all paradigms through the 10-fold outer / 5-fold inner CV loop.

    Steps:
      1. Load dataset via canonical pipeline.
      2. Build outer CV splitter (stratified on y_incident).
      3. For each paradigm: run full CV, collect metrics.
      4. Save per-fold and summary results.
      5. Generate comparison figures.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = config["results"]["root"] / "screening_paradigms"
    out_dir = base_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to: %s", out_dir)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("[1/5] Loading dataset ...")
    _, df_ml = load_ml_dataset()
    feature_cols = get_feature_columns(df_ml)
    logger.info("  Feature columns available: %d / %d", len(feature_cols), len(ALL_FEATURE_COLS))
    logger.info("  Cohort: %d subjects", len(df_ml))

    # ── 2. Outer CV splitter ──────────────────────────────────────────────────
    logger.info("[2/5] Building outer CV splitter (stratified on y_incident) ...")
    # Stratify on incident label so each fold has proportional incident cases.
    # Prevalent cases are treated as 0 for stratification (they are excluded
    # from evaluation regardless).
    y_stratify = df_ml["y_incident"].astype(int).values
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ── 3. Run paradigms ──────────────────────────────────────────────────────
    if FOCAL_LOSS_ENABLED:
        logger.info("[3/5] Running %d paradigms (with Focal Loss) ...", len(PARADIGMS))
    else:
        logger.info("[3/5] Running %d paradigms ...", len(PARADIGMS))
    all_metrics: Dict[str, List[FoldMetrics]] = {}

    for paradigm in PARADIGMS:
        fold_metrics = run_paradigm(
            paradigm=paradigm,
            df_ml=df_ml,
            feature_cols=feature_cols,
            outer_cv=outer_cv,
            y_stratify=y_stratify,
        )
        all_metrics[paradigm.name] = fold_metrics

    # ── 4. Save results ───────────────────────────────────────────────────────
    logger.info("[4/5] Saving results ...")
    folds_csv = out_dir / f"{timestamp}_paradigm_comparison_folds.csv"
    save_results(all_metrics, out_dir, timestamp)

    # ── 5. Generate figures and report ────────────────────────────────────────
    logger.info("[5/5] Generating comparison figures and report ...")
    run_plots(folds_csv=folds_csv)

    # Compute summary and generate detailed report
    summary_df = compile_results_table(all_metrics)
    best_idx = summary_df["roc_auc_mean"].idxmax()
    best_paradigm = summary_df.loc[best_idx, "paradigm"]
    logger.info("Best paradigm (by ROC-AUC): %s", best_paradigm)
    report_path = generate_full_report(all_metrics, summary_df, best_paradigm, out_dir)
    logger.info("Generated detailed report → %s", report_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
