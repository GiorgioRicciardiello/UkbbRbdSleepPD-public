"""
run_ml_p1_combined.py
=====================

Top-level training script: P1 Combined paradigm for ml_cross_sectional.

Training strategy
-----------------
* Cases = incident + prevalent positives (both treated as y=1).
* Controls = 1:N random matching per outer fold from the ``control`` column.
* Outer CV stratified on incident labels; test fold = incident + controls only.
* Inner CV via Optuna (PR-AUC objective), identical to the standard pipeline.
* All four feature sets are swept; results are saved per model per feature set.
* Cross-feature-set comparison plots (8 figures per model) are written after
  all feature sets complete.

Usage
-----
Edit the ``# --- Configuration ---`` block below, then run::

    C:/Users/riccig01/anaconda3/envs/stats_env/python.exe run_ml_p1_combined.py

For a quick smoke test::

    python run_ml_p1_combined.py --smoke

(The ``--smoke`` flag reduces outer/inner folds and Optuna trials.)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --- Project imports ---------------------------------------------------------
from library.ml_cross_sectional.dataset import convert_to_cross_sectional
from library.ml_cross_sectional.explainability import compute_permutation_importance, compute_shap
from library.ml_cross_sectional.feature_sets import FEATURE_SETS
from library.ml_cross_sectional.features import ImputerPipeline, get_feature_matrix
from library.ml_cross_sectional.metrics import compute_cohort_stats
from library.ml_cross_sectional.models import ALL_MODELS
from library.ml_cross_sectional.models.base import ModelBase
from library.ml_cross_sectional.plots import plot_all_for_run
from library.ml_cross_sectional.report.cross_fs_plots import generate_all_cross_fs_plots
from library.ml_cross_sectional.storage import ResultsWriter
from library.ml_cross_sectional.training import NestedCVResult, _compute_sample_weight
from library.ml_cross_sectional.training_p1 import P1CombinedConfig, P1CombinedTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_ROOT = Path("results/ml_cross_sectional_p1")

# =============================================================================
# --- Configuration -----------------------------------------------------------
# Edit these values; do NOT add CLI parsing for production runs.
# =============================================================================

DEFAULT_CFG = P1CombinedConfig(
    outcome_name="pd_only",        # see library/ml_cross_sectional/outcomes.py
    n_outer=5,
    n_inner=3,
    n_trials=50,
    controls_per_case=10,          # 1:10 case:control matching
    random_state=42,
    tte_strategy="exclude",        # drop time_to_event_log feature
    file_name="ehr_diag_pd_rbd_only_all",
)

FEATURE_SETS_TO_RUN: tuple[str, ...] = ()  # empty = all registered feature sets

SKIP_MODELS: tuple[str, ...] = ("svm_rbf",)  # SVM is intractable on the full cohort

# =============================================================================


# --- Data loading ------------------------------------------------------------

def load_and_prepare(cfg: P1CombinedConfig) -> tuple[
    pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series
]:
    """
    Load data and return (X, y_label, y_incident, y_prevalent, y_control).

    All series are aligned with X (same positional index after reset).

    Parameters
    ----------
    cfg :
        P1CombinedConfig with file_name and outcome_name.

    Returns
    -------
    X :
        Pre-imputation feature matrix.
    y_label :
        Combined binary outcome (incident | prevalent).
    y_incident :
        Incident-only binary labels.
    y_prevalent :
        Prevalent-only binary labels.
    y_control :
        Boolean control flag (outcome-agnostic ``control`` column).
    """
    from library.risk.risk_helpers import get_clean_risk_data
    _, df = get_clean_risk_data(file_name=cfg.file_name)
    df = df.drop_duplicates(subset=["eid"], keep="first")
    logger.info("Loaded %d subjects from %s", len(df), cfg.file_name)

    frame = convert_to_cross_sectional(df, outcome_name=cfg.outcome_name)
    # When tte_strategy=="exclude", omit TTE from X entirely so that cohort
    # stats and distribution reports reflect only the features actually used.
    _include_tte = cfg.tte_strategy != "exclude"
    X, y_label = get_feature_matrix(frame, include_tte=_include_tte)

    # Extract P1 auxiliary series aligned with the frame's reset index.
    df_frame = frame.df
    y_incident = _get_binary_col(df_frame, frame.incident_col, "y_incident")
    y_prevalent = _get_binary_col(df_frame, frame.prevalent_col, "y_prevalent")
    y_control = _get_bool_col(df_frame, frame.control_col, "y_control")

    logger.info(
        "Outcome: %s | incident=%d | prevalent=%d | controls=%d",
        cfg.outcome_name,
        int(y_incident.sum()),
        int(y_prevalent.sum()),
        int(y_control.sum()),
    )
    return X, y_label, y_incident, y_prevalent, y_control


def _get_binary_col(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    if col and col in df.columns:
        return df[col].fillna(0).astype(int).reset_index(drop=True)
    logger.warning("Column %r not found; treating %s as all-zero.", col, name)
    return pd.Series(np.zeros(len(df), dtype=int), name=name)


def _get_bool_col(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    if col and col in df.columns:
        return df[col].fillna(False).astype(bool).reset_index(drop=True)
    logger.warning("Column %r not found; treating %s as all-True.", col, name)
    return pd.Series(np.ones(len(df), dtype=bool), name=name)


# --- Per-model runner --------------------------------------------------------

def run_model_p1(
    model_cls: type[ModelBase],
    X: pd.DataFrame,
    y_label: pd.Series,
    y_incident: pd.Series,
    y_prevalent: pd.Series,
    y_control: pd.Series,
    cfg: P1CombinedConfig,
    feature_set: str,
    results_root: Path = RESULTS_ROOT,
) -> tuple[Path, NestedCVResult]:
    """
    Run P1 Combined nested CV for a single model + feature set.

    Returns
    -------
    run_dir :
        Directory where results were saved.
    result :
        NestedCVResult for downstream cross-feature-set comparison.
    """
    print(f"\n=== Model: {model_cls.name} | feature_set={feature_set} ===", flush=True)

    trainer = P1CombinedTrainer(model_cls=model_cls, cfg=cfg)
    result = trainer.fit(X, y_label, y_incident, y_prevalent, y_control)

    metrics = result.metrics_frame()
    print(
        metrics[["fold", "auc_pr", "auc_roc", "f1", "sensitivity", "specificity"]].to_string(
            index=False
        )
    )
    mean = result.mean_metrics()
    print(f"\nMean PR-AUC:  {mean.loc['auc_pr','mean']:.4f} +/- {mean.loc['auc_pr','sd']:.4f}")
    print(f"Mean ROC-AUC: {mean.loc['auc_roc','mean']:.4f} +/- {mean.loc['auc_roc','sd']:.4f}")

    # Final global refit for SHAP/permutation (uses last fold's best params).
    final_imp = ImputerPipeline(
        tte_strategy=cfg.tte_strategy, random_state=cfg.random_state,
    ).fit(X, y_label)
    X_full = final_imp.transform(X)
    final_model = model_cls(params=result.folds[-1].best_params, random_state=cfg.random_state)
    sw = _compute_sample_weight(y_label) if final_model.supports_sample_weight else None
    final_model.fit(X_full, y_label, sample_weight=sw)

    last = result.folds[-1]
    te_idx = last.val_indices
    train_mask = np.ones(len(X_full), dtype=bool)
    train_mask[te_idx] = False
    shap_result = compute_shap(final_model, X_full.iloc[train_mask], X_full.iloc[~train_mask])
    perm = compute_permutation_importance(
        final_model, X_full.iloc[~train_mask], y_label.iloc[~train_mask],
    )

    cohort = compute_cohort_stats(X, y_label)
    fs_dir = results_root / cfg.outcome_name / feature_set
    writer = ResultsWriter(base_dir=fs_dir, model_name=model_cls.name)
    run_dir = writer.save_all(
        result=result,
        cohort=cohort,
        shap_result=shap_result,
        perm=perm,
        X=X,
        y=y_label,
    )

    figs = plot_all_for_run(shap_result, run_dir)
    if figs:
        print(f"  SHAP figures: {len(figs)} written to {run_dir / 'figures'}")

    print(f"Saved → {run_dir}", flush=True)
    return run_dir, result


# --- Feature set + model sweep -----------------------------------------------

def run_all_p1(
    df_raw: pd.DataFrame,
    cfg: P1CombinedConfig,
    feature_sets: Iterable[str] | None = None,
    models: Iterable[type[ModelBase]] = ALL_MODELS,
    skip_models: Iterable[str] = SKIP_MODELS,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, dict[str, tuple[Path, NestedCVResult]]]:
    """
    Sweep all feature sets × all models using P1 Combined training.

    After all models complete for a feature set, 8 cross-feature-set comparison
    plots are written per model to ``results_root/{outcome}/cross_fs_comparison/``.

    Parameters
    ----------
    df_raw :
        Raw subject-level DataFrame from get_clean_risk_data (de-duplicated).
    cfg :
        P1CombinedConfig.
    feature_sets :
        Feature set names to sweep. None = all registered sets.
    models :
        Model classes to train.
    skip_models :
        Model names to skip (e.g. "svm_rbf" for tractability).
    results_root :
        Root directory for results.

    Returns
    -------
    dict[feature_set, dict[model_name, (run_dir, NestedCVResult)]]
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS_TO_RUN or tuple(FEATURE_SETS.keys())
    skip_set = set(skip_models)

    # Accumulate per-model results across feature sets.
    # Structure: {model_name: {feature_set: NestedCVResult}}
    model_fs_results: dict[str, dict[str, NestedCVResult]] = {
        m.name: {} for m in models if m.name not in skip_set
    }
    out: dict[str, dict[str, tuple[Path, NestedCVResult]]] = {}

    for fs in feature_sets:
        if fs not in FEATURE_SETS:
            print(f"\n!!! Unknown feature set: {fs!r} (skipping)")
            continue
        print(f"\n########## FEATURE SET: {fs} ##########")

        # Build feature-set-specific X from the shared raw df.
        frame_fs = convert_to_cross_sectional(
            df_raw, outcome_name=cfg.outcome_name, feature_set=fs,
        )
        _include_tte = cfg.tte_strategy != "exclude"
        X_fs, y_fs = get_feature_matrix(frame_fs, include_tte=_include_tte)

        y_inc_fs = _get_binary_col(frame_fs.df, frame_fs.incident_col, "y_incident")
        y_prev_fs = _get_binary_col(frame_fs.df, frame_fs.prevalent_col, "y_prevalent")
        y_ctrl_fs = _get_bool_col(frame_fs.df, frame_fs.control_col, "y_control")

        out[fs] = {}
        for model_cls in models:
            if model_cls.name in skip_set:
                print(f"\n--- Skipping {model_cls.name} ---")
                continue
            run_dir, result = run_model_p1(
                model_cls, X_fs, y_fs, y_inc_fs, y_prev_fs, y_ctrl_fs,
                cfg=cfg, feature_set=fs, results_root=results_root,
            )
            out[fs][model_cls.name] = (run_dir, result)
            model_fs_results[model_cls.name][fs] = result

    # --- Cross-feature-set comparison plots ----------------------------------
    print("\n########## CROSS-FEATURE-SET COMPARISON PLOTS ##########")
    cross_fs_dir = results_root / cfg.outcome_name / "cross_fs_comparison"
    for model_name, fs_results in model_fs_results.items():
        if len(fs_results) < 2:
            continue  # need at least 2 feature sets to compare
        print(f"  {model_name}: generating 8 comparison figures …")
        try:
            paths = generate_all_cross_fs_plots(
                results=fs_results,
                out_dir=cross_fs_dir,
                model_name=model_name,
            )
            print(f"  → {len(paths)} figures written to {cross_fs_dir}")
        except Exception as exc:
            print(f"  [WARN] Cross-FS plots failed for {model_name}: {exc}")

    # Summary table.
    print("\n########## SUMMARY ##########")
    for fs, model_results in out.items():
        print(f"\nFeature set: {fs}")
        for model_name, (run_dir, result) in model_results.items():
            mean = result.mean_metrics()
            try:
                pr = mean.loc["auc_pr", "mean"]
                roc = mean.loc["auc_roc", "mean"]
                pr_sd = mean.loc["auc_pr", "sd"]
                roc_sd = mean.loc["auc_roc", "sd"]
                print(f"  {model_name:>15s}  PR-AUC={pr:.4f}±{pr_sd:.4f}  ROC-AUC={roc:.4f}±{roc_sd:.4f}")
            except Exception:
                print(f"  {model_name:>15s}  (metrics unavailable)")

    return out


# --- Smoke test --------------------------------------------------------------

def run_smoke_test(cfg: P1CombinedConfig | None = None) -> None:
    """Run a quick sanity check on a small random subset."""
    if cfg is None:
        cfg = P1CombinedConfig(n_outer=2, n_inner=2, n_trials=3, controls_per_case=5)
    X, y_label, y_incident, y_prevalent, y_control = load_and_prepare(cfg)

    # Subsample for speed: 200 cases + 200 controls from the valid control pool.
    rng = np.random.default_rng(42)
    pos_idx = np.where(y_label.values == 1)[0]
    neg_idx = np.where(y_control.values)[0]
    pos_sel = rng.choice(pos_idx, size=min(200, len(pos_idx)), replace=False)
    neg_sel = rng.choice(neg_idx, size=min(200, len(neg_idx)), replace=False)
    keep = np.sort(np.concatenate([pos_sel, neg_sel]))

    X_s = X.iloc[keep].reset_index(drop=True)
    y_label_s = y_label.iloc[keep].reset_index(drop=True)
    y_inc_s = y_incident.iloc[keep].reset_index(drop=True)
    y_prev_s = y_prevalent.iloc[keep].reset_index(drop=True)
    y_ctrl_s = y_control.iloc[keep].reset_index(drop=True)

    print(f"Smoke subset: n={len(keep)} | label_pos={int(y_label_s.sum())} | "
          f"incident={int(y_inc_s.sum())} | prevalent={int(y_prev_s.sum())} | "
          f"controls={int(y_ctrl_s.sum())}")

    from library.ml_cross_sectional.models.xgboost_model import XGBoostModel
    from library.ml_cross_sectional.report.distribution import fold_composition_table
    trainer = P1CombinedTrainer(model_cls=XGBoostModel, cfg=cfg)
    result = trainer.fit(X_s, y_label_s, y_inc_s, y_prev_s, y_ctrl_s)
    print("Smoke test passed. Fold metrics:")
    print(result.metrics_frame()[["fold", "auc_pr", "auc_roc"]].to_string(index=False))
    print("\nFold composition:")
    print(fold_composition_table(result).to_string(index=False))


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P1 Combined training script")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on a small subset")
    args = parser.parse_args()

    if args.smoke:
        smoke_cfg = P1CombinedConfig(n_outer=2, n_inner=2, n_trials=5, controls_per_case=5)
        run_smoke_test(cfg=smoke_cfg)
    else:
        cfg = DEFAULT_CFG
        from library.risk.risk_helpers import get_clean_risk_data
        _, df_raw = get_clean_risk_data(file_name=cfg.file_name)
        df_raw = df_raw.drop_duplicates(subset=["eid"], keep="first")
        logger.info("Loaded %d subjects from %s", len(df_raw), cfg.file_name)
        run_all_p1(
            df_raw=df_raw,
            cfg=cfg,
            feature_sets=FEATURE_SETS_TO_RUN or None,
            skip_models=SKIP_MODELS,
            results_root=RESULTS_ROOT,
        )
