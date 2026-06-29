"""
pipeline.py
===========

Orchestrator script for the ml_cross_sectional library. Runs the full
nested-CV + Optuna sweep over all registered models for each feature set and
writes results to ``results/ml_cross_sectional/<feature_set>/<model>_<timestamp>/``.

Usage
-----
This is a script, not a CLI. Edit the ``RUN_CONFIG`` block at the bottom and
execute with::

    C:/Users/riccig01/anaconda3/envs/stats_env/python.exe -m library.ml_cross_sectional.pipeline

For a smoke test on a small subset, see ``library.ml_cross_sectional.pipeline.run_smoke_test``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import uuid

import numpy as np
import pandas as pd

from .dataset import convert_to_cross_sectional
from .explainability import compute_permutation_importance, compute_shap
from .feature_sets import FEATURE_SETS
from .features import TIME_TO_EVENT_COL, ImputerPipeline, get_feature_matrix
from .metrics import compute_cohort_stats
from .models import ALL_MODELS
from .models.base import ModelBase
from .plots import plot_all_for_run
from .report.cross_fs_plots import generate_all_cross_fs_plots
from .storage import ResultsWriter
from .training import NestedCVResult, NestedCVTrainer
from .training_p1 import P1CombinedTrainer, P1CombinedConfig

RESULTS_ROOT = Path("results/ml_cross_sectional")

#: Enable or disable imputation of missing values
IMPUTATION_ENABLED = False


def generate_run_id() -> str:
    """
    Generate unique run ID: YYYYMMDD_HHMMSS_XXXXX.

    Format: Date + time + 5-char hash.
    This allows multiple runs per day without collision.

    Returns: e.g., "20260420_150000_a3k7m"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = str(uuid.uuid4())[:5]
    return f"{timestamp}_{unique_suffix}"


def list_all_runs(results_root: Path = RESULTS_ROOT) -> dict[str, Path]:
    """
    List all timestamped final report runs.

    Returns
    -------
    dict[str, Path]
        Mapping of run ID to report directory.
        Sorted by date (most recent first).
    """
    runs = {}
    for report_dir in sorted(results_root.glob("_final_report_*"), reverse=True):
        run_id = report_dir.name.replace("_final_report_", "")
        runs[run_id] = report_dir
    return runs


@dataclass(frozen=True)
class ModelRunResult:
    """Bundles the on-disk run directory with the in-memory CV result."""

    run_dir: Path
    cv_result: NestedCVResult


# --- Configuration -----------------------------------------------------------

#: Time-to-event handling: only ``"exclude"`` (drop time_to_event_log feature).
TTE_MODE: str = "exclude"


@dataclass(frozen=True)
class RunConfig:
    """User-tunable knobs for one full pipeline invocation."""

    n_outer: int = 5
    n_inner: int = 3
    n_trials: int = 50
    random_state: int = 42
    file_name: str = "ehr_diag_pd_rbd_only_all"
    outcome_name: str = "pd_only"  # outcome name (see outcomes.py)
    skip_models: tuple[str, ...] = ()  # e.g. ("svm_rbf",)
    feature_set: str | None = None  # feature set name (see feature_sets.py)
    feature_sets: tuple[str, ...] = ()  # for run_all_feature_sets


# --- Loaders -----------------------------------------------------------------

def load_data(file_name: str) -> pd.DataFrame:
    """Load df_risk via the project's standard helper."""
    from library.risk.risk_helpers import get_clean_risk_data  # local import (heavy)
    _, df = get_clean_risk_data(file_name=file_name)
    df = df.drop_duplicates(subset=['eid'], keep='first')
    return df


def _apply_complete_case_analysis(
    X: pd.DataFrame,
    *series: pd.Series,
) -> tuple[pd.DataFrame | pd.Series, ...]:
    """
    Drop rows where any predictor value is NaN, keeping all series aligned.

    Used when IMPUTATION_ENABLED=False. Drops rows with missing values
    in X (complete case analysis), which is statistically valid for small
    fractions of missing data (<1%) and avoids silent NaN propagation into
    models that cannot handle missing values.

    Parameters
    ----------
    X :
        Feature matrix (n_samples × n_features).
    *series :
        Any number of pd.Series to keep row-aligned with X after dropping.

    Returns
    -------
    tuple of (X_clean, *series_clean) with consistent integer index.
    """
    nan_mask = X.isna().any(axis=1)
    n_dropped = int(nan_mask.sum())
    if n_dropped > 0:
        pct = n_dropped / len(X) * 100
        missing_cols = X.columns[X.isna().any()].tolist()
        print(
            f"[Complete case] Dropped {n_dropped} rows ({pct:.3f}%) with missing "
            f"predictor values (imputation disabled). Columns: {missing_cols}"
        )
    keep = ~nan_mask
    X_clean = X[keep].reset_index(drop=True)
    return (X_clean,) + tuple(s[keep].reset_index(drop=True) for s in series)


def build_xy(
    df: pd.DataFrame,
    outcome_name: str | None = None,
    feature_set: str | None = None,
    include_tte: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert df_risk → cross-sectional (X, y).

    Parameters
    ----------
    df :
        The input dataframe from get_clean_risk_data.
    outcome_name :
        Outcome name (e.g., "pd_only", "pd_ad", "ad_only").
        If None, uses default outcome (backward compatible).
    feature_set :
        Feature set name (e.g., "rbd_alone", "rbd_prs", etc.).
        If None, uses default KEEP_FEATURES (backward compatible).
    include_tte :
        Whether to append the engineered ``time_to_event_log`` column.
        Set to ``False`` when ``TTE_MODE == "exclude"`` so that cohort
        stats reflect the actual feature set used in training.
    """
    frame = convert_to_cross_sectional(
        df, outcome_name=outcome_name, feature_set=feature_set
    )
    X, y = get_feature_matrix(frame, include_tte=include_tte)
    if not IMPUTATION_ENABLED:
        X, y = _apply_complete_case_analysis(X, y)
    return X, y


def build_xy_p1(
    df: pd.DataFrame,
    outcome_name: str | None = None,
    feature_set: str | None = None,
    include_tte: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Build X, y_label, y_incident, y_prevalent, y_control for P1 Combined training.

    Returns
    -------
    X : Feature matrix
    y_label : Combined outcome (incident | prevalent)
    y_incident : Binary incident case indicator
    y_prevalent : Binary prevalent case indicator
    y_control : Boolean control indicator (valid matching pool)
    """
    frame = convert_to_cross_sectional(
        df, outcome_name=outcome_name, feature_set=feature_set
    )
    X, y_label = get_feature_matrix(frame, include_tte=include_tte)

    # Extract P1-specific columns from the frame (aligned with X before any drops)
    df_work = frame.df
    y_incident = pd.Series(0, index=X.index, name="incident", dtype=int)
    y_prevalent = pd.Series(0, index=X.index, name="prevalent", dtype=int)
    y_control = pd.Series(False, index=X.index, name="control", dtype=bool)

    if frame.incident_col and frame.incident_col in df_work.columns:
        y_incident = df_work[frame.incident_col].astype(int).reset_index(drop=True)
    if frame.prevalent_col and frame.prevalent_col in df_work.columns:
        y_prevalent = df_work[frame.prevalent_col].astype(int).reset_index(drop=True)
    if frame.control_col and frame.control_col in df_work.columns:
        y_control = df_work[frame.control_col].astype(bool).reset_index(drop=True)

    if not IMPUTATION_ENABLED:
        X, y_label, y_incident, y_prevalent, y_control = _apply_complete_case_analysis(
            X, y_label, y_incident, y_prevalent, y_control
        )

    return X, y_label, y_incident, y_prevalent, y_control


# --- Per-model runner --------------------------------------------------------

def _apply_tte_mode(X: pd.DataFrame, tte_mode: str) -> tuple[pd.DataFrame, str]:
    """
    Translate a high-level ``tte_mode`` into an ``(X, imputer_strategy)`` pair.

    * ``exclude``  -> drop ``time_to_event_log`` column entirely
    * ``jittered`` -> keep column, imputer uses ``jittered_q3_max``
    """
    if tte_mode == "exclude":
        if TIME_TO_EVENT_COL in X.columns:
            X = X.drop(columns=[TIME_TO_EVENT_COL])
        return X, "jittered_q3_max"  # strategy irrelevant (column dropped)
    if tte_mode == "jittered":
        return X, "jittered_q3_max"
    raise ValueError(f"Unknown tte_mode: {tte_mode!r}. Expected one of ('exclude', 'jittered').")


def run_model(
    model_cls: type[ModelBase],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: RunConfig,
    feature_set: str | None = None,
    results_root: Path = RESULTS_ROOT,
    run_id: str | None = None,
    y_incident: pd.Series | None = None,
    y_prevalent: pd.Series | None = None,
) -> ModelRunResult:
    """
    Run nested CV for a single model and persist results.

    Parameters
    ----------
    model_cls :
        Model class to train.
    X, y :
        Feature matrix and target vector.
    cfg :
        Run configuration.
    feature_set :
        Feature set name for organizing results. If None, uses cfg.feature_set.
    results_root :
        Root directory for results.

    Results land under ``results_root / <feature_set> / <model>_<timestamp>``.
    """
    # Use provided feature_set, fall back to cfg.feature_set for backward compatibility.
    fs = feature_set or cfg.feature_set or "default"
    X_use, imputer_strategy = _apply_tte_mode(X, TTE_MODE)
    print(f"\n=== Model: {model_cls.name} | feature_set={fs} ===", flush=True)

    trainer = NestedCVTrainer(
        model_cls=model_cls,
        n_outer=cfg.n_outer,
        n_inner=cfg.n_inner,
        n_trials=cfg.n_trials,
        random_state=cfg.random_state,
        tte_strategy=imputer_strategy,
        imputation_enabled=IMPUTATION_ENABLED,
    )
    result = trainer.fit(X_use, y)
    metrics = result.metrics_frame()
    print(metrics[["fold", "auc_pr", "auc_roc", "f1", "sensitivity", "specificity"]].to_string(index=False))
    mean = result.mean_metrics()
    print(f"\nMean PR-AUC: {mean.loc['auc_pr','mean']:.4f} +/- {mean.loc['auc_pr','sd']:.4f}")
    print(f"Mean ROC-AUC: {mean.loc['auc_roc','mean']:.4f} +/- {mean.loc['auc_roc','sd']:.4f}")

    # Refit on full data for SHAP/permutation explainers (one final model).
    final_imp = ImputerPipeline(
        tte_strategy=imputer_strategy, random_state=cfg.random_state,
        enabled=IMPUTATION_ENABLED,
    ).fit(X_use, y)
    X_full = final_imp.transform(X_use)
    final_model = model_cls(params=result.folds[-1].best_params, random_state=cfg.random_state)
    from .training import _compute_sample_weight
    sw = _compute_sample_weight(y) if final_model.supports_sample_weight else None
    final_model.fit(X_full, y, sample_weight=sw)

    # Use the last outer fold's val indices as the explainer hold-out.
    last = result.folds[-1]
    test_idx = last.val_indices
    train_mask = np.ones(len(X_full), dtype=bool)
    train_mask[test_idx] = False
    shap_result = compute_shap(final_model, X_full.iloc[train_mask], X_full.iloc[~train_mask])
    perm = compute_permutation_importance(final_model, X_full.iloc[~train_mask], y.iloc[~train_mask])

    cohort = compute_cohort_stats(X_use, y, y_incident=y_incident, y_prevalent=y_prevalent)
    fs_dir = results_root / fs
    writer = ResultsWriter(base_dir=fs_dir, model_name=model_cls.name, timestamp=run_id or "")
    run_dir = writer.save_all(result=result, cohort=cohort, shap_result=shap_result, perm=perm)

    # Figures: SHAP importance + CI, plus interaction scatters.
    figs = plot_all_for_run(shap_result, run_dir)
    if figs:
        print(f"  figures: {len(figs)} written to {run_dir / 'figures'}")

    print(f"Saved -> {run_dir}", flush=True)
    return ModelRunResult(run_dir=run_dir, cv_result=result)


# --- Top-level orchestrator --------------------------------------------------

def run_all(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: RunConfig,
    feature_set: str | None = None,
    models: Iterable[type[ModelBase]] = ALL_MODELS,
    results_root: Path = RESULTS_ROOT,
    run_id: str | None = None,
    y_incident: pd.Series | None = None,
    y_prevalent: pd.Series | None = None,
) -> dict[str, ModelRunResult]:
    """
    Run every model in ``models`` for a single feature set.

    Parameters
    ----------
    X, y :
        Feature matrix and target vector.
    cfg :
        Run configuration.
    feature_set :
        Feature set name. If None, uses cfg.feature_set.
    models :
        Model classes to train.
    results_root :
        Root directory for results.
    """
    cohort = compute_cohort_stats(X, y)
    print(f"\nCohort: n={cohort.n_subjects} | cases={cohort.n_cases} | "
          f"controls={cohort.n_controls} | prevalence={cohort.prevalence:.4%}")
    print(f"Imputation: {'DISABLED' if not IMPUTATION_ENABLED else 'ENABLED'}")
    print("\nFeature stats (including missingness):")
    print(cohort.feature_stats.to_string())

    # Print features with >0% missingness
    missing_features = cohort.feature_stats[cohort.feature_stats['missing_pct'] > 0.0]
    if not missing_features.empty:
        print("\nFeatures with missing values:")
        for feat, row in missing_features.iterrows():
            print(f"  {feat:40s} | missing: {row['missing_pct']:6.2f}%")
    else:
        print("\nNo missing values detected.")

    out: dict[str, ModelRunResult] = {}
    for cls in models:
        if cls.name in cfg.skip_models:
            print(f"\n--- Skipping {cls.name} ---")
            continue
        out[cls.name] = run_model(
            cls, X, y, cfg, feature_set=feature_set, results_root=results_root, run_id=run_id,
            y_incident=y_incident, y_prevalent=y_prevalent
        )

    fs_label = feature_set or cfg.feature_set or "default"
    print(f"\n=== Summary ({fs_label}) ===")
    for name, mr in out.items():
        try:
            mm = pd.read_csv(mr.run_dir / "mean_metrics.csv", index_col=0)
            print(f"{name:>15s}  PR-AUC={mm.loc['auc_pr','mean']:.4f}  "
                  f"ROC-AUC={mm.loc['auc_roc','mean']:.4f}  "
                  f"F1={mm.loc['f1','mean']:.4f}")
        except Exception as e:  # pragma: no cover
            print(f"{name:>15s}  (failed to read mean_metrics.csv: {e})")
    return out


def run_all_feature_sets(
    df: pd.DataFrame,
    cfg: RunConfig,
    feature_sets: Iterable[str] | None = None,
    models: Iterable[type[ModelBase]] = ALL_MODELS,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, dict[str, ModelRunResult]]:
    """
    Sweep every feature set and run the full model suite for each.

    Parameters
    ----------
    df :
        Raw subject-level DataFrame from ``load_data`` (de-duplicated).
        X is rebuilt per feature set so that cohort stats and training always
        reflect exactly the columns defined in ``feature_sets.py``.
    cfg :
        Run configuration.
    feature_sets :
        Feature set names to sweep. If None, uses cfg.feature_sets or all available.
    models :
        Model classes to train.
    results_root :
        Root directory for results.

    Returns
    -------
    dict[str, dict[str, ModelRunResult]]
        Nested dict: {feature_set: {model: ModelRunResult}}
    """
    if feature_sets is None:
        feature_sets = cfg.feature_sets or tuple(FEATURE_SETS.keys())

    # Include TTE only when TTE_MODE is not "exclude" so cohort stats mirror training.
    _include_tte = TTE_MODE != "exclude"

    results: dict[str, dict[str, ModelRunResult]] = {}
    for fs in feature_sets:
        if fs not in FEATURE_SETS:
            print(f"\n!!! Unknown feature set: {fs} (skipping)")
            continue
        print(f"\n########## FEATURE SET: {fs} ##########")
        # Build feature-set-specific X so stats and training match exactly.
        X_fs, y_fs = build_xy(
            df, outcome_name=cfg.outcome_name, feature_set=fs, include_tte=_include_tte
        )

        # Extract incident / prevalent series from the CrossSectionalFrame so
        # they are persisted in cohort_stats.json for transparency reporting.
        _frame = convert_to_cross_sectional(df, outcome_name=cfg.outcome_name, feature_set=fs)
        _df_work = _frame.data
        _y_incident: pd.Series | None = None
        _y_prevalent: pd.Series | None = None
        if _frame.incident_col and _frame.incident_col in _df_work.columns:
            _y_incident = _df_work[_frame.incident_col].astype(int).reset_index(drop=True)
        if _frame.prevalent_col and _frame.prevalent_col in _df_work.columns:
            _y_prevalent = _df_work[_frame.prevalent_col].astype(int).reset_index(drop=True)

        results[fs] = run_all(
            X_fs,
            y_fs,
            cfg,
            feature_set=fs,
            models=models,
            results_root=results_root,
            y_incident=_y_incident,
            y_prevalent=_y_prevalent,
        )

    # Cross-feature-set comparison table (text).
    print("\n########## CROSS-FEATURE-SET COMPARISON ##########")
    print(f"{'model':>15s}  " + "  ".join(f"{fs:>22s}" for fs in feature_sets))
    model_names = sorted({m for d in results.values() for m in d.keys()})
    for name in model_names:
        row = [f"{name:>15s}"]
        for fs in feature_sets:
            mr = results.get(fs, {}).get(name)
            if mr is None:
                row.append(f"{'—':>22s}")
                continue
            try:
                mm = pd.read_csv(mr.run_dir / "mean_metrics.csv", index_col=0)
                pr = mm.loc["auc_pr", "mean"]
                roc = mm.loc["auc_roc", "mean"]
                row.append(f"PR={pr:.3f} ROC={roc:.3f}")
            except Exception:
                row.append(f"{'err':>22s}")
        print("  ".join(row))

    # Generate cross-feature-set comparison figures (8 plots per model).
    print("\n########## CROSS-FEATURE-SET FIGURES ##########")
    cross_fs_dir = results_root / "cross_fs_comparison"
    run_id = generate_run_id()
    for model_name in model_names:
        fs_cv_results: dict[str, NestedCVResult] = {
            fs: d[model_name].cv_result
            for fs, d in results.items()
            if model_name in d
        }
        if len(fs_cv_results) < 2:
            print(f"  {model_name}: skipped (fewer than 2 feature sets available)")
            continue
        written = generate_all_cross_fs_plots(fs_cv_results, cross_fs_dir, model_name, run_id=run_id)
        print(f"  {model_name}: {len(written)} figures -> {cross_fs_dir / run_id}")

    # Final comparison figure: best model per feature set (ROC + CM).
    print("\n########## FINAL COMPARISON FIGURE ##########")
    from .results_collector.runner import run_final_report
    run_id = generate_run_id()
    final_dir = run_final_report(
        results_root=results_root,
        feature_sets=tuple(results.keys()),
        report_timestamp=run_id,
    )
    print(f"  -> {final_dir / 'figure_feature_set_comparison.png'}")
    print(f"  Run ID: {run_id}")

    return results


# --- P1 Combined training (incident vs prevalent handling) -------------------

def run_model_p1(
    model_cls: type[ModelBase],
    X: pd.DataFrame,
    y_label: pd.Series,
    y_incident: pd.Series,
    y_prevalent: pd.Series,
    y_control: pd.Series,
    cfg: RunConfig,
    feature_set: str | None = None,
    results_root: Path = RESULTS_ROOT,
    run_id: str | None = None,
) -> ModelRunResult:
    """
    Run P1 Combined nested CV for a single model.

    Parameters
    ----------
    model_cls :
        Model class to train.
    X, y_label, y_incident, y_prevalent, y_control :
        Feature matrix and outcome labels (from build_xy_p1).
    cfg :
        Run configuration.
    feature_set :
        Feature set name for organizing results.
    results_root :
        Root directory for results.
    """
    fs = feature_set or cfg.feature_set or "default"
    print(f"\n=== Model: {model_cls.name} | feature_set={fs} (P1 Combined) ===", flush=True)

    p1_cfg = P1CombinedConfig(
        n_outer=cfg.n_outer,
        n_inner=cfg.n_inner,
        n_trials=cfg.n_trials,
        random_state=cfg.random_state,
        outcome_name=cfg.outcome_name,
        feature_set=feature_set,
    )

    trainer = P1CombinedTrainer(model_cls=model_cls, cfg=p1_cfg)
    result = trainer.fit(X, y_label, y_incident, y_prevalent, y_control)
    metrics = result.metrics_frame()
    print(metrics[["fold", "auc_pr", "auc_roc", "f1", "sensitivity", "specificity"]].to_string(index=False))
    mean = result.mean_metrics()
    print(f"\nMean PR-AUC: {mean.loc['auc_pr','mean']:.4f} +/- {mean.loc['auc_pr','sd']:.4f}")
    print(f"Mean ROC-AUC: {mean.loc['auc_roc','mean']:.4f} +/- {mean.loc['auc_roc','sd']:.4f}")

    # Full-data refit for SHAP and permutation importance.
    # Train on all subjects (incident + prevalent + controls) using best params
    # from the last outer fold. Evaluate explainers on the last fold's
    # incident-only held-out set, consistent with P1 evaluation semantics.
    from .training import _compute_sample_weight
    final_imp = ImputerPipeline(
        tte_strategy="jittered_q3_max", random_state=cfg.random_state,
        enabled=IMPUTATION_ENABLED,
    ).fit(X, y_label)
    X_full = final_imp.transform(X)
    last = result.folds[-1]
    final_model = model_cls(params=last.best_params, random_state=cfg.random_state)
    sw = _compute_sample_weight(y_label) if final_model.supports_sample_weight else None
    final_model.fit(X_full, y_label, sample_weight=sw)

    test_idx = last.val_indices
    train_mask = np.ones(len(X_full), dtype=bool)
    train_mask[test_idx] = False
    shap_result = compute_shap(final_model, X_full.iloc[train_mask], X_full.iloc[~train_mask])
    perm = compute_permutation_importance(
        final_model, X_full.iloc[~train_mask], y_label.iloc[~train_mask]
    )

    cohort = compute_cohort_stats(X, y_label)
    fs_dir = results_root / fs
    writer = ResultsWriter(base_dir=fs_dir, model_name=model_cls.name, timestamp=run_id or "")
    run_dir = writer.save_all(result=result, cohort=cohort, shap_result=shap_result, perm=perm)

    figs = plot_all_for_run(shap_result, run_dir)
    if figs:
        print(f"  figures: {len(figs)} written to {run_dir / 'figures'}")

    print(f"Saved -> {run_dir}", flush=True)
    return ModelRunResult(run_dir=run_dir, cv_result=result)


def run_all_p1(
    X: pd.DataFrame,
    y_label: pd.Series,
    y_incident: pd.Series,
    y_prevalent: pd.Series,
    y_control: pd.Series,
    cfg: RunConfig,
    feature_set: str | None = None,
    models: Iterable[type[ModelBase]] = ALL_MODELS,
    results_root: Path = RESULTS_ROOT,
    run_id: str | None = None,
) -> dict[str, ModelRunResult]:
    """
    Run every model for a single feature set using P1 Combined paradigm.

    Parameters
    ----------
    X, y_label, y_incident, y_prevalent, y_control :
        Feature matrix and outcome labels.
    cfg :
        Run configuration.
    feature_set :
        Feature set name.
    models :
        Model classes to train.
    results_root :
        Root directory for results.
    """
    cohort = compute_cohort_stats(X, y_label)
    print(f"\nCohort: n={cohort.n_subjects} | cases={cohort.n_cases} | "
          f"controls={cohort.n_controls} | prevalence={cohort.prevalence:.4%}")
    print(f"Imputation: {'DISABLED' if not IMPUTATION_ENABLED else 'ENABLED'}")
    print("\nFeature stats (including missingness):")
    print(cohort.feature_stats.to_string())

    # Print features with >0% missingness
    missing_features = cohort.feature_stats[cohort.feature_stats['missing_pct'] > 0.0]
    if not missing_features.empty:
        print("\nFeatures with missing values:")
        for feat, row in missing_features.iterrows():
            print(f"  {feat:40s} | missing: {row['missing_pct']:6.2f}%")
    else:
        print("\nNo missing values detected.")

    out: dict[str, ModelRunResult] = {}
    for cls in models:
        if cls.name in cfg.skip_models:
            print(f"\n--- Skipping {cls.name} ---")
            continue
        out[cls.name] = run_model_p1(
            cls, X, y_label, y_incident, y_prevalent, y_control,
            cfg, feature_set=feature_set, results_root=results_root, run_id=run_id
        )

    fs_label = feature_set or cfg.feature_set or "default"
    print(f"\n=== Summary ({fs_label}, P1 Combined) ===")
    for name, mr in out.items():
        try:
            mm = pd.read_csv(mr.run_dir / "mean_metrics.csv", index_col=0)
            print(f"{name:>15s}  PR-AUC={mm.loc['auc_pr','mean']:.4f}  "
                  f"ROC-AUC={mm.loc['auc_roc','mean']:.4f}  "
                  f"F1={mm.loc['f1','mean']:.4f}")
        except Exception as e:  # pragma: no cover
            print(f"{name:>15s}  (failed to read mean_metrics.csv: {e})")
    return out


def run_all_feature_sets_p1(
    df: pd.DataFrame,
    cfg: RunConfig,
    feature_sets: Iterable[str] | None = None,
    models: Iterable[type[ModelBase]] = ALL_MODELS,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, dict[str, ModelRunResult]]:
    """
    Sweep every feature set using P1 Combined paradigm.

    Parameters
    ----------
    df :
        Raw subject-level DataFrame from ``load_data``.
    cfg :
        Run configuration.
    feature_sets :
        Feature set names to sweep. If None, uses all available.
    models :
        Model classes to train.
    results_root :
        Root directory for results.

    Returns
    -------
    dict[str, dict[str, ModelRunResult]]
        Nested dict: {feature_set: {model: ModelRunResult}}
    """
    if feature_sets is None:
        feature_sets = cfg.feature_sets or tuple(FEATURE_SETS.keys())

    _include_tte = TTE_MODE != "exclude"

    results: dict[str, dict[str, ModelRunResult]] = {}
    for fs in feature_sets:
        if fs not in FEATURE_SETS:
            print(f"\n!!! Unknown feature set: {fs} (skipping)")
            continue
        print(f"\n########## FEATURE SET: {fs} (P1 Combined) ##########")
        # Build feature-set-specific X, y for P1
        X_fs, y_label, y_inc, y_prev, y_ctrl = build_xy_p1(
            df, outcome_name=cfg.outcome_name, feature_set=fs, include_tte=_include_tte
        )
        results[fs] = run_all_p1(
            X_fs, y_label, y_inc, y_prev, y_ctrl,
            cfg,
            feature_set=fs,
            models=models,
            results_root=results_root,
        )

    return results


# --- Smoke test --------------------------------------------------------------

def subsample_balanced(
    X: pd.DataFrame,
    y: pd.Series,
    n_cases: int,
    n_controls: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Randomly subsample ``n_cases`` positives + ``n_controls`` negatives.

    Used for smoke-test runs. ``n_controls`` are drawn uniformly at random
    from the negative pool (not stratified by any covariate).
    """
    rng = np.random.default_rng(random_state)
    pos_pool = np.where(y.values == 1)[0]
    neg_pool = np.where(y.values == 0)[0]
    if len(pos_pool) < n_cases:
        raise ValueError(f"Not enough positives: have {len(pos_pool)}, want {n_cases}")
    if len(neg_pool) < n_controls:
        raise ValueError(f"Not enough negatives: have {len(neg_pool)}, want {n_controls}")
    pos_idx = rng.choice(pos_pool, size=n_cases, replace=False)
    neg_idx = rng.choice(neg_pool, size=n_controls, replace=False)
    keep = np.sort(np.concatenate([pos_idx, neg_idx]))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def run_smoke_test(
    n_cases: int = 200,
    n_controls: int = 200,
    cfg: RunConfig | None = None,
    feature_sets: Iterable[str] | None = None,
    use_parallel: bool = False,
    n_workers: int = 2,
    selection_metric: str = "auc_roc",
) -> dict[str, dict[str, Path]]:
    """
    Run the full pipeline on a small random subset, sweeping every feature set.

    Defaults to 2 outer x 2 inner x 5 trials to keep runtime manageable.

    Parameters
    ----------
    n_cases, n_controls :
        Subset sizes for case/control stratification.
    cfg :
        Run configuration. Defaults to 2x2x5 if not provided.
    feature_sets :
        Feature sets to sweep. If None, uses all available.
    use_parallel :
        Whether to use parallel execution (ProcessPoolExecutor).
    n_workers :
        Number of worker processes (default: 2).
    selection_metric :
        Metric for selecting best model per feature set (default: "auc_roc", options: "auc_pr", "youden").
    """
    if cfg is None:
        cfg = RunConfig(n_outer=2, n_inner=2, n_trials=5)
    df = load_data(cfg.file_name)

    # Build a base X (no feature_set) solely to identify balanced subset row indices.
    # TTE is excluded here because it's not needed for index selection.
    # The actual per-feature-set X is rebuilt inside run_all_feature_sets.
    X_base, y_base = build_xy(df, outcome_name=cfg.outcome_name, include_tte=False)
    X_sub, y_sub = subsample_balanced(
        X_base, y_base, n_cases=n_cases, n_controls=n_controls,
        random_state=cfg.random_state,
    )
    # Map the reset-index positions back to the original df rows.
    # subsample_balanced preserves relative order, so iloc is safe after sort.
    rng = np.random.default_rng(cfg.random_state)
    pos_pool = np.where(y_base.values == 1)[0]
    neg_pool = np.where(y_base.values == 0)[0]
    pos_sel = rng.choice(pos_pool, size=min(n_cases, len(pos_pool)), replace=False)
    neg_sel = rng.choice(neg_pool, size=min(n_controls, len(neg_pool)), replace=False)
    keep = np.sort(np.concatenate([pos_sel, neg_sel]))
    df_sub = df.iloc[keep].reset_index(drop=True)

    print(f"Smoke test subset: n={len(df_sub)} | cases={int(y_sub.sum())} | "
          f"controls={int((y_sub == 0).sum())}")

    if use_parallel:
        from .parallel_pipeline import run_all_feature_sets_parallel
        return run_all_feature_sets_parallel(df_sub, cfg, feature_sets=feature_sets, n_workers=n_workers, selection_metric=selection_metric)
    else:
        return run_all_feature_sets(df_sub, cfg, feature_sets=feature_sets)


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    cfg = RunConfig()
    df = load_data(cfg.file_name)
    run_all_feature_sets(df, cfg)
