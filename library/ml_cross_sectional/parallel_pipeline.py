"""
parallel_pipeline.py
====================

Parallel feature-set execution for ML cross-sectional pipeline.

Spawns one worker process per feature set. Each worker runs all models
sequentially (with internal joblib parallelism for CV folds). Optimized
for 100k-row datasets where pickling is cheap.

Usage
-----
Instead of::

    run_all_feature_sets(df, cfg)

Use::

    run_all_feature_sets_parallel(df, cfg, n_workers=2)

Expected: 3-4x speedup on 16-core machine.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable
    import pandas as pd
    from .pipeline import RunConfig, ModelRunResult


def _init_worker() -> None:
    """
    Worker process initializer: prevent BLAS thread oversubscription.

    Each worker will run CPU-intensive CV loops. Without this, BLAS (OpenMP)
    will spawn threads, competing with other workers for cores.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _run_fs_worker(
    fs: str,
    df: pd.DataFrame,
    cfg: RunConfig,
    outcome_name: str,
    results_root: Path,
    run_id: str,
) -> tuple[str, dict[str, ModelRunResult]]:
    """
    Worker: run all models for one feature set.

    Parameters
    ----------
    fs : Feature set name
    df : Pickled DataFrame (100k rows, ~20 MB, <2s to unpickle)
    cfg : RunConfig
    outcome_name : Outcome for this run
    results_root : Results directory

    Returns
    -------
    tuple[str, dict]
        (feature_set_name, {model_name: ModelRunResult})
    """
    from .pipeline import build_xy, run_all, TTE_MODE
    from .models import ALL_MODELS

    print(f"\n[Worker PID={os.getpid()}] Starting feature set: {fs}", flush=True)

    # Build feature-set-specific X, y — mirror serial pipeline's TTE logic
    _include_tte = TTE_MODE != "exclude"
    X_fs, y_fs = build_xy(
        df,
        outcome_name=outcome_name,
        feature_set=fs,
        include_tte=_include_tte,
    )

    # Run all models for this feature set (models run sequentially in this worker)
    results_fs = run_all(
        X_fs,
        y_fs,
        cfg,
        feature_set=fs,
        models=ALL_MODELS,
        results_root=results_root,
        run_id=run_id,
    )

    print(f"[Worker PID={os.getpid()}] Complete: {fs}", flush=True)
    return fs, results_fs


def run_all_feature_sets_parallel(
    df: pd.DataFrame,
    cfg: RunConfig,
    feature_sets: Iterable[str] | None = None,
    models: "Iterable[type[ModelBase]]" | None = None,
    results_root: Path | None = None,
    n_workers: int = 2,
    selection_metric: str = "auc_roc",
) -> dict[str, dict[str, ModelRunResult]]:
    """
    Parallel feature-set sweep using process pool.

    Spawns n_workers worker processes, each running all models for one
    feature set. For 100k-row datasets, pickling is fast (<2s/worker).

    Parameters
    ----------
    df : DataFrame from load_data() (~100k rows)
    cfg : RunConfig
    feature_sets : Feature set names (default: all from cfg.feature_sets or FEATURE_SETS.keys())
    models : Model classes (default: ALL_MODELS, respects cfg.skip_models)
    results_root : Results directory (default: RESULTS_ROOT)
    n_workers : Number of worker processes (2-3 recommended for 100k rows on 16-core machine)
    selection_metric : Metric for selecting best model per feature set (default: "auc_roc", options: "auc_pr", "youden")

    Returns
    -------
    dict[str, dict[str, ModelRunResult]]
        Nested dict: {feature_set: {model_name: ModelRunResult}}

    Notes
    -----
    - Each worker receives a pickled copy of df (~20 MB, ~1-2s overhead).
    - Models run sequentially within each worker (no nested process pools).
    - Expected speedup: 3-4x on typical hardware.
    """
    from .feature_sets import FEATURE_SETS
    from .pipeline import RESULTS_ROOT as DEFAULT_RESULTS_ROOT
    from .models import ALL_MODELS

    if feature_sets is None:
        feature_sets = cfg.feature_sets or tuple(FEATURE_SETS.keys())

    if models is None:
        models = ALL_MODELS

    if results_root is None:
        results_root = DEFAULT_RESULTS_ROOT

    # Validate feature sets
    feature_sets = list(feature_sets)
    invalid = [fs for fs in feature_sets if fs not in FEATURE_SETS]
    if invalid:
        print(f"\n!!! Unknown feature sets (skipping): {invalid}")
        feature_sets = [fs for fs in feature_sets if fs in FEATURE_SETS]

    if not feature_sets:
        print("\n!!! No valid feature sets to run.")
        return {}

    results: dict[str, dict[str, ModelRunResult]] = {}

    from .pipeline import generate_run_id
    run_id = generate_run_id()

    print(f"\n{'='*60}")
    print(f"  Parallel Mode: {len(feature_sets)} feature sets, {n_workers} workers")
    print(f"  Run ID: {run_id}")
    print(f"{'='*60}")

    # Create pool and submit tasks
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        # Map futures to feature set names for tracking
        future_to_fs = {
            executor.submit(
                _run_fs_worker,
                fs,
                df,
                cfg,
                cfg.outcome_name,
                results_root,
                run_id,
            ): fs
            for fs in feature_sets
        }

        # Collect results as workers complete
        for future in as_completed(future_to_fs):
            fs = future_to_fs[future]
            try:
                fs_name, models_dict = future.result()
                results[fs_name] = models_dict
                print(f"\n[OK] [{fs}] All models complete")
            except Exception as e:
                print(f"\n[ERROR] [{fs}] FAILED: {e}")
                raise

    print(f"\n{'='*60}")
    print(f"  Parallel execution complete")
    print(f"{'='*60}")

    # Generate cross-feature-set comparison figures and final report
    print(f"\n########## CROSS-FEATURE-SET FIGURES ##########")
    from .report.cross_fs_plots import generate_all_cross_fs_plots
    from .results_collector.runner import run_final_report
    from .training import NestedCVResult

    # Reuse the run_id generated before the pool — all feature sets share it.
    cross_fs_dir = results_root / "cross_fs_comparison"

    # Extract model results for cross-feature-set plots
    for model_name in set(m for models_dict in results.values() for m in models_dict.keys()):
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

    # Generate final comparison report
    print(f"\n########## FINAL COMPARISON FIGURE ##########")
    from datetime import datetime

    final_dir = run_final_report(
        results_root=results_root,
        feature_sets=tuple(results.keys()),
        selection_metric=selection_metric,
        report_timestamp=run_id,
    )

    # Extract and display date/time from run ID
    try:
        date_str = run_id[:8]  # YYYYMMDD
        time_str = run_id[9:15]  # HHMMSS
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_formatted = date_obj.strftime("%Y-%m-%d")
        time_formatted = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
    except Exception:
        date_formatted = "unknown"
        time_formatted = "unknown"

    print(f"\n{'='*60}")
    print(f"  FINAL REPORT GENERATED")
    print(f"  Date: {date_formatted}")
    print(f"  Time: {time_formatted}")
    print(f"  Run ID: {run_id}")
    print(f"  Selection Metric: {selection_metric}")
    print(f"  Output: {final_dir}")
    print(f"{'='*60}")

    return results


def _run_fs_worker_p1(
    fs: str,
    df: pd.DataFrame,
    cfg: RunConfig,
    outcome_name: str,
    results_root: Path,
    run_id: str,
) -> tuple[str, dict[str, ModelRunResult]]:
    """
    Worker: run all models for one feature set using P1 Combined paradigm.

    Parameters
    ----------
    fs : Feature set name
    df : Pickled DataFrame (100k rows, ~20 MB, <2s to unpickle)
    cfg : RunConfig
    outcome_name : Outcome for this run
    results_root : Results directory

    Returns
    -------
    tuple[str, dict]
        (feature_set_name, {model_name: ModelRunResult})
    """
    from .pipeline import build_xy_p1, run_all_p1, TTE_MODE
    from .models import ALL_MODELS

    print(f"\n[Worker PID={os.getpid()}] Starting feature set (P1): {fs}", flush=True)

    # Build feature-set-specific X, y — mirror serial pipeline's TTE logic
    _include_tte = TTE_MODE != "exclude"
    X_fs, y_label, y_incident, y_prevalent, y_control = build_xy_p1(
        df,
        outcome_name=outcome_name,
        feature_set=fs,
        include_tte=_include_tte,
    )

    # Run all models for this feature set using P1 paradigm
    results_fs = run_all_p1(
        X_fs,
        y_label,
        y_incident,
        y_prevalent,
        y_control,
        cfg,
        feature_set=fs,
        models=ALL_MODELS,
        results_root=results_root,
        run_id=run_id,
    )

    print(f"[Worker PID={os.getpid()}] Complete (P1): {fs}", flush=True)
    return fs, results_fs


def run_all_feature_sets_p1_parallel(
    df: pd.DataFrame,
    cfg: RunConfig,
    feature_sets: Iterable[str] | None = None,
    models: "Iterable[type[ModelBase]]" | None = None,
    results_root: Path | None = None,
    n_workers: int = 2,
    selection_metric: str = "auc_roc",
) -> dict[str, dict[str, ModelRunResult]]:
    """
    Parallel feature-set sweep using P1 Combined paradigm with process pool.

    Spawns n_workers worker processes, each running all models for one
    feature set using the P1 Combined training approach (incident+prevalent
    training, incident-only evaluation, per-fold control matching).

    Parameters
    ----------
    df : DataFrame from load_data() (~100k rows)
    cfg : RunConfig
    feature_sets : Feature set names (default: all from cfg.feature_sets or FEATURE_SETS.keys())
    models : Model classes (default: ALL_MODELS, respects cfg.skip_models)
    results_root : Results directory (default: RESULTS_ROOT)
    n_workers : Number of worker processes (2-3 recommended for 100k rows on 16-core machine)
    selection_metric : Metric for selecting best model per feature set (default: "auc_roc", options: "auc_pr", "youden")

    Returns
    -------
    dict[str, dict[str, ModelRunResult]]
        Nested dict: {feature_set: {model_name: ModelRunResult}}

    Notes
    -----
    - Each worker receives a pickled copy of df (~20 MB, ~1-2s overhead).
    - Models run sequentially within each worker (no nested process pools).
    - Expected speedup: 3-4x on typical hardware.
    - Uses P1 Combined paradigm: stratifies on incident cases, trains on
      incident+prevalent, evaluates on incident only.
    """
    from .feature_sets import FEATURE_SETS
    from .pipeline import RESULTS_ROOT as DEFAULT_RESULTS_ROOT
    from .models import ALL_MODELS

    if feature_sets is None:
        feature_sets = cfg.feature_sets or tuple(FEATURE_SETS.keys())

    if models is None:
        models = ALL_MODELS

    if results_root is None:
        results_root = DEFAULT_RESULTS_ROOT

    # Validate feature sets
    feature_sets = list(feature_sets)
    invalid = [fs for fs in feature_sets if fs not in FEATURE_SETS]
    if invalid:
        print(f"\n!!! Unknown feature sets (skipping): {invalid}")
        feature_sets = [fs for fs in feature_sets if fs in FEATURE_SETS]

    if not feature_sets:
        print("\n!!! No valid feature sets to run.")
        return {}

    results: dict[str, dict[str, ModelRunResult]] = {}

    from .pipeline import generate_run_id
    run_id = generate_run_id()

    print(f"\n{'='*60}")
    print(f"  Parallel P1 Combined Mode: {len(feature_sets)} feature sets, {n_workers} workers")
    print(f"  Run ID: {run_id}")
    print(f"{'='*60}")

    # Create pool and submit tasks
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        # Map futures to feature set names for tracking
        future_to_fs = {
            executor.submit(
                _run_fs_worker_p1,
                fs,
                df,
                cfg,
                cfg.outcome_name,
                results_root,
                run_id,
            ): fs
            for fs in feature_sets
        }

        # Collect results as workers complete
        for future in as_completed(future_to_fs):
            fs = future_to_fs[future]
            try:
                fs_name, models_dict = future.result()
                results[fs_name] = models_dict
                print(f"\n[OK] [{fs}] All models complete (P1)")
            except Exception as e:
                print(f"\n[ERROR] [{fs}] FAILED: {e}")
                raise

    print(f"\n{'='*60}")
    print(f"  Parallel P1 execution complete")
    print(f"{'='*60}")

    # Generate final comparison report (cross-fs plots skipped: P1 uses
    # incident-only evaluation metrics which are not comparable via the
    # standard cross_fs_plots that assume shared NestedCVResult structure).
    print(f"\n########## FINAL COMPARISON FIGURE (P1) ##########")
    from datetime import datetime

    from .results_collector.runner import run_final_report

    # Reuse the run_id generated before the pool — all feature sets share it.
    final_dir = run_final_report(
        results_root=results_root,
        feature_sets=tuple(results.keys()),
        selection_metric=selection_metric,
        report_timestamp=run_id,
    )

    try:
        date_str = run_id[:8]
        time_str = run_id[9:15]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        date_formatted = date_obj.strftime("%Y-%m-%d")
        time_formatted = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
    except Exception:
        date_formatted = "unknown"
        time_formatted = "unknown"

    print(f"\n{'='*60}")
    print(f"  FINAL REPORT GENERATED")
    print(f"  Date: {date_formatted}")
    print(f"  Time: {time_formatted}")
    print(f"  Run ID: {run_id}")
    print(f"  Selection Metric: {selection_metric}")
    print(f"  Output: {final_dir}")
    print(f"{'='*60}")

    return results
