"""
runner.py
=========

Entry point for generating all reporting artifacts (tables + figures)
for one or more feature sets.

Usage
-----
Run from the project root::

    PYTHONPATH=. python -m library.ml_cross_sectional.results_collector.runner

Or call programmatically::

    from library.ml_cross_sectional.results_collector.runner import run_report
    run_report("rbd_alone")
    run_report("rbd_prs")
"""
from __future__ import annotations

import sys
from pathlib import Path

# Support both `python -m library.ml_cross_sectional.results_collector.runner`
# (relative imports work) and `python runner.py` (direct script execution,
# where relative imports fail). The try/except handles both cases.
try:
    from .collector import RESULTS_ROOT, load_all_models
    from .figures import plot_roc_and_cm
    from .tables import save_tables
    from .plot_utils import _palette
    from ..feature_sets import get_feature_set
except ImportError:
    _project_root = Path(__file__).resolve().parents[3]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from library.ml_cross_sectional.results_collector.collector import (
        RESULTS_ROOT, load_all_models,
    )
    from library.ml_cross_sectional.results_collector.figures import plot_roc_and_cm
    from library.ml_cross_sectional.results_collector.tables import save_tables
    from library.ml_cross_sectional.feature_sets import get_feature_set

#: Feature sets to report when running all.
#: This should be kept in sync with FEATURE_SETS in feature_sets.py.
DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "rbd_alone",
    "rbd_prodromal",
    "rbd_prs",
    "rbd_prs_prodromal",
    "rbd_trail_ratio",
)


def run_report(
    feature_set: str,
    results_root: Path = RESULTS_ROOT,
    timestamp: str | None = None,
) -> Path:
    """
    Generate all reporting artifacts for a single feature set.

    Parameters
    ----------
    feature_set :
        Feature set name (e.g., "rbd_alone", "rbd_prs", etc.).
    results_root :
        Root of the ML results tree.
    timestamp :
        If provided, select runs matching this exact timestamp.
        If ``None`` (default), use the latest run per model.

    Returns
    -------
    Path
        The output directory where tables and figures were saved.
    """
    print(f"\n{'='*60}")
    print(f"  Reporting: feature_set={feature_set}")
    print(f"{'='*60}")

    runs = load_all_models(feature_set, results_root, timestamp=timestamp)
    print(f"  Loaded {len(runs)} models: {[r.model_name for r in runs]}")

    out_dir = results_root / feature_set / "_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tables.
    saved_tables = save_tables(runs, out_dir)
    for name, path in saved_tables.items():
        print(f"  [table] {name} -> {path}")

    # Figure: ROC + confusion matrices (grid layout, 2 CMs per row by default).
    fig_path = out_dir / "figure_roc_cm.png"
    fs_config = get_feature_set(feature_set)
    fs_label = fs_config.get("label", feature_set)
    plot_roc_and_cm(
        runs=runs,
        out_path=fig_path,
        font_scale=1.5,
        dpi=300,
        title="ROC + Confusion Matrices",
        feature_set_label=fs_label,
    )
    print(f"  [figure] roc_cm -> {fig_path}")

    print(f"  Done. Output -> {out_dir}")
    return out_dir


def run_all_reports(
    results_root: Path = RESULTS_ROOT,
    timestamp: str | None = None,
    feature_sets: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """
    Generate reports for all feature sets.

    Parameters
    ----------
    results_root :
        Root of the ML results tree.
    timestamp :
        Optional exact timestamp filter.
    feature_sets :
        Feature sets to report. If None, uses DEFAULT_FEATURE_SETS.

    Returns
    -------
    dict
        ``{feature_set: output_dir}``
    """
    if feature_sets is None:
        feature_sets = DEFAULT_FEATURE_SETS

    out: dict[str, Path] = {}
    for fs in feature_sets:
        fs_dir = results_root / fs
        if not fs_dir.is_dir():
            print(f"  Skipping {fs} (directory not found: {fs_dir})")
            continue
        out[fs] = run_report(fs, results_root, timestamp=timestamp)
    return out


def run_final_report(
    results_root: Path = RESULTS_ROOT,
    feature_sets: tuple[str, ...] | None = None,
    selection_metric: str = "auc_roc",
    bar_metrics: tuple[str, ...] = ("accuracy", "ppv", "f1"),
    cm_max_per_row: int = 2,
    report_timestamp: str | None = None,
    include_pr_curve: bool = False,
) -> Path:
    """
    Generate final report: cross-feature-set comparison figure, supplemental figure, and summary table.

    Loads latest model runs for each model type (across all feature sets).
    Report is assigned a unique run ID for traceability.

    Parameters
    ----------
    results_root :
        Root of the ML results tree.
    feature_sets :
        Feature sets to compare. If None, uses DEFAULT_FEATURE_SETS.
    selection_metric :
        Metric used to select best model: ``"auc_roc"`` (default),
        ``"auc_pr"``, or ``"youden"``.
    bar_metrics :
        Metrics shown in the bar chart. Default: accuracy, ppv (precision), f1.
    cm_max_per_row :
        Maximum number of confusion matrices per row (default: 2).
    report_timestamp :
        Optional explicit run ID for report directory name. If None, auto-generates.
    include_pr_curve :
        Include precision-recall curves in main figure (default: False).

    Returns
    -------
    Path
        Output directory where the figures and table were saved.
    """
    if feature_sets is None:
        feature_sets = DEFAULT_FEATURE_SETS

    from .final_figure import (
        plot_feature_set_comparison,
        plot_supplemental_figure,
        make_best_model_summary_table,
        make_all_models_summary_table,
        _load_best_models,
    )
    from ..pipeline import generate_run_id

    print(f"\n{'='*60}")
    print(f"  Final Report: Cross-Feature-Set Comparison")
    print(f"{'='*60}")

    # Load best models once (shared across all outputs)
    best_models = _load_best_models(results_root, list(feature_sets), selection_metric)

    # Extract run IDs from loaded models to identify common timestamp
    if best_models:
        run_ids = []
        for fs, (best_run, _) in best_models.items():
            run_dir_name = best_run.run_dir.name  # e.g., "xgboost_20260418_121649_k9m5p"
            parts = run_dir_name.rsplit("_", 1)
            if len(parts) == 2:
                # Check if second part is a UUID suffix (5 alphanumeric chars)
                if len(parts[1]) == 5 and parts[1].isalnum():
                    timestamp = "_".join(parts)
                else:
                    timestamp = run_dir_name
            else:
                timestamp = run_dir_name
            run_ids.append(timestamp)

        # Use the most recent run ID (sorted) if available, otherwise generate new one
        if run_ids:
            common_timestamp = sorted(set(run_ids))[-1]  # Most recent unique timestamp
        else:
            common_timestamp = generate_run_id()
    else:
        common_timestamp = generate_run_id()

    # Use provided timestamp or the common one from models
    out_timestamp = report_timestamp or common_timestamp

    # Create timestamped output directory
    out_dir = results_root / f"_final_report_{out_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not best_models:
        print(f"  No valid feature sets found (tried {feature_sets})")
        return out_dir

    colors = _palette(list(best_models.keys()))

    # Show which models are being used
    print(f"  Loaded {len(best_models)} best models:")
    for fs, (best_run, fs_label) in best_models.items():
        print(f"    - {fs_label}: {best_run.model_name} ({best_run.run_dir.name})")

    # Main figure: ROC + PR + confusion matrices
    fig_path = out_dir / "figure_feature_set_comparison.png"
    plot_feature_set_comparison(
        results_root=results_root,
        feature_sets=feature_sets,
        out_path=fig_path,
        selection_metric=selection_metric,
        bar_metrics=bar_metrics,
        cm_max_per_row=cm_max_per_row,
        font_scale=1.5,
        dpi=300,
        title="Cross-Feature-Set Comparison — Best Model per Configuration",
        best_models=best_models,
        include_pr_curve=include_pr_curve,
    )
    print(f"  [figure] feature_set_comparison -> {fig_path}")

    # Supplemental figure: metric bars, SHAP, calibration, cohort
    supp_path = out_dir / "figure_feature_set_supplemental.png"
    plot_supplemental_figure(
        best_models=best_models,
        colors=colors,
        out_path=supp_path,
        font_scale=1.3,
        dpi=300,
        title="Supplemental: Feature-Set Comparison Panels",
    )
    print(f"  [figure] supplemental -> {supp_path}")

    # Summary tables
    table_df = make_best_model_summary_table(best_models)
    table_path = out_dir / "table_best_model_summary.csv"
    table_df.to_csv(table_path, index=False)
    print(f"  [table] best_model_summary -> {table_path}")

    all_models_df = make_all_models_summary_table(results_root=results_root, feature_sets=feature_sets)
    all_models_path = out_dir / "table_all_models_summary.csv"
    all_models_df.to_csv(all_models_path, index=False)
    print(f"  [table] all_models_summary -> {all_models_path}")

    # Data transparency tables
    from .data_report import write_data_transparency_report
    write_data_transparency_report(best_models=best_models, out_dir=out_dir)

    print(f"  Done. Output -> {out_dir}")
    print(f"  Run ID: {out_timestamp}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path

    # When run directly (python runner.py), relative imports fail because
    # the file is not inside a package context. Fix by inserting the project
    # root (4 levels up from this file) into sys.path and re-importing via
    # absolute names.
    _project_root = _Path(__file__).resolve().parents[3]
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))

    from library.ml_cross_sectional.results_collector.collector import (  # noqa: E402
        RESULTS_ROOT as _RESULTS_ROOT,
        load_all_models as _load_all_models,
    )
    from library.ml_cross_sectional.results_collector.figures import (  # noqa: E402
        plot_roc_and_cm as _plot_roc_and_cm,
    )
    from library.ml_cross_sectional.results_collector.tables import (  # noqa: E402
        save_tables as _save_tables,
    )
    from library.ml_cross_sectional.feature_sets import get_feature_set as _get_feature_set

    _FEATURE_SETS = (
        "rbd_alone",
        "rbd_prodromal",
        "rbd_prs",
        "rbd_prs_prodromal",
        "rbd_trail_ratio",
    )

    for _fs in _FEATURE_SETS:
        _fs_dir = _RESULTS_ROOT / _fs
        if not _fs_dir.is_dir():
            print(f"  Skipping {_fs} (directory not found: {_fs_dir})")
            continue
        print(f"\n{'='*60}")
        print(f"  Reporting: feature_set={_fs}")
        print(f"{'='*60}")
        _runs = _load_all_models(_fs, _RESULTS_ROOT)
        print(f"  Loaded {len(_runs)} models: {[r.model_name for r in _runs]}")
        _out_dir = _RESULTS_ROOT / _fs / "_report"
        _out_dir.mkdir(parents=True, exist_ok=True)
        _saved = _save_tables(_runs, _out_dir)
        for _name, _path in _saved.items():
            print(f"  [table] {_name} -> {_path}")
        _fig_path = _out_dir / "figure_roc_cm.png"
        _fs_config = _get_feature_set(_fs)
        _fs_label = _fs_config.get("label", _fs)
        _plot_roc_and_cm(
            runs=_runs,
            out_path=_fig_path,
            font_scale=1.5,
            dpi=300,
            title="ROC + Confusion Matrices",
            feature_set_label=_fs_label,
        )
        print(f"  [figure] roc_cm -> {_fig_path}")
        print(f"  Done. Output -> {_out_dir}")
