"""
collector.py
============

Discover and load persisted ML run artifacts from disk.

Each model run lives under::

    results/ml_cross_sectional/{feature_set}/{model}_{YYYYMMDD_HHMMSS}/

This module provides helpers to:

* discover the latest run directory per model for a given feature set,
* load standard CSV / JSON artifacts into pandas / dict structures.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

RESULTS_ROOT = Path("results/ml_cross_sectional")

#: Regex to parse ``<model>_<YYYYMMDD_HHMMSS>`` or ``<model>_<YYYYMMDD_HHMMSS_UUID>`` directory names.
_RUN_DIR_RE = re.compile(r"^(.+?)_(\d{8}_\d{6})(?:_([a-zA-Z0-9]{5}))?$")

#: Canonical display order for models.
MODEL_ORDER: tuple[str, ...] = (
    "logistic",
    "elasticnet",
    "xgboost",
    "random_forest",
    "svm_rbf",
)

#: Display names for models (for publication tables / figures).
MODEL_DISPLAY: dict[str, str] = {
    "logistic": "Logistic Regression",
    "elasticnet": "Elastic Net",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM (RBF)",
}


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def _parse_run_dir(name: str) -> tuple[str, str] | None:
    """Return ``(model_name, timestamp_str)`` or ``None`` if unparseable."""
    m = _RUN_DIR_RE.match(name)
    if m is None:
        return None
    return m.group(1), m.group(2)


def find_latest_runs(
    feature_set: str,
    results_root: Path = RESULTS_ROOT,
    timestamp: str | None = None,
) -> dict[str, Path]:
    """
    Discover model run directories for one feature set.

    Parameters
    ----------
    feature_set :
        Feature set name (e.g., "rbd_alone", "rbd_prs", etc.).
    results_root :
        Root of the ML results tree.
    timestamp :
        If provided, select runs matching this exact timestamp
        (format ``YYYYMMDD_HHMMSS``).  If ``None``, pick the latest
        timestamp per model.

    Returns
    -------
    dict
        ``{model_name: run_dir_path}`` for each discovered model.
    """
    fs_dir = results_root / feature_set
    if not fs_dir.is_dir():
        raise FileNotFoundError(f"Feature set directory not found: {fs_dir}")

    # Collect all (model, timestamp, path) triples.
    entries: list[tuple[str, str, Path]] = []
    for child in sorted(fs_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_run_dir(child.name)
        if parsed is None:
            continue
        entries.append((parsed[0], parsed[1], child))

    if timestamp is not None:
        # Exact match on timestamp.
        return {
            model: path
            for model, ts, path in entries
            if ts == timestamp
        }

    # Latest per model: entries are sorted lexicographically, so last wins.
    latest: dict[str, Path] = {}
    for model, _ts, path in entries:
        latest[model] = path  # later timestamps overwrite earlier
    return latest


# ---------------------------------------------------------------------------
# Individual artifact loaders
# ---------------------------------------------------------------------------

def load_mean_metrics(run_dir: Path) -> pd.DataFrame:
    """Load ``mean_metrics.csv`` (index = metric name, cols = mean, sd)."""
    return pd.read_csv(run_dir / "mean_metrics.csv", index_col=0)


def load_metrics_per_fold(run_dir: Path) -> pd.DataFrame:
    """Load ``metrics_per_fold.csv`` (one row per outer fold)."""
    return pd.read_csv(run_dir / "metrics_per_fold.csv")


def load_shap_summary(run_dir: Path) -> pd.DataFrame:
    """Load ``shap_summary.csv``; returns empty DataFrame if absent (e.g. P1 paradigm)."""
    p = run_dir / "shap_summary.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_permutation_importance(run_dir: Path) -> pd.DataFrame:
    """Load ``permutation_importance.csv``; returns empty DataFrame if absent."""
    p = run_dir / "permutation_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_confusion_matrices(run_dir: Path) -> dict[str, dict]:
    """Load ``confusion_matrices.json`` -> ``{fold_0: {tp, fp, tn, fn, threshold}, ...}``."""
    with open(run_dir / "confusion_matrices.json") as f:
        return json.load(f)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load ``predictions_per_fold.csv`` (cols: fold, row_index, y_true, y_pred_proba)."""
    return pd.read_csv(run_dir / "predictions_per_fold.csv")


def load_cohort_stats(run_dir: Path) -> dict:
    """Load ``cohort_stats.json``."""
    with open(run_dir / "cohort_stats.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Aggregated loader
# ---------------------------------------------------------------------------

@dataclass
class ModelRunData:
    """All artifacts from a single model run."""

    model_name: str
    run_dir: Path
    mean_metrics: pd.DataFrame = field(repr=False)
    metrics_per_fold: pd.DataFrame = field(repr=False)
    shap_summary: pd.DataFrame = field(repr=False)
    permutation_importance: pd.DataFrame = field(repr=False)
    confusion_matrices: dict[str, dict] = field(repr=False)
    predictions: pd.DataFrame = field(repr=False)
    cohort_stats: dict = field(repr=False)


def select_best_model(
    runs: list[ModelRunData],
    metric: str = "auc_roc",
) -> ModelRunData:
    """
    Select the best model from a list of runs by a scalar metric.

    Parameters
    ----------
    runs :
        List of ModelRunData, one per model.
    metric :
        Scoring metric used for selection. Options:

        - ``"auc_roc"`` (default) — mean ROC-AUC across outer folds.
        - ``"auc_pr"``            — mean PR-AUC across outer folds.
        - ``"youden"``            — Youden index = mean(sensitivity + specificity − 1).

        All values are read from ``ModelRunData.mean_metrics``
        (index=metric name, column="mean").

    Returns
    -------
    ModelRunData
        The run with the highest metric value.

    Raises
    ------
    ValueError
        If the metric is not found or runs list is empty.
    """
    if not runs:
        raise ValueError("runs list is empty")

    def _score(run: ModelRunData) -> float:
        mm = run.mean_metrics
        if metric == "youden":
            return mm.loc["sensitivity", "mean"] + mm.loc["specificity", "mean"] - 1.0
        if metric not in mm.index:
            raise ValueError(f"Metric {metric!r} not found in mean_metrics")
        return mm.loc[metric, "mean"]

    return max(runs, key=_score)


def load_all_models(
    feature_set: str,
    results_root: Path = RESULTS_ROOT,
    timestamp: str | None = None,
    model_order: Sequence[str] = MODEL_ORDER,
) -> list[ModelRunData]:
    """
    Load artifacts for all discovered models in one feature set.

    Parameters
    ----------
    feature_set :
        Feature set name (e.g., "rbd_alone", "rbd_prs", etc.).
    results_root :
        Root results directory.
    timestamp :
        Optional exact timestamp filter.
    model_order :
        Desired ordering of models in the returned list.
        Models not found on disk are silently skipped.

    Returns
    -------
    list[ModelRunData]
        Sorted according to ``model_order``.
    """
    runs = find_latest_runs(feature_set, results_root, timestamp=timestamp)
    if not runs:
        raise FileNotFoundError(
            f"No model runs found for feature_set={feature_set!r} "
            f"(root={results_root}, timestamp={timestamp!r})"
        )

    loaded: dict[str, ModelRunData] = {}
    for model_name, run_dir in runs.items():
        loaded[model_name] = ModelRunData(
            model_name=model_name,
            run_dir=run_dir,
            mean_metrics=load_mean_metrics(run_dir),
            metrics_per_fold=load_metrics_per_fold(run_dir),
            shap_summary=load_shap_summary(run_dir),
            permutation_importance=load_permutation_importance(run_dir),
            confusion_matrices=load_confusion_matrices(run_dir),
            predictions=load_predictions(run_dir),
            cohort_stats=load_cohort_stats(run_dir),
        )

    # Sort by model_order, appending any extras at the end.
    ordered: list[ModelRunData] = []
    for name in model_order:
        if name in loaded:
            ordered.append(loaded.pop(name))
    for remaining in sorted(loaded.keys()):
        ordered.append(loaded[remaining])
    return ordered
