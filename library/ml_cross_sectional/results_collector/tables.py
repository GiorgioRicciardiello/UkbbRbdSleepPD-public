"""
tables.py
=========

Publication-ready summary tables from ML cross-sectional results.

All tables are returned as ``pd.DataFrame`` and optionally saved to CSV.

Table 1 — **Model performance**: one row per model, metrics as columns,
           cells formatted as ``mean (sd)``. AUC metrics as raw proportions;
           sensitivity, specificity, accuracy, F1, PPV, NPV, Brier in %.
Table 2 — **Permutation importance**: single merged table, features as rows,
           one ``mean (sd)`` column per model.
Table 3 — **SHAP summary**: features as rows, models as columns,
           cells formatted as ``mean (sd)`` of mean |SHAP|.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .collector import MODEL_DISPLAY, ModelRunData

#: Metrics reported as raw proportions (not converted to %).
_RAW_METRICS: frozenset[str] = frozenset({"auc_roc", "auc_pr"})

#: Metrics shown in the performance table and their display labels.
DISPLAY_METRICS: tuple[tuple[str, str], ...] = (
    ("auc_roc", "AUC-ROC"),
    ("auc_pr", "AUC-PR"),
    ("f1", "F1 (%)"),
    ("accuracy", "Accuracy (%)"),
    ("sensitivity", "Sensitivity (%)"),
    ("specificity", "Specificity (%)"),
    ("ppv", "PPV (%)"),
    ("npv", "NPV (%)"),
    ("brier", "Brier (%)"),
)


def _fmt(mean: float, sd: float, decimals: int = 2) -> str:
    """Format a value as ``mean (sd)`` rounded to *decimals*."""
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"


def _fmt_pct(mean: float, sd: float, decimals: int = 2) -> str:
    """Format a proportion as percentage ``mean (sd)`` rounded to *decimals*."""
    return f"{mean * 100:.{decimals}f} ({sd * 100:.{decimals}f})"


# ---------------------------------------------------------------------------
# Table 1: model performance
# ---------------------------------------------------------------------------

def make_metrics_table(
    runs: Sequence[ModelRunData],
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Cross-model performance table.

    Parameters
    ----------
    runs :
        Loaded model run data (one per model).
    decimals :
        Rounding precision.

    Returns
    -------
    pd.DataFrame
        Index = model display name, columns = metric display labels,
        cells = ``"mean (sd)"`` strings. Percentage metrics scaled x100.
    """
    rows: list[dict[str, str]] = []
    index_labels: list[str] = []

    for run in runs:
        mm = run.mean_metrics
        row: dict[str, str] = {}
        for metric_key, metric_label in DISPLAY_METRICS:
            if metric_key not in mm.index:
                row[metric_label] = "—"
                continue
            m, s = mm.loc[metric_key, "mean"], mm.loc[metric_key, "sd"]
            if metric_key in _RAW_METRICS:
                row[metric_label] = _fmt(m, s, decimals)
            else:
                row[metric_label] = _fmt_pct(m, s, decimals)
        rows.append(row)
        index_labels.append(MODEL_DISPLAY.get(run.model_name, run.model_name))

    df = pd.DataFrame(rows, index=index_labels)
    df.index.name = "Model"
    return df


# ---------------------------------------------------------------------------
# Table 2: permutation importance (merged across models)
# ---------------------------------------------------------------------------

def make_permutation_table(
    runs: Sequence[ModelRunData],
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Merged permutation importance table across all models.

    Parameters
    ----------
    runs :
        Loaded model run data.
    decimals :
        Rounding precision.

    Returns
    -------
    pd.DataFrame
        Index = feature name (sorted by max importance descending),
        columns = model display names, cells = ``"mean (sd)"`` strings.
    """
    model_series: dict[str, pd.Series] = {}
    for run in runs:
        pi = run.permutation_importance
        if pi.empty:
            continue
        display = MODEL_DISPLAY.get(run.model_name, run.model_name)
        formatted = pi.set_index("feature").apply(
            lambda r: _fmt(r["importance_mean"], r["importance_std"], decimals),
            axis=1,
        )
        model_series[display] = formatted

    if not model_series:
        return pd.DataFrame()

    combined = pd.DataFrame(model_series)

    # Sort by max raw importance across models (descending).
    raw_means: dict[str, pd.Series] = {}
    for run in runs:
        display = MODEL_DISPLAY.get(run.model_name, run.model_name)
        raw_means[display] = run.permutation_importance.set_index("feature")[
            "importance_mean"
        ]
    raw_df = pd.DataFrame(raw_means)
    sort_key = raw_df.max(axis=1).reindex(combined.index)
    combined = combined.loc[sort_key.sort_values(ascending=False).index]

    combined.index.name = "Feature"
    return combined


# ---------------------------------------------------------------------------
# Table 3: SHAP summary (features x models) with std
# ---------------------------------------------------------------------------

def make_shap_table(
    runs: Sequence[ModelRunData],
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Cross-model SHAP summary with bootstrap std.

    Parameters
    ----------
    runs :
        Loaded model run data.
    decimals :
        Rounding precision.

    Returns
    -------
    pd.DataFrame
        Index = feature name (sorted by max mean |SHAP| descending),
        columns = model display names, cells = ``"mean (sd)"``.
    """
    model_formatted: dict[str, pd.Series] = {}
    model_raw_mean: dict[str, pd.Series] = {}

    for run in runs:
        shap_values_path = run.run_dir / "shap_values.npy"
        shap_x_path = run.run_dir / "shap_X_eval.csv"
        display = MODEL_DISPLAY.get(run.model_name, run.model_name)

        if shap_values_path.exists() and shap_x_path.exists():
            # Compute mean and std from raw SHAP values.
            sv = np.load(shap_values_path)
            x_eval = pd.read_csv(shap_x_path)
            features = list(x_eval.columns)
            abs_sv = np.abs(sv)
            means = abs_sv.mean(axis=0)
            stds = abs_sv.std(axis=0)
            formatted = pd.Series(
                [_fmt(m, s, decimals) for m, s in zip(means, stds)],
                index=features,
            )
            model_formatted[display] = formatted
            model_raw_mean[display] = pd.Series(means, index=features)
        else:
            # Fallback: use shap_summary.csv (mean only, no std).
            shap = run.shap_summary
            if shap.empty or "mean_abs_shap" not in shap.columns:
                continue
            s = shap.set_index("feature")["mean_abs_shap"]
            model_formatted[display] = s.map(
                lambda v: f"{v:.{decimals}f}" if pd.notna(v) else "—"
            )
            model_raw_mean[display] = s

    if not model_formatted:
        return pd.DataFrame()

    combined = pd.DataFrame(model_formatted)
    raw_df = pd.DataFrame(model_raw_mean)

    # Sort features by max mean |SHAP| across models (descending).
    sort_key = raw_df.max(axis=1).reindex(combined.index)
    combined = combined.loc[sort_key.sort_values(ascending=False).index]

    combined.index.name = "Feature"
    return combined


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_tables(
    runs: Sequence[ModelRunData],
    out_dir: Path,
    decimals: int = 2,
) -> dict[str, Path]:
    """
    Generate and save all tables to *out_dir*.

    Returns
    -------
    dict
        ``{table_name: csv_path}`` for each saved table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # Table 1: metrics.
    metrics_df = make_metrics_table(runs, decimals)
    p = out_dir / "table_model_metrics.csv"
    metrics_df.to_csv(p)
    saved["metrics"] = p

    # Table 2: permutation importance (single merged table).
    perm_df = make_permutation_table(runs, decimals)
    p = out_dir / "table_permutation_importance.csv"
    perm_df.to_csv(p)
    saved["permutation"] = p

    # Table 3: SHAP summary with std.
    shap_df = make_shap_table(runs, decimals)
    p = out_dir / "table_shap_summary.csv"
    shap_df.to_csv(p)
    saved["shap_summary"] = p

    return saved
