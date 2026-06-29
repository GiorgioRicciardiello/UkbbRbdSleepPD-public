"""
data_report.py
==============

Generate data-transparency tables for the cross-feature-set final report.

Produces four CSV files written to the report output directory:

* ``data_cohort_summary.csv``   — cohort size, case/control split, prevalence per feature set
* ``data_fold_distribution.csv``— per-fold n, case/control counts, class rate, threshold
* ``data_fold_metrics.csv``     — all performance metrics per fold (consolidated)
* ``data_feature_stats.csv``    — feature-level descriptive statistics per feature set

Note on prevalent / incident counts
------------------------------------
These are not persisted in the ML run artifacts (``cohort_stats.json`` stores
n_cases / n_controls only).  The ML pipeline operates on incident cases after
prevalent exclusion has already been applied upstream in the dataset-build step.
Prevalent / incident counts should be reported from the dataset-build output
(``data/pp/res_build_final_dataset/``), not from the ML artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .collector import ModelRunData, MODEL_DISPLAY


# ---------------------------------------------------------------------------
# 1. Cohort summary
# ---------------------------------------------------------------------------

def make_cohort_summary(
    best_models: dict[str, tuple[ModelRunData, str]],
) -> pd.DataFrame:
    """
    One row per feature set: total N, n_cases, n_controls, prevalence.

    Parameters
    ----------
    best_models :
        ``{fs: (best_run, fs_label)}``.

    Returns
    -------
    pd.DataFrame
        Columns: Feature Set, Model, N Total, N Cases, N Controls,
        Prevalence (%), Case:Control Ratio.
    """
    rows = []
    for fs, (best_run, fs_label) in best_models.items():
        cs = best_run.cohort_stats
        n_total = cs.get("n_subjects", cs.get("n_total", None))
        n_cases = cs.get("n_cases", cs.get("n_pos", None))
        n_controls = cs.get("n_controls", cs.get("n_neg", None))

        # Fall back to mean_metrics if cohort_stats is incomplete.
        mm = best_run.mean_metrics
        if n_cases is None:
            n_cases = int(round(mm.loc["n_pos", "mean"]))
        if n_total is None:
            n_total = int(round(mm.loc["n", "mean"]))
        if n_controls is None:
            n_controls = int(n_total) - int(n_cases)

        prevalence_pct = 100.0 * int(n_cases) / int(n_total) if int(n_total) > 0 else float("nan")
        ratio = f"1:{int(n_controls) // max(int(n_cases), 1)}"

        n_incident = cs.get("n_incident", None)
        n_prevalent = cs.get("n_prevalent", None)

        rows.append({
            "Feature Set": fs_label,
            "Model (best)": MODEL_DISPLAY.get(best_run.model_name, best_run.model_name),
            "N Total": int(n_total),
            "N Cases": int(n_cases),
            "N Incident": int(n_incident) if n_incident is not None else "N/A",
            "N Prevalent": int(n_prevalent) if n_prevalent is not None else "N/A",
            "N Controls": int(n_controls),
            "Prevalence (%)": round(prevalence_pct, 3),
            "Case:Control Ratio": ratio,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Per-fold distribution
# ---------------------------------------------------------------------------

def make_fold_distribution(
    best_models: dict[str, tuple[ModelRunData, str]],
) -> pd.DataFrame:
    """
    One row per (feature set, fold): n_total, n_cases, n_controls,
    case rate, threshold used.

    Derived from ``confusion_matrices`` (TP+FN = cases in fold,
    TN+FP = controls in fold) and ``metrics_per_fold``.

    Parameters
    ----------
    best_models :
        ``{fs: (best_run, fs_label)}``.

    Returns
    -------
    pd.DataFrame
        Columns: Feature Set, Fold, N Total, N Cases, N Controls,
        Case Rate (%), Threshold, AUC-ROC, Sensitivity, Specificity.
    """
    rows = []
    for fs, (best_run, fs_label) in best_models.items():
        cm_dict = best_run.confusion_matrices
        mpf = best_run.metrics_per_fold

        for fold_key, cm in cm_dict.items():
            fold_idx = int(fold_key.split("_")[1])

            tp = cm.get("tp", 0)
            fn = cm.get("fn", 0)
            fp = cm.get("fp", 0)
            tn = cm.get("tn", 0)
            threshold = cm.get("threshold", float("nan"))

            n_cases = tp + fn
            n_controls = fp + tn
            n_total = n_cases + n_controls
            case_rate = 100.0 * n_cases / max(n_total, 1)

            # Pull per-fold AUC/sensitivity/specificity from metrics_per_fold.
            fold_row = mpf[mpf["fold"] == fold_idx]
            auc_roc = fold_row["auc_roc"].iloc[0] if not fold_row.empty else float("nan")
            sensitivity = fold_row["sensitivity"].iloc[0] if not fold_row.empty else float("nan")
            specificity = fold_row["specificity"].iloc[0] if not fold_row.empty else float("nan")

            rows.append({
                "Feature Set": fs_label,
                "Fold": fold_idx,
                "N Total": n_total,
                "N Cases": n_cases,
                "N Controls": n_controls,
                "Case Rate (%)": round(case_rate, 3),
                "Threshold": round(threshold, 4),
                "AUC-ROC": round(auc_roc, 4),
                "Sensitivity": round(sensitivity, 4),
                "Specificity": round(specificity, 4),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["Feature Set", "Fold"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Consolidated metrics per fold
# ---------------------------------------------------------------------------

def make_fold_metrics(
    best_models: dict[str, tuple[ModelRunData, str]],
) -> pd.DataFrame:
    """
    All per-fold metrics for each feature set's best model, consolidated
    into one table.

    Parameters
    ----------
    best_models :
        ``{fs: (best_run, fs_label)}``.

    Returns
    -------
    pd.DataFrame
        Feature Set and Model prepended to each row of metrics_per_fold.
    """
    frames: list[pd.DataFrame] = []
    for fs, (best_run, fs_label) in best_models.items():
        mpf = best_run.metrics_per_fold.copy()
        mpf.insert(0, "Model (best)", MODEL_DISPLAY.get(best_run.model_name, best_run.model_name))
        mpf.insert(0, "Feature Set", fs_label)
        frames.append(mpf)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Feature descriptive statistics
# ---------------------------------------------------------------------------

def make_feature_stats(
    best_models: dict[str, tuple[ModelRunData, str]],
) -> pd.DataFrame:
    """
    Feature-level descriptive statistics (mean, SD, median, IQR, missing %)
    per feature set, loaded from ``feature_stats.csv`` in each run directory.

    Parameters
    ----------
    best_models :
        ``{fs: (best_run, fs_label)}``.

    Returns
    -------
    pd.DataFrame
        Columns: Feature Set, Feature, Mean, SD, Median, IQR, Missing (%).
    """
    frames: list[pd.DataFrame] = []
    for fs, (best_run, fs_label) in best_models.items():
        p = best_run.run_dir / "feature_stats.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(0, "Feature Set", fs_label)
        # Standardise column names regardless of how pipeline wrote them.
        df = df.rename(columns={
            "mean": "Mean",
            "sd": "SD",
            "median": "Median",
            "iqr": "IQR",
            "missing_pct": "Missing (%)",
            "feature": "Feature",
        })
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Top-level writer
# ---------------------------------------------------------------------------

def write_data_transparency_report(
    best_models: dict[str, tuple[ModelRunData, str]],
    out_dir: Path,
) -> None:
    """
    Write all data-transparency tables to ``out_dir``.

    Files written
    -------------
    * ``data_cohort_summary.csv``
    * ``data_fold_distribution.csv``
    * ``data_fold_metrics.csv``
    * ``data_feature_stats.csv``

    Parameters
    ----------
    best_models :
        ``{fs: (best_run, fs_label)}``.
    out_dir :
        Directory where CSV files are written (created if absent).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: list[tuple[str, pd.DataFrame]] = [
        ("data_cohort_summary.csv",    make_cohort_summary(best_models)),
        ("data_fold_distribution.csv", make_fold_distribution(best_models)),
        ("data_fold_metrics.csv",      make_fold_metrics(best_models)),
        ("data_feature_stats.csv",     make_feature_stats(best_models)),
    ]

    for fname, df in tables:
        path = out_dir / fname
        df.to_csv(path, index=False)
        n_rows, n_cols = df.shape
        print(f"  [data] {fname} — {n_rows} rows × {n_cols} cols -> {path}")
