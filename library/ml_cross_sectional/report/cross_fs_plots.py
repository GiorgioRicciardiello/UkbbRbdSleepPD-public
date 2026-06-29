"""
cross_fs_plots.py
=================

Eight cross-feature-set comparison figures, analogous to the paradigm-comparison
figures in ``src/screening/plot_results.py`` but with "feature_set" on the x-axis
instead of "paradigm".

Input
-----
``dict[str, NestedCVResult]`` — keyed by feature-set name, one result per set.
Each NestedCVResult carries ``folds: list[FoldResult]`` with per-fold metrics
and cohort composition counts.

Output
------
One PNG per figure, saved to the caller-supplied ``out_dir``.

Figures
-------
1. Metric box plots       — per-metric, feature sets on x-axis, folds = data points
2. Metric trajectories    — metric per fold per feature set (stability view)
3. Heatmap ROC-AUC        — feature_set × fold colour grid
4. Heatmap PR-AUC         — feature_set × fold colour grid
5. Summary bars           — mean ± 95% CI per metric per feature set
6. Calibration (Brier)    — dot plot per feature set ± CI, reference at 0
7. Training composition   — stacked bar: mean n_cases + n_controls per feature set
8. Summary table          — matplotlib table: feature_set × metric "mean ± SD"

All figures are saved at 300 dpi.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from ..training import NestedCVResult

logger = logging.getLogger(__name__)

# Metrics to include in comparison figures.
COMPARE_METRICS: tuple[str, ...] = ("auc_roc", "auc_pr", "f1", "sensitivity", "specificity")
METRIC_LABELS: dict[str, str] = {
    "auc_roc": "ROC-AUC",
    "auc_pr": "PR-AUC",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "brier": "Brier Score",
}

PALETTE: list[str] = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#956CB4",
                       "#8C613C", "#DC7EC0", "#797979"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_long_df(results: dict[str, "NestedCVResult"]) -> pd.DataFrame:
    """Flatten results to a long-format DataFrame with one row per fold × feature_set."""
    rows = []
    for fs_name, res in results.items():
        for fr in res.folds:
            row = {"feature_set": fs_name, "fold": fr.fold}
            row.update(fr.metrics.to_dict())
            row["n_train_cases"] = fr.n_train_cases
            row["n_train_controls"] = fr.n_train_controls
            rows.append(row)
    return pd.DataFrame(rows)


def _mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lower_CI, upper_CI) using Student's t."""
    n = len(values)
    m = float(np.mean(values))
    if n < 2:
        return m, m, m
    se = float(stats.sem(values, ddof=1))
    t = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m, m - t * se, m + t * se


# ---------------------------------------------------------------------------
# Figure 1: Metric box plots
# ---------------------------------------------------------------------------

def plot_metric_boxplots(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> Path:
    """4-panel box plots: one metric per panel, feature sets on x-axis."""
    metrics = [m for m in COMPARE_METRICS if m in long_df.columns]
    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    fs_order = sorted(long_df["feature_set"].unique())
    colors = {fs: PALETTE[i % len(PALETTE)] for i, fs in enumerate(fs_order)}

    for ax, metric in zip(axes, metrics):
        data_by_fs = [long_df.loc[long_df["feature_set"] == fs, metric].dropna().values
                      for fs in fs_order]
        bps = ax.boxplot(data_by_fs, patch_artist=True, widths=0.5, notch=False)
        for patch, fs in zip(bps["boxes"], fs_order):
            patch.set_facecolor(colors[fs])
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(fs_order) + 1))
        ax.set_xticklabels(fs_order, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"Metric Distribution by Feature Set — {model_name}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{model_name}_fig1_metric_boxplots.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Metric trajectories (per fold)
# ---------------------------------------------------------------------------

def plot_metric_trajectories(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> Path:
    """Line plot: metric value per fold, one line per feature set."""
    metrics = [m for m in COMPARE_METRICS[:4] if m in long_df.columns]
    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    fs_order = sorted(long_df["feature_set"].unique())

    for ax, metric in zip(axes, metrics):
        for i, fs in enumerate(fs_order):
            sub = long_df[long_df["feature_set"] == fs].sort_values("fold")
            ax.plot(sub["fold"], sub[metric], marker="o", label=fs,
                    color=PALETTE[i % len(PALETTE)], linewidth=1.5)
        ax.set_xlabel("Outer fold")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)

    fig.suptitle(f"Metric Trajectories by Feature Set — {model_name}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{model_name}_fig2_trajectories.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3a/3b: Heatmaps
# ---------------------------------------------------------------------------

def _plot_heatmap(
    long_df: pd.DataFrame,
    metric: str,
    out_dir: Path,
    model_name: str,
    suffix: str,
) -> Path:
    """Feature_set × fold heatmap for a single metric."""
    fs_order = sorted(long_df["feature_set"].unique())
    folds = sorted(long_df["fold"].unique())

    mat = np.full((len(fs_order), len(folds)), fill_value=float("nan"))
    for i, fs in enumerate(fs_order):
        for j, fold in enumerate(folds):
            vals = long_df.loc[(long_df["feature_set"] == fs) & (long_df["fold"] == fold), metric]
            if len(vals):
                mat[i, j] = float(vals.iloc[0])

    fig, ax = plt.subplots(figsize=(max(5, len(folds) * 0.9), max(3, len(fs_order) * 0.8)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_yticks(range(len(fs_order)))
    ax.set_yticklabels(fs_order)
    ax.set_xlabel("Outer fold")
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {model_name}", fontsize=10)

    # Annotate cells.
    for i in range(len(fs_order)):
        for j in range(len(folds)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    out_path = out_dir / f"{model_name}_{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_heatmaps(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> list[Path]:
    """Generate ROC-AUC and PR-AUC heatmaps."""
    paths = []
    for metric, suffix in [("auc_roc", "fig3a_heatmap_roc"), ("auc_pr", "fig3b_heatmap_pr")]:
        if metric in long_df.columns:
            paths.append(_plot_heatmap(long_df, metric, out_dir, model_name, suffix))
    return paths


# ---------------------------------------------------------------------------
# Figure 4: Summary bar chart (mean ± 95% CI)
# ---------------------------------------------------------------------------

def plot_summary_bars(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> Path:
    """Grouped bar chart: mean ± 95% CI per metric per feature set."""
    metrics = [m for m in COMPARE_METRICS if m in long_df.columns]
    fs_order = sorted(long_df["feature_set"].unique())
    n_fs = len(fs_order)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_fs)
    for ax, metric in zip(axes, metrics):
        means, lows, highs = [], [], []
        for fs in fs_order:
            vals = long_df.loc[long_df["feature_set"] == fs, metric].dropna().values
            m, lo, hi = _mean_ci(vals)
            means.append(m)
            lows.append(m - lo)
            highs.append(hi - m)
        bars = ax.bar(x, means, yerr=[lows, highs], capsize=4, color=PALETTE[:n_fs], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fs_order, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Mean ± 95% CI by Feature Set — {model_name}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{model_name}_fig4_summary_bars.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 5: Calibration (Brier score)
# ---------------------------------------------------------------------------

def plot_calibration(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> Path:
    """Horizontal dot plot: Brier score per feature set (lower is better)."""
    if "brier" not in long_df.columns:
        logger.warning("Brier score column not found; skipping calibration plot.")
        return out_dir / f"{model_name}_fig5_calibration_MISSING.png"

    fs_order = sorted(long_df["feature_set"].unique())
    fig, ax = plt.subplots(figsize=(7, max(3, len(fs_order) * 0.7)))

    y_pos = np.arange(len(fs_order))
    for i, fs in enumerate(fs_order):
        vals = long_df.loc[long_df["feature_set"] == fs, "brier"].dropna().values
        m, lo, hi = _mean_ci(vals)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o",
                    color=PALETTE[i % len(PALETTE)], capsize=4, markersize=6)
        # Jittered fold dots.
        ax.scatter(vals, np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)),
                   color=PALETTE[i % len(PALETTE)], alpha=0.4, s=20)

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fs_order, fontsize=9)
    ax.set_xlabel("Brier Score (lower = better)")
    ax.set_title(f"Calibration (Brier) by Feature Set — {model_name}", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / f"{model_name}_fig5_calibration.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 6: Training composition
# ---------------------------------------------------------------------------

def plot_training_composition(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> Path:
    """Stacked bar: mean n_train_cases + n_train_controls per feature set."""
    fs_order = sorted(long_df["feature_set"].unique())
    means_cases, means_ctrl = [], []
    for fs in fs_order:
        sub = long_df[long_df["feature_set"] == fs]
        means_cases.append(float(sub["n_train_cases"].mean()))
        means_ctrl.append(float(sub["n_train_controls"].mean()))

    x = np.arange(len(fs_order))
    fig, ax = plt.subplots(figsize=(max(5, len(fs_order) * 1.2), 5))
    ax.bar(x, means_cases, label="Cases", color="#4878D0", alpha=0.85)
    ax.bar(x, means_ctrl, bottom=means_cases, label="Controls", color="#EE854A", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(fs_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean n per training fold")
    ax.set_title(f"Training Composition by Feature Set — {model_name}", fontsize=10)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / f"{model_name}_fig6_training_composition.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 7: ROC vs PR scatter
# ---------------------------------------------------------------------------

def plot_roc_vs_pr_scatter(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> Path:
    """Mean ROC-AUC vs mean PR-AUC per feature set with ± 95% CI error bars."""
    fs_order = sorted(long_df["feature_set"].unique())
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, fs in enumerate(fs_order):
        sub = long_df[long_df["feature_set"] == fs]
        roc_m, roc_lo, roc_hi = _mean_ci(sub["auc_roc"].dropna().values)
        pr_m, pr_lo, pr_hi = _mean_ci(sub["auc_pr"].dropna().values)
        ax.errorbar(
            roc_m, pr_m,
            xerr=[[roc_m - roc_lo], [roc_hi - roc_m]],
            yerr=[[pr_m - pr_lo], [pr_hi - pr_m]],
            fmt="o", label=fs, color=PALETTE[i % len(PALETTE)],
            capsize=4, markersize=8,
        )

    ax.set_xlabel("Mean ROC-AUC")
    ax.set_ylabel("Mean PR-AUC")
    ax.set_title(f"ROC-AUC vs PR-AUC by Feature Set — {model_name}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / f"{model_name}_fig7_roc_vs_pr.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 8: Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(
    long_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
) -> Path:
    """Matplotlib table: feature_set × metric as 'mean ± SD'."""
    metrics = [m for m in COMPARE_METRICS if m in long_df.columns]
    fs_order = sorted(long_df["feature_set"].unique())

    cell_text = []
    for fs in fs_order:
        row = []
        for metric in metrics:
            vals = long_df.loc[long_df["feature_set"] == fs, metric].dropna().values
            if len(vals):
                row.append(f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}")
            else:
                row.append("—")
        cell_text.append(row)

    col_labels = [METRIC_LABELS.get(m, m) for m in metrics]
    fig_h = max(2, len(fs_order) * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 2), fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=fs_order,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    ax.set_title(f"Summary Table — {model_name}", fontsize=11, pad=12)
    fig.tight_layout()
    out_path = out_dir / f"{model_name}_fig8_summary_table.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_cross_fs_plots(
    results: dict[str, "NestedCVResult"],
    out_dir: Path,
    model_name: str,
    run_id: str | None = None,
) -> list[Path]:
    """
    Generate all 8 cross-feature-set comparison figures.

    Parameters
    ----------
    results :
        ``{feature_set_name: NestedCVResult}`` — one result per feature set,
        all from the same model type.
    out_dir :
        Base directory for output. If run_id is provided, creates subdirectory
        with timestamp (e.g., ``cross_fs_comparison/20260420_150000_a3k7m/``).
    model_name :
        Model name for filename prefixes.
    run_id :
        Optional run ID for timestamped output. If None, uses out_dir directly.
        Format: YYYYMMDD_HHMMSS_XXXXX
        Used as filename prefix and in figure titles.

    Returns
    -------
    list[Path]
        Paths to the written PNG files.
    """
    out_dir = Path(out_dir)

    # Create timestamped subdirectory if run_id is provided
    if run_id:
        out_dir = out_dir / run_id

    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = _build_long_df(results)
    paths: list[Path] = []

    try:
        paths.append(plot_metric_boxplots(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 1 failed: %s", exc)

    try:
        paths.append(plot_metric_trajectories(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 2 failed: %s", exc)

    try:
        paths.extend(plot_heatmaps(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 3 failed: %s", exc)

    try:
        paths.append(plot_summary_bars(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 4 failed: %s", exc)

    try:
        paths.append(plot_calibration(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 5 failed: %s", exc)

    try:
        paths.append(plot_training_composition(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 6 failed: %s", exc)

    try:
        paths.append(plot_roc_vs_pr_scatter(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 7 failed: %s", exc)

    try:
        paths.append(plot_summary_table(long_df, out_dir, model_name))
    except Exception as exc:
        logger.warning("Fig 8 failed: %s", exc)

    logger.info("Cross-FS plots written to %s (%d files)", out_dir, len(paths))
    return paths
