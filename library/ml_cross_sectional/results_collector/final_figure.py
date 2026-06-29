"""
final_figure.py
===============

Cross-feature-set comparison figure showing the best model per feature set.

Generates a single publication-ready figure comparing feature sets by:
1. Selecting the best model for each feature set (by AUC-ROC, AUC-PR, or Youden).
2. Overlaying their ROC curves (one per feature set).
3. Arranging their confusion matrices as rows.
4. Displaying metrics (Accuracy, Precision, F1 by default) as a grouped bar chart.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .collector import MODEL_DISPLAY, load_all_models, select_best_model, ModelRunData
from .plot_utils import (
    AXES_BG,
    FIG_BG,
    _cumulative_confusion,
    _draw_cm,
    _mean_roc,
    _mean_pr,
    _palette,
    _threshold_point_on_curve,
    _wrap,
)
from .tables import _fmt, _fmt_pct
from library.ml_cross_sectional.feature_sets import FEATURE_SETS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("results/ml_cross_sectional")

#: Default feature sets to sweep (in order).
DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "rbd_alone",
    "rbd_prodromal",
    "rbd_prs",
    "rbd_prs_prodromal",
    "rbd_trail_ratio",
)

#: Short labels for feature sets (from FEATURE_SETS, abbreviated).
FS_SHORT_LABELS: dict[str, str] = {
    "rbd_alone":           "RBD alone",
    "rbd_prodromal":       "RBD + Prodromal",
    "rbd_prs":             "RBD + PRS",
    "rbd_prs_prodromal":   "RBD + PRS + Prodromal",
    # "rbd_trail_ratio":     "RBD + TMT",  # Commented out
}

#: Default metrics shown in the right-side bar chart.
DEFAULT_BAR_METRICS: tuple[str, ...] = ("accuracy", "ppv", "f1")

#: Display names for metrics.
METRIC_DISPLAY: dict[str, str] = {
    "accuracy": "Accuracy",
    "ppv": "Precision (PPV)",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
}

_DEFAULT_PALETTE = "Dark2"
FONT_SCALE = 1.3


# ---------------------------------------------------------------------------
# Helper: load and select best models
# ---------------------------------------------------------------------------

def _load_best_models(
    results_root: Path,
    feature_sets: list[str],
    selection_metric: str = "auc_roc",
) -> dict[str, tuple[ModelRunData, str]]:
    """
    Load all models for feature sets and select best per feature set.

    Loads the latest model run for each model (across all feature sets).
    Note: Latest runs may be from different timestamps due to staggered execution.

    Parameters
    ----------
    results_root :
        Root of the ML results tree.
    feature_sets :
        Feature set names to load.
    selection_metric :
        Metric for model selection (auc_roc, auc_pr, youden).

    Returns
    -------
    dict
        ``{fs: (best_run, fs_label)}`` for successfully loaded feature sets.
    """
    best_models: dict[str, tuple] = {}
    for fs in feature_sets:
        try:
            runs = load_all_models(fs, results_root)
            best = select_best_model(runs, metric=selection_metric)
            fs_label = FEATURE_SETS.get(fs, {}).get("label", FS_SHORT_LABELS.get(fs, fs))
            best_models[fs] = (best, fs_label)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load models for {fs}: {e}")
            continue
    return best_models


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_feature_set_comparison(
    results_root: Path = RESULTS_ROOT,
    feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS,
    out_path: Path | None = None,
    selection_metric: str = "auc_roc",
    bar_metrics: Sequence[str] = DEFAULT_BAR_METRICS,
    cm_max_per_row: int = 2,
    font_scale: float = FONT_SCALE,
    dpi: int = 300,
    palette: str = _DEFAULT_PALETTE,
    fig_bg: str = FIG_BG,
    axes_bg: str = AXES_BG,
    xlim: tuple[float, float] = (0.0, 0.8),
    ylim: tuple[float, float] = (0.2, 1.0),
    class_names: dict[int, str] | None = None,
    title: str = "Cross-Feature-Set Comparison — Best Model per Configuration",
    best_models: dict[str, tuple] | None = None,
    include_pr_curve: bool = True,
) -> plt.Figure:
    """
    Generate cross-feature-set comparison figure with ROC and optionally PR curves and confusion matrices.

    Selects the best model per feature set (by selection_metric), overlays
    their ROC and PR curves, and arranges CMs in a grid with metrics displayed in titles.

    Layout (N_cm_rows × (2 + N_cm_cols) cols):
    - Columns 0–1 (merged, span all rows): nested left panel
      * Top: ROC curves
      * Bottom: PR curves (optional, controlled by include_pr_curve)
    - Columns 2 to 2+N_cm_cols-1 (grid): Confusion matrices with metrics in subtitle
      * Metrics: Accuracy, Precision, F1 (rounded to 1 decimal)
      * Axis labels shown only on leftmost column (y-labels) and bottom row (x-labels)

    Parameters
    ----------
    results_root :
        Root of the ML results tree (used only if best_models is None).
    feature_sets :
        Feature set names to include (in order).
    out_path :
        Output PNG path. If None, figure is not saved.
    selection_metric :
        Metric used to select best model per feature set (used only if best_models is None).
        Options: ``"auc_roc"`` (default), ``"auc_pr"``, ``"youden"``.
    bar_metrics :
        Metrics to display in the right-side bar chart.
        Default: ``("accuracy", "ppv", "f1")``.
    cm_max_per_row :
        Maximum number of confusion matrices per row (default: 2).
    font_scale :
        Global font multiplier.
    dpi :
        Resolution.
    palette :
        Seaborn palette name for feature set colors.
    fig_bg, axes_bg :
        Figure and axes background colours.
    xlim, ylim :
        ROC axis limits.
    class_names :
        ``{0: "Neg", 1: "Pos"}`` labels for CM axes.
    title :
        Figure suptitle.
    best_models :
        Pre-loaded ``{fs: (best_run, fs_label)}`` dict. If None, loads from disk.
    include_pr_curve :
        If True (default), include PR curve panel below ROC. If False, show only ROC panel.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    class_names = class_names or {0: "Neg", 1: "Pos"}
    feature_sets = list(feature_sets)
    n_features = len(feature_sets)
    bar_metrics = list(bar_metrics)

    # Load and select best model for each feature set if not provided.
    if best_models is None:
        best_models = _load_best_models(results_root, feature_sets, selection_metric)

    if not best_models:
        raise ValueError(f"No valid feature sets found (tried {feature_sets})")

    # Build feature set colors.
    colors = _palette(list(best_models.keys()), name=palette)

    # Grid layout: n_cm_rows rows × (2 ROC+PR + n_cm_cols CM) cols.
    # CMs arranged in grid: ceil(n_feature_sets / cm_max_per_row) rows, up to cm_max_per_row per row.
    n_feature_sets = len(best_models)
    n_cm_cols = min(n_feature_sets, cm_max_per_row)
    n_cm_rows = math.ceil(n_feature_sets / cm_max_per_row)
    n_cols = 2 + n_cm_cols  # left panel (2 virtual cols) + CM cols
    # Left panel gets 1.8× the width of each CM col so ROC/PR curves are legible.
    width_ratios = [1.8, 1.8] + [1.0] * n_cm_cols
    figsize = (5.0 * n_cols, max(8.0, 4.0 * n_cm_rows))

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
    }

    with plt.rc_context(rc):
        # Pure GridSpec avoids the plt.subplots+ax.remove() pattern, which
        # leaves phantom row/col constraints that shrink the left panel.
        fig = plt.figure(figsize=figsize, facecolor=fig_bg)
        gs = matplotlib.gridspec.GridSpec(
            n_cm_rows, n_cols,
            figure=fig,
            width_ratios=width_ratios,
            hspace=0.45,
            wspace=0.38,
        )

        # ── Left panel: nested GridSpec spanning ALL rows, first 2 cols ──
        n_inner_rows = 2 if include_pr_curve else 1
        inner_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            n_inner_rows, 1,
            subplot_spec=gs[:, 0:2],
            height_ratios=[1.0, 0.75] if include_pr_curve else [1.0],
            hspace=0.38 if include_pr_curve else 0,
        )
        roc_ax = fig.add_subplot(inner_gs[0, 0])
        pr_ax = fig.add_subplot(inner_gs[1, 0]) if include_pr_curve else None
        roc_ax.set_facecolor(axes_bg)
        if pr_ax is not None:
            pr_ax.set_facecolor(axes_bg)

        # Pre-build all CM axes (used/unused) so hide logic works uniformly.
        cm_ax_map: dict[tuple[int, int], plt.Axes] = {}
        for _k in range(n_cm_rows * n_cm_cols):
            _r, _c = _k // n_cm_cols, 2 + (_k % n_cm_cols)
            _ax = fig.add_subplot(gs[_r, _c])
            _ax.set_facecolor(axes_bg)
            cm_ax_map[(_r, _c)] = _ax

        # ── ROC panel ──
        roc_ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

        roc_legend_lines = []
        for fs, (best_run, fs_label) in best_models.items():
            color = colors[fs]
            preds = best_run.predictions

            fpr_grid, mean_tpr, std_tpr, mean_auc, std_auc = _mean_roc(preds)

            # Legend label: bold feature set name, then model and metrics.
            # Spaces replaced with ~ so mathbf renders them correctly in mathtext.
            model_display = MODEL_DISPLAY.get(best_run.model_name, best_run.model_name)
            tau = best_run.mean_metrics.loc["threshold", "mean"]
            fs_bold = "$\\mathbf{" + fs_label.replace(" ", "~") + "}$"
            legend_label = f"{fs_bold} | {model_display}\n(AUC-ROC={mean_auc:.3f}, τ*={tau:.3f})"

            line, = roc_ax.plot(
                fpr_grid, mean_tpr,
                color=color, linewidth=2.5,
                label=legend_label,
            )
            roc_legend_lines.append(line)

            # Std band.
            roc_ax.fill_between(
                fpr_grid,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=color, alpha=0.10,
            )

            # Diamond marker at Youden threshold.
            fpr_t, tpr_t = _threshold_point_on_curve(preds, tau, fpr_grid, mean_tpr)
            roc_ax.scatter(
                fpr_t, tpr_t,
                s=80, color=color,
                edgecolor="black", linewidths=1.0,
                marker="D", zorder=5,
            )

        roc_ax.set_xlabel("False Positive Rate (1 - Specificity)")
        roc_ax.set_ylabel("True Positive Rate (Sensitivity)")
        roc_ax.set_title("ROC — Best Model per Feature Set")
        if xlim:
            roc_ax.set_xlim(*xlim)
        if ylim:
            roc_ax.set_ylim(*ylim)
        roc_ax.grid(alpha=0.3)
        leg = roc_ax.legend(loc="lower right", frameon=True)
        for legline in leg.get_lines():
            legline.set_linewidth(4.0)

        # ── PR panel (optional) ──
        if include_pr_curve and pr_ax is not None:
            for fs, (best_run, fs_label) in best_models.items():
                color = colors[fs]
                preds = best_run.predictions

                recall_grid, mean_prec, std_prec, mean_auc_pr, std_auc_pr = _mean_pr(preds)

                model_display = MODEL_DISPLAY.get(best_run.model_name, best_run.model_name)
                tau = best_run.mean_metrics.loc["threshold", "mean"]
                legend_label = f"{fs_label} | {model_display}\n(AUC-PR={mean_auc_pr:.3f})"

                pr_ax.plot(
                    recall_grid, mean_prec,
                    color=color, linewidth=2.5,
                    label=legend_label,
                )

                # Std band.
                pr_ax.fill_between(
                    recall_grid,
                    np.clip(mean_prec - std_prec, 0, 1),
                    np.clip(mean_prec + std_prec, 0, 1),
                    color=color, alpha=0.10,
                )

                # No-skill baseline: y = n_pos / n_total
                n_pos = best_run.mean_metrics.loc["n_pos", "mean"]
                n_total = best_run.mean_metrics.loc["n", "mean"]
                no_skill = n_pos / max(n_total, 1)
                pr_ax.axhline(no_skill, color=color, linestyle="--", linewidth=1.0, alpha=0.6)

                # Diamond marker at operating threshold (approximate via closest recall/precision on grid).
                # For PR curves, we compute the operating point at the threshold.
                try:
                    preds_copy = preds.copy()
                    preds_copy["y_pred"] = (preds_copy["y_pred_proba"] >= tau).astype(int)
                    from sklearn.metrics import precision_recall_curve as prc_sklearn
                    for _, chunk in preds_copy.groupby("fold"):
                        y_true = chunk["y_true"].to_numpy(dtype=int)
                        y_score = chunk["y_pred_proba"].to_numpy(dtype=float)
                        if len(np.unique(y_true)) >= 2:
                            prec, rec, thresholds = prc_sklearn(y_true, y_score)
                            # Find closest threshold to tau
                            idx = int(np.argmin(np.abs(thresholds - tau)))
                            # Snap to grid
                            rec_snap = rec[idx]
                            prec_snap = prec[idx]
                            # Find nearest grid point
                            grid_idx = int(np.argmin((recall_grid - rec_snap) ** 2))
                            pr_ax.scatter(
                                recall_grid[grid_idx], mean_prec[grid_idx],
                                s=80, color=color,
                                edgecolor="black", linewidths=1.0,
                                marker="D", zorder=5,
                            )
                            break
                except Exception:
                    pass

            pr_ax.set_xlabel("Recall (Sensitivity)")
            pr_ax.set_ylabel("Precision (PPV)")
            pr_ax.set_title("PR — Best Model per Feature Set")
            pr_ax.set_xlim(0, 1)
            pr_ax.set_ylim(0, 1.05)
            pr_ax.grid(alpha=0.3)
            pr_ax.legend(loc="upper right", frameon=True, fontsize=8)

        # ── Confusion matrices (grid layout with metrics in title) ────────
        for i, (fs, (best_run, fs_label)) in enumerate(best_models.items()):
            row_idx = i // cm_max_per_row
            col_idx = 2 + (i % cm_max_per_row)
            ax = cm_ax_map[(row_idx, col_idx)]

            color = colors[fs]
            model_display = MODEL_DISPLAY.get(best_run.model_name, best_run.model_name)

            # Extract metrics (rounded to 1 decimal).
            accuracy = best_run.mean_metrics.loc["accuracy", "mean"]
            precision = best_run.mean_metrics.loc["ppv", "mean"]
            f1 = best_run.mean_metrics.loc["f1", "mean"]

            # Convert predictions to binary using threshold first (need it for CM and n).
            preds = best_run.predictions.copy()
            threshold = best_run.mean_metrics.loc["threshold", "mean"]
            preds["y_pred"] = (preds["y_pred_proba"] >= threshold).astype(int)
            cm = _cumulative_confusion(preds)
            n_samples = int(cm.sum())

            # Title: model name and metrics (feature set shown in ROC legend)
            # Metrics on separate lines to reduce horizontal span
            cm_title = (
                f"{model_display}\n"
                f"n={n_samples} | Acc={accuracy:.2f}\n"
                f"Prec={precision:.2f} F1={f1:.2f}"
            )

            # Determine label visibility based on position.
            show_y_labels = (col_idx == 2)  # Leftmost CM column
            show_x_labels = (row_idx == n_cm_rows - 1)  # Bottom row

            _draw_cm(
                ax, cm,
                title=cm_title,
                class_names=class_names,
                base_color=color,
                fontsize=10 * font_scale,
                show_x_labels=show_x_labels,
                show_y_labels=show_y_labels,
            )

        # ── Hide unused CM cells in the last row ────────────────────────
        for j in range(n_feature_sets, n_cm_rows * n_cm_cols):
            row_idx = j // n_cm_cols
            col_idx = 2 + (j % n_cm_cols)
            cm_ax_map[(row_idx, col_idx)].axis("off")

        # ── Finalise ───────────────────────────────────────────────────
        # tight_layout is intentionally omitted: GridSpec hspace/wspace
        # already controls spacing; tight_layout can conflict with suptitle.
        fig.suptitle(title, fontsize=15 * font_scale, y=1.01)

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig_bg)

    return fig


# ---------------------------------------------------------------------------
# Supplemental figure (4-panel: metrics, SHAP, calibration, cohort)
# ---------------------------------------------------------------------------

def plot_supplemental_figure(
    best_models: dict[str, tuple[ModelRunData, str]],
    colors: dict[str, tuple] | None = None,
    out_path: Path | None = None,
    font_scale: float = FONT_SCALE,
    dpi: int = 300,
    fig_bg: str = FIG_BG,
    axes_bg: str = AXES_BG,
    top_n_shap: int = 5,
    title: str = "Supplemental: Feature-Set Comparison Panels",
) -> plt.Figure:
    """
    Generate supplemental figure: metric bars (top-left), Brier Score (bottom-left),
    and Top-5 SHAP importance spanning full right column (2 rows × 1 col).

    Layout (2 rows × 2 cols via GridSpec):
    - [0, 0]: Key metrics grouped bar chart
    - [1, 0]: Brier Score (mean ± SD) per feature set
    - [:, 1]: Top-5 SHAP feature importance per feature set (spans both rows)

    Parameters
    ----------
    best_models :
        Dict of ``{fs: (best_run, fs_label)}``.
    colors :
        Dict mapping fs to color tuples. If None, auto-generated.
    out_path :
        Output PNG path. If None, figure is not saved.
    font_scale :
        Global font multiplier.
    dpi :
        Resolution.
    fig_bg, axes_bg :
        Figure and axes background colours.
    top_n_shap :
        Number of top features to show per feature set in the SHAP panel.
    title :
        Figure suptitle.

    Returns
    -------
    plt.Figure
        The generated figure object.
    """
    if colors is None:
        colors = _palette(list(best_models.keys()))

    fs_labels = [fs_label for _, (_, fs_label) in best_models.items()]
    fs_keys = list(best_models.keys())
    n_fs = len(best_models)

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
    }

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(17, 13), facecolor=fig_bg)
        gs = matplotlib.gridspec.GridSpec(
            2, 2,
            figure=fig,
            width_ratios=[1.1, 1.0],
            height_ratios=[1.0, 1.0],
        )

        ax_metrics = fig.add_subplot(gs[0, 0])
        ax_calib = fig.add_subplot(gs[1, 0])
        # SHAP spans both rows in the right column.
        ax_shap = fig.add_subplot(gs[:, 1])

        # Manual margins: right margin leaves room for SHAP section labels.
        fig.subplots_adjust(left=0.07, right=0.84, top=0.93, bottom=0.09, hspace=0.42, wspace=0.38)

        ax_metrics.set_facecolor(axes_bg)
        ax_calib.set_facecolor(axes_bg)
        ax_shap.set_facecolor(axes_bg)

        # ── [0,0] Metric bars: grouped bar chart ──
        metrics_to_plot = ["sensitivity", "specificity", "f1", "accuracy"]
        metric_displays = ["Sensitivity", "Specificity", "F1", "Accuracy"]

        x_groups = np.arange(len(metrics_to_plot))
        bar_width = 0.8 / n_fs
        for i, fs in enumerate(fs_keys):
            best_run, fs_label = best_models[fs]
            color = colors[fs]
            values = [best_run.mean_metrics.loc[m, "mean"] * 100 for m in metrics_to_plot]
            sds = [best_run.mean_metrics.loc[m, "sd"] * 100 for m in metrics_to_plot]
            x_pos = x_groups + (i - n_fs / 2 + 0.5) * bar_width
            ax_metrics.bar(x_pos, values, bar_width, label=fs_label, color=color, alpha=0.8, yerr=sds)

        ax_metrics.set_xlabel("Metric")
        ax_metrics.set_ylabel("Score (%)")
        ax_metrics.set_title("Key Metrics — Best Model per Feature Set")
        ax_metrics.set_xticks(x_groups)
        ax_metrics.set_xticklabels(metric_displays, rotation=15, ha="right")
        ax_metrics.set_ylim(0, 110)
        ax_metrics.grid(axis="y", alpha=0.3)
        ax_metrics.legend()

        # ── [1,0] Brier Skill Score: BSS = 1 − Brier_model / Brier_ref ──
        # Brier_ref = prevalence × (1 − prevalence) — the score of a no-skill model
        # that always predicts the marginal event rate.  BSS=0 → no improvement over
        # baseline; BSS=1 → perfect; BSS<0 → worse than baseline.
        # SD propagates linearly: sd_BSS = sd_Brier / Brier_ref.
        x_pos = np.arange(n_fs)
        bss_values: list[float] = []
        bss_sds: list[float] = []
        bar_colors: list[tuple] = []

        for fs in fs_keys:
            best_run, _ = best_models[fs]
            mm = best_run.mean_metrics
            n_pos = mm.loc["n_pos", "mean"]
            n_total = mm.loc["n", "mean"]
            prevalence = n_pos / max(n_total, 1)
            brier_ref = prevalence * (1.0 - prevalence)

            brier_mean = mm.loc["brier", "mean"]
            brier_sd = mm.loc["brier", "sd"]

            bss = 1.0 - (brier_mean / brier_ref) if brier_ref > 0 else 0.0
            bss_sd = (brier_sd / brier_ref) if brier_ref > 0 else 0.0
            bss_values.append(bss)
            bss_sds.append(bss_sd)
            bar_colors.append(colors[fs])

        bars = ax_calib.bar(
            x_pos, bss_values,
            color=bar_colors, alpha=0.8,
            yerr=bss_sds, capsize=5, error_kw={"elinewidth": 1.5},
        )
        ax_calib.axhline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.6,
                         label="No skill (BSS = 0)")
        ax_calib.axhline(1.0, color="green", linestyle="--", linewidth=1.2, alpha=0.6,
                         label="Perfect (BSS = 1)")
        ax_calib.set_xlabel("Feature Set")
        ax_calib.set_ylabel("Brier Skill Score")
        ax_calib.set_title("Calibration: Brier Skill Score (mean ± SD)\n(higher = better; 0 = no-skill baseline)")
        ax_calib.set_xticks(x_pos)
        ax_calib.set_xticklabels([fs_labels[i] for i in range(n_fs)], rotation=15, ha="right")
        ax_calib.grid(axis="y", alpha=0.3)
        ax_calib.legend(fontsize=8 * font_scale)

        # ── [:, 1] SHAP importance — spans full right column ──
        y_pos = 0
        y_ticks: list[float] = []
        y_labels: list[str] = []

        for fs_idx, fs in enumerate(fs_keys):
            best_run, fs_label = best_models[fs]
            color = colors[fs]

            shap_df = best_run.shap_summary
            if not shap_df.empty and "mean_abs_shap" in shap_df.columns:
                top_features = shap_df.nlargest(top_n_shap, "mean_abs_shap")
            else:
                perm_df = best_run.permutation_importance
                if not perm_df.empty:
                    top_features = perm_df.nlargest(top_n_shap, "importance_mean")
                    top_features = top_features.rename(
                        columns={"importance_mean": "mean_abs_shap"}
                    )
                else:
                    continue

            # Feature-set section header (annotated text, not an extra tick).
            section_start = y_pos
            for _, row in top_features.iterrows():
                feat_name = _wrap(str(row["feature"]), width=20)
                ax_shap.barh(y_pos, row["mean_abs_shap"], color=color, alpha=0.75)
                y_labels.append(feat_name)
                y_ticks.append(y_pos)
                y_pos += 1

            # Label the section with the feature set name on the right.
            section_mid = (section_start + y_pos - 1) / 2
            ax_shap.annotate(
                fs_label,
                xy=(1.01, section_mid),
                xycoords=("axes fraction", "data"),
                fontsize=8 * font_scale,
                color=color,
                va="center",
                fontweight="bold",
            )

            if fs_idx < len(fs_keys) - 1:
                y_pos += 0.8  # gap between feature sets

        ax_shap.set_yticks(y_ticks)
        ax_shap.set_yticklabels(y_labels, fontsize=8 * font_scale)
        ax_shap.set_xlabel("Mean |SHAP| / Importance")
        ax_shap.set_title("Top-5 Feature Importance per Feature Set")
        ax_shap.grid(axis="x", alpha=0.3)

        # ── Finalise ───────────────────────────────────────────────────
        fig.suptitle(title, fontsize=15 * font_scale)

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig_bg)
            plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_best_model_summary_table(
    best_models: dict[str, tuple[ModelRunData, str]],
) -> pd.DataFrame:
    """
    Generate summary table for best models per feature set.

    One row per feature set, columns include: model, AUC-ROC, AUC-PR, metrics,
    confusion matrix counts, threshold.

    Parameters
    ----------
    best_models :
        Dict of ``{fs: (best_run, fs_label)}``.

    Returns
    -------
    pd.DataFrame
        Summary table, one row per feature set, no index.
    """
    rows = []

    for fs, (best_run, fs_label) in best_models.items():
        mm = best_run.mean_metrics
        row = {
            "Feature Set": fs_label,
            "Model": MODEL_DISPLAY.get(best_run.model_name, best_run.model_name),
            "AUC-ROC": _fmt(mm.loc["auc_roc", "mean"], mm.loc["auc_roc", "sd"], 3),
            "AUC-PR": _fmt(mm.loc["auc_pr", "mean"], mm.loc["auc_pr", "sd"], 3),
            "Sensitivity (%)": _fmt_pct(mm.loc["sensitivity", "mean"], mm.loc["sensitivity", "sd"], 1),
            "Specificity (%)": _fmt_pct(mm.loc["specificity", "mean"], mm.loc["specificity", "sd"], 1),
            "PPV (%)": _fmt_pct(mm.loc["ppv", "mean"], mm.loc["ppv", "sd"], 1),
            "F1 (%)": _fmt_pct(mm.loc["f1", "mean"], mm.loc["f1", "sd"], 1),
            "Accuracy (%)": _fmt_pct(mm.loc["accuracy", "mean"], mm.loc["accuracy", "sd"], 1),
            "Brier": _fmt(mm.loc["brier", "mean"], mm.loc["brier", "sd"], 3),
            "TP": str(int(round(mm.loc["tp", "mean"]))),
            "FP": str(int(round(mm.loc["fp", "mean"]))),
            "TN": str(int(round(mm.loc["tn", "mean"]))),
            "FN": str(int(round(mm.loc["fn", "mean"]))),
            "N": str(int(round(mm.loc["n", "mean"]))),
            "Threshold*": f"{mm.loc['threshold', 'mean']:.3f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def make_all_models_summary_table(
    results_root: Path = RESULTS_ROOT,
    feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS,
) -> pd.DataFrame:
    """
    Generate summary table for all models across all feature sets.

    One row per (feature_set, model) combination. Loads latest model runs.

    Parameters
    ----------
    results_root :
        Root of the ML results tree.
    feature_sets :
        Feature set names to include.

    Returns
    -------
    pd.DataFrame
        Summary table, one row per model per feature set, no index.
    """
    rows = []

    for fs in feature_sets:
        fs_label = FS_SHORT_LABELS.get(fs, fs)
        try:
            runs = load_all_models(fs, results_root)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load models for {fs}: {e}")
            continue

        for run in runs:
            mm = run.mean_metrics
            row = {
                "Feature Set": fs_label,
                "Model": MODEL_DISPLAY.get(run.model_name, run.model_name),
                "AUC-ROC": _fmt(mm.loc["auc_roc", "mean"], mm.loc["auc_roc", "sd"], 3),
                "AUC-PR": _fmt(mm.loc["auc_pr", "mean"], mm.loc["auc_pr", "sd"], 3),
                "Sensitivity (%)": _fmt_pct(mm.loc["sensitivity", "mean"], mm.loc["sensitivity", "sd"], 1),
                "Specificity (%)": _fmt_pct(mm.loc["specificity", "mean"], mm.loc["specificity", "sd"], 1),
                "PPV (%)": _fmt_pct(mm.loc["ppv", "mean"], mm.loc["ppv", "sd"], 1),
                "F1 (%)": _fmt_pct(mm.loc["f1", "mean"], mm.loc["f1", "sd"], 1),
                "Accuracy (%)": _fmt_pct(mm.loc["accuracy", "mean"], mm.loc["accuracy", "sd"], 1),
                "Brier": _fmt(mm.loc["brier", "mean"], mm.loc["brier", "sd"], 3),
                "TP": str(int(round(mm.loc["tp", "mean"]))),
                "FP": str(int(round(mm.loc["fp", "mean"]))),
                "TN": str(int(round(mm.loc["tn", "mean"]))),
                "FN": str(int(round(mm.loc["fn", "mean"]))),
                "N": str(int(round(mm.loc["n", "mean"]))),
                "Threshold*": f"{mm.loc['threshold', 'mean']:.3f}",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df
