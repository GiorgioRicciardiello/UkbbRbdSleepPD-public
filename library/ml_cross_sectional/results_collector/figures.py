"""
figures.py
==========

Multi-panel ROC + confusion-matrix figure for ML cross-sectional results.

Layout (N_rows × (2 + N_cm_cols + 1) cols, where N_rows = ceil(N_models / cm_max_per_row))::

    ┌──────────────────┬──────┬──────┬──────┐
    │  ROC (merged     │  CM  │ (CM) │ Bar  │
    │  2 cols, spans   │  M1  │      │(Acc/ │
    │  all rows)       │      │      │ F1,  │
    │                  │      │      │spans │
    ├──────────────────┼──────┼──────┤ all  │
    │                  │  CM  │ (CM) │ rows)
    │                  │  M2  │      │      │
    │                  │      │      │      │
    └──────────────────┴──────┴──────┴──────┘

* The ROC panel shows the mean ROC curve per model (across CV folds),
  spanning all rows and the merged left columns.
* CMs are arranged N_cm_cols per row (configurable via cm_max_per_row).
* The bar chart spans all rows on the right.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .collector import MODEL_DISPLAY, ModelRunData
from .plot_utils import (
    AXES_BG,
    FIG_BG,
    _cumulative_confusion,
    _draw_cm,
    _mean_roc,
    _palette,
    _threshold_point_on_curve,
    _wrap,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE = "Dark2"
FONT_SCALE = 1.3

#: Single optimization label used throughout (Youden-J threshold).
_OPT_LABEL = "youden"

MODEL_SHORT: dict[str, str] = {
    "logistic": "LR",
    "elasticnet": "ElasticNet",
    "xgboost": "XGB",
    "random_forest": "RF",
    "svm_rbf": "SVM",
}


# ---------------------------------------------------------------------------
# Data adapter — ModelRunData → long-format DataFrames
# ---------------------------------------------------------------------------

def _build_dataframes(
    runs: Sequence[ModelRunData],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a list of ``ModelRunData`` objects into the long-format
    DataFrames expected by the plotting functions.

    Parameters
    ----------
    runs :
        One entry per model, loaded by ``collector.load_all_models``.

    Returns
    -------
    df_predictions :
        Long-format frame with columns:
        ``config, model_type, optimization, fold, row_index,
          y_true, y_pred_proba, y_pred``.
    df_selected :
        One row per model: ``config, model_type, optimization``.
    """
    pred_frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []

    for run in runs:
        threshold = run.mean_metrics.loc["threshold", "mean"]
        preds = run.predictions.copy()
        preds["config"] = run.model_name
        preds["model_type"] = run.model_name
        preds["optimization"] = _OPT_LABEL
        preds["y_pred"] = (preds["y_pred_proba"] >= threshold).astype(int)
        pred_frames.append(preds)

        selected_rows.append({
            "config": run.model_name,
            "model_type": run.model_name,
            "optimization": _OPT_LABEL,
        })

    df_predictions = pd.concat(pred_frames, ignore_index=True)
    df_selected = pd.DataFrame(selected_rows)
    return df_predictions, df_selected


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_roc_and_cm(
    runs: Sequence[ModelRunData],
    out_path: Path,
    cm_max_per_row: int = 2,
    dpi: int = 300,
    font_scale: float = FONT_SCALE,
    palette: str = _DEFAULT_PALETTE,
    fig_bg: str = FIG_BG,
    axes_bg: str = AXES_BG,
    title: str = "ROC + Confusion Matrices",
    fig_size: tuple[float, float] | None = None,
    xlim: tuple[float, float] = (0.0, 0.8),
    ylim: tuple[float, float] = (0.5, 1.0),
    class_names: dict[int, str] | None = None,
    feature_set_label: str | None = None,
) -> Path:
    """
    Generate the multi-panel ROC + confusion-matrix figure with configurable CM layout.

    Layout: ROC panel on the left (spanning all rows, merged from 2 cols),
    CMs arranged in a grid (configurable: 1 or more per row),
    plus a bar chart of accuracy & F1 on the far right (spanning all rows).

    Parameters
    ----------
    runs :
        Loaded model run data, one per model.
    out_path :
        Output PNG path.
    cm_max_per_row :
        Maximum number of confusion matrices per row (default: 2).
        Layout adjusts to ceil(n_models / cm_max_per_row) rows.
    dpi :
        Resolution.
    font_scale :
        Global font multiplier.
    palette :
        Seaborn palette name.
    fig_bg, axes_bg :
        Figure and axes background colours.
    title :
        Figure suptitle.
    fig_size :
        ``(width, height)`` in inches. ``None`` = auto.
    xlim, ylim :
        ROC axis limits.
    class_names :
        ``{0: "Neg", 1: "Pos"}`` labels for CM axes.
    feature_set_label :
        Optional feature set label (e.g., "RBD + PRS + Prodromal").
        If provided, appended to the title.

    Returns
    -------
    Path
        Saved figure path.
    """
    class_names = class_names or {0: "Neg", 1: "Pos"}

    df_predictions, df_selected = _build_dataframes(runs)

    configs = [run.model_name for run in runs]   # ordered
    n_models = len(configs)
    colors = _palette(configs, name=palette)

    # Layout: 2 ROC cols (merged) + N_cm_cols CM cols.
    n_cm_cols = min(n_models, cm_max_per_row)
    n_cm_rows = math.ceil(n_models / cm_max_per_row)
    n_total_cols = 2 + n_cm_cols
    width_ratios = [1.0, 1.0] + [1.0] * n_cm_cols

    # Figure height scales with number of CM rows.
    _fig_size = fig_size or (4.0 * n_total_cols, max(5.0, 4.0 * n_cm_rows))

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
        fig, axes = plt.subplots(
            n_cm_rows, n_total_cols,
            figsize=_fig_size,
            facecolor=fig_bg,
            gridspec_kw={"width_ratios": width_ratios},
        )
        axes = np.atleast_2d(axes)

        # ── ROC panel: merge first 2 cols, span all rows ───────────────
        gs = axes[0, 0].get_gridspec()
        for row in range(n_cm_rows):
            for col in range(2):
                axes[row, col].remove()
        roc_ax = fig.add_subplot(gs[:, 0:2])
        roc_ax.set_facecolor(axes_bg)

        roc_ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

        for run in runs:
            cfg = run.model_name
            display = MODEL_DISPLAY.get(cfg, cfg)
            short = MODEL_SHORT.get(cfg, cfg)
            color = colors[cfg]

            preds_cfg = df_predictions[df_predictions["config"] == cfg]
            fpr_grid, mean_tpr, std_tpr, mean_auc, std_auc = _mean_roc(preds_cfg)

            # Mean ROC line with bold label.
            roc_ax.plot(
                fpr_grid, mean_tpr,
                color=color, linewidth=2.5,
                label=(
                    f"$\\bf{{{display}}}$\n"
                    f"AUC: {mean_auc:.3f} ({std_auc:.3f}) | "
                    f"{short} | "
                    r"$\tau^{*}$"
                    f": {run.mean_metrics.loc['threshold', 'mean']:.3f}"
                ),
            )

            # Std band.
            roc_ax.fill_between(
                fpr_grid,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=color, alpha=0.10,
            )

            # Diamond marker snapped onto the mean curve.
            tau = run.mean_metrics.loc["threshold", "mean"]
            fpr_t, tpr_t = _threshold_point_on_curve(
                preds_cfg, tau, fpr_grid, mean_tpr,
            )
            roc_ax.scatter(
                fpr_t, tpr_t,
                s=80, color=color,
                edgecolor="black", linewidths=1.0,
                marker="D", zorder=5,
            )

        roc_ax.set_xlabel("False Positive Rate (1 - Specificity)")
        roc_ax.set_ylabel("True Positive Rate (Sensitivity)")
        roc_ax.set_title("ROC — best model per configuration")
        if xlim:
            roc_ax.set_xlim(*xlim)
        if ylim:
            roc_ax.set_ylim(*ylim)
        roc_ax.grid(alpha=0.3)
        leg = roc_ax.legend(loc="lower right", frameon=True)
        for legline in leg.get_lines():
            legline.set_linewidth(4.0)

        # ── Confusion matrices with metrics in title ────────────────────
        for i, run in enumerate(runs):
            row_idx = i // cm_max_per_row
            col_idx = 2 + (i % cm_max_per_row)
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(axes_bg)

            cfg = run.model_name
            display = MODEL_DISPLAY.get(cfg, cfg)
            color = colors[cfg]

            # Extract metrics (rounded to 1 decimal).
            accuracy = run.mean_metrics.loc["accuracy", "mean"]
            precision = run.mean_metrics.loc["ppv", "mean"]
            f1 = run.mean_metrics.loc["f1", "mean"]

            preds_cfg = df_predictions[df_predictions["config"] == cfg]
            cm = _cumulative_confusion(preds_cfg)
            n_samples = int(cm.sum())

            # Title with formatting: italic model, normal metrics + sample size.
            cm_title = (
                f"$\\mathit{{{display}}}$\n"
                f"n={n_samples} | Acc={accuracy:.1f} Prec={precision:.1f} F1={f1:.1f}"
            )

            # Determine label visibility based on position.
            show_y_labels = (col_idx == 2)  # Leftmost CM column
            show_x_labels = (row_idx == n_cm_rows - 1)  # Bottom row

            _draw_cm(
                ax, cm,
                title=cm_title,
                class_names=class_names,
                base_color=color,
                fontsize=11 * font_scale,
                show_x_labels=show_x_labels,
                show_y_labels=show_y_labels,
            )

        # Hide any unused CM cells in the last row.
        for j in range(n_models, n_cm_rows * n_cm_cols):
            axes[j // n_cm_cols, 2 + (j % n_cm_cols)].axis("off")

        # ── Finalise ───────────────────────────────────────────────────
        if feature_set_label:
            full_title = f"{title}\n{feature_set_label}"
        else:
            full_title = title
        fig.suptitle(full_title, fontsize=15 * font_scale, y=1.002)
        fig.tight_layout()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig_bg)
        plt.close(fig)

    return out_path
