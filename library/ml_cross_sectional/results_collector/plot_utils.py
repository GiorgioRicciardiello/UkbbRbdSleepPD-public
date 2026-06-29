"""
plot_utils.py
=============

Shared plotting utilities for ML cross-sectional visualization modules.

Contains all common helper functions and constants used by:
- figures.py (single feature set comparison)
- final_figure.py (cross-feature-set comparison)

Ported from ``MignotLab/CataplexyQuestionnaire/module/src/nestedfs/plotting.py``.
"""
from __future__ import annotations

import textwrap
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIG_BG: str = "#f7f7f8"
AXES_BG: str = "#eeeeee"
_DEFAULT_PALETTE: str = "Dark2"


# ---------------------------------------------------------------------------
# Helpers — ported from nestedfs/plotting.py
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 22) -> str:
    """Wrap text to specified width using textwrap."""
    return "\n".join(textwrap.wrap(text, width=width))


def _palette(
    keys: Sequence[str],
    name: str = _DEFAULT_PALETTE,
) -> dict[str, tuple[float, ...]]:
    """
    Build a color palette mapping.

    Parameters
    ----------
    keys :
        Sequence of keys to assign colors to.
    name :
        Seaborn palette name (default: "Dark2").

    Returns
    -------
    dict
        Mapping of keys to RGBA tuples.
    """
    keys = list(keys)
    colors = sns.color_palette(name, max(len(keys), 3))
    return {k: colors[i % len(colors)] for i, k in enumerate(keys)}


def _mean_roc(
    df_preds: pd.DataFrame,
    *,
    fold_col: str = "fold",
    y_true_col: str = "y_true",
    y_score_col: str = "y_pred_proba",
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Mean ROC curve across CV folds via linear interpolation.

    Parameters
    ----------
    df_preds :
        Predictions frame with fold, y_true, y_pred_proba columns.
    fold_col :
        Name of the fold column.
    y_true_col :
        Name of the true label column.
    y_score_col :
        Name of the predicted probability column.
    n_points :
        Number of points in the FPR grid.

    Returns
    -------
    fpr_grid :
        Common FPR grid, shape (n_points,).
    mean_tpr :
        Mean TPR at each grid point.
    std_tpr :
        Std TPR at each grid point.
    mean_auc :
        Mean AUC across folds.
    std_auc :
        Std AUC across folds.
    """
    fpr_grid = np.linspace(0.0, 1.0, n_points)
    tpr_interp: list[np.ndarray] = []
    aucs: list[float] = []

    for _, chunk in df_preds.groupby(fold_col):
        y_true = chunk[y_true_col].to_numpy(dtype=int)
        y_score = chunk[y_score_col].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp = np.interp(fpr_grid, fpr, tpr)
        interp[0] = 0.0
        tpr_interp.append(interp)
        aucs.append(float(auc(fpr, tpr)))

    if not tpr_interp:
        return fpr_grid, np.zeros_like(fpr_grid), np.zeros_like(fpr_grid), float("nan"), float("nan")

    arr = np.array(tpr_interp)
    mean_tpr = arr.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = arr.std(axis=0)
    return fpr_grid, mean_tpr, std_tpr, float(np.mean(aucs)), float(np.std(aucs))


def _mean_pr(
    df_preds: pd.DataFrame,
    *,
    fold_col: str = "fold",
    y_true_col: str = "y_true",
    y_score_col: str = "y_pred_proba",
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Mean PR curve across CV folds via linear interpolation on recall grid.

    Parameters
    ----------
    df_preds :
        Predictions frame with fold, y_true, y_pred_proba columns.
    fold_col, y_true_col, y_score_col :
        Column names.
    n_points :
        Number of points in the recall grid.

    Returns
    -------
    recall_grid :
        Common recall grid, shape (n_points,).
    mean_precision :
        Mean precision at each grid point.
    std_precision :
        Std precision at each grid point.
    mean_auc_pr :
        Mean AUC-PR across folds.
    std_auc_pr :
        Std AUC-PR across folds.
    """
    recall_grid = np.linspace(0.0, 1.0, n_points)
    prec_interp: list[np.ndarray] = []
    auc_prs: list[float] = []

    for _, chunk in df_preds.groupby(fold_col):
        y_true = chunk[y_true_col].to_numpy(dtype=int)
        y_score = chunk[y_score_col].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns in decreasing threshold order (decreasing recall).
        # Reverse to get increasing recall, then interpolate.
        rec_sorted = rec[::-1]
        prec_sorted = prec[::-1]
        interp = np.interp(recall_grid, rec_sorted, prec_sorted)
        interp[0] = 1.0  # precision at recall=0 is undefined, set to 1.0 (conservative)
        prec_interp.append(interp)
        auc_prs.append(float(average_precision_score(y_true, y_score)))

    if not prec_interp:
        return recall_grid, np.zeros_like(recall_grid), np.zeros_like(recall_grid), float("nan"), float("nan")

    arr = np.array(prec_interp)
    mean_prec = arr.mean(axis=0)
    std_prec = arr.std(axis=0)
    return recall_grid, mean_prec, std_prec, float(np.mean(auc_prs)), float(np.std(auc_prs))


def _threshold_point_on_curve(
    df_preds: pd.DataFrame,
    threshold: float,
    base_fpr: np.ndarray,
    mean_tpr: np.ndarray,
    fold_col: str = "fold",
    y_true_col: str = "y_true",
    y_score_col: str = "y_pred_proba",
) -> tuple[float, float]:
    """
    Find the mean (FPR, TPR) at *threshold* across folds and snap it to
    the nearest point on the plotted mean ROC curve (so the marker lies
    exactly on the line).

    Parameters
    ----------
    df_preds :
        Predictions frame.
    threshold :
        Decision threshold value.
    base_fpr :
        FPR grid used for the plotted curve.
    mean_tpr :
        Plotted mean TPR values.

    Returns
    -------
    fpr_snapped, tpr_snapped :
        Snapped (FPR, TPR) coordinates on the mean curve.
    """
    fprs: list[float] = []
    tprs: list[float] = []
    for _, chunk in df_preds.groupby(fold_col):
        y_true = chunk[y_true_col].to_numpy(dtype=int)
        y_score = chunk[y_score_col].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr_f, tpr_f, thr_f = roc_curve(y_true, y_score)
        idx = int(np.argmin(np.abs(thr_f - threshold)))
        fprs.append(float(fpr_f[idx]))
        tprs.append(float(tpr_f[idx]))

    if not fprs:
        return float("nan"), float("nan")

    raw_fpr = float(np.mean(fprs))
    raw_tpr = float(np.mean(tprs))
    # Snap to the nearest point on the plotted curve.
    dist = (base_fpr - raw_fpr) ** 2 + (mean_tpr - raw_tpr) ** 2
    snap = int(np.argmin(dist))
    return float(base_fpr[snap]), float(mean_tpr[snap])


def _cumulative_confusion(
    df_preds: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> np.ndarray:
    """
    Sum of per-fold confusion matrices → 2×2 array [[TN, FP], [FN, TP]].

    Parameters
    ----------
    df_preds :
        Predictions frame with y_true and y_pred columns.

    Returns
    -------
    np.ndarray
        2×2 confusion matrix.
    """
    y_true = df_preds[y_true_col].to_numpy(dtype=int)
    y_pred = df_preds[y_pred_col].to_numpy(dtype=int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def _draw_cm(
    ax: plt.Axes,
    cm: np.ndarray,
    *,
    title: str,
    class_names: dict[int, str],
    base_color: str | tuple,
    fontsize: float,
    show_x_labels: bool = True,
    show_y_labels: bool = True,
) -> None:
    """
    Draw a 2×2 confusion matrix heatmap with count + row-% annotations.
    Matches the style from nestedfs/plotting.py.

    Parameters
    ----------
    ax :
        Matplotlib axes to draw on.
    cm :
        2×2 confusion matrix.
    title :
        Panel title.
    class_names :
        Mapping of class labels (0, 1) to display names.
    base_color :
        Base color for the heatmap (passed to sns.light_palette).
    fontsize :
        Font size for labels.
    show_x_labels :
        Whether to show x-axis (Pred) labels (default: True).
    show_y_labels :
        Whether to show y-axis (True) labels (default: True).
    """
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = cm / row_sums * 100.0
    labels = np.array([
        [f"{cm[0, 0]:,}\n{pct[0, 0]:.1f}%", f"{cm[0, 1]:,}\n{pct[0, 1]:.1f}%"],
        [f"{cm[1, 0]:,}\n{pct[1, 0]:.1f}%", f"{cm[1, 1]:,}\n{pct[1, 1]:.1f}%"],
    ])
    cmap = sns.light_palette(base_color, as_cmap=True)
    sns.heatmap(
        cm, ax=ax, cmap=cmap, cbar=False, square=True,
        linewidths=2, linecolor="white",
        annot=labels, fmt="",
        vmin=0, vmax=max(1, int(cm.max())),
        annot_kws={"fontsize": fontsize, "ha": "center", "va": "center", "color": "black"},
    )
    ax.set_title(title, fontsize=fontsize * 1.1)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])

    if show_x_labels:
        ax.set_xticklabels(
            [f"Pred {class_names[0]}", f"Pred {class_names[1]}"],
            fontsize=fontsize * 0.9,
        )
    else:
        ax.set_xticklabels([])

    if show_y_labels:
        ax.set_yticklabels(
            [f"True {class_names[0]}", f"True {class_names[1]}"],
            fontsize=fontsize * 0.9,
        )
    else:
        ax.set_yticklabels([])

    ax.set_xlabel("")
    ax.set_ylabel("")
