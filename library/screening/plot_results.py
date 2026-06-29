"""
Screening paradigm comparison — results visualisation.

Reads the per-fold CSV produced by ``main.py`` and generates a suite of
publication-quality plots:

  Figure 1  (4-panel)  : Box plots of all metrics across paradigms
  Figure 2  (4-panel)  : Fold-level metric trajectories (stability)
  Figure 3  (1-panel)  : Metric heatmap (paradigms × folds)
  Figure 4  (1-panel)  : Summary bar chart (mean ± 95 % CI)
  Figure 5  (1-panel)  : Calibration slope comparison

All figures are saved to the same directory as the input CSV.
If no CSV path is supplied, the script auto-detects the most recent run
in ``results/screening_paradigms/``.

Usage
-----
Run standalone::

    python -m src.screening.plot_results
    python -m src.screening.plot_results --folds_csv results/screening_paradigms/20260415_120000_paradigm_comparison_folds.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

METRIC_LABELS: Dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier_score": "Brier Score",
    "calibration_slope": "Calibration Slope",
}

METRIC_COLS: List[str] = list(METRIC_LABELS.keys())

# Per-metric: (lower_is_better, reference_line_value, y_label)
METRIC_META: Dict[str, Tuple[bool, Optional[float], str]] = {
    "roc_auc":            (False, None,  "ROC-AUC"),
    "pr_auc":             (False, None,  "PR-AUC (Average Precision)"),
    "brier_score":        (True,  None,  "Brier Score (↓ better)"),
    "calibration_slope":  (False, 1.0,   "Calibration Slope (1 = ideal)"),
}

# Consistent color palette per paradigm
_PALETTE_COLORS = [
    "#2E86AB",  # blue       – P1 combined
    "#A23B72",  # purple     – P2 incident only
    "#F18F01",  # orange     – P3 weighted α=0.30
    "#C73E1D",  # red        – P3 weighted α=0.10
    "#3B1F2B",  # dark brown – P4 subsample r5
    "#44BBA4",  # teal       – P6 prevalent train
]

# Human-readable labels for paradigm IDs
PARADIGM_DISPLAY: Dict[str, str] = {
    "p1_combined":               "P1: Combined\n(incident + prevalent)",
    "p2_incident_only":          "P2: Incident only",
    "p3_weighted_a030":          "P3: Weighted α=0.30",
    "p3_weighted_a010":          "P3: Weighted α=0.10",
    "p4_subsample_r5":           "P4: Subsample (1:5)",
    "p6_prevalent_train":        "P6: Prevalent→Incident",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _find_latest_folds_csv() -> Path:
    """Return the most recent folds CSV in results/screening_paradigms/."""
    results_root = Path("results") / "screening_paradigms"
    candidates = sorted(results_root.glob("*_paradigm_comparison_folds.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No paradigm comparison CSVs found in {results_root}. "
            "Run src.screening.main first."
        )
    return candidates[-1]


def load_folds(folds_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the per-fold results CSV.

    Parameters
    ----------
    folds_csv : Path, optional
        Explicit path to the folds CSV.  If None, auto-detects latest run.

    Returns
    -------
    pd.DataFrame
        Columns: fold, paradigm, n_cases, n_controls, roc_auc, pr_auc,
                 brier_score, calibration_slope, (optional param_* columns).
    """
    path = folds_csv or _find_latest_folds_csv()
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df, path


def _paradigm_order(df: pd.DataFrame) -> List[str]:
    """Return paradigm names in canonical order (P1 … P6), unknowns at end."""
    canonical = list(PARADIGM_DISPLAY.keys())
    present = df["paradigm"].unique().tolist()
    ordered = [p for p in canonical if p in present]
    ordered += [p for p in present if p not in canonical]
    return ordered


def _colors_for(paradigms: List[str]) -> List[str]:
    """Map each paradigm to a colour (cycle if > 6)."""
    base = list(PARADIGM_DISPLAY.keys())
    return [
        _PALETTE_COLORS[base.index(p) % len(_PALETTE_COLORS)]
        if p in base
        else _PALETTE_COLORS[len(base) % len(_PALETTE_COLORS)]
        for p in paradigms
    ]


def _display_label(paradigm: str) -> str:
    return PARADIGM_DISPLAY.get(paradigm, paradigm)


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame, paradigms: List[str]) -> pd.DataFrame:
    """
    Compute mean, SD, and 95 % CI for each metric × paradigm.

    95 % CI uses t-distribution (appropriate for n ≈ 10 folds).

    Returns
    -------
    pd.DataFrame  with columns: paradigm, metric, mean, sd, ci_lo, ci_hi, n
    """
    rows = []
    for paradigm in paradigms:
        sub = df[df["paradigm"] == paradigm]
        for metric in METRIC_COLS:
            vals = sub[metric].dropna().values
            n = len(vals)
            if n == 0:
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n)
            t_crit = float(stats.t.ppf(0.975, df=max(n - 1, 1)))
            ci_lo = mean - t_crit * sem
            ci_hi = mean + t_crit * sem
            rows.append(dict(paradigm=paradigm, metric=metric,
                             mean=mean, sd=sd, ci_lo=ci_lo, ci_hi=ci_hi, n=n))
    return pd.DataFrame(rows)


# ── Figure 1: Box plots (4-panel) ─────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame,
    paradigms: List[str],
    colors: List[str],
    out_path: Path,
) -> None:
    """
    4-panel figure: one box plot per metric, paradigms on x-axis.

    Each box shows the distribution across outer CV folds.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRIC_COLS):
        lower_better, ref_line, ylabel = METRIC_META[metric]

        data_per_paradigm = [
            df.loc[df["paradigm"] == p, metric].dropna().values
            for p in paradigms
        ]

        bp = ax.boxplot(
            data_per_paradigm,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        if ref_line is not None:
            ax.axhline(ref_line, color="black", linestyle="--", linewidth=1,
                       label=f"Reference = {ref_line}")
            ax.legend(fontsize=8)

        display_labels = [_display_label(p) for p in paradigms]
        ax.set_xticks(range(1, len(paradigms) + 1))
        ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("")

    fig.suptitle(
        "Paradigm Comparison — Metric Distribution Across Outer CV Folds",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 1 (boxplots) → %s", out_path)


# ── Figure 2: Fold-level trajectories ─────────────────────────────────────────

def plot_fold_trajectories(
    df: pd.DataFrame,
    paradigms: List[str],
    colors: List[str],
    out_path: Path,
) -> None:
    """
    4-panel figure: metric value per fold for each paradigm (line plot).

    Shows stability of each paradigm across the 10 outer folds.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRIC_COLS):
        _, ref_line, ylabel = METRIC_META[metric]

        for paradigm, color in zip(paradigms, colors):
            sub = df[df["paradigm"] == paradigm].sort_values("fold")
            folds = sub["fold"].values + 1   # 1-based for display
            vals = sub[metric].values
            ax.plot(folds, vals, marker="o", color=color, linewidth=1.5,
                    markersize=4, label=_display_label(paradigm), alpha=0.85)

        if ref_line is not None:
            ax.axhline(ref_line, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel("Outer fold")
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(1, df["fold"].max() + 2))

    # Shared legend below the panels
    handles = [
        mpatches.Patch(color=c, label=_display_label(p), alpha=0.85)
        for p, c in zip(paradigms, colors)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.suptitle(
        "Metric Stability Across Outer CV Folds",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 2 (trajectories) → %s", out_path)


# ── Figure 3: Metric heatmap ──────────────────────────────────────────────────

def plot_metric_heatmap(
    df: pd.DataFrame,
    paradigms: List[str],
    out_path: Path,
    metric: str = "roc_auc",
) -> None:
    """
    Heatmap of a single metric value for each (paradigm, fold) cell.

    Reveals which paradigm × fold combinations are strongest/weakest.
    """
    n_folds = df["fold"].max() + 1
    matrix = np.full((len(paradigms), n_folds), np.nan)

    for i, paradigm in enumerate(paradigms):
        sub = df[df["paradigm"] == paradigm]
        for _, row in sub.iterrows():
            matrix[i, int(row["fold"])] = row[metric]

    _, _, ylabel = METRIC_META[metric]

    fig, ax = plt.subplots(figsize=(min(n_folds + 2, 14), len(paradigms) * 0.85 + 1.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

    # Annotate cells
    for i in range(len(paradigms)):
        for j in range(n_folds):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7.5, color="black")

    ax.set_xticks(range(n_folds))
    ax.set_xticklabels([f"Fold {k+1}" for k in range(n_folds)], fontsize=9)
    ax.set_yticks(range(len(paradigms)))
    ax.set_yticklabels([_display_label(p) for p in paradigms], fontsize=9)
    ax.set_title(f"{ylabel} — Paradigm × Fold Heatmap",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=ylabel)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 3 (heatmap) → %s", out_path)


# ── Figure 4: Summary bar chart ───────────────────────────────────────────────

def plot_summary_bars(
    summary: pd.DataFrame,
    paradigms: List[str],
    colors: List[str],
    out_path: Path,
) -> None:
    """
    4-panel grouped bar chart: mean ± 95 % CI per metric × paradigm.

    Suitable as a compact summary figure for publication.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, metric in zip(axes, METRIC_COLS):
        _, ref_line, ylabel = METRIC_META[metric]
        sub = summary[summary["metric"] == metric]
        sub = sub.set_index("paradigm").reindex(paradigms).reset_index()

        x = np.arange(len(paradigms))
        means = sub["mean"].values
        ci_lo = sub["ci_lo"].values
        ci_hi = sub["ci_hi"].values
        yerr_lo = means - ci_lo
        yerr_hi = ci_hi - means
        yerr = np.array([np.abs(yerr_lo), np.abs(yerr_hi)])

        bars = ax.bar(x, means, yerr=yerr, color=colors, alpha=0.8,
                      error_kw=dict(ecolor="black", capsize=4, linewidth=1.2),
                      width=0.65)

        if ref_line is not None:
            ax.axhline(ref_line, color="black", linestyle="--", linewidth=1)

        # Value annotations on bars
        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{mean_val:.3f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_display_label(p).replace("\n", " ") for p in paradigms],
            rotation=40, ha="right", fontsize=7.5,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(METRIC_LABELS[metric])

    fig.suptitle(
        "Screening Paradigm Comparison — Mean ± 95 % CI (10-fold outer CV)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 4 (summary bars) → %s", out_path)


# ── Figure 5: Calibration slope detail ────────────────────────────────────────

def plot_calibration_detail(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    paradigms: List[str],
    colors: List[str],
    out_path: Path,
) -> None:
    """
    Horizontal dot-plot of calibration slope per paradigm.

    Individual fold values shown as jittered dots;
    mean ± 95 % CI shown as error bars.
    Reference line at 1.0 (perfect calibration).
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(paradigms) * 0.9 + 1)))
    y_positions = np.arange(len(paradigms))
    rng = np.random.default_rng(0)

    for idx, (paradigm, color) in enumerate(zip(paradigms, colors)):
        sub_folds = df[df["paradigm"] == paradigm]["calibration_slope"].dropna().values
        sub_summ = summary[(summary["paradigm"] == paradigm) &
                           (summary["metric"] == "calibration_slope")]

        if len(sub_folds):
            jitter = rng.uniform(-0.15, 0.15, size=len(sub_folds))
            ax.scatter(sub_folds, np.full_like(sub_folds, idx) + jitter,
                       color=color, alpha=0.5, s=30, zorder=3)

        if not sub_summ.empty:
            mean_val = float(sub_summ["mean"].iloc[0])
            ci_lo = float(sub_summ["ci_lo"].iloc[0])
            ci_hi = float(sub_summ["ci_hi"].iloc[0])
            ax.errorbar(mean_val, idx, xerr=[[mean_val - ci_lo], [ci_hi - mean_val]],
                        fmt="D", color=color, markersize=8, linewidth=2,
                        capsize=5, zorder=4)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5,
               label="Perfect calibration (slope = 1)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_display_label(p) for p in paradigms])
    ax.set_xlabel("Calibration Slope")
    ax.set_title("Calibration Slope — Individual Folds and Mean ± 95 % CI",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 5 (calibration detail) → %s", out_path)


# ── Figure 6: Training set composition ────────────────────────────────────────

def plot_training_composition(
    df: pd.DataFrame,
    paradigms: List[str],
    colors: List[str],
    out_path: Path,
) -> None:
    """
    Stacked bar chart showing mean n_cases and n_controls per paradigm.

    Reveals the case-control ratio actually achieved after fold-level matching.
    """
    summary_comp = (
        df.groupby("paradigm")[["n_cases", "n_controls"]]
        .mean()
        .reindex(paradigms)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(paradigms))
    w = 0.6

    bars_ctrl = ax.bar(x, summary_comp["n_controls"], w,
                       label="Controls", color="#CCCCCC", alpha=0.8)
    bars_case = ax.bar(x, summary_comp["n_cases"], w,
                       bottom=summary_comp["n_controls"],
                       label="Cases (positive label)", color=[c for c in colors], alpha=0.85)

    for i, paradigm in enumerate(paradigms):
        n_c = summary_comp.loc[paradigm, "n_cases"]
        n_ctrl = summary_comp.loc[paradigm, "n_controls"]
        if not np.isnan(n_c) and not np.isnan(n_ctrl):
            ratio = n_ctrl / max(n_c, 1)
            ax.text(x[i], n_c + n_ctrl + 5, f"1:{ratio:.0f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_display_label(p).replace("\n", " ") for p in paradigms],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean subjects per training fold")
    ax.set_title("Training Set Composition (mean per fold)\nRatio = controls : cases",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 6 (training composition) → %s", out_path)


# ── Figure 7: ROC-AUC vs PR-AUC scatter ──────────────────────────────────────

def plot_roc_vs_pr(
    summary: pd.DataFrame,
    paradigms: List[str],
    colors: List[str],
    out_path: Path,
) -> None:
    """
    Scatter of mean ROC-AUC vs mean PR-AUC per paradigm.

    PR-AUC is the primary metric under severe imbalance; ROC-AUC may be
    over-optimistic.  Paradigms in the top-right quadrant are preferred.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    roc_sub = summary[summary["metric"] == "roc_auc"].set_index("paradigm")
    pr_sub  = summary[summary["metric"] == "pr_auc"].set_index("paradigm")

    for paradigm, color in zip(paradigms, colors):
        if paradigm not in roc_sub.index or paradigm not in pr_sub.index:
            continue
        roc_mean = roc_sub.loc[paradigm, "mean"]
        roc_err  = [[roc_mean - roc_sub.loc[paradigm, "ci_lo"]],
                    [roc_sub.loc[paradigm, "ci_hi"] - roc_mean]]
        pr_mean  = pr_sub.loc[paradigm, "mean"]
        pr_err   = [[pr_mean - pr_sub.loc[paradigm, "ci_lo"]],
                    [pr_sub.loc[paradigm, "ci_hi"] - pr_mean]]

        ax.errorbar(roc_mean, pr_mean,
                    xerr=roc_err, yerr=pr_err,
                    fmt="o", color=color, markersize=9,
                    capsize=4, linewidth=1.5, alpha=0.9,
                    label=_display_label(paradigm))
        ax.annotate(
            _display_label(paradigm).split("\n")[0],
            (roc_mean, pr_mean),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7.5, color=color,
        )

    ax.set_xlabel("Mean ROC-AUC")
    ax.set_ylabel("Mean PR-AUC (Average Precision)")
    ax.set_title("ROC-AUC vs PR-AUC — Paradigm Comparison\n"
                 "(error bars = 95 % CI; top-right = better)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 7 (ROC vs PR scatter) → %s", out_path)


# ── Figure 8: Summary table (text figure) ─────────────────────────────────────

def plot_summary_table(
    summary: pd.DataFrame,
    paradigms: List[str],
    out_path: Path,
) -> None:
    """
    Render a formatted summary table as a matplotlib figure.

    Columns: paradigm, ROC-AUC, PR-AUC, Brier, CalSlope — all as mean ± SD.
    """
    rows = []
    for paradigm in paradigms:
        sub = summary[summary["paradigm"] == paradigm].set_index("metric")
        row = [_display_label(paradigm).replace("\n", " ")]
        for metric in METRIC_COLS:
            if metric in sub.index:
                m = sub.loc[metric, "mean"]
                s = sub.loc[metric, "sd"]
                row.append(f"{m:.3f} ± {s:.3f}")
            else:
                row.append("—")
        rows.append(row)

    col_labels = ["Paradigm"] + [METRIC_LABELS[m] for m in METRIC_COLS]
    fig, ax = plt.subplots(figsize=(13, max(3, len(paradigms) * 0.6 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2E86AB")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(rows) + 1):
        bg = "#F0F4F8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)

    ax.set_title(
        "Screening Paradigm Comparison — Mean ± SD (10-fold outer CV)",
        fontsize=11, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Figure 8 (summary table) → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(folds_csv: Optional[Path] = None) -> None:
    """
    Generate all figures from a paradigm comparison folds CSV.

    Parameters
    ----------
    folds_csv : Path, optional
        Path to ``*_paradigm_comparison_folds.csv``.  Auto-detected if None.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    df, csv_path = load_folds(folds_csv)
    out_dir = csv_path.parent
    stem = csv_path.stem.replace("_paradigm_comparison_folds", "")

    paradigms = _paradigm_order(df)
    colors = _colors_for(paradigms)

    logger.info("Paradigms detected (%d): %s", len(paradigms), paradigms)
    logger.info("Folds: %s", sorted(df["fold"].unique()))

    summary = compute_summary(df, paradigms)

    # ── Render figures ────────────────────────────────────────────────────────
    plot_boxplots(
        df, paradigms, colors,
        out_dir / f"{stem}_fig1_boxplots.png",
    )
    plot_fold_trajectories(
        df, paradigms, colors,
        out_dir / f"{stem}_fig2_trajectories.png",
    )
    plot_metric_heatmap(
        df, paradigms,
        out_dir / f"{stem}_fig3_heatmap_roc.png",
        metric="roc_auc",
    )
    plot_metric_heatmap(
        df, paradigms,
        out_dir / f"{stem}_fig3_heatmap_pr.png",
        metric="pr_auc",
    )
    plot_summary_bars(
        summary, paradigms, colors,
        out_dir / f"{stem}_fig4_summary_bars.png",
    )
    plot_calibration_detail(
        df, summary, paradigms, colors,
        out_dir / f"{stem}_fig5_calibration.png",
    )
    plot_training_composition(
        df, paradigms, colors,
        out_dir / f"{stem}_fig6_training_composition.png",
    )
    plot_roc_vs_pr(
        summary, paradigms, colors,
        out_dir / f"{stem}_fig7_roc_vs_pr.png",
    )
    plot_summary_table(
        summary, paradigms,
        out_dir / f"{stem}_fig8_summary_table.png",
    )

    logger.info("All figures saved to %s", out_dir)
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot paradigm comparison results."
    )
    parser.add_argument(
        "--folds_csv",
        type=Path,
        default=None,
        help="Path to *_paradigm_comparison_folds.csv (auto-detects latest if omitted).",
    )
    args = parser.parse_args()
    run(args.folds_csv)
