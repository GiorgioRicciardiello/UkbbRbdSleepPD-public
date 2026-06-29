"""
Figures for the LR-MDS analysis.

All figures use a consistent style. Saved as high-resolution PNGs.

Figures produced:
1. lr_profile        — LR+ vs z-score threshold with Heinzel benchmarks
2. sex_lr_comparison — LR+/LR- CIs for overall / female / male
3. fagan_nomogram    — Pre-test → post-test probability via LR+
4. posteriors        — Posterior probability density: cases vs controls (C1 and C2)
5. empirical_vs_published — Empirical vs Heinzel prodromal marker LRs
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library.lr_analysis.config import HEINZEL_LRS, PRODROMAL_LABELS
from library.lr_analysis.lr_metrics import EmpiricalMarkerLR, LRResult, LogisticORResult
from library.lr_analysis.risk_group_lr import StratumLR

# ── Style constants ───────────────────────────────────────────────────────────
_FIG_DPI = 300
_FONT_SIZE = 9
_COLOR_CASE = "#C0392B"     # red for cases
_COLOR_CTRL = "#2980B9"     # blue for controls
_COLOR_OVERALL = "#2C3E50"  # dark for overall
_COLOR_MALE = "#1ABC9C"
_COLOR_FEMALE = "#E67E22"
_COLOR_LR_POS = "#2980B9"   # blue for LR+
_COLOR_LR_NEG = "#8E44AD"   # purple for LR-
_COLOR_OR = "#C0392B"       # red for OR

plt.rcParams.update({
    "font.size": _FONT_SIZE,
    "axes.labelsize": _FONT_SIZE,
    "axes.titlesize": _FONT_SIZE + 1,
    "xtick.labelsize": _FONT_SIZE - 1,
    "ytick.labelsize": _FONT_SIZE - 1,
    "legend.fontsize": _FONT_SIZE - 1,
    "figure.dpi": _FIG_DPI,
})


# ── 1. LR profile ─────────────────────────────────────────────────────────────

def plot_lr_profile(
    lr_profile: pd.DataFrame,
    youden_threshold: float,
    out_path: Path,
) -> None:
    """LR+ at each z-score threshold with Heinzel benchmarks.

    Horizontal reference lines at:
    - PSG-proven RBD (LR+ = 130)

    Parameters
    ----------
    lr_profile : pd.DataFrame
        Output of ``compute_lr_profile``.
    youden_threshold : float
        Optimal threshold (marked with a vertical dashed line).
    out_path : Path
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, metric, ylabel in [
        (axes[0], "lr_pos", "LR+"),
        (axes[1], "lr_neg", "LR−"),
    ]:
        lci_col = f"{metric[3:]}_pos_lci" if metric == "lr_pos" else "lr_neg_lci"
        uci_col = f"{metric[3:]}_pos_uci" if metric == "lr_pos" else "lr_neg_uci"
        # Recompute correct col names
        lci_col = "lr_pos_lci" if metric == "lr_pos" else "lr_neg_lci"
        uci_col = "lr_pos_uci" if metric == "lr_pos" else "lr_neg_uci"

        x = lr_profile["threshold"].values
        y = lr_profile[metric].values
        y_lo = lr_profile[lci_col].values
        y_hi = lr_profile[uci_col].values

        ax.plot(x, y, "o-", color=_COLOR_OVERALL, lw=1.5, ms=4, label="Actigraphy RBD")
        ax.fill_between(x, y_lo, y_hi, alpha=0.2, color=_COLOR_OVERALL)

        # Heinzel benchmarks (LR+ panel only)
        if metric == "lr_pos":
            ax.axhline(130, ls=":", color="gray", lw=1.0,
                       label="PSG-proven RBD (LR+=130)")

        # Youden threshold
        ax.axvline(youden_threshold, ls="--", color="black", lw=0.9, alpha=0.7,
                   label=f"Youden threshold (z={youden_threshold:.2f})")

        ax.set_xlabel("RBD z-score threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs z-score threshold")
        if metric == "lr_pos":
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ── 1b. Combined LR+ and LR- forest plot ──────────────────────────────────────

def plot_combined_lr_forest(
    risk_group_lrs_3g: list[StratumLR],
    or_unadj: LogisticORResult,
    or_adj: LogisticORResult,
    empirical_lrs: list[EmpiricalMarkerLR],
    out_path: Path,
) -> None:
    """Combined forest plot: LR+ and LR- side-by-side with OR.

    Shows:
    - RBD Risk Group (3g): High, Intermediate, Low
    - Actigraphy z-score Logistic OR: Unadjusted, Adjusted
    - Prodromal Markers: LR+/LR- for empirical markers

    Layout:
    - Single axis with two x-regions: LR+ (left) and LR- (right)
    - Log10 scale with offset positioning
    - Blue circles for LR+
    - Purple circles for LR-
    - Red squares for OR (LR+ side only; LR- side shows N/A)

    Parameters
    ----------
    risk_group_lrs_3g : list[StratumLR]
        Risk group LRs (3-group scheme) ordered [Low, Intermediate, High].
    or_unadj : LogisticORResult
        Unadjusted logistic OR per 1 SD increase in RBD z-score.
    or_adj : LogisticORResult
        Adjusted logistic OR per 1 SD increase in RBD z-score.
    empirical_lrs : list[EmpiricalMarkerLR]
        Empirically computed LRs for prodromal markers.
    out_path : Path
        Output file path.
    """
    # ── Data assembly ────────────────────────────────────────────────────────
    rows = []

    # 1. RBD Risk Groups (3g) — reverse order so High is at top
    for stratum in reversed(risk_group_lrs_3g):
        rows.append({
            "section": "RBD Risk Group",
            "label": f"{stratum.category} Risk",
            "type": "lr",  # LR+/LR-
            "lr_pos": stratum.lr,
            "lr_pos_lci": stratum.lr_lci,
            "lr_pos_uci": stratum.lr_uci,
            "lr_neg": np.nan,  # Risk groups only show LR
            "lr_neg_lci": np.nan,
            "lr_neg_uci": np.nan,
            "or": np.nan,
            "or_lci": np.nan,
            "or_uci": np.nan,
        })

    # 2. Actigraphy z-score — Logistic OR
    rows.append({
        "section": "Actigraphy z-score",
        "label": "Unadjusted OR\n(per 1 SD z-score)",
        "type": "or",  # Only OR, not LR+/LR-
        "lr_pos": np.nan,
        "lr_pos_lci": np.nan,
        "lr_pos_uci": np.nan,
        "lr_neg": np.nan,
        "lr_neg_lci": np.nan,
        "lr_neg_uci": np.nan,
        "or": or_unadj.or_estimate,
        "or_lci": or_unadj.or_lci,
        "or_uci": or_unadj.or_uci,
    })
    rows.append({
        "section": "Actigraphy z-score",
        "label": "Adjusted OR\n(per 1 SD z-score; age, sex, BMI)",
        "type": "or",
        "lr_pos": np.nan,
        "lr_pos_lci": np.nan,
        "lr_pos_uci": np.nan,
        "lr_neg": np.nan,
        "lr_neg_lci": np.nan,
        "lr_neg_uci": np.nan,
        "or": or_adj.or_estimate,
        "or_lci": or_adj.or_lci,
        "or_uci": or_adj.or_uci,
    })

    # 3. Prodromal Markers
    for marker in empirical_lrs:
        rows.append({
            "section": "Prodromal Marker",
            "label": marker.label,
            "type": "lr",
            "lr_pos": marker.lr_pos,
            "lr_pos_lci": marker.lr_pos_ci[0],
            "lr_pos_uci": marker.lr_pos_ci[1],
            "lr_neg": marker.lr_neg,
            "lr_neg_lci": marker.lr_neg_ci[0],
            "lr_neg_uci": marker.lr_neg_ci[1],
            "or": np.nan,
            "or_lci": np.nan,
            "or_uci": np.nan,
        })

    df = pd.DataFrame(rows)

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), dpi=_FIG_DPI)

    # Y-positions (reversed so first row is at top)
    n_rows = len(df)
    y_positions = np.arange(n_rows - 1, -1, -1) * 0.8

    # Log10 scale regions
    # Left side: LR+ region (log10 scale centered at 0)
    # Right side: LR- region (log10 scale centered at offset)
    offset_x_pos = 1.5  # Center of LR+ region
    offset_x_neg = 5.0  # Center of LR- region

    # ── Background section highlights ───────────────────────────────────────
    # Define pastel colors for each section
    pastel_rbd = "#E8F4F8"       # Light blue
    pastel_continuous = "#F0E8F8"  # Light lavender
    pastel_prodromal = "#F8F4E8"   # Light cream

    # Identify section boundaries
    sections_with_indices = []
    current_section = None
    start_idx = 0
    for idx, section in enumerate(df["section"]):
        if section != current_section:
            if current_section is not None:
                sections_with_indices.append((current_section, start_idx, idx - 1))
            current_section = section
            start_idx = idx
    if current_section is not None:
        sections_with_indices.append((current_section, start_idx, len(df) - 1))

    # Draw background rectangles
    section_colors = {
        "RBD Risk Group": pastel_rbd,
        "Actigraphy z-score": pastel_continuous,
        "Prodromal Marker": pastel_prodromal,
    }

    for section_name, start_idx, end_idx in sections_with_indices:
        color = section_colors.get(section_name, "white")
        y_min = y_positions[end_idx] - 0.4
        y_max = y_positions[start_idx] + 0.4
        ax.axhspan(y_min, y_max, color=color, alpha=0.3, zorder=0)

    # Reference lines at LR = 1 (log10(1) = 0)
    ax.axvline(offset_x_pos + 0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(offset_x_neg + 0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)

    # Vertical divider between LR+ and LR- regions
    ax.axvline((offset_x_pos + offset_x_neg) / 2, color="black", linestyle="-",
               linewidth=1.2, alpha=0.3)

    # ── Plot data ────────────────────────────────────────────────────────────
    for idx, (i, row) in enumerate(df.iterrows()):
        y = y_positions[idx]

        # LR+ (left side)
        if row["type"] == "lr" and np.isfinite(row["lr_pos"]):
            x_pos = offset_x_pos + np.log10(row["lr_pos"])
            x_pos_lo = offset_x_pos + np.log10(row["lr_pos_lci"])
            x_pos_hi = offset_x_pos + np.log10(row["lr_pos_uci"])

            ax.plot([x_pos_lo, x_pos_hi], [y, y], color=_COLOR_LR_POS, linewidth=2.5, zorder=2)
            ax.scatter(x_pos, y, marker="o", s=100, color=_COLOR_LR_POS,
                      edgecolor="white", linewidth=1.2, zorder=3)

            # Annotation
            ax.text(x_pos_hi + 0.15, y,
                   f"{row['lr_pos']:.2f}\n[{row['lr_pos_lci']:.2f}–{row['lr_pos_uci']:.2f}]",
                   fontsize=_FONT_SIZE - 1, ha="left", va="center", family="monospace")

        elif row["type"] == "or" and np.isfinite(row["or"]):
            x_or = offset_x_pos + np.log10(row["or"])
            x_or_lo = offset_x_pos + np.log10(row["or_lci"])
            x_or_hi = offset_x_pos + np.log10(row["or_uci"])

            ax.plot([x_or_lo, x_or_hi], [y, y], color=_COLOR_OR, linewidth=2.5, zorder=2)
            ax.scatter(x_or, y, marker="s", s=120, color=_COLOR_OR,
                      edgecolor="white", linewidth=1.2, zorder=3)

            # Annotation
            ax.text(x_or_hi + 0.15, y,
                   f"{row['or']:.2f}\n[{row['or_lci']:.2f}–{row['or_uci']:.2f}]",
                   fontsize=_FONT_SIZE - 1, ha="left", va="center", family="monospace")

        # LR- (right side)
        if row["type"] == "lr" and np.isfinite(row["lr_neg"]):
            x_neg = offset_x_neg + np.log10(row["lr_neg"])
            x_neg_lo = offset_x_neg + np.log10(row["lr_neg_lci"])
            x_neg_hi = offset_x_neg + np.log10(row["lr_neg_uci"])

            ax.plot([x_neg_lo, x_neg_hi], [y, y], color=_COLOR_LR_NEG, linewidth=2.5, zorder=2)
            ax.scatter(x_neg, y, marker="o", s=100, color=_COLOR_LR_NEG,
                      edgecolor="white", linewidth=1.2, zorder=3)

            # Annotation
            ax.text(x_neg_hi + 0.15, y,
                   f"{row['lr_neg']:.3f}\n[{row['lr_neg_lci']:.3f}–{row['lr_neg_uci']:.3f}]",
                   fontsize=_FONT_SIZE - 1, ha="left", va="center", family="monospace")

        elif row["type"] == "or":
            # OR rows: show N/A on LR- side
            ax.text(offset_x_neg, y, "N/A", fontsize=_FONT_SIZE - 1,
                   ha="center", va="center", style="italic", color="gray")

    # ── Add section separators ──────────────────────────────────────────────
    section_breaks = []
    for i, (label, section) in enumerate(zip(df["label"], df["section"])):
        if i > 0 and df["section"].iloc[i-1] != section:
            section_breaks.append(y_positions[i] + 0.4)

    for y_break in section_breaks:
        ax.axhline(y=y_break, color="lightgray", linestyle="-", linewidth=0.8, alpha=0.4)

    # ── Axes setup ───────────────────────────────────────────────────────────
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"].values, fontsize=_FONT_SIZE)

    # X-axis: combined log scale with labeled regions
    # LR+ region ticks
    lr_pos_ticks = np.array([0.5, 1, 2, 5, 10])
    lr_pos_log_ticks = offset_x_pos + np.log10(lr_pos_ticks)

    # LR- region ticks
    lr_neg_ticks = np.array([0.05, 0.1, 0.2, 0.5, 1])
    lr_neg_log_ticks = offset_x_neg + np.log10(lr_neg_ticks)

    ax.set_xticks(np.concatenate([lr_pos_log_ticks, lr_neg_log_ticks]))
    ax.set_xticklabels(
        [f"{v:.2f}" for v in lr_pos_ticks] + [f"{v:.2f}" for v in lr_neg_ticks],
        fontsize=_FONT_SIZE - 1
    )

    # Set x-axis limits
    ax.set_xlim(offset_x_pos - 1.2, offset_x_neg + 1.5)

    # Labels
    ax.set_xlabel("Likelihood Ratio (log10 scale)", fontsize=_FONT_SIZE + 1, fontweight="bold")

    # Add region labels above axis
    ax.text(offset_x_pos, y_positions[-1] + 1.2, "LR+", fontsize=_FONT_SIZE + 1,
           ha="center", fontweight="bold")
    ax.text(offset_x_neg, y_positions[-1] + 1.2, "LR−", fontsize=_FONT_SIZE + 1,
           ha="center", fontweight="bold")

    # Remove grid and spines
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)

    # Title and legend
    fig.suptitle(
        "Likelihood Ratios: RBD Risk Groups and Prodromal Markers",
        fontsize=_FONT_SIZE + 2, fontweight="bold", y=0.98
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLOR_LR_POS,
              markersize=6, label="LR+", linewidth=2.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLOR_LR_NEG,
              markersize=6, label="LR−", linewidth=2.5),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_COLOR_OR,
              markersize=6, label="OR per 1 SD", linewidth=2.5),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="LR = 1"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=5,
              fontsize=_FONT_SIZE - 0.5, frameon=True, fancybox=False,
              edgecolor="#CCCCCC", framealpha=0.90, bbox_to_anchor=(0.5, 0.93))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor="white")
    print(f"[OK] Combined LR forest plot saved: {out_path}")
    plt.close(fig)


# ── 2. Sex-stratified LR comparison ──────────────────────────────────────────

def plot_sex_stratified_lr(
    overall: LRResult,
    female: LRResult,
    male: LRResult,
    out_path: Path,
) -> None:
    """Bar chart of LR+/LR− by stratum with 95% CI error bars."""
    strata = ["Overall", "Female", "Male"]
    colors = [_COLOR_OVERALL, _COLOR_FEMALE, _COLOR_MALE]
    results = [overall, female, male]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, attr, lci_attr, uci_attr, title in [
        (axes[0], "lr_pos", "lr_pos_ci", "lr_pos_ci", "LR+"),
        (axes[1], "lr_neg", "lr_neg_ci", "lr_neg_ci", "LR−"),
    ]:
        vals, lo_errs, hi_errs = [], [], []
        for r in results:
            v = getattr(r, attr)
            ci = getattr(r, lci_attr)
            vals.append(v)
            lo_errs.append(max(0.0, v - ci[0]))
            hi_errs.append(max(0.0, ci[1] - v))

        x = np.arange(len(strata))
        bars = ax.bar(x, vals, color=colors, alpha=0.8, width=0.5)
        ax.errorbar(
            x, vals,
            yerr=[lo_errs, hi_errs],
            fmt="none", color="black", capsize=4, lw=1.2,
        )

        # Reference line at 1 (no information)
        ax.axhline(1.0, ls="--", color="gray", lw=0.9, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(strata)
        ax.set_ylabel(title)
        ax.set_title(f"{title} by sex stratum")
        ax.grid(True, axis="y", alpha=0.3, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ── 3. Fagan nomogram ─────────────────────────────────────────────────────────

def _prob_to_logodds(p: float) -> float:
    return np.log10(p / (1.0 - p))


def _logodds_to_prob(lo: float) -> float:
    return 10**lo / (1.0 + 10**lo)


def plot_fagan_nomogram(
    lr_pos: float,
    lr_neg: float,
    prior_probs: list[float],
    out_path: Path,
) -> None:
    """Simplified Fagan nomogram for LR+.

    Shows how the post-test probability changes from each prior probability
    given LR+ and LR-.

    Parameters
    ----------
    lr_pos : float
    lr_neg : float
    prior_probs : list[float]
        Representative prior probabilities to annotate (e.g. [0.002, 0.008]).
    out_path : Path
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    prior_range = np.linspace(0.0005, 0.10, 300)
    post_pos = [_logodds_to_prob(_prob_to_logodds(p) + np.log10(lr_pos))
                for p in prior_range]
    post_neg = [_logodds_to_prob(_prob_to_logodds(p) + np.log10(lr_neg))
                for p in prior_range]

    ax.plot(prior_range * 100, [p * 100 for p in post_pos],
            color=_COLOR_CASE, lw=1.8, label=f"Post-test if actigraphy+ (LR+={lr_pos:.1f})")
    ax.plot(prior_range * 100, [p * 100 for p in post_neg],
            color=_COLOR_CTRL, lw=1.8, ls="--",
            label=f"Post-test if actigraphy− (LR−={lr_neg:.2f})")
    ax.plot(prior_range * 100, prior_range * 100,
            color="gray", lw=0.8, ls=":", label="No change (LR=1)")

    # Annotate representative priors
    for prior in prior_probs:
        post = _logodds_to_prob(_prob_to_logodds(prior) + np.log10(lr_pos))
        ax.annotate(
            f"{prior*100:.1f}% → {post*100:.1f}%",
            xy=(prior * 100, post * 100),
            xytext=(prior * 100 + 0.5, post * 100 + 0.5),
            fontsize=7, color=_COLOR_CASE,
        )

    ax.set_xlabel("Pre-test probability (%)")
    ax.set_ylabel("Post-test probability (%)")
    ax.set_title("Fagan nomogram — actigraphy RBD score")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ── 4. Posterior probability distributions ────────────────────────────────────

def plot_posteriors(
    posterior_c2: pd.Series,
    posterior_c1: pd.Series,
    is_case: pd.Series,
    out_path: Path,
) -> None:
    """Density plot of posterior probabilities — cases vs controls, C1 and C2.

    Parameters
    ----------
    posterior_c2 : pd.Series
        Per-subject posteriors from C2 (Hybrid).
    posterior_c1 : pd.Series
        Per-subject posteriors from C1 (Empirical).
    is_case : pd.Series[bool]
    out_path : Path
    """
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    labels = ["C2: Hybrid (Heinzel LRs)", "C1: Empirical (UKBB LRs)"]

    for ax, posterior, label in zip(axes, [posterior_c2, posterior_c1], labels):
        for mask, group_label, color in [
            (is_case, "Incident PD", _COLOR_CASE),
            (~is_case, "Controls", _COLOR_CTRL),
        ]:
            vals = posterior[mask].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            # Clip to [0, 0.3] for readability (most posteriors are small)
            vals_clipped = np.clip(vals, 0, 0.30)
            xgrid = np.linspace(0, 0.30, 300)
            try:
                kde = gaussian_kde(vals_clipped, bw_method=0.3)
                ax.plot(xgrid * 100, kde(xgrid), color=color, lw=1.5,
                        label=f"{group_label} (N={len(vals):,})")
                ax.fill_between(xgrid * 100, kde(xgrid), alpha=0.15, color=color)
            except Exception:
                ax.hist(vals_clipped * 100, bins=30, density=True,
                        color=color, alpha=0.5, label=group_label)

        ax.set_xlabel("Posterior probability (%)")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ── 5. Empirical vs published prodromal LRs ───────────────────────────────────

def plot_empirical_vs_published(
    empirical_lrs: list[EmpiricalMarkerLR],
    out_path: Path,
) -> None:
    """Compare empirically-derived LRs from UKBB to published Heinzel values.

    Shows LR+ for each viable prodromal marker as paired bars (empirical vs
    Heinzel) with 95% CI for the empirical estimate.

    Parameters
    ----------
    empirical_lrs : list[EmpiricalMarkerLR]
    out_path : Path
    """
    from library.lr_analysis.config import PRODROMAL_COL_TO_HEINZEL

    markers = []
    emp_pos, emp_pos_lo, emp_pos_hi = [], [], []
    hein_pos = []

    for e in empirical_lrs:
        heinzel_key = PRODROMAL_COL_TO_HEINZEL.get(e.col)
        if heinzel_key is None:
            continue
        h = HEINZEL_LRS.get(heinzel_key)
        if h is None:
            continue
        markers.append(PRODROMAL_LABELS.get(e.col, e.col))
        emp_pos.append(e.lr_pos)
        emp_pos_lo.append(e.lr_pos - e.lr_pos_ci[0] if np.isfinite(e.lr_pos_ci[0]) else 0)
        emp_pos_hi.append(e.lr_pos_ci[1] - e.lr_pos if np.isfinite(e.lr_pos_ci[1]) else 0)
        hein_pos.append(h.lr_pos)

    if not markers:
        return

    n = len(markers)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), 4))
    ax.bar(x - width / 2, emp_pos, width, label="Empirical (UKBB)",
           color=_COLOR_OVERALL, alpha=0.8)
    ax.errorbar(
        x - width / 2, emp_pos,
        yerr=[emp_pos_lo, emp_pos_hi],
        fmt="none", color="black", capsize=4, lw=1.2,
    )
    ax.bar(x + width / 2, hein_pos, width, label="Published (Heinzel 2019)",
           color="lightgray", edgecolor="gray", alpha=0.9)

    ax.axhline(1.0, ls="--", color="gray", lw=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(markers, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("LR+")
    ax.set_title("Empirical vs published LR+ for prodromal markers")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
