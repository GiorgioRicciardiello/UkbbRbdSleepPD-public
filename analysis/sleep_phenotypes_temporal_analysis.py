"""
Temporal Validation: RBD Score Reliability Across Measurement Gaps
==================================================================
Evaluates whether the temporal gap between actigraphy-derived RBD scores
and sleep questionnaire assessments modulates prediction reliability.

Research questions
------------------
1. **Temporal landscape** — What is the distribution of measurement gaps?
   Is the gap confounded with phenotype response?
2. **Data-driven temporal windows** — Does effect size (Cohen's d) degrade
   as the actigraphy-questionnaire gap increases? Quantile-binned analysis.
3. **Continuous modulation** — Is RBD score correlated with temporal gap?
   Does the gap modulate the score-phenotype relationship?
4. **PD outcome** — Does RBD score predict PD? Does temporal gap modulate
   the association?

All analysis at **subject level** (one row per eid).
Seed: 42 (all stochastic operations).
"""

import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde, spearmanr

from config.config import config
from library.risk.risk_helpers import get_clean_risk_data

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Configuration ────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/sleep_phenotypes_temporal")
FIG_DPI = 300
SEED = 42

# Column names
RBD_COL = "abk_rbd_score_mean"
DREAM_COL = "cov_dream_enactment_freq_30557"
VIOLENT_COL = "cov_violent_sleep_freq_30558"
QUEST_DATE_COL = "cov_sleep_quest_complete_32122"
ACTIG_DATE_COL = "wear_time_start"
OUTCOME_EVENT_COL = "outcome_1a_pd_only__surv_event"
OUTCOME_DAYS_COL = "outcome_1a_pd_only__surv_days"

PHENOTYPE_CONFIG: Dict[str, Dict] = {
    "dream_enactment": {
        "raw_col": DREAM_COL,
        "binary_col": "dream_enactment_binary",
        "label": "Dream Enactment",
    },
    "violent_sleep": {
        "raw_col": VIOLENT_COL,
        "binary_col": "violent_sleep_binary",
        "label": "Violent Sleep",
    },
}

# Palette (colorblind-safe)
C_NO = "#4C72B0"
C_YES = "#DD8452"
C_PD = "#C44E52"
C_NOPD = "#55A868"
C_PHENO = {"dream_enactment": "#4C72B0", "violent_sleep": "#DD8452"}


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD) between two groups."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return np.nan
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (group_b.mean() - group_a.mean()) / pooled_std


def classify_effect(d: float) -> str:
    """Cohen's d effect strength label."""
    if np.isnan(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.2:
        return "Negligible"
    if ad < 0.5:
        return "Small"
    if ad < 0.8:
        return "Medium"
    return "Large"


def compute_mann_whitney_effect(
    vals_no: np.ndarray,
    vals_yes: np.ndarray,
) -> Dict:
    """Mann-Whitney U + Cohen's d for two groups."""
    n_no, n_yes = len(vals_no), len(vals_yes)
    if n_no < 3 or n_yes < 3:
        return {
            "u_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan,
            "effect": "Insufficient N", "n_no": n_no, "n_yes": n_yes,
            "mean_no": np.nan, "mean_yes": np.nan,
        }
    u_stat, p_value = stats.mannwhitneyu(vals_no, vals_yes, alternative="two-sided")
    d = compute_cohens_d(vals_no, vals_yes)
    return {
        "u_stat": u_stat, "p_value": p_value, "cohens_d": d,
        "effect": classify_effect(d),
        "n_no": n_no, "n_yes": n_yes,
        "mean_no": vals_no.mean(), "mean_yes": vals_yes.mean(),
        "std_no": vals_no.std(ddof=1), "std_yes": vals_yes.std(ddof=1),
    }


def bootstrap_cohens_d_ci(
    vals_no: np.ndarray,
    vals_yes: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = SEED,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        b_no = rng.choice(vals_no, size=len(vals_no), replace=True)
        b_yes = rng.choice(vals_yes, size=len(vals_yes), replace=True)
        ds[i] = compute_cohens_d(b_no, b_yes)
    alpha = (1 - ci) / 2
    return float(np.nanpercentile(ds, alpha * 100)), float(np.nanpercentile(ds, (1 - alpha) * 100))


def _sig_stars(p: float) -> str:
    """Significance stars."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def bin_by_quantiles(
    series: pd.Series,
    n_bins: int,
    prefix: str = "Q",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Bin a numeric series into equal-frequency quantile bins.

    Returns
    -------
    bin_labels : pd.Series
        Categorical labels (e.g. Q1, Q2, ...)
    bin_info : pd.DataFrame
        Rows = bins, cols = label, lo, hi, midpoint, n
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = series.quantile(quantiles).values
    # Ensure unique edges
    edges = np.unique(edges)
    actual_bins = len(edges) - 1

    labels = [f"{prefix}{i + 1}" for i in range(actual_bins)]
    bin_labels = pd.cut(series, bins=edges, labels=labels, include_lowest=True)

    info_rows = []
    for i in range(actual_bins):
        info_rows.append({
            "label": labels[i],
            "lo": edges[i],
            "hi": edges[i + 1],
            "midpoint": (edges[i] + edges[i + 1]) / 2,
        })
    bin_info = pd.DataFrame(info_rows)
    return bin_labels, bin_info


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def annotate_box(
    ax: plt.Axes,
    position: float,
    values: np.ndarray,
    color: str = "black",
    fontsize: int = 8,
) -> None:
    """Annotate a boxplot position with N, mean +/- std."""
    if len(values) == 0:
        return
    q3 = np.percentile(values, 75)
    txt = f"N={len(values):,}\n$\\mu$={values.mean():.3f}\n$\\sigma$={values.std(ddof=1):.3f}"
    ax.text(
        position, q3 + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
        txt, ha="center", va="bottom", fontsize=fontsize, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, edgecolor="none"),
    )


def _effect_ref_lines(ax: plt.Axes) -> None:
    """Add horizontal reference lines for Cohen's d thresholds."""
    for val, lbl in [(0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
        ax.axhline(val, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[1], val, f" {lbl}", va="center", fontsize=7, color="gray")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: TEMPORAL LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════════

def plot_temporal_gap_distribution(
    delta_years: np.ndarray,
    output_path: Path,
) -> None:
    """Histogram of signed temporal gap with KDE overlay and summary stats."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    ax.hist(delta_years, bins=80, density=True, alpha=0.6, color="#7FB3D8",
            edgecolor="white", linewidth=0.3, label="Histogram")

    # KDE overlay
    kde = gaussian_kde(delta_years[~np.isnan(delta_years)], bw_method=0.15)
    x_grid = np.linspace(delta_years.min() - 0.5, delta_years.max() + 0.5, 500)
    ax.plot(x_grid, kde(x_grid), color="#2C5F8A", lw=2, label="KDE")

    # Reference lines
    mean_v = np.nanmean(delta_years)
    median_v = np.nanmedian(delta_years)
    q1 = np.nanpercentile(delta_years, 25)
    q3 = np.nanpercentile(delta_years, 75)
    ax.axvline(mean_v, color="#C44E52", ls="-", lw=2, label=f"Mean = {mean_v:.2f}y")
    ax.axvline(median_v, color="#4C72B0", ls="--", lw=2, label=f"Median = {median_v:.2f}y")
    ax.axvline(q1, color="gray", ls=":", lw=1.5, label=f"Q1 = {q1:.2f}y")
    ax.axvline(q3, color="gray", ls=":", lw=1.5, label=f"Q3 = {q3:.2f}y")

    # Zero line
    ax.axvline(0, color="black", ls="-", lw=1, alpha=0.3)

    # Summary box
    iqr = q3 - q1
    txt = (f"N = {len(delta_years):,}\n"
           f"Mean = {mean_v:.2f}y\nMedian = {median_v:.2f}y\n"
           f"SD = {np.nanstd(delta_years):.2f}y\nIQR = {iqr:.2f}y\n"
           f"Range = [{np.nanmin(delta_years):.1f}, {np.nanmax(delta_years):.1f}]y")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    ax.set_xlabel("Temporal Gap (years): Actigraphy -> Questionnaire", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Actigraphy-Questionnaire Temporal Gap", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2, ls="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def plot_temporal_gap_by_phenotype(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """2x2 panel: temporal gap distribution by binary phenotype response."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, (pname, pcfg) in enumerate(PHENOTYPE_CONFIG.items()):
        ax = axes[idx]
        bcol = pcfg["binary_col"]
        df_plot = df.dropna(subset=[bcol, "delta_years"]).copy()

        vals_no = df_plot.loc[df_plot[bcol] == "No", "delta_years"].values
        vals_yes = df_plot.loc[df_plot[bcol] == "Yes", "delta_years"].values

        data = [vals_no, vals_yes]
        bp = ax.boxplot(data, tick_labels=["No", "Yes"], patch_artist=True, widths=0.5,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="white", markersize=5))
        bp["boxes"][0].set_facecolor(C_NO)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(C_YES)
        bp["boxes"][1].set_alpha(0.7)

        # MW test: is temporal gap confounded with phenotype?
        if len(vals_no) >= 3 and len(vals_yes) >= 3:
            _, p_val = stats.mannwhitneyu(vals_no, vals_yes, alternative="two-sided")
            d = compute_cohens_d(vals_no, vals_yes)
            ax.set_title(f"{pcfg['label']}\nMW p={p_val:.2e}, d={d:.4f} ({classify_effect(d)})",
                         fontsize=10, fontweight="bold")
        else:
            ax.set_title(pcfg["label"], fontsize=10, fontweight="bold")

        # Annotate
        for i, (vals, lbl) in enumerate(zip(data, ["No", "Yes"])):
            ax.text(i + 1, np.percentile(vals, 75) + 0.1,
                    f"N={len(vals):,}\n$\\mu$={vals.mean():.2f}",
                    ha="center", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, edgecolor="none"))

        ax.set_xlabel("Phenotype Response", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Temporal Gap (years)", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2, ls="--")

    fig.suptitle("Temporal Gap by Phenotype Response (Confounding Check)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def export_temporal_descriptive(
    df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Export descriptive statistics of temporal gap, overall and by subgroups."""
    rows = []

    def _row(label: str, vals: np.ndarray) -> Dict:
        return {
            "group": label, "N": len(vals),
            "mean_years": np.nanmean(vals), "median_years": np.nanmedian(vals),
            "sd_years": np.nanstd(vals, ddof=1), "iqr_years": np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25),
            "min_years": np.nanmin(vals), "max_years": np.nanmax(vals),
            "pct_total": len(vals) / len(df) * 100,
        }

    all_delta = df["delta_years"].dropna().values
    rows.append(_row("Overall", all_delta))

    # By measurement order
    for order in ["RBD first", "Questionnaire first", "Same day"]:
        mask = df["measurement_order"] == order
        if mask.sum() > 0:
            rows.append(_row(f"Order: {order}", df.loc[mask, "delta_years"].values))

    # By temporal quartile
    if "temporal_quartile" in df.columns:
        for q in sorted(df["temporal_quartile"].dropna().unique()):
            mask = df["temporal_quartile"] == q
            rows.append(_row(f"Quartile: {q}", df.loc[mask, "delta_years"].values))

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False, float_format="%.4f")
    print(f"  [Tab] {output_path.name}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA-DRIVEN TEMPORAL WINDOWS
# ═══════════════════════════════════════════════════════════════════════════

def compute_effect_by_bins(
    df: pd.DataFrame,
    bin_col: str,
    bin_info: pd.DataFrame,
    phenotype_configs: Dict,
    rbd_col: str,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Compute MW effect + bootstrap CI for each bin x phenotype."""
    rows = []
    for _, brow in bin_info.iterrows():
        label = brow["label"]
        mask = df[bin_col] == label
        df_bin = df.loc[mask]
        n_bin = mask.sum()

        for pname, pcfg in phenotype_configs.items():
            bcol = pcfg["binary_col"]
            df_clean = df_bin.dropna(subset=[bcol, rbd_col])
            vals_no = df_clean.loc[df_clean[bcol] == "No", rbd_col].values
            vals_yes = df_clean.loc[df_clean[bcol] == "Yes", rbd_col].values

            eff = compute_mann_whitney_effect(vals_no, vals_yes)

            ci_lo, ci_hi = np.nan, np.nan
            if len(vals_no) >= 10 and len(vals_yes) >= 10:
                ci_lo, ci_hi = bootstrap_cohens_d_ci(vals_no, vals_yes, n_boot=n_boot)

            rows.append({
                "bin": label,
                "bin_lo": brow["lo"], "bin_hi": brow["hi"],
                "bin_midpoint": brow["midpoint"],
                "phenotype": pname,
                "phenotype_label": pcfg["label"],
                "N": len(df_clean),
                "N_no": eff["n_no"], "N_yes": eff["n_yes"],
                "mean_rbd_no": eff["mean_no"], "mean_rbd_yes": eff["mean_yes"],
                "cohens_d": eff["cohens_d"],
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "p_value": eff["p_value"],
                "effect": eff["effect"],
            })

    return pd.DataFrame(rows)


def plot_effect_by_quartile(
    df_eff: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart: Cohen's d by temporal quartile, grouped by phenotype."""
    phenotypes = df_eff["phenotype"].unique()
    bins = df_eff["bin"].unique()
    n_bins = len(bins)
    n_pheno = len(phenotypes)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(n_bins)

    for i, pname in enumerate(phenotypes):
        sub = df_eff.loc[df_eff["phenotype"] == pname].sort_values("bin")
        offset = (i - (n_pheno - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, sub["cohens_d"].values, bar_width,
            label=sub["phenotype_label"].iloc[0],
            color=C_PHENO.get(pname, "#999999"), alpha=0.8, edgecolor="black", linewidth=0.5,
        )
        # Error bars (bootstrap CI)
        if sub["ci_lo"].notna().any():
            yerr_lo = sub["cohens_d"].values - sub["ci_lo"].values
            yerr_hi = sub["ci_hi"].values - sub["cohens_d"].values
            ax.errorbar(
                x + offset, sub["cohens_d"].values,
                yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=3, lw=1,
            )
        # N annotations
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.text(x[j] + offset, row["cohens_d"] + 0.015,
                    f"N={row['N']:,}", ha="center", va="bottom", fontsize=7, rotation=45)

    # Build x-tick labels with bin ranges
    range_labels = []
    for b in bins:
        sub = df_eff.loc[df_eff["bin"] == b].iloc[0]
        range_labels.append(f"{b}\n[{sub['bin_lo']:.1f}, {sub['bin_hi']:.1f}]y")

    ax.set_xticks(x)
    ax.set_xticklabels(range_labels, fontsize=9)
    ax.set_xlabel("Temporal Gap Quartile (absolute years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cohen's d (RBD Yes - No)", fontsize=11, fontweight="bold")
    ax.set_title("Stratification Strength by Temporal Gap Quartile\n(Does effect degrade with longer gap?)",
                 fontsize=12, fontweight="bold")

    # Reference lines
    for val, lbl in [(0.2, "Small d=0.2"), (0.5, "Medium d=0.5")]:
        ax.axhline(val, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, 0.2, "Small", ha="right", fontsize=7, color="gray")
    ax.text(ax.get_xlim()[1] * 0.98, 0.5, "Medium", ha="right", fontsize=7, color="gray")

    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2, ls="--")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def plot_effect_trend_deciles(
    df_eff: pd.DataFrame,
    output_path: Path,
) -> None:
    """Line plot: Cohen's d across decile midpoints with CI band + Spearman."""
    phenotypes = df_eff["phenotype"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    for pname in phenotypes:
        sub = df_eff.loc[df_eff["phenotype"] == pname].sort_values("bin_midpoint")
        x = sub["bin_midpoint"].values
        y = sub["cohens_d"].values
        label_str = sub["phenotype_label"].iloc[0]

        color = C_PHENO.get(pname, "#999999")
        ax.plot(x, y, "o-", color=color, lw=2, markersize=6, label=label_str)

        # CI band
        if sub["ci_lo"].notna().all():
            ax.fill_between(x, sub["ci_lo"].values, sub["ci_hi"].values,
                            color=color, alpha=0.15)

        # Spearman correlation
        valid = ~np.isnan(y)
        if valid.sum() >= 4:
            rho, p = spearmanr(x[valid], y[valid])
            ax.text(0.98, 0.02 + 0.06 * list(phenotypes).index(pname),
                    f"{label_str}: rho={rho:.3f}, p={p:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    color=color, fontweight="bold",
                    bbox=dict(fc="white", alpha=0.8, edgecolor="none"))

    # Reference lines
    for val in [0.2, 0.5]:
        ax.axhline(val, color="gray", ls="--", lw=0.8, alpha=0.5)

    ax.set_xlabel("Temporal Gap Midpoint (years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cohen's d (RBD Yes - No)", fontsize=11, fontweight="bold")
    ax.set_title("Effect Size Trend Across Temporal Deciles\n(Spearman correlation: gap vs stratification strength)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, ls="--")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: CONTINUOUS TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_scatter_delta_vs_rbd(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """1x2 scatter: delta_years vs RBD score colored by binary phenotype."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    rng = np.random.default_rng(SEED)

    for idx, (pname, pcfg) in enumerate(PHENOTYPE_CONFIG.items()):
        ax = axes[idx]
        bcol = pcfg["binary_col"]
        df_plot = df.dropna(subset=[bcol, RBD_COL, "delta_years"])

        for group, color, zorder in [("No", C_NO, 1), ("Yes", C_YES, 2)]:
            mask = df_plot[bcol] == group
            x = df_plot.loc[mask, "delta_years"].values
            y = df_plot.loc[mask, RBD_COL].values
            alpha_val = 0.03 if len(x) > 5000 else 0.1
            ax.scatter(x, y, s=3, alpha=alpha_val, color=color, zorder=zorder,
                       label=f"{group} (N={len(x):,})", rasterized=True)

            # Trend line (binned means for clarity)
            if len(x) > 100:
                n_trend_bins = 20
                bin_edges = np.linspace(x.min(), x.max(), n_trend_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_means = np.array([
                    y[(x >= bin_edges[i]) & (x < bin_edges[i + 1])].mean()
                    if ((x >= bin_edges[i]) & (x < bin_edges[i + 1])).sum() > 0
                    else np.nan
                    for i in range(n_trend_bins)
                ])
                valid_trend = ~np.isnan(bin_means)
                ax.plot(bin_centers[valid_trend], bin_means[valid_trend],
                        "-", color=color, lw=2.5, alpha=0.9, zorder=3)

            # Spearman
            rho, p = spearmanr(x, y)
            ax.text(0.02, 0.98 - 0.08 * (0 if group == "No" else 1),
                    f"{group}: rho={rho:.3f}, p={p:.2e}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    color=color, fontweight="bold",
                    bbox=dict(fc="white", alpha=0.8, edgecolor="none"))

        ax.set_xlabel("Temporal Gap (years)", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Mean RBD Score", fontsize=10, fontweight="bold")
        ax.set_title(pcfg["label"], fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, markerscale=5)
        ax.grid(alpha=0.2, ls="--")

    fig.suptitle("RBD Score vs Temporal Gap by Phenotype Response",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def plot_rbd_kde_by_quartile(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """2xN panel: KDE of RBD score for Yes/No, one column per quartile."""
    quartiles = sorted(df["temporal_quartile"].dropna().unique())
    n_q = len(quartiles)
    n_pheno = len(PHENOTYPE_CONFIG)

    fig, axes = plt.subplots(n_pheno, n_q, figsize=(4 * n_q, 4 * n_pheno), sharey="row", sharex=True)
    if n_pheno == 1:
        axes = axes[np.newaxis, :]

    for row, (pname, pcfg) in enumerate(PHENOTYPE_CONFIG.items()):
        bcol = pcfg["binary_col"]
        for col, q in enumerate(quartiles):
            ax = axes[row, col]
            df_q = df.loc[(df["temporal_quartile"] == q) & df[bcol].notna()].copy()

            for group, color in [("No", C_NO), ("Yes", C_YES)]:
                vals = df_q.loc[df_q[bcol] == group, RBD_COL].dropna().values
                if len(vals) < 5:
                    continue
                try:
                    kde = gaussian_kde(vals, bw_method=0.3)
                    x_grid = np.linspace(vals.min() - 1, vals.max() + 1, 300)
                    ax.fill_between(x_grid, kde(x_grid), alpha=0.3, color=color)
                    ax.plot(x_grid, kde(x_grid), color=color, lw=1.5,
                            label=f"{group} (N={len(vals):,})")
                    ax.axvline(vals.mean(), color=color, ls="--", lw=1, alpha=0.7)
                except np.linalg.LinAlgError:
                    pass

            if row == 0:
                # Get quartile range
                q_mask = df["temporal_quartile"] == q
                lo = df.loc[q_mask, "abs_delta_years"].min()
                hi = df.loc[q_mask, "abs_delta_years"].max()
                ax.set_title(f"{q}\n[{lo:.1f}-{hi:.1f}y]", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{pcfg['label']}\nDensity", fontsize=9, fontweight="bold")
            if row == n_pheno - 1:
                ax.set_xlabel("RBD Score", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.2, ls="--")

    fig.suptitle("RBD Score Distributions by Phenotype Response Across Temporal Quartiles",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def export_continuous_correlations(
    df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Spearman correlations: delta_years vs RBD score, stratified."""
    rows = []

    def _corr(label: str, x: np.ndarray, y: np.ndarray) -> Dict:
        valid = ~(np.isnan(x) | np.isnan(y))
        x_v, y_v = x[valid], y[valid]
        if len(x_v) < 10:
            return {"group": label, "N": len(x_v), "rho": np.nan, "p_value": np.nan}
        rho, p = spearmanr(x_v, y_v)
        return {"group": label, "N": len(x_v), "rho": rho, "p_value": p}

    delta = df["delta_years"].values
    rbd = df[RBD_COL].values
    rows.append(_corr("Overall", delta, rbd))

    for pname, pcfg in PHENOTYPE_CONFIG.items():
        bcol = pcfg["binary_col"]
        for grp in ["No", "Yes"]:
            mask = df[bcol] == grp
            rows.append(_corr(f"{pcfg['label']} = {grp}", delta[mask.values], rbd[mask.values]))

    # By PD outcome
    for pd_val, pd_lbl in [(0, "No PD"), (1, "PD")]:
        mask = df["pd_outcome"] == pd_val
        rows.append(_corr(f"PD Outcome = {pd_lbl}", delta[mask.values], rbd[mask.values]))

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False, float_format="%.6f")
    print(f"  [Tab] {output_path.name}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: PD OUTCOME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_rbd_by_pd_outcome(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Boxplot + strip: RBD score by PD outcome with MW test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Boxplot
    ax = axes[0]
    no_pd = df.loc[df["pd_outcome"] == 0, RBD_COL].dropna().values
    yes_pd = df.loc[df["pd_outcome"] == 1, RBD_COL].dropna().values

    bp = ax.boxplot([no_pd, yes_pd], tick_labels=["No PD", "PD"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=6))
    bp["boxes"][0].set_facecolor(C_NOPD)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_PD)
    bp["boxes"][1].set_alpha(0.7)

    eff = compute_mann_whitney_effect(no_pd, yes_pd)
    ci_lo, ci_hi = bootstrap_cohens_d_ci(no_pd, yes_pd)
    ax.set_title(
        f"RBD Score by PD Outcome\n"
        f"MW p={eff['p_value']:.2e} {_sig_stars(eff['p_value'])}, "
        f"d={eff['cohens_d']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] ({eff['effect']})",
        fontsize=10, fontweight="bold")

    for i, (vals, lbl) in enumerate(zip([no_pd, yes_pd], ["No PD", "PD"])):
        ax.text(i + 1, np.percentile(vals, 75) + 0.15,
                f"N={len(vals):,}\n$\\mu$={vals.mean():.3f}\n$\\sigma$={vals.std(ddof=1):.3f}",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, edgecolor="none"))

    ax.set_ylabel("Mean RBD Score", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, ls="--")

    # Panel B: KDE
    ax = axes[1]
    for vals, color, lbl in [(no_pd, C_NOPD, f"No PD (N={len(no_pd):,})"),
                              (yes_pd, C_PD, f"PD (N={len(yes_pd):,})")]:
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals, bw_method=0.3)
        x_grid = np.linspace(min(no_pd.min(), yes_pd.min()) - 1,
                             max(no_pd.max(), yes_pd.max()) + 1, 500)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.3, color=color)
        ax.plot(x_grid, kde(x_grid), color=color, lw=2, label=lbl)
        ax.axvline(vals.mean(), color=color, ls="--", lw=1.5, alpha=0.7)

    ax.set_xlabel("Mean RBD Score", fontsize=10, fontweight="bold")
    ax.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax.set_title("RBD Score Distribution by PD Outcome", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, ls="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def plot_temporal_gap_by_pd(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Boxplot: temporal gap by PD outcome."""
    fig, ax = plt.subplots(figsize=(7, 5))

    no_pd = df.loc[df["pd_outcome"] == 0, "delta_years"].dropna().values
    yes_pd = df.loc[df["pd_outcome"] == 1, "delta_years"].dropna().values

    bp = ax.boxplot([no_pd, yes_pd], tick_labels=["No PD", "PD"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=6))
    bp["boxes"][0].set_facecolor(C_NOPD)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_PD)
    bp["boxes"][1].set_alpha(0.7)

    if len(no_pd) >= 3 and len(yes_pd) >= 3:
        _, p_val = stats.mannwhitneyu(no_pd, yes_pd, alternative="two-sided")
        d = compute_cohens_d(no_pd, yes_pd)
        ax.set_title(f"Temporal Gap by PD Outcome\nMW p={p_val:.2e}, d={d:.4f} ({classify_effect(d)})",
                     fontsize=11, fontweight="bold")

    for i, (vals, lbl) in enumerate(zip([no_pd, yes_pd], ["No PD", "PD"])):
        ax.text(i + 1, np.percentile(vals, 75) + 0.1,
                f"N={len(vals):,}\n$\\mu$={vals.mean():.2f}y",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, edgecolor="none"))

    ax.set_ylabel("Temporal Gap (years)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, ls="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def plot_pd_combined_panel(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """2x2 combined panel: RBD-by-PD, KDE-by-PD, gap-by-PD, scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rng = np.random.default_rng(SEED)

    no_pd_rbd = df.loc[df["pd_outcome"] == 0, RBD_COL].dropna().values
    yes_pd_rbd = df.loc[df["pd_outcome"] == 1, RBD_COL].dropna().values
    no_pd_gap = df.loc[df["pd_outcome"] == 0, "delta_years"].dropna().values
    yes_pd_gap = df.loc[df["pd_outcome"] == 1, "delta_years"].dropna().values

    # (0,0) Boxplot RBD by PD
    ax = axes[0, 0]
    bp = ax.boxplot([no_pd_rbd, yes_pd_rbd], tick_labels=["No PD", "PD"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5))
    bp["boxes"][0].set_facecolor(C_NOPD)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_PD)
    bp["boxes"][1].set_alpha(0.7)
    eff = compute_mann_whitney_effect(no_pd_rbd, yes_pd_rbd)
    ax.set_title(f"RBD by PD | d={eff['cohens_d']:.3f}, p={eff['p_value']:.2e}", fontsize=9, fontweight="bold")
    ax.set_ylabel("RBD Score", fontsize=9)
    ax.grid(axis="y", alpha=0.2, ls="--")

    # (0,1) KDE RBD by PD
    ax = axes[0, 1]
    for vals, color, lbl in [(no_pd_rbd, C_NOPD, f"No PD (N={len(no_pd_rbd):,})"),
                              (yes_pd_rbd, C_PD, f"PD (N={len(yes_pd_rbd):,})")]:
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals, bw_method=0.3)
        x_grid = np.linspace(min(no_pd_rbd.min(), yes_pd_rbd.min()) - 1,
                             max(no_pd_rbd.max(), yes_pd_rbd.max()) + 1, 400)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.3, color=color)
        ax.plot(x_grid, kde(x_grid), color=color, lw=1.5, label=lbl)
        ax.axvline(vals.mean(), color=color, ls="--", lw=1, alpha=0.7)
    ax.set_title("RBD Score Density by PD", fontsize=9, fontweight="bold")
    ax.set_xlabel("RBD Score", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2, ls="--")

    # (1,0) Gap by PD
    ax = axes[1, 0]
    bp = ax.boxplot([no_pd_gap, yes_pd_gap], tick_labels=["No PD", "PD"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5))
    bp["boxes"][0].set_facecolor(C_NOPD)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_PD)
    bp["boxes"][1].set_alpha(0.7)
    if len(no_pd_gap) >= 3 and len(yes_pd_gap) >= 3:
        _, p_gap = stats.mannwhitneyu(no_pd_gap, yes_pd_gap, alternative="two-sided")
        d_gap = compute_cohens_d(no_pd_gap, yes_pd_gap)
        ax.set_title(f"Temporal Gap by PD | d={d_gap:.4f}, p={p_gap:.2e}", fontsize=9, fontweight="bold")
    ax.set_ylabel("Gap (years)", fontsize=9)
    ax.grid(axis="y", alpha=0.2, ls="--")

    # (1,1) Scatter: delta vs RBD colored by PD
    ax = axes[1, 1]
    for pd_val, color, lbl in [(0, C_NOPD, "No PD"), (1, C_PD, "PD")]:
        mask = df["pd_outcome"] == pd_val
        x = df.loc[mask, "delta_years"].values
        y = df.loc[mask, RBD_COL].values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        alpha_v = 0.02 if len(x) > 5000 else 0.2
        zord = 1 if pd_val == 0 else 2
        ax.scatter(x, y, s=3 if pd_val == 0 else 8, alpha=alpha_v, color=color,
                   label=f"{lbl} (N={len(x):,})", zorder=zord, rasterized=True)
    ax.set_xlabel("Temporal Gap (years)", fontsize=9)
    ax.set_ylabel("RBD Score", fontsize=9)
    ax.set_title("RBD vs Gap colored by PD Outcome", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(alpha=0.2, ls="--")

    fig.suptitle("PD Outcome: RBD Score and Temporal Gap Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


def export_pd_outcome_summary(
    df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """PD outcome summary: overall + stratified by temporal quartile."""
    rows = []

    def _pd_row(label: str, df_sub: pd.DataFrame) -> Dict:
        no_rbd = df_sub.loc[df_sub["pd_outcome"] == 0, RBD_COL].dropna().values
        yes_rbd = df_sub.loc[df_sub["pd_outcome"] == 1, RBD_COL].dropna().values
        eff = compute_mann_whitney_effect(no_rbd, yes_rbd)
        return {
            "stratum": label,
            "N_total": len(df_sub),
            "N_pd_cases": int((df_sub["pd_outcome"] == 1).sum()),
            "N_pd_controls": int((df_sub["pd_outcome"] == 0).sum()),
            "pct_pd": (df_sub["pd_outcome"] == 1).mean() * 100,
            "mean_rbd_controls": eff["mean_no"],
            "mean_rbd_cases": eff["mean_yes"],
            "cohens_d": eff["cohens_d"],
            "p_value": eff["p_value"],
            "effect": eff["effect"],
        }

    rows.append(_pd_row("Overall", df))

    if "temporal_quartile" in df.columns:
        for q in sorted(df["temporal_quartile"].dropna().unique()):
            rows.append(_pd_row(f"Quartile: {q}", df.loc[df["temporal_quartile"] == q]))

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False, float_format="%.6f")
    print(f"  [Tab] {output_path.name}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(
    df: pd.DataFrame,
    df_quartile_eff: pd.DataFrame,
    df_decile_eff: pd.DataFrame,
    output_path: Path,
) -> None:
    """3x2 master summary dashboard."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    delta = df["delta_years"].dropna().values
    no_pd_rbd = df.loc[df["pd_outcome"] == 0, RBD_COL].dropna().values
    yes_pd_rbd = df.loc[df["pd_outcome"] == 1, RBD_COL].dropna().values

    # (0,0) Temporal gap histogram
    ax = axes[0, 0]
    ax.hist(delta, bins=60, density=True, alpha=0.6, color="#7FB3D8", edgecolor="white", linewidth=0.3)
    kde = gaussian_kde(delta, bw_method=0.15)
    x_grid = np.linspace(delta.min() - 0.5, delta.max() + 0.5, 400)
    ax.plot(x_grid, kde(x_grid), color="#2C5F8A", lw=2)
    ax.axvline(np.mean(delta), color="#C44E52", ls="-", lw=2)
    ax.set_title(f"Temporal Gap Distribution (N={len(delta):,})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Gap (years)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2, ls="--")

    # (0,1) Effect by quartile
    ax = axes[0, 1]
    phenotypes = df_quartile_eff["phenotype"].unique()
    bins = sorted(df_quartile_eff["bin"].unique())
    x = np.arange(len(bins))
    bw = 0.35
    for i, pname in enumerate(phenotypes):
        sub = df_quartile_eff.loc[df_quartile_eff["phenotype"] == pname].sort_values("bin")
        offset = (i - 0.5) * bw
        ax.bar(x + offset, sub["cohens_d"].values, bw, color=C_PHENO.get(pname, "#999"),
               alpha=0.8, label=sub["phenotype_label"].iloc[0], edgecolor="black", linewidth=0.5)
        if sub["ci_lo"].notna().any():
            yerr_lo = sub["cohens_d"].values - sub["ci_lo"].values
            yerr_hi = sub["ci_hi"].values - sub["cohens_d"].values
            ax.errorbar(x + offset, sub["cohens_d"].values,
                        yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=8)
    ax.axhline(0.2, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Effect Size by Temporal Quartile", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cohen's d")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2, ls="--")

    # (1,0) Effect trend deciles
    ax = axes[1, 0]
    for pname in phenotypes:
        sub = df_decile_eff.loc[df_decile_eff["phenotype"] == pname].sort_values("bin_midpoint")
        color = C_PHENO.get(pname, "#999")
        ax.plot(sub["bin_midpoint"].values, sub["cohens_d"].values, "o-", color=color, lw=2, markersize=4,
                label=sub["phenotype_label"].iloc[0])
        if sub["ci_lo"].notna().all():
            ax.fill_between(sub["bin_midpoint"].values, sub["ci_lo"].values, sub["ci_hi"].values,
                            color=color, alpha=0.15)
    ax.axhline(0.2, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Effect Size Trend Across Deciles", fontsize=10, fontweight="bold")
    ax.set_xlabel("Gap midpoint (years)")
    ax.set_ylabel("Cohen's d")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, ls="--")

    # (1,1) RBD by PD boxplot
    ax = axes[1, 1]
    bp = ax.boxplot([no_pd_rbd, yes_pd_rbd], tick_labels=["No PD", "PD"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5))
    bp["boxes"][0].set_facecolor(C_NOPD)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_PD)
    bp["boxes"][1].set_alpha(0.7)
    eff_pd = compute_mann_whitney_effect(no_pd_rbd, yes_pd_rbd)
    ax.set_title(f"RBD by PD | d={eff_pd['cohens_d']:.3f}, p={eff_pd['p_value']:.2e}",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("RBD Score")
    ax.grid(axis="y", alpha=0.2, ls="--")

    # (2,0) Scatter delta vs RBD (dream enactment)
    ax = axes[2, 0]
    pname = "dream_enactment"
    bcol = PHENOTYPE_CONFIG[pname]["binary_col"]
    df_plot = df.dropna(subset=[bcol, RBD_COL, "delta_years"])
    for group, color in [("No", C_NO), ("Yes", C_YES)]:
        mask = df_plot[bcol] == group
        ax.scatter(df_plot.loc[mask, "delta_years"].values, df_plot.loc[mask, RBD_COL].values,
                   s=2, alpha=0.02, color=color, label=f"{group}", rasterized=True)
    ax.set_title(f"Gap vs RBD ({PHENOTYPE_CONFIG[pname]['label']})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Gap (years)")
    ax.set_ylabel("RBD Score")
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(alpha=0.2, ls="--")

    # (2,1) Gap by PD
    ax = axes[2, 1]
    no_pd_gap = df.loc[df["pd_outcome"] == 0, "delta_years"].dropna().values
    yes_pd_gap = df.loc[df["pd_outcome"] == 1, "delta_years"].dropna().values
    bp = ax.boxplot([no_pd_gap, yes_pd_gap], tick_labels=["No PD", "PD"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5))
    bp["boxes"][0].set_facecolor(C_NOPD)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_PD)
    bp["boxes"][1].set_alpha(0.7)
    if len(no_pd_gap) >= 3 and len(yes_pd_gap) >= 3:
        _, p_g = stats.mannwhitneyu(no_pd_gap, yes_pd_gap, alternative="two-sided")
        d_g = compute_cohens_d(no_pd_gap, yes_pd_gap)
        ax.set_title(f"Gap by PD | d={d_g:.4f}, p={p_g:.2e}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Gap (years)")
    ax.grid(axis="y", alpha=0.2, ls="--")

    fig.suptitle("Temporal Validation Summary Dashboard", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig] {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run temporal validation of RBD score reliability across measurement gaps."""
    # ── Setup ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TEMPORAL VALIDATION: RBD SCORE x QUESTIONNAIRE MEASUREMENT GAP")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────
    thresholds, df_risk = get_clean_risk_data(file_name="ehr_diag_pd_rbd_only_all")
    n_total = df_risk.shape[0]
    n_eid_total = df_risk["eid"].nunique()
    print(f"  Loaded: {n_total:,} rows, {n_eid_total:,} subjects")

    # Missingness report
    n_actig = df_risk[ACTIG_DATE_COL].notna().sum()
    n_quest = df_risk[QUEST_DATE_COL].notna().sum()
    n_both = (df_risk[ACTIG_DATE_COL].notna() & df_risk[QUEST_DATE_COL].notna()).sum()
    print(f"  Actigraphy date present: {n_actig:,}/{n_total:,} ({n_actig/n_total*100:.1f}%)")
    print(f"  Questionnaire date present: {n_quest:,}/{n_total:,} ({n_quest/n_total*100:.1f}%)")
    print(f"  Both dates present: {n_both:,}/{n_total:,} ({n_both/n_total*100:.1f}%)")

    # Filter for temporal data
    df_risk = df_risk.loc[
        df_risk[ACTIG_DATE_COL].notna() & df_risk[QUEST_DATE_COL].notna()
    ].copy()
    print(f"  After temporal filter: {df_risk.shape[0]:,} rows, {df_risk['eid'].nunique():,} subjects")

    # ── Subject-level aggregation ──────────────────────────────────────────
    rbd_raw_col = RBD_COL.replace("_mean", "")  # abk_rbd_score
    agg_dict = {
        RBD_COL: (rbd_raw_col, "mean"),
        DREAM_COL: (DREAM_COL, "first"),
        VIOLENT_COL: (VIOLENT_COL, "first"),
        QUEST_DATE_COL: (QUEST_DATE_COL, "first"),
        ACTIG_DATE_COL: (ACTIG_DATE_COL, "first"),
        OUTCOME_EVENT_COL: (OUTCOME_EVENT_COL, "first"),
        OUTCOME_DAYS_COL: (OUTCOME_DAYS_COL, "first"),
    }
    df_subj = df_risk.groupby("eid", as_index=False).agg(**agg_dict)
    df_subj = df_subj.dropna(subset=[RBD_COL])
    print(f"  Subject-level: {df_subj.shape[0]:,} subjects")

    # ── Temporal metrics ───────────────────────────────────────────────────
    quest_dt = pd.to_datetime(df_subj[QUEST_DATE_COL], errors="coerce")
    actig_dt = pd.to_datetime(df_subj[ACTIG_DATE_COL], errors="coerce")
    df_subj["delta_days"] = (quest_dt - actig_dt).dt.days
    df_subj["delta_years"] = df_subj["delta_days"] / 365.25
    df_subj["abs_delta_years"] = df_subj["delta_years"].abs()

    df_subj["measurement_order"] = np.where(
        df_subj["delta_days"] > 0, "RBD first",
        np.where(df_subj["delta_days"] < 0, "Questionnaire first", "Same day")
    )

    # Binary phenotypes
    df_subj["dream_enactment_binary"] = np.where(
        df_subj[DREAM_COL] == 0, "No",
        np.where(df_subj[DREAM_COL] >= 1, "Yes", None)
    )
    df_subj["violent_sleep_binary"] = np.where(
        df_subj[VIOLENT_COL] == 0, "No",
        np.where(df_subj[VIOLENT_COL] >= 1, "Yes", None)
    )

    # PD outcome
    df_subj["pd_outcome"] = df_subj[OUTCOME_EVENT_COL].fillna(0).astype(int).clip(0, 1)

    # Temporal quartiles and deciles
    valid_delta = df_subj["abs_delta_years"].dropna()
    df_subj["temporal_quartile"], quartile_info = bin_by_quantiles(df_subj["abs_delta_years"], 4, "Q")
    df_subj["temporal_decile"], decile_info = bin_by_quantiles(df_subj["abs_delta_years"], 10, "D")

    # ── Summary print ──────────────────────────────────────────────────────
    delta_vals = df_subj["delta_years"].dropna().values
    print(f"\n  Temporal gap summary:")
    print(f"    Mean = {delta_vals.mean():.2f}y, Median = {np.median(delta_vals):.2f}y, "
          f"SD = {delta_vals.std():.2f}y")
    print(f"    Range = [{delta_vals.min():.1f}, {delta_vals.max():.1f}]y")
    print(f"    IQR = [{np.percentile(delta_vals, 25):.2f}, {np.percentile(delta_vals, 75):.2f}]y")
    print(f"\n  Measurement order:")
    for order, count in df_subj["measurement_order"].value_counts().items():
        print(f"    {order}: {count:,} ({count/len(df_subj)*100:.1f}%)")
    print(f"\n  PD outcome: {df_subj['pd_outcome'].sum():,} cases "
          f"({df_subj['pd_outcome'].mean()*100:.2f}%)")
    print(f"\n  Temporal quartile bins:")
    for _, row in quartile_info.iterrows():
        n_q = (df_subj["temporal_quartile"] == row["label"]).sum()
        print(f"    {row['label']}: [{row['lo']:.2f}, {row['hi']:.2f}]y, N={n_q:,}")

    # ==================================================================
    # SECTION 1: TEMPORAL LANDSCAPE
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("[1] TEMPORAL LANDSCAPE")
    print("=" * 80)

    plot_temporal_gap_distribution(
        df_subj["delta_years"].dropna().values,
        OUTPUT_DIR / "temporal_gap_distribution.png",
    )

    plot_temporal_gap_by_phenotype(
        df_subj,
        OUTPUT_DIR / "temporal_gap_by_phenotype.png",
    )

    export_temporal_descriptive(
        df_subj,
        OUTPUT_DIR / "temporal_descriptive_summary.csv",
    )

    # ==================================================================
    # SECTION 2: DATA-DRIVEN TEMPORAL WINDOWS
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("[2] STRATIFICATION STRENGTH BY TEMPORAL QUARTILE / DECILE")
    print("=" * 80)

    print("\n  Computing effect sizes by quartile (with bootstrap CI)...")
    df_quartile_eff = compute_effect_by_bins(
        df_subj, "temporal_quartile", quartile_info, PHENOTYPE_CONFIG, RBD_COL, n_boot=2000,
    )
    df_quartile_eff.to_csv(OUTPUT_DIR / "stratification_by_temporal_quartile.csv",
                           index=False, float_format="%.6f")
    print(f"  [Tab] stratification_by_temporal_quartile.csv")

    for _, row in df_quartile_eff.iterrows():
        print(f"    {row['bin']} | {row['phenotype_label']:20s} | "
              f"d={row['cohens_d']:.4f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] | "
              f"p={row['p_value']:.2e} | N={row['N']:,}")

    plot_effect_by_quartile(df_quartile_eff, OUTPUT_DIR / "effect_by_temporal_quartile.png")

    print("\n  Computing effect sizes by decile (with bootstrap CI)...")
    df_decile_eff = compute_effect_by_bins(
        df_subj, "temporal_decile", decile_info, PHENOTYPE_CONFIG, RBD_COL, n_boot=2000,
    )
    df_decile_eff.to_csv(OUTPUT_DIR / "stratification_by_temporal_decile.csv",
                         index=False, float_format="%.6f")
    print(f"  [Tab] stratification_by_temporal_decile.csv")

    plot_effect_trend_deciles(df_decile_eff, OUTPUT_DIR / "effect_trend_deciles.png")

    # Spearman: does effect size correlate with temporal gap?
    for pname in PHENOTYPE_CONFIG:
        sub = df_decile_eff.loc[df_decile_eff["phenotype"] == pname].dropna(subset=["cohens_d"])
        if len(sub) >= 4:
            rho, p = spearmanr(sub["bin_midpoint"].values, sub["cohens_d"].values)
            print(f"    Spearman ({pname}): rho={rho:.4f}, p={p:.4f}")

    # ==================================================================
    # SECTION 3: CONTINUOUS TEMPORAL ANALYSIS
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("[3] CONTINUOUS TEMPORAL ANALYSIS")
    print("=" * 80)

    plot_scatter_delta_vs_rbd(df_subj, OUTPUT_DIR / "scatter_temporal_gap_vs_rbd.png")
    plot_rbd_kde_by_quartile(df_subj, OUTPUT_DIR / "rbd_dist_by_temporal_quartile.png")
    export_continuous_correlations(df_subj, OUTPUT_DIR / "continuous_temporal_correlations.csv")

    # ==================================================================
    # SECTION 4: PD OUTCOME
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("[4] PD OUTCOME ANALYSIS")
    print("=" * 80)

    plot_rbd_by_pd_outcome(df_subj, OUTPUT_DIR / "rbd_score_by_pd_outcome.png")
    plot_temporal_gap_by_pd(df_subj, OUTPUT_DIR / "temporal_gap_by_pd_outcome.png")
    plot_pd_combined_panel(df_subj, OUTPUT_DIR / "rbd_temporal_pd_combined.png")
    export_pd_outcome_summary(df_subj, OUTPUT_DIR / "pd_outcome_summary.csv")

    # ==================================================================
    # SECTION 5: SUMMARY DASHBOARD
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("[5] SUMMARY DASHBOARD")
    print("=" * 80)

    plot_summary_dashboard(
        df_subj, df_quartile_eff, df_decile_eff,
        OUTPUT_DIR / "temporal_analysis_dashboard.png",
    )

    # ==================================================================
    # FINAL REPORT
    # ==================================================================
    n_figs = len(list(OUTPUT_DIR.glob("*.png")))
    n_tabs = len(list(OUTPUT_DIR.glob("*.csv")))
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Figures: {n_figs} PNG")
    print(f"  Tables: {n_tabs} CSV")
    print("=" * 80)


if __name__ == "__main__":
    main()
