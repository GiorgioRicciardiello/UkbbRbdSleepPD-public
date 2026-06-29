"""
Sleep Phenotype vs RBD Score Evaluation
=======================================
Evaluates whether ML-derived RBD probability scores from actigraphy
properly stratify self-reported sleep questionnaire responses.

Core validation: if RBD scores cannot separate dream-enactment Yes vs No,
the model is not capturing RBD-related behavior.

All analysis is at the **subject level** (one row per eid).
"""

import shutil
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import gaussian_kde

from config.config import config
from library.risk.risk_helpers import get_clean_risk_data

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Output directory ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/sleep_phenotypes_nostf")

# ── Global plot style ─────────────────────────────────────────────────────
PALETTE = {
    "No": "#4C72B0",
    "Yes": "#DD8452",
    "box_all": "#8DA0CB",
    "density_no": "#4C72B0",
    "density_yes": "#DD8452",
}
FIG_DPI = 300

# ── Phenotype label mappings ─────────────────────────────────────────────
DREAM_ENACTMENT_LABELS: Dict[int, str] = {
    -3: "Prefer not to answer",
    -2: "Do not know",
    -1: "Not applicable",
    0: "Never/rarely",
    1: "Sometimes/often/always",
}

VIOLENT_SLEEP_LABELS: Dict[int, str] = {
    -3: "Prefer not to answer",
    -2: "Do not know",
    -1: "Not applicable",
    0: "Never/rarely",
    1: "1-2 nights/week",
    2: "3-4 nights/week",
    3: "5+ nights/week",
    4: ">=4 nights/week (alt)",
    5: "More frequent",
    6: "Frequent",
    7: "Very frequent",
}

# Columns to aggregate at subject level (first value per eid)
SLEEP_COVARIATES: List[str] = [
    "cov_dream_enactment_freq_30557",
    "cov_violent_sleep_freq_30558",
    "cov_sleepwalk_freq_30555",
    "cov_teeth_grind_freq_30556",
    "cov_nightmare_freq_30559",
]

RBD_COL = "abk_rbd_score_mean"


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_stratified_statistics(
    df_data: pd.DataFrame,
    rbd_col: str,
    strat_col: str,
    label_map: Dict[int, str],
) -> Tuple[pd.DataFrame, dict]:
    """
    Stratified summary stats + Kruskal-Wallis test + eta-squared.

    Parameters
    ----------
    df_data : pd.DataFrame
        Subject-level data.
    rbd_col : str
        RBD score column.
    strat_col : str
        Stratification column (integer-coded).
    label_map : dict
        Code -> label mapping.

    Returns
    -------
    summary : pd.DataFrame
        Per-group descriptive statistics.
    test_results : dict
        KW H-stat, p-value, eta-squared, sample sizes.
    """
    df_clean = df_data.dropna(subset=[strat_col, rbd_col]).copy()

    summary = df_clean.groupby(strat_col)[rbd_col].agg(
        N="count",
        Mean="mean",
        Median="median",
        Std="std",
        SE=lambda x: x.std() / np.sqrt(len(x)),
        Min="min",
        Max="max",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
    ).round(4)

    summary["Group"] = summary.index.map(label_map).fillna("Unknown")
    # 95% CI on the mean
    summary["CI95_lo"] = summary["Mean"] - 1.96 * summary["SE"]
    summary["CI95_hi"] = summary["Mean"] + 1.96 * summary["SE"]

    groups = [
        df_clean.loc[df_clean[strat_col] == code, rbd_col].values
        for code in sorted(df_clean[strat_col].unique())
    ]
    # Need at least 2 non-empty groups for KW
    non_empty = [g for g in groups if len(g) > 0]
    if len(non_empty) >= 2:
        h_stat, p_value = stats.kruskal(*non_empty)
    else:
        h_stat, p_value = np.nan, np.nan

    grand_mean = df_clean[rbd_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in non_empty)
    ss_total = ((df_clean[rbd_col] - grand_mean) ** 2).sum()
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    test_results = {
        "test": "Kruskal-Wallis",
        "h_statistic": h_stat,
        "p_value": p_value,
        "eta_squared": eta_squared,
        "n_groups": len(non_empty),
        "total_n": len(df_clean),
        "missing_n": len(df_data) - len(df_clean),
    }
    return summary, test_results


def compute_binary_statistics(
    df_data: pd.DataFrame,
    rbd_col: str,
    binary_col: str,
) -> dict:
    """
    Mann-Whitney U test + Cohen's d for a binary (Yes/No) column.

    Parameters
    ----------
    df_data : pd.DataFrame
        Subject-level data.
    rbd_col : str
        RBD score column.
    binary_col : str
        Binary column with values "No" / "Yes".

    Returns
    -------
    dict with U-stat, p-value, Cohen's d, group Ns, means, stds.
    """
    df_clean = df_data.dropna(subset=[binary_col, rbd_col]).copy()
    no_vals = df_clean.loc[df_clean[binary_col] == "No", rbd_col].values
    yes_vals = df_clean.loc[df_clean[binary_col] == "Yes", rbd_col].values

    if len(no_vals) == 0 or len(yes_vals) == 0:
        return {"test": "Mann-Whitney U", "u_statistic": np.nan, "p_value": np.nan,
                "cohens_d": np.nan}

    u_stat, p_value = stats.mannwhitneyu(no_vals, yes_vals, alternative="two-sided")

    # Cohen's d (pooled SD)
    n0, n1 = len(no_vals), len(yes_vals)
    pooled_std = np.sqrt(((n0 - 1) * no_vals.std(ddof=1) ** 2 +
                          (n1 - 1) * yes_vals.std(ddof=1) ** 2) / (n0 + n1 - 2))
    cohens_d = (yes_vals.mean() - no_vals.mean()) / pooled_std if pooled_std > 0 else 0.0

    return {
        "test": "Mann-Whitney U",
        "u_statistic": u_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_no": n0,
        "n_yes": n1,
        "mean_no": no_vals.mean(),
        "mean_yes": yes_vals.mean(),
        "std_no": no_vals.std(ddof=1),
        "std_yes": yes_vals.std(ddof=1),
    }


def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _annotate_boxes(
    ax: plt.Axes,
    data_groups: List[np.ndarray],
    labels: List[str],
    y_offset_frac: float = 0.02,
) -> None:
    """
    Add N, mean +/- std annotation above each box in a boxplot.

    Parameters
    ----------
    ax : plt.Axes
        The axes containing the boxplot.
    data_groups : list of np.ndarray
        Data arrays, one per box (same order as boxes).
    labels : list of str
        Group labels (unused here but kept for API consistency).
    y_offset_frac : float
        Fraction of y-range to offset text above the box.
    """
    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo

    for i, vals in enumerate(data_groups):
        if len(vals) == 0:
            continue
        x_pos = i + 1  # boxplot positions are 1-indexed
        q3 = np.percentile(vals, 75)
        text = (
            f"N={len(vals):,}\n"
            f"\u03bc={vals.mean():.3f}\n"
            f"\u03c3={vals.std(ddof=1):.3f}"
        )
        ax.text(
            x_pos, q3 + y_range * y_offset_frac, text,
            ha="center", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )


def plot_binary_comparison(
    df: pd.DataFrame,
    rbd_col: str,
    binary_col: str,
    phenotype_name: str,
    output_path: Path,
) -> dict:
    """
    Binary Yes/No boxplot with mean/std annotations and Mann-Whitney test.

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level data.
    rbd_col : str
        RBD score column.
    binary_col : str
        Binary column ("No"/"Yes").
    phenotype_name : str
        Human-readable name for plot title.
    output_path : Path
        Save path for figure.

    Returns
    -------
    dict with binary test statistics.
    """
    test = compute_binary_statistics(df, rbd_col, binary_col)

    df_plot = df.dropna(subset=[binary_col, rbd_col]).copy()
    groups_order = ["No", "Yes"]
    data_groups = [df_plot.loc[df_plot[binary_col] == g, rbd_col].values for g in groups_order]

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot(
        data_groups,
        tick_labels=groups_order,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=6),
    )
    colors = [PALETTE["No"], PALETTE["Yes"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    _annotate_boxes(ax, data_groups, groups_order)

    sig = _sig_stars(test["p_value"])
    ax.set_title(
        f"{phenotype_name} (Binary)\n"
        f"Mann-Whitney p={test['p_value']:.2e} ({sig})  |  Cohen's d={test['cohens_d']:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("RBD Probability Score (subject mean)", fontsize=10, fontweight="bold")
    ax.set_xlabel(phenotype_name, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure saved] {output_path}")
    return test


def plot_all_levels(
    df: pd.DataFrame,
    rbd_col: str,
    strat_col: str,
    label_map: Dict[int, str],
    phenotype_name: str,
    output_path: Path,
) -> Tuple[pd.DataFrame, dict]:
    """
    Multi-level boxplot (valid codes only, excluding -3/-2/-1) with annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level data.
    rbd_col : str
        RBD score column.
    strat_col : str
        Integer-coded stratification column.
    label_map : dict
        Code -> label.
    phenotype_name : str
        Title label.
    output_path : Path
        Save path.

    Returns
    -------
    summary : pd.DataFrame
        Per-level stats.
    test_results : dict
        KW test output.
    """
    # Filter to valid response codes (>= 0)
    df_valid = df.loc[df[strat_col] >= 0].copy()
    summary, test_results = compute_stratified_statistics(df_valid, rbd_col, strat_col, label_map)

    codes_sorted = sorted(df_valid[strat_col].dropna().unique())
    labels = [label_map.get(int(c), f"Code {int(c)}") for c in codes_sorted]
    data_groups = [df_valid.loc[df_valid[strat_col] == c, rbd_col].dropna().values for c in codes_sorted]

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(codes_sorted)), 6))
    bp = ax.boxplot(
        data_groups,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["box_all"])
        patch.set_alpha(0.7)

    _annotate_boxes(ax, data_groups, labels)

    sig = _sig_stars(test_results["p_value"])
    ax.set_title(
        f"{phenotype_name} — All Response Levels\n"
        f"Kruskal-Wallis H={test_results['h_statistic']:.1f}, "
        f"p={test_results['p_value']:.2e} ({sig})  |  "
        f"\u03b7\u00b2={test_results['eta_squared']:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("RBD Probability Score (subject mean)", fontsize=10, fontweight="bold")
    ax.set_xlabel(phenotype_name, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure saved] {output_path}")
    return summary, test_results


def plot_density_by_group(
    df: pd.DataFrame,
    rbd_col: str,
    binary_col: str,
    phenotype_name: str,
    output_path: Path,
) -> None:
    """
    KDE density overlay for binary groups with vertical mean lines.

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level data.
    rbd_col : str
        RBD score column.
    binary_col : str
        Binary column ("No"/"Yes").
    phenotype_name : str
        Title label.
    output_path : Path
        Save path.
    """
    df_plot = df.dropna(subset=[binary_col, rbd_col]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    for group, color_key in [("No", "density_no"), ("Yes", "density_yes")]:
        vals = df_plot.loc[df_plot[binary_col] == group, rbd_col].values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method="scott")
        x_grid = np.linspace(vals.min() - 0.02, vals.max() + 0.02, 500)
        density = kde(x_grid)
        mu = vals.mean()
        ax.fill_between(x_grid, density, alpha=0.35, color=PALETTE[color_key])
        ax.plot(x_grid, density, color=PALETTE[color_key], lw=2,
                label=f"{group} (N={len(vals):,}, \u03bc={mu:.3f})")
        ax.axvline(mu, color=PALETTE[color_key], ls="--", lw=1.5, alpha=0.8)

    ax.set_xlabel("RBD Probability Score (subject mean)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax.set_title(
        f"RBD Score Distribution — {phenotype_name} (Binary)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure saved] {output_path}")


def plot_combined_panel(
    df: pd.DataFrame,
    rbd_col: str,
    strat_col: str,
    binary_col: str,
    label_map: Dict[int, str],
    phenotype_name: str,
    output_path: Path,
) -> None:
    """
    Combined 2x2 panel: binary box | all-levels box | density | violin+strip.

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level data.
    rbd_col : str
        RBD score column.
    strat_col : str
        Integer-coded column.
    binary_col : str
        Binary column ("No"/"Yes").
    label_map : dict
        Code -> label.
    phenotype_name : str
        Title label.
    output_path : Path
        Save path.
    """
    df_valid = df.loc[df[strat_col] >= 0].dropna(subset=[rbd_col]).copy()
    codes_sorted = sorted(df_valid[strat_col].dropna().unique())
    labels_all = [label_map.get(int(c), f"Code {int(c)}") for c in codes_sorted]
    data_all = [df_valid.loc[df_valid[strat_col] == c, rbd_col].values for c in codes_sorted]

    df_bin = df.dropna(subset=[binary_col, rbd_col]).copy()
    data_bin = [df_bin.loc[df_bin[binary_col] == g, rbd_col].values for g in ["No", "Yes"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Panel A: Binary boxplot ───────────────────────────────────────────
    ax = axes[0, 0]
    bp = ax.boxplot(data_bin, tick_labels=["No", "Yes"], patch_artist=True, widths=0.45,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                   markeredgecolor="black", markersize=5))
    for patch, col in zip(bp["boxes"], [PALETTE["No"], PALETTE["Yes"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    _annotate_boxes(ax, data_bin, ["No", "Yes"])
    bin_test = compute_binary_statistics(df, rbd_col, binary_col)
    sig = _sig_stars(bin_test["p_value"])
    ax.set_title(f"A. Binary — MW p={bin_test['p_value']:.2e} ({sig}), d={bin_test['cohens_d']:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("RBD Score", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel B: All-levels boxplot ───────────────────────────────────────
    ax = axes[0, 1]
    bp2 = ax.boxplot(data_all, tick_labels=labels_all, patch_artist=True, widths=0.5,
                     showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="black",
                                    markeredgecolor="black", markersize=4))
    for patch in bp2["boxes"]:
        patch.set_facecolor(PALETTE["box_all"])
        patch.set_alpha(0.7)
    _annotate_boxes(ax, data_all, labels_all)
    df_kw = df_valid.copy()
    non_empty_groups = [df_kw.loc[df_kw[strat_col] == c, rbd_col].values
                        for c in codes_sorted if len(df_kw.loc[df_kw[strat_col] == c]) > 0]
    if len(non_empty_groups) >= 2:
        h_stat, kw_p = stats.kruskal(*non_empty_groups)
    else:
        h_stat, kw_p = np.nan, np.nan
    ax.set_title(f"B. All Levels — KW p={kw_p:.2e} ({_sig_stars(kw_p)})",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("RBD Score", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel C: KDE density ──────────────────────────────────────────────
    ax = axes[1, 0]
    for group, color_key in [("No", "density_no"), ("Yes", "density_yes")]:
        vals = df_bin.loc[df_bin[binary_col] == group, rbd_col].values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method="scott")
        x_grid = np.linspace(vals.min() - 0.02, vals.max() + 0.02, 500)
        density = kde(x_grid)
        mu = vals.mean()
        ax.fill_between(x_grid, density, alpha=0.35, color=PALETTE[color_key])
        ax.plot(x_grid, density, color=PALETTE[color_key], lw=2,
                label=f"{group} (N={len(vals):,}, \u03bc={mu:.3f})")
        ax.axvline(mu, color=PALETTE[color_key], ls="--", lw=1.5, alpha=0.8)
    ax.set_title("C. RBD Score Density (Binary)", fontsize=10, fontweight="bold")
    ax.set_xlabel("RBD Score", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel D: Violin + strip ───────────────────────────────────────────
    ax = axes[1, 1]
    vp = ax.violinplot(data_bin, positions=[1, 2], showmeans=True, showmedians=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor([PALETTE["No"], PALETTE["Yes"]][i])
        body.set_alpha(0.5)
    # Strip (jittered scatter)
    rng = np.random.default_rng(seed=42)
    for i, (group, col_key) in enumerate([("No", "No"), ("Yes", "Yes")]):
        vals = data_bin[i]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i + 1) + jitter, vals,
                   alpha=0.08, s=4, color=PALETTE[col_key], edgecolors="none")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_title("D. Violin + Strip Plot (Binary)", fontsize=10, fontweight="bold")
    ax.set_ylabel("RBD Score", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"RBD Score Stratification by {phenotype_name}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure saved] {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# TABLE EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_summary_table(
    summary: pd.DataFrame,
    test_results: dict,
    output_path: Path,
) -> None:
    """
    Save stratified summary + test results to CSV.

    Parameters
    ----------
    summary : pd.DataFrame
        Per-group statistics from compute_stratified_statistics.
    test_results : dict
        Statistical test output.
    output_path : Path
        Save path.
    """
    out = summary.copy()
    out = out[["Group", "N", "Mean", "Median", "Std", "SE", "CI95_lo", "CI95_hi",
               "Q1", "Q3", "IQR", "Min", "Max"]]
    out.index.name = "Code"
    out.to_csv(output_path)
    print(f"  [Table saved] {output_path}")


def export_binary_table(
    binary_results: List[dict],
    output_path: Path,
) -> None:
    """
    Save binary comparison results to CSV.

    Parameters
    ----------
    binary_results : list of dict
        Each dict from compute_binary_statistics, with extra 'phenotype' key.
    output_path : Path
        Save path.
    """
    df_out = pd.DataFrame(binary_results)
    df_out.to_csv(output_path, index=False)
    print(f"  [Table saved] {output_path}")


def export_all_tests_table(
    all_tests: List[dict],
    output_path: Path,
) -> None:
    """
    Save all statistical test results to CSV.

    Parameters
    ----------
    all_tests : list of dict
        Combined test results with 'phenotype' and 'level' keys.
    output_path : Path
        Save path.
    """
    df_out = pd.DataFrame(all_tests)
    df_out.to_csv(output_path, index=False)
    print(f"  [Table saved] {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run sleep phenotype vs RBD score evaluation."""
    # ── Prepare output directory ──────────────────────────────────────────
    if OUTPUT_DIR.exists():
        # Windows can deny rmtree if Explorer or a prior process holds a
        # handle.  Strip read-only bits first, then retry on error.
        def _force_remove(func, path, _exc_info):
            import stat
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass  # best-effort; mkdir below recreates the tree anyway

        shutil.rmtree(OUTPUT_DIR, onexc=_force_remove)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LOADING RISK DATA FOR PHENOTYPE ANALYSIS")
    print("=" * 80)

    try:
        thresholds, df_risk = get_clean_risk_data(file_name="ehr_diag_pd_rbd_only_all")
        print(f"  [OK] Loaded: {df_risk.shape[0]:,} rows, {df_risk.shape[1]} cols")
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        raise

    # ── Subject-level aggregation ─────────────────────────────────────────
    # Keep only rows with valid wear_time and violent sleep (mirroring original filter)
    df_risk = df_risk.loc[
        df_risk["wear_time_start"].notna() &
        df_risk["cov_violent_sleep_freq_30558"].notna()
    ]

    agg_dict = {"abk_rbd_score": ("abk_rbd_score", "mean")}
    for col in SLEEP_COVARIATES:
        if col in df_risk.columns:
            agg_dict[col] = (col, "first")

    df_subj = df_risk.groupby("eid", as_index=False).agg(**agg_dict)
    df_subj = df_subj.rename(columns={"abk_rbd_score": RBD_COL})
    df_subj = df_subj.dropna(subset=[RBD_COL])
    print(f"  [OK] Subject-level: {df_subj.shape[0]:,} subjects")

    # # remove the stanford ids as double check -> we get the same results
    # path_stf_rbd = config.get('paths')['actig_extracted']['root'].joinpath('ActigStfRecords', 'RBD_Sleep_Score_avg_abk_merged.parquet')
    # df_rbd_stf = pd.read_parquet(path_stf_rbd)
    # # df_rbd_stf['eid'] = df_rbd_stf['eid'].astype(int)
    # df_rbd_stf['eid'] = df_rbd_stf['ID'].apply(lambda x: x.split('_')[0])
    # df_rbd_stf['eid'] = df_rbd_stf['eid'].astype(int)
    # df_subj = df_subj.loc[~df_subj['eid'].isin(df_rbd_stf['eid']), :]
    # '32122'
    #
    # df_risk['cov_sleep_quest_complete_32122']
    # [col for col in df_risk.columns if '32122' in col ]
    # ── Binary recoding ───────────────────────────────────────────────────
    dream_col = "cov_dream_enactment_freq_30557"
    violent_col = "cov_violent_sleep_freq_30558"

    # Dream enactment: 0 -> No, 1 -> Yes (already binary in UKBB coding)
    df_subj["dream_enactment_binary"] = df_subj[dream_col].map(
        lambda v: "No" if v == 0 else ("Yes" if v >= 1 else None)
    )
    # Violent sleep: 0 -> No, >=1 -> Yes (collapse frequency)
    df_subj["violent_sleep_binary"] = df_subj[violent_col].map(
        lambda v: "No" if v == 0 else ("Yes" if v >= 1 else None)
    )

    # ── Analysis containers ───────────────────────────────────────────────
    all_binary_results: List[dict] = []
    all_test_results: List[dict] = []

    # ═══════════════════════════════════════════════════════════════════════
    # DREAM ENACTMENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    phenotype = "Dream Enactment"
    print(f"\n{'=' * 80}")
    print(f"[1] {phenotype.upper()} FREQUENCY STRATIFICATION")
    print("=" * 80)

    if dream_col in df_subj.columns:
        # Figure A: Binary boxplot
        bin_test = plot_binary_comparison(
            df_subj, RBD_COL, "dream_enactment_binary", phenotype,
            OUTPUT_DIR / "dream_enactment_binary_boxplot.png",
        )
        bin_test["phenotype"] = phenotype
        all_binary_results.append(bin_test)
        all_test_results.append({**bin_test, "level": "binary"})

        # Figure B: All-levels boxplot
        dream_summary, dream_kw = plot_all_levels(
            df_subj, RBD_COL, dream_col, DREAM_ENACTMENT_LABELS, phenotype,
            OUTPUT_DIR / "dream_enactment_all_levels_boxplot.png",
        )
        all_test_results.append({**dream_kw, "phenotype": phenotype, "level": "all_levels"})

        # Figure C: Density
        plot_density_by_group(
            df_subj, RBD_COL, "dream_enactment_binary", phenotype,
            OUTPUT_DIR / "dream_enactment_density.png",
        )

        # Figure D: Combined panel
        plot_combined_panel(
            df_subj, RBD_COL, dream_col, "dream_enactment_binary",
            DREAM_ENACTMENT_LABELS, phenotype,
            OUTPUT_DIR / "dream_enactment_combined_panel.png",
        )

        # Export summary table (includes ALL codes for reference)
        full_summary, full_kw = compute_stratified_statistics(
            df_subj, RBD_COL, dream_col, DREAM_ENACTMENT_LABELS,
        )
        export_summary_table(full_summary, full_kw,
                             OUTPUT_DIR / "summary_dream_enactment.csv")

        # Print binary results
        print(f"\n  Binary comparison (No vs Yes):")
        print(f"    N(No)={bin_test.get('n_no', 0):,}  N(Yes)={bin_test.get('n_yes', 0):,}")
        print(f"    Mean(No)={bin_test.get('mean_no', 0):.4f}  Mean(Yes)={bin_test.get('mean_yes', 0):.4f}")
        print(f"    Mann-Whitney p={bin_test['p_value']:.2e} ({_sig_stars(bin_test['p_value'])})")
        print(f"    Cohen's d={bin_test['cohens_d']:.4f}")
    else:
        print(f"  [SKIP] Column '{dream_col}' not found")

    # ═══════════════════════════════════════════════════════════════════════
    # VIOLENT SLEEP ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    phenotype = "Violent Sleep"
    print(f"\n{'=' * 80}")
    print(f"[2] {phenotype.upper()} FREQUENCY STRATIFICATION")
    print("=" * 80)

    if violent_col in df_subj.columns:
        # Figure A: Binary boxplot
        bin_test = plot_binary_comparison(
            df_subj, RBD_COL, "violent_sleep_binary", phenotype,
            OUTPUT_DIR / "violent_sleep_binary_boxplot.png",
        )
        bin_test["phenotype"] = phenotype
        all_binary_results.append(bin_test)
        all_test_results.append({**bin_test, "level": "binary"})

        # Figure B: All-levels boxplot
        violent_summary, violent_kw = plot_all_levels(
            df_subj, RBD_COL, violent_col, VIOLENT_SLEEP_LABELS, phenotype,
            OUTPUT_DIR / "violent_sleep_all_levels_boxplot.png",
        )
        all_test_results.append({**violent_kw, "phenotype": phenotype, "level": "all_levels"})

        # Figure C: Density
        plot_density_by_group(
            df_subj, RBD_COL, "violent_sleep_binary", phenotype,
            OUTPUT_DIR / "violent_sleep_density.png",
        )

        # Figure D: Combined panel
        plot_combined_panel(
            df_subj, RBD_COL, violent_col, "violent_sleep_binary",
            VIOLENT_SLEEP_LABELS, phenotype,
            OUTPUT_DIR / "violent_sleep_combined_panel.png",
        )

        # Export summary table
        full_summary, full_kw = compute_stratified_statistics(
            df_subj, RBD_COL, violent_col, VIOLENT_SLEEP_LABELS,
        )
        export_summary_table(full_summary, full_kw,
                             OUTPUT_DIR / "summary_violent_sleep.csv")

        # Print binary results
        print(f"\n  Binary comparison (No vs Yes):")
        print(f"    N(No)={bin_test.get('n_no', 0):,}  N(Yes)={bin_test.get('n_yes', 0):,}")
        print(f"    Mean(No)={bin_test.get('mean_no', 0):.4f}  Mean(Yes)={bin_test.get('mean_yes', 0):.4f}")
        print(f"    Mann-Whitney p={bin_test['p_value']:.2e} ({_sig_stars(bin_test['p_value'])})")
        print(f"    Cohen's d={bin_test['cohens_d']:.4f}")
    else:
        print(f"  [SKIP] Column '{violent_col}' not found")

    # ── Export combined tables ────────────────────────────────────────────
    export_binary_table(all_binary_results, OUTPUT_DIR / "binary_comparison.csv")
    export_all_tests_table(all_test_results, OUTPUT_DIR / "statistical_tests.csv")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"  Figures: {len(list(OUTPUT_DIR.glob('*.png')))} PNG files")
    print(f"  Tables:  {len(list(OUTPUT_DIR.glob('*.csv')))} CSV files")
    print("\nKey: *** p<0.001 | ** p<0.01 | * p<0.05 | ns = not significant")


if __name__ == "__main__":
    main()
