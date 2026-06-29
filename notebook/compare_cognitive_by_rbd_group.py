"""
Cognitive Performance by RBD Risk Group — Controls Only
=========================================================
Compare cognitive test scores across three RBD risk strata (Low / Intermediate / High)
in subjects free of all neurodegenerative outcomes (controls only).

Source  : data/pp/res_build_final_dataset/ehr_diag_pd_rbd_only_all.parquet
          (produced by run_merge_ukbb_rbd.py; contains rg_pctl3 + cog_*_latest)
Grouping: rg_pctl3  (Low 0-90 / Intermediate 90-99 / High 99-100 percentile)
Cohort  : control == True after standard exclusions (get_clean_risk_data)
Tests   : Kruskal-Wallis global; Dunn post-hoc with Bonferroni within each variable
Adj.    : Benjamini-Hochberg FDR across the 9 cognitive variables
Confound: age (cov_age_recruitment_21022) + sex (cov_sex_31) — OLS residualisation
Outputs : results/cognitive_rbd_group/{table_raw, table_adjusted, fig_raw, fig_adjusted}
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive — no pop-up windows

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from config.config import config
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level

# ── Constants ──────────────────────────────────────────────────────────────────

FILE_NAME: str = "ehr_diag_pd_rbd_only_all"
RG_COL: str = "rg_pctl3"
AGE_COL: str = "cov_age_recruitment_21022"
SEX_COL: str = "cov_sex_31"
SEED: int = 42

GROUP_ORDER: List[str] = ["Low (0,90%)", "Intermediate (90,99%)", "High (99,100%)"]
GROUP_COLORS: Dict[str, str] = {
    "Low (0,90%)":           "#4878d0",
    "Intermediate (90,99%)": "#ee854a",
    "High (99,100%)":        "#d65f5f",
}
GROUP_SHORT: Dict[str, str] = {
    "Low (0,90%)":           "Low",
    "Intermediate (90,99%)": "Mid",
    "High (99,100%)":        "High",
}

# Variables with 0% non-null in this cohort are excluded automatically.
COG_VARS: Dict[str, str] = {
    "cog_fluid_intelligence_latest":  "Fluid Intelligence",
    "cog_react_time_latest":          "Reaction Time (ms)",
    "cog_numeric_memory_latest":      "Numeric Memory",
    "cog_pairs_status_latest":        "Pairs Matching (errors)",
    "cog_sds_accuracy_latest":        "SDS Accuracy",
    "cog_tmt1_dur_latest":            "TMT-A Duration (s)",
    "cog_tmt2_dur_latest":            "TMT-B Duration (s)",
    "cog_tmt_ratio_log_latest":       "Log TMT-B/A Ratio",
}

OUT_DIR: Path = Path("results/cognitive_rbd_group")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_controls() -> pd.DataFrame:
    """
    Load the production merged parquet, apply standard exclusions via
    get_clean_risk_data, collapse to subject level, and retain controls only.

    Returns
    -------
    pd.DataFrame
        One row per control subject. Contains rg_pctl3 and all cog_*_latest cols.
    """
    _, df_night = get_clean_risk_data(file_name=FILE_NAME)
    df = make_subject_level(df_night, id_col="eid", prob_col="abk_rbd_score_mean")
    df = df[df["control"] == True].reset_index(drop=True)
    print(f"  Controls after exclusions: {len(df):,}")
    print(f"\n  RBD group distribution:\n{df[RG_COL].value_counts().to_string()}\n")
    return df


def filter_available_cog_vars(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return the subset of COG_VARS that have ≥ 1 non-null value in df.
    Drops variables with 0% non-null (e.g. sds_correct_per_min, prospective_memory).
    """
    available = {
        col: label
        for col, label in COG_VARS.items()
        if col in df.columns and df[col].notna().any()
    }
    dropped = set(COG_VARS) - set(available)
    if dropped:
        print(f"  Dropped (all-null): {dropped}")
    return available


# ── Residualisation ────────────────────────────────────────────────────────────

def residualise_age_sex(
    df: pd.DataFrame,
    cog_vars: Dict[str, str],
    age_col: str = AGE_COL,
    sex_col: str = SEX_COL,
) -> pd.DataFrame:
    """
    OLS-residualise each cognitive variable on age + sex.
    Adds a column '{var}_adj' for each variable.
    Subjects with missing age or sex are excluded from each model.

    Assumptions stated:
      - Linearity of age effect on cognitive score (reasonable over 39-70 age range).
      - Sex is treated as binary categorical (UKBB coding: 0=female, 1=male).
      - Residuals retain any group-level variance not explained by age/sex.
    """
    df_out = df.copy()
    for var in cog_vars:
        sub = df[[var, age_col, sex_col]].dropna().copy()
        if len(sub) < 100:
            df_out[f"{var}_adj"] = np.nan
            warnings.warn(f"  {var}: too few complete cases ({len(sub)}) — _adj set to NaN")
            continue
        # Cast nullable integer types to plain numpy float/int so patsy can handle them.
        sub[var] = sub[var].astype(float)
        sub[age_col] = sub[age_col].astype(float)
        sub[sex_col] = sub[sex_col].astype(float)
        formula = f"{var} ~ {age_col} + {sex_col}"
        model = ols(formula, data=sub).fit()
        df_out.loc[sub.index, f"{var}_adj"] = model.resid.values
    return df_out


# ── Statistics ─────────────────────────────────────────────────────────────────

def _group_arrays(
    df: pd.DataFrame,
    var: str,
    rg_col: str = RG_COL,
) -> Dict[str, np.ndarray]:
    """Extract non-null values per group as numpy arrays, in GROUP_ORDER."""
    return {
        g: df.loc[(df[rg_col] == g) & df[var].notna(), var].to_numpy(dtype=float)
        for g in GROUP_ORDER
        if g in df[rg_col].values
    }


def kruskal_wallis(groups: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Run Kruskal-Wallis and return (H, p)."""
    arrays = [arr for arr in groups.values() if len(arr) > 0]
    if len(arrays) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*arrays)
    return float(h), float(p)


def epsilon_squared(h: float, n_total: int) -> float:
    """
    Epsilon-squared effect size for Kruskal-Wallis.
    Formula: ε² = H / (N - 1)   [Tomczak & Tomczak, 2014].
    Interpretation: <0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, >0.14 large.
    """
    if np.isnan(h) or n_total <= 1:
        return np.nan
    return float(h / (n_total - 1))


def dunn_pairwise(groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Dunn's post-hoc test for all pairwise comparisons.

    Ranks all observations jointly, then computes z-scores using the pooled
    rank variance estimate from the Kruskal-Wallis framework:
        z_ij = (R̄_i - R̄_j) / sqrt[(N(N+1)/12) * (1/n_i + 1/n_j)]

    Bonferroni correction applied within each variable (multiplied by number
    of pairwise comparisons = 3 for three groups).

    Returns
    -------
    pd.DataFrame with columns: group_a, group_b, n_a, n_b, mean_rank_a,
        mean_rank_b, z, p_uncorrected, p_bonf
    """
    all_vals = np.concatenate(list(groups.values()))
    N = len(all_vals)
    ranks = stats.rankdata(all_vals)

    idx = 0
    mean_ranks: Dict[str, float] = {}
    ns: Dict[str, int] = {}
    for name, arr in groups.items():
        n = len(arr)
        mean_ranks[name] = ranks[idx : idx + n].mean()
        ns[name] = n
        idx += n

    names = list(groups.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            na, nb = ns[a], ns[b]
            se = np.sqrt((N * (N + 1) / 12.0) * (1.0 / na + 1.0 / nb))
            z = (mean_ranks[a] - mean_ranks[b]) / se
            p_unc = float(2.0 * stats.norm.sf(abs(z)))
            rows.append(
                dict(
                    group_a=a,
                    group_b=b,
                    n_a=na,
                    n_b=nb,
                    mean_rank_a=round(mean_ranks[a], 1),
                    mean_rank_b=round(mean_ranks[b], 1),
                    z=round(z, 3),
                    p_uncorrected=p_unc,
                )
            )

    df = pd.DataFrame(rows)
    n_comparisons = len(df)
    df["p_bonf"] = np.minimum(df["p_uncorrected"] * n_comparisons, 1.0)
    return df


# ── Summary table ──────────────────────────────────────────────────────────────

def _fmt_mean_sd(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "—"
    return f"{np.mean(arr):.2f} ± {np.std(arr, ddof=1):.2f}"


def _fmt_median_iqr(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "—"
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    return f"{q2:.2f} [{q1:.2f}–{q3:.2f}]"


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def build_summary_table(
    df: pd.DataFrame,
    cog_vars: Dict[str, str],
    var_suffix: str = "",
) -> pd.DataFrame:
    """
    Build the main results table.
    One row per cognitive variable. Columns:
      - N, mean±SD, median[IQR] per group
      - KW H, KW p, FDR-adj p, epsilon²
      - Pairwise Bonferroni-corrected p for each pair
    """
    rows = []
    kw_pvalues = []
    var_list = []

    for var, label in cog_vars.items():
        col = f"{var}{var_suffix}" if var_suffix else var
        if col not in df.columns or not df[col].notna().any():
            continue
        groups = _group_arrays(df, col)
        h, p_kw = kruskal_wallis(groups)
        n_total = sum(len(a) for a in groups.values())
        eps2 = epsilon_squared(h, n_total)
        dunn = dunn_pairwise(groups)

        row = {"variable": label}
        for g in GROUP_ORDER:
            if g not in groups:
                row[f"n_{GROUP_SHORT[g]}"] = 0
                row[f"mean_sd_{GROUP_SHORT[g]}"] = "—"
                row[f"median_iqr_{GROUP_SHORT[g]}"] = "—"
            else:
                arr = groups[g]
                row[f"n_{GROUP_SHORT[g]}"] = len(arr)
                row[f"mean_sd_{GROUP_SHORT[g]}"] = _fmt_mean_sd(arr)
                row[f"median_iqr_{GROUP_SHORT[g]}"] = _fmt_median_iqr(arr)

        row["KW_H"] = round(h, 2)
        row["KW_p"] = p_kw
        row["epsilon2"] = round(eps2, 4)

        # Pairwise columns
        for _, pair in dunn.iterrows():
            key = f"p_{GROUP_SHORT[pair['group_a']]}_vs_{GROUP_SHORT[pair['group_b']]}"
            row[key] = pair["p_bonf"]

        rows.append(row)
        kw_pvalues.append(p_kw)
        var_list.append(var)

    df_table = pd.DataFrame(rows)

    # FDR across all variables
    if kw_pvalues:
        _, fdr_pvals, _, _ = multipletests(kw_pvalues, method="fdr_bh")
        df_table["FDR_p"] = fdr_pvals

    return df_table


# ── Figures ────────────────────────────────────────────────────────────────────

def _annotate_pair(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y_top: float,
    p: float,
    bar_height: float = 0.03,
) -> float:
    """Draw a significance bracket between two x positions. Returns new y ceiling."""
    stars = _sig_stars(p)
    if stars == "ns":
        return y_top
    y = y_top * (1 + bar_height)
    ax.plot([x1, x1, x2, x2], [y_top, y, y, y_top], lw=0.8, c="k")
    ax.text((x1 + x2) / 2, y * (1 + bar_height / 2), stars,
            ha="center", va="bottom", fontsize=7)
    return y * (1 + bar_height)


def plot_cognitive_by_rbd_group(
    df: pd.DataFrame,
    cog_vars: Dict[str, str],
    var_suffix: str = "",
    title_suffix: str = "",
    out_path: Optional[Path] = None,
) -> None:
    """
    3×3 grid of violin + box plots, one per cognitive variable.
    Groups are color-coded by LOW / INTERMEDIATE / HIGH.
    Significance brackets drawn for Bonferroni-corrected Dunn pairwise p < 0.05.
    """
    avail = [
        (var, label)
        for var, label in cog_vars.items()
        if (f"{var}{var_suffix}" if var_suffix else var) in df.columns
        and df[(f"{var}{var_suffix}" if var_suffix else var)].notna().any()
    ]

    ncols = 3
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()

    for idx, (var, label) in enumerate(avail):
        ax = axes[idx]
        col = f"{var}{var_suffix}" if var_suffix else var
        groups = _group_arrays(df, col)
        dunn = dunn_pairwise(groups)

        xs = list(range(len(GROUP_ORDER)))
        data_by_x = []
        for xi, g in enumerate(GROUP_ORDER):
            arr = groups.get(g, np.array([]))
            data_by_x.append(arr)

            if len(arr) == 0:
                continue

            # Violin
            parts = ax.violinplot(
                arr, positions=[xi], widths=0.7,
                showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(GROUP_COLORS[g])
                pc.set_alpha(0.4)
                pc.set_edgecolor("none")

            # Box (thin)
            bp = ax.boxplot(
                arr, positions=[xi], widths=0.18,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor=GROUP_COLORS[g], alpha=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=1, alpha=0.2,
                                markerfacecolor=GROUP_COLORS[g]),
            )

        # Significance brackets — only Bonferroni-corrected significant pairs
        y_top = max(
            (np.percentile(a, 95) for a in data_by_x if len(a) > 0),
            default=1.0,
        )
        pair_positions = {
            (GROUP_ORDER[0], GROUP_ORDER[1]): (0, 1),
            (GROUP_ORDER[0], GROUP_ORDER[2]): (0, 2),
            (GROUP_ORDER[1], GROUP_ORDER[2]): (1, 2),
        }
        for _, row in dunn.iterrows():
            pair = (row["group_a"], row["group_b"])
            if pair not in pair_positions:
                continue
            xi, xj = pair_positions[pair]
            y_top = _annotate_pair(ax, xi, xj, y_top, row["p_bonf"])

        # KW global p in title
        h, p_kw = kruskal_wallis(groups)
        n_total = sum(len(a) for a in data_by_x)
        eps2 = epsilon_squared(h, n_total)
        pstr = f"p={p_kw:.3g}" if p_kw >= 0.001 else "p<0.001"
        ax.set_title(f"{label}\n{pstr}, ε²={eps2:.3f}", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER], fontsize=8)
        ax.set_ylabel(label if not var_suffix else f"{label} (adj.)", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused axes
    for idx in range(len(avail), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    handles = [
        mpatches.Patch(facecolor=GROUP_COLORS[g], label=g, alpha=0.7)
        for g in GROUP_ORDER
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)

    suptitle = f"Cognitive Performance by RBD Risk Group — Controls Only"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved -> {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full cognitive × RBD risk group comparison pipeline."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  COGNITIVE PERFORMANCE × RBD RISK GROUP — CONTROLS ONLY")
    print("=" * 72)

    # 1. Load data
    print("\n[1] Loading data ...")
    df = load_controls()
    cog_vars = filter_available_cog_vars(df)
    print(f"  Cognitive variables available: {len(cog_vars)}")

    # 2. Age/sex residualisation
    print("\n[2] Residualising cognitive scores on age + sex ...")
    df = residualise_age_sex(df, list(cog_vars.keys()))

    # 3. Raw analysis
    print("\n[3] Building raw summary table (Kruskal-Wallis + Dunn + FDR) ...")
    table_raw = build_summary_table(df, cog_vars, var_suffix="")
    path_table_raw = OUT_DIR / "table_cognitive_by_rg3_raw.csv"
    table_raw.to_csv(path_table_raw, index=False)
    print(f"  Saved -> {path_table_raw}")

    # 4. Adjusted analysis
    print("\n[4] Building age/sex-adjusted summary table ...")
    table_adj = build_summary_table(df, cog_vars, var_suffix="_adj")
    path_table_adj = OUT_DIR / "table_cognitive_by_rg3_adjusted.csv"
    table_adj.to_csv(path_table_adj, index=False)
    print(f"  Saved -> {path_table_adj}")

    # 5. Figures
    print("\n[5] Generating figures ...")
    plot_cognitive_by_rbd_group(
        df, cog_vars,
        var_suffix="",
        title_suffix="Raw scores",
        out_path=OUT_DIR / "fig_cognitive_by_rg3_raw.png",
    )
    plot_cognitive_by_rbd_group(
        df, cog_vars,
        var_suffix="_adj",
        title_suffix="Age- and sex-adjusted residuals",
        out_path=OUT_DIR / "fig_cognitive_by_rg3_adjusted.png",
    )

    # 6. Print headline results
    print("\n" + "=" * 72)
    print("  HEADLINE RESULTS (raw scores)")
    print("=" * 72)
    cols_show = ["variable", "KW_H", "KW_p", "FDR_p", "epsilon2"]
    pairwise_cols = [c for c in table_raw.columns if c.startswith("p_") and "_vs_" in c]
    print(table_raw[cols_show + pairwise_cols].to_string(index=False))

    print("\n" + "=" * 72)
    print("  HEADLINE RESULTS (age/sex adjusted)")
    print("=" * 72)
    cols_show_adj = ["variable", "KW_H", "KW_p", "FDR_p", "epsilon2"]
    pairwise_cols_adj = [c for c in table_adj.columns if c.startswith("p_") and "_vs_" in c]
    print(table_adj[cols_show_adj + pairwise_cols_adj].to_string(index=False))

    print(f"\n  All outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
