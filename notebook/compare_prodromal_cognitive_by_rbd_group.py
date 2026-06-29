"""
RBD Risk Group × Post-Baseline Prodromal + Cognitive Comparison
================================================================
Compare post-baseline cognitive (follow-up + change) and incident prodromal
markers across three RBD risk strata (Low / Mid / High).

Cohorts:
  1. Full cohort  — all subjects after standard exclusions
  2. Controls only — subjects with control == True

Source  : data/pp/res_build_final_dataset/ehr_diag_pd_rbd_only_all.parquet
Grouping: rg_pctl3 (Low / Mid / High)

Cognitive variables (all post-baseline):
  _fu    : follow-up scores at imaging visit i2 (~29–36% coverage)
  _delta : change from baseline i0→i2; TMT delta included (online i0 vs
           clinic i2 paradigm difference noted in outputs)

Prodromal variables (binary, incident post-baseline):
  Post-baseline = first event > wear_time_start + 182d AND before
  censoring/event date AND baseline flag == 0.
  dream_enactment and hyposmia excluded (structural zeros in HES).

Statistical framework:
  Continuous (cog _fu, _delta, prodromal_burden_post):
    Kruskal-Wallis global → Dunn post-hoc (Bonferroni within var)
    → BH-FDR across vars. Effect size: epsilon-squared (ε²).
    Run raw and age/sex OLS-residualised.
  Binary (prodromal _post):
    Chi-square global (3 groups) → pairwise 2×2 chi-square
    (Bonferroni × 3) → BH-FDR across vars. Effect size: Cramér's V.

Outputs (per cohort):
  results/rbd_group_comparison/{full_cohort,controls_only}/
    table_cognitive_fu_by_rg3_{raw,adj}.csv
    table_cognitive_delta_by_rg3_{raw,adj}.csv
    table_prodromal_burden_post_by_rg3.csv
    table_prodromal_post_by_rg3.csv
    fig_cognitive_fu_by_rg3_{raw,adj}.png
    fig_cognitive_delta_by_rg3_{raw,adj}.png
    fig_prodromal_post_by_rg3.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

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

GROUP_ORDER: List[str] = ["Low", "Mid", "High"]
GROUP_COLORS: Dict[str, str] = {
    "Low":  "#4878d0",
    "Mid":  "#ee854a",
    "High": "#d65f5f",
}

COG_FU_VARS: Dict[str, str] = {
    "cog_fluid_intelligence_fu": "Fluid Intelligence (FU)",
    "cog_react_time_fu":         "Reaction Time (ms, FU)",
    "cog_tmt1_dur_fu":           "TMT-A Duration (s, FU)",
    "cog_tmt2_dur_fu":           "TMT-B Duration (s, FU)",
    "cog_tmt_ratio_log_fu":      "Log TMT-B/A Ratio (FU)",
}

COG_DELTA_VARS: Dict[str, str] = {
    "cog_fluid_intelligence_delta": "Fluid Intelligence (Δ)",
    "cog_react_time_delta":         "Reaction Time (Δ ms)",
    "cog_tmt1_dur_delta":           "TMT-A Duration (Δ s)",
    "cog_tmt2_dur_delta":           "TMT-B Duration (Δ s)",
    "cog_tmt_ratio_log_delta":      "Log TMT-B/A Ratio (Δ)",
}

# TMT delta not pre-built in parquet (FI and RT deltas are); compute from _fu - _bl.
TMT_DELTA_COMPUTE: Dict[str, Tuple[str, str]] = {
    "cog_tmt1_dur_delta":      ("cog_tmt1_dur_fu",      "cog_tmt1_dur_bl"),
    "cog_tmt2_dur_delta":      ("cog_tmt2_dur_fu",      "cog_tmt2_dur_bl"),
    "cog_tmt_ratio_log_delta": ("cog_tmt_ratio_log_fu", "cog_tmt_ratio_log_bl"),
}

# dream_enactment and hyposmia excluded — structural zeros in inpatient HES.
PRODROMAL_POST_VARS: Dict[str, str] = {
    "prodromal_constipation_post":         "Constipation",
    "prodromal_depression_post":           "Depression",
    "prodromal_anxiety_post":              "Anxiety",
    "prodromal_orthostatic_post":          "Orthostatic Hypotension",
    "prodromal_erectile_dysfunction_post": "Erectile Dysfunction",
    "prodromal_anosmia_post":              "Anosmia",
}

PRODROMAL_BURDEN_COL: str = "prodromal_burden_post"

OUT_ROOT: Path = Path("results/rbd_group_comparison")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_cohort(controls_only: bool) -> pd.DataFrame:
    """
    Load merged parquet, apply standard exclusions via get_clean_risk_data,
    collapse to subject level, and optionally filter to controls.

    Parameters
    ----------
    controls_only : bool
        If True, retain only rows where control == True.

    Returns
    -------
    pd.DataFrame
        One row per subject.
    """
    _, df_night = get_clean_risk_data(file_name=FILE_NAME)
    df = make_subject_level(df_night, id_col="eid", prob_col="abk_rbd_score_mean")
    if controls_only:
        df = df[df["control"] == True].reset_index(drop=True)
    print(f"  N = {len(df):,}  |  controls_only={controls_only}")
    counts = df[RG_COL].value_counts().reindex(GROUP_ORDER)
    print(f"  RBD group distribution:\n{counts.to_string()}\n")
    return df


def add_tmt_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TMT delta columns (fu - bl) that are not pre-built in the parquet.
    FI and RT deltas are already present; this adds TMT-A, TMT-B, and ratio deltas.
    NaN propagates where either endpoint is missing.
    """
    df = df.copy()
    for col, (fu_col, bl_col) in TMT_DELTA_COMPUTE.items():
        if col not in df.columns:
            fu = pd.to_numeric(
                df.get(fu_col, pd.Series(np.nan, index=df.index)), errors="coerce"
            )
            bl = pd.to_numeric(
                df.get(bl_col, pd.Series(np.nan, index=df.index)), errors="coerce"
            )
            df[col] = fu - bl
    return df


def filter_available_vars(
    df: pd.DataFrame, vars_dict: Dict[str, str]
) -> Dict[str, str]:
    """Return vars that exist in df and have at least one non-null value."""
    available = {
        col: label
        for col, label in vars_dict.items()
        if col in df.columns and df[col].notna().any()
    }
    dropped = set(vars_dict) - set(available)
    if dropped:
        print(f"  Dropped (all-null or absent): {dropped}")
    return available


# ── Residualisation ────────────────────────────────────────────────────────────

def residualise_age_sex(
    df: pd.DataFrame,
    cog_vars: List[str],
    age_col: str = AGE_COL,
    sex_col: str = SEX_COL,
) -> pd.DataFrame:
    """
    OLS-residualise each continuous variable on age + sex.
    Adds column '{var}_adj' for each variable.

    Assumptions:
    - Linearity of age effect on the outcome (reasonable over 39–70 age range).
    - Sex treated as binary (UKBB: 0=female, 1=male).
    - Residuals retain group-level variance not explained by age/sex.
    """
    df_out = df.copy()
    for var in cog_vars:
        sub = df[[var, age_col, sex_col]].dropna().copy()
        if len(sub) < 100:
            df_out[f"{var}_adj"] = np.nan
            warnings.warn(
                f"  {var}: too few complete cases ({len(sub)}) — _adj set to NaN"
            )
            continue
        sub[var] = sub[var].astype(float)
        sub[age_col] = sub[age_col].astype(float)
        sub[sex_col] = sub[sex_col].astype(float)
        model = ols(f"{var} ~ {age_col} + {sex_col}", data=sub).fit()
        df_out.loc[sub.index, f"{var}_adj"] = model.resid.values
    return df_out


# ── Continuous statistics ──────────────────────────────────────────────────────

def _group_arrays(
    df: pd.DataFrame, var: str, rg_col: str = RG_COL
) -> Dict[str, np.ndarray]:
    """Extract non-null values per group as numpy arrays, preserving GROUP_ORDER."""
    return {
        g: df.loc[(df[rg_col] == g) & df[var].notna(), var].to_numpy(dtype=float)
        for g in GROUP_ORDER
        if g in df[rg_col].values
    }


def kruskal_wallis(groups: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Run Kruskal-Wallis across group arrays. Returns (H statistic, p-value)."""
    arrays = [arr for arr in groups.values() if len(arr) > 0]
    if len(arrays) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*arrays)
    return float(h), float(p)


def epsilon_squared(h: float, n_total: int) -> float:
    """
    Epsilon-squared effect size for Kruskal-Wallis: ε² = H / (N - 1).
    Interpretation: <0.01 negligible, 0.01–0.06 small, 0.06–0.14 medium, >0.14 large.
    Reference: Tomczak & Tomczak (2014).
    """
    if np.isnan(h) or n_total <= 1:
        return np.nan
    return float(h / (n_total - 1))


def dunn_pairwise(groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Dunn's post-hoc test for all pairwise comparisons.

    z_ij = (R̄_i - R̄_j) / sqrt[(N(N+1)/12) * (1/n_i + 1/n_j)]

    Bonferroni correction applied within each variable (multiplied by the
    number of pairwise comparisons = 3 for three groups).

    Returns
    -------
    pd.DataFrame with columns: group_a, group_b, n_a, n_b,
        mean_rank_a, mean_rank_b, z, p_uncorrected, p_bonf
    """
    all_vals = np.concatenate(list(groups.values()))
    N = len(all_vals)
    ranks = stats.rankdata(all_vals)

    idx = 0
    mean_ranks: Dict[str, float] = {}
    ns: Dict[str, int] = {}
    for name, arr in groups.items():
        n = len(arr)
        mean_ranks[name] = ranks[idx: idx + n].mean()
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
            rows.append(dict(
                group_a=a, group_b=b,
                n_a=na, n_b=nb,
                mean_rank_a=round(mean_ranks[a], 1),
                mean_rank_b=round(mean_ranks[b], 1),
                z=round(z, 3),
                p_uncorrected=p_unc,
            ))

    df_dunn = pd.DataFrame(rows)
    df_dunn["p_bonf"] = np.minimum(df_dunn["p_uncorrected"] * len(df_dunn), 1.0)
    return df_dunn


def _fmt_median_iqr(arr: np.ndarray) -> str:
    """Format as 'median [Q1–Q3]'."""
    if len(arr) == 0:
        return "—"
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    return f"{q2:.2f} [{q1:.2f}–{q3:.2f}]"


def _sig_stars(p: float) -> str:
    """Return significance stars for annotation."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def build_continuous_table(
    df: pd.DataFrame,
    vars_dict: Dict[str, str],
    var_suffix: str = "",
) -> pd.DataFrame:
    """
    Build results table for continuous variables.

    One row per variable. Columns: variable, N and median[IQR] per group,
    KW H, KW p, FDR-adjusted p, ε², pairwise Bonferroni-corrected Dunn p.
    BH-FDR correction applied across all variables in vars_dict.
    """
    rows = []
    kw_pvalues: List[float] = []

    for var, label in vars_dict.items():
        col = f"{var}{var_suffix}" if var_suffix else var
        if col not in df.columns or not df[col].notna().any():
            continue

        groups = _group_arrays(df, col)
        h, p_kw = kruskal_wallis(groups)
        n_total = sum(len(a) for a in groups.values())
        eps2 = epsilon_squared(h, n_total)
        dunn = dunn_pairwise(groups)

        row: Dict = {"variable": label}
        for g in GROUP_ORDER:
            arr = groups.get(g, np.array([]))
            row[f"n_{g}"] = len(arr)
            row[f"median_iqr_{g}"] = _fmt_median_iqr(arr)

        row["KW_H"] = round(h, 3) if not np.isnan(h) else np.nan
        row["KW_p"] = p_kw
        row["epsilon2"] = round(eps2, 4) if not np.isnan(eps2) else np.nan

        for _, pair_row in dunn.iterrows():
            row[f"p_{pair_row['group_a']}_vs_{pair_row['group_b']}"] = pair_row["p_bonf"]

        rows.append(row)
        kw_pvalues.append(p_kw)

    df_table = pd.DataFrame(rows)
    if kw_pvalues:
        valid = [not np.isnan(p) for p in kw_pvalues]
        fdr_vals = np.full(len(kw_pvalues), np.nan)
        if any(valid):
            _, corrected, _, _ = multipletests(
                [p for p, v in zip(kw_pvalues, valid) if v], method="fdr_bh"
            )
            fi = 0
            for i, is_valid in enumerate(valid):
                if is_valid:
                    fdr_vals[i] = corrected[fi]
                    fi += 1
        df_table["FDR_p"] = fdr_vals

    return df_table


# ── Binary statistics ──────────────────────────────────────────────────────────

def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    """Cramér's V = sqrt(χ² / (n * (min(r, c) - 1)))."""
    denom = n * (min(n_rows, n_cols) - 1)
    if denom <= 0 or np.isnan(chi2):
        return np.nan
    return float(np.sqrt(chi2 / denom))


def analyse_binary(
    df: pd.DataFrame, vars_dict: Dict[str, str]
) -> pd.DataFrame:
    """
    Chi-square test across RBD risk groups for each binary prodromal variable.

    Global test: 3-group contingency chi-square.
    Pairwise: 2×2 chi-square with Bonferroni correction (3 comparisons per var).
    BH-FDR applied across variables (global p-values).
    Reports N (%), chi2, p, FDR p, Cramér's V, pairwise Bonferroni p.
    """
    pairs: List[Tuple[str, str]] = [
        (GROUP_ORDER[i], GROUP_ORDER[j])
        for i in range(len(GROUP_ORDER))
        for j in range(i + 1, len(GROUP_ORDER))
    ]
    n_pairs = len(pairs)

    rows: List[Dict] = []
    global_pvals: List[float] = []

    for var, label in vars_dict.items():
        if var not in df.columns or not df[var].notna().any():
            continue

        sub = df[df[var].notna() & df[RG_COL].isin(GROUP_ORDER)].copy()
        sub[var] = sub[var].astype(float)

        row: Dict = {"variable": label}
        for g in GROUP_ORDER:
            g_mask = sub[RG_COL] == g
            n_g = int(g_mask.sum())
            n_pos = int(sub.loc[g_mask, var].sum())
            pct = (n_pos / n_g * 100) if n_g > 0 else np.nan
            row[f"n_{g}"] = n_g
            row[f"n_pos_{g}"] = n_pos
            row[f"pct_{g}"] = round(pct, 2) if not np.isnan(pct) else np.nan

        contingency = pd.crosstab(
            sub[RG_COL].reindex(sub.index), sub[var]
        ).reindex(index=[g for g in GROUP_ORDER if g in sub[RG_COL].values])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            row.update({"chi2": np.nan, "chi2_p": np.nan, "cramers_v": np.nan})
            for a, b in pairs:
                row[f"p_{a}_vs_{b}"] = np.nan
            rows.append(row)
            global_pvals.append(np.nan)
            continue

        chi2_stat, p_chi2, _, _ = stats.chi2_contingency(contingency)
        cv = cramers_v(chi2_stat, len(sub), contingency.shape[0], contingency.shape[1])

        row["chi2"] = round(chi2_stat, 3)
        row["chi2_p"] = p_chi2
        row["cramers_v"] = round(cv, 4) if not np.isnan(cv) else np.nan

        for a, b in pairs:
            pair_sub = sub[sub[RG_COL].isin([a, b])]
            ct = pd.crosstab(pair_sub[RG_COL], pair_sub[var])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                row[f"p_{a}_vs_{b}"] = np.nan
            else:
                _, p_pair, _, _ = stats.chi2_contingency(ct)
                row[f"p_{a}_vs_{b}"] = min(p_pair * n_pairs, 1.0)

        rows.append(row)
        global_pvals.append(p_chi2)

    df_table = pd.DataFrame(rows)
    valid = [not np.isnan(p) for p in global_pvals]
    fdr_vals = np.full(len(global_pvals), np.nan)
    if any(valid):
        _, corrected, _, _ = multipletests(
            [p for p, v in zip(global_pvals, valid) if v], method="fdr_bh"
        )
        fi = 0
        for i, is_valid in enumerate(valid):
            if is_valid:
                fdr_vals[i] = corrected[fi]
                fi += 1
    df_table["FDR_p"] = fdr_vals
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
    """Draw a significance bracket between two x positions. Returns updated y ceiling."""
    stars = _sig_stars(p)
    if stars == "ns":
        return y_top
    y = y_top * (1 + bar_height)
    ax.plot([x1, x1, x2, x2], [y_top, y, y, y_top], lw=0.8, c="k")
    ax.text(
        (x1 + x2) / 2, y * (1 + bar_height / 2), stars,
        ha="center", va="bottom", fontsize=7,
    )
    return y * (1 + bar_height)


def plot_continuous(
    df: pd.DataFrame,
    vars_dict: Dict[str, str],
    var_suffix: str = "",
    title_suffix: str = "",
    out_path: Optional[Path] = None,
) -> None:
    """
    3-column violin + box grid, one panel per variable.
    Significance brackets drawn for Bonferroni-corrected Dunn p < 0.05.
    """
    avail = [
        (var, label)
        for var, label in vars_dict.items()
        if (f"{var}{var_suffix}" if var_suffix else var) in df.columns
        and df[(f"{var}{var_suffix}" if var_suffix else var)].notna().any()
    ]
    if not avail:
        return

    ncols = 3
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (var, label) in enumerate(avail):
        ax = axes_flat[idx]
        col = f"{var}{var_suffix}" if var_suffix else var
        groups = _group_arrays(df, col)
        dunn = dunn_pairwise(groups)

        data_by_x = []
        for xi, g in enumerate(GROUP_ORDER):
            arr = groups.get(g, np.array([]))
            data_by_x.append(arr)
            if len(arr) == 0:
                continue
            parts = ax.violinplot(
                arr, positions=[xi], widths=0.7,
                showmedians=False, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(GROUP_COLORS[g])
                pc.set_alpha(0.4)
                pc.set_edgecolor("none")
            ax.boxplot(
                arr, positions=[xi], widths=0.18, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor=GROUP_COLORS[g], alpha=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=1, alpha=0.2,
                                markerfacecolor=GROUP_COLORS[g]),
            )

        y_top = max(
            (np.percentile(a, 95) for a in data_by_x if len(a) > 0), default=1.0
        )
        pair_positions = {
            (GROUP_ORDER[0], GROUP_ORDER[1]): (0, 1),
            (GROUP_ORDER[0], GROUP_ORDER[2]): (0, 2),
            (GROUP_ORDER[1], GROUP_ORDER[2]): (1, 2),
        }
        for _, pair_row in dunn.iterrows():
            pair = (pair_row["group_a"], pair_row["group_b"])
            if pair in pair_positions:
                xi, xj = pair_positions[pair]
                y_top = _annotate_pair(ax, xi, xj, y_top, pair_row["p_bonf"])

        h, p_kw = kruskal_wallis(groups)
        n_total = sum(len(a) for a in data_by_x)
        eps2 = epsilon_squared(h, n_total)
        pstr = f"p={p_kw:.3g}" if not np.isnan(p_kw) and p_kw >= 0.001 else "p<0.001"
        eps_str = f", ε²={eps2:.3f}" if not np.isnan(eps2) else ""
        ax.set_title(f"{label}\n{pstr}{eps_str}", fontsize=9)
        ax.set_xticks(list(range(len(GROUP_ORDER))))
        ax.set_xticklabels(GROUP_ORDER, fontsize=8)
        ax.set_ylabel(f"{label} (adj.)" if var_suffix else label, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(len(avail), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=GROUP_COLORS[g], label=g, alpha=0.7)
        for g in GROUP_ORDER
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)
    suptitle = "Cognitive Performance by RBD Risk Group"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved -> {out_path}")
    plt.close(fig)


def plot_prodromal(
    df: pd.DataFrame,
    vars_dict: Dict[str, str],
    out_path: Optional[Path] = None,
) -> None:
    """
    Grouped bar chart showing % prevalence of each incident prodromal marker
    per RBD risk group.
    """
    avail = [
        (var, label)
        for var, label in vars_dict.items()
        if var in df.columns and df[var].notna().any()
    ]
    if not avail:
        return

    n_vars = len(avail)
    bar_width = 0.25
    x = np.arange(n_vars)

    fig, ax = plt.subplots(figsize=(max(10, n_vars * 2), 5))

    for gi, g in enumerate(GROUP_ORDER):
        g_sub = df[df[RG_COL] == g]
        pcts = []
        for var, _ in avail:
            col_data = g_sub[var].dropna()
            pct = col_data.mean() * 100 if len(col_data) > 0 else 0.0
            pcts.append(pct)
        offset = (gi - 1) * bar_width
        ax.bar(x + offset, pcts, width=bar_width,
               color=GROUP_COLORS[g], alpha=0.8, label=g)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in avail], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Prevalence (%)", fontsize=10)
    ax.set_title("Post-Baseline Incident Prodromal Markers by RBD Risk Group", fontsize=11)
    ax.legend(title="RBD Risk", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved -> {out_path}")
    plt.close(fig)


def plot_prodromal_transitions(
    df: pd.DataFrame,
    vars_dict: Dict[str, str],
    out_path: Optional[Path] = None,
) -> None:
    """
    Slope chart (Sankey-style) showing the shift in prodromal marker prevalence
    from baseline (_bl) to incident post-baseline (_post) for each RBD risk group.

    For each prodromal variable (one subplot):
      - X-axis: two time points — Baseline and Post-baseline
      - Y-axis: prevalence (%) in that group at that time point
      - Three lines, one per RBD risk group (Low / Mid / High)

    Baseline prevalence  = % of subjects in group with _bl == 1
    Post-baseline rate   = % of subjects in group with _bl == 0 who have _post == 1
                           (incident only; denominator is subjects free at baseline)

    Note: these are cross-sectional proportions at two time windows, not
    repeated measures on the same subjects.
    """
    # Build parallel list of (post_col, bl_col, label) for available pairs
    avail: List[Tuple[str, str, str]] = []
    for post_col, label in vars_dict.items():
        # Derive the _bl counterpart by replacing the _post suffix
        bl_col = post_col.replace("_post", "_bl")
        if (
            post_col in df.columns and df[post_col].notna().any()
            and bl_col in df.columns and df[bl_col].notna().any()
        ):
            avail.append((post_col, bl_col, label))

    if not avail:
        return

    ncols = 3
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (post_col, bl_col, label) in enumerate(avail):
        ax = axes_flat[idx]

        for g in GROUP_ORDER:
            g_mask = df[RG_COL] == g
            g_sub = df[g_mask]

            # Baseline prevalence: % with _bl == 1
            bl_valid = g_sub[bl_col].dropna()
            pct_bl = bl_valid.mean() * 100 if len(bl_valid) > 0 else np.nan

            # Post-baseline incident rate: among those with _bl == 0, % with _post == 1
            at_risk = g_sub[g_sub[bl_col] == 0]
            post_valid = at_risk[post_col].dropna()
            pct_post = post_valid.mean() * 100 if len(post_valid) > 0 else np.nan

            if np.isnan(pct_bl) or np.isnan(pct_post):
                continue

            ax.plot(
                [0, 1], [pct_bl, pct_post],
                color=GROUP_COLORS[g], linewidth=2.0,
                marker="o", markersize=6, label=g,
            )
            # Annotate endpoints with value
            ax.text(-0.05, pct_bl, f"{pct_bl:.1f}%", ha="right", va="center",
                    fontsize=7, color=GROUP_COLORS[g])
            ax.text(1.05, pct_post, f"{pct_post:.1f}%", ha="left", va="center",
                    fontsize=7, color=GROUP_COLORS[g])

        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Post-baseline"], fontsize=9)
        ax.set_ylabel("Prevalence (%)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(len(avail), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=GROUP_COLORS[g], label=g, alpha=0.85)
        for g in GROUP_ORDER
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9, title="RBD Risk",
               framealpha=0.8)
    fig.suptitle(
        "Prodromal Marker Prevalence: Baseline → Post-Baseline by RBD Risk Group",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved -> {out_path}")
    plt.close(fig)


# ── Cohort runner ──────────────────────────────────────────────────────────────

def run_cohort(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    """Run the full analysis pipeline for one cohort and write all outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}")
    print(f"  COHORT: {label.upper()}")
    print(f"{'='*72}")

    df = add_tmt_deltas(df)

    cog_fu = filter_available_vars(df, COG_FU_VARS)
    cog_delta = filter_available_vars(df, COG_DELTA_VARS)
    prod_post = filter_available_vars(df, PRODROMAL_POST_VARS)

    print(f"  Cognitive FU vars: {len(cog_fu)}")
    print(f"  Cognitive delta vars: {len(cog_delta)}")
    print(f"  Prodromal post vars: {len(prod_post)}")

    # Residualise all continuous variables
    all_cont_vars = list({**cog_fu, **cog_delta}.keys())
    has_burden = (
        PRODROMAL_BURDEN_COL in df.columns
        and df[PRODROMAL_BURDEN_COL].notna().any()
    )
    if has_burden:
        all_cont_vars.append(PRODROMAL_BURDEN_COL)

    print("  Residualising continuous vars on age + sex ...")
    df = residualise_age_sex(df, all_cont_vars)

    # Cognitive follow-up
    if cog_fu:
        print("  Cognitive FU — raw ...")
        build_continuous_table(df, cog_fu).to_csv(
            out_dir / "table_cognitive_fu_by_rg3_raw.csv", index=False
        )
        print("  Cognitive FU — age/sex adjusted ...")
        build_continuous_table(df, cog_fu, var_suffix="_adj").to_csv(
            out_dir / "table_cognitive_fu_by_rg3_adj.csv", index=False
        )
        plot_continuous(
            df, cog_fu, var_suffix="",
            title_suffix="Follow-up scores — Raw",
            out_path=out_dir / "fig_cognitive_fu_by_rg3_raw.png",
        )
        plot_continuous(
            df, cog_fu, var_suffix="_adj",
            title_suffix="Follow-up scores — Age/sex adjusted",
            out_path=out_dir / "fig_cognitive_fu_by_rg3_adj.png",
        )

    # Cognitive delta
    if cog_delta:
        print("  Cognitive delta — raw ...")
        build_continuous_table(df, cog_delta).to_csv(
            out_dir / "table_cognitive_delta_by_rg3_raw.csv", index=False
        )
        print("  Cognitive delta — age/sex adjusted ...")
        build_continuous_table(df, cog_delta, var_suffix="_adj").to_csv(
            out_dir / "table_cognitive_delta_by_rg3_adj.csv", index=False
        )
        plot_continuous(
            df, cog_delta, var_suffix="",
            title_suffix="Change scores (FU − BL) — Raw",
            out_path=out_dir / "fig_cognitive_delta_by_rg3_raw.png",
        )
        plot_continuous(
            df, cog_delta, var_suffix="_adj",
            title_suffix="Change scores (FU − BL) — Age/sex adjusted",
            out_path=out_dir / "fig_cognitive_delta_by_rg3_adj.png",
        )

    # Prodromal burden (continuous)
    if has_burden:
        print("  Prodromal burden (continuous) ...")
        build_continuous_table(
            df, {PRODROMAL_BURDEN_COL: "Prodromal Burden (post-baseline count)"}
        ).to_csv(out_dir / "table_prodromal_burden_post_by_rg3.csv", index=False)

    # Prodromal binary
    if prod_post:
        print("  Prodromal post — binary ...")
        analyse_binary(df, prod_post).to_csv(
            out_dir / "table_prodromal_post_by_rg3.csv", index=False
        )
        plot_prodromal(
            df, prod_post,
            out_path=out_dir / "fig_prodromal_post_by_rg3.png",
        )
        plot_prodromal_transitions(
            df, prod_post,
            out_path=out_dir / "fig_prodromal_transitions_by_rg3.png",
        )

    print(f"\n  All outputs -> {out_dir.resolve()}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run analysis for full cohort then controls-only cohort."""
    cohorts = [
        (False, "Full cohort",    "full_cohort"),
        (True,  "Controls only",  "controls_only"),
    ]
    for controls_only, label, subdir in cohorts:
        print(f"\n[Loading] {label} ...")
        df = load_cohort(controls_only=controls_only)
        run_cohort(df, label=label, out_dir=OUT_ROOT / subdir)


if __name__ == "__main__":
    main()
