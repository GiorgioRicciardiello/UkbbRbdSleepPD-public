"""
Cognitive Delta Analysis by RBD Group — Controls Only

Pipeline:
1. Load controls from parquet
2. Compute cognitive deltas (i2 - i0)
3. Generate audit tables (coverage, outliers, distribution)
4. Residualize deltas on age + sex
5. Run Kruskal-Wallis by RBD group per variable
6. Run Dunn post-hoc pairwise comparisons
7. Apply Benjamini-Hochberg FDR across 5 variables
8. Generate results tables + figures
9. Create comprehensive report

Output:
- audit_cognitive_delta_overall.csv
- audit_cognitive_delta_by_rbd_group.csv
- results_cognitive_delta_kruskal_wallis.csv
- results_cognitive_delta_pairwise.csv
- results_cognitive_delta_summary.csv
- figures (violin+box plots)
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import linregress

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_DIR: Path = Path(__file__).parent
PROJECT_DIR: Path = SCRIPT_DIR.parent
PARQUET_PATH: Path = PROJECT_DIR / "data" / "pp" / "res_build_final_dataset" / "ehr_diag_pd_rbd_only_all.parquet"
RG_COL: str = "rg_pctl3"
AGE_COL: str = "cov_age_recruitment_21022"
SEX_COL: str = "cov_sex_31"

# Cognitive variables with _bl and _fu variants
COG_VARS: dict[str, tuple[str, str]] = {
    "Reaction Time (ms)": ("cog_react_time_bl", "cog_react_time_fu"),
    "Fluid Intelligence": ("cog_fluid_intelligence_bl", "cog_fluid_intelligence_fu"),
    "TMT-A Duration (s)": ("cog_tmt1_dur_bl", "cog_tmt1_dur_fu"),
    "TMT-B Duration (s)": ("cog_tmt2_dur_bl", "cog_tmt2_dur_fu"),
    "Log TMT-B/A Ratio": ("cog_tmt_ratio_log_bl", "cog_tmt_ratio_log_fu"),
}

GROUP_ORDER: list[str] = ["Low", "Mid", "High"]
GROUP_COLORS: dict[str, str] = {
    "Low": "#4878d0",
    "Mid": "#ee854a",
    "High": "#d65f5f",
}
GROUP_SHORT: dict[str, str] = {
    "Low": "Low",
    "Mid": "Int",
    "High": "High",
}

OUT_DIR: Path = SCRIPT_DIR / "results" / "cognitive_delta_analysis"
FIG_DIR: Path = OUT_DIR / "figures"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING & PREP
# ──────────────────────────────────────────────────────────────────────────────

def load_controls() -> pd.DataFrame:
    """Load controls from parquet."""
    df = pd.read_parquet(PARQUET_PATH)
    df = df.groupby("eid").first().reset_index()
    df = df[df["control"] == True].reset_index(drop=True)
    return df


def compute_delta(df: pd.DataFrame, col_bl: str, col_fu: str) -> pd.Series:
    """Compute delta = fu - bl."""
    return df[col_fu] - df[col_bl]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: AUDIT (EMBEDDED)
# ──────────────────────────────────────────────────────────────────────────────

def audit_delta_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Audit overall delta coverage and distribution."""
    rows = []

    for var_name, (col_bl, col_fu) in COG_VARS.items():
        has_both = df[col_bl].notna() & df[col_fu].notna()
        n_total = has_both.sum()

        if n_total == 0:
            rows.append({
                "variable": var_name,
                "n_total": 0,
                "median_delta": np.nan,
                "q25_delta": np.nan,
                "q75_delta": np.nan,
                "min_delta": np.nan,
                "max_delta": np.nan,
                "pct_outliers_pos": np.nan,
                "pct_outliers_neg": np.nan,
            })
            continue

        delta = compute_delta(df.loc[has_both], col_bl, col_fu)
        mean_delta = delta.mean()
        std_delta = delta.std()
        outliers_pos = (delta > mean_delta + 3 * std_delta).sum()
        outliers_neg = (delta < mean_delta - 3 * std_delta).sum()

        rows.append({
            "variable": var_name,
            "n_total": int(n_total),
            "median_delta": float(delta.median()),
            "q25_delta": float(delta.quantile(0.25)),
            "q75_delta": float(delta.quantile(0.75)),
            "min_delta": float(delta.min()),
            "max_delta": float(delta.max()),
            "pct_outliers_pos": round(100 * outliers_pos / n_total, 2),
            "pct_outliers_neg": round(100 * outliers_neg / n_total, 2),
        })

    return pd.DataFrame(rows)


def audit_delta_by_rbd(df: pd.DataFrame) -> pd.DataFrame:
    """Audit delta by RBD group."""
    rows = []

    for var_name, (col_bl, col_fu) in COG_VARS.items():
        has_both = df[col_bl].notna() & df[col_fu].notna()
        df_pairs = df.loc[has_both].copy()
        df_pairs["delta"] = compute_delta(df_pairs, col_bl, col_fu)

        for rg in sorted(df_pairs[RG_COL].unique()):
            df_rg = df_pairs[df_pairs[RG_COL] == rg]
            delta = df_rg["delta"]

            rows.append({
                "variable": var_name,
                "rbd_group": str(rg),
                "n": len(df_rg),
                "median_delta": float(delta.median()),
                "q25": float(delta.quantile(0.25)),
                "q75": float(delta.quantile(0.75)),
                "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std()),
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: RESIDUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def residualize_delta_age_sex(
    df: pd.DataFrame,
    delta_col: str,
    age_col: str = AGE_COL,
    sex_col: str = SEX_COL,
) -> pd.Series:
    """Residualize delta on age + sex via linear regression."""
    sub = df[[delta_col, age_col, sex_col]].dropna().copy()
    if len(sub) < 100:
        return pd.Series(np.nan, index=df.index)

    y = sub[delta_col].astype(float).values
    X_age = sub[age_col].astype(float).values
    X_sex = sub[sex_col].astype(float).values

    # Fit regression: y ~ age + sex
    X = np.column_stack([np.ones(len(X_age)), X_age, X_sex])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ coef
    residuals = y - y_pred

    # Assign residuals back to original index
    result = pd.Series(np.nan, index=df.index)
    result.loc[sub.index] = residuals
    return result


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: KRUSKAL-WALLIS & PAIRWISE
# ──────────────────────────────────────────────────────────────────────────────

def _group_arrays(
    df: pd.DataFrame,
    var: str,
    rg_col: str = RG_COL,
) -> dict[str, np.ndarray]:
    """Extract non-null values per group."""
    return {
        g: df.loc[(df[rg_col] == g) & df[var].notna(), var].to_numpy(dtype=float)
        for g in GROUP_ORDER
        if g in df[rg_col].values
    }


def kruskal_wallis(groups: dict[str, np.ndarray]) -> tuple[float, float]:
    """Run Kruskal-Wallis test."""
    arrays = [arr for arr in groups.values() if len(arr) > 0]
    if len(arrays) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*arrays)
    return float(h), float(p)


def epsilon_squared(h: float, n_total: int) -> float:
    """Epsilon-squared effect size."""
    if np.isnan(h) or n_total <= 1:
        return np.nan
    return float(h / (n_total - 1))


def dunn_pairwise(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Dunn post-hoc pairwise comparisons."""
    # Handle empty groups
    if not groups or all(len(arr) == 0 for arr in groups.values()):
        return pd.DataFrame()

    all_vals = np.concatenate([arr for arr in groups.values() if len(arr) > 0])
    N = len(all_vals)
    ranks = stats.rankdata(all_vals)

    idx = 0
    mean_ranks: dict[str, float] = {}
    ns: dict[str, int] = {}
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
            rows.append({
                "group_a": a,
                "group_b": b,
                "n_a": na,
                "n_b": nb,
                "z": round(z, 3),
                "p_uncorrected": p_unc,
            })

    df_pairs = pd.DataFrame(rows)
    n_comparisons = len(df_pairs)
    df_pairs["p_bonf"] = np.minimum(df_pairs["p_uncorrected"] * n_comparisons, 1.0)
    return df_pairs


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: MAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_cognitive_delta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full cognitive delta analysis."""
    kw_results = []
    pairwise_results = []

    # Compute deltas and residualize
    for var_name, (col_bl, col_fu) in COG_VARS.items():
        delta_col = f"{var_name}_delta"
        df[delta_col] = compute_delta(df, col_bl, col_fu)
        resid_col = f"{var_name}_delta_resid"
        df[resid_col] = residualize_delta_age_sex(df, delta_col)

    # Kruskal-Wallis per variable
    kw_pvalues = []
    var_list = []

    for var_name, (col_bl, col_fu) in COG_VARS.items():
        resid_col = f"{var_name}_delta_resid"
        groups = _group_arrays(df, resid_col)
        h, p_kw = kruskal_wallis(groups)
        n_total = sum(len(a) for a in groups.values())
        eps2 = epsilon_squared(h, n_total)

        kw_results.append({
            "variable": var_name,
            "n_total": int(n_total),
            "kw_h": round(h, 2),
            "kw_p": p_kw,
            "epsilon2": round(eps2, 4),
        })

        kw_pvalues.append(p_kw)
        var_list.append(var_name)

        # Pairwise comparisons
        dunn = dunn_pairwise(groups)
        for _, row in dunn.iterrows():
            pairwise_results.append({
                "variable": var_name,
                "group_a": row["group_a"],
                "group_b": row["group_b"],
                "z": row["z"],
                "p_uncorrected": round(row["p_uncorrected"], 4),
                "p_bonf": round(row["p_bonf"], 4),
            })

    # Apply FDR across all variables
    df_kw = pd.DataFrame(kw_results)
    if kw_pvalues:
        _, fdr_pvals, _, _ = multipletests(kw_pvalues, method="fdr_bh")
        df_kw["fdr_p"] = [round(p, 4) for p in fdr_pvals]

    df_pairwise = pd.DataFrame(pairwise_results)

    return df_kw, df_pairwise, df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: FIGURES
# ──────────────────────────────────────────────────────────────────────────────

def plot_delta_by_rbd(
    df: pd.DataFrame,
    var_name: str,
    resid_col: str,
    out_path: Path | None = None,
) -> None:
    """Plot delta distribution by RBD group."""
    groups = _group_arrays(df, resid_col)

    fig, ax = plt.subplots(figsize=(8, 6))

    xs = list(range(len(GROUP_ORDER)))
    data_by_x = []
    for xi, g in enumerate(GROUP_ORDER):
        arr = groups.get(g, np.array([]))
        data_by_x.append(arr)

        if len(arr) == 0:
            continue

        # Violin
        parts = ax.violinplot(arr, positions=[xi], widths=0.7, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(GROUP_COLORS[g])
            pc.set_alpha(0.4)
            pc.set_edgecolor("none")

        # Box
        bp = ax.boxplot(
            arr, positions=[xi], widths=0.18, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor=GROUP_COLORS[g], alpha=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )

    h, p_kw = kruskal_wallis(groups)
    n_total = sum(len(a) for a in data_by_x)
    eps2 = epsilon_squared(h, n_total)
    pstr = f"p={p_kw:.3g}" if p_kw >= 0.001 else "p<0.001"

    ax.set_title(f"{var_name}\n{pstr}, epsilon-sq={eps2:.4f}", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER], fontsize=9)
    ax.set_ylabel("Residualized Delta", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run full analysis pipeline."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  COGNITIVE DELTA ANALYSIS BY RBD GROUP — CONTROLS ONLY")
    print("=" * 80)

    # 1. Load
    print("\n[1] Loading controls ...")
    df = load_controls()
    print(f"  Controls: {len(df):,}")

    # 2. Audit (generate and save)
    print("\n[2] Generating audit tables ...")
    df_audit_overall = audit_delta_overall(df)
    df_audit_rbd = audit_delta_by_rbd(df)

    path_audit_overall = OUT_DIR / "audit_cognitive_delta_overall.csv"
    df_audit_overall.to_csv(path_audit_overall, index=False)
    print(f"  Saved: {path_audit_overall}")

    path_audit_rbd = OUT_DIR / "audit_cognitive_delta_by_rbd_group.csv"
    df_audit_rbd.to_csv(path_audit_rbd, index=False)
    print(f"  Saved: {path_audit_rbd}")

    print("\n  Audit Summary:")
    print(df_audit_overall[["variable", "n_total", "median_delta"]].to_string(index=False))

    # 3. Analysis
    print("\n[3] Running Kruskal-Wallis analysis ...")
    df_kw, df_pairwise, df_with_residuals = analyze_cognitive_delta(df)

    path_kw = OUT_DIR / "results_cognitive_delta_kruskal_wallis.csv"
    df_kw.to_csv(path_kw, index=False)
    print(f"  Saved: {path_kw}")

    path_pairwise = OUT_DIR / "results_cognitive_delta_pairwise.csv"
    df_pairwise.to_csv(path_pairwise, index=False)
    print(f"  Saved: {path_pairwise}")

    print("\n  Results:")
    print(df_kw[["variable", "n_total", "kw_p", "fdr_p", "epsilon2"]].to_string(index=False))

    # 4. Figures
    print("\n[4] Generating figures ...")
    for var_name, (col_bl, col_fu) in COG_VARS.items():
        resid_col = f"{var_name}_delta_resid"
        fig_path = FIG_DIR / f"delta_{var_name.replace(' ', '_').lower()}.png"
        plot_delta_by_rbd(df_with_residuals, var_name, resid_col, fig_path)

    print(f"  Figures saved to: {FIG_DIR}")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
