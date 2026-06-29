"""
Prodromal Delta Analysis by RBD Group — Controls Only

Pipeline:
1. Load controls from parquet
2. Compute prodromal burden (baseline + incident)
3. Generate audit tables (coverage, prevalence)
4. Residualize delta on age + sex
5. Run Kruskal-Wallis on prodromal burden delta by RBD group
6. Run chi-square tests on individual markers by RBD group
7. Apply Benjamini-Hochberg FDR across all tests
8. Generate results tables + figures
9. Create comprehensive report

Output:
- audit_prodromal_delta_overall.csv
- audit_prodromal_delta_by_rbd_group.csv
- audit_prodromal_markers_overall.csv
- audit_prodromal_markers_by_rbd_group.csv
- results_prodromal_burden_kruskal_wallis.csv
- results_prodromal_markers_chisquare.csv
- results_prodromal_delta_summary.csv
- figures (bar charts)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_DIR: Path = Path(__file__).parent
PROJECT_DIR: Path = SCRIPT_DIR.parent
PARQUET_PATH: Path = PROJECT_DIR / "data" / "pp" / "res_build_final_dataset" / "ehr_diag_pd_rbd_only_all.parquet"
RG_COL: str = "rg_pctl3"
AGE_COL: str = "cov_age_recruitment_21022"
SEX_COL: str = "cov_sex_31"

PRODROMAL_MARKERS: list[str] = [
    "constipation",
    "depression",
    "anxiety",
    "orthostatic",
    "erectile_dysfunction",
    "dream_enactment",
    "anosmia",
    "hyposmia",
]

GROUP_ORDER: list[str] = ["Low", "Mid", "High"]
GROUP_COLORS: dict[str, str] = {
    "Low": "#4878d0",
    "Mid": "#ee854a",
    "High": "#d65f5f",
}

OUT_DIR: Path = SCRIPT_DIR / "results" / "prodromal_delta_analysis"
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


def compute_prodromal_burden(
    df: pd.DataFrame,
    markers: list[str],
    suffix: str = "_bl",
) -> pd.Series:
    """Compute sum of prodromal markers for given suffix."""
    cols = [f"prodromal_{m}{suffix}" for m in markers]
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return pd.Series(np.nan, index=df.index)
    return df[existing_cols].fillna(0).sum(axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: AUDIT (EMBEDDED)
# ──────────────────────────────────────────────────────────────────────────────

def audit_burden_overall(df: pd.DataFrame) -> dict:
    """Audit prodromal burden overall."""
    burden_bl = compute_prodromal_burden(df, PRODROMAL_MARKERS, "_bl")
    burden_post = compute_prodromal_burden(df, PRODROMAL_MARKERS, "_post")

    return {
        "n_controls": len(df),
        "n_with_bl_data": int(burden_bl.notna().sum()),
        "n_with_post_data": int(burden_post.notna().sum()),
        "median_burden_bl": float(burden_bl.median()),
        "median_burden_post": float(burden_post.median()),
        "pct_with_any_bl": round(100 * (burden_bl > 0).sum() / len(df), 1),
        "pct_with_any_post": round(100 * (burden_post > 0).sum() / len(df), 1),
    }


def audit_burden_by_rbd(df: pd.DataFrame) -> pd.DataFrame:
    """Audit burden by RBD group."""
    burden_bl = compute_prodromal_burden(df, PRODROMAL_MARKERS, "_bl")
    burden_post = compute_prodromal_burden(df, PRODROMAL_MARKERS, "_post")
    burden_delta = burden_post

    rows = []
    for rg in sorted(df[RG_COL].unique()):
        df_rg = df[df[RG_COL] == rg]
        bl_rg = burden_bl.loc[df_rg.index]
        delta_rg = burden_delta.loc[df_rg.index]

        rows.append({
            "rbd_group": str(rg),
            "n": len(df_rg),
            "median_burden_bl": float(bl_rg.median()),
            "median_burden_delta": float(delta_rg.median()),
            "pct_with_any_post": round(100 * (delta_rg > 0).sum() / len(df_rg), 1),
            "mean_burden_post": round(burden_post.loc[df_rg.index].mean(), 2),
        })

    return pd.DataFrame(rows)


def audit_markers_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Audit individual markers."""
    rows = []

    for marker in PRODROMAL_MARKERS:
        col_bl = f"prodromal_{marker}_bl"
        col_post = f"prodromal_{marker}_post"

        has_bl = col_bl in df.columns
        has_post = col_post in df.columns

        if not (has_bl and has_post):
            continue

        n_bl = (df[col_bl] == 1).sum()
        n_post = (df[col_post] == 1).sum()

        rows.append({
            "marker": marker,
            "n_bl": int(n_bl),
            "pct_bl": round(100 * n_bl / len(df), 2),
            "n_post": int(n_post),
            "pct_post": round(100 * n_post / len(df), 2),
        })

    return pd.DataFrame(rows)


def audit_markers_by_rbd(df: pd.DataFrame) -> pd.DataFrame:
    """Audit markers by RBD group."""
    rows = []

    for marker in PRODROMAL_MARKERS:
        col_post = f"prodromal_{marker}_post"

        if col_post not in df.columns:
            continue

        for rg in sorted(df[RG_COL].unique()):
            df_rg = df[df[RG_COL] == rg]
            n_pos = (df_rg[col_post] == 1).sum()
            n_total = len(df_rg)
            pct = 100 * n_pos / n_total if n_total > 0 else 0

            rows.append({
                "marker": marker,
                "rbd_group": str(rg),
                "n_group": n_total,
                "n_with_marker": int(n_pos),
                "pct_with_marker": round(pct, 2),
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: RESIDUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def residualize_burden_age_sex(
    df: pd.DataFrame,
    burden_col: str,
    age_col: str = AGE_COL,
    sex_col: str = SEX_COL,
) -> pd.Series:
    """Residualize burden on age + sex."""
    sub = df[[burden_col, age_col, sex_col]].dropna().copy()
    if len(sub) < 100:
        return pd.Series(np.nan, index=df.index)

    sub[burden_col] = sub[burden_col].astype(float)
    sub[age_col] = sub[age_col].astype(float)
    sub[sex_col] = sub[sex_col].astype(float)

    formula = f"{burden_col} ~ {age_col} + {sex_col}"
    model = ols(formula, data=sub).fit()

    result = pd.Series(np.nan, index=df.index)
    result.loc[sub.index] = model.resid.values
    return result


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: KRUSKAL-WALLIS ON BURDEN DELTA
# ──────────────────────────────────────────────────────────────────────────────

def analyze_burden_delta(df: pd.DataFrame) -> tuple[float, float, float]:
    """Run Kruskal-Wallis on residualized prodromal burden delta."""
    burden_post = compute_prodromal_burden(df, PRODROMAL_MARKERS, "_post")
    delta_resid = residualize_burden_age_sex(df, "prodromal_burden_delta")
    df["prodromal_burden_delta"] = burden_post
    delta_resid = residualize_burden_age_sex(df, "prodromal_burden_delta")

    groups = {
        g: df.loc[(df[RG_COL] == g) & delta_resid.notna(), "tmp"].fillna(0).values
        for g in GROUP_ORDER
        if g in df[RG_COL].values
    }

    # Recreate groups with residualized delta
    arrays = []
    for g in GROUP_ORDER:
        if g in df[RG_COL].values:
            mask = (df[RG_COL] == g) & delta_resid.notna()
            arrays.append(delta_resid[mask].values)

    if len(arrays) < 2:
        return np.nan, np.nan, np.nan

    h, p = stats.kruskal(*arrays)
    n_total = sum(len(a) for a in arrays)
    eps2 = h / (n_total - 1) if n_total > 1 else np.nan

    return float(h), float(p), float(eps2)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: CHI-SQUARE ON INDIVIDUAL MARKERS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_marker_chisquare(df: pd.DataFrame) -> pd.DataFrame:
    """Run chi-square tests for each marker by RBD group."""
    results = []

    for marker in PRODROMAL_MARKERS:
        col_post = f"prodromal_{marker}_post"

        if col_post not in df.columns:
            continue

        # Create contingency table
        contingency = pd.crosstab(df[RG_COL], df[col_post])

        if contingency.shape[1] < 2:
            continue  # Skip if only 1 category

        chi2, p_chi2, dof, expected = chi2_contingency(contingency)

        # Calculate % with marker per group
        marker_pcts = {}
        for rg in GROUP_ORDER:
            if rg in df[RG_COL].values:
                n_pos = (df[df[RG_COL] == rg][col_post] == 1).sum()
                n_total = len(df[df[RG_COL] == rg])
                marker_pcts[rg] = round(100 * n_pos / n_total, 2) if n_total > 0 else 0

        results.append({
            "marker": marker,
            "chi2": round(chi2, 2),
            "p_chi2": p_chi2,
            "dof": dof,
            "pct_low": marker_pcts.get(GROUP_ORDER[0], 0),
            "pct_int": marker_pcts.get(GROUP_ORDER[1], 0),
            "pct_high": marker_pcts.get(GROUP_ORDER[2], 0),
        })

    df_results = pd.DataFrame(results)

    # Apply FDR
    if len(df_results) > 0 and "p_chi2" in df_results.columns:
        _, fdr_pvals, _, _ = multipletests(df_results["p_chi2"], method="fdr_bh")
        df_results["fdr_p"] = [round(p, 4) for p in fdr_pvals]

    return df_results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: FIGURES
# ──────────────────────────────────────────────────────────────────────────────

def plot_marker_by_rbd(
    df_markers_rbd: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    """Bar plot: % with incident marker by RBD group."""
    markers_present = df_markers_rbd["marker"].unique()

    n_markers = len(markers_present)
    ncols = 3
    nrows = int(np.ceil(n_markers / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for idx, marker in enumerate(markers_present):
        ax = axes[idx]
        df_m = df_markers_rbd[df_markers_rbd["marker"] == marker]

        x_pos = np.arange(len(GROUP_ORDER))
        heights = []
        for rg in GROUP_ORDER:
            row = df_m[df_m["rbd_group"] == rg]
            pct = row["pct_with_marker"].values[0] if len(row) > 0 else 0
            heights.append(pct)

        colors = [GROUP_COLORS[rg] for rg in GROUP_ORDER]
        ax.bar(x_pos, heights, color=colors, alpha=0.7, edgecolor="black", linewidth=1)

        ax.set_title(f"{marker.replace('_', ' ').title()}", fontsize=10)
        ax.set_ylabel("% with incident marker", fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([g.split("(")[0].strip() for g in GROUP_ORDER], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for idx in range(n_markers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Incident Prodromal Markers by RBD Group", fontsize=12, y=0.995)
    fig.tight_layout()

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
    print("  PRODROMAL DELTA ANALYSIS BY RBD GROUP — CONTROLS ONLY")
    print("=" * 80)

    # 1. Load
    print("\n[1] Loading controls ...")
    df = load_controls()
    print(f"  Controls: {len(df):,}")

    # 2. Audit (generate and save)
    print("\n[2] Generating audit tables ...")
    audit_overall = audit_burden_overall(df)
    df_audit_burden_rbd = audit_burden_by_rbd(df)
    df_audit_markers = audit_markers_overall(df)
    df_audit_markers_rbd = audit_markers_by_rbd(df)

    path_audit_burden_rbd = OUT_DIR / "audit_prodromal_burden_by_rbd_group.csv"
    df_audit_burden_rbd.to_csv(path_audit_burden_rbd, index=False)
    print(f"  Saved: {path_audit_burden_rbd}")

    path_audit_markers = OUT_DIR / "audit_prodromal_markers_overall.csv"
    df_audit_markers.to_csv(path_audit_markers, index=False)
    print(f"  Saved: {path_audit_markers}")

    path_audit_markers_rbd = OUT_DIR / "audit_prodromal_markers_by_rbd_group.csv"
    df_audit_markers_rbd.to_csv(path_audit_markers_rbd, index=False)
    print(f"  Saved: {path_audit_markers_rbd}")

    print("\n  Audit Summary:")
    print(f"    Total controls: {audit_overall['n_controls']:,}")
    print(f"    With baseline prodromal data: {audit_overall['n_with_bl_data']:,}")
    print(f"    With incident prodromal data: {audit_overall['n_with_post_data']:,}")
    print(f"    % with any incident marker: {audit_overall['pct_with_any_post']:.1f}%")

    # 3. Analysis
    print("\n[3] Running prodromal analyses ...")

    # Burden Kruskal-Wallis
    burden_post = compute_prodromal_burden(df, PRODROMAL_MARKERS, "_post")
    df["prodromal_burden_delta"] = burden_post
    delta_resid = residualize_burden_age_sex(df, "prodromal_burden_delta")

    arrays = []
    for g in GROUP_ORDER:
        if g in df[RG_COL].values:
            mask = (df[RG_COL] == g) & delta_resid.notna()
            arrays.append(delta_resid[mask].values)

    h, p_kw = stats.kruskal(*arrays) if len(arrays) >= 2 else (np.nan, np.nan)
    n_total = sum(len(a) for a in arrays)
    eps2 = h / (n_total - 1) if n_total > 1 and not np.isnan(h) else np.nan

    df_burden = pd.DataFrame([{
        "test": "Prodromal Burden Delta",
        "n_total": int(n_total),
        "kw_h": round(h, 2),
        "kw_p": round(p_kw, 4) if not np.isnan(p_kw) else np.nan,
        "epsilon2": round(eps2, 4) if not np.isnan(eps2) else np.nan,
    }])

    path_burden = OUT_DIR / "results_prodromal_burden_kruskal_wallis.csv"
    df_burden.to_csv(path_burden, index=False)
    print(f"  Saved: {path_burden}")

    print(f"    Burden delta KW p = {p_kw:.4f}")

    # Individual markers chi-square
    df_markers_chi2 = analyze_marker_chisquare(df)

    path_markers_chi2 = OUT_DIR / "results_prodromal_markers_chisquare.csv"
    df_markers_chi2.to_csv(path_markers_chi2, index=False)
    print(f"  Saved: {path_markers_chi2}")

    if len(df_markers_chi2) > 0:
        print(f"\n  Marker chi-square results (FDR-corrected):")
        for _, row in df_markers_chi2.iterrows():
            sig = "***" if row["fdr_p"] < 0.05 else "ns"
            print(f"    {row['marker']:25s}  FDR p={row['fdr_p']:.4f}  {sig}")

    # 4. Figures
    print("\n[4] Generating figures ...")
    fig_path = FIG_DIR / "prodromal_markers_by_rbd.png"
    plot_marker_by_rbd(df_audit_markers_rbd, fig_path)
    print(f"  Saved: {fig_path}")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
