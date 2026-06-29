"""
Dual Prodromal Analysis: Prevalence at Follow-up + Incident Cases

Pipeline:
1. Load controls with valid actigraphy
2. Compute prevalence burden (sum of _post flags)
3. Compute incident burden (sum of _post where _bl=0)
4. Audit both analyses
5. Residualize on age + sex
6. Q1: Kruskal-Wallis + chi-square on prevalence
7. Q2: Kruskal-Wallis + chi-square on incident
8. Separate FDR correction per question
9. Generate all results tables + figures
10. Create comprehensive interpretation

Output:
- results/prodromal_prevalence_and_incident_analysis/
  - results_prodromal_prevalence_*.csv
  - results_prodromal_incident_*.csv
  - audit_prodromal_prevalence_*.csv
  - audit_prodromal_incident_*.csv
  - figures/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

OUT_DIR: Path = SCRIPT_DIR / "results" / "prodromal_prevalence_and_incident_analysis"
FIG_DIR: Path = OUT_DIR / "figures"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING & PREP
# ──────────────────────────────────────────────────────────────────────────────

def load_controls() -> pd.DataFrame:
    """Load controls from parquet, deduplicate by eid."""
    df = pd.read_parquet(PARQUET_PATH)
    df = df.groupby("eid").first().reset_index()
    df = df[df["control"] == True].reset_index(drop=True)
    return df


def compute_burden(
    df: pd.DataFrame,
    markers: list[str],
    suffix: str = "_post",
) -> pd.Series:
    """Compute sum of prodromal markers for given suffix."""
    cols = [f"prodromal_{m}{suffix}" for m in markers]
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return pd.Series(np.nan, index=df.index)
    return df[existing_cols].fillna(0).sum(axis=1)


def compute_incident_burden(df: pd.DataFrame, markers: list[str]) -> pd.Series:
    """Compute incident burden: sum of markers where post=1 AND baseline=0."""
    incident_flags = []
    for m in markers:
        col_bl = f"prodromal_{m}_bl"
        col_post = f"prodromal_{m}_post"
        if col_bl in df.columns and col_post in df.columns:
            is_incident = (df[col_post] == 1) & (df[col_bl] != 1)
            incident_flags.append(is_incident.astype(int))
    if not incident_flags:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(incident_flags, axis=1).sum(axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: AUDIT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def audit_prevalence_overall(df: pd.DataFrame) -> dict:
    """Audit prevalence burden overall."""
    burden_bl = compute_burden(df, PRODROMAL_MARKERS, "_bl")
    burden_post = compute_burden(df, PRODROMAL_MARKERS, "_post")

    return {
        "n_controls": len(df),
        "n_with_bl_data": int(burden_bl.notna().sum()),
        "n_with_post_data": int(burden_post.notna().sum()),
        "median_burden_bl": float(burden_bl.median()),
        "median_burden_post": float(burden_post.median()),
        "pct_with_any_bl": round(100 * (burden_bl > 0).sum() / len(df), 1),
        "pct_with_any_post": round(100 * (burden_post > 0).sum() / len(df), 1),
    }


def audit_incident_overall(df: pd.DataFrame) -> dict:
    """Audit incident burden overall."""
    incident_burden = compute_incident_burden(df, PRODROMAL_MARKERS)
    burden_bl = compute_burden(df, PRODROMAL_MARKERS, "_bl")

    return {
        "n_controls": len(df),
        "median_incident_burden": float(incident_burden.median()),
        "pct_developing_any": round(100 * (incident_burden > 0).sum() / len(df), 1),
        "median_burden_bl": float(burden_bl.median()),
    }


def audit_prevalence_by_rbd(df: pd.DataFrame) -> pd.DataFrame:
    """Audit prevalence by RBD group."""
    burden_bl = compute_burden(df, PRODROMAL_MARKERS, "_bl")
    burden_post = compute_burden(df, PRODROMAL_MARKERS, "_post")

    rows = []
    for rg in sorted(df[RG_COL].unique()):
        df_rg = df[df[RG_COL] == rg]
        bl_rg = burden_bl.loc[df_rg.index]
        post_rg = burden_post.loc[df_rg.index]

        rows.append({
            "rbd_group": str(rg),
            "n": len(df_rg),
            "median_burden_bl": float(bl_rg.median()),
            "median_burden_post": float(post_rg.median()),
            "pct_with_any_post": round(100 * (post_rg > 0).sum() / len(df_rg), 1),
        })

    return pd.DataFrame(rows)


def audit_incident_by_rbd(df: pd.DataFrame) -> pd.DataFrame:
    """Audit incident burden by RBD group."""
    incident_burden = compute_incident_burden(df, PRODROMAL_MARKERS)

    rows = []
    for rg in sorted(df[RG_COL].unique()):
        df_rg = df[df[RG_COL] == rg]
        incident_rg = incident_burden.loc[df_rg.index]

        rows.append({
            "rbd_group": str(rg),
            "n": len(df_rg),
            "median_incident_burden": float(incident_rg.median()),
            "pct_developing_any": round(100 * (incident_rg > 0).sum() / len(df_rg), 1),
        })

    return pd.DataFrame(rows)


def audit_markers_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Audit individual markers: baseline and post prevalence."""
    rows = []

    for marker in PRODROMAL_MARKERS:
        col_bl = f"prodromal_{marker}_bl"
        col_post = f"prodromal_{marker}_post"

        if not (col_bl in df.columns and col_post in df.columns):
            continue

        n_bl = (df[col_bl] == 1).sum()
        n_post = (df[col_post] == 1).sum()
        n_at_risk = ((df[col_bl] != 1) & df[col_post].notna()).sum()
        n_incident = ((df[col_post] == 1) & (df[col_bl] != 1)).sum()

        rows.append({
            "marker": marker,
            "n_bl": int(n_bl),
            "pct_bl": round(100 * n_bl / len(df), 2),
            "n_post": int(n_post),
            "pct_post": round(100 * n_post / len(df), 2),
            "n_at_risk": int(n_at_risk),
            "n_incident": int(n_incident),
            "pct_incident": round(100 * n_incident / n_at_risk, 2) if n_at_risk > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def audit_markers_by_rbd(df: pd.DataFrame, analysis_type: str = "prevalence") -> pd.DataFrame:
    """Audit markers by RBD group. analysis_type: 'prevalence' or 'incident'."""
    rows = []

    for marker in PRODROMAL_MARKERS:
        col_bl = f"prodromal_{marker}_bl"
        col_post = f"prodromal_{marker}_post"

        if col_post not in df.columns:
            continue

        for rg in sorted(df[RG_COL].unique()):
            df_rg = df[df[RG_COL] == rg]

            if analysis_type == "prevalence":
                n_pos = (df_rg[col_post] == 1).sum()
                n_total = len(df_rg)
                pct = 100 * n_pos / n_total if n_total > 0 else np.nan

                rows.append({
                    "marker": marker,
                    "rbd_group": str(rg),
                    "n": n_total,
                    "n_with_marker": int(n_pos),
                    "pct_with_marker": round(pct, 2) if not np.isnan(pct) else np.nan,
                })

            elif analysis_type == "incident":
                if col_bl not in df.columns:
                    continue
                n_at_risk = ((df_rg[col_bl] != 1) & df_rg[col_post].notna()).sum()
                n_incident = ((df_rg[col_post] == 1) & (df_rg[col_bl] != 1)).sum()
                pct = 100 * n_incident / n_at_risk if n_at_risk > 0 else np.nan

                rows.append({
                    "marker": marker,
                    "rbd_group": str(rg),
                    "n_at_risk": int(n_at_risk),
                    "n_incident": int(n_incident),
                    "pct_incident": round(pct, 2) if not np.isnan(pct) else np.nan,
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
    """Residualize burden on age + sex using OLS."""
    sub = df[[burden_col, age_col, sex_col]].dropna().copy()
    if len(sub) < 100:
        return pd.Series(np.nan, index=df.index)

    sub[burden_col] = sub[burden_col].astype(float)
    sub[age_col] = sub[age_col].astype(float)
    sub[sex_col] = sub[sex_col].astype(float)

    formula = f"{burden_col} ~ {age_col} + {sex_col}"
    try:
        model = ols(formula, data=sub).fit()
        result = pd.Series(np.nan, index=df.index)
        result.loc[sub.index] = model.resid.values
        return result
    except Exception:
        return pd.Series(np.nan, index=df.index)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Q1 - PREVALENCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_prevalence_burden(df: pd.DataFrame) -> tuple[float, float, float]:
    """Run Kruskal-Wallis on residualized prevalence burden."""
    burden_post = compute_burden(df, PRODROMAL_MARKERS, "_post")
    df_temp = df.copy()
    df_temp["prevalence_burden"] = burden_post
    delta_resid = residualize_burden_age_sex(df_temp, "prevalence_burden")

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


def analyze_prevalence_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Run chi-square tests on prevalence of individual markers."""
    results = []

    for marker in PRODROMAL_MARKERS:
        col_post = f"prodromal_{marker}_post"

        if col_post not in df.columns:
            continue

        contingency = pd.crosstab(df[RG_COL], df[col_post])

        if contingency.shape[1] < 2:
            continue

        chi2, p_chi2, dof, _ = chi2_contingency(contingency)

        marker_pcts = {}
        for rg in GROUP_ORDER:
            if rg in df[RG_COL].values:
                n_pos = (df[df[RG_COL] == rg][col_post] == 1).sum()
                n_total = len(df[df[RG_COL] == rg])
                marker_pcts[rg] = round(100 * n_pos / n_total, 2) if n_total > 0 else np.nan

        results.append({
            "marker": marker,
            "chi2": round(chi2, 2),
            "p_chi2": p_chi2,
            "dof": dof,
            "pct_low": marker_pcts.get("Low", np.nan),
            "pct_mid": marker_pcts.get("Mid", np.nan),
            "pct_high": marker_pcts.get("High", np.nan),
        })

    df_results = pd.DataFrame(results)

    if len(df_results) > 0 and "p_chi2" in df_results.columns:
        _, fdr_pvals, _, _ = multipletests(df_results["p_chi2"], method="fdr_bh")
        df_results["fdr_p"] = [round(p, 4) for p in fdr_pvals]

    return df_results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Q2 - INCIDENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_incident_burden(df: pd.DataFrame) -> tuple[float, float, float]:
    """Run Kruskal-Wallis on residualized incident burden."""
    incident_burden = compute_incident_burden(df, PRODROMAL_MARKERS)
    df_temp = df.copy()
    df_temp["incident_burden"] = incident_burden
    delta_resid = residualize_burden_age_sex(df_temp, "incident_burden")

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


def analyze_incident_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Run chi-square tests on incident markers (at-risk denominator)."""
    results = []

    for marker in PRODROMAL_MARKERS:
        col_bl = f"prodromal_{marker}_bl"
        col_post = f"prodromal_{marker}_post"

        if not (col_bl in df.columns and col_post in df.columns):
            continue

        df_at_risk = df[(df[col_bl] != 1) & df[col_post].notna()].copy()
        if len(df_at_risk) < 10:
            continue

        contingency = pd.crosstab(df_at_risk[RG_COL], df_at_risk[col_post])

        if contingency.shape[1] < 2:
            continue

        chi2, p_chi2, dof, _ = chi2_contingency(contingency)

        marker_pcts = {}
        n_at_risk_dict = {}
        for rg in GROUP_ORDER:
            df_rg_at_risk = df_at_risk[df_at_risk[RG_COL] == rg]
            if len(df_rg_at_risk) > 0:
                n_incident = (df_rg_at_risk[col_post] == 1).sum()
                n_at_risk = len(df_rg_at_risk)
                marker_pcts[rg] = round(100 * n_incident / n_at_risk, 2)
                n_at_risk_dict[rg] = n_at_risk

        results.append({
            "marker": marker,
            "chi2": round(chi2, 2),
            "p_chi2": p_chi2,
            "dof": dof,
            "pct_incident_low": marker_pcts.get("Low", np.nan),
            "pct_incident_mid": marker_pcts.get("Mid", np.nan),
            "pct_incident_high": marker_pcts.get("High", np.nan),
            "n_at_risk_low": n_at_risk_dict.get("Low", np.nan),
            "n_at_risk_mid": n_at_risk_dict.get("Mid", np.nan),
            "n_at_risk_high": n_at_risk_dict.get("High", np.nan),
        })

    df_results = pd.DataFrame(results)

    if len(df_results) > 0 and "p_chi2" in df_results.columns:
        _, fdr_pvals, _, _ = multipletests(df_results["p_chi2"], method="fdr_bh")
        df_results["fdr_p"] = [round(p, 4) for p in fdr_pvals]

    return df_results


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: FIGURES
# ──────────────────────────────────────────────────────────────────────────────

def plot_burden_by_rbd(
    df: pd.DataFrame,
    burden_col: str,
    title: str,
    out_path: Path | None = None,
) -> None:
    """Violin+box plot of burden by RBD group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_group = []
    for g in GROUP_ORDER:
        mask = (df[RG_COL] == g) & df[burden_col].notna()
        data_by_group.append(df.loc[mask, burden_col].values)

    positions = np.arange(len(GROUP_ORDER))
    parts = ax.violinplot(data_by_group, positions=positions, widths=0.7, showmeans=False, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor("#8888ff")
        pc.set_alpha(0.6)

    bp = ax.boxplot(data_by_group, positions=positions, widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor="#ccccff", alpha=0.7),
                    medianprops=dict(color="red", linewidth=2))

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_ORDER, fontsize=11)
    ax.set_ylabel("Burden (count)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_markers_by_rbd(
    df_markers_rbd: pd.DataFrame,
    title: str,
    pct_col: str,
    out_path: Path | None = None,
) -> None:
    """Bar plot: % with marker by RBD group."""
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
            pct = row[pct_col].values[0] if len(row) > 0 else 0
            heights.append(pct)

        colors = [GROUP_COLORS[rg] for rg in GROUP_ORDER]
        ax.bar(x_pos, heights, color=colors, alpha=0.7, edgecolor="black", linewidth=1)

        ax.set_title(f"{marker.replace('_', ' ').title()}", fontsize=10)
        ax.set_ylabel(pct_col.replace("_", " ").title(), fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(GROUP_ORDER, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(n_markers, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute complete dual prodromal analysis."""
    print("=" * 80)
    print("DUAL PRODROMAL ANALYSIS: PREVALENCE + INCIDENT")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/10] Loading controls...")
    df = load_controls()
    print(f"  N controls: {len(df)}")

    print("\n[2/10] Audit: Overall statistics...")
    prev_audit = audit_prevalence_overall(df)
    incident_audit = audit_incident_overall(df)
    print(f"  Prevalence burden (baseline median): {prev_audit['median_burden_bl']}")
    print(f"  Prevalence burden (post median): {prev_audit['median_burden_post']}")
    print(f"  Incident burden (median): {incident_audit['median_incident_burden']}")

    print("\n[3/10] Audit: By RBD group...")
    df_audit_prev_rbd = audit_prevalence_by_rbd(df)
    df_audit_inc_rbd = audit_incident_by_rbd(df)
    print(df_audit_prev_rbd.to_string(index=False))
    print("\nIncident by RBD:")
    print(df_audit_inc_rbd.to_string(index=False))

    print("\n[4/10] Audit: Individual markers...")
    df_audit_markers = audit_markers_overall(df)
    print(df_audit_markers.to_string(index=False))

    df_audit_markers_prev = audit_markers_by_rbd(df, analysis_type="prevalence")
    df_audit_markers_inc = audit_markers_by_rbd(df, analysis_type="incident")

    print("\n[5/10] Q1: Prevalence Burden Kruskal-Wallis...")
    h_prev, p_prev, eps2_prev = analyze_prevalence_burden(df)
    print(f"  H-stat: {h_prev:.2f}, p-value: {p_prev:.4f}, epsilon²: {eps2_prev:.4f}")

    print("\n[6/10] Q1: Prevalence Markers Chi-square (FDR separate)...")
    df_prev_markers = analyze_prevalence_markers(df)
    print(df_prev_markers[["marker", "chi2", "p_chi2", "fdr_p", "pct_low", "pct_mid", "pct_high"]].to_string(index=False))

    print("\n[7/10] Q2: Incident Burden Kruskal-Wallis...")
    h_inc, p_inc, eps2_inc = analyze_incident_burden(df)
    print(f"  H-stat: {h_inc:.2f}, p-value: {p_inc:.4f}, epsilon²: {eps2_inc:.4f}")

    print("\n[8/10] Q2: Incident Markers Chi-square (FDR separate)...")
    df_inc_markers = analyze_incident_markers(df)
    print(df_inc_markers[["marker", "chi2", "p_chi2", "fdr_p", "pct_incident_low", "pct_incident_mid", "pct_incident_high"]].to_string(index=False))

    print("\n[9/10] Writing outputs...")

    # Summary tables
    df_prev_summary = pd.DataFrame([{
        "analysis": "prevalence_burden",
        "h_stat": round(h_prev, 2),
        "p_value": round(p_prev, 4),
        "epsilon2": round(eps2_prev, 4),
    }])

    df_inc_summary = pd.DataFrame([{
        "analysis": "incident_burden",
        "h_stat": round(h_inc, 2),
        "p_value": round(p_inc, 4),
        "epsilon2": round(eps2_inc, 4),
    }])

    df_prev_summary.to_csv(OUT_DIR / "results_prodromal_prevalence_burden_kruskal_wallis.csv", index=False)
    df_inc_summary.to_csv(OUT_DIR / "results_prodromal_incident_burden_kruskal_wallis.csv", index=False)
    df_prev_markers.to_csv(OUT_DIR / "results_prodromal_prevalence_markers_chisquare.csv", index=False)
    df_inc_markers.to_csv(OUT_DIR / "results_prodromal_incident_markers_chisquare.csv", index=False)

    df_audit_prev_rbd.to_csv(OUT_DIR / "audit_prodromal_prevalence_by_rbd_group.csv", index=False)
    df_audit_inc_rbd.to_csv(OUT_DIR / "audit_prodromal_incident_by_rbd_group.csv", index=False)
    df_audit_markers.to_csv(OUT_DIR / "audit_prodromal_markers_overall.csv", index=False)

    print("\n[10/10] Generating figures...")

    burden_post = compute_burden(df, PRODROMAL_MARKERS, "_post")
    df_plot = df.copy()
    df_plot["prevalence_burden"] = burden_post
    plot_burden_by_rbd(df_plot, "prevalence_burden",
                       "Prevalence: Prodromal Burden at Follow-up by RBD Group",
                       FIG_DIR / "prodromal_prevalence_burden_by_rbd.png")

    incident_burden = compute_incident_burden(df, PRODROMAL_MARKERS)
    df_plot["incident_burden"] = incident_burden
    plot_burden_by_rbd(df_plot, "incident_burden",
                       "Incident: New Prodromal Symptoms Developed by RBD Group",
                       FIG_DIR / "prodromal_incident_burden_by_rbd.png")

    plot_markers_by_rbd(df_audit_markers_prev,
                        "Prevalence: % with Prodromal Marker at Follow-up by RBD Group",
                        "pct_with_marker",
                        FIG_DIR / "prodromal_prevalence_markers_by_rbd.png")

    plot_markers_by_rbd(df_audit_markers_inc,
                        "Incident: % Who Developed Prodromal Marker by RBD Group",
                        "pct_incident",
                        FIG_DIR / "prodromal_incident_markers_by_rbd.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs written to: {OUT_DIR}")
    print(f"Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
