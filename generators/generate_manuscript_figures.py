"""
Generate all figures and extended data tables cited in the manuscript.

Reads pre-computed result tables from the Cox prodromal pipeline output
and produces publication-quality figures (PDF + PNG) and formatted Excel
tables matching the naming convention in docs/publication/manuscript.md.

Output
------
    docs/publication/figures/
        Figure_1a_KM_RBD_PD.pdf
        Figure_1b_forest_cross_outcome.pdf
        Figure_2a_interaction_heatmap.pdf
        Figure_2b_RERI_forest.pdf
        Figure_3a_threshold_stability.pdf
        Figure_3b_CIF_vs_KM.pdf
    docs/publication/tables/
        Table_1.xlsx
        Table_2.xlsx
    docs/publication/extended_data/
        Extended_Data_Table_1.xlsx  ..  Extended_Data_Table_10.xlsx
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import sys
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config.config import (  # noqa: E402
    RBD_RISK_COLORS,
    outcomes,
    outcomes_formal_names,
    outcomes_short_names,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(
    r"C:\Users\riccig01\OneDrive\Projects\MtSinai\During\UkbbRbdSleepPD"
    r"\results\cox_prodromal_abk_03_27_2026_13_06_48"
)
PUB_DIR = Path(
    r"C:\Users\riccig01\OneDrive\Projects\MtSinai\During\UkbbRbdSleepPD"
    r"\docs\publication"
)
FIG_DIR = PUB_DIR / "figures"
TBL_DIR = PUB_DIR / "tables"
EDT_DIR = PUB_DIR / "extended_data"

for d in (FIG_DIR, TBL_DIR, EDT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Style ─────────────────────────────────────────────────────────────────────

# Nature-style defaults
PALETTE = {
    "rbd_low": RBD_RISK_COLORS["Low"],
    "rbd_mid": RBD_RISK_COLORS["Mid"],
    "rbd_high": RBD_RISK_COLORS["High"],
}
# Colors assigned per outcome key — must match config.outcomes names exactly.
OUTCOME_COLORS: Dict[str, str] = {
    "outcome_1a_pd_only":              "#B2182B",
    "outcome_1b_pd_ad":                "#E08214",
    "outcome_2a_vasculardementia":     "#2166AC",
    "outcome_2b_pd_vasculardementia":  "#984EA3",
    "outcome_4a_ad_only":              "#4DAF4A",
}

# Sourced from config.config (single source of truth).
OUTCOME_LABELS: Dict[str, str] = outcomes_short_names        # short names for axes/legends
OUTCOME_FORMAL_NAMES: Dict[str, str] = outcomes_formal_names  # full names for tables

# Formal covariate names (raw column codes → publication label)
COVARIATE_FORMAL_NAMES: Dict[str, str] = {
    # ── RBD risk groups ──────────────────────────────────────────────────
    "rbd_High (99,100%)":          "RBD High (99th–100th percentile)",
    "rbd_Intermediate (90,99%)":   "RBD Intermediate (90th–99th percentile)",
    "rbd_Low (0,90%)":             "RBD Low (0th–90th percentile, reference)",
    "rbd_High (90,100%)":          "RBD High (90th–100th percentile)",
    # ── Adjustment covariates ─────────────────────────────────────────────
    "cov_age_recruitment_21022":   "Age at recruitment (years)",
    "cov_sex_31":                  "Sex (male vs. female)",
    "bmi_imp_23104_i0":            "Body mass index (kg/m²)",
    "bmi_imp_23104_bl":            "Body mass index (kg/m²)",
    "bmi_21001_bl":                "Body mass index (kg/m²)",
    "cov_smoking_20116_i0":        "Smoking status",
    "cov_smoking_20116_bl":        "Smoking status",
    "cov_alcohol_20117_i0":        "Alcohol drinker status",
    "cov_alcohol_20117_bl":        "Alcohol drinker status",
    # ── Prodromal cognitive markers (legacy _i0 keys; retained) ───────────
    "cov_fluid_intelligence_20016_i0":     "Fluid intelligence score",
    "cov_react_time_mean_20023_i0":        "Reaction time (mean, ms)",
    "cov_fi_questions_attempted_20128_i0": "Fluid intelligence questions attempted",
    "cov_numeric_memory_max_20240_i0":     "Numeric memory (max digits)",
    "trail_making_errors_trail1_i2":       "Trail Making Test errors (Trial 1)",
    "cov_pairs_status_20244_i0":           "Pairs matching completion status",
    # ── Prodromal cognitive markers — current _bl names ───────────────────
    # trail_making_errors_trail1_i2 keeps _i2 (derived from p6348, not renamed).
    "cog_fluid_intelligence_bl":           "Fluid intelligence score",
    "cog_react_time_bl":                   "Reaction time (mean, ms)",
    "cov_fi_questions_attempted_20128_bl": "Fluid intelligence questions attempted",
    "cog_numeric_memory_bl":               "Numeric memory (max digits)",
    "cog_pairs_matching_bl":               "Pairs matching completion status",
    "cog_tmt_ratio_log_bl":                "Trail Making Test B/A ratio (log)",
    # ── Prodromal binary markers (legacy keys; retained) ──────────────────
    "prodromal_constipation":         "Constipation (pre-baseline HES/medication)",
    "prodromal_depression":           "Depression (pre-baseline HES/medication)",
    "prodromal_anxiety":              "Anxiety disorder (pre-baseline HES/medication)",
    "prodromal_orthostatic":          "Orthostatic hypotension (pre-baseline HES)",
    "prodromal_erectile_dysfunction": "Erectile dysfunction (pre-baseline HES/medication)",
    "prodromal_dream_enactment":      "Dream enactment behaviour (pre-baseline HES)",
    "prodromal_anosmia":              "Anosmia (pre-baseline HES)",
    "prodromal_hyposmia":             "Hyposmia (pre-baseline HES)",
    # ── Prodromal binary markers — current _bl names ──────────────────────
    "prodromal_constipation_bl":         "Constipation (pre-baseline HES/medication)",
    "prodromal_depression_bl":           "Depression (pre-baseline HES/medication)",
    "prodromal_anxiety_bl":              "Anxiety disorder (pre-baseline HES/medication)",
    "prodromal_orthostatic_bl":          "Orthostatic hypotension (pre-baseline HES)",
    "prodromal_erectile_dysfunction_bl": "Erectile dysfunction (pre-baseline HES/medication)",
    "prodromal_dream_enactment_bl":      "Dream enactment behaviour (pre-baseline HES)",
    "prodromal_anosmia_bl":              "Anosmia (pre-baseline HES)",
    "prodromal_hyposmia_bl":             "Hyposmia (pre-baseline HES)",
    # ── Prodromal group levels ────────────────────────────────────────────
    "prod_High":   "High tertile (vs. Low)",
    "prod_Medium": "Middle tertile (vs. Low)",
    "prod_Yes":    "Present (vs. Absent)",
    "prod_1":      "Present (vs. Absent)",
    # ── Continuous RBD ───────────────────────────────────────────────────
    "rbd_prob":    "RBD probability score (actigraphy-derived)",
    # ── Abbreviated codes (competing risk model) ──────────────────────────
    "cov_smoking": "Smoking status",
    "cov_alcohol": "Alcohol drinker status",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
})

# Figure widths (inches) for Nature single/double column
SINGLE_COL = 3.5
DOUBLE_COL = 7.2


# ── Helper ────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF (vector) and PNG (raster)."""
    fig.savefig(FIG_DIR / f"{name}.pdf", format="pdf")
    fig.savefig(FIG_DIR / f"{name}.png", format="png")
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


def _save_xlsx(df: pd.DataFrame, name: str, directory: Path = EDT_DIR) -> None:
    """Save DataFrame to Excel."""
    path = directory / f"{name}.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    print(f"  Saved: {path.name}")


def _format_p(p: float) -> str:
    """Format p-value for display."""
    if not np.isfinite(p):
        return "--"
    if p < 1e-30:
        return f"{p:.1e}"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _apply_formal_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace raw column codes with publication-ready formal names.

    Operates on the raw (pre-rename) DataFrame columns:
        - "outcome"          → OUTCOME_FORMAL_NAMES
        - "covariate"        → COVARIATE_FORMAL_NAMES
        - "prodromal_var"    → COVARIATE_FORMAL_NAMES
        - "prodromal_label"  → left as-is (already human-readable)

    Returns a copy; does not mutate the input.
    """
    df = df.copy()
    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].map(
            lambda x: OUTCOME_FORMAL_NAMES.get(x, x)
        )
    if "covariate" in df.columns:
        df["covariate"] = df["covariate"].map(
            lambda x: COVARIATE_FORMAL_NAMES.get(x, x)
        )
    if "prodromal_var" in df.columns:
        df["prodromal_var"] = df["prodromal_var"].map(
            lambda x: COVARIATE_FORMAL_NAMES.get(x, x)
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1a: KM survival curves by RBD risk group (PD)
# ══════════════════════════════════════════════════════════════════════════════

def figure_1a() -> None:
    """
    Kaplan-Meier survival curves for incident PD by RBD 3-group.

    Uses the pre-computed KM plot from the pipeline as basis,
    but re-generates from survival data for publication quality.
    Falls back to copying the existing PNG if raw data unavailable.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    # Load the analytical dataset
    from config.config import config
    from library.risk.risk_helpers import get_clean_risk_data, make_subject_level
    from library.column_registry import col_risk_group_agnostic, col_incident, col_surv_time

    dir_final = config["pp"]["final_dir"]
    dir_thresh = config["pp"]["thresholds"]["root"]

    _, df_risk = get_clean_risk_data(
        file_name="ehr_diag_pd_rbd_only_all",
        thresholds_root=dir_thresh,
        final_dir=dir_final,
    )
    df_subj = make_subject_level(df_risk, id_col="eid", prob_col="abk_rbd_score_mean")

    # Apply analytical cohort filter (incident + control)
    outcome = "outcome_1a_pd_only"
    incident_col = col_incident(outcome)
    surv_col = col_surv_time(outcome)
    cohort_mask = (
        df_subj[incident_col].fillna(False).astype(bool)
        | df_subj["control"].fillna(False).astype(bool)
    )
    df_subj = df_subj[cohort_mask].copy()
    df_subj = df_subj[df_subj[surv_col].notna()].copy()

    # Prepare survival columns
    group_col = col_risk_group_agnostic("percentile_3g")
    df_subj["time"] = pd.to_numeric(df_subj[surv_col], errors="coerce") / 365.25
    df_subj["event"] = df_subj[incident_col].fillna(0).astype(int)
    df_plot = df_subj.dropna(subset=["time", "event", group_col]).copy()
    df_plot[group_col] = df_plot[group_col].astype(str)

    # Order groups
    group_order = sorted(
        df_plot[group_col].unique(),
        key=lambda x: 0 if "low" in x.lower() else (1 if "mid" in x.lower() or "inter" in x.lower() else 2)
    )
    colors = [PALETTE["rbd_low"], PALETTE["rbd_mid"], PALETTE["rbd_high"]]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    kmf = KaplanMeierFitter()

    for i, grp in enumerate(group_order):
        mask = df_plot[group_col] == grp
        kmf.fit(
            df_plot.loc[mask, "time"],
            df_plot.loc[mask, "event"],
            label=grp,
        )
        kmf.plot_survival_function(
            ax=ax,
            color=colors[i],
            linewidth=1.2,
            ci_show=True,
            ci_alpha=0.15,
        )

    # Log-rank test
    lr = multivariate_logrank_test(
        df_plot["time"], df_plot[group_col], df_plot["event"]
    )
    ax.text(
        0.98, 0.02,
        f"Log-rank P = {_format_p(lr.p_value)}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6, fontstyle="italic",
    )

    ax.set_xlabel("Time since accelerometry (years)")
    ax.set_ylabel("Survival probability")
    ax.set_title("a", fontweight="bold", loc="left", fontsize=10)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.96, 1.002)
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, "Figure_1a_KM_RBD_PD")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1b: Forest plot of HR per SD across outcomes
# ══════════════════════════════════════════════════════════════════════════════

def figure_1b() -> None:
    """Forest plot: HR per 1-SD RBD score across 4 neurodegenerative outcomes."""
    df = pd.read_excel(RESULTS_DIR / "rbd_continuous.xlsx")

    # Order sourced from config.outcomes (PD first, then decreasing HR)
    outcome_order = [o for o in outcomes if o in df["outcome"].values]
    df = df.set_index("outcome").loc[outcome_order].reset_index()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    y_pos = np.arange(len(df))[::-1]

    for i, row in df.iterrows():
        color = OUTCOME_COLORS.get(row["outcome"], "#333333")
        label = OUTCOME_LABELS.get(row["outcome"], row["outcome"])

        ax.errorbar(
            row["hr_per_sd"], y_pos[i],
            xerr=[[row["hr_per_sd"] - row["hr_lci"]],
                   [row["hr_uci"] - row["hr_per_sd"]]],
            fmt="s", color=color, markersize=6, capsize=3,
            linewidth=1.0, markeredgewidth=0.5,
        )
        # Annotation: HR (CI) | C-index
        txt = (f"{row['hr_per_sd']:.2f} ({row['hr_lci']:.2f}-{row['hr_uci']:.2f})"
               f"  C={row['c_index']:.2f}")
        ax.text(
            row["hr_uci"] + 0.03, y_pos[i],
            txt, va="center", fontsize=6,
        )

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([OUTCOME_LABELS.get(o, o) for o in df["outcome"]])
    ax.set_xlabel("Hazard Ratio per 1-SD increase in RBD score")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=10)
    ax.set_xlim(0.95, 1.40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, "Figure_1b_forest_cross_outcome")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2a: Interaction heatmap
# ══════════════════════════════════════════════════════════════════════════════

def figure_2a() -> None:
    """Heatmap of multiplicative interaction HRs (Model D) for PD outcome."""
    df = pd.read_excel(RESULTS_DIR / "interaction_cox.xlsx")

    # Filter: PD outcome, percentile_3g, interaction terms only
    df = df[
        (df["outcome"] == "outcome_1a_pd_only")
        & (df["method"] == "percentile_3g")
        & (df["covariate"].str.contains("__x__", na=False))
    ].copy()

    if df.empty:
        print("  WARNING: No interaction terms found for heatmap. Skipping Figure 2a.")
        return

    # Parse interaction terms: rbd_group__x__prodromal_level
    df["rbd_group"] = df["covariate"].str.extract(r"rbd_(.+?)__x__")
    df["prod_level"] = df["covariate"].str.extract(r"__x__prod_(.+)")
    df["prodromal"] = df["prodromal_label"]

    # Build pivot: rows = prodromal markers, cols = RBD group x prodromal level
    # Simplify: just use prodromal_label and rbd_group
    pivot_data = df.pivot_table(
        index="prodromal_label",
        columns="rbd_group",
        values="HR",
        aggfunc="first",
    )

    # Build significance matrix
    pivot_p = df.pivot_table(
        index="prodromal_label",
        columns="rbd_group",
        values="p",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, 3.5))

    # Log-transform HR for symmetric color scale around 1.
    # Clip HR to [0.1, 10] before log2 to avoid extreme values (e.g. HR~0)
    # distorting the color scale.
    hr_clipped = np.clip(pivot_data.values.astype(float), 0.1, 10.0)
    log_hr = np.log2(hr_clipped)
    vmax = max(abs(np.nanmin(log_hr)), abs(np.nanmax(log_hr)))
    vmax = max(vmax, 0.5)

    im = ax.imshow(
        log_hr,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
    )

    # Annotate cells
    for i in range(log_hr.shape[0]):
        for j in range(log_hr.shape[1]):
            hr_val = pivot_data.values[i, j]
            p_val = pivot_p.values[i, j]
            if np.isfinite(hr_val):
                stars = ""
                if np.isfinite(p_val):
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"
                ax.text(
                    j, i, f"{hr_val:.2f}{stars}",
                    ha="center", va="center", fontsize=6,
                    color="white" if abs(log_hr[i, j]) > vmax * 0.6 else "black",
                )

    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(
        [f"RBD {c}" for c in pivot_data.columns],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index, fontsize=7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("log2(HR)", fontsize=7)

    ax.set_title("a", fontweight="bold", loc="left", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, "Figure_2a_interaction_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2b: RERI forest plot
# ══════════════════════════════════════════════════════════════════════════════

def figure_2b() -> None:
    """Forest plot of RERI (additive interaction) with bootstrap CIs."""
    df = pd.read_excel(RESULTS_DIR / "additive_interaction.xlsx")
    df = df[df["outcome"] == "outcome_1a_pd_only"].copy()

    if df.empty:
        print("  WARNING: No RERI data found. Skipping Figure 2b.")
        return

    # Sort by RERI ascending (most negative first)
    df = df.sort_values("reri", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

    for i, row in df.iterrows():
        # Color: red if CI excludes 0, grey Vascularwise
        sig = (row["reri_lci"] > 0) or (row["reri_uci"] < 0)
        color = "#B2182B" if sig else "#999999"

        ax.errorbar(
            row["reri"], i,
            xerr=[[row["reri"] - row["reri_lci"]],
                   [row["reri_uci"] - row["reri"]]],
            fmt="o", color=color, markersize=5, capsize=3,
            linewidth=1.0, markeredgewidth=0.5,
        )

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["prodromal_label"], fontsize=7)
    ax.set_xlabel("RERI (Relative Excess Risk due to Interaction)")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: sub-additive region
    ax.text(
        0.02, 0.98,
        "Sub-additive",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=6, fontstyle="italic", color="#B2182B",
    )
    ax.annotate(
        "", xy=(-3, len(df) - 0.5), xytext=(0, len(df) - 0.5),
        arrowprops=dict(arrowstyle="<-", color="#B2182B", lw=0.5),
    )

    fig.tight_layout()
    _save_fig(fig, "Figure_2b_RERI_forest")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3a: Threshold stability
# ══════════════════════════════════════════════════════════════════════════════

def figure_3a() -> None:
    """Bar/forest plot: RBD HR at 5th, 10th, 15th percentile cutoffs (PD)."""
    df = pd.read_excel(RESULTS_DIR / "rbd_threshold_stability.xlsx")
    df = df[df["outcome"] == "outcome_1a_pd_only"].copy()

    if df.empty:
        print("  WARNING: No threshold stability data. Skipping Figure 3a.")
        return

    df = df.sort_values("percentile").reset_index(drop=True)
    y_pos = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))

    colors_grad = [RBD_RISK_COLORS["High"], RBD_RISK_COLORS["Mid"], RBD_RISK_COLORS["Low"]]
    for i, row in df.iterrows():
        c = colors_grad[i] if i < len(colors_grad) else "#999999"
        ax.errorbar(
            row["hr"], i,
            xerr=[[row["hr"] - row["lci"]], [row["uci"] - row["hr"]]],
            fmt="s", color=c, markersize=6, capsize=3,
            linewidth=1.0, markeredgewidth=0.5,
        )
        ax.text(
            row["uci"] + 0.05, i,
            f"HR {row['hr']:.2f} ({row['lci']:.2f}-{row['uci']:.2f})",
            va="center", fontsize=6,
        )

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"Top {100 - row['percentile']:.0f}% (P{row['percentile']:.0f})"
         for _, row in df.iterrows()],
        fontsize=7,
    )
    ax.set_xlabel("Hazard Ratio (High vs Low RBD)")
    ax.set_title("a", fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, "Figure_3a_threshold_stability")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3b: CIF vs 1-KM comparison
# ══════════════════════════════════════════════════════════════════════════════

def figure_3b() -> None:
    """Compare Aalen-Johansen CIF vs 1-KM estimates by RBD group."""
    df = pd.read_excel(RESULTS_DIR / "competing_risk_cif_vs_km.xlsx")
    df = df[df["outcome"] == "outcome_1a_pd_only"].copy()

    if df.empty:
        print("  WARNING: No CIF vs KM data. Skipping Figure 3b.")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

    groups = df["group"].unique()
    x_vals = sorted(df["timepoint"].unique())
    width = 0.15
    x_arr = np.arange(len(x_vals))

    def _cif_color(label: str) -> str:
        ll = label.lower()
        if "high" in ll:
            return PALETTE["rbd_high"]
        if "low" in ll:
            return PALETTE["rbd_low"]
        return "#999999"

    for gi, grp in enumerate(groups):
        dg = df[df["group"] == grp].sort_values("timepoint")
        color = _cif_color(grp)
        offset = (gi - 0.5) * width * 2.5

        cif_vals = dg["CIF_AJ_pct"].values
        km_vals = dg["CIF_KM_pct"].values

        bars_cif = ax.bar(
            x_arr + offset - width / 2, cif_vals, width,
            color=color, alpha=0.8, edgecolor="black", linewidth=0.3,
            label=f"{grp} (AJ CIF)" if gi == 0 else f"{grp} (AJ CIF)",
        )
        bars_km = ax.bar(
            x_arr + offset + width / 2, km_vals, width,
            color=color, alpha=0.3, edgecolor="black", linewidth=0.3,
            hatch="//",
            label=f"{grp} (1-KM)" if gi == 0 else f"{grp} (1-KM)",
        )

        # Annotate values
        for j, (c, k) in enumerate(zip(cif_vals, km_vals)):
            ax.text(x_arr[j] + offset, max(c, k) + 0.05,
                    f"{c:.2f}%", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"{int(t)}-year" for t in x_vals])
    ax.set_ylabel("Cumulative incidence (%)")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=5, ncol=2, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, "Figure_3b_CIF_vs_KM")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABLES (Table 1 & Table 2)
# ══════════════════════════════════════════════════════════════════════════════

def table_1() -> None:
    """Table 1: RBD stratification + prodromal marker associations."""
    # Model A: RBD-only
    df_rbd = pd.read_excel(RESULTS_DIR / "rbd_only_cox.xlsx")
    df_rbd = df_rbd[
        (df_rbd["outcome"] == "outcome_1a_pd_only")
        & (df_rbd["method"] == "percentile_3g")
    ].copy()
    # Keep only risk group covariates
    df_rbd = df_rbd[df_rbd["covariate"].str.contains("risk_group", na=False)].copy()

    # RBD continuous
    df_cont = pd.read_excel(RESULTS_DIR / "rbd_continuous.xlsx")
    df_cont = df_cont[df_cont["outcome"] == "outcome_1a_pd_only"].copy()

    # Model B: Prodromal baseline
    df_prod = pd.read_excel(RESULTS_DIR / "baseline_cox_HRs.xlsx")
    df_prod = df_prod[df_prod["outcome"] == "outcome_1a_pd_only"].copy()
    # Keep only the prodromal variable rows (not covariates)
    df_prod = df_prod[df_prod["covariate"] == df_prod["prodromal_var"]].copy()

    rows = []
    # Section A
    rows.append({"Variable": "Model A: RBD risk group (3-tier)", "HR": "", "95% CI": "", "P": "", "P_FDR": ""})
    rows.append({"Variable": "  Low (0-90th percentile)", "HR": "1.00", "95% CI": "ref", "P": "--", "P_FDR": "--"})
    for _, r in df_rbd.iterrows():
        label = r["covariate"]
        if "mid" in label.lower() or "inter" in label.lower():
            display = "  Intermediate (90th-99th)"
        elif "high" in label.lower():
            display = "  High (99th-100th)"
        else:
            continue
        rows.append({
            "Variable": display,
            "HR": f"{r['HR']:.2f}",
            "95% CI": f"{r['HR_lower']:.2f}-{r['HR_upper']:.2f}",
            "P": _format_p(r["p"]),
            "P_FDR": "--",
        })
    # Continuous
    if not df_cont.empty:
        r = df_cont.iloc[0]
        rows.append({
            "Variable": "  RBD continuous (per 1-SD)",
            "HR": f"{r['hr_per_sd']:.2f}",
            "95% CI": f"{r['hr_lci']:.2f}-{r['hr_uci']:.2f}",
            "P": _format_p(r["p"]),
            "P_FDR": "--",
        })

    # Section B
    rows.append({"Variable": "Model B: Prodromal markers (individual)", "HR": "", "95% CI": "", "P": "", "P_FDR": ""})
    df_prod = df_prod.sort_values("p")
    for _, r in df_prod.iterrows():
        rows.append({
            "Variable": f"  {r['prodromal_label']}",
            "HR": f"{r['HR']:.2f}",
            "95% CI": f"{r['HR_lower']:.2f}-{r['HR_upper']:.2f}",
            "P": _format_p(r["p"]),
            "P_FDR": _format_p(r["p_fdr"]) if pd.notna(r.get("p_fdr")) else "--",
        })

    df_out = pd.DataFrame(rows)
    _save_xlsx(df_out, "Table_1", directory=TBL_DIR)


def table_2() -> None:
    """Table 2: Additive interaction measures (RERI)."""
    df = pd.read_excel(RESULTS_DIR / "additive_interaction.xlsx")
    df = df[df["outcome"] == "outcome_1a_pd_only"].copy()

    out_rows = []
    for _, r in df.iterrows():
        out_rows.append({
            "Prodromal marker": r["prodromal_label"],
            "HR_11": f"{r['hr_11']:.2f}",
            "HR_10": f"{r['hr_10']:.2f}",
            "HR_01": f"{r['hr_01']:.2f}",
            "RERI": f"{r['reri']:.2f}",
            "RERI 95% CI": f"{r['reri_lci']:.2f} to {r['reri_uci']:.2f}",
            "AP": f"{r['ap']:.2f}",
            "N": r["N"],
            "Events": r["events"],
        })
    df_out = pd.DataFrame(out_rows)
    _save_xlsx(df_out, "Table_2", directory=TBL_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED DATA TABLES 1-10
# ══════════════════════════════════════════════════════════════════════════════

def extended_data_table_1() -> None:
    """EDT1: Threshold stability across all outcomes."""
    df = pd.read_excel(RESULTS_DIR / "rbd_threshold_stability.xlsx")
    df = _apply_formal_names(df)
    df = df.rename(columns={
        "percentile": "Percentile cutoff",
        "threshold_value": "Threshold value",
        "hr": "HR", "lci": "95% CI lower", "uci": "95% CI upper",
        "p": "P", "n_high": "N (High)", "n_low": "N (Low)",
        "events": "Events", "outcome": "Outcome",
    })
    _save_xlsx(df, "Extended_Data_Table_1")


def extended_data_table_2() -> None:
    """EDT2: C-index comparison."""
    df = pd.read_excel(RESULTS_DIR / "c_index.xlsx")
    df = _apply_formal_names(df)
    df = df.rename(columns={
        "outcome": "Outcome",
        "prodromal_var": "Prodromal variable (UKBB field code)",
        "prodromal_label": "Prodromal marker",
        "c_index_full": "C-index (full model)",
        "c_index_null": "C-index (base model)",
        "c_index_incremental": "Delta C-index",
        "N": "N", "events": "Events",
    })
    _save_xlsx(df, "Extended_Data_Table_2")


def extended_data_table_3() -> None:
    """EDT3: Lag sensitivity analysis."""
    df = pd.read_excel(RESULTS_DIR / "lag_sensitivity.xlsx")
    df = _apply_formal_names(df)
    df = df.rename(columns={
        "outcome": "Outcome",
        "prodromal_var": "Prodromal variable (UKBB field code)",
        "prodromal_label": "Prodromal marker",
        "covariate": "Covariate",
        "HR_primary": "HR (primary, no lag)",
        "HR_lag2y": "HR (2-year lag)",
        "HR_lag2y_lower": "95% CI lower (lag)",
        "HR_lag2y_upper": "95% CI upper (lag)",
        "p_lag2y": "P (2-year lag)",
        "N_lag": "N (lag cohort)", "events_lag": "Events (lag cohort)",
    })
    _save_xlsx(df, "Extended_Data_Table_3")


def extended_data_table_4() -> None:
    """EDT4: HES-active subcohort sensitivity."""
    df = pd.read_excel(RESULTS_DIR / "sensitivity_hes_active.xlsx")
    # Keep only the prodromal variable rows (not all covariates)
    df_prod = df[df["covariate"] == df["prodromal_var"]].copy()
    if df_prod.empty:
        df_prod = df.copy()
    df_prod = _apply_formal_names(df_prod)
    df_out = df_prod.rename(columns={
        "outcome": "Outcome",
        "prodromal_label": "Prodromal marker",
        "analysis": "Analysis",
        "N_sensitivity": "N (HES-active subcohort)",
        "N_excluded_gap": "N excluded (HES gap > 4 yr)",
        "events_sensitivity": "Events (HES-active subcohort)",
        "HR": "HR", "HR_lower": "95% CI lower", "HR_upper": "95% CI upper",
        "p": "P",
    })
    cols_keep = [c for c in df_out.columns if c not in ("prodromal_var", "covariate", "N", "events")]
    _save_xlsx(df_out[cols_keep], "Extended_Data_Table_4")


def extended_data_table_5() -> None:
    """EDT5: Proportional hazards diagnostic tests."""
    df = pd.read_excel(RESULTS_DIR / "ph_diagnostics.xlsx")
    df = _apply_formal_names(df)
    df = df.rename(columns={
        "outcome": "Outcome",
        "model": "Model",
        "prodromal_var": "Prodromal variable (UKBB field code)",
        "covariate": "Covariate",
        "ph_stat": "Schoenfeld statistic",
        "ph_p": "P (Schoenfeld test)",
        "ph_violation": "PH violation (P < 0.05)",
        "prodromal_label": "Prodromal marker",
    })
    _save_xlsx(df, "Extended_Data_Table_5")


def extended_data_table_6() -> None:
    """EDT6: Additive model (Model C) full results – Parkinson's disease, 3-group stratification."""
    df = pd.read_excel(RESULTS_DIR / "additive_cox.xlsx")
    df_pd = df[
        (df["outcome"] == "outcome_1a_pd_only")
        & (df["method"] == "percentile_3g")
    ].copy()
    # Keep RBD risk group rows and prodromal variable main-effect rows only
    mask = (
        df_pd["covariate"].str.startswith("rbd_", na=False)
        | (df_pd["covariate"] == df_pd["prodromal_var"])
    )
    df_pd = _apply_formal_names(df_pd[mask])
    df_out = df_pd.rename(columns={
        "prodromal_label": "Prodromal marker added",
        "covariate": "Covariate",
        "HR": "HR", "HR_lower": "95% CI lower", "HR_upper": "95% CI upper",
        "p": "P", "N": "N", "events": "Events",
    })
    cols = ["Prodromal marker added", "Covariate", "HR", "95% CI lower", "95% CI upper", "P", "N", "Events"]
    _save_xlsx(df_out[[c for c in cols if c in df_out.columns]], "Extended_Data_Table_6")


def extended_data_table_7() -> None:
    """EDT7: Multiplicative interaction (Model D) full results – PD, 3-group."""
    df = pd.read_excel(RESULTS_DIR / "interaction_cox.xlsx")
    df_pd = df[
        (df["outcome"] == "outcome_1a_pd_only")
        & (df["method"] == "percentile_3g")
    ].copy()
    # Keep interaction terms + RBD main effects + prodromal main effects
    mask = (
        df_pd["covariate"].str.contains("__x__", na=False)
        | df_pd["covariate"].str.startswith("rbd_", na=False)
        | (df_pd["covariate"] == df_pd["prodromal_var"])
    )
    # Apply formal names before rename (outcome and prodromal_var columns)
    df_named = _apply_formal_names(df_pd[mask])
    # Clean up interaction term labels: replace comma notation
    df_named["covariate"] = (
        df_named["covariate"]
        .str.replace(r"\((\d+),(\d+)%\)", r"(\1–\2th pct)", regex=True)
    )
    df_out = df_named.rename(columns={
        "prodromal_label": "Prodromal marker",
        "covariate": "Model term",
        "HR": "HR", "HR_lower": "95% CI lower", "HR_upper": "95% CI upper",
        "p": "P", "N": "N", "events": "Events",
    })
    cols = ["Prodromal marker", "Model term", "HR", "95% CI lower", "95% CI upper", "P", "N", "Events"]
    _save_xlsx(df_out[[c for c in cols if c in df_out.columns]], "Extended_Data_Table_7")


def extended_data_table_8() -> None:
    """EDT8: Spline analysis LR tests (linearity assumption)."""
    df_prod = pd.read_excel(RESULTS_DIR / "spline_cox.xlsx")
    df_rbd = pd.read_excel(RESULTS_DIR / "rbd_spline.xlsx")

    df_rbd["prodromal_var"] = "rbd_prob"
    df_rbd["prodromal_label"] = "RBD probability score (actigraphy-derived)"

    cols_common = ["outcome", "prodromal_var", "prodromal_label",
                   "c_index_spline", "c_index_linear", "lr_stat", "lr_p", "N", "events"]
    parts = []
    for d in [df_rbd, df_prod]:
        for c in cols_common:
            if c not in d.columns:
                d[c] = np.nan
        parts.append(d[cols_common])

    df_out = pd.concat(parts, ignore_index=True)
    df_out = _apply_formal_names(df_out)
    df_out = df_out.rename(columns={
        "outcome": "Outcome",
        "prodromal_label": "Variable",
        "c_index_spline": "C-index (spline model)",
        "c_index_linear": "C-index (linear model)",
        "lr_stat": "LR statistic (spline vs. linear)",
        "lr_p": "LR P-value (non-linearity)",
        "N": "N", "events": "Events",
    })
    cols_out = ["Outcome", "Variable", "C-index (spline model)", "C-index (linear model)",
                "LR statistic (spline vs. linear)", "LR P-value (non-linearity)", "N", "Events"]
    _save_xlsx(df_out[[c for c in cols_out if c in df_out.columns]], "Extended_Data_Table_8")


def extended_data_table_9() -> None:
    """EDT9: Data availability and missingness."""
    df = pd.read_excel(RESULTS_DIR / "data_availability_report.xlsx")
    # Apply formal names to "variable" column if it contains raw field codes
    if "variable" in df.columns:
        df["variable"] = df["variable"].map(
            lambda x: COVARIATE_FORMAL_NAMES.get(x, x)
        )
    df = df.rename(columns={
        "variable": "Variable",
        "label": "Label",
        "n_available": "N available",
        "pct_available": "% available",
        "in_dataset": "In dataset",
        "pct_hes_active": "% HES-active subcohort",
    })
    _save_xlsx(df, "Extended_Data_Table_9")


def extended_data_table_10() -> None:
    """EDT10: Cross-outcome competing risk Cox HRs."""
    df = pd.read_excel(RESULTS_DIR / "competing_risk_cox.xlsx")
    df = _apply_formal_names(df)
    df = df.rename(columns={
        "outcome": "Outcome",
        "covariate": "Covariate",
        "HR": "HR", "HR_lower": "95% CI lower", "HR_upper": "95% CI upper",
        "p": "P", "c_index": "C-index", "N": "N", "events": "Events",
    })
    _save_xlsx(df, "Extended_Data_Table_10")


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE FIGURES (multi-panel)
# ══════════════════════════════════════════════════════════════════════════════

def figure_1_composite() -> None:
    """
    Figure 1 composite: (a) KM curves + (b) cross-outcome forest plot.

    This is the combined two-panel figure for the manuscript.
    Individual panels are also saved separately.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    from config.config import config
    from library.risk.risk_helpers import get_clean_risk_data, make_subject_level
    from library.column_registry import col_risk_group_agnostic, col_incident, col_surv_time

    dir_final = config["pp"]["final_dir"]
    dir_thresh = config["pp"]["thresholds"]["root"]

    _, df_risk = get_clean_risk_data(
        file_name="ehr_diag_pd_rbd_only_all",
        thresholds_root=dir_thresh,
        final_dir=dir_final,
    )
    df_subj = make_subject_level(df_risk, id_col="eid", prob_col="abk_rbd_score_mean")

    outcome = "outcome_1a_pd_only"
    incident_col = col_incident(outcome)
    surv_col = col_surv_time(outcome)
    cohort_mask = (
        df_subj[incident_col].fillna(False).astype(bool)
        | df_subj["control"].fillna(False).astype(bool)
    )
    df_subj = df_subj[cohort_mask & df_subj[surv_col].notna()].copy()

    group_col = col_risk_group_agnostic("percentile_3g")
    df_subj["time"] = pd.to_numeric(df_subj[surv_col], errors="coerce") / 365.25
    df_subj["event"] = df_subj[incident_col].fillna(0).astype(int)
    df_plot = df_subj.dropna(subset=["time", "event", group_col]).copy()
    df_plot[group_col] = df_plot[group_col].astype(str)

    group_order = sorted(
        df_plot[group_col].unique(),
        key=lambda x: 0 if "low" in x.lower() else (1 if "mid" in x.lower() or "inter" in x.lower() else 2)
    )
    colors = [PALETTE["rbd_low"], PALETTE["rbd_mid"], PALETTE["rbd_high"]]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})

    # ── Panel a: KM ──────────────────────────────────────────────────────
    ax = axes[0]
    kmf = KaplanMeierFitter()
    for i, grp in enumerate(group_order):
        mask = df_plot[group_col] == grp
        kmf.fit(df_plot.loc[mask, "time"], df_plot.loc[mask, "event"], label=grp)
        kmf.plot_survival_function(ax=ax, color=colors[i], linewidth=1.2, ci_show=True, ci_alpha=0.12)

    lr = multivariate_logrank_test(df_plot["time"], df_plot[group_col], df_plot["event"])
    ax.text(0.98, 0.02, f"Log-rank P = {_format_p(lr.p_value)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, fontstyle="italic")
    ax.set_xlabel("Time since accelerometry (years)")
    ax.set_ylabel("Survival probability")
    ax.set_title("a", fontweight="bold", loc="left", fontsize=10)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.96, 1.002)
    ax.legend(loc="lower left", frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel b: forest ──────────────────────────────────────────────────
    ax = axes[1]
    df_cont = pd.read_excel(RESULTS_DIR / "rbd_continuous.xlsx")
    outcome_order = [o for o in outcomes if o in df_cont["outcome"].values]
    df_cont = df_cont.set_index("outcome").loc[outcome_order].reset_index()
    y_pos = np.arange(len(df_cont))[::-1]

    for i, row in df_cont.iterrows():
        color = OUTCOME_COLORS.get(row["outcome"], "#333333")
        ax.errorbar(
            row["hr_per_sd"], y_pos[i],
            xerr=[[row["hr_per_sd"] - row["hr_lci"]], [row["hr_uci"] - row["hr_per_sd"]]],
            fmt="s", color=color, markersize=5, capsize=3, linewidth=1.0, markeredgewidth=0.5,
        )
        txt = f"{row['hr_per_sd']:.2f} ({row['hr_lci']:.2f}-{row['hr_uci']:.2f})  C={row['c_index']:.2f}"
        ax.text(row["hr_uci"] + 0.02, y_pos[i], txt, va="center", fontsize=5.5)

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([OUTCOME_LABELS.get(o, o) for o in df_cont["outcome"]], fontsize=7)
    ax.set_xlabel("HR per 1-SD RBD score")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=10)
    ax.set_xlim(0.95, 1.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, "Figure_1")


def figure_2_composite() -> None:
    """Figure 2 composite: (a) interaction heatmap + (b) RERI forest."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Panel a: Heatmap ─────────────────────────────────────────────────
    ax = axes[0]
    df = pd.read_excel(RESULTS_DIR / "interaction_cox.xlsx")
    df = df[
        (df["outcome"] == "outcome_1a_pd_only")
        & (df["method"] == "percentile_3g")
        & (df["covariate"].str.contains("__x__", na=False))
    ].copy()

    if not df.empty:
        df["rbd_group"] = df["covariate"].str.extract(r"rbd_(.+?)__x__")
        pivot_data = df.pivot_table(index="prodromal_label", columns="rbd_group", values="HR", aggfunc="first")
        pivot_p = df.pivot_table(index="prodromal_label", columns="rbd_group", values="p", aggfunc="first")

        hr_clipped = np.clip(pivot_data.values.astype(float), 0.1, 10.0)
        log_hr = np.log2(hr_clipped)
        vmax = max(abs(np.nanmin(log_hr)), abs(np.nanmax(log_hr)), 0.5)

        im = ax.imshow(log_hr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(log_hr.shape[0]):
            for j in range(log_hr.shape[1]):
                hr_val = pivot_data.values[i, j]
                p_val = pivot_p.values[i, j]
                if np.isfinite(hr_val):
                    stars = ""
                    if np.isfinite(p_val):
                        if p_val < 0.001: stars = "***"
                        elif p_val < 0.01: stars = "**"
                        elif p_val < 0.05: stars = "*"
                    ax.text(j, i, f"{hr_val:.2f}{stars}", ha="center", va="center", fontsize=5,
                            color="white" if abs(log_hr[i, j]) > vmax * 0.6 else "black")

        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([f"RBD {c}" for c in pivot_data.columns], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index, fontsize=6)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("log2(HR)", fontsize=6)

    ax.set_title("a", fontweight="bold", loc="left", fontsize=10)

    # ── Panel b: RERI forest ─────────────────────────────────────────────
    ax = axes[1]
    df_reri = pd.read_excel(RESULTS_DIR / "additive_interaction.xlsx")
    df_reri = df_reri[df_reri["outcome"] == "outcome_1a_pd_only"].sort_values("reri").reset_index(drop=True)

    for i, row in df_reri.iterrows():
        sig = (row["reri_lci"] > 0) or (row["reri_uci"] < 0)
        color = "#B2182B" if sig else "#999999"
        ax.errorbar(row["reri"], i,
                    xerr=[[row["reri"] - row["reri_lci"]], [row["reri_uci"] - row["reri"]]],
                    fmt="o", color=color, markersize=4, capsize=3, linewidth=0.8, markeredgewidth=0.5)

    ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_yticks(range(len(df_reri)))
    ax.set_yticklabels(df_reri["prodromal_label"], fontsize=6)
    ax.set_xlabel("RERI")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, "Figure_2")


def figure_3_composite() -> None:
    """Figure 3 composite: (a) threshold stability + (b) CIF vs KM."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    # ── Panel a: threshold stability ─────────────────────────────────────
    ax = axes[0]
    df = pd.read_excel(RESULTS_DIR / "rbd_threshold_stability.xlsx")
    df = df[df["outcome"] == "outcome_1a_pd_only"].sort_values("percentile").reset_index(drop=True)

    colors_grad = [RBD_RISK_COLORS["High"], RBD_RISK_COLORS["Mid"], RBD_RISK_COLORS["Low"]]
    for i, row in df.iterrows():
        c = colors_grad[i] if i < len(colors_grad) else "#999999"
        ax.errorbar(row["hr"], i,
                    xerr=[[row["hr"] - row["lci"]], [row["uci"] - row["hr"]]],
                    fmt="s", color=c, markersize=5, capsize=3, linewidth=1.0, markeredgewidth=0.5)
        ax.text(row["uci"] + 0.05, i,
                f"HR {row['hr']:.2f} ({row['lci']:.2f}-{row['uci']:.2f})",
                va="center", fontsize=5.5)

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"Top {100 - row['percentile']:.0f}% (P{row['percentile']:.0f})" for _, row in df.iterrows()],
        fontsize=6,
    )
    ax.set_xlabel("Hazard Ratio (High vs Low)")
    ax.set_title("a", fontweight="bold", loc="left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel b: CIF vs KM ──────────────────────────────────────────────
    ax = axes[1]
    df_cif = pd.read_excel(RESULTS_DIR / "competing_risk_cif_vs_km.xlsx")
    df_cif = df_cif[df_cif["outcome"] == "outcome_1a_pd_only"].copy()

    groups = df_cif["group"].unique()
    x_vals = sorted(df_cif["timepoint"].unique())
    width = 0.15
    x_arr = np.arange(len(x_vals))
    def _cif_color_c(label: str) -> str:
        ll = label.lower()
        if "high" in ll:
            return PALETTE["rbd_high"]
        if "low" in ll:
            return PALETTE["rbd_low"]
        return "#999999"

    for gi, grp in enumerate(groups):
        dg = df_cif[df_cif["group"] == grp].sort_values("timepoint")
        color = _cif_color_c(grp)
        offset = (gi - 0.5) * width * 2.5

        ax.bar(x_arr + offset - width / 2, dg["CIF_AJ_pct"].values, width,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.bar(x_arr + offset + width / 2, dg["CIF_KM_pct"].values, width,
               color=color, alpha=0.3, edgecolor="black", linewidth=0.3, hatch="//")

        for j, (c, k) in enumerate(zip(dg["CIF_AJ_pct"].values, dg["CIF_KM_pct"].values)):
            ax.text(x_arr[j] + offset, max(c, k) + 0.03,
                    f"{c:.2f}%", ha="center", va="bottom", fontsize=4.5)

    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"{int(t)}-year" for t in x_vals])
    ax.set_ylabel("Cumulative incidence (%)")
    ax.set_title("b", fontweight="bold", loc="left", fontsize=10)

    # Legend
    handles = [
        mpatches.Patch(color=PALETTE["rbd_high"], alpha=0.8, label="High RBD (AJ CIF)"),
        mpatches.Patch(color=PALETTE["rbd_high"], alpha=0.3, label="High RBD (1-KM)", hatch="//"),
        mpatches.Patch(color=PALETTE["rbd_low"], alpha=0.8, label="Low RBD (AJ CIF)"),
        mpatches.Patch(color=PALETTE["rbd_low"], alpha=0.3, label="Low RBD (1-KM)", hatch="//"),
    ]
    ax.legend(handles=handles, fontsize=4.5, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, "Figure_3")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Generate all manuscript figures and tables."""
    print("=" * 70)
    print("  GENERATING MANUSCRIPT FIGURES AND TABLES")
    print("=" * 70)

    # ── Individual panels ──────────────────────────────────────────────────
    print("\n[1/5] Generating individual figure panels ...")
    figure_1a()
    figure_1b()
    figure_2a()
    figure_2b()
    figure_3a()
    figure_3b()

    # ── Composite figures ──────────────────────────────────────────────────
    print("\n[2/5] Generating composite figures ...")
    figure_1_composite()
    figure_2_composite()
    figure_3_composite()

    # ── Main tables ────────────────────────────────────────────────────────
    print("\n[3/5] Generating main tables ...")
    table_1()
    table_2()

    # ── Extended data tables ───────────────────────────────────────────────
    print("\n[4/5] Generating extended data tables ...")
    extended_data_table_1()
    extended_data_table_2()
    extended_data_table_3()
    extended_data_table_4()
    extended_data_table_5()
    extended_data_table_6()
    extended_data_table_7()
    extended_data_table_8()
    extended_data_table_9()
    extended_data_table_10()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[5/5] Summary")
    print(f"  Figures:  {FIG_DIR}")
    print(f"  Tables:   {TBL_DIR}")
    print(f"  Extended: {EDT_DIR}")

    n_figs = len(list(FIG_DIR.glob("*.pdf")))
    n_tbls = len(list(TBL_DIR.glob("*.xlsx")))
    n_edts = len(list(EDT_DIR.glob("*.xlsx")))
    print(f"\n  Total: {n_figs} figures (PDF+PNG), {n_tbls} main tables, {n_edts} extended data tables")
    print("\nDone.")


if __name__ == "__main__":
    main()
