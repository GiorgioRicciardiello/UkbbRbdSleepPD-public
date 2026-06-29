"""
Publication-quality figures for the RBD–PRS biological strength analysis.

Figures
───────
F1  Joint scatter (RBD score vs. PRS_{PD,RBD}), coloured by risk group,
    with marginal KDE and Spearman ρ annotation.
F2  Spearman ρ forest plot stratified by rg_pctl3 with 95 % CIs.
F3  OLS partial regression plots (added-variable plots), one per PRS.
F4  GAM smooth curves for PRS → RBD score with pointwise 95 % CI,
    faceted by PRS type and stratum (full / high-risk).
F5  Violin + strip plot of RBD score and PRS by risk group × case/control.

Design principles
─────────────────
- Pastel palette defined in config.PALETTE.
- seaborn 'whitegrid' theme, axes spines trimmed.
- All text ≥ 9pt; titles ≤ 12pt; axis labels 10pt.
- DPI = 300 for print-ready output.
- No chart junk: no unnecessary grid lines on scatter plots.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from pygam import LinearGAM, s, l

from library.rbd_prs_association.analysis import AnalysisResults, GAMResult, SpearmanResult
from library.rbd_prs_association.config import (
    ADJUSTMENT_COVARIATES,
    FIGURE_DPI,
    FIGURE_STYLE,
    GAM_LAMBDA_GRID,
    GAM_MAX_ITER,
    GAM_N_SPLINES,
    HIGH_RISK_LABEL,
    PALETTE,
    PRS_PD_COL,
    PRS_RBD_COL,
    RBD_SCORE_COL,
    RG_ORDER,
    RG_SHORT,
    RISK_GROUP_COL,
)

logger = logging.getLogger(__name__)

_PRS_LABELS = {
    PRS_PD_COL: "PD Polygenic Risk Score (z-score)",
    PRS_RBD_COL: "RBD Polygenic Risk Score (z-score)",
}
_RBD_LABEL = "RBD Probability Score"

sns.set_theme(style=FIGURE_STYLE, font_scale=0.95)

_RG_COLORS = {rg: PALETTE[rg] for rg in RG_ORDER}
_RG_COLORS_SHORT = {RG_SHORT[k]: v for k, v in _RG_COLORS.items()}


def _save(fig: plt.Figure, name: str, figures_dir: Path) -> Path:
    path = figures_dir / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def _annotate_spearman(ax: plt.Axes, rho: float, p: float, n: int) -> None:
    """Add Spearman ρ annotation box to an axes."""
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    text = f"ρ = {rho:+.3f} {sig}\nN = {n:,}"
    ax.text(
        0.97, 0.04, text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#BBBBBB", alpha=0.85),
    )


# ── Figure 1: Joint scatter with marginal KDE ────────────────────────────────

def plot_joint_scatter(
    df: pd.DataFrame,
    results: AnalysisResults,
    figures_dir: Path,
) -> List[Path]:
    """F1: Joint scatter RBD score vs. PRS_{PD,RBD}, coloured by rg_pctl3.

    Produces one figure per PRS column (2 total).
    """
    paths = []
    # Map risk group to short label for legend
    df = df.copy()
    df["rg_short"] = df[RISK_GROUP_COL].map(RG_SHORT).fillna("Unknown")
    rg_short_order = [RG_SHORT[rg] for rg in RG_ORDER if RG_SHORT[rg] in df["rg_short"].unique()]

    for prs_col in [PRS_PD_COL, PRS_RBD_COL]:
        if prs_col not in df.columns:
            logger.warning("Column %s missing; skipping F1 for this PRS.", prs_col)
            continue

        valid = df[[RBD_SCORE_COL, prs_col, "rg_short", "case_control"]].dropna()

        # Build colour list for scatter
        point_colors = [_RG_COLORS_SHORT.get(rg, "#CCCCCC") for rg in valid["rg_short"]]

        fig = plt.figure(figsize=(8, 7))
        # 3-panel layout: main scatter + top marginal + right marginal
        gs = fig.add_gridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # Scatter (subsample if large to avoid overplotting)
        n_plot = min(len(valid), 8000)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(valid), n_plot, replace=False) if len(valid) > n_plot else np.arange(len(valid))
        sub = valid.iloc[idx]
        sub_colors = [point_colors[i] for i in idx]

        ax_main.scatter(
            sub[prs_col], sub[RBD_SCORE_COL],
            c=sub_colors, alpha=0.35, s=10, linewidths=0, rasterized=True,
        )
        # Add OLS regression line over full valid set
        x_line = np.linspace(valid[prs_col].min(), valid[prs_col].max(), 200)
        coefs = np.polyfit(valid[prs_col], valid[RBD_SCORE_COL], 1)
        ax_main.plot(x_line, np.polyval(coefs, x_line),
                     color=PALETTE["regression_line"], lw=1.5, ls="--", zorder=5, label="OLS trend")

        ax_main.set_xlabel(_PRS_LABELS[prs_col], fontsize=10)
        ax_main.set_ylabel(_RBD_LABEL, fontsize=10)
        sns.despine(ax=ax_main)

        # Annotate Spearman ρ
        full_corr = next((r for r in results.spearman if r.prs_col == prs_col and r.stratum == "Full cohort"), None)
        if full_corr:
            _annotate_spearman(ax_main, full_corr.rho, full_corr.p_permutation, full_corr.n)

        # Marginal KDEs by risk group
        for rg_short, color in _RG_COLORS_SHORT.items():
            sub_rg = valid.loc[valid["rg_short"] == rg_short]
            if len(sub_rg) < 10:
                continue
            try:
                sns.kdeplot(x=sub_rg[prs_col], ax=ax_top, color=color, fill=True, alpha=0.4, linewidth=1.2)
                sns.kdeplot(y=sub_rg[RBD_SCORE_COL], ax=ax_right, color=color, fill=True, alpha=0.4, linewidth=1.2)
            except Exception:
                pass

        ax_top.set_ylabel("Density", fontsize=8)
        ax_right.set_xlabel("Density", fontsize=8)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        sns.despine(ax=ax_top, left=False)
        sns.despine(ax=ax_right, bottom=False)

        # Legend
        patches = [
            mpatches.Patch(facecolor=_RG_COLORS_SHORT[rg], label=rg, alpha=0.8)
            for rg in rg_short_order if rg in _RG_COLORS_SHORT
        ]
        ax_main.legend(handles=patches, title="RBD Risk Group", fontsize=8,
                       title_fontsize=8.5, loc="upper left", framealpha=0.9)

        prs_tag = "pd" if prs_col == PRS_PD_COL else "rbd"
        fig.suptitle(
            f"RBD Probability Score vs. {_PRS_LABELS[prs_col].split('(')[0].strip()}\n"
            f"coloured by RBD Risk Group (rg_pctl3)",
            fontsize=11, y=1.01,
        )
        paths.append(_save(fig, f"fig1_joint_scatter_{prs_tag}.png", figures_dir))

    return paths


# ── Figure 2: Spearman ρ forest plot ─────────────────────────────────────────

def plot_correlation_forest(results: AnalysisResults, figures_dir: Path) -> Path:
    """F2: Forest plot of Spearman ρ stratified by risk group with 95 % CI."""
    rows = [r for r in results.spearman if r.stratum in list(RG_SHORT.values()) + ["Full cohort"]]
    if not rows:
        logger.warning("No correlation results for forest plot; skipping F2.")
        return None

    df_plot = pd.DataFrame([
        {
            "prs": r.prs_col,
            "stratum": r.stratum,
            "rho": r.rho,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "p": r.p_permutation,
            "n": r.n,
        }
        for r in rows
    ])

    prs_cols_present = df_plot["prs"].unique()
    fig, axes = plt.subplots(1, len(prs_cols_present), figsize=(5 * len(prs_cols_present), 5), sharey=True)
    if len(prs_cols_present) == 1:
        axes = [axes]

    strata_order = ["Full cohort"] + [RG_SHORT[rg] for rg in RG_ORDER if RG_SHORT[rg] in df_plot["stratum"].values]
    palette_order = {
        "Full cohort": "#888888",
        **_RG_COLORS_SHORT,
    }

    for ax, prs_col in zip(axes, prs_cols_present):
        sub = df_plot.loc[df_plot["prs"] == prs_col].copy()
        sub["y"] = sub["stratum"].map({s: i for i, s in enumerate(strata_order)})
        sub = sub.sort_values("y")

        for _, row in sub.iterrows():
            color = palette_order.get(row["stratum"], "#999999")
            ax.errorbar(
                x=row["rho"],
                y=row["y"],
                xerr=[[row["rho"] - row["ci_lower"]], [row["ci_upper"] - row["rho"]]],
                fmt="o", color=color, markersize=7, capsize=4, lw=1.8,
            )
            # Significance annotation
            sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            if sig:
                ax.text(row["ci_upper"] + 0.005, row["y"], sig, va="center", fontsize=9, color=color)

        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Spearman ρ  (95% CI)", fontsize=10)
        ax.set_title(_PRS_LABELS[prs_col].replace(" (z-score)", ""), fontsize=10)
        ax.set_yticks(range(len(strata_order)))
        ax.set_yticklabels(strata_order, fontsize=9)
        sns.despine(ax=ax)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("Spearman ρ: RBD Probability Score × PRS\nby Risk Group Stratum", fontsize=11)
    plt.tight_layout()
    return _save(fig, "fig2_correlation_forest.png", figures_dir)


# ── Figure 3: OLS partial regression (added-variable) plots ─────────────────

def plot_partial_regression(
    df: pd.DataFrame,
    active_covariates: List[str],
    figures_dir: Path,
) -> List[Path]:
    """F3: Added-variable (partial regression) plots for each PRS predictor.

    An added-variable plot removes the linear effect of all other covariates
    from both the outcome and the predictor, showing the unique contribution
    of the PRS after adjustment.
    """
    import statsmodels.api as sm

    paths = []
    cov_cols = [c for c in active_covariates if c in df.columns]

    for prs_col in [PRS_PD_COL, PRS_RBD_COL]:
        if prs_col not in df.columns:
            continue
        needed = [RBD_SCORE_COL, prs_col] + cov_cols
        valid = df[needed].dropna()
        if len(valid) < 50:
            continue

        y = valid[RBD_SCORE_COL].to_numpy(dtype=float)
        x_prs = valid[prs_col].to_numpy(dtype=float)
        X_cov = sm.add_constant(valid[cov_cols].to_numpy(dtype=float))

        # Residualise outcome on covariates
        res_y = sm.OLS(y, X_cov).fit().resid
        # Residualise PRS on covariates
        res_x = sm.OLS(x_prs, X_cov).fit().resid

        # OLS on residuals (slope = partial β)
        slope, intercept = np.polyfit(res_x, res_y, 1)
        x_line = np.linspace(res_x.min(), res_x.max(), 200)

        # Subsample for plotting
        n_plot = min(len(valid), 6000)
        idx = np.random.default_rng(42).choice(len(valid), n_plot, replace=False)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            res_x[idx], res_y[idx],
            color=PALETTE["rbd_score"], alpha=0.3, s=8, linewidths=0, rasterized=True,
        )
        ax.plot(x_line, slope * x_line + intercept,
                color=PALETTE["regression_line"], lw=2, label=f"Partial β = {slope:+.4f}")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")
        ax.axvline(0, color="#AAAAAA", lw=0.8, ls=":")
        ax.set_xlabel(f"{_PRS_LABELS[prs_col]} | covariates", fontsize=10)
        ax.set_ylabel(f"{_RBD_LABEL} | covariates", fontsize=10)
        ax.set_title(
            f"Partial Regression: {_PRS_LABELS[prs_col].split('(')[0].strip()}\n"
            f"(adjusted for age, sex, BMI, PC1–PC10, lifestyle)",
            fontsize=10,
        )
        ax.legend(fontsize=9, framealpha=0.9)
        sns.despine(ax=ax)
        prs_tag = "pd" if prs_col == PRS_PD_COL else "rbd"
        paths.append(_save(fig, f"fig3_partial_regression_{prs_tag}.png", figures_dir))

    return paths


# ── Figure 4: GAM smooth curves ───────────────────────────────────────────────

def plot_gam_smooth(
    df: pd.DataFrame,
    active_covariates: List[str],
    gam_results: List[GAMResult],
    figures_dir: Path,
) -> List[Path]:
    """F4: GAM spline curves PRS → RBD score with 95 % CI, faceted by stratum.

    The smooth is evaluated at the mean of all covariates to show the
    marginal relationship between PRS and RBD score.
    """
    import statsmodels.api as sm

    paths = []
    strata_to_plot = [
        ("Full cohort", df),
        (f"High-risk ({HIGH_RISK_LABEL})", df.loc[df[RISK_GROUP_COL] == HIGH_RISK_LABEL]),
    ]
    cov_cols = [c for c in active_covariates if c in df.columns]

    for prs_col in [PRS_PD_COL, PRS_RBD_COL]:
        if prs_col not in df.columns:
            continue

        n_strata = len(strata_to_plot)
        fig, axes = plt.subplots(1, n_strata, figsize=(6 * n_strata, 5), sharey=True)
        if n_strata == 1:
            axes = [axes]

        for ax, (stratum_label, sub) in zip(axes, strata_to_plot):
            needed = [RBD_SCORE_COL, prs_col] + cov_cols
            valid = sub[needed].dropna()
            if len(valid) < 50:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
                continue

            y = valid[RBD_SCORE_COL].to_numpy(dtype=float)
            X = np.column_stack([
                valid[prs_col].to_numpy(dtype=float)
            ] + [valid[c].to_numpy(dtype=float) for c in cov_cols])

            n_covs = len(cov_cols)
            terms = s(0, n_splines=GAM_N_SPLINES, spline_order=3)
            for i in range(1, 1 + n_covs):
                terms = terms + l(i)

            # Retrieve best lambda from stored results if available
            gam_res = next(
                (r for r in gam_results if r.prs_col == prs_col and r.stratum == stratum_label),
                None,
            )
            best_lam = gam_res.best_lambda if gam_res else 0.6
            lam_list = [best_lam] + [0.6] * n_covs

            gam = LinearGAM(terms, max_iter=GAM_MAX_ITER)
            gam.lam = [np.array([l]) for l in lam_list]
            gam.fit(X, y)

            # Evaluate smooth at 200 points of prs_col, other covariates at mean
            x_grid = np.linspace(valid[prs_col].min(), valid[prs_col].max(), 200)
            cov_means = valid[cov_cols].mean().to_numpy(dtype=float)
            X_grid = np.column_stack([x_grid] + [np.full(200, m) for m in cov_means])

            y_pred = gam.predict(X_grid)
            try:
                ci = gam.prediction_intervals(X_grid, width=0.95)
                y_lo = ci[:, 0]
                y_hi = ci[:, 1]
            except Exception:
                y_lo = y_pred
                y_hi = y_pred

            # OLS line for comparison
            coefs_ols = np.polyfit(valid[prs_col], valid[RBD_SCORE_COL], 1)
            y_ols = np.polyval(coefs_ols, x_grid)

            color = PALETTE["prs_pd"] if prs_col == PRS_PD_COL else PALETTE["prs_rbd"]
            ax.fill_between(x_grid, y_lo, y_hi, color=color, alpha=0.3, label="95% CI (GAM)")
            ax.plot(x_grid, y_pred, color=color, lw=2, label="GAM smooth")
            ax.plot(x_grid, y_ols, color=PALETTE["regression_line"], lw=1.5, ls="--", label="OLS linear")
            ax.set_xlabel(_PRS_LABELS[prs_col], fontsize=10)
            ax.set_ylabel(_RBD_LABEL if ax == axes[0] else "", fontsize=10)
            ax.set_title(stratum_label, fontsize=10)
            sns.despine(ax=ax)

            # Annotate edf
            if gam_res:
                ax.text(
                    0.03, 0.97,
                    f"edf = {gam_res.edf:.2f}\np_nonlin = {gam_res.nonlinearity_p:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#BBBBBB", alpha=0.85),
                )

        axes[0].legend(fontsize=8, framealpha=0.9)
        fig.suptitle(
            f"GAM Smooth: {_PRS_LABELS[prs_col].split('(')[0].strip()} → RBD Score\n"
            f"(adjusted for age, sex, BMI, PC1–PC10; evaluated at covariate means)",
            fontsize=11,
        )
        plt.tight_layout()
        prs_tag = "pd" if prs_col == PRS_PD_COL else "rbd"
        paths.append(_save(fig, f"fig4_gam_smooth_{prs_tag}.png", figures_dir))

    return paths


# ── Figure 5: Violin + strip by risk group × case/control ────────────────────

def plot_violin_outcome(df: pd.DataFrame, figures_dir: Path) -> Path:
    """F5: Violin + jittered strip plot for RBD score and PRS by rg_pctl3 × outcome.

    Three panels: abk_rbd_score_mean, prs_score_pd, prs_score_rbd.
    X-axis: risk group; hue: case/control.
    """
    if "case_control" not in df.columns:
        logger.warning("case_control column missing; skipping F5.")
        return None

    df = df.copy()
    df["rg_short"] = df[RISK_GROUP_COL].map(RG_SHORT).fillna("Unknown")
    rg_short_order = [RG_SHORT[rg] for rg in RG_ORDER if RG_SHORT[rg] in df["rg_short"].unique()]

    plot_vars = [
        (RBD_SCORE_COL, _RBD_LABEL),
        (PRS_PD_COL, "PD PRS (z-score)"),
        (PRS_RBD_COL, "RBD PRS (z-score)"),
    ]
    plot_vars = [(col, label) for col, label in plot_vars if col in df.columns]

    fig, axes = plt.subplots(1, len(plot_vars), figsize=(5.5 * len(plot_vars), 5.5))
    if len(plot_vars) == 1:
        axes = [axes]

    hue_palette = {"case": PALETTE["case"], "control": PALETTE["control"]}

    for ax, (col, ylabel) in zip(axes, plot_vars):
        valid = df[["rg_short", "case_control", col]].dropna()
        valid = valid.loc[valid["rg_short"].isin(rg_short_order)]
        valid[col] = pd.to_numeric(valid[col], errors="coerce")

        sns.violinplot(
            data=valid, x="rg_short", y=col, hue="case_control",
            order=rg_short_order, palette=hue_palette,
            inner=None, split=True, linewidth=0.8,
            density_norm="width", ax=ax,
        )
        # Subsample strip for legibility
        n_strip = min(len(valid), 2000)
        idx = np.random.default_rng(42).choice(len(valid), n_strip, replace=False)
        sub_strip = valid.iloc[idx]
        sns.stripplot(
            data=sub_strip, x="rg_short", y=col, hue="case_control",
            order=rg_short_order, palette=hue_palette,
            dodge=True, size=2, alpha=0.4, jitter=True, legend=False, ax=ax,
        )
        # Add median lines
        for i, rg in enumerate(rg_short_order):
            for j, cc in enumerate(["control", "case"]):
                vals = valid.loc[(valid["rg_short"] == rg) & (valid["case_control"] == cc), col].dropna()
                if len(vals) == 0:
                    continue
                med = vals.median()
                offset = -0.2 + 0.4 * j
                ax.plot([i + offset - 0.1, i + offset + 0.1], [med, med],
                        color="black", lw=1.5, zorder=10)

        ax.set_xlabel("RBD Risk Group (rg_pctl3)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10)
        sns.despine(ax=ax)

        # Update legend only on last panel
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(),
                  title="Incident PD", fontsize=8, title_fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Distribution of RBD Score and PRS by Risk Group × Incident PD",
        fontsize=11,
    )
    plt.tight_layout()
    return _save(fig, "fig5_violin_outcome.png", figures_dir)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def generate_all_figures(
    df: pd.DataFrame,
    results: AnalysisResults,
    active_covariates: List[str],
    figures_dir: Path,
) -> None:
    """Generate and save all five figures.

    Parameters
    ----------
    figures_dir : Path
        Directory where PNG figures are written.  Must already exist.
    """
    logger.info("Generating Figure 1: joint scatter ...")
    plot_joint_scatter(df, results, figures_dir)

    logger.info("Generating Figure 2: correlation forest plot ...")
    plot_correlation_forest(results, figures_dir)

    logger.info("Generating Figure 3: partial regression plots ...")
    plot_partial_regression(df, active_covariates, figures_dir)

    logger.info("Generating Figure 4: GAM smooth curves ...")
    plot_gam_smooth(df, active_covariates, results.gam, figures_dir)

    logger.info("Generating Figure 5: violin outcome plot ...")
    plot_violin_outcome(df, figures_dir)

    logger.info("All figures saved to: %s", figures_dir)
