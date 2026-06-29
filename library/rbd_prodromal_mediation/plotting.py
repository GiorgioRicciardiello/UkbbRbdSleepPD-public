"""
Publication-grade plotting for RBD-Prodromal mediation analysis.

Figures produced
----------------
1. plot_mediation_steps_forest  — 4-panel forest: a-path β, b-path HR,
                                   c-path HR, c'-path HR per variable.
2. plot_indirect_effect_forest  — Bootstrap HR_indirect + PM% with CIs.
3. plot_3group_bpath_forest     — 3-group b-path (Intermediate/High vs Low).
4. plot_model_performance_delta — ΔC-index lollipop (both interpretations).
5. plot_association_overview    — OLS β and logistic OR dot-plot overview.
6. run_mediation_plots          — Orchestrator: load from disk, save all figures.

Style
-----
- Pastel palette; DejaVu Sans; 300 dpi; whitegrid.
- Every title contains N and event count.
- Significance coded: *** p<0.001, ** p<0.01, * p<0.05, ns.
- FDR-adjusted p-values used for star annotations where available.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ── Palette ────────────────────────────────────────────────────────────────

PASTEL = SimpleNamespace(
    # Interpretation A (binary prodromal) — warm
    interp_a="#E8916A",
    interp_a_light="#F5D0BF",
    # Interpretation C (cognitive) — cool
    interp_c="#7BAEC8",
    interp_c_light="#C4DBE8",
    # Mediation paths
    c_path="#D4A373",         # total effect — warm sand
    a_path="#95C4B0",         # a-path — teal-green
    b_path="#9BB3CE",         # b-path — slate blue
    cprime_path="#C9A8D4",    # direct effect — lavender
    indirect="#F2B880",       # indirect — peach
    inconsistent="#CC8EC4",   # inconsistent mediation — mauve
    # Significance / neutral
    sig_pos="#2D6A4F",        # significant positive — dark green
    sig_neg="#7B2D8B",        # significant negative — purple
    nonsig="#BBBBBB",         # non-significant — light grey
    ref_line="#999999",       # reference line at HR=1
    grid="#EEEEEE",           # gridline
    text_dark="#222222",
    text_muted="#666666",
)


# ── Style helpers ──────────────────────────────────────────────────────────

def _setup_style() -> None:
    """Apply project-wide publication rcParams."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.color": PASTEL.grid,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
    })


def _save(fig: Figure, path: Path, dpi: int = 300) -> None:
    """Save and close figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [plot] {path.name}")


def _pstar(p: float, fdr: Optional[float] = None) -> str:
    """Return significance stars (uses FDR if provided and < raw p)."""
    pval = fdr if (fdr is not None and not np.isnan(fdr)) else p
    if np.isnan(pval):
        return ""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"


def _fmt_p(p: float) -> str:
    """Format p-value for display."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def _clean_label(label: str, max_len: int = 26) -> str:
    """Truncate long variable labels for axis ticks."""
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def _var_color(row: pd.Series, interp: str = "A") -> str:
    """Color a forest dot by significance (FDR of c-path)."""
    p = row.get("p_c_fdr", row.get("p_c", np.nan))
    if pd.isna(p):
        return PASTEL.nonsig
    if p < 0.05:
        if row.get("inconsistent_mediation", False):
            return PASTEL.inconsistent
        hr = row.get("hr_c", 1.0)
        return PASTEL.sig_pos if hr >= 1.0 else PASTEL.sig_neg
    return PASTEL.nonsig


# ── Figure 1: Mediation steps forest ──────────────────────────────────────

def plot_mediation_steps_forest(
    df_steps: pd.DataFrame,
    out_path: Path,
    interp_label: str = "A",
    outcome_label: str = "PD (outcome_1a)",
) -> None:
    """
    4-panel forest plot of Baron-Kenny mediation steps.

    Panels (left to right):
    - Panel A: a-path OLS β (prodromal → RBD z-score) ± 95% CI
    - Panel B: b-path HR (RBD → PD | prodromal) ± 95% CI (log scale)
    - Panel C: c-path HR total effect (prodromal → PD) ± 95% CI (log scale)
    - Panel D: c'-path HR direct effect (prodromal → PD | RBD) ± 95% CI (log scale)

    Parameters
    ----------
    df_steps : pd.DataFrame
        mediation_steps.xlsx output.
    out_path : Path
        Destination PNG file.
    interp_label : str
        "A" or "C" — used in title.
    outcome_label : str
        Human-readable outcome string for title.
    """
    _setup_style()
    if df_steps is None or df_steps.empty:
        return

    df = df_steps.copy().reset_index(drop=True)
    n_vars = len(df)
    y_pos = np.arange(n_vars)
    labels = [_clean_label(str(r["label"])) for _, r in df.iterrows()]

    # Colour per row (significance of c-path FDR)
    dot_colors = [_var_color(row, interp_label) for _, row in df.iterrows()]

    # Compute OLS 95% CI from se_a (±1.96·se)
    beta_a_lo = df["beta_a"] - 1.96 * df["se_a"]
    beta_a_hi = df["beta_a"] + 1.96 * df["se_a"]

    total_n = int(df["n"].iloc[0]) if "n" in df.columns else 0
    total_ev = int(df["events"].iloc[0]) if "events" in df.columns else 0

    interp_color = PASTEL.interp_a if interp_label == "A" else PASTEL.interp_c

    fig, axes = plt.subplots(
        1, 4,
        figsize=(15, max(3.5, 0.65 * n_vars + 1.2)),
        gridspec_kw={"width_ratios": [1, 1, 1, 1]},
    )
    fig.suptitle(
        f"Mediation Steps — Interpretation {interp_label} | {outcome_label}\n"
        f"N={total_n:,}  events={total_ev}",
        fontsize=11, fontweight="bold", y=1.01,
    )

    panel_specs = [
        # (ax, x_vals, x_lo, x_hi, ref, xlabel, log_scale, title)
        (
            axes[0],
            df["beta_a"].values,
            beta_a_lo.values, beta_a_hi.values,
            0.0, "β (OLS)", False,
            "a-path\nProdromal → RBD",
        ),
        (
            axes[1],
            df["hr_b"].values,
            df["hr_b_lower"].values, df["hr_b_upper"].values,
            1.0, "HR", True,
            "b-path\nRBD → PD | Prodromal",
        ),
        (
            axes[2],
            df["hr_c"].values,
            df["hr_c_lower"].values, df["hr_c_upper"].values,
            1.0, "HR", True,
            "c-path (total)\nProdromal → PD",
        ),
        (
            axes[3],
            df["hr_cprime"].values,
            df["hr_cprime_lower"].values, df["hr_cprime_upper"].values,
            1.0, "HR", True,
            "c'-path (direct)\nProdromal → PD | RBD",
        ),
    ]

    for ax, x_vals, x_lo, x_hi, ref_val, xlabel, log_scale, panel_title in panel_specs:
        x_lo = np.where(np.isnan(x_lo), x_vals, x_lo)
        x_hi = np.where(np.isnan(x_hi), x_vals, x_hi)

        xerr_lo = np.clip(x_vals - x_lo, 0, None)
        xerr_hi = np.clip(x_hi - x_vals, 0, None)

        if log_scale:
            safe_x = np.where(x_vals > 0, x_vals, np.nan)
            ax.set_xscale("log")
        else:
            safe_x = x_vals

        for i, (xi, lo, hi, col) in enumerate(
            zip(safe_x, xerr_lo, xerr_hi, dot_colors)
        ):
            ax.errorbar(
                xi, y_pos[i],
                xerr=[[lo], [hi]],
                fmt="o", color=col, ecolor=col,
                elinewidth=1.2, capsize=3,
                markersize=6, markeredgewidth=0.5,
                markeredgecolor="white",
                alpha=0.85,
            )

        ax.axvline(ref_val, color=PASTEL.ref_line, linestyle="--",
                   linewidth=0.9, alpha=0.7, zorder=0)

        # Significance stars on rightmost edge
        p_col = "p_a_fdr" if "a-path" in panel_title else "p_c_fdr"
        p_raw_col = "p_a" if "a-path" in panel_title else "p_c"
        if "c'-path" in panel_title:
            p_col, p_raw_col = "p_cprime", "p_cprime"
        if "b-path" in panel_title:
            p_col, p_raw_col = "p_b", "p_b"

        for i, (_, row) in enumerate(df.iterrows()):
            p_fdr = row.get(p_col, np.nan)
            p_raw = row.get(p_raw_col, np.nan)
            star = _pstar(p_raw, p_fdr)
            ax.text(
                ax.get_xlim()[1] if log_scale else (
                    x_hi.max() + abs(x_vals).max() * 0.05
                ),
                y_pos[i], f" {star}",
                va="center", ha="left", fontsize=7,
                color=PASTEL.text_dark if star != "ns" else PASTEL.text_muted,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            labels if ax is axes[0] else [""] * n_vars,
            fontsize=8,
        )
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_title(panel_title, fontsize=9, fontweight="bold", pad=6)
        ax.set_ylim(-0.6, n_vars - 0.4)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")

    # Legend
    legend_handles = [
        mpatches.Patch(color=PASTEL.sig_pos, label="Sig. positive (FDR p<0.05)"),
        mpatches.Patch(color=PASTEL.sig_neg, label="Sig. negative (FDR p<0.05)"),
        mpatches.Patch(color=PASTEL.inconsistent, label="Inconsistent mediation"),
        mpatches.Patch(color=PASTEL.nonsig, label="Not significant"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4,
        fontsize=7.5, framealpha=0.8,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    _save(fig, out_path)


# ── Figure 2: Indirect effect forest ──────────────────────────────────────

def plot_indirect_effect_forest(
    df_steps: pd.DataFrame,
    df_boot: pd.DataFrame,
    out_path: Path,
    interp_label: str = "A",
    outcome_label: str = "PD (outcome_1a)",
    pm_cap: float = 150.0,
) -> None:
    """
    Two-panel forest: bootstrap HR_indirect (top) and PM% (bottom).

    Bootstrap CIs are used when available; falls back to point estimate.
    PM% values outside ±pm_cap are displayed as triangular markers at cap
    with a label showing the true value — preventing axis distortion from
    extreme PM% under inconsistent mediation.

    Parameters
    ----------
    df_steps : pd.DataFrame
        mediation_steps.xlsx — provides point estimates and metadata.
    df_boot : pd.DataFrame
        mediation_indirect.xlsx — provides bootstrap CIs.
    out_path : Path
        Destination PNG.
    interp_label : str
        "A" or "C".
    outcome_label : str
        Human-readable outcome string.
    pm_cap : float
        Axis cap for PM% display (default ±150%).
    """
    _setup_style()
    if df_steps is None or df_steps.empty:
        return

    # Merge steps + bootstrap.
    # Strategy: rename ALL overlapping columns in df_boot before merge so that
    # no suffix collisions occur. CI columns (lci/uci) only exist in df_boot,
    # so they merge in cleanly. Point estimates in df_boot override df_steps.
    CI_COLS = ["hr_indirect_lci", "hr_indirect_uci",
               "pm_pct_lci", "pm_pct_uci", "n_converged", "n_bootstrap"]

    merged = df_steps.copy()

    if df_boot is not None and not df_boot.empty:
        # Columns that exist in both DataFrames and need renaming in boot
        overlap = {"variable", "label", "hr_indirect", "pm_pct"}
        boot = df_boot.copy()
        boot = boot.drop(
            columns=[c for c in boot.columns if c in overlap - {"variable"}],
            errors="ignore",
        )
        boot = boot.rename(columns={
            "hr_indirect": "hr_indirect_boot",
            "pm_pct": "pm_pct_boot",
        })
        merged = merged.merge(boot, on="variable", how="left")

        # Prefer bootstrap values over step-level point estimates
        if "hr_indirect_boot" in merged.columns:
            merged["hr_indirect"] = merged["hr_indirect_boot"].fillna(
                merged["hr_indirect"]
            )
        if "pm_pct_boot" in merged.columns:
            merged["pm_pct"] = merged["pm_pct_boot"].fillna(merged["pm_pct"])

    # Ensure CI columns exist (fill NaN if boot wasn't available)
    for col in CI_COLS:
        if col not in merged.columns:
            merged[col] = np.nan

    n_vars = len(merged)
    y_pos = np.arange(n_vars)
    labels = [_clean_label(str(r["label"])) for _, r in merged.iterrows()]

    total_n = int(merged["n"].iloc[0]) if "n" in merged.columns else 0
    total_ev = int(merged["events"].iloc[0]) if "events" in merged.columns else 0

    fig, (ax_hr, ax_pm) = plt.subplots(
        2, 1,
        figsize=(8, max(4.5, 0.9 * n_vars + 2.5)),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.suptitle(
        f"Indirect Effect & Proportion Mediated — Interpretation {interp_label}\n"
        f"{outcome_label}  |  N={total_n:,}  events={total_ev}  "
        f"(Bootstrap B={int(merged['n_bootstrap'].dropna().iloc[0]) if 'n_bootstrap' in merged.columns and merged['n_bootstrap'].notna().any() else 'N/A'})",
        fontsize=10, fontweight="bold",
    )

    # ── Top panel: HR_indirect ───────────────────────────────────────────
    ax_hr.set_xscale("log")
    ax_hr.axvline(1.0, color=PASTEL.ref_line, linestyle="--",
                  linewidth=0.9, alpha=0.7, zorder=0)

    for i, (_, row) in enumerate(merged.iterrows()):
        inconsist = bool(row.get("inconsistent_mediation", False))
        color = PASTEL.inconsistent if inconsist else (
            PASTEL.interp_a if interp_label == "A" else PASTEL.interp_c
        )
        hr = row.get("hr_indirect", np.nan)
        lci = row.get("hr_indirect_lci", hr)
        uci = row.get("hr_indirect_uci", hr)
        if pd.isna(lci):
            lci = hr
        if pd.isna(uci):
            uci = hr

        if pd.notna(hr) and hr > 0:
            ax_hr.errorbar(
                hr, y_pos[i],
                xerr=[[max(0, hr - lci)], [max(0, uci - hr)]],
                fmt="D" if inconsist else "o",
                color=color, ecolor=color,
                elinewidth=1.2, capsize=3,
                markersize=6.5, markeredgewidth=0.5,
                markeredgecolor="white", alpha=0.85,
                label="Inconsistent" if inconsist else None,
            )

        # Annotation: HR [LCI–UCI] anchored just right of the CI upper bound
        lci_str = f"{lci:.3f}" if pd.notna(lci) else "?"
        uci_str = f"{uci:.3f}" if pd.notna(uci) else "?"
        hr_str = f"{hr:.3f}" if pd.notna(hr) else "?"
        x_ann = (uci * 1.05) if (pd.notna(uci) and uci > 0) else 1.05
        ax_hr.text(
            x_ann, y_pos[i],
            f"  HR={hr_str} [{lci_str}-{uci_str}]",
            va="center", ha="left", fontsize=6.5,
            color=PASTEL.text_muted,
        )

    ax_hr.set_yticks(y_pos)
    ax_hr.set_yticklabels(labels, fontsize=8)
    ax_hr.set_xlabel("HR_indirect (bootstrap 95% CI)", fontsize=8.5)
    ax_hr.set_title("Bootstrap Indirect Effect (HR_indirect via RBD)", fontsize=9,
                    fontweight="bold")
    ax_hr.set_ylim(-0.6, n_vars - 0.4)
    ax_hr.invert_yaxis()
    ax_hr.grid(True, axis="x", alpha=0.3)
    ax_hr.grid(False, axis="y")

    # ── Bottom panel: PM% ────────────────────────────────────────────────
    ax_pm.axvline(0.0, color=PASTEL.ref_line, linestyle="--",
                  linewidth=0.9, alpha=0.7, zorder=0)
    ax_pm.axvline(100.0, color=PASTEL.ref_line, linestyle=":",
                  linewidth=0.7, alpha=0.5, zorder=0)

    for i, (_, row) in enumerate(merged.iterrows()):
        inconsist = bool(row.get("inconsistent_mediation", False))
        color = PASTEL.inconsistent if inconsist else (
            PASTEL.interp_a if interp_label == "A" else PASTEL.interp_c
        )
        pm = row.get("pm_pct", np.nan)
        lci = row.get("pm_pct_lci", pm)
        uci = row.get("pm_pct_uci", pm)

        # Cap extreme values
        pm_disp = np.clip(pm, -pm_cap, pm_cap) if pd.notna(pm) else np.nan
        lci_disp = np.clip(lci, -pm_cap, pm_cap) if pd.notna(lci) else np.nan
        uci_disp = np.clip(uci, -pm_cap, pm_cap) if pd.notna(uci) else np.nan

        if pd.notna(pm_disp):
            ax_pm.errorbar(
                pm_disp, y_pos[i],
                xerr=[[max(0, pm_disp - lci_disp)],
                      [max(0, uci_disp - pm_disp)]],
                fmt="^" if (abs(pm) > pm_cap if pd.notna(pm) else False) else "o",
                color=color, ecolor=color,
                elinewidth=1.2, capsize=3,
                markersize=6.5, markeredgewidth=0.5,
                markeredgecolor="white", alpha=0.85,
            )
            # Label actual value if capped
            if pd.notna(pm) and abs(pm) > pm_cap:
                ax_pm.text(
                    pm_disp, y_pos[i] - 0.25,
                    f"  {pm:.0f}%", va="top", ha="left",
                    fontsize=6.5, color=color,
                    fontstyle="italic",
                )

        # Value annotation
        pm_str = f"{pm:.1f}%" if pd.notna(pm) else "N/A"
        lci_s = f"{lci:.1f}" if pd.notna(lci) else "?"
        uci_s = f"{uci:.1f}" if pd.notna(uci) else "?"
        ax_pm.text(
            pm_cap + 5, y_pos[i],
            f"{pm_str} [{lci_s}–{uci_s}]",
            va="center", ha="left", fontsize=6.5, color=PASTEL.text_muted,
        )

    ax_pm.set_yticks(y_pos)
    ax_pm.set_yticklabels(labels, fontsize=8)
    ax_pm.set_xlabel("Proportion Mediated % (PM%)", fontsize=8.5)
    ax_pm.set_title(
        f"Proportion Mediated via RBD (PM%)\n"
        f"Capped at ±{pm_cap:.0f}% for display; triangle = capped value",
        fontsize=9, fontweight="bold",
    )
    ax_pm.set_xlim(-pm_cap * 1.1, pm_cap * 1.65)
    ax_pm.set_ylim(-0.6, n_vars - 0.4)
    ax_pm.invert_yaxis()
    ax_pm.grid(True, axis="x", alpha=0.3)
    ax_pm.grid(False, axis="y")

    # Inconsistent mediation legend patch
    legend_handles = [
        mpatches.Patch(
            color=PASTEL.interp_a if interp_label == "A" else PASTEL.interp_c,
            label="Consistent mediation",
        ),
        mpatches.Patch(color=PASTEL.inconsistent, label="Inconsistent mediation"),
    ]
    ax_hr.legend(
        handles=legend_handles, loc="lower right",
        fontsize=7.5, framealpha=0.8,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_path)


# ── Figure 3: 3-group b-path forest ────────────────────────────────────────

def plot_3group_bpath_forest(
    df_3g: pd.DataFrame,
    out_path: Path,
    interp_label: str = "A",
    outcome_label: str = "PD (outcome_1a)",
) -> None:
    """
    Grouped forest plot for the 3-group b-path Cox model.

    Shows HR_Intermediate and HR_High vs Low for each variable,
    demonstrating a dose-response of RBD groups on PD risk after
    adjusting for each prodromal marker.

    Parameters
    ----------
    df_3g : pd.DataFrame
        supplementary_3g_bpath.xlsx output.
    out_path : Path
        Destination PNG.
    interp_label : str
        "A" or "C".
    outcome_label : str
        Outcome label for title.
    """
    _setup_style()
    if df_3g is None or df_3g.empty:
        return

    df = df_3g.copy().reset_index(drop=True)
    n_vars = len(df)
    labels = [_clean_label(str(r["label"])) for _, r in df.iterrows()]

    total_n = int(df["n"].iloc[0]) if "n" in df.columns else 0
    total_ev = int(df["events"].iloc[0]) if "events" in df.columns else 0

    # Staggered y-positions: variable i → two rows at i*2+0 and i*2+1
    y_intermed = np.arange(n_vars) * 2.2 + 0.4
    y_high = np.arange(n_vars) * 2.2

    fig, ax = plt.subplots(
        figsize=(8, max(4.0, 0.8 * n_vars * 2 + 1.5)),
    )
    fig.suptitle(
        f"3-Group RBD b-Path (Cox) — Interpretation {interp_label}\n"
        f"RBD: Low / Intermediate (≥p90) / High (≥p99) | {outcome_label}\n"
        f"N={total_n:,}  events={total_ev}",
        fontsize=10, fontweight="bold",
    )

    color_intermed = "#9BB3CE"   # slate blue
    color_high = "#E8916A"       # salmon-orange

    for i, (_, row) in enumerate(df.iterrows()):
        # Intermediate vs Low
        hr_i = row.get("hr_intermediate_vs_low", np.nan)
        lo_i = row.get("hr_intermediate_lci", hr_i)
        hi_i = row.get("hr_intermediate_uci", hr_i)
        p_i = row.get("p_intermediate", np.nan)
        if pd.notna(hr_i) and hr_i > 0:
            ax.errorbar(
                hr_i, y_intermed[i],
                xerr=[[max(0, hr_i - (lo_i or hr_i))],
                      [max(0, (hi_i or hr_i) - hr_i)]],
                fmt="s", color=color_intermed, ecolor=color_intermed,
                elinewidth=1.2, capsize=3,
                markersize=6, markeredgewidth=0.5,
                markeredgecolor="white", alpha=0.85,
            )
            star_i = _pstar(p_i)
            ax.text(
                hi_i + 0.05 if pd.notna(hi_i) else hr_i + 0.1,
                y_intermed[i],
                f"  {hr_i:.2f} {star_i}",
                va="center", ha="left", fontsize=7,
                color=color_intermed,
            )

        # High vs Low
        hr_h = row.get("hr_high_vs_low", np.nan)
        lo_h = row.get("hr_high_lci", hr_h)
        hi_h = row.get("hr_high_uci", hr_h)
        p_h = row.get("p_high", np.nan)
        if pd.notna(hr_h) and hr_h > 0:
            ax.errorbar(
                hr_h, y_high[i],
                xerr=[[max(0, hr_h - (lo_h or hr_h))],
                      [max(0, (hi_h or hr_h) - hr_h)]],
                fmt="o", color=color_high, ecolor=color_high,
                elinewidth=1.2, capsize=3,
                markersize=6, markeredgewidth=0.5,
                markeredgecolor="white", alpha=0.85,
            )
            star_h = _pstar(p_h)
            ax.text(
                hi_h + 0.05 if pd.notna(hi_h) else hr_h + 0.1,
                y_high[i],
                f"  {hr_h:.2f} {star_h}",
                va="center", ha="left", fontsize=7,
                color=color_high,
            )

    ax.axvline(1.0, color=PASTEL.ref_line, linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=0)
    ax.set_xscale("log")

    # Y-axis labels centred between the two dots per variable
    y_mid = (y_intermed + y_high) / 2
    ax.set_yticks(y_mid)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Hazard Ratio vs RBD Low (log scale)", fontsize=8.5)
    ax.set_title(
        "b-path: RBD 3-Group → PD | Prodromal variable adjusted",
        fontsize=9, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(False, axis="y")

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=color_intermed,
            markersize=7, label="Intermediate vs Low (≥p90)",
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color_high,
            markersize=7, label="High vs Low (≥p99)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8, framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_path)


# ── Figure 4: ΔC-index lollipop ────────────────────────────────────────────

def plot_model_performance_delta(
    df_perf_summary: pd.DataFrame,
    out_path: Path,
    outcome_label: str = "PD (outcome_1a)",
) -> None:
    """
    Lollipop chart of ΔC-index (joint − c-path) for all variables.

    Shows the discrimination gain from adding the RBD mediator to the
    prodromal Cox model. Variables are colour-coded by interpretation
    (A = binary prodromal, C = cognitive).

    Parameters
    ----------
    df_perf_summary : pd.DataFrame
        model_performance_summary.xlsx (combined A + C).
    out_path : Path
        Destination PNG.
    outcome_label : str
        Outcome label for title.
    """
    _setup_style()
    if df_perf_summary is None or df_perf_summary.empty:
        return

    df = df_perf_summary.copy().sort_values("delta_c", ascending=True)
    df = df.reset_index(drop=True)
    n_vars = len(df)
    y_pos = np.arange(n_vars)

    # Determine which interpretation each variable belongs to
    is_interp_a = df["variable"].str.startswith("prodromal_")
    colors = [PASTEL.interp_a if a else PASTEL.interp_c for a in is_interp_a]
    labels = [_clean_label(str(r["label"])) for _, r in df.iterrows()]

    fig, (ax_delta, ax_cindex) = plt.subplots(
        1, 2, figsize=(12, max(3.5, 0.6 * n_vars + 1.5)),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    fig.suptitle(
        f"Model Discrimination — Joint vs Prodromal-Only Cox | {outcome_label}",
        fontsize=11, fontweight="bold",
    )

    # ── Left: ΔC lollipop ───────────────────────────────────────────────
    ax_delta.axvline(0.0, color=PASTEL.ref_line, linestyle="--",
                     linewidth=0.9, alpha=0.7, zorder=0)

    for i, (delta_c, col) in enumerate(zip(df["delta_c"].values, colors)):
        ax_delta.plot([0, delta_c], [y_pos[i], y_pos[i]],
                      color=col, linewidth=1.8, alpha=0.7, zorder=1)
        ax_delta.scatter(delta_c, y_pos[i],
                         color=col, s=55, zorder=2,
                         edgecolors="white", linewidth=0.5)
        ax_delta.text(
            delta_c + 0.0005, y_pos[i],
            f"  +{delta_c:.4f}",
            va="center", ha="left", fontsize=7.5, color=col,
        )

    ax_delta.set_yticks(y_pos)
    ax_delta.set_yticklabels(labels, fontsize=8)
    ax_delta.set_xlabel("ΔC-index (joint − prodromal-only)", fontsize=8.5)
    ax_delta.set_title("Discrimination Gain\n(adding RBD mediator)", fontsize=9,
                       fontweight="bold")
    ax_delta.set_xlim(-0.001, df["delta_c"].max() * 1.6)
    ax_delta.invert_yaxis()
    ax_delta.grid(True, axis="x", alpha=0.3)
    ax_delta.grid(False, axis="y")

    # ── Right: Absolute C-index comparison ──────────────────────────────
    c_base_col = "c_path_total_c_index"
    c_joint_col = "joint_cprime_b_c_index"

    if c_base_col in df.columns and c_joint_col in df.columns:
        x_min = df[c_base_col].min() - 0.005
        x_max = df[c_joint_col].max() + 0.005

        for i, (_, row) in enumerate(df.iterrows()):
            c_base = row.get(c_base_col, np.nan)
            c_joint = row.get(c_joint_col, np.nan)
            col = colors[i]

            if pd.notna(c_base) and pd.notna(c_joint):
                # Arrow from base → joint
                ax_cindex.annotate(
                    "",
                    xy=(c_joint, y_pos[i]),
                    xytext=(c_base, y_pos[i]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=col, lw=1.5,
                    ),
                )
                ax_cindex.scatter(c_base, y_pos[i],
                                  color=col, s=40, alpha=0.5,
                                  marker="o", edgecolors="white")
                ax_cindex.scatter(c_joint, y_pos[i],
                                  color=col, s=55, alpha=0.9,
                                  marker="D", edgecolors="white")
                ax_cindex.text(
                    c_joint + 0.0003, y_pos[i],
                    f"  {c_joint:.4f}",
                    va="center", ha="left", fontsize=7, color=col,
                )

        ax_cindex.set_yticks(y_pos)
        ax_cindex.set_yticklabels(labels, fontsize=8)
        ax_cindex.set_xlabel("C-index", fontsize=8.5)
        ax_cindex.set_title(
            "C-index: circle=prodromal-only, diamond=joint\n(arrow = improvement from RBD)",
            fontsize=9, fontweight="bold",
        )
        ax_cindex.set_xlim(x_min, x_max + 0.01)
        ax_cindex.invert_yaxis()
        ax_cindex.grid(True, axis="x", alpha=0.3)
        ax_cindex.grid(False, axis="y")

    # Legend
    legend_handles = [
        mpatches.Patch(color=PASTEL.interp_a, label="Interpretation A (binary)"),
        mpatches.Patch(color=PASTEL.interp_c, label="Interpretation C (cognitive)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2,
        fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(fig, out_path)


# ── Figure 5: Association overview ─────────────────────────────────────────

def plot_association_overview(
    df_linear: pd.DataFrame,
    df_logistic: pd.DataFrame,
    df_logistic_3g: pd.DataFrame,
    out_path: Path,
    interp_label: str = "A",
    outcome_label: str = "PD (outcome_1a)",
) -> None:
    """
    Three-panel association dot plot: OLS β, binary logistic OR, 3g OR.

    Panel A: OLS beta (prodromal → RBD z-score, linear association).
    Panel B: Logistic OR (prodromal → RBD binary high ≥p99).
    Panel C: Multinomial OR (prodromal → RBD 3-group: Intermediate & High vs Low).

    Parameters
    ----------
    df_linear : pd.DataFrame
        assoc_linear.xlsx.
    df_logistic : pd.DataFrame
        assoc_logistic.xlsx.
    df_logistic_3g : pd.DataFrame
        assoc_logistic_3g.xlsx.
    out_path : Path
        Destination PNG.
    interp_label : str
        "A" or "C".
    outcome_label : str
        Outcome label for title.
    """
    _setup_style()
    if df_linear is None or df_linear.empty:
        return

    n_vars = len(df_linear)
    y_pos = np.arange(n_vars)
    labels = [_clean_label(str(r["label"])) for _, r in df_linear.iterrows()]

    # Pull total N from linear model (same for all)
    total_n = int(df_linear["n"].iloc[0]) if "n" in df_linear.columns else 0

    interp_color = PASTEL.interp_a if interp_label == "A" else PASTEL.interp_c

    fig, axes = plt.subplots(
        1, 3, figsize=(14, max(3.5, 0.65 * n_vars + 1.5)),
        gridspec_kw={"width_ratios": [1, 1, 1.4]},
    )
    fig.suptitle(
        f"Association with RBD — Interpretation {interp_label} | {outcome_label}\n"
        f"N={total_n:,}  (Model 1a: OLS on RBD z-score; 1b: Logistic on RBD ≥p99; "
        f"1b-3g: Multinomial)",
        fontsize=9.5, fontweight="bold",
    )

    # ── Panel A: OLS β ──────────────────────────────────────────────────
    ax = axes[0]
    ax.axvline(0.0, color=PASTEL.ref_line, linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=0)

    df_sorted_lin = df_linear.sort_values("beta", ascending=True).reset_index(drop=True)
    lin_labels = [_clean_label(str(r["label"])) for _, r in df_sorted_lin.iterrows()]

    for i, (_, row) in enumerate(df_sorted_lin.iterrows()):
        p = row.get("p_fdr", row.get("p", np.nan))
        col = interp_color if (pd.notna(p) and p < 0.05) else PASTEL.nonsig
        beta = row["beta"]
        lo = row["ci_lower"]
        hi = row["ci_upper"]
        ax.errorbar(
            beta, i,
            xerr=[[max(0, beta - lo)], [max(0, hi - beta)]],
            fmt="o", color=col, ecolor=col,
            elinewidth=1.2, capsize=3,
            markersize=5.5, markeredgewidth=0.4,
            markeredgecolor="white", alpha=0.85,
        )
        star = _pstar(row.get("p", np.nan), p)
        ax.text(hi + abs(beta) * 0.05, i, f"  {star}",
                va="center", fontsize=7,
                color=col if star != "ns" else PASTEL.text_muted)

    ax.set_yticks(np.arange(len(df_sorted_lin)))
    ax.set_yticklabels(lin_labels, fontsize=8)
    ax.set_xlabel("β (95% CI)", fontsize=8.5)
    ax.set_title("Model 1a: OLS\nProdromal → RBD z-score", fontsize=9,
                 fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(False, axis="y")

    # ── Panel B: Logistic OR ─────────────────────────────────────────────
    ax = axes[1]
    ax.set_xscale("log")
    ax.axvline(1.0, color=PASTEL.ref_line, linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=0)

    if df_logistic is not None and not df_logistic.empty:
        df_sorted_log = df_logistic.sort_values(
            "or_val", ascending=True
        ).reset_index(drop=True)
        log_labels = [_clean_label(str(r["label"])) for _, r in df_sorted_log.iterrows()]

        for i, (_, row) in enumerate(df_sorted_log.iterrows()):
            p = row.get("p_fdr", row.get("p", np.nan))
            col = interp_color if (pd.notna(p) and p < 0.05) else PASTEL.nonsig
            or_v = row["or_val"]
            or_lo = row["or_lower"]
            or_hi = row["or_upper"]
            n_h = int(row.get("n_high", 0))
            ax.errorbar(
                or_v, i,
                xerr=[[max(0, or_v - or_lo)], [max(0, or_hi - or_v)]],
                fmt="o", color=col, ecolor=col,
                elinewidth=1.2, capsize=3,
                markersize=5.5, markeredgewidth=0.4,
                markeredgecolor="white", alpha=0.85,
            )
            star = _pstar(row.get("p", np.nan), p)
            ax.text(or_hi * 1.02, i, f"  {star}",
                    va="center", fontsize=7,
                    color=col if star != "ns" else PASTEL.text_muted)

        ax.set_yticks(np.arange(len(df_sorted_log)))
        ax.set_yticklabels(log_labels, fontsize=8)
        n_high_note = int(df_logistic.get("n_high", pd.Series([0])).iloc[0])
        ax.set_xlabel("OR (95% CI, log scale)", fontsize=8.5)
        ax.set_title(
            f"Model 1b: Logistic\nProdromal → RBD High (≥p99, n={n_high_note:,})",
            fontsize=9, fontweight="bold",
        )
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")

    # ── Panel C: 3-group multinomial OR ─────────────────────────────────
    ax = axes[2]
    ax.set_xscale("log")
    ax.axvline(1.0, color=PASTEL.ref_line, linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=0)

    color_intermed = "#9BB3CE"
    color_high = "#E8916A"

    if df_logistic_3g is not None and not df_logistic_3g.empty:
        # One row per variable in the unique-variable ordering
        vars_ordered = df_logistic.sort_values("or_val", ascending=True)["variable"].tolist() \
            if df_logistic is not None and not df_logistic.empty \
            else df_logistic_3g["variable"].unique().tolist()

        for i, var in enumerate(vars_ordered):
            sub = df_logistic_3g[df_logistic_3g["variable"] == var]
            intermed_row = sub[sub["contrast"].str.contains("Intermediate", na=False)]
            high_row = sub[sub["contrast"].str.contains("High", na=False)]

            y_i = i * 2.2 + 0.4
            y_h = i * 2.2

            for row_df, color, y in [
                (intermed_row, color_intermed, y_i),
                (high_row, color_high, y_h),
            ]:
                if row_df.empty:
                    continue
                r = row_df.iloc[0]
                p = r.get("p_fdr", r.get("p", np.nan))
                col = color if (pd.notna(p) and p < 0.05) else PASTEL.nonsig
                or_v = r["or_val"]
                or_lo = r["or_lower"]
                or_hi = r["or_upper"]
                ax.errorbar(
                    or_v, y,
                    xerr=[[max(0, or_v - or_lo)], [max(0, or_hi - or_v)]],
                    fmt="s" if color == color_intermed else "o",
                    color=col, ecolor=col,
                    elinewidth=1.2, capsize=3,
                    markersize=5.5, markeredgewidth=0.4,
                    markeredgecolor="white", alpha=0.85,
                )
                star = _pstar(r.get("p", np.nan), p)
                ax.text(or_hi * 1.02, y, f"  {star}",
                        va="center", fontsize=7,
                        color=col if star != "ns" else PASTEL.text_muted)

        y_mids = np.arange(len(vars_ordered)) * 2.2 + 0.2
        threeg_labels = []
        for var in vars_ordered:
            sub = df_logistic_3g[df_logistic_3g["variable"] == var]
            if not sub.empty:
                threeg_labels.append(_clean_label(str(sub.iloc[0]["label"])))
            else:
                threeg_labels.append(var)
        ax.set_yticks(y_mids)
        ax.set_yticklabels(threeg_labels, fontsize=8)
        ax.set_xlabel("OR (95% CI, log scale)", fontsize=8.5)
        ax.set_title(
            "Model 1b-3g: Multinomial\nProdromal → RBD (Intermed/High vs Low)",
            fontsize=9, fontweight="bold",
        )
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")

        legend_handles_3g = [
            plt.Line2D(
                [0], [0], marker="s", color="w", markerfacecolor=color_intermed,
                markersize=6, label="Intermediate vs Low",
            ),
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color_high,
                markersize=6, label="High vs Low",
            ),
        ]
        ax.legend(handles=legend_handles_3g, loc="lower right",
                  fontsize=7.5, framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_path)


# ── Orchestrator ────────────────────────────────────────────────────────────

def run_mediation_plots(
    results_dir: Path,
    outcome_label: str = "PD (outcome_1a)",
    dpi: int = 300,
) -> None:
    """
    Load mediation result tables from disk and generate all publication figures.

    Expects the directory structure produced by run_mediation():
        results_dir/
          mediation/
            interpretation_A/   *.xlsx
            interpretation_C/   *.xlsx
            report/             *.xlsx

    Figures are saved to:
        results_dir/mediation/figures/

    Parameters
    ----------
    results_dir : Path
        Parent results directory (e.g. results/cox_prodromal_abk_<timestamp>).
    outcome_label : str
        Human-readable outcome string for all figure titles.
    dpi : int
        Output DPI (default 300).
    """
    med_dir = results_dir / "mediation"
    fig_dir = med_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _load(path: Path) -> Optional[pd.DataFrame]:
        """Load Excel if it exists, else return None."""
        if path.exists():
            return pd.read_excel(path)
        print(f"  [skip] not found: {path.name}")
        return None

    # ── Load all tables ──────────────────────────────────────────────────
    dir_a = med_dir / "interpretation_A"
    dir_c = med_dir / "interpretation_C"
    dir_rep = med_dir / "report"

    tables: Dict[str, Optional[pd.DataFrame]] = {
        "a_steps": _load(dir_a / "mediation_steps.xlsx"),
        "a_indirect": _load(dir_a / "mediation_indirect.xlsx"),
        "a_linear": _load(dir_a / "assoc_linear.xlsx"),
        "a_logistic": _load(dir_a / "assoc_logistic.xlsx"),
        "a_logistic_3g": _load(dir_a / "assoc_logistic_3g.xlsx"),
        "a_3g_bpath": _load(dir_a / "supplementary_3g_bpath.xlsx"),
        "c_steps": _load(dir_c / "mediation_steps.xlsx"),
        "c_indirect": _load(dir_c / "mediation_indirect.xlsx"),
        "c_linear": _load(dir_c / "assoc_linear.xlsx"),
        "c_logistic": _load(dir_c / "assoc_logistic.xlsx"),
        "c_logistic_3g": _load(dir_c / "assoc_logistic_3g.xlsx"),
        "c_3g_bpath": _load(dir_c / "supplementary_3g_bpath.xlsx"),
        "perf_summary": _load(dir_rep / "model_performance_summary.xlsx"),
    }

    print(f"\n[Mediation Plots] Output -> {fig_dir}")

    # ── Figure 1a: Mediation steps forest — Interpretation A ────────────
    if tables["a_steps"] is not None:
        plot_mediation_steps_forest(
            df_steps=tables["a_steps"],
            out_path=fig_dir / "fig1a_mediation_steps_interp_A.png",
            interp_label="A",
            outcome_label=outcome_label,
        )

    # ── Figure 1b: Mediation steps forest — Interpretation C ────────────
    if tables["c_steps"] is not None:
        plot_mediation_steps_forest(
            df_steps=tables["c_steps"],
            out_path=fig_dir / "fig1b_mediation_steps_interp_C.png",
            interp_label="C",
            outcome_label=outcome_label,
        )

    # ── Figure 2a: Indirect effect — Interpretation A ───────────────────
    if tables["a_steps"] is not None:
        plot_indirect_effect_forest(
            df_steps=tables["a_steps"],
            df_boot=tables["a_indirect"],
            out_path=fig_dir / "fig2a_indirect_effect_interp_A.png",
            interp_label="A",
            outcome_label=outcome_label,
        )

    # ── Figure 2b: Indirect effect — Interpretation C ───────────────────
    if tables["c_steps"] is not None:
        plot_indirect_effect_forest(
            df_steps=tables["c_steps"],
            df_boot=tables["c_indirect"],
            out_path=fig_dir / "fig2b_indirect_effect_interp_C.png",
            interp_label="C",
            outcome_label=outcome_label,
        )

    # ── Figure 3a: 3-group b-path — Interpretation A ────────────────────
    if tables["a_3g_bpath"] is not None:
        plot_3group_bpath_forest(
            df_3g=tables["a_3g_bpath"],
            out_path=fig_dir / "fig3a_3g_bpath_interp_A.png",
            interp_label="A",
            outcome_label=outcome_label,
        )

    # ── Figure 3b: 3-group b-path — Interpretation C ────────────────────
    if tables["c_3g_bpath"] is not None:
        plot_3group_bpath_forest(
            df_3g=tables["c_3g_bpath"],
            out_path=fig_dir / "fig3b_3g_bpath_interp_C.png",
            interp_label="C",
            outcome_label=outcome_label,
        )

    # ── Figure 4: ΔC-index — combined A + C ─────────────────────────────
    if tables["perf_summary"] is not None:
        plot_model_performance_delta(
            df_perf_summary=tables["perf_summary"],
            out_path=fig_dir / "fig4_delta_cindex_combined.png",
            outcome_label=outcome_label,
        )

    # ── Figure 5a: Association overview — Interpretation A ──────────────
    if tables["a_linear"] is not None:
        plot_association_overview(
            df_linear=tables["a_linear"],
            df_logistic=tables["a_logistic"],
            df_logistic_3g=tables["a_logistic_3g"],
            out_path=fig_dir / "fig5a_association_overview_interp_A.png",
            interp_label="A",
            outcome_label=outcome_label,
        )

    # ── Figure 5b: Association overview — Interpretation C ──────────────
    if tables["c_linear"] is not None:
        plot_association_overview(
            df_linear=tables["c_linear"],
            df_logistic=tables["c_logistic"],
            df_logistic_3g=tables["c_logistic_3g"],
            out_path=fig_dir / "fig5b_association_overview_interp_C.png",
            interp_label="C",
            outcome_label=outcome_label,
        )

    print(f"[Mediation Plots] All figures saved -> {fig_dir}")
