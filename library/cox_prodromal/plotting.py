"""
Harmonized plotting functions for Cox prodromal analysis.

All plots use the centralized ``RISK_PALETTE`` from config for consistent
color mapping across figures. KM curves include risk ratio (RR) annotations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

plt.rcParams["svg.fonttype"] = "none"
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from library.cox_prodromal.cox_config import (
    ABSOLUTE_RISK_TIMEPOINTS,
    RISK_PALETTE,
)


# ── Color mapping ──────────────────────────────────────────────────────────

def get_rbd_group_color(label: str) -> str:
    """
    Map an RBD risk group label to its palette color.

    Parameters
    ----------
    label : str
        Group label (e.g. 'High', 'Low', 'Mid').

    Returns
    -------
    str
        Hex color code.
    """
    lower = label.lower().strip()
    if "high" in lower:
        return RISK_PALETTE.rbd_high
    if "mid" in lower or "medium" in lower or "inter" in lower:
        return RISK_PALETTE.rbd_mid
    if "low" in lower:
        return RISK_PALETTE.rbd_low
    return "#999999"


def get_prodromal_color(label: str) -> str:
    """
    Map a prodromal marker label to its palette color.

    Parameters
    ----------
    label : str
        Group label (e.g. 'Yes', 'No', 'High', 'Low').

    Returns
    -------
    str
        Hex color code.
    """
    lower = label.lower().strip()
    if lower in ("yes", "high"):
        return RISK_PALETTE.prodromal_yes
    if lower in ("no", "low"):
        return RISK_PALETTE.prodromal_no
    if "mid" in lower or "medium" in lower:
        return RISK_PALETTE.rbd_mid
    return "#999999"


def get_combined_color(label: str) -> str:
    """
    Map a combined RBD/prodromal label to a color.

    Uses the RBD component to determine the base hue, with opacity
    or saturation shift for prodromal status.

    Parameters
    ----------
    label : str
        Label in format 'RBD_group / Prodromal_group'.

    Returns
    -------
    str
        Hex color code.
    """
    from config.config import RBD_RISK_COLORS_COMBINED  # local import avoids circular
    parts = label.split("/")
    if len(parts) >= 2:
        rbd_part = parts[0].strip().lower()
        prod_part = parts[1].strip().lower()
        if "high" in rbd_part and prod_part in ("yes", "high"):
            return RBD_RISK_COLORS_COMBINED["High_Yes"]
        if "high" in rbd_part:
            return RBD_RISK_COLORS_COMBINED["High"]
        if "low" in rbd_part and prod_part in ("no", "low"):
            return RBD_RISK_COLORS_COMBINED["Low_No"]
        if "low" in rbd_part:
            return RBD_RISK_COLORS_COMBINED["Low"]
        if "mid" in rbd_part and prod_part in ("yes", "high"):
            return RBD_RISK_COLORS_COMBINED["Mid_Yes"]
        if "mid" in rbd_part:
            return RBD_RISK_COLORS_COMBINED["Mid"]
    return get_rbd_group_color(label)


def _select_color_fn(group_col: str) -> Callable[[str], str]:
    """Choose the correct color function based on column name."""
    lower = group_col.lower()
    if "combined" in lower:
        return get_combined_color
    if "rbd" in lower or "risk_group" in lower:
        return get_rbd_group_color
    return get_prodromal_color


# ── Ghost figure helper ────────────────────────────────────────────────────

def _save_ghost_copy(fig: plt.Figure, path_stem: Path) -> None:
    """Save a stripped, transparent-background version of *fig*.

    Keeps curves, CI bands, threshold lines, spines, and tick marks.
    Strips title, suptitle, legend, axis labels, and grid from all axes.
    Saves as ``<path_stem>.png`` (300 dpi) and ``<path_stem>.svg``.

    Must be called *after* the primary ``fig.savefig()`` — this function
    mutates the figure in-place.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Already-saved figure to derive the ghost from.
    path_stem : Path
        Output path without extension (extensions added automatically).
    """
    path_stem = Path(path_stem)

    # Hide figure-level text (suptitle lives in fig.texts)
    for txt in fig.texts:
        txt.set_visible(False)

    for ax in fig.axes:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        lgd = ax.get_legend()
        if lgd is not None:
            lgd.remove()
        ax.grid(False)
        ax.patch.set_alpha(0.0)

    fig.patch.set_alpha(0.0)

    fig.savefig(path_stem.with_suffix(".png"), dpi=300,
                bbox_inches="tight", transparent=True)
    fig.savefig(path_stem.with_suffix(".svg"),
                bbox_inches="tight", transparent=True)


# ── Risk ratio computation ─────────────────────────────────────────────────

def _compute_rr_at_timepoint(
    kmf_dict: Dict[str, KaplanMeierFitter],
    ref_group: str,
    t0: float,
) -> Dict[str, Optional[float]]:
    """
    Compute cumulative incidence risk ratio (RR) at a fixed timepoint.

    RR = CIF_group(t0) / CIF_ref(t0)  where CIF = 1 - S(t).

    Parameters
    ----------
    kmf_dict : dict
        {group_label: fitted KaplanMeierFitter}.
    ref_group : str
        Reference group label.
    t0 : float
        Timepoint in years.

    Returns
    -------
    dict
        {group_label: RR} (ref group = 1.0).
    """
    rr: Dict[str, Optional[float]] = {}

    ref_sf = kmf_dict.get(ref_group)
    if ref_sf is None:
        return rr
    ref_at_t = ref_sf.survival_function_[ref_sf.survival_function_.index <= t0]
    if ref_at_t.empty:
        return rr
    ref_cif = 1 - float(ref_at_t.iloc[-1].values[0])
    if ref_cif <= 0:
        return rr

    for grp, kmf_fitted in kmf_dict.items():
        if grp == ref_group:
            rr[grp] = 1.0
            continue
        sf_at_t = kmf_fitted.survival_function_[
            kmf_fitted.survival_function_.index <= t0
        ]
        if sf_at_t.empty:
            rr[grp] = None
            continue
        grp_cif = 1 - float(sf_at_t.iloc[-1].values[0])
        rr[grp] = round(grp_cif / ref_cif, 2) if ref_cif > 0 else None

    return rr


def _identify_ref_group(groups: List[str]) -> str:
    """Pick the lowest-risk group as reference for RR computation."""
    for g in groups:
        gl = g.lower()
        if "low" in gl or gl.startswith("no"):
            return g
    return groups[0]


# ── KM panel ───────────────────────────────────────────────────────────────

def _adaptive_ylim_lower(kmf_dict: Dict[str, "KaplanMeierFitter"]) -> float:
    """
    Compute an adaptive y-axis lower bound from a set of fitted KMFs.

    Logic: take the minimum terminal survival across all groups, subtract a
    fixed margin of 0.02, then floor to the nearest 0.01 grid step.
    This keeps the axis tight around the data regardless of event rate —
    a 0.05 grid would snap a terminal survival of 0.96 down to 0.90,
    wasting most of the plot area.
    Clipped to [0, 0.99] so the axis is never inverted or trivially narrow.
    """
    term_survs = [
        float(kmf.survival_function_.iloc[-1].values[0])
        for kmf in kmf_dict.values()
        if not kmf.survival_function_.empty
    ]
    if not term_survs:
        return 0.85

    point_clearance_below_lowest_curve = 0.01
    return float(np.clip(
        np.floor((min(term_survs) - point_clearance_below_lowest_curve) / 0.01) * 0.01,
        0.0, 0.99,
    ))


def plot_km_panel(
    ax: Axes,
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    title: str,
    color_fn: Optional[Callable[[str], str]] = None,
    show_rr: bool = True,
    t0_years: float = 5.0,
    ylim_lower: Optional[float] = None,
) -> float:
    """
    Plot Kaplan-Meier curves with harmonized colors and optional RR annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    df : pd.DataFrame
        Must contain ``time_col``, ``event_col``, ``group_col``.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    group_col : str
        Grouping variable.
    title : str
        Panel title.
    color_fn : callable, optional
        Maps group label -> hex color. If None, auto-selected.
    show_rr : bool
        Whether to annotate legend with RR(t0).
    t0_years : float
        Timepoint for RR computation (default 5.0).
    ylim_lower : float or None
        Override the adaptive lower y-axis bound.  When None (default) the
        bound is derived from the data via ``_adaptive_ylim_lower``.  Pass a
        precomputed global minimum when aligning multiple panels to a common
        scale.

    Returns
    -------
    float
        Log-rank p-value (NaN if < 2 groups).
    """
    df_plot = df.dropna(subset=[time_col, event_col, group_col]).copy()
    groups = sorted(df_plot[group_col].astype(str).unique())
    p_val = np.nan

    if not groups:
        ax.axis("off")
        return p_val

    if color_fn is None:
        color_fn = _select_color_fn(group_col)

    kmf_dict: Dict[str, KaplanMeierFitter] = {}
    for grp in groups:
        mask = df_plot[group_col].astype(str) == grp
        kmf = KaplanMeierFitter()
        kmf.fit(df_plot.loc[mask, time_col], df_plot.loc[mask, event_col])
        kmf_dict[grp] = kmf

    # Compute RR
    rr_dict: Dict[str, Optional[float]] = {}
    if show_rr and len(groups) >= 2:
        ref = _identify_ref_group(groups)
        rr_dict = _compute_rr_at_timepoint(kmf_dict, ref, t0_years)

    # Plot each group
    for grp in groups:
        kmf = kmf_dict[grp]
        mask = df_plot[group_col].astype(str) == grp
        n = int(mask.sum())
        ev = int(df_plot.loc[mask, event_col].sum())
        rr_str = ""
        if grp in rr_dict and rr_dict[grp] is not None:
            rr_val = rr_dict[grp]
            if rr_val != 1.0:
                rr_str = f", RR={rr_val:.1f}"
            else:
                rr_str = ", ref"
        label = f"{grp} (n={n}, e={ev}{rr_str})"
        kmf.plot_survival_function(
            ax=ax, ci_show=True, color=color_fn(grp), label=label, lw=2.5
        )

    # Log-rank test
    if len(groups) >= 2:
        try:
            res = multivariate_logrank_test(
                df_plot[time_col],
                df_plot[group_col].astype(str),
                df_plot[event_col],
            )
            p_val = res.p_value
        except Exception:
            pass

    p_str = f"p = {p_val:.3g}" if np.isfinite(p_val) else ""
    ax.set_title(f"{title}\n{p_str}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")

    lower = ylim_lower if ylim_lower is not None else _adaptive_ylim_lower(kmf_dict)
    ax.set_ylim(lower, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    return p_val


# ── Three-panel KM ─────────────────────────────────────────────────────────

def plot_three_panel_km(
    df: pd.DataFrame,
    df_full: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    prod_grp_col: str,
    combined_grp_col: str,
    outcome_label: str,
    method_label: str,
    prod_label: str,
    save_path: str,
    t0_years: float = 5.0,
) -> Dict[str, float]:
    """
    Create a 4-panel KM figure (2x2):
      A: By RBD (complete-case subset)
      B: By Prodromal marker (complete-case subset)
      C: Combined RBD x Prodromal (complete-case subset)
      D: By RBD — full cohort, no covariate/prodromal NA filter

    Panels A–C are restricted to subjects with no missing covariates or
    prodromal variable (same subset used in Cox models). Panel D shows the
    RBD stratification on the maximum available sample, making the
    covariate-completeness cost visible at a glance.

    Parameters
    ----------
    df : pd.DataFrame
        Complete-case survival dataset (covariates + prodromal + rbd_col non-null).
    df_full : pd.DataFrame
        Full outcome cohort dropna on [time_col, event_col, rbd_col] only.
    rbd_col : str
        RBD risk group column.
    prod_grp_col : str
        Prodromal group column.
    combined_grp_col : str
        Combined group column (e.g. 'High / Yes').
    outcome_label : str
        Display label for the outcome.
    method_label : str
        Display label for the method.
    prod_label : str
        Display label for the prodromal variable.
    save_path : str
        Full path for the saved figure.
    t0_years : float
        Timepoint for RR computation.

    Returns
    -------
    dict
        Keys: logrank_rbd_p, logrank_prod_p, logrank_combined_p, logrank_full_rbd_p.
    """
    n_cc = len(df)
    n_full = len(df_full)
    n_events_cc = int(df[event_col].sum())
    n_events_full = int(df_full[event_col].sum())

    # ── Compute shared y-limit across all 4 panels ──────────────────────────
    # Fit KMFs for each panel dataset/grouping to find the global minimum
    # terminal survival, then snap to a common adaptive lower bound.
    def _fit_kmfs(data: pd.DataFrame, grp_col: str) -> Dict[str, KaplanMeierFitter]:
        d = data.dropna(subset=[time_col, event_col, grp_col]).copy()
        out: Dict[str, KaplanMeierFitter] = {}
        for g in d[grp_col].astype(str).unique():
            mask = d[grp_col].astype(str) == g
            kmf = KaplanMeierFitter()
            kmf.fit(d.loc[mask, time_col], d.loc[mask, event_col])
            out[g] = kmf
        return out

    all_kmfs: Dict[str, KaplanMeierFitter] = {}
    all_kmfs.update(_fit_kmfs(df, rbd_col))
    all_kmfs.update(_fit_kmfs(df, prod_grp_col))
    all_kmfs.update(_fit_kmfs(df, combined_grp_col))
    all_kmfs.update(_fit_kmfs(df_full, rbd_col))
    shared_lower = _adaptive_ylim_lower(all_kmfs)

    fig, axes = plt.subplots(2, 2, figsize=(16.1, 11.9))
    axes_flat = axes.flatten()  # [A, B, C, D]
    fig.suptitle(
        f"{outcome_label} | {method_label} | {prod_label}\n"
        f"Complete-case N={n_cc:,} (events={n_events_cc})  "
        f"Full-cohort N={n_full:,} (events={n_events_full})",
        fontsize=11, fontweight="bold",
    )

    p_a = plot_km_panel(
        axes_flat[0], df, time_col, event_col, rbd_col,
        "A. By RBD Risk Group (complete-case)",
        color_fn=get_rbd_group_color, t0_years=t0_years, ylim_lower=shared_lower,
    )
    p_b = plot_km_panel(
        axes_flat[1], df, time_col, event_col, prod_grp_col,
        f"B. By {prod_label} (complete-case)",
        color_fn=get_prodromal_color, t0_years=t0_years, ylim_lower=shared_lower,
    )
    p_c = plot_km_panel(
        axes_flat[2], df, time_col, event_col, combined_grp_col,
        "C. Combined RBD x Prodromal (complete-case)",
        color_fn=get_combined_color, t0_years=t0_years, ylim_lower=shared_lower,
    )
    p_d = plot_km_panel(
        axes_flat[3], df_full, time_col, event_col, rbd_col,
        "D. By RBD Risk Group (full cohort, no covariate filter)",
        color_fn=get_rbd_group_color, t0_years=t0_years, ylim_lower=shared_lower,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "logrank_rbd_p": p_a,
        "logrank_prod_p": p_b,
        "logrank_combined_p": p_c,
        "logrank_full_rbd_p": p_d,
    }


# ── Full-cohort standalone KM ─────────────────────────────────────────────

def plot_rbd_only_km_full(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    outcome_label: str,
    method_label: str,
    save_path: str,
    t0_years: float = 5.0,
) -> float:
    """
    Standalone single-panel KM of RBD risk groups on the full outcome cohort.

    This figure is generated once per outcome × method — before any covariate
    or prodromal variable is applied — so it reflects the maximum available
    sample and is not affected by complete-case attrition.

    Parameters
    ----------
    df : pd.DataFrame
        Full outcome cohort, already dropna on [time_col, event_col, rbd_col].
    time_col, event_col : str
        Duration (years) and event indicator columns.
    rbd_col : str
        RBD risk group column (string-encoded labels expected).
    outcome_label : str
        Display label for the outcome (used in title).
    method_label : str
        Display label for the stratification method (used in title).
    save_path : str
        Full path for the saved figure (PNG).
    t0_years : float
        Timepoint (years) for cumulative incidence RR annotation.

    Returns
    -------
    float
        Log-rank p-value across RBD groups.
    """
    n_total = len(df)
    n_events = int(df[event_col].sum())

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 8.5))
    fig.suptitle(
        f"Full-Cohort KM | {outcome_label} | {method_label}\n"
        f"N={n_total:,}  events={n_events}  (no covariate filtering)",
        fontsize=11, fontweight="bold",
    )

    p_val = plot_km_panel(
        ax, df, time_col, event_col, rbd_col,
        "By RBD Risk Group",
        color_fn=get_rbd_group_color,
        show_rr=True,
        t0_years=t0_years,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _save_ghost_copy(fig, Path(save_path).with_name(Path(save_path).stem + "_ghost"))
    plt.close(fig)
    return p_val


# ── Cumulative incidence (1 − KM) figures ─────────────────────────────────

def plot_cumulative_incidence_rbd(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    outcome_label: str,
    method_label: str,
    save_path: str,
    t0_years: float = 5.0,
) -> float:
    """
    Standalone single-panel cumulative incidence (1 − KM) by RBD risk group.

    Complements ``plot_rbd_only_km_full``.  Cumulative incidence is preferred
    for visual communication when event rates are low (<5%): curves start at 0
    and rise, making absolute risk differences between groups directly legible
    without zooming into the top of a survival plot.

    Y-axis bounds are data-adaptive: upper bound is snapped to the nearest 0.01
    above the maximum observed cumulative incidence, with a 20% relative margin.
    This keeps the scale tight around the data range regardless of outcome rate.

    Parameters
    ----------
    df : pd.DataFrame
        Full outcome cohort, already dropna on [time_col, event_col, rbd_col].
    time_col, event_col : str
        Duration (years) and binary event indicator (0/1).
    rbd_col : str
        RBD risk group column (string-encoded labels expected).
    outcome_label : str
        Display label for the outcome (used in title).
    method_label : str
        Display label for the stratification method (used in title).
    save_path : str
        Full path for the saved figure (PNG).
    t0_years : float
        Timepoint (years) used for absolute risk annotation in the legend.

    Returns
    -------
    float
        Log-rank p-value across RBD groups (identical to KM log-rank).
    """
    df_plot = df.dropna(subset=[time_col, event_col, rbd_col]).copy()
    groups = sorted(df_plot[rbd_col].astype(str).unique())

    n_total = len(df_plot)
    n_events = int(df_plot[event_col].sum())

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 8.5))
    fig.suptitle(
        f"Cumulative Incidence (1−KM) | {outcome_label} | {method_label}\n"
        f"N={n_total:,}  events={n_events}  (no covariate filtering)",
        fontsize=11, fontweight="bold",
    )

    kmf_dict: Dict[str, KaplanMeierFitter] = {}
    for grp in groups:
        mask = df_plot[rbd_col].astype(str) == grp
        kmf = KaplanMeierFitter()
        kmf.fit(df_plot.loc[mask, time_col], df_plot.loc[mask, event_col])
        kmf_dict[grp] = kmf

    # Adaptive upper y-limit: ceiling to nearest 0.01, with a 20% relative margin
    max_ci = max(
        float(1 - kmf.survival_function_.iloc[-1].values[0])
        for kmf in kmf_dict.values()
        if not kmf.survival_function_.empty
    )
    upper = float(np.ceil(max_ci * 1.20 / 0.01) * 0.01)
    upper = float(np.clip(upper, 0.01, 1.0))

    for grp in groups:
        kmf = kmf_dict[grp]
        mask = df_plot[rbd_col].astype(str) == grp
        n = int(mask.sum())
        ev = int(df_plot.loc[mask, event_col].sum())

        # Absolute risk at t0_years from cumulative density
        ci_t0 = np.nan
        cd = kmf.cumulative_density_
        t_idx = cd.index.searchsorted(t0_years, side="right") - 1
        if 0 <= t_idx < len(cd):
            ci_t0 = float(cd.iloc[t_idx].values[0])
        ar_str = f", CI({t0_years:.0f}y)={ci_t0:.1%}" if np.isfinite(ci_t0) else ""

        label = f"{grp} (n={n}, e={ev}{ar_str})"
        kmf.plot_cumulative_density(
            ax=ax, ci_show=True, color=get_rbd_group_color(grp), label=label
        )

    # Log-rank test (same statistic as KM)
    p_val = np.nan
    if len(groups) >= 2:
        try:
            res = multivariate_logrank_test(
                df_plot[time_col],
                df_plot[rbd_col].astype(str),
                df_plot[event_col],
            )
            p_val = res.p_value
        except Exception:
            pass

    p_str = f"p = {p_val:.3g}" if np.isfinite(p_val) else ""
    ax.set_title(f"By RBD Risk Group\n{p_str}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Cumulative incidence")
    ax.set_ylim(0.0, upper)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    _save_ghost_copy(fig, Path(save_path).with_name(Path(save_path).stem + "_ghost"))
    plt.close(fig)
    return p_val


# ── Forest plot ────────────────────────────────────────────────────────────

def plot_forest_hr(
    ax: Axes,
    results_df: pd.DataFrame,
    title: str = "Forest Plot: Hazard Ratios",
    label_col: str = "prodromal_label",
    hr_col: str = "HR",
    lci_col: str = "HR_lower",
    uci_col: str = "HR_upper",
) -> None:
    """
    Forest plot of hazard ratios with 95% CI.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    results_df : pd.DataFrame
        One row per covariate with HR, LCI, UCI columns.
    title : str
        Plot title.
    label_col : str
        Column for y-axis labels.
    hr_col, lci_col, uci_col : str
        Column names for HR and its confidence interval bounds.
    """
    df_plot = results_df.dropna(subset=[hr_col, lci_col, uci_col]).copy()
    if df_plot.empty:
        ax.axis("off")
        return

    df_plot = df_plot.sort_values(hr_col, ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_plot))

    ax.errorbar(
        df_plot[hr_col], y_pos,
        xerr=[
            df_plot[hr_col] - df_plot[lci_col],
            df_plot[uci_col] - df_plot[hr_col],
        ],
        fmt="o", color="#2166AC", ecolor="#999999",
        capsize=3, markersize=6,
    )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot[label_col], fontsize=8)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")


# ── Dose-response plot ─────────────────────────────────────────────────────

def plot_rbd_dose_response(
    ax: Axes,
    hr_curve: pd.DataFrame,
    title: str = "RBD Dose-Response Curve",
    xlabel: str = "RBD Probability",
) -> None:
    """
    Plot the spline-derived HR curve for continuous RBD probability.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    hr_curve : pd.DataFrame
        Must contain columns: rbd_prob, HR, HR_LCI, HR_UCI.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    """
    if hr_curve.empty:
        ax.axis("off")
        return

    ax.plot(hr_curve["rbd_prob"], hr_curve["HR"],
            color=RISK_PALETTE.rbd_high, linewidth=2, label="HR")
    ax.fill_between(
        hr_curve["rbd_prob"],
        hr_curve["HR_LCI"], hr_curve["HR_UCI"],
        alpha=0.2, color=RISK_PALETTE.rbd_high, label="95% CI",
    )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Hazard Ratio (vs minimum)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


# ── Absolute risk table (KM-based) ────────────────────────────────────────

def compute_absolute_risks_km(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    timepoints: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    KM cumulative incidence (%) at each timepoint per group.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    group_col : str
        Grouping variable.
    timepoints : list[float], optional
        Fixed timepoints (default: [5.0, 10.0]).

    Returns
    -------
    pd.DataFrame
        Columns: group, timepoint_years, cum_inc_pct, ci_lower_pct,
        ci_upper_pct, n, events.
    """
    timepoints = timepoints or ABSOLUTE_RISK_TIMEPOINTS
    kmf = KaplanMeierFitter()
    rows: List[Dict[str, Any]] = []
    df_clean = df.dropna(subset=[time_col, event_col, group_col])

    for grp in sorted(df_clean[group_col].astype(str).unique()):
        mask = df_clean[group_col].astype(str) == grp
        sub = df_clean[mask]
        n = int(mask.sum())
        n_ev = int(sub[event_col].sum())
        kmf.fit(sub[time_col], sub[event_col])
        sf = kmf.survival_function_
        ci_df = kmf.confidence_interval_survival_function_

        for t in timepoints:
            sf_t = sf[sf.index <= t]
            ci_t = ci_df[ci_df.index <= t]
            if sf_t.empty:
                cum_inc = ci_lo = ci_hi = np.nan
            else:
                surv = float(sf_t.iloc[-1].values[0])
                cum_inc = (1 - surv) * 100
                lo_col = ci_t.columns[ci_t.columns.str.contains("lower")][0]
                hi_col = ci_t.columns[ci_t.columns.str.contains("upper")][0]
                ci_lo = (1 - float(ci_t.iloc[-1][hi_col])) * 100
                ci_hi = (1 - float(ci_t.iloc[-1][lo_col])) * 100
            rows.append({
                "group": str(grp),
                "timepoint_years": t,
                "cum_inc_pct": round(cum_inc, 2) if not np.isnan(cum_inc) else np.nan,
                "ci_lower_pct": round(ci_lo, 2) if not np.isnan(ci_lo) else np.nan,
                "ci_upper_pct": round(ci_hi, 2) if not np.isnan(ci_hi) else np.nan,
                "n": n,
                "events": n_ev,
            })


# ── RBD score distribution ─────────────────────────────────────────────────

def _infer_thresholds_from_groups(
    df: pd.DataFrame,
    prob_col: str,
    group_col: str,
    group_order: List[str],
) -> List[float]:
    """Infer group boundary thresholds from the max score in each non-final group.

    Returns N-1 threshold values for N groups, sorted ascending.
    """
    thresholds: List[float] = []
    for grp in group_order[:-1]:
        max_val = df.loc[df[group_col] == grp, prob_col].max()
        thresholds.append(float(max_val))
    return sorted(thresholds)


def make_risk_legend(
    group_labels: List[str],
    summary: Dict[str, Dict[str, Any]],
    colors: List[str],
) -> Tuple[List[Line2D], List[str]]:
    """Build legend handles for the RBD distribution plot.

    Each entry shows n, incident cases, controls, crude risk, and RR
    (Low group as reference).

    Parameters
    ----------
    group_labels : list[str]
        Ordered group labels Low → High.
    summary : dict
        ``{group: {"n": int, "cases": int, "controls": int}}``.
    colors : list[str]
        Hex color per group, same order as ``group_labels``.

    Returns
    -------
    handles, labels : list[Line2D], list[str]
    """
    risks: Dict[str, float] = {}
    for grp in group_labels:
        vals = summary.get(grp, {"n": 0, "cases": 0})
        n = vals["n"]
        risks[grp] = (vals["cases"] / n) if n > 0 else np.nan

    # Reference = lowest-risk group with at least 1 case
    ref_grp = next(
        (g for g in group_labels if summary.get(g, {}).get("cases", 0) > 0),
        None,
    )
    ref_risk = risks[ref_grp] if ref_grp is not None else np.nan

    handles: List[Line2D] = []
    labels: List[str] = []
    for grp, color in zip(group_labels, colors):
        vals = summary.get(grp, {"n": 0, "cases": 0, "controls": 0})
        rr = (risks[grp] / ref_risk) if (np.isfinite(ref_risk) and ref_risk > 0) else np.nan
        risk_str = f"risk={risks[grp]:.3f}" if np.isfinite(risks[grp]) else "risk=NA"
        rr_str = f"RR={rr:.2f}" if np.isfinite(rr) else "RR=NA"
        star = " *" if grp == ref_grp else ""
        handles.append(Line2D([0], [0], marker="o", linestyle="",
                               markersize=10, color=color))
        labels.append(
            f"{grp}{star}  n={vals['n']}\n"
            f"  cases={vals['cases']}, ctrl={vals['controls']}\n"
            f"  {risk_str}, {rr_str}"
        )
    return handles, labels


def plot_rbd_distribution_single(
    df: pd.DataFrame,
    prob_col: str = "rbd_prob",
    group_col: str = "rg_pctl3",
    incident_col: str = "outcome_1a_pd_only__incident",
    group_order: Optional[List[str]] = None,
    bins: int = 40,
    figsize: Tuple[float, float] = (7.0, 4.8),
    save_path: Optional[Path] = None,
    filename_stem: str = "rbd_score_distribution",
) -> None:
    """Publication-grade RBD score distribution by risk group.

    Overlapping histograms + KDE curves coloured by ``rg_pctl3`` group,
    with vertical dashed lines at group boundaries and a summary legend.
    Saves PDF, PNG, and ghost (PNG + SVG) variants.

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level DataFrame containing ``prob_col``, ``group_col``,
        and ``incident_col``.
    prob_col : str
        Column with RBD probability scores.
    group_col : str
        Column with risk group labels (``"Low"`` / ``"Mid"`` / ``"High"``).
    incident_col : str
        Binary incident-outcome column (0/1).
    group_order : list[str] | None
        Ordered labels Low → High.  Defaults to ``["Low", "Mid", "High"]``.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size in inches.
    save_path : Path | None
        Directory to write output files.  Skips saving if ``None``.
    filename_stem : str
        Base filename without extension.
    """
    import seaborn as sns
    from config.config import RBD_RISK_COLORS as _RC

    if group_order is None:
        group_order = ["Low", "Mid", "High"]

    _base_colors = [_RC["Low"], _RC["Mid"], _RC["High"]]
    colors = [_base_colors[i] if i < len(_base_colors) else "#999999"
              for i in range(len(group_order))]

    for col in (prob_col, group_col, incident_col):
        if col not in df.columns:
            raise KeyError(f"Required column not found: {col!r}")

    keep_cols = [prob_col, group_col, incident_col]
    if "control" in df.columns:
        keep_cols.append("control")
    df_plot = df[keep_cols].dropna(subset=[prob_col, group_col])

    thr_vals = _infer_thresholds_from_groups(df_plot, prob_col, group_col, group_order)

    summary: Dict[str, Dict[str, Any]] = {}
    for grp in group_order:
        sub = df_plot[df_plot[group_col] == grp]
        n = len(sub)
        cases = int(np.nansum(sub[incident_col].values)) if incident_col in sub.columns else 0
        summary[grp] = {"n": n, "cases": cases, "controls": n - cases}

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.25)

    fig, ax = plt.subplots(figsize=figsize)

    x_min = df_plot[prob_col].min()
    x_max = df_plot[prob_col].max()
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    for grp, color in zip(group_order, colors):
        sub = df_plot[df_plot[group_col] == grp][prob_col]
        ax.hist(sub, bins=bin_edges, color=color, alpha=0.45,
                edgecolor="white", linewidth=0.3, zorder=2)

    for grp, color in zip(group_order, colors):
        sub = df_plot[df_plot[group_col] == grp][prob_col]
        if len(sub) > 5:
            sns.kdeplot(sub, ax=ax, color=color, linewidth=2.0, alpha=0.9, zorder=3)

    for thr in thr_vals:
        ax.axvline(thr, color="#333333", linestyle="--",
                   linewidth=1.6, alpha=0.85, zorder=4)

    handles, legend_labels = make_risk_legend(group_order, summary, colors)
    ax.legend(handles, legend_labels, fontsize=9, loc="upper right",
              frameon=True, framealpha=0.95, edgecolor="0.75",
              title="Risk Group (3-tier percentile)", title_fontsize=9.5,
              handletextpad=0.6, labelspacing=0.9)

    ax.set_xlabel("RBD score", fontsize=11, labelpad=6)
    ax.set_ylabel("Participant count", fontsize=11, labelpad=6)
    ax.set_title(
        "Distribution of RBD scores by risk group\n"
        "Incident PD (outcome 1a) | Percentile 3-tier stratification",
        fontsize=11, fontweight="bold", loc="left", pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, length=4)
    ax.grid(axis="y", alpha=0.2, linestyle="-", zorder=0)

    fig.tight_layout()

    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / filename_stem
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
        _save_ghost_copy(fig, out_dir / f"{filename_stem}_ghost")
        print(f"Saved: {stem}.{{pdf,png}} + ghost.{{png,svg}}")

    plt.close(fig)
