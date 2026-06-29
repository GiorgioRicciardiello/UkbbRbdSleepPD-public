"""
Forest plots for Models A–D (3-group percentile stratification).

One figure per model, primary outcome = outcome_1a_pd_only, method = percentile_3g.

    Model A  –  RBD-only Cox           (rbd_only_cox.xlsx)
    Model B  –  Prodromal-only Cox     (baseline_cox_HRs.xlsx)
    Model C  –  Additive Cox           (additive_cox.xlsx)
    Model D  –  Interaction Cox        (interaction_cox.xlsx)

Output
------
    docs/publication/figures/
        Forest_A_RBD_only.pdf / .png
        Forest_B_Prodromal_only.pdf / .png
        Forest_C_Additive.pdf / .png
        Forest_D_Interaction.pdf / .png
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(
    r"C:\Users\riccig01\OneDrive\Projects\MtSinai\During\UkbbRbdSleepPD"
    r"\results\cox_prodromal_abk_03_27_2026_13_06_48"
)
FIG_DIR = Path(
    r"C:\Users\riccig01\OneDrive\Projects\MtSinai\During\UkbbRbdSleepPD"
    r"\docs\publication\figures"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_OUTCOME = "outcome_1a_pd_only"
METHOD = "percentile_3g"

# ── Style ──────────────────────────────────────────────────────────────────────

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from config.config import RBD_RISK_COLORS as _RBD_COLORS  # noqa: E402

C_HIGH   = _RBD_COLORS["High"]   # RBD High / interaction High
C_MID    = _RBD_COLORS["Mid"]    # RBD Intermediate / interaction Intermediate
C_LOW    = _RBD_COLORS["Low"]    # RBD Low
C_PROD   = "#2166AC"              # prodromal marker (not an RBD risk group)
C_REF    = "#666666"              # null line

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
})


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _p_str(p: float) -> str:
    """Format p-value for annotation."""
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as PDF and PNG."""
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"  Saved: {path.name}")


def _draw_forest(
    ax: plt.Axes,
    rows: List[dict],
    *,
    xmin: float = 0.2,
    xmax: float = 12.0,
    ref_x: float = 1.0,
    title: str = "",
    xlabel: str = "Hazard Ratio (95% CI, log scale)",
    annot_hr: bool = True,
) -> None:
    """
    Draw a horizontal forest plot on *ax*.

    Each element of *rows* is a dict with keys:
        label   : str   – y-axis label
        hr      : float
        lo      : float – CI lower
        hi      : float – CI upper
        p       : float – p-value (optional, shown as annotation)
        color   : str   – dot/errorbar color
        is_sep  : bool  – if True, draw a faint separator line instead
        bold    : bool  – bold y-label
    """
    n = len(rows)
    y_positions = list(range(n - 1, -1, -1))  # top→bottom

    for i, (row, yp) in enumerate(zip(rows, y_positions)):
        if row.get("is_sep"):
            ax.axhline(yp, color="#cccccc", linewidth=0.4, zorder=0)
            ax.text(
                xmin * 1.05, yp,
                row["label"],
                va="center", ha="left",
                fontsize=7.5, fontweight="bold",
                color="#333333",
            )
            continue

        hr, lo, hi = row["hr"], row["lo"], row["hi"]
        color = row.get("color", C_HIGH)
        p_val = row.get("p", None)

        ax.errorbar(
            hr, yp,
            xerr=[[hr - lo], [hi - hr]],
            fmt="o",
            color=color,
            markersize=5,
            linewidth=1.0,
            capsize=2.5,
            capthick=0.8,
            zorder=3,
        )

        if annot_hr:
            ci_str = f"{hr:.2f} ({lo:.2f}–{hi:.2f})"
            if p_val is not None:
                ci_str += f"  p={_p_str(p_val)}"
            ax.text(
                xmax * 0.98, yp,
                ci_str,
                va="center", ha="right",
                fontsize=6.5, color="#333333",
            )

    # Y labels
    labels = []
    for row in rows:
        if row.get("is_sep"):
            labels.append("")
        else:
            fw = "bold" if row.get("bold") else "normal"
            labels.append(row["label"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("bold" if row.get("bold") else "normal")

    # Axes
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.8, n - 0.2)
    ax.axvline(ref_x, color=C_REF, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linewidth=0.3, color="#dddddd", zorder=0)


# ── Model A: RBD-only ──────────────────────────────────────────────────────────

def forest_model_a() -> None:
    """HR for RBD High and Intermediate vs Low (reference), PD outcome, 3g."""
    df = pd.read_excel(RESULTS_DIR / "rbd_only_cox.xlsx")
    df = df[
        (df["outcome"] == PRIMARY_OUTCOME) &
        (df["method"] == METHOD) &
        df["covariate"].str.startswith("rbd_")
    ].copy()

    # Group label cleanup
    df["group_label"] = df["covariate"].str.replace(r"rbd_", "", regex=True)

    rows = [
        {
            "label": "RBD High (99–100th pct)",
            "hr": row["HR"], "lo": row["HR_lower"], "hi": row["HR_upper"],
            "p": row["p"], "color": C_HIGH,
        }
        for _, row in df[df["group_label"].str.startswith("High")].iterrows()
    ] + [
        {
            "label": "RBD Intermediate (90–99th pct)",
            "hr": row["HR"], "lo": row["HR_lower"], "hi": row["HR_upper"],
            "p": row["p"], "color": C_MID,
        }
        for _, row in df[df["group_label"].str.startswith("Intermediate")].iterrows()
    ]
    rows = rows[::-1]  # Intermediate on top, High below — reversed for readability

    n_events = int(df["events"].iloc[0])
    n_total  = int(df["N"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.2, 1.4))
    _draw_forest(
        ax, rows,
        xmin=0.5, xmax=20.0,
        title=(
            f"Model A – RBD-only  "
            f"(N={n_total:,}, events={n_events})  "
            f"Reference: Low (0–90th pct)"
        ),
    )
    legend = [
        mpatches.Patch(color=C_HIGH, label="RBD High (99–100th pct)"),
        mpatches.Patch(color=C_MID,  label="RBD Intermediate (90–99th pct)"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=6.5)

    fig.tight_layout()
    _save(fig, "Forest_A_RBD_only")
    plt.close(fig)


# ── Model B: Prodromal-only ────────────────────────────────────────────────────

# Human-readable row labels for prodromal covariate codes
_PROD_COVARIATE_LABELS = {
    "prod_High":   "(High vs Low)",
    "prod_Medium": "(Medium vs Low)",
    "prod_Yes":    "(Yes vs No)",
    "prod_1":      "(Present vs Absent)",
}

# Desired display order for prodromal markers
_PROD_ORDER = [
    "Numeric Memory",
    "Reaction Time (ms)",
    "Fluid Intelligence",
    "FI Questions Attempted",
    "Trail Making",
    "Pairs Matching Status",
    "Constipation",
    "Orthostatic Hypotension",
    "Erectile Dysfunction",
    "Depression",
    "Anxiety",
]


def forest_model_b() -> None:
    """HR for each prodromal marker (prodromal-only model), PD outcome."""
    df = pd.read_excel(RESULTS_DIR / "baseline_cox_HRs.xlsx")
    df = df[
        (df["outcome"] == PRIMARY_OUTCOME) &
        df["covariate"].str.startswith("prod_")
    ].copy()

    # Build rows in desired order
    rows: list[dict] = []
    for marker in _PROD_ORDER:
        sub = df[df["prodromal_label"] == marker]
        if sub.empty:
            continue
        rows.append({"label": marker, "is_sep": True})
        for _, row in sub.iterrows():
            grp_suffix = _PROD_COVARIATE_LABELS.get(row["covariate"], row["covariate"])
            sig = row["p_fdr"] < 0.05 if not pd.isna(row["p_fdr"]) else False
            rows.append({
                "label": f"  {grp_suffix}",
                "hr": row["HR"], "lo": row["HR_lower"], "hi": row["HR_upper"],
                "p": row["p"], "color": C_PROD, "bold": sig,
            })

    height = max(3.5, len(rows) * 0.28)
    fig, ax = plt.subplots(figsize=(7.2, height))
    _draw_forest(
        ax, rows,
        xmin=0.1, xmax=15.0,
        title="Model B – Prodromal markers only  (Reference: Low / No / Absent)",
    )
    legend = [
        mpatches.Patch(color=C_PROD, label="Prodromal marker HR"),
        mpatches.Patch(color="white", label="Bold = FDR < 0.05", linewidth=0),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=6.5)
    fig.tight_layout()
    _save(fig, "Forest_B_Prodromal_only")
    plt.close(fig)


# ── Model C: Additive (RBD + prodromal) ───────────────────────────────────────

def forest_model_c() -> None:
    """
    For each prodromal marker, show three HRs:
        RBD High, RBD Intermediate, Prodromal effect.

    Grouped vertically by prodromal marker.
    """
    df = pd.read_excel(RESULTS_DIR / "additive_cox.xlsx")
    df = df[
        (df["outcome"] == PRIMARY_OUTCOME) &
        (df["method"] == METHOD)
    ].copy()

    rows: list[dict] = []
    for marker in _PROD_ORDER:
        sub = df[df["prodromal_label"] == marker]
        if sub.empty:
            continue

        rows.append({"label": marker, "is_sep": True})

        rbd_high = sub[sub["covariate"].str.startswith("rbd_High")]
        rbd_int  = sub[sub["covariate"].str.startswith("rbd_Intermediate")]
        prod_rows = sub[sub["covariate"].str.startswith("prod_")]

        for _, r in rbd_high.iterrows():
            rows.append({
                "label": "  RBD High (99–100th pct)",
                "hr": r["HR"], "lo": r["HR_lower"], "hi": r["HR_upper"],
                "p": r["p"], "color": C_HIGH,
            })
        for _, r in rbd_int.iterrows():
            rows.append({
                "label": "  RBD Intermediate (90–99th pct)",
                "hr": r["HR"], "lo": r["HR_lower"], "hi": r["HR_upper"],
                "p": r["p"], "color": C_MID,
            })
        for _, r in prod_rows.iterrows():
            grp = _PROD_COVARIATE_LABELS.get(r["covariate"], r["covariate"])
            rows.append({
                "label": f"  Prodromal {grp}",
                "hr": r["HR"], "lo": r["HR_lower"], "hi": r["HR_upper"],
                "p": r["p"], "color": C_PROD,
            })

    height = max(5.0, len(rows) * 0.27)
    fig, ax = plt.subplots(figsize=(7.2, height))
    _draw_forest(
        ax, rows,
        xmin=0.1, xmax=20.0,
        title="Model C – Additive (RBD + prodromal)  |  RBD reference: Low (0–90th pct)",
    )
    legend = [
        mpatches.Patch(color=C_HIGH, label="RBD High"),
        mpatches.Patch(color=C_MID,  label="RBD Intermediate"),
        mpatches.Patch(color=C_PROD, label="Prodromal marker"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=6.5)
    fig.tight_layout()
    _save(fig, "Forest_C_Additive")
    plt.close(fig)


# ── Model D: Interaction terms ─────────────────────────────────────────────────

def forest_model_d() -> None:
    """
    Interaction HR for each prodromal marker × RBD group term.

    Degenerate terms (HR == 0 or HR > 100) are excluded.
    Grouped by prodromal marker.
    """
    df = pd.read_excel(RESULTS_DIR / "interaction_cox.xlsx")
    df = df[
        (df["outcome"] == PRIMARY_OUTCOME) &
        (df["method"] == METHOD) &
        df["covariate"].str.contains("__x__")
    ].copy()

    # Exclude degenerate numerical fits
    df = df[(df["HR"] > 0.01) & (df["HR"] < 100)].copy()

    # Parse RBD group from covariate name
    df["rbd_group"] = df["covariate"].str.extract(r"rbd_(.+?)__x__")
    df["rbd_group_clean"] = df["rbd_group"].str.replace(
        r"\((\d+),(\d+)%\)", r"(\1–\2th pct)", regex=True
    )

    rows: list[dict] = []
    for marker in _PROD_ORDER:
        sub = df[df["prodromal_label"] == marker]
        if sub.empty:
            continue

        rows.append({"label": marker, "is_sep": True})

        high_sub = sub[sub["rbd_group"].str.startswith("High")]
        int_sub  = sub[sub["rbd_group"].str.startswith("Intermediate")]

        for _, r in high_sub.iterrows():
            rows.append({
                "label": "  RBD High × Prodromal",
                "hr": r["HR"], "lo": r["HR_lower"], "hi": r["HR_upper"],
                "p": r["p"], "color": C_HIGH,
                "bold": r["p"] < 0.05,
            })
        for _, r in int_sub.iterrows():
            rows.append({
                "label": "  RBD Intermediate × Prodromal",
                "hr": r["HR"], "lo": r["HR_lower"], "hi": r["HR_upper"],
                "p": r["p"], "color": C_MID,
                "bold": r["p"] < 0.05,
            })

    height = max(5.0, len(rows) * 0.27)
    fig, ax = plt.subplots(figsize=(7.2, height))
    _draw_forest(
        ax, rows,
        xmin=0.1, xmax=20.0,
        title=(
            "Model D – Interaction (RBD × Prodromal)  "
            "|  Bold = nominal p < 0.05  "
            "|  Reference: Low RBD × Prodromal Low/No"
        ),
    )
    ax.axvline(1.0, color=C_REF, linewidth=0.8, linestyle="--", zorder=1)
    legend = [
        mpatches.Patch(color=C_HIGH, label="RBD High (99–100th pct) × Prodromal"),
        mpatches.Patch(color=C_MID,  label="RBD Intermediate (90–99th pct) × Prodromal"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=6.5)
    fig.tight_layout()
    _save(fig, "Forest_D_Interaction")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Model forest plots (3g, outcome_1a_pd_only)...\n")

    print("Model A – RBD-only")
    forest_model_a()

    print("\nModel B – Prodromal-only")
    forest_model_b()

    print("\nModel C – Additive")
    forest_model_c()

    print("\nModel D – Interaction")
    forest_model_d()

    print(f"\nDone. Figures saved to: {FIG_DIR}")
