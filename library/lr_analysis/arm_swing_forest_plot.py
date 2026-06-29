"""
Forest plot for arm swing analysis: main effects + RBD interaction effects.

Visualizes:
- Main effect OR (overall effect across all RBD groups)
- Interaction effect OR for Mid RBD stratum
- Interaction effect OR for High RBD stratum
(Low RBD is reference, implicit OR = 1.0)

Organized by variable with CI whiskers and log-scale axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

from config.config import config as project_config
from library.lr_analysis.config import RESULTS_SUBDIR

_FIG_DPI = 300
_FONT = 10

plt.rcParams.update({
    "font.size": _FONT,
    "axes.labelsize": _FONT,
    "axes.titlesize": _FONT + 1,
    "xtick.labelsize": _FONT - 1,
    "ytick.labelsize": _FONT - 1,
    "legend.fontsize": _FONT - 1,
    "figure.dpi": _FIG_DPI,
})

# Effect type order and colors
_EFFECT_ORDER = ["Main Effect", "Mid RBD", "High RBD"]
_EFFECT_COLOR = {
    "Main Effect": "#1a5276",      # Dark blue
    "Mid RBD":     "#2e86c1",      # Medium blue
    "High RBD":    "#c0392b",      # Red
}

_REFERENCE_LINE_LABEL = "OR = 1  (no effect)"


def load_arm_swing_interaction(results_dir: Path) -> pd.DataFrame:
    """Load arm_swing_interaction.csv."""
    path = results_dir / "arm_swing" / "arm_swing_interaction.csv"
    if not path.exists():
        raise FileNotFoundError(f"Arm swing interaction results not found: {path}")
    return pd.read_csv(path)


def build_forest_data(interaction_df: pd.DataFrame) -> pd.DataFrame:
    """Transform interaction data into forest plot format.

    For each variable, create 3 rows:
    - Main effect (main_g_or / lci / uci)
    - Mid RBD interaction (interaction_or_Mid / lci_Mid / uci_Mid)
    - High RBD interaction (interaction_or_High / lci_High / uci_High)

    Returns DataFrame with columns:
    variable_label, effect_type, or_estimate, or_lci, or_uci, n_total
    """
    rows = []
    for _, r in interaction_df.iterrows():
        var_label = r["label"]

        # Main effect
        rows.append({
            "variable": var_label,
            "effect_type": "Main Effect",
            "or_estimate": float(r["main_g_or"]),
            "or_lci": float(r["main_g_lci"]),
            "or_uci": float(r["main_g_uci"]),
            "n_total": int(r["n_total"]),
        })

        # Mid RBD interaction
        rows.append({
            "variable": var_label,
            "effect_type": "Mid RBD",
            "or_estimate": float(r["interaction_or_Mid"]),
            "or_lci": float(r["interaction_lci_Mid"]),
            "or_uci": float(r["interaction_uci_Mid"]),
            "n_total": int(r["n_total"]),
        })

        # High RBD interaction
        rows.append({
            "variable": var_label,
            "effect_type": "High RBD",
            "or_estimate": float(r["interaction_or_High"]),
            "or_lci": float(r["interaction_lci_High"]),
            "or_uci": float(r["interaction_uci_High"]),
            "n_total": int(r["n_total"]),
        })

    return pd.DataFrame(rows)


def plot_arm_swing_forest(forest_data: pd.DataFrame, out_path: Path) -> None:
    """Render horizontal forest plot for arm swing interaction effects."""

    # Build y-axis positions: group by variable, with gaps between variables
    y_positions = {}
    current_y = 0.0
    prev_var = None

    for _, row in forest_data.iterrows():
        var = row["variable"]
        if var != prev_var:
            current_y += 0.8  # Gap before new variable
            prev_var = var

        key = (row["variable"], row["effect_type"])
        y_positions[key] = current_y
        current_y += 1.0

    fig_h = max(9, len(forest_data) * 0.35 + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # Reference line at OR = 1
    ax.axvline(1.0, color="#555555", linestyle="--", lw=1.2, alpha=0.7,
               label=_REFERENCE_LINE_LABEL, zorder=1)

    # Blended transform for annotations: data coords on x-axis, data coords on y-axis
    ann_transform = blended_transform_factory(ax.transData, ax.transData)

    # Plot each row
    for _, row in forest_data.iterrows():
        y = y_positions[(row["variable"], row["effect_type"])]
        effect = row["effect_type"]
        color = _EFFECT_COLOR.get(effect, "#555555")

        or_est = row["or_estimate"]
        or_lci = row["or_lci"]
        or_uci = row["or_uci"]

        # CI whisker
        ax.plot([or_lci, or_uci], [y, y], color=color, lw=1.8, alpha=0.75, zorder=3)
        ax.plot([or_lci, or_lci], [y - 0.15, y + 0.15], color=color, lw=1.2, zorder=3)
        ax.plot([or_uci, or_uci], [y - 0.15, y + 0.15], color=color, lw=1.2, zorder=3)

        # Point estimate (circle)
        ax.plot(or_est, y, marker="o", color=color, ms=7,
                mec="white", mew=1.0, zorder=5)

        # Annotation: OR [95% CI] — positioned right of plot frame
        ann = f"{or_est:.3f} [{or_lci:.3f}–{or_uci:.3f}]"
        ax.text(
            3.6, y, ann,
            transform=ann_transform,
            va="center", ha="left",
            fontsize=7.5, color=color, clip_on=False,
            family="monospace"
        )

    # Y-axis: variable name + effect type
    y_ticks = []
    y_labels = []
    for key, y in sorted(y_positions.items(), key=lambda x: x[1]):
        var_label, effect_type = key
        y_ticks.append(y)
        # Indent effect types under variable names
        label = f"  {effect_type}" if effect_type != "Main Effect" else var_label
        y_labels.append(label)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8.5)

    # X-axis (log scale)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_xlabel("Odds Ratio (per 1 SD z-score)", fontsize=_FONT, fontweight="bold")

    # Y-limits and inversion (top-to-bottom)
    y_max = max(y_positions.values())
    ax.set_ylim(y_max + 0.5, -0.5)

    # Grid
    ax.grid(True, axis="x", alpha=0.25, lw=0.5, zorder=0)

    # Legend
    main_patch = mpatches.Patch(color=_EFFECT_COLOR["Main Effect"],
                                label="Main effect (overall)")
    mid_patch = mpatches.Patch(color=_EFFECT_COLOR["Mid RBD"],
                               label="Interaction: Mid RBD stratum")
    high_patch = mpatches.Patch(color=_EFFECT_COLOR["High RBD"],
                                label="Interaction: High RBD stratum")
    ax.legend(handles=[main_patch, mid_patch, high_patch],
              loc="lower right", fontsize=8.5, framealpha=0.9)

    # Reference line legend
    ref_line = plt.Line2D([0], [0], color="#555555", linestyle="--", lw=1.2,
                          label=_REFERENCE_LINE_LABEL)
    ax.get_legend().get_texts()[0].set_visible(False)
    ax.plot([], [], color="#555555", linestyle="--", lw=1.2,
            label=_REFERENCE_LINE_LABEL)

    # Title
    fig.suptitle(
        "Arm Swing Analysis: Main Effects and RBD Interaction Effects\n"
        "(Incident Parkinson Disease, UKBB Cohort)",
        fontsize=_FONT + 1, fontweight="bold", y=0.98
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Arm swing forest plot: {out_path}")


def run_arm_swing_forest_plot() -> None:
    """Execute arm swing forest plot generation."""
    results_dir = project_config["results"]["root"] / RESULTS_SUBDIR
    fig_dir = results_dir / "arm_swing" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Arm Swing Forest Plot: Main Effects + RBD Interactions")
    print("=" * 70)

    # Load and transform
    interaction_df = load_arm_swing_interaction(results_dir)
    forest_data = build_forest_data(interaction_df)

    print(f"  Loaded {len(interaction_df)} arm swing variables")
    print(f"  Forest data: {len(forest_data)} rows ({len(interaction_df)} variables x 3 effects)")

    # Render
    out_path = fig_dir / "arm_swing_forest_plot.png"
    plot_arm_swing_forest(forest_data, out_path)

    # Summary
    print("\nEffect counts by type:")
    for effect in _EFFECT_ORDER:
        count = len(forest_data[forest_data["effect_type"] == effect])
        print(f"  {effect}: {count} rows")

    print("\n  Forest plot complete")


if __name__ == "__main__":
    run_arm_swing_forest_plot()
