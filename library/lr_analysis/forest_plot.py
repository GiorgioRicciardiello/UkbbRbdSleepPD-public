"""
Forest plot of all empirical LR estimates and consolidated master table.

Sources
-------
risk_group_lr.csv           -> stratum-specific LR for 2g/3g RBD risk groups
lr_at_youden.csv            -> binary LR+/LR- at Youden threshold (overall, female, male)
lr_profile.csv              -> binary LR+/LR- across z-score grid (overall only)
empirical_prodromal_lrs.csv -> binary LR+/LR- for prodromal markers
logistic_or.csv             -> OR per 1 SD of z-score (unadjusted + adjusted for age/sex/BMI)

Outputs
-------
results/lr_analysis/master_lr_table.csv
results/lr_analysis/figures/forest_plot.png
Excel workbook: new sheet "Master_LR_Table"

Note on mixed metrics
---------------------
LR+ rows (binary threshold) and OR rows (continuous z-score, logistic regression)
are both ratio measures centred at 1 on a log scale, but are not directly comparable.
OR rows are plotted in the LR+ panel with a distinct marker and clearly labelled.
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
_FONT = 9

plt.rcParams.update({
    "font.size": _FONT,
    "axes.labelsize": _FONT,
    "axes.titlesize": _FONT + 1,
    "xtick.labelsize": _FONT - 1,
    "ytick.labelsize": _FONT - 1,
    "legend.fontsize": _FONT - 1,
    "figure.dpi": _FIG_DPI,
})

# Thresholds from the profile grid to include in the forest plot
_PROFILE_PLOT_THRESHOLDS = {1.0, 1.5, 2.0, 2.5}

# Group display order and colors
_GROUP_ORDER = [
    "RBD Risk Group (3g)",
    "RBD Risk Group (2g)",
    "Actigraphy z-score — Logistic OR (continuous)",
    "Actigraphy z-score — Youden threshold",
    "Actigraphy z-score — threshold profile",
    "Prodromal marker (empirical UKBB)",
]

_GROUP_COLOR = {
    "RBD Risk Group (3g)":                          "#1a5276",
    "RBD Risk Group (2g)":                          "#2e86c1",
    "Actigraphy z-score — Logistic OR (continuous)": "#c0392b",
    "Actigraphy z-score — Youden threshold":         "#884ea0",
    "Actigraphy z-score — threshold profile":        "#6e2fa1",
    "Prodromal marker (empirical UKBB)":             "#1e8449",
}

# OR rows use a distinct square marker so they're visually separable from LR rows
_OR_GROUP = "Actigraphy z-score — Logistic OR (continuous)"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "youden":      pd.read_csv(results_dir / "lr_at_youden.csv"),
        "profile":     pd.read_csv(results_dir / "lr_profile.csv"),
        "empirical":   pd.read_csv(results_dir / "empirical_prodromal_lrs.csv"),
        "risk_groups": pd.read_csv(results_dir / "risk_group_lr.csv"),
        "logistic_or": pd.read_csv(results_dir / "logistic_or.csv"),
    }


# ── Master table assembly ─────────────────────────────────────────────────────

def build_master_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Consolidate all LR estimates into a single table.

    Columns
    -------
    group, label, variable, stratum, n_total, n_cases, n_controls,
    lr_pos, lr_pos_lci, lr_pos_uci, lr_neg, lr_neg_lci, lr_neg_uci,
    stable, lr_type, note
    """
    rows: list[dict] = []

    # 1. Risk groups — stratum-specific LR (no LR- equivalent)
    for _, r in data["risk_groups"].iterrows():
        scheme = r["scheme"]
        cat = r["category"]
        pctile = ">= p99" if cat == "High" and scheme == "3g" else (
                 ">= p90" if cat in ("High", "Intermediate") else "< p90")
        rows.append({
            "group":        f"RBD Risk Group ({scheme})",
            "label":        f"{cat} ({scheme})",
            "variable":     "abk_rbd_score_mean",
            "stratum":      cat,
            "threshold":    pctile,
            "n_total":      int(r["N"]),
            "n_cases":      int(r["n_cases"]),
            "n_controls":   int(r["n_controls"]),
            "lr_pos":       r["LR"],
            "lr_pos_lci":   r["LR_lci"],
            "lr_pos_uci":   r["LR_uci"],
            "lr_neg":       None,
            "lr_neg_lci":   None,
            "lr_neg_uci":   None,
            "stable":       r["stable"],
            "lr_type":      "stratum-specific",
            "note":         "Same thresholds as Cox survival analysis",
        })

    # 2. Youden threshold (overall + sex-stratified)
    for _, r in data["youden"].iterrows():
        rows.append({
            "group":       "Actigraphy z-score — Youden threshold",
            "label":       f"z >= {r['threshold']:.2f} — {r['stratum']}",
            "variable":    "abk_rbd_score_mean (z-score)",
            "stratum":     r["stratum"],
            "threshold":   f"z = {r['threshold']:.3f}",
            "n_total":     int(r["n_cases"] + r["n_controls"]),
            "n_cases":     int(r["n_cases"]),
            "n_controls":  int(r["n_controls"]),
            "lr_pos":      r["lr_pos"],
            "lr_pos_lci":  r["lr_pos_lci"],
            "lr_pos_uci":  r["lr_pos_uci"],
            "lr_neg":      r["lr_neg"],
            "lr_neg_lci":  r["lr_neg_lci"],
            "lr_neg_uci":  r["lr_neg_uci"],
            "stable":      r["stable"],
            "lr_type":     "binary",
            "note":        "Youden-optimal threshold",
        })

    # 3. Logistic OR (unadjusted + adjusted for age/sex/BMI)
    _adjustment_labels = {
        "unadjusted": "Unadjusted OR (per 1 SD z-score)",
        "adjusted":   "Adjusted OR (per 1 SD z-score; age, sex, BMI)",
    }
    for _, r in data["logistic_or"].iterrows():
        model = r["model_type"]
        rows.append({
            "group":       _OR_GROUP,
            "label":       _adjustment_labels.get(model, model),
            "variable":    "abk_rbd_score_mean (z-score, continuous)",
            "stratum":     "overall",
            "threshold":   "continuous (per 1 SD)",
            "n_total":     int(r["n"]),
            "n_cases":     int(r["n_cases"]),
            "n_controls":  int(r["n"]) - int(r["n_cases"]),
            "lr_pos":      r["or_estimate"],
            "lr_pos_lci":  r["or_lci"],
            "lr_pos_uci":  r["or_uci"],
            "lr_neg":      None,
            "lr_neg_lci":  None,
            "lr_neg_uci":  None,
            "stable":      bool(r["converged"]),
            "lr_type":     "logistic OR",
            "note":        f"p < 0.001; converged={r['converged']}",
        })

    # 4. Threshold profile (overall, all thresholds — full table; plot uses subset)
    profile_overall = data["profile"][data["profile"]["stratum"] == "overall"]
    for _, r in profile_overall.iterrows():
        thr = float(r["threshold"])
        rows.append({
            "group":       "Actigraphy z-score — threshold profile",
            "label":       f"z >= {thr:.1f} — overall",
            "variable":    "abk_rbd_score_mean (z-score)",
            "stratum":     "overall",
            "threshold":   f"z = {thr:.1f}",
            "n_total":     int(r["n_cases"] + r["n_controls"]),
            "n_cases":     int(r["n_cases"]),
            "n_controls":  int(r["n_controls"]),
            "lr_pos":      r["lr_pos"],
            "lr_pos_lci":  r["lr_pos_lci"],
            "lr_pos_uci":  r["lr_pos_uci"],
            "lr_neg":      r["lr_neg"],
            "lr_neg_lci":  r["lr_neg_lci"],
            "lr_neg_uci":  r["lr_neg_uci"],
            "stable":      r["stable"],
            "lr_type":     "binary",
            "note":        "",
        })

    # 4. Prodromal markers (empirical UKBB)
    for _, r in data["empirical"].iterrows():
        male_only = "erectile" in r["col"]
        rows.append({
            "group":       "Prodromal marker (empirical UKBB)",
            "label":       r["label"] + (" (males only)" if male_only else ""),
            "variable":    r["col"],
            "stratum":     "male" if male_only else "overall",
            "threshold":   "positive vs negative",
            "n_total":     int(r["tp"] + r["fp"] + r["fn"] + r["tn"]),
            "n_cases":     int(r["tp"] + r["fn"]),
            "n_controls":  int(r["fp"] + r["tn"]),
            "lr_pos":      r["lr_pos"],
            "lr_pos_lci":  r["lr_pos_lci"],
            "lr_pos_uci":  r["lr_pos_uci"],
            "lr_neg":      r["lr_neg"],
            "lr_neg_lci":  r["lr_neg_lci"],
            "lr_neg_uci":  r["lr_neg_uci"],
            "stable":      r["stable"],
            "lr_type":     "binary",
            "note":        "Males only" if male_only else "",
        })

    return pd.DataFrame(rows)


# ── Forest plot ───────────────────────────────────────────────────────────────

def _select_plot_rows(master: pd.DataFrame) -> pd.DataFrame:
    """Subset master table to forest plot rows."""
    parts = []

    # Risk groups — both schemes
    parts.append(master[master["group"].str.startswith("RBD Risk Group")])

    # Logistic OR — both models
    parts.append(master[master["group"] == _OR_GROUP])

    # Youden — all strata
    parts.append(master[master["group"] == "Actigraphy z-score — Youden threshold"])

    # Profile — selected thresholds
    profile = master[master["group"] == "Actigraphy z-score — threshold profile"].copy()
    profile["_thr"] = profile["threshold"].str.extract(r"z = (.+)").astype(float)
    parts.append(profile[profile["_thr"].isin(_PROFILE_PLOT_THRESHOLDS)].drop(columns="_thr"))

    # Prodromal markers
    parts.append(master[master["group"] == "Prodromal marker (empirical UKBB)"])

    return pd.concat(parts, ignore_index=True)


def _build_plot_rows(
    forest_df: pd.DataFrame,
) -> tuple[list[dict], list[str]]:
    """Sort and group forest rows, return (rows_with_meta, group_header_indices)."""
    group_rank = {g: i for i, g in enumerate(_GROUP_ORDER)}
    df = forest_df.copy()
    df["_grank"] = df["group"].map(group_rank).fillna(99)
    df = df.sort_values(["_grank", "lr_pos"], ascending=[True, False])

    rows_out = []
    section_breaks: dict[int, str] = {}  # row_index -> group label
    prev_group = None
    for _, row in df.iterrows():
        g = row["group"]
        if g != prev_group:
            section_breaks[len(rows_out)] = g
            prev_group = g
        rows_out.append(row.to_dict())

    return rows_out, section_breaks


def plot_forest(master: pd.DataFrame, out_path: Path) -> None:
    """Render two-panel horizontal forest plot (LR+ | LR-)."""
    forest_df = _select_plot_rows(master)
    plot_rows, section_breaks = _build_plot_rows(forest_df)

    n = len(plot_rows)
    # y-positions: leave 0.7-unit gap at each section break
    y_pos: list[float] = []
    current_y = 0.0
    for i in range(n):
        if i in section_breaks and i > 0:
            current_y += 0.7
        y_pos.append(current_y)
        current_y += 1.0

    fig_h = max(7, n * 0.42 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h),
                             gridspec_kw={"wspace": 0.55})

    panels = [
        (axes[0], "lr_pos", "lr_pos_lci", "lr_pos_uci",
         "LR+  /  Stratum-specific LR",
         [(1.0, "#555555", "--", "LR = 1  (no change)"),
          (2.8, "#c0672b", ":", "Questionnaire RBD benchmark  (LR+ = 2.8)")]),
        (axes[1], "lr_neg", "lr_neg_lci", "lr_neg_uci",
         "LR−",
         [(1.0, "#555555", "--", "LR = 1  (no change)")]),
    ]

    for ax, val_col, lci_col, uci_col, title, ref_lines in panels:
        # Blended transform: axes-x (fraction 0-1) × data-y — keeps annotation column fixed
        ann_transform = blended_transform_factory(ax.transAxes, ax.transData)

        for i, row in enumerate(plot_rows):
            y = y_pos[i]
            g = row["group"]
            c = _GROUP_COLOR.get(g, "#555555")
            stable = bool(row.get("stable", True))

            v = row.get(val_col)
            lo = row.get(lci_col)
            hi = row.get(uci_col)

            if v is None or (isinstance(v, float) and np.isnan(v)):
                ax.text(
                    1.0, y, "N/A",
                    va="center", ha="center",
                    fontsize=7, color="#aaaaaa", style="italic",
                )
                continue

            # CI whisker
            has_ci = (
                lo is not None and hi is not None
                and not np.isnan(float(lo)) and not np.isnan(float(hi))
            )
            if has_ci:
                ax.plot([lo, hi], [y, y], color=c, lw=1.6, alpha=0.75, zorder=3)
                ax.plot([lo, lo], [y - 0.15, y + 0.15], color=c, lw=1.0, zorder=3)
                ax.plot([hi, hi], [y - 0.15, y + 0.15], color=c, lw=1.0, zorder=3)

            # Point estimate — square for OR rows, diamond for unstable, circle otherwise
            is_or_row = row.get("lr_type") == "logistic OR"
            if is_or_row:
                marker, ms = "s", 7
            elif not stable:
                marker, ms = "D", 5
            else:
                marker, ms = "o", 6
            ax.plot(v, y, marker=marker, color=c, ms=ms,
                    mec="white", mew=0.8, zorder=5)

            # Annotation: effect size [95% CI] — placed just right of the panel frame
            if has_ci:
                ann = f"{v:.2f} [{lo:.2f}–{hi:.2f}]"
            else:
                ann = f"{v:.2f}"
            ax.text(
                1.02, y, ann,
                transform=ann_transform,
                va="center", ha="left",
                fontsize=6.0, color=c,
                clip_on=False,
            )

        # Reference lines
        for rv, rc, rls, rlbl in ref_lines:
            ax.axvline(rv, color=rc, ls=rls, lw=1.0, alpha=0.7,
                       label=rlbl, zorder=1)

        ax.set_xscale("log")
        ax.set_xlim(0.25, 18)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:g}"
        ))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [r["label"] for r in plot_rows],
            fontsize=7.5,
        )
        ax.set_ylim(max(y_pos) + 0.8, -0.8)  # top-to-bottom
        ax.set_xlabel(title, fontsize=_FONT)
        ax.set_title(title, fontsize=_FONT + 1, fontweight="bold", pad=8)
        ax.grid(True, axis="x", alpha=0.25, lw=0.5, zorder=0)
        if ref_lines:
            ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)

        # Section header annotations (left panel only)
        if ax is axes[0]:
            for idx, group_name in section_breaks.items():
                y_header = y_pos[idx] - 0.55
                color = _GROUP_COLOR.get(group_name, "#333333")
                ax.text(
                    0.01, y_header,
                    group_name,
                    transform=ax.get_yaxis_transform(),
                    fontsize=7, fontweight="bold",
                    color=color, va="bottom", ha="left",
                )

    # Legend: marker types
    lr_patch = plt.Line2D(
        [0], [0], marker="o", color="gray",
        linestyle="none", ms=6, mec="white", mew=0.8,
        label="LR (2x2 table, binary or stratum-specific)",
    )
    or_patch = plt.Line2D(
        [0], [0], marker="s", color="#c0392b",
        linestyle="none", ms=7, mec="white", mew=0.8,
        label="OR per 1 SD z-score (logistic regression)",
    )
    unstable_patch = plt.Line2D(
        [0], [0], marker="D", color="gray",
        linestyle="none", ms=5, label="Unstable (sparse cells)",
    )
    axes[1].legend(
        handles=[lr_patch, or_patch, unstable_patch],
        fontsize=6.5, loc="upper right", framealpha=0.85,
    )

    fig.suptitle(
        "Likelihood Ratios and Odds Ratios: Actigraphy RBD Score and Prodromal Markers\n"
        "(Incident Parkinson disease, UKBB cohort)\n"
        "OR rows (square): per 1 SD of RBD z-score; LR rows (circle): 2x2 table at specified threshold",
        fontsize=_FONT, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Forest plot: {out_path}")


# ── Output writer ─────────────────────────────────────────────────────────────

def save_master_table(master: pd.DataFrame, out_dir: Path) -> None:
    """Save master table to CSV and as a new Excel sheet."""
    csv_path = out_dir / "master_lr_table.csv"
    master.to_csv(csv_path, index=False)
    print(f"  Master table CSV: {csv_path}")

    excel_path = out_dir / "lr_analysis_tables.xlsx"
    if excel_path.exists():
        from openpyxl import load_workbook
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            master.to_excel(writer, sheet_name="Master_LR_Table", index=False)
        print(f"  Master table appended to {excel_path}")
    else:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            master.to_excel(writer, sheet_name="Master_LR_Table", index=False)
        print(f"  Master table Excel: {excel_path}")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_forest_plot() -> None:
    """Build master table and render forest plot from saved result CSVs."""
    out_dir = project_config["results"]["root"] / RESULTS_SUBDIR
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Forest plot + master LR table")
    print("=" * 60)

    data = _load_results(out_dir)
    master = build_master_table(data)
    print(f"  Master table: {len(master)} rows across {master['group'].nunique()} groups")

    save_master_table(master, out_dir)
    plot_forest(master, fig_dir / "forest_plot.png")

    print("\nSummary:")
    for g, sub in master.groupby("group"):
        print(f"  {g}: {len(sub)} rows")


if __name__ == "__main__":
    run_forest_plot()
