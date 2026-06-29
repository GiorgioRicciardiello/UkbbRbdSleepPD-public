"""
This module provides functions for statistical analysis and visualization
of risk stratification data, including computation of risk ratio, confidence
intervals, p-values, and creation of forest plots for visualizing results.

Functions:
- compute_rr_ci: Compute the risk ratio (RR) and 95% confidence interval (CI)
  using a standard log method for a 2?2 contingency table.
- fisher_p: Perform Fisher's Exact Test to calculate p-values for contingency tables.
- compute_rr_table: Compute RR statistics per group without plotting.
- forest_cell_plot: Generate a forest plot for a single contingency table
  AND return RR statistics.
- forest_panels_per_outcome: Generate panel forest plots for multiple outcomes
  and methods, and return a DataFrame with all statistics.

Dependencies:
- pandas
- numpy
- matplotlib
- scipy
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

from matplotlib.axes import Axes
from pathlib import Path

from library.risk.risk_helpers import (
    compute_risk_groups,
    get_group_labels,
)


# ===============================================================
# RR + CI COMPUTATION
# ===============================================================

def compute_rr_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float, float]:
    """
    Compute RR and 95% CI using standard log method.

    2?2 table:
             cases   controls
    group      a         b
    ref        c         d

    Returns:
        RR, LCI, UCI, risk_group (r1), risk_ref (r0)
    """
    r1 = a / (a + b) if (a + b) > 0 else np.nan
    r0 = c / (c + d) if (c + d) > 0 else np.nan

    if np.isnan(r0) or r0 == 0:
        RR = np.nan
    else:
        RR = r1 / r0

    if a == 0 or c == 0:
        return RR, np.nan, np.nan, r1, r0

    SE = np.sqrt((1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + d)))
    LCI = np.exp(np.log(RR) - 1.96 * SE)
    UCI = np.exp(np.log(RR) + 1.96 * SE)

    return RR, LCI, UCI, r1, r0



# ===============================================================
# FISHER EXACT TEST
# ===============================================================

def fisher_p(a: int, b: int, c: int, d: int) -> float:
    """
    Returns Fisher exact p-value for the 2x2 table.
    """
    try:
        _, p = fisher_exact([[a, b], [c, d]])
        return p
    except Exception:
        return np.nan



# ===============================================================
# COMPUTE RR TABLE (NO PLOT)
# ===============================================================

def compute_rr_table(
    df: pd.DataFrame,
    outcome: str,
    thr_dict: Dict[str, float],
    prob_col: str,
    subset: str,
    method: str
) -> List[Dict[str, Any]]:
    """
    Compute RR statistics for all risk groups.

    Returns:
        A list of dictionaries (one per group).
    """

    df = df.copy()
    df["risk_group"] = compute_risk_groups(df[prob_col], thr_dict)

    groups = get_group_labels(len(thr_dict) + 1)

    counts = {
        g: {
            "cases": int(df[df["risk_group"] == g][outcome].sum()),
            "controls": int(df[df["risk_group"] == g]["control"].sum()),
            "n": df[df["risk_group"] == g].shape[0],
        }
        for g in groups
    }

    # Determine reference group
    ref_group = None
    for g in groups:
        if counts[g]["cases"] > 0:
            ref_group = g
            break

    # If no cases anywhere
    if ref_group is None:
        return [
            {
                "outcome": outcome,
                "method": method,
                "subset": subset,
                "group": g,
                "cases": counts[g]["cases"],
                "controls": counts[g]["controls"],
                "n": counts[g]["n"],
                "risk": np.nan,
                "RR": np.nan,
                "LCI": np.nan,
                "UCI": np.nan,
                "p": np.nan,
                "ref_group": None,
            }
            for g in groups
        ]

    ref_cases = counts[ref_group]["cases"]
    ref_ctrls = counts[ref_group]["controls"]

    results = []

    for g in groups:
        a = counts[g]["cases"]
        b = counts[g]["controls"]
        c = ref_cases
        d = ref_ctrls

        RR, LCI, UCI, r1, r0 = compute_rr_ci(a, b, c, d)
        p = fisher_p(a, b, c, d)

        results.append(
            {
                "outcome": outcome,
                "method": method,
                "subset": subset,
                "group": g,
                "cases": a,
                "controls": b,
                "n": counts[g]["n"],
                "risk": r1,
                "RR": RR,
                "LCI": LCI,
                "UCI": UCI,
                "p": p,
                "ref_group": ref_group,
            }
        )

    return results



# ===============================================================
# FOREST CELL PLOT (PLOT + RETURN TABLE)
# ===============================================================

def forest_cell_plot(
    ax: Axes,
    df: pd.DataFrame,
    outcome: str,
    thr_dict: Dict[str, float],
    prob_col: str = "rbd_prob"
) -> List[Dict[str, Any]]:
    """
    Draw a forest plot inside `ax` AND return group-level statistics.

    Returns:
        A list of dictionaries with RR, CI, counts, p-values.
    """

    df = df.copy()
    df["risk_group"] = compute_risk_groups(df[prob_col], thr_dict)

    groups = get_group_labels(len(thr_dict) + 1)

    # Count
    counts = {}
    for g in groups:
        sub = df[df["risk_group"] == g]
        counts[g] = {
            "cases": int(sub[outcome].sum()),
            "controls": int(sub["control"].sum()),
            "n": len(sub),
        }

    # Determine reference
    ref = None
    for g in groups:
        if counts[g]["cases"] > 0:
            ref = g
            break

    # No valid groups
    if ref is None:
        ax.text(0.5, 0.5, "No cases -> RR undefined", ha="center")
        ax.axis("off")
        return [
            {
                "group": g,
                "ref_group": None,
                "cases": counts[g]["cases"],
                "controls": counts[g]["controls"],
                "n": counts[g]["n"],
                "risk": np.nan,
                "RR": np.nan,
                "LCI": np.nan,
                "UCI": np.nan,
                "p": np.nan,
            }
            for g in groups
        ]

    ref_cases = counts[ref]["cases"]
    ref_ctrls = counts[ref]["controls"]

    # Compute RR
    results = []
    for g in groups:
        a = counts[g]["cases"]
        b = counts[g]["controls"]
        c = ref_cases
        d = ref_ctrls

        RR, LCI, UCI, r1, r0 = compute_rr_ci(a, b, c, d)
        p = fisher_p(a, b, c, d)

        results.append(
            {
                "group": g,
                "ref_group": ref,
                "cases": a,
                "controls": b,
                "n": counts[g]["n"],
                "risk": r1,
                "RR": RR,
                "LCI": LCI,
                "UCI": UCI,
                "p": p,
            }
        )

    # --------- PLOT ---------
    y = np.arange(len(groups))
    ax.axvline(1, color="gray", linestyle="--")

    # CI lines
    for i, r in enumerate(results):
        if not (np.isnan(r["LCI"]) or np.isnan(r["UCI"])):
            ax.plot([r["LCI"], r["UCI"]], [i, i], color="black", linewidth=1)

    # RR points
    ax.scatter([r["RR"] for r in results], y, s=30, color="black", zorder=3)

    # Labels
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{r['group']}\nRR={r['RR']:.2f} (CI {r['LCI']:.2f}-{r['UCI']:.2f})"
            f"\ncases={r['cases']}, ctrl={r['controls']}\n"
            f"risk={r['risk']:.4f}, p={r['p']:.3g}"
            for r in results
        ],
        fontsize=7,
    )

    ax.set_xlabel("Risk Ratio (RR) with 95% CI", fontsize=8)
    ax.grid(alpha=0.3)

    return results



# ===============================================================
# MASTER FUNCTION ? PANELS + RETURN FULL TABLE
# ===============================================================

def forest_panels_per_outcome(
    df: pd.DataFrame,
    outcomes: List[str],
    thresholds: Dict[str, Dict[str, Any]],
    prob_col: str = "rbd_prob",
    figsize: tuple[int, int] = (12, 3),
    verbose: bool = True,
    case_type: str = "incident",
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate all forest panels and return a DataFrame containing all RR statistics.

    Behavior:
        - verbose=True  -> show figures interactively.
        - verbose=False -> do NOT show figures, only save them (if save_path is provided).

    Returns:
        pd.DataFrame with columns:
            outcome, method, subset, group, RR, LCI, UCI, cases, controls, n, risk, p
    """

    all_rows: List[Dict[str, Any]] = []
    methods = list(thresholds.keys())

    for outcome in outcomes:
        fig, axes = plt.subplots(
            nrows=len(methods),
            ncols=2,
            figsize=(figsize[0], figsize[1] * len(methods))
        )
        axes = np.atleast_2d(axes)

        fig.suptitle(f"Forest Plots for {outcome}", fontsize=14)

        for i, method in enumerate(methods):
            if outcome not in thresholds[method]:
                axes[i, 0].axis("off")
                axes[i, 1].axis("off")
                continue

            thr_dict = thresholds[method][outcome]["all"]

            # -----------------------
            # VALIDATION SUBSET
            # -----------------------
            df_val = df[df["val"] == True]
            res_val = forest_cell_plot(
                ax=axes[i, 0],
                df=df_val,
                outcome=outcome + f'_{case_type}',
                thr_dict=thr_dict,
                prob_col=prob_col
            )

            for r in res_val:
                all_rows.append({
                    **r,
                    "outcome": outcome + f'_{case_type}',
                    "method": method,
                    "subset": "validation"
                })

            # -----------------------
            # NON-VALIDATION SUBSET
            # -----------------------
            df_non = df[df["val"] == False]
            res_non = forest_cell_plot(
                ax=axes[i, 1],
                df=df_non,
                outcome=outcome + f'_{case_type}',
                thr_dict=thr_dict,
                prob_col=prob_col
            )

            for r in res_non:
                all_rows.append({
                    **r,
                    "outcome": outcome + f'_{case_type}',
                    "method": method,
                    "subset": "nonvalidation"
                })

        fig.tight_layout()

        # -----------------------
        # SAVE IF REQUESTED
        # -----------------------
        if save_path is not None:
            out_file = save_path / f"forest_panel_{outcome}.png"
            fig.savefig(out_file, dpi=300)

        # -----------------------
        # SHOW OR CLOSE
        # -----------------------
        if verbose:
            plt.show()
        else:
            plt.close(fig)

    return pd.DataFrame(all_rows)
