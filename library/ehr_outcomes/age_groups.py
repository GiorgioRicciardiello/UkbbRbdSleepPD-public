import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from textwrap import fill
from pathlib import Path
from tabulate import tabulate
from config.config import outcomes
from library.column_registry import col_prevalent, col_incident, col_dx

def create_age_groups(
    df: pd.DataFrame,
save_dir:Path = None,
    age_col: str = "age_recruitment"
) -> pd.DataFrame:
    """
    Adds age group variables with:
      1) Three-level age groups: <50, 50?60, >60
      2) Collapsed binary age group: ?60 vs >60

    3-category:
        - age_group_3
        - age_group_3_cat
        - age_group_3_suffix

    2-category (collapsed):
        - age_group_2
        - age_group_2_cat
        - age_group_2_suffix

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new columns:
            - age_group
            - age_group_cat
            - age_group_cat_suffix
            - age_group_2cat
            - age_group_2cat_cat
    """


    df = df.copy()
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    # ==================================================
    # 3-level age groups: <50, 50?60, >60
    # ==================================================
    bins_3 = [-np.inf, 50, 60, np.inf]
    labels_3 = ["<50", "50-60", ">60"]
    suffix_map_3 = {
        "<50": "low_50",
        "50-60": "btw_50_60",
        ">60": "ge_60",
    }

    df["age_group_3"] = pd.cut(
        df[age_col],
        bins=bins_3,
        labels=labels_3,
        include_lowest=True,
        right=False,  # 50 -> 50?60
    )

    df["age_group_3_cat"] = pd.Categorical(
        df["age_group_3"],
        categories=labels_3,
        ordered=True,
    )

    df["age_group_3_suffix"] = df["age_group_3"].map(suffix_map_3)

    # ==================================================
    # 2-level collapsed age groups: ?60 vs >60
    # ==================================================
    labels_2 = ["le_60", "gt_60"]
    suffix_map_2 = {
        "le_60": "le_60",
        "gt_60": "gt_60",
    }

    df["age_group_2"] = np.where(
        df["age_group_3"].isin(["<50", "50-60"]),
        "le_60",
        "gt_60",
    )

    df["age_group_2_cat"] = pd.Categorical(
        df["age_group_2"],
        categories=labels_2,
        ordered=True,
    )

    df["age_group_2_suffix"] = df["age_group_2"].map(suffix_map_2)

    # all ages groups
    df["age_group_none"] = "All ages"
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

        _ = _plot_age_outcome_heatmap(df,
                                     age_group_col="age_group_2_cat",
                                     figsize=(12, 5),
                                     out_path=save_dir,
                                     outcomes=outcomes)

        _ = _plot_age_outcome_heatmap(df,
                                     age_group_col="age_group_3_cat",
                                     figsize=(12, 5),
                                     out_path=save_dir,
                                     outcomes=outcomes)

    return df


def _plot_age_outcome_heatmap(df: pd.DataFrame,
                              age_group_col: str = "age_group_cat",
                              outcomes: list = None,
                              wrap_width: int = 12,
                              figsize=(16, 12),
                              out_path: Path = None,
                              cmap="Blues"):
    """
    Produces:
      1) A long summary table:
            age_group ? outcome ? {diagnosed, prevalent, incident} counts
      2) A wide matrix summary for tabulate printing
      3) A 3-row subplot:
            Row 1 -> diagnosed heatmap
            Row 2 -> prevalent heatmap
            Row 3 -> incident heatmap

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain columns:
            <outcome>_diagnosed
            <outcome>_prevalent
            <outcome>_incident
    age_group_col : str
        Column identifying the age groups.
    outcomes : list
        Outcome names as in config.outcomes, e.g. ['outcome_1a_pd_only', ...]
    cmap : str
        Colormap for heatmaps.
    wrap_width : int
        Width (chars) for wrapping long outcome names.
    """

    if outcomes is None:
        raise ValueError("Must pass the outcome list.")

        # ----------------------------------------------------------------------
        # Preset row order ? pulled from the categorical definition
        # ----------------------------------------------------------------------
    age_order = df[age_group_col].cat.categories.tolist()  # ['<50','50-60','>60']

    # ----------------------------------------------------------------------
    # BUILD LONG SUMMARY TABLE
    # ----------------------------------------------------------------------
    long_rows = []
    for age_grp in age_order:
        sub = df[df[age_group_col] == age_grp]

        for outcome in outcomes:
            outcome_l = outcome.lower()

            long_rows.append({
                "age_group": age_grp,
                "outcome": outcome,
                "case_type": "diagnosed",
                "count": int(sub[col_dx(outcome_l)].sum())
            })
            long_rows.append({
                "age_group": age_grp,
                "outcome": outcome,
                "case_type": "prevalent",
                "count": int(sub[col_prevalent(outcome_l)].sum())
            })
            long_rows.append({
                "age_group": age_grp,
                "outcome": outcome,
                "case_type": "incident",
                "count": int(sub[col_incident(outcome_l)].sum())
            })

    summary_long = pd.DataFrame(long_rows)

    # ----------------------------------------------------------------------
    # WIDE MATRICES (WRAPPED COLS + FIXED ROW ORDER)
    # ----------------------------------------------------------------------
    # wrap = lambda s: fill(s, wrap_width)

    def make_matrix(case_type):
        mat = (summary_long[summary_long["case_type"] == case_type]
               .pivot(index="age_group", columns="outcome", values="count")
               # .rename(columns=wrap)
               )
        return mat.reindex(age_order)  # enforce row order

    diagnosed_mat = make_matrix("diagnosed")
    prevalent_mat = make_matrix("prevalent")
    incident_mat = make_matrix("incident")

    # ----------------------------------------------------------------------
    # PRINT TABLES WITH TABULATE
    # ----------------------------------------------------------------------
    print("\n### DIAGNOSED COUNTS ###\n")
    print(tabulate(diagnosed_mat, headers="keys", tablefmt="github"))

    print("\n### PREVALENT COUNTS ###\n")
    print(tabulate(prevalent_mat, headers="keys", tablefmt="github"))

    print("\n### INCIDENT COUNTS ###\n")
    print(tabulate(incident_mat, headers="keys", tablefmt="github"))

    # ----------------------------------------------------------------------
    # HEATMAPS (3 ROWS, FIXED ORDER)
    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    mats = [diagnosed_mat, prevalent_mat, incident_mat]
    titles = ["Diagnosed Cases", "Prevalent Cases", "Incident Cases"]

    for i, (ax, mat, title) in enumerate(zip(axes, mats, titles)):
        sns.heatmap(mat,
                    annot=True,
                    fmt="d",
                    cmap=cmap,
                    cbar=False,
                    ax=ax)

        ax.set_title(title, fontsize=18, pad=20)
        ax.set_ylabel("Age Group", fontsize=12, labelpad=10)

        # Only bottom plot gets x-labels and x-ticks
        if i < len(axes) - 1:
            ax.set_xlabel("")  # remove label
            ax.set_xticklabels([])  # remove tick labels
            ax.tick_params(axis='x', length=0)  # hide tick marks
        else:
            ax.set_xlabel("Outcome", fontsize=12, labelpad=10)
            # Get current tick labels, replace '_' with '\n'

            labels_plot = [underscore_to_mixed_breaks(t.get_text())
                           for t in ax.get_xticklabels()]

            # Apply them, keep horizontal alignment and rotation if desired
            ax.set_xticklabels(labels_plot, ha="center", rotation=0)

            # Y ticks always horizontal
        ax.set_yticklabels(ax.get_yticklabels(), ha="center", rotation=0)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path.joinpath('age_group_outcome_strata.png'), dpi=300)
        summary_long.to_csv(out_path.joinpath('age_group_outcome_counts.csv'), index=False)
        diagnosed_mat.to_csv(out_path.joinpath('age_group_diagnosed_counts.csv'), index=True)
        prevalent_mat.to_csv(out_path.joinpath('age_group_prevalent_counts.csv'), index=True)
        incident_mat.to_csv(out_path.joinpath('age_group_incident_counts.csv'), index=True)

    plt.show()

    return {
        "summary_long": summary_long,
        "diagnosed_matrix": diagnosed_mat,
        "prevalent_matrix": prevalent_mat,
        "incident_matrix": incident_mat
    }






def underscore_to_mixed_breaks(s: str) -> str:
    """
    Replace only the FIRST and LAST '_' with newline ('\\n'),
    and replace any MIDDLE '_' with a space (' ').

    Examples:
      "A_B"         -> "A\\nB"
      "A_B_C"       -> "A\\nB\\nC"
      "A_B_C_D"     -> "A\\nB C\\nD"
      "No_Underscore" -> "No\\nUnderscore"
      "A__B__C"     -> "A\\n B \\nC"   # empty middle parts preserved as spaces
    """
    parts = s.split('_')
    n = len(parts)

    if n == 1:
        # No underscores
        return s
    elif n == 2:
        # One underscore -> newline between first and last
        return f"{parts[0]}\n{parts[1]}"
    else:
        # Multiple underscores:
        # First and last separated by newlines, middle parts joined by spaces
        middle = ' '.join(parts[1:-1])
        return f"{parts[0]}\n{middle}\n{parts[-1]}"



def get_age_group_columns(g: int) -> dict:
    """
    Resolve age-group column names and labels given g = 2 or 3.
    """
    if g == 3:
        return {
            "group": "age_group_3",
            "cat": "age_group_3_cat",
            "suffix": "age_group_3_suffix",
            "high_label": ">60",
            "high_suffix": "ge_60",
        }

    if g == 2:
        return {
            "group": "age_group_2",
            "cat": "age_group_2_cat",
            "suffix": "age_group_2_suffix",
            "high_label": "gt_60",
            "high_suffix": "gt_60",
        }

    raise ValueError("g must be 2 or 3")


