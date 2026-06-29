from config.config import config, features, outcomes_short_names
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from library.column_registry import col_incident, rename_legacy_columns

# %% utils
def make_subject_level(df, id_col="eid", prob_col="prob_mean"):
    """
    Generates a subject-level DataFrame by aggregating by id_col and select the first value of prob_col.
    prob_col is expected to be the *average* probability across the nights, so it's a single value

    This function groups the input DataFrame by a specified identifier column, retaining
    the first occurrence of each identifier. It also renames a given probability column
    to a predefined name in the resulting DataFrame for standardization purposes.

    :param df: The input pandas DataFrame to process.
    :type df: pandas.DataFrame
    :param id_col: The name of the column used as the unique identifier for grouping.
        Defaults to "eid".
    :type id_col: str, optional
    :param prob_col: The name of the column containing probability values, which will
        be renamed in the resulting DataFrame. Defaults to "prob_mean".
    :type prob_col: str, optional
    :return: A new pandas DataFrame containing one entry per unique identifier, with
        the specified probability column renamed to "rbd_prob".
    :rtype: pandas.DataFrame
    """
    df_subj = (
        df.groupby(id_col, as_index=False)
          .first()
    )
    df_subj = df_subj.rename(columns={prob_col: "rbd_prob"})
    return df_subj



# %% thresholds reding

def load_and_normalize_thresholds(threshold_dict_paths: Dict[str, Path], file_name: str | None = None) -> Dict[str, Dict[str, Any]]:
    """
    Loads threshold JSON files and normalizes them into a unified structure.

    Output structure:
    thresholds_normalized[method][outcome] = {
        "type": method,
        "primary": float,     # main threshold to use in 2-class plots
        "all": dict           # all threshold values (percentile, quartile, etc.)
    }
    threshold_paths
            'thresholds': {
            'percentile': ...\'risk_percentile.json'),
            'roc': ...\'risk_roc.json'),
            'pr': ...\'risk_pr.json'),
            'f1': ...\'risk_f1.json'),
            'surv': ...\'risk_surv.json'),
            'quartile': ...\'risk_quartile.json'),
        },

    """

    thresholds_normalized = {}

    for method, path in threshold_dict_paths.items():
        if method == 'root' or method == 'collection':
            # ignore the root and collection
            # we are creating collection
            continue
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        thresholds_normalized[method] = {}

        # Check if the data is a single-outcome flat dictionary (has "outcome" and "thresholds"/"threshold")
        # If so, wrap it in a dict to match the multi-outcome structure: {outcome_name: data}
        if "outcome" in data and ("thresholds" in data or "threshold" in data):
             items_to_process = {data["outcome"]: data}
        else:
             items_to_process = data

        for outcome, values in items_to_process.items():

            # CASE 1 ? ROC, PR, F1, SURVIVAL
            if "threshold" in values:
                thresholds_normalized[method][outcome] = {
                    "type": method,
                    "primary": values["threshold"],
                    "all": {"threshold": values["threshold"]}
                }

            # CASE 2 ? percentile or quartile thresholds
            elif "thresholds" in values:
                # For plotting we select one "primary" threshold
                # convention: we choose the highest threshold
                thr_dict = values["thresholds"]
                ordered = sorted(thr_dict.items(), key=lambda x: x[1])
                primary_key, primary_val = ordered[-1]

                thresholds_normalized[method][outcome] = {
                    "type": method,
                    "primary": primary_val,
                    "all": thr_dict
                }

            else:
                raise ValueError(f"Unknown threshold format for method={method}, outcome={outcome}")

    if file_name:
        thresholds_normalized["file_name"] = file_name


    return thresholds_normalized


def print_thresholds(thresholds: dict):
    """
    Pretty-print the normalized thresholds dictionary produced by
    load_and_normalize_thresholds().
    """
    for method, outcomes_dict in thresholds.items():
        print(f"\n=== METHOD: {method.upper()} ===")
        for outcome, meta in outcomes_dict.items():
            primary = meta.get("primary")
            all_vals = meta.get("all")

            print(f"\nOutcome: {outcome}")
            print(f"  Primary: {primary:.4f}" if isinstance(primary, float) else f"  Primary: {primary}")
            print("  All thresholds:")
            for k, v in all_vals.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")



def get_threshold(dir_thresholds:Path) -> Dict[str, Path]:
    """
    Retrieve a dictionary mapping methods to their corresponding threshold JSON file paths.

    This function scans the given directory for JSON files and creates a dictionary that maps
    the method names extracted from the filenames to their respective file paths. If the file
    name starts with "risk_", the prefix is removed to derive the method name.

    :param dir_thresholds: Directory containing threshold JSON files.
    :type dir_thresholds: Path
    :return: A dictionary where the keys are method names (derived from JSON filenames) and
             the values are their corresponding file paths.
    :rtype: Dict[str, Path]
    """
    threshold_dict_paths = {}
    for p in dir_thresholds.glob("*.json"):
        # Example filename: risk_percentile_2g.json
        name = p.stem  # risk_percentile_2g

        # Remove common prefix if present
        if name.startswith("risk_"):
            method = name.replace("risk_", "")
        else:
            method = name

        threshold_dict_paths[method] = p

    if not threshold_dict_paths:
        raise FileNotFoundError(f"No threshold JSON files found in {dir_thresholds}")

    return threshold_dict_paths


def get_clean_risk_data(thresholds_root:Optional[Path]= None,
                        final_dir:Optional[Path]= None,
                        file_name: str = 'file_name_risk_data') -> tuple[dict, pd.DataFrame]:
    """
    Retrieve the normalized risk thresholds and the associated dataframe for a given file_name.
    Thresholds are in the structure:
                            <method>            <outcome>
        thresholds.get('percentile_3g').get('outcome_1a_pd_only')


    Args:
        file_name (str): The identifier for the file (e.g., 'ehr_diag_pd_rbd_only_val').

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing the normalized thresholds dictionary
                                   and the loaded DataFrame.
    """
    # 1. Paths from config
    if thresholds_root is None:
        thresholds_root = config['pp']['thresholds']['root']
    if final_dir is None:
        final_dir = config['pp']['final_dir']
    
    # 2. Construct paths
    # Collection.json is inside a folder named after the file_name
    collection_path = thresholds_root / file_name / 'risk_collection.json'
    parquet_path = final_dir / f"{file_name}.parquet"
    
    # 3. Load Thresholds (Collection)
    if not collection_path.exists():
        raise FileNotFoundError(f"Risk collection not found at: {collection_path}")
        
    with open(collection_path, 'r') as f:
        thresholds = json.load(f)
        
    # 4. Load DataFrame
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataframe parquet not found at: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    # remove the features from the matrix
    col_feat =  [f for f in features if f in df.columns]
    if len(col_feat) > 0:
        df = df.drop(columns=col_feat)
    # Exclude neurologically ineligible subjects (prevalent neuro disease at baseline).
    # train_sleep is NOT applied here: the ABK model was not trained on any UKBB subject,
    # so all actigraphy participants are valid for analysis regardless of train_sleep.
    df = df[df['neuro_exclude'] == 0].copy()
    print(f'  Excluded neuro_exclude subjects. Remaining: {df.shape[0]:,}')

    # Exclude subjects with poor actigraphy recording quality.
    # acc_bad_quality = True if ANY of: insufficient wear time (p90015), failed calibration
    # (p90016), not calibrated on own data (p90017), daylight-savings crossover (p90018),
    # unreliable device size (p90002), or non-zero recording problems (p90180).
    # Subjects failing these criteria produced unreliable actigraphy signals; their RBD
    # probability scores cannot be trusted for the survival analysis.
    if 'acc_bad_quality' in df.columns:
        n_before_acc = df.shape[0]
        df = df[df['acc_bad_quality'] != True].copy()
        n_excl_acc = n_before_acc - df.shape[0]
        print(f'  Excluded acc_bad_quality subjects: {n_excl_acc:,}. Remaining: {df.shape[0]:,}')
    else:
        print('  Warning: acc_bad_quality column not found — quality exclusion skipped.')

    # Exclude subjects doing night shifts at the time of actigraphy (instance 2).
    # Field p3426 = night shift work at current job; instance i2 = imaging visit when actigraphy
    # was recorded (~2014+). Night shift workers have disrupted circadian sleep architecture;
    # their actigraphy signal does not reflect physiological sleep patterns, making their
    # RBD probability scores unreliable for survival analysis.
    # Instance 0 (2006-2010) is intentionally NOT used: shift status 8+ years before recording
    # is not informative about actigraphy quality at recording time.
    _night_shift_col = 'shift_any_i2_p3426'
    if _night_shift_col in df.columns:
        n_before_ns = df.shape[0]
        df = df[df[_night_shift_col] != 1].copy()
        n_excl_ns = n_before_ns - df.shape[0]
        print(f'  Excluded night-shift (i2) subjects: {n_excl_ns:,}. Remaining: {df.shape[0]:,}')
    else:
        print(f'  Warning: {_night_shift_col} column not found — night-shift exclusion skipped.')

    # Migrate legacy column names to the new __-separated convention.
    # This is a safety net for stale parquet files that have not been
    # regenerated after the naming convention change.
    df = rename_legacy_columns(df)

    return thresholds, df


# %% Risk groups selections

def compute_risk_groups(prob:pd.Series, thresholds_dict:Dict[str, float]):
    """Compute risk groups from thresholds (works for raw scores or probabilities)."""

    thr_vals = sorted(set(thresholds_dict.values()))

    # Get data range
    min_val = prob.min()
    max_val = prob.max()

    # Build edges using actual data range
    group_edges = [min_val] + thr_vals + [max_val]

    # Ensure strictly increasing
    group_edges = np.unique(group_edges)

    if len(group_edges) < 2:
        raise ValueError("Not enough unique bin edges.")

    n_groups = len(group_edges) - 1
    labels = get_group_labels(n_groups)

    return pd.cut(
        prob,
        bins=group_edges,
        labels=labels,
        include_lowest=True,
        right=True
    )


def get_group_labels(n_groups) -> List[str]:
    """ Auto labels for N risk groups"""
    if n_groups == 2:
        return ["Low", "High"]
    if n_groups == 3:
        return ["Low", "Mid", "High"]
    return [f"Group {i+1}" for i in range(n_groups)]


# %% risk legends from groups
def make_risk_legend(group_labels, summary, colors):
    """
    Legend creator with:
      - colored dots
      - risk per group
      - risk ratios (RR)
      - reference group marked with "*"
      - adaptive reference: lowest group with >=1 case

    summary[group] must contain:
        {"n": int, "cases": int, "controls": int}
    """

    handles = []
    labels = []

    # ---------------------------------------------------------
    # Compute risks for each group
    # ---------------------------------------------------------
    risks = {}
    for grp in group_labels:
        vals = summary.get(grp, {"n": 0, "cases": 0})
        n = vals["n"]
        cases = vals["cases"]
        risks[grp] = (cases / n) if n > 0 else np.nan

    # ---------------------------------------------------------
    # Determine reference group:
    # lowest-risk group with at least 1 case
    # ---------------------------------------------------------
    ref_grp = None
    for grp in group_labels:
        if summary[grp]["cases"] > 0:
            ref_grp = grp
            break

    # If *no* group contains any cases -> no valid RR
    if ref_grp is None:
        ref_risk = np.nan
    else:
        ref_risk = risks[ref_grp]

    # ---------------------------------------------------------
    # Compute RRs relative to the reference
    # ---------------------------------------------------------
    RRs = {}
    for grp in group_labels:
        if np.isnan(ref_risk) or ref_risk == 0:
            RRs[grp] = np.nan
        else:
            RRs[grp] = risks[grp] / ref_risk

    # ---------------------------------------------------------
    # Build legend entries
    # ---------------------------------------------------------
    for grp, color in zip(group_labels, colors):

        vals = summary.get(grp, {"n": 0, "cases": 0, "controls": 0})

        handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=10,
            color=color
        )

        # Format risk & RR
        if not np.isnan(risks[grp]):
            risk_str = f"risk={risks[grp]:.3f}"
        else:
            risk_str = "risk=NA"

        if not np.isnan(RRs[grp]):
            rr_str = f"RR={RRs[grp]:.2f}"
        else:
            rr_str = "RR=NA"

        # Reference group marker
        star = " *" if grp == ref_grp else ""

        # Build label
        label = (
            rf"$\bf{{{grp}{star}}}$: n={vals['n']} "
            f" \n(cases={vals['cases']}, ctrl={vals['controls']})\n"
            f" {risk_str}, {rr_str}"
        )

        handles.append(handle)
        labels.append(label)

    return handles, labels


def make_risk_legend_counts(group_labels, counts, colors):
    """
    Simpler legend with just counts (N) and percentage.
    counts: dict {group_label: n}
    """
    
    handles = []
    labels = []
    
    total = sum(counts.values())
    
    for grp, color in zip(group_labels, colors):
        n = counts.get(grp, 0)
        pct = (n / total * 100) if total > 0 else 0
        
        handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=10,
            color=color
        )
        
        label = f"{grp}: N={n} ({pct:.1f}%)"
        handles.append(handle)
        labels.append(label)
        
    return handles, labels


# %% Plotting
def get_group_colors(n_groups: int) -> list:
    """Return canonical risk-group colors ordered Low → (Mid) → High.

    For 2 and 3 groups returns the project palette from config.config.
    For other counts falls back to a 'Reds' gradient.
    """
    from config.config import RBD_RISK_COLORS
    _fixed: dict = {
        2: [RBD_RISK_COLORS["Low"], RBD_RISK_COLORS["High"]],
        3: [RBD_RISK_COLORS["Low"], RBD_RISK_COLORS["Mid"], RBD_RISK_COLORS["High"]],
    }
    if n_groups in _fixed:
        return _fixed[n_groups]
    cmap = plt.cm.get_cmap("Reds")
    return [cmap(0.3 + 0.6 * (i / max(n_groups - 1, 1))) for i in range(n_groups)]

def add_risk_shading(
    ax,
    thresholds_dict,
    colors,
    x_min: float = 0,
    x_max: float = 1,
    alpha: float = 0.08,
):
    """
    Generalized shading for N risk groups defined by thresholds.

    Parameters
    ----------
    ax : matplotlib axis
    thresholds_dict : dict
        {"p90": ..., "p99": ...}
    colors : list
        List of colors for each group
    x_min : float
        Minimum x-axis value
    x_max : float
        Maximum x-axis value
    alpha : float
        Transparency level for shading (recommended 0.05–0.12 for publications)
    """

    thr_vals = sorted(set(thresholds_dict.values()))

    # Build edges from actual data range
    edges = [x_min] + thr_vals + [x_max]

    # Ensure strictly increasing
    edges = np.unique(edges)

    for i in range(len(edges) - 1):
        ax.axvspan(
            edges[i],
            edges[i + 1],
            color=colors[i],
            alpha=alpha,
            zorder=0,   # ensures shading stays behind histogram
        )


def plot_rbd_thresholds_methods_separately_per_outcome(
    df: pd.DataFrame,
    outcomes: list,
    thresholds: dict,
    prob_col: str = "rbd_prob",
    figsize: tuple[int, float] = (12, 2.6),
    font_scale: float = 1.0,
    save_path=None
):
    """
    One figure per OUTCOME.
    Rows = methods
    Columns = Validation & Non-validation.
    Font sizes controlled by font_scale.
    """

    method_list = list(thresholds.keys())
    n_methods = len(method_list)

    # global default font scaling
    base_title = int(14 * font_scale)
    base_label = int(10 * font_scale)
    base_ticks = int(8 * font_scale)
    base_legend = int(7 * font_scale)

    for outcome in outcomes:

        fig, axes = plt.subplots(
            n_methods, 2,
            figsize=(figsize[0], figsize[1] * n_methods),
            sharey=True
        )
        axes = np.atleast_2d(axes)

        fig.suptitle(
            f"Outcome: {outcome}",
            fontsize=base_title + 2,
            y=0.995
        )

        for i, method in enumerate(method_list):

            if outcome not in thresholds[method]:
                continue

            thr_dict = thresholds[method][outcome]["all"]
            thr_vals = sorted(thr_dict.values())

            n_groups = len(thr_vals) + 1
            colors = get_group_colors(n_groups)
            group_labels = get_group_labels(n_groups)

            for side, split_name in zip([0, 1], ["Validation", "Non-validation"]):

                ax = axes[i, side]

                if split_name == "Validation":
                    df_split = df[df["val"] == 1].copy()
                else:
                    df_split = df[df["val"] == 0].copy()

                df_split["risk_group"] = compute_risk_groups(df_split[prob_col], thr_dict)

                summary = {}
                for grp in group_labels:
                    sub = df_split[df_split["risk_group"] == grp]
                    summary[grp] = {
                        "n": len(sub),
                        "cases": int(sub[col_incident(outcome)].sum()),
                        "controls": int(sub["control"].sum())
                    }

                x_min = df_split[prob_col].min()
                x_max = df_split[prob_col].max()

                add_risk_shading(
                    ax=ax,
                    thresholds_dict=thr_dict,
                    colors=colors,
                    x_min=x_min,
                    x_max=x_max
                )

                ax.hist(
                    df_split[prob_col],
                    bins=60,
                    color="#4A90E2",
                    edgecolor="black",
                    alpha=0.65
                )

                for thr in thr_vals:
                    ax.axvline(thr, color="black", linestyle="--", linewidth=1.2)

                # labels & formatting
                if side == 0:
                    ax.set_ylabel(
                        f"{method.upper()}\nCount",
                        fontsize=base_label,
                        rotation=0,
                        labelpad=40
                    )

                ax.set_title(split_name, fontsize=base_label + 1)
                ax.set_xlabel("Predicted Probability", fontsize=base_label)
                ax.tick_params(axis="both", labelsize=base_ticks)
                ax.grid(alpha=0.25)

                # legend
                handles, labels_txt = make_risk_legend(group_labels, summary, colors)
                ax.legend(
                    handles,
                    labels_txt,
                    fontsize=base_legend,
                    loc="upper right",
                    title="Risk Groups",
                    title_fontsize=base_legend
                )

        fig.tight_layout(rect=[0, 0, 1, 0.98])

        if save_path:
            fname = f"outcome_{outcome}.png"
            fpath = save_path / fname
            fig.savefig(fpath, dpi=300)
            print(f"Saved {fpath}")

        plt.show()


def plot_rbd_thresholds_methods_separately_per_outcome_all_data(
    df: pd.DataFrame,
    outcomes: list,
    thresholds: dict,
    prob_col: str = "rbd_prob",
    max_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.2, 2.6),
    font_scale: float = 1.0,
    save_path=None
):
    """
    One figure per OUTCOME.
    Panels = methods (adaptive grid).
    All data pooled (no validation split).
    Histogram + thresholds + risk shading identical to previous logic.
    """

    import math

    method_list = list(thresholds.keys())
    n_methods = len(method_list)

    # font scaling
    base_title = int(14 * font_scale)
    base_label = int(10 * font_scale)
    base_ticks = int(8 * font_scale)
    base_legend = int(7 * font_scale)

    # adaptive grid
    n_cols = min(max_cols, n_methods)
    n_rows = math.ceil(n_methods / n_cols)

    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1] * n_rows

    for outcome in outcomes:

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            sharey=True
        )
        axes = np.atleast_2d(axes)

        fig.suptitle(
            f"Outcome: {outcome}",
            fontsize=base_title + 2,
            y=0.995
        )

        for idx, method in enumerate(method_list):

            if outcome not in thresholds[method]:
                continue

            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]

            thr_dict = thresholds[method][outcome]["all"]
            thr_vals = sorted(thr_dict.values())

            n_groups = len(thr_vals) + 1
            colors = get_group_colors(n_groups)
            group_labels = get_group_labels(n_groups)

            # risk groups
            df_local = df.copy()
            df_local["risk_group"] = compute_risk_groups(
                df_local[prob_col], thr_dict
            )

            # summary for legend
            summary = {}
            for grp in group_labels:
                sub = df_local[df_local["risk_group"] == grp]
                summary[grp] = {
                    "n": len(sub),
                    "cases": int(sub[col_incident(outcome)].sum()),
                    "controls": int(sub["control"].sum())
                }

            # background shading
            add_risk_shading(
                ax=ax,
                thresholds_dict=thr_dict,
                colors=colors,
                x_min=df_local[prob_col].min(),
                x_max=df_local[prob_col].max()
            )
            # histogram
            ax.hist(
                df_local[prob_col],
                bins=60,
                color="#4A90E2",
                edgecolor="black",
                alpha=0.65
            )

            # threshold lines
            for thr in thr_vals:
                ax.axvline(thr, color="black", linestyle="--", linewidth=1.2)

            ax.set_title(method.upper(), fontsize=base_label + 1)
            ax.set_xlabel("Predicted Probability", fontsize=base_label)
            ax.tick_params(axis="both", labelsize=base_ticks)
            ax.grid(alpha=0.25)

            # legend
            handles, labels_txt = make_risk_legend(
                group_labels, summary, colors
            )
            ax.legend(
                handles,
                labels_txt,
                fontsize=base_legend,
                loc="upper right",
                title="Risk Groups",
                title_fontsize=base_legend
            )

        # remove empty panels
        for j in range(idx + 1, n_rows * n_cols):
            r = j // n_cols
            c = j % n_cols
            fig.delaxes(axes[r, c])

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            fpath = save_path / f"outcome_{outcome}_methods_grid.png"
            fig.savefig(fpath, dpi=300)
            print(f"Saved {fpath}")

        plt.show()



def plot_rbd_thresholds_methods_separately(
    df: pd.DataFrame,
    thresholds: dict,
    outcomes: List[str],
    prob_col: str = "rbd_prob",
    max_cols: int = 3,
    figsize_per_panel: tuple = (5.5, 3.6),
    font_scale: float = 1.0,
    save_path=None,
):
    """
    One figure per OUTCOME. Panels = methods (adaptive grid). All data pooled.

    Thresholds are outcome-agnostic (derived from the RBD score distribution,
    stored under the 'rbd_only_distribution' key), but N, events, controls,
    risk and risk ratio are computed per real outcome so the annotation
    structure matches plot_rbd_thresholds_methods_separately_per_outcome_all_data.

    Cases = sum of incident_col
    Controls = sum of control
    RR computed inside make_risk_legend()

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level DataFrame.  Must contain ``prob_col``,
        ``{outcome}_incident``, and ``control`` columns.
    thresholds : dict
        Normalised threshold dict from load_and_normalize_thresholds.
        Methods keys map to a single 'rbd_only_distribution' outcome key.
    outcomes : list[str]
        Real outcome identifiers (e.g. ['outcome_1a_pd_only', ...]).
    prob_col : str
        Column with predicted RBD probability (default 'rbd_prob').
    max_cols : int
        Maximum panels per row.
    figsize_per_panel : tuple
        (width, height) in inches per panel.
    font_scale : float
        Global font scaling factor.
    save_path : Path or str, optional
        Directory to save figures.  One PNG per outcome saved as
        ``rbd_only_{outcome}.png``.
    """

    import math
    import seaborn as sns

    sns.set_style("whitegrid")

    method_list = [
        k for k, v in thresholds.items()
        if isinstance(v, dict) and len(v) > 0
    ]

    if not method_list:
        raise ValueError("No valid threshold methods found.")

    n_methods = len(method_list)

    base_title = int(14 * font_scale)
    base_label = int(11 * font_scale)
    base_ticks = int(9 * font_scale)
    base_legend = int(8 * font_scale)

    n_cols = min(max_cols, n_methods)
    n_rows = math.ceil(n_methods / n_cols)

    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1] * n_rows

    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for outcome in outcomes:

        incident_col = col_incident(outcome)
        if incident_col not in df.columns:
            print(f"  [SKIP] {outcome}: missing column {incident_col}")
            continue

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            sharey=True,
        )

        axes = np.atleast_2d(axes)

        fig.suptitle(
            f"RBD-based Risk Stratification Thresholds {outcomes_short_names.get(outcome)} Incident",
            fontsize=base_title + 2,
            y=0.99,
            fontweight="bold",
        )

        last_idx = 0

        for idx, method in enumerate(method_list):

            last_idx = idx
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]

            outcome_keys = [
                k for k in thresholds[method]
                if isinstance(thresholds[method][k], dict)
            ]

            if not outcome_keys:
                ax.axis("off")
                continue

            thr_block = thresholds[method][outcome_keys[0]]

            if "all" not in thr_block:
                ax.axis("off")
                continue

            thr_dict = thr_block["all"]
            thr_vals = sorted(
                v for v in thr_dict.values()
                if isinstance(v, (int, float))
            )

            n_groups = len(thr_vals) + 1
            colors = get_group_colors(n_groups)
            group_labels = get_group_labels(n_groups)

            df_local = df.copy()

            df_local["risk_group"] = compute_risk_groups(
                df_local[prob_col],
                thr_dict,
            )

            # --------------------------------------------------
            # Histogram
            # --------------------------------------------------

            sns.histplot(
                df_local[prob_col],
                bins=35,
                color="#6E8FBF",
                edgecolor="white",
                linewidth=0.4,
                alpha=0.75,
                ax=ax,
            )

            # Density curve
            sns.kdeplot(
                df_local[prob_col],
                color="black",
                linewidth=1.5,
                ax=ax,
            )

            # --------------------------------------------------
            # Risk group shading
            # --------------------------------------------------

            add_risk_shading(
                ax,
                thr_dict,
                colors,
                x_min=df_local[prob_col].min(),
                x_max=df_local[prob_col].max(),
                alpha=0.08,
            )

            # --------------------------------------------------
            # Threshold lines
            # --------------------------------------------------

            for thr in thr_vals:
                ax.axvline(
                    thr,
                    color="black",
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.9,
                )

            # --------------------------------------------------
            # Group statistics
            # --------------------------------------------------

            summary = {}

            for grp in group_labels:
                sub = df_local[df_local["risk_group"] == grp]

                summary[grp] = {
                    "n": len(sub),
                    "cases": int(sub[incident_col].sum()),
                    "controls": int(sub["control"].sum())
                    if "control" in sub.columns
                    else 0,
                }

            handles, labels_txt = make_risk_legend(
                group_labels,
                summary,
                colors,
            )

            ax.legend(
                handles,
                labels_txt,
                fontsize=base_legend,
                loc="upper right",
                frameon=True,
                framealpha=0.95,
                edgecolor="0.8",
                title="Risk Groups",
                title_fontsize=base_legend + 1,
            )

            # --------------------------------------------------
            # Titles and labels
            # --------------------------------------------------

            panel_label = panel_letters[idx]

            ax.set_title(
                f"{panel_label}. {method.replace('_',' ').title()}",
                fontsize=base_label + 1,
                loc="left",
                fontweight="bold",
            )

            ax.set_xlabel(
                "Model RBD scores",
                fontsize=base_label,
            )

            if c == 0:
                ax.set_ylabel(
                    "Participant count",
                    fontsize=base_label,
                )

            ax.tick_params(
                axis="both",
                labelsize=base_ticks,
            )

            ax.grid(
                alpha=0.15,
                linestyle="-",
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # remove empty panels
        for j in range(last_idx + 1, n_rows * n_cols):
            fig.delaxes(axes[j // n_cols, j % n_cols])

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path is not None:

            out_dir = Path(save_path)
            out_dir.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                out_dir / f"rbd_only_{outcome}.pdf",
                bbox_inches="tight",
            )

            fig.savefig(
                out_dir / f"rbd_only_{outcome}.png",
                dpi=400,
                bbox_inches="tight",
            )

        plt.show()



def plot_rbd_thresholds_publication(
    df: pd.DataFrame,
    thresholds: dict,
    outcomes: List[str],
    prob_col: str = "rbd_prob",
    figsize: tuple = (7.5, 2.8),
    font_scale: float = 1.0,
    save_path: Optional[Path] = None,
    file_format: str = "png",
) -> None:
    """
    Publication-quality three-panel histogram figure for RBD risk stratification.

    Produces one figure per outcome with horizontal panels (A–C), one panel per
    stratification method (percentile_2g, percentile_3g, quartile).

    Design rationale:
    - Gray semi-transparent histogram (35 bins) with a scaled KDE density overlay
      separates the distribution shape from threshold markers.
    - Colorblind-safe Vega-10 palette (blue / orange / red) encodes risk level
      consistently across all panels and threshold lines.
    - Threshold lines are colored by the risk group they bound on the right,
      with rotated inline labels; no background shading is used.
    - High-risk RR and case count shown as a compact annotation per panel.
    - Top/right spines removed; horizontal y-grid only (reduced chart junk).
    - Export: PNG at 300 dpi by default (PDF/SVG also supported for publication).

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level DataFrame. Must contain ``prob_col``,
        ``{outcome}_incident``, and ``control`` columns.
    thresholds : dict
        Normalised threshold dict from ``load_and_normalize_thresholds``.
        Methods keys map to a single ``rbd_only_distribution`` outcome key.
    outcomes : list[str]
        Real outcome identifiers; one figure is produced per outcome.
    prob_col : str
        Column with predicted probability (default ``"rbd_prob"``).
    figsize : tuple[float, float]
        Figure size in inches. Default (7.5, 2.8) ≈ 190 mm × 71 mm.
    font_scale : float
        Multiplicative font-size scaling (default 1.0).
    save_path : Path or str, optional
        Output directory for figures. Filename: ``rbd_thresholds_pub_{outcome}.{fmt}``.
        If None, only displays via plt.show() (no save).
    file_format : str
        ``"pdf"``, ``"svg"``, or ``"png"`` (default ``"png"``).
    """
    from scipy.stats import gaussian_kde
    import matplotlib.patches as mpatches

    # ── Risk group palette (Low → Mid → High) — sourced from config.config ───
    from config.config import RBD_RISK_COLORS as _RC
    _CB: Dict[int, List[str]] = {
        2: [_RC["Low"], _RC["High"]],
        3: [_RC["Low"], _RC["Mid"], _RC["High"]],
        4: [_RC["Low"], _RC["Mid"], "#f7c88b", _RC["High"]],
    }

    _PANEL_TITLE: Dict[str, str] = {
        "percentile_2g": "Percentile Threshold (2 groups)",
        "percentile_3g": "Percentile Threshold (3 groups)",
        "quartile": "Quartile Threshold",
    }
    _THRESHOLD_LABELS: Dict[int, List[str]] = {
        1: ["Risk threshold"],
        2: ["Low \u2192 Mid", "Mid \u2192 High"],
        3: ["T\u2081", "T\u2082", "T\u2083"],
    }
    _ABC = "ABCDEFGHIJ"

    # ── Font-size hierarchy ───────────────────────────────────────────────────
    fs_abc   = int(13 * font_scale)
    fs_title = int(11 * font_scale)
    fs_axis  = int(10 * font_scale)
    fs_tick  = int(9  * font_scale)
    fs_annot = int(8  * font_scale)
    fs_leg   = int(8  * font_scale)

    method_list = [
        k for k, v in thresholds.items()
        if isinstance(v, dict) and len(v) > 0
    ]
    if not method_list:
        raise ValueError("No valid threshold methods found in thresholds dict.")

    n_panels = len(method_list)

    for outcome in outcomes:
        incident_col = col_incident(outcome)
        if incident_col not in df.columns:
            print(f"  [SKIP] {outcome}: missing column {incident_col}")
            continue

        fig, axes = plt.subplots(
            1, n_panels,
            figsize=figsize,
            sharey=True,
        )
        axes = np.atleast_1d(axes)

        # Global score range → consistent x-axis across panels
        scores_all = df[prob_col].dropna()
        x_min = float(scores_all.min())
        x_max = float(scores_all.max())
        x_pad = (x_max - x_min) * 0.03
        n_bins = 35

        # Pre-compute histogram max for a shared y-ceiling
        hist_counts, _ = np.histogram(scores_all.values, bins=n_bins)
        y_ceil = int(hist_counts.max() * 1.30)

        for panel_idx, method in enumerate(method_list):
            ax = axes[panel_idx]

            # ── Retrieve thresholds for this method ───────────────────────────
            outcome_keys = [
                k for k in thresholds[method]
                if isinstance(thresholds[method][k], dict)
            ]
            if not outcome_keys:
                ax.axis("off")
                continue

            thr_block = thresholds[method][outcome_keys[0]]
            if "all" not in thr_block or not isinstance(thr_block["all"], dict):
                ax.axis("off")
                continue

            thr_dict = thr_block["all"]
            thr_vals = sorted(
                v for v in thr_dict.values()
                if isinstance(v, (int, float))
            )
            n_groups = len(thr_vals) + 1
            colors = _CB.get(n_groups, _CB[3])
            group_labels = get_group_labels(n_groups)

            # ── Assign risk groups and compute per-group statistics ───────────
            keep_cols = [prob_col, incident_col] + (
                ["control"] if "control" in df.columns else []
            )
            df_local = df[keep_cols].dropna(subset=[prob_col]).copy()
            df_local["risk_group"] = compute_risk_groups(df_local[prob_col], thr_dict)

            summary: Dict[str, Dict] = {}
            for grp in group_labels:
                sub = df_local[df_local["risk_group"] == grp]
                n = len(sub)
                cases = int(sub[incident_col].sum()) if n > 0 else 0
                summary[grp] = {
                    "n": n,
                    "cases": cases,
                    "rate": cases / n if n > 0 else 0.0,
                }

            # Risk ratio vs. the lowest (reference) group
            ref_rate = summary[group_labels[0]]["rate"]
            for grp in group_labels:
                r = summary[grp]["rate"]
                summary[grp]["rr"] = r / ref_rate if ref_rate > 0 else float("nan")

            # ── Histogram ─────────────────────────────────────────────────────
            ax.hist(
                df_local[prob_col].values,
                bins=n_bins,
                range=(x_min - x_pad, x_max + x_pad),
                color="#CCCCCC",
                edgecolor="white",
                linewidth=0.3,
                alpha=0.85,
                zorder=2,
            )

            # ── KDE density curve (scaled to count units) ─────────────────────
            vals = df_local[prob_col].values
            if len(vals) > 10:
                kde_fn = gaussian_kde(vals, bw_method="scott")
                x_kde = np.linspace(x_min - x_pad, x_max + x_pad, 500)
                y_kde = kde_fn(x_kde)
                bin_width = (x_max - x_min + 2 * x_pad) / n_bins
                ax.plot(
                    x_kde,
                    y_kde * len(vals) * bin_width,
                    color="black",
                    linewidth=1.2,
                    zorder=5,
                )

            # ── Colored dashed threshold lines with inline annotations ─────────
            # Line i separates group[i] from group[i+1]; color = upper group.
            thr_plot_labels = _THRESHOLD_LABELS.get(
                len(thr_vals),
                [f"T{i+1}" for i in range(len(thr_vals))],
            )
            for t_idx, (thr, t_lbl) in enumerate(zip(thr_vals, thr_plot_labels)):
                line_color = colors[t_idx + 1]
                ax.axvline(
                    thr,
                    color=line_color,
                    linestyle="--",
                    linewidth=1.4,
                    zorder=6,
                    alpha=0.9,
                )
                ax.text(
                    thr - (x_max - x_min) * 0.012,
                    y_ceil * 0.90,
                    t_lbl,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=fs_annot,
                    color=line_color,
                    zorder=7,
                )

            # ── High-risk RR annotation (upper-right corner) ──────────────────
            high_grp = group_labels[-1]
            high_rr = summary[high_grp]["rr"]
            high_cases = summary[high_grp]["cases"]
            if not np.isnan(high_rr):
                ax.text(
                    0.98,
                    0.97,
                    f"High-risk group\nRR = {high_rr:.2f}\ncases = {high_cases}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=fs_annot,
                    color=colors[-1],
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor=colors[-1],
                        alpha=0.85,
                        linewidth=0.8,
                    ),
                    zorder=8,
                )

            # ── Legend: group membership counts ───────────────────────────────
            legend_patches = [
                mpatches.Patch(
                    facecolor=c,
                    alpha=0.85,
                    label=f"{grp}  N={summary[grp]['n']:,}",
                )
                for grp, c in zip(group_labels, colors)
            ]
            ax.legend(
                handles=legend_patches,
                fontsize=fs_leg,
                loc="upper left",
                framealpha=0.85,
                edgecolor="#CCCCCC",
                handlelength=1.0,
                handleheight=0.85,
            )

            # ── Panel letter (A/B/C) and title ────────────────────────────────
            panel_title = _PANEL_TITLE.get(method, method.upper())
            ax.set_title(panel_title, fontsize=fs_title, fontweight="bold", pad=5)
            ax.text(
                -0.06,
                1.08,
                _ABC[panel_idx],
                transform=ax.transAxes,
                fontsize=fs_abc,
                fontweight="bold",
                va="top",
                ha="left",
            )

            # ── Axis cosmetics ────────────────────────────────────────────────
            ax.set_xlabel("Model Predicted Probability Score", fontsize=fs_axis)
            if panel_idx == 0:
                ax.set_ylabel("Participant Count", fontsize=fs_axis)
            ax.tick_params(axis="both", labelsize=fs_tick)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(0, y_ceil)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.45, zorder=0)

        fig.tight_layout(w_pad=2.5)

        if save_path is not None:
            out_dir = Path(save_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"rbd_thresholds_pub_{outcome}.{file_format}"
            out_path = out_dir / fname
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"  [Saved] {out_path}")

        plt.show()
