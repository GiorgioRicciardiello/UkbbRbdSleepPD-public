"""
CONSORT-style reporting utilities for EHR-based cohort studies.

Creates:
1) CONSORT participant flow table
2) Follow-up summary table (events, censoring, median follow-up)
3) Staging flow table (counts after each processing stage)
4) Detailed outcome breakdown (diagnosis, prevalence, incidence, TTE stats)

This module performs REPORTING ONLY.
No filtering or mutation of the cohort.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from library.column_registry import col_incident
from tabulate import tabulate
from library.risk.risk_helpers import make_subject_level

# %%

# -----------------------------
# Core helpers (SHARED)
# -----------------------------

def _safe_loss(before: int, after: int) -> int:
    return max(before - after, 0)

def _pct_loss(before: int, after: int) -> float:
    return 100 * _safe_loss(before, after) / before if before > 0 else np.nan

def _count_subjects(df: pd.DataFrame, id_col: str) -> int:
    return int(df[id_col].nunique())

def _count_outcomes(
    df: pd.DataFrame,
    outcomes: List[str],
    suffixes: List[str] = ["diagnosed", "prevalent", "incident"]
) -> Dict[str, int]:
    """
    Returns a flat dict:
    {
        "pd_diagnosed": 259,
        "pd_prevalent": 59,
        "pd_incident": 200,
        ...
    }
    """
    counts = {}
    for out in outcomes:
        for sfx in suffixes:
            col = f"{out}_{sfx}"
            counts[col] = int(df[col].sum()) if col in df.columns else 0
    return counts



def _build_consort_transitions(
    stages: List[tuple],
    outcomes: List[str],
    id_col: str
) -> pd.DataFrame:
    """
    . Generic stage-to-stage CONSORT transition function
    stages = [
        ("Stage name", dataframe),
        ...
    ]
    """

    rows = []

    for (stage_b, df_b), (stage_a, df_a) in zip(stages[:-1], stages[1:]):

        # -------------------------
        # Subjects
        # -------------------------
        n_b = _count_subjects(df_b, id_col)
        n_a = _count_subjects(df_a, id_col)

        rows.append({
            "Metric": "Subjects",
            "Before": n_b,
            "After": n_a,
            "Lost": _safe_loss(n_b, n_a),
            "% Lost": _pct_loss(n_b, n_a),
            "Stage transition": f"{stage_b} -> {stage_a}"
        })

        # -------------------------
        # Outcomes (shared logic)
        # -------------------------
        counts_b = _count_outcomes(df_b, outcomes)
        counts_a = _count_outcomes(df_a, outcomes)

        for key in counts_b.keys():
            rows.append({
                "Metric": key.replace("_", " ").upper(),
                "Before": counts_b[key],
                "After": counts_a[key],
                "Lost": _safe_loss(counts_b[key], counts_a[key]),
                "% Lost": _pct_loss(counts_b[key], counts_a[key]),
                "Stage transition": f"{stage_b} -> {stage_a}"
            })

    df = pd.DataFrame(rows)

    # ordering
    metric_order = (
        ["Subjects"] +
        [f"{out}_{sfx}".replace("_", " ").upper()
         for out in outcomes
         for sfx in ["diagnosed", "prevalent", "incident"]]
    )

    df["Metric"] = pd.Categorical(
        df["Metric"],
        categories=metric_order,
        ordered=True
    )

    return (
        df
        .sort_values(["Stage transition", "Metric"])
        .reset_index(drop=True)
    )


def _night_cleaning_attrition_table(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    outcomes: List[str],
    id_col: str
) -> pd.DataFrame:
    """
    Wrapper that enforces identical counting logic
    as CONSORT stage transitions.
    """
    return _build_consort_transitions(
        stages=[
            ("Before night cleaning", df_before),
            ("After night cleaning", df_after),
        ],
        outcomes=outcomes,
        id_col=id_col
    )

def generate_consort_tables(
    df_ehr: pd.DataFrame,
    df_ehr_out: pd.DataFrame,
    df_ehr_out_ne: pd.DataFrame,
    df_ehr_out_ne_contr_cov: pd.DataFrame,
    df_ehr_out_ne_contr_cov_age: pd.DataFrame,
    df_ehr_out_ne_contr_cov_age_nights: pd.DataFrame,
    df_ehr_out_ne_contr_cov_age_nights_clean: pd.DataFrame,
    outcomes: List[str],
    id_col: str = "eid",
    save_dir: Path | None = None,
    verbose: bool = True
) -> pd.DataFrame:

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    df_consort_long = _build_consort_transitions(
        stages=[
            ("Initial cohort (with actigraphy)", df_ehr_out_ne_contr_cov_age),
            ("Neuro exclusions", df_ehr_out_ne_contr_cov_age.loc[
                df_ehr_out_ne["neuro_exclude"] == False
            ]),
            ("Night cleaning", df_ehr_out_ne_contr_cov_age_nights_clean.loc[
                df_ehr_out_ne_contr_cov_age_nights_clean["neuro_exclude"] == False
            ]),
        ],
        outcomes=outcomes,
        id_col=id_col
    )

    if save_dir:
        df_consort_long.to_csv(save_dir / "staging_consort_long.csv", index=False)

    if verbose:
        print(tabulate(df_consort_long, headers="keys", tablefmt="github", showindex=False))

    return df_consort_long



# %% RBD analysis
def consort_rbd_risk_flow(
    df: pd.DataFrame,
    outcomes: list[str],
    id_col: str = "eid",
    save_dir: Path | None = None,
    verbose: bool = True,
):
    """
    CONSORT-style long attrition table for RBD risk pipeline.
    Structure:
        Metric | Before | After | Lost | % Lost | Stage transition
    """

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def safe_loss(b, a):
        return max(b - a, 0)

    def pct_loss(b, a):
        return 100 * safe_loss(b, a) / b if b > 0 else np.nan

    rows = []

    # --------------------------------------------------
    # Define stages (PROCESSING ORDER)
    # --------------------------------------------------
    stages = [
        (
            "After night QC",
            df,
        ),
        (
            "With RBD prediction",
            df.loc[df["rbd_bin"].notna()],
        ),
        (
            "Eligible for risk analysis (non-validation)",
            df.loc[df["val"] == False],
        ),
    ]

    # --------------------------------------------------
    # Stage-to-stage transitions
    # --------------------------------------------------
    for (stage_b, df_b), (stage_a, df_a) in zip(stages[:-1], stages[1:]):

        # -------------------------
        # Subjects
        # -------------------------
        b = df_b[id_col].nunique()
        a = df_a[id_col].nunique()

        rows.append({
            "Metric": "Subjects",
            "Before": b,
            "After": a,
            "Lost": safe_loss(b, a),
            "% Lost": pct_loss(b, a),
            "Stage transition": f"{stage_b} -> {stage_a}"
        })

        # -------------------------
        # Nights
        # -------------------------
        b_n = len(df_b)
        a_n = len(df_a)

        rows.append({
            "Metric": "Nights",
            "Before": b_n,
            "After": a_n,
            "Lost": safe_loss(b_n, a_n),
            "% Lost": pct_loss(b_n, a_n),
            "Stage transition": f"{stage_b} -> {stage_a}"
        })

    # --------------------------------------------------
    # Outcome-specific eligibility (incident-based)
    # --------------------------------------------------
    df_eligible = stages[-1][1]

    for outcome in outcomes:
        inc_col = col_incident(outcome)
        if inc_col not in df_eligible.columns:
            continue

        df_inc = df_eligible.loc[df_eligible[inc_col].notna()]

        b = df_eligible[id_col].nunique()
        a = df_inc[id_col].nunique()

        rows.append({
            "Metric": f"{outcome.upper()} eligible subjects",
            "Before": b,
            "After": a,
            "Lost": safe_loss(b, a),
            "% Lost": pct_loss(b, a),
            "Stage transition": "Risk-eligible -> Outcome-eligible"
        })

        rows.append({
            "Metric": f"{outcome.upper()} incident cases",
            "Before": df_inc[inc_col].sum() + (df_inc[inc_col] == 0).sum(),
            "After": df_inc[inc_col].sum(),
            "Lost": np.nan,
            "% Lost": np.nan,
            "Stage transition": "Outcome-eligible"
        })

    # --------------------------------------------------
    # Final table
    # --------------------------------------------------
    df_consort = pd.DataFrame(rows)

    metric_order = (
        ["Subjects", "Nights"] +
        [f"{out.upper()} eligible subjects" for out in outcomes] +
        [f"{out.upper()} incident cases" for out in outcomes]
    )

    df_consort["Metric"] = pd.Categorical(
        df_consort["Metric"],
        categories=metric_order,
        ordered=True
    )

    df_consort = (
        df_consort
        .sort_values(["Stage transition", "Metric"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    if verbose:
        print("\nCONSORT ? RBD RISK PIPELINE (LONG FORMAT)")
        print(tabulate(df_consort, headers="keys", tablefmt="github", showindex=False))

    if save_dir:
        df_consort.to_csv(save_dir / "consort_rbd_risk_long.csv", index=False)

    return df_consort

# def consort_rbd_risk_flow_old(
#     df: pd.DataFrame,
#     outcomes: list[str],
#     save_dir: Path | None = None,
#     verbose: bool = True,
# ):
#     """
#     CONSORT-style flow table for RBD risk analysis.
#     Tracks subjects and nights through prediction, validation exclusion,
#     and outcome-specific eligibility.
#     """
#     save_dir.mkdir(parents=True, exist_ok=True)
#     rows_global = []
#
#     # --------------------------------------------------
#     # Stage 0 ? Input
#     # --------------------------------------------------
#     rows_global.append({
#         "stage": "After night QC",
#         "subjects": df["eid"].nunique(),
#         "nights": len(df)
#     })
#
#     # --------------------------------------------------
#     # Stage 1 ? RBD prediction availability
#     # --------------------------------------------------
#     has_pred = df["rbd_bin"].notna()
#
#     rows_global.append({
#         "stage": "With RBD prediction",
#         "subjects": df.loc[has_pred, "eid"].nunique(),
#         "nights": has_pred.sum()
#     })
#
#     # --------------------------------------------------
#     # Stage 2 ? Validation exclusion
#     # --------------------------------------------------
#     df_no_val = df.loc[df["val"] == False]
#
#     rows_global.append({
#         "stage": "Excluded validation subjects",
#         "subjects": df.loc[df["val"] == True, "eid"].nunique(),
#         "nights": df.loc[df["val"] == True].shape[0]
#     })
#
#     rows_global.append({
#         "stage": "Eligible for risk analysis",
#         "subjects": df_no_val["eid"].nunique(),
#         "nights": len(df_no_val)
#     })
#
#     consort_global = pd.DataFrame(rows_global)
#
#     # --------------------------------------------------
#     # Outcome-specific CONSORT
#     # --------------------------------------------------
#     outcome_rows = []
#
#     for outcome in outcomes:
#         inc_col = col_incident(outcome)
#
#         eligible = df_no_val.loc[
#             df_no_val[inc_col].notna()
#         ]
#
#         outcome_rows.append({
#             "outcome": outcome,
#             "subjects_eligible": eligible["eid"].nunique(),
#             "incident_cases": eligible[inc_col].sum(),
#             "controls": (eligible[inc_col] == 0).sum(),
#             "nights_used": len(eligible),
#             "subjects_with_nights": eligible["eid"].nunique()
#         })
#
#     consort_outcome = pd.DataFrame(outcome_rows)
#
#     # --------------------------------------------------
#     # Save / print
#     # --------------------------------------------------
#     if verbose:
#         print("\nCONSORT ? RBD RISK PIPELINE (GLOBAL)")
#         print(tabulate(consort_global, headers="keys", tablefmt="github", showindex=False))
#
#         print("\nCONSORT ? OUTCOME-SPECIFIC")
#         print(tabulate(consort_outcome, headers="keys", tablefmt="github", showindex=False))
#
#     if save_dir:
#         save_dir.mkdir(parents=True, exist_ok=True)
#         consort_global.to_csv(save_dir / "consort_rbd_global.csv", index=False)
#         consort_outcome.to_csv(save_dir / "consort_rbd_outcome.csv", index=False)
#
#     return consort_global, consort_outcome


# ── STROBE/CONSORT-compliant participant flow ─────────────────────────────────

# Expected flag columns in the processed DataFrame
_FLAG_NEURO    = "neuro_exclude"    # bool: True = subject has neuro dx before baseline
_FLAG_ACC      = "acc_bad_quality"  # bool: True = poor accelerometer recording
_SURV_TEMPLATE = "{outcome}__surv_days"   # NaN for prevalent cases (excluded from follow-up)
_INC_TEMPLATE  = "{outcome}__incident"    # 1 = incident case
_PREV_TEMPLATE = "{outcome}__prevalent"   # 1 = prevalent case (excluded per outcome)
_DX_TEMPLATE   = "{outcome}__dx"          # 1 = any diagnosis (incident + prevalent)

# STROBE Section labels (Altman 1996 / STROBE 2007 Item 13)
_SECTION_ENROL   = "Enrollment"
_SECTION_GLOBAL  = "Global exclusions"
_SECTION_OUTCOME = "Outcome-specific"


def _row(
    step: float,
    section: str,
    description: str,
    exclusion_criterion: str,
    n_before: int,
    n_excluded: int,
    n_after: int,
    outcome: str = "",
    n_incident: int = -1,
    n_prevalent_excluded: int = -1,
    median_follow_up_years: float = np.nan,
    notes: str = "",
) -> Dict:
    """Construct a single CONSORT flow row."""
    pct = round(100.0 * n_excluded / n_before, 2) if n_before > 0 else np.nan
    return {
        "step":                   step,
        "section":                section,
        "description":            description,
        "exclusion_criterion":    exclusion_criterion,
        "n_before":               n_before,
        "n_excluded":             n_excluded,
        "pct_excluded":           pct if n_excluded > 0 else np.nan,
        "n_after":                n_after,
        "outcome":                outcome,
        "n_incident_cases":       n_incident       if n_incident       >= 0 else np.nan,
        "n_prevalent_excluded":   n_prevalent_excluded if n_prevalent_excluded >= 0 else np.nan,
        "median_follow_up_years": median_follow_up_years,
        "notes":                  notes,
    }


def _median_fu(df: pd.DataFrame, time_col: str) -> float:
    """Median follow-up among censored (event-free) subjects (years)."""
    censored = df.loc[df[time_col.replace("__surv_days", "__incident")] == 0, time_col]
    if censored.empty:
        return np.nan
    return round(float(censored.median()) / 365.25, 2)


def generate_ehr_consort_flow(
    df: pd.DataFrame,
    outcomes: List[str],
    outcome_labels: Optional[Dict[str, str]] = None,
    n_all_ukbb: Optional[int] = None,
    id_col: str = "eid",
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate a STROBE/CONSORT-compliant participant flow table for the
    EHR-based UKBB cohort.

    Reconstructs the full attrition cascade from flag columns already
    present in the final processed DataFrame (output of
    ``UkbDataProcessor.apply_processing_pipeline()``).  No filtering is
    applied — only counting.

    Exclusion hierarchy (applied cumulatively):
      Step 0 — Total UKBB enrolled (optional, requires ``n_all_ukbb``)
      Step 1 — Actigraphy cohort (= ``len(df)``, all subjects with
                ``wear_time_start`` not NaT)
      Step 2 — After neurological exclusion (``neuro_exclude == False``)
      Step 3 — After poor actigraphy quality (``acc_bad_quality == False``)
                *Note: quality filter applied at night-level in actigraphy
                pipeline; subject-level count shown for transparency.*
      Step 4+ — Per-outcome analytical cohort (prevalent cases removed via
                 ``{outcome}__surv_days.notna()``)

    For each step the function reports:
      - N subjects entering the step
      - N excluded and % excluded
      - N subjects remaining
      - Per-outcome: N incident cases, N prevalent excluded,
        median censored follow-up (years)

    This satisfies STROBE Checklist Item 13 (numbers at each study stage)
    and CONSORT 2010 Item 13a (participants enrolled, allocated, followed
    up, analysed).

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level, fully processed DataFrame.  Must contain:
        ``neuro_exclude``, ``acc_bad_quality`` (if available),
        ``{outcome}__incident``, ``{outcome}__prevalent``,
        ``{outcome}__surv_days`` for each outcome in ``outcomes``.
    outcomes : list[str]
        Outcome keys (e.g. ['outcome_1a_pd_only', ...]).
    outcome_labels : dict[str, str], optional
        Human-readable names for each outcome key.  Falls back to the
        outcome key if not provided.
    n_all_ukbb : int, optional
        Total UKBB participants before the actigraphy filter.  Adds an
        extra Step 0 row showing the full source population.
    id_col : str
        Subject identifier column.
    save_dir : Path, optional
        If provided, saves ``consort_flow_table.csv`` and
        ``consort_flow_table.xlsx`` there.
    verbose : bool
        Print the table to stdout.

    Returns
    -------
    pd.DataFrame
        One row per attrition step.  Columns: step, section, description,
        exclusion_criterion, n_before, n_excluded, pct_excluded, n_after,
        outcome, n_incident_cases, n_prevalent_excluded,
        median_follow_up_years, notes.
    """
    labels = outcome_labels or {}
    rows: List[Dict] = []

    # ── STEP 0: Source population (optional) ─────────────────────────────────
    n_actig = int(df[id_col].nunique())

    if n_all_ukbb is not None:
        rows.append(_row(
            step=0,
            section=_SECTION_ENROL,
            description="Total UK Biobank registered participants",
            exclusion_criterion="",
            n_before=n_all_ukbb,
            n_excluded=n_all_ukbb - n_actig,
            n_after=n_actig,
            notes="Restricted to participants with valid actigraphy data "
                  "(wear_time_start not missing)",
        ))

    # ── STEP 1: Actigraphy cohort (entry point if no Step 0) ─────────────────
    rows.append(_row(
        step=1,
        section=_SECTION_ENROL,
        description="Participants with valid actigraphy (wear_time_start)",
        exclusion_criterion="",
        n_before=n_actig,
        n_excluded=0,
        n_after=n_actig,
        notes="All participants with non-missing wear_time_start "
              "(actigraphy baseline date)",
    ))

    # ── STEP 2: Neurological exclusion ───────────────────────────────────────
    if _FLAG_NEURO not in df.columns:
        raise KeyError(
            f"'{_FLAG_NEURO}' column not found. "
            "Run add_neuro_exclusion() before calling this function."
        )

    n_neuro_excl = int(df[_FLAG_NEURO].sum())
    n_post_neuro = n_actig - n_neuro_excl

    rows.append(_row(
        step=2,
        section=_SECTION_GLOBAL,
        description="After neurological exclusion",
        exclusion_criterion=(
            "Prevalent neurological/neurodegenerative diagnosis "
            "(ICD-10 criteria) prior to actigraphy baseline"
        ),
        n_before=n_actig,
        n_excluded=n_neuro_excl,
        n_after=n_post_neuro,
        notes="neuro_exclude == True",
    ))

    # ── STEP 3: Poor actigraphy quality (informational) ──────────────────────
    # This is a NIGHT-LEVEL filter in the actigraphy pipeline.
    # At the subject level we report it for transparency, but note that
    # subjects with acc_bad_quality are NOT removed from the EHR cohort;
    # they are excluded at the night-level preprocessing step.
    n_post_excl = n_post_neuro   # analytical base before acc_quality
    base_mask   = df[_FLAG_NEURO] == False  # noqa: E712  (bool column)

    if _FLAG_ACC in df.columns:
        n_acc_bad = int(df.loc[base_mask, _FLAG_ACC].sum())
        rows.append(_row(
            step=3,
            section=_SECTION_GLOBAL,
            description="Subjects with poor actigraphy quality (informational)",
            exclusion_criterion=(
                "≥1 of: unreliable device data, failed calibration, "
                "daylight-savings crossover, recording problems (p90002, "
                "p90015–p90018, p90180)"
            ),
            n_before=n_post_neuro,
            n_excluded=n_acc_bad,
            n_after=n_post_neuro - n_acc_bad,
            notes=(
                "acc_bad_quality == True.  These subjects are RETAINED in "
                "the subject-level analytical cohort but their nightly "
                "records are removed during actigraphy night-level QC."
            ),
        ))

    # ── STEP 4+: Per-outcome analytical cohort ────────────────────────────────
    # For each outcome, apply prevalent exclusion on top of neuro exclusion.
    for step_idx, outcome in enumerate(outcomes, start=4):
        lbl = labels.get(outcome, outcome)

        surv_col = _SURV_TEMPLATE.format(outcome=outcome)
        inc_col  = _INC_TEMPLATE.format(outcome=outcome)
        prev_col = _PREV_TEMPLATE.format(outcome=outcome)

        if surv_col not in df.columns:
            rows.append(_row(
                step=float(step_idx),
                section=_SECTION_OUTCOME,
                description=f"Analytical cohort — {lbl}",
                exclusion_criterion="",
                n_before=n_post_excl,
                n_excluded=0,
                n_after=0,
                outcome=outcome,
                notes=f"MISSING: '{surv_col}' column not found.",
            ))
            continue

        # Base = neuro-excluded cohort
        df_base = df.loc[base_mask].copy()
        n_base  = int(df_base[id_col].nunique())

        # Prevalent exclusion: surv_days is NaN for prevalent cases
        mask_prevalent = df_base[surv_col].isna()
        n_prevalent    = int(mask_prevalent.sum())
        df_analytic    = df_base.loc[~mask_prevalent]
        n_analytic     = int(df_analytic[id_col].nunique())

        # Incident cases and median follow-up in the analytical cohort
        n_incident = int(df_analytic[inc_col].sum()) if inc_col in df_analytic.columns else -1
        n_controls = n_analytic - n_incident

        # Median censored follow-up (years) — censored = event-free subjects
        median_fu = np.nan
        if inc_col in df_analytic.columns:
            censored_times = df_analytic.loc[
                df_analytic[inc_col] == 0, surv_col
            ]
            if not censored_times.empty:
                median_fu = round(float(censored_times.median()) / 365.25, 2)

        rows.append(_row(
            step=float(step_idx),
            section=_SECTION_OUTCOME,
            description=f"Analytical cohort — {lbl}",
            exclusion_criterion=(
                f"Prevalent {lbl} at actigraphy baseline "
                f"(diagnosis before wear_time_start)"
            ),
            n_before=n_base,
            n_excluded=n_prevalent,
            n_after=n_analytic,
            outcome=outcome,
            n_incident=n_incident,
            n_prevalent_excluded=n_prevalent,
            median_follow_up_years=median_fu,
            notes=(
                f"Incident cases: {n_incident:,}  |  "
                f"Event-free (controls): {n_controls:,}  |  "
                f"Median censored follow-up: {median_fu:.2f} years"
            ),
        ))

    # ── Assemble table ────────────────────────────────────────────────────────
    df_consort = pd.DataFrame(rows)

    # ── Print ─────────────────────────────────────────────────────────────────
    if verbose:
        display_cols = [
            "step", "description", "n_before",
            "n_excluded", "pct_excluded", "n_after",
        ]
        print("\n" + "=" * 90)
        print("  CONSORT / STROBE PARTICIPANT FLOW TABLE")
        print("=" * 90)
        print(tabulate(
            df_consort[display_cols],
            headers="keys", tablefmt="github", showindex=False, floatfmt=".1f",
        ))
        # Outcome-specific detail
        outcome_rows = df_consort[df_consort["section"] == _SECTION_OUTCOME]
        if not outcome_rows.empty:
            print("\n  Outcome-specific analytical cohorts:")
            detail_cols = [
                "description", "n_after", "n_prevalent_excluded",
                "n_incident_cases", "median_follow_up_years",
            ]
            print(tabulate(
                outcome_rows[detail_cols],
                headers="keys", tablefmt="github", showindex=False, floatfmt=".2f",
            ))
        print("=" * 90 + "\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        df_consort.to_csv(save_dir / "consort_flow_table.csv", index=False)

        try:
            df_consort.to_excel(
                save_dir / "consort_flow_table.xlsx", index=False
            )
        except Exception:
            pass  # openpyxl optional

        print(f"  CONSORT table saved → {save_dir / 'consort_flow_table.csv'}")

    return df_consort
