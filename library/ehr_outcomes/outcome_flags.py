"""
Outcome flagging for PD, AD, and Dementia from UKBB diagnosis fields.

Data sources (prioritised):
  1. First-occurrence fields (data until 2025):
       PD (G20): p131022 (date), p131023 (source)
       AD (G30): p131036 (date), p131037 (source)
  2. HES ICD-10 array fields (data until ~2013, used for DEM/DLB):
       p41270 (codes, pipe-separated) + p41280_a* (dates)

Creates:
    - PD_flag, AD_flag, DEM_flag, DLB_flag
    - Diagnosis dates
    - Composite outcomes 1a-4a
"""

from config.config import outcomes, DISEASE_DATE_COLS
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Optional
from tabulate import tabulate
from library.column_registry import (
    col_dx, col_prevalent, col_incident, col_competing,
    col_tte_days, col_surv_time, col_surv_event,
)


# ---------------------------------------------------------
# LEGACY: First-occurrence field mapping (superseded by DISEASE_DATE_COLS
# in config.config, 2026-03-29).
#
# The old approach merged HES ICD-10 arrays (p41270/p41280) with
# first-occurrence fields (Category 2410).  Outcome dates are now extracted
# directly from the algo-defined (p42xxx) and first-occurrence (p13xxx)
# columns via priority-based selection in _priority_date().
# Retained for reference only — not called by the active pipeline.
# ---------------------------------------------------------
# FIRST_OCCURRENCE_DATE: Dict[str, List[str]] = {
#     "pd":  ["p131022"],
#     "ad":  ["p131036"],
#     "dem": ["p42022", "p42024"],
# }
#
#
# def _get_first_occurrence_date(
#     df: pd.DataFrame,
#     field_prefixes: List[str],
# ) -> pd.Series:
#     """Extract earliest diagnosis date across UKB first-occurrence fields."""
#     all_series: List[pd.Series] = []
#     for prefix in field_prefixes:
#         fo_cols = sorted(
#             c for c in df.columns
#             if c == prefix or c.startswith(f"{prefix}_")
#         )
#         if not fo_cols:
#             continue
#         date_parts = [pd.to_datetime(df[c], errors="coerce") for c in fo_cols]
#         if len(date_parts) == 1:
#             all_series.append(date_parts[0])
#         else:
#             all_series.append(pd.concat(date_parts, axis=1).min(axis=1))
#     if not all_series:
#         return pd.Series(pd.NaT, index=df.index)
#     if len(all_series) == 1:
#         return all_series[0]
#     return pd.concat(all_series, axis=1).min(axis=1)


# ---------------------------------------------------------
# LEGACY: HES ICD-10 row-wise helpers (p41270/p41280).
# Outcomes no longer use HES arrays — they are sourced from algo-defined
# (p42xxx) and first-occurrence (p13xxx) fields via _priority_date().
# These helpers are retained because other pipeline components
# (neuro_exclusion, prodromal covariates via add_alpha_syn_covariates)
# still scan p41270/p41280.
# ---------------------------------------------------------
# def _contains_code(row, codes):
#     for v in row:
#         s = str(v)
#         if any(code in s for code in codes):
#             return True
#     return False
#
#
# def _get_first_dx_date(df, codes, icd_cols, date_cols):
#     suffix_map = {}
#     for col in icd_cols:
#         suffix = col.replace("41270", "")
#         date_col = "41280" + suffix
#         if date_col in date_cols:
#             suffix_map[col] = date_col
#     dates = []
#     for icd_col, dt_col in suffix_map.items():
#         mask = df[icd_col].astype(str).apply(lambda x: any(code in x for code in codes))
#         dt = pd.to_datetime(df.loc[mask, dt_col], errors="coerce")
#         dates.append(dt)
#     if not dates:
#         return pd.Series([pd.NaT] * len(df))
#     all_dates = pd.concat(dates, axis=1)
#     return all_dates.min(axis=1)


# ---------------------------------------------------------
# Priority-based date extraction
# ---------------------------------------------------------

def _priority_date(
    df: pd.DataFrame,
    primary_col: str,
    fallback_col: str,
) -> pd.Series:
    """
    Return the primary date column where available, else the fallback column.

    Priority rule: algo-defined fields (p42xxx, coverage ~2024) are preferred
    because they are adjudicated by UKB.  When a subject has no algo-defined
    date (NaT), the first-occurrence field (p13xxx, coverage ~2025) is used
    to extend temporal coverage.

    Args:
        df: DataFrame containing the date columns.
        primary_col: Algo-defined date column name (e.g. ``"p42032"``).
        fallback_col: First-occurrence date column name (e.g. ``"p131022"``).

    Returns:
        Series of pd.Timestamp — primary date where not NaT, else fallback.
    """
    primary = (
        pd.to_datetime(df[primary_col], errors="coerce")
        if primary_col in df.columns
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )
    fallback = (
        pd.to_datetime(df[fallback_col], errors="coerce")
        if fallback_col in df.columns
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    )
    return primary.where(primary.notna(), fallback)


# ---------------------------------------------------------
# LEGACY: HES-based outcome flagging (superseded 2026-03-29).
# This version scanned p41270/p41280 ICD-10 HES arrays for PD/AD/DEM/DLB.
# Replaced by _priority_date() extraction from algo-defined (p42xxx) and
# first-occurrence (p13xxx) columns.  Retained for reference.
# ---------------------------------------------------------
def _add_outcome_flags_hes(
        df: pd.DataFrame,
        save_dir: Path | None = None,
        verbose: bool = True,
        overwrite: bool = True,
        censor_date: str | pd.Timestamp = "2025-11-30",
) -> pd.DataFrame:
    """
    [LEGACY] ICD-10 HES-based outcome flags.

    Uses:
      - p41270 : ICD-10 codes (pipe-separated, array-aligned)
      - p41280_a* : date of first occurrence for each ICD-10 code

    Outcomes:
      PD / AD / DEM / DLB (base)
      Composite outcomes (1a–4a)

    Survival variables per outcome:
      _diagnosed   bool   has the disease and a recorded date
      _prevalent   bool   diagnosed before wear_time_start
      _incident    bool   diagnosed within [start, censor_date]
      _competing   bool   died during follow-up without the outcome (surv_event=2)
      _tte_days    int    days from start to diagnosis (NaN for prevalent)
      _surv_time   float  days used in Cox/KM (NaN for prevalent)
      _surv_event  int    0=censored, 1=event, 2=competing event (death)

    Additional columns:
      death_date   Timestamp   earliest p40000_* date (NaT if no death recorded)
      death_flag   bool        True when death_date is not NaT
    """

    # ---------------------------------------------------------
    # Outcome definitions
    # ---------------------------------------------------------
    path_file = save_dir.joinpath("2_neuro_exclude.parquet")
    if not overwrite and path_file.exists():
        df_report = pd.read_csv(save_dir / "outcome_summary.csv")

        df = pd.read_parquet(path_file)

        print("\n" + "=" * 100)
        print("OUTCOME REPORT")
        print("=" * 100)
        print(tabulate(df_report, headers="keys", tablefmt="github", showindex=False))
        print("=" * 100)
        return df

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------
    df = df.copy()
    df["wear_time_start"] = pd.to_datetime(df["wear_time_start"], errors="coerce")
    if isinstance(censor_date, str):
        censor_date = pd.to_datetime(censor_date, errors="coerce")

    # ---------------------------------------------------------
    # Study admin end: max diagnosis date across ALL outcome date columns,
    # capped at the configured ceiling.
    # Flatten FIRST_OCCURRENCE_DATE (values are now lists of prefixes).
    # ---------------------------------------------------------
    _fo_prefixes_flat = [p for prefixes in FIRST_OCCURRENCE_DATE.values() for p in prefixes]
    _outcome_date_prefixes = _fo_prefixes_flat + ["p41280"]
    _outcome_date_cols = [
        c for c in df.columns
        if any(c == p or c.startswith(f"{p}_") for p in _outcome_date_prefixes)
    ]

    if _outcome_date_cols:
        data_max = pd.to_datetime(
            pd.concat(
                [pd.to_datetime(df[c], errors="coerce") for c in _outcome_date_cols],
                axis=1,
            ).max().max()
        )
        study_admin_end: pd.Timestamp = min(data_max, censor_date)
    else:
        study_admin_end = censor_date

    if verbose:
        print(f"[CENSOR] Study admin end (data max capped at config): {study_admin_end.date()}")

    # ---------------------------------------------------------
    # Death date — needed before per-patient censor computation.
    # Field 40000: date of death (p40000_i0, p40000_i1, …)
    # ---------------------------------------------------------
    death_cols = sorted([c for c in df.columns if c.startswith("p40000")])
    if death_cols:
        df["death_date"] = pd.to_datetime(
            pd.concat(
                [pd.to_datetime(df[c], errors="coerce") for c in death_cols],
                axis=1,
            ).min(axis=1)
        )
    else:
        df["death_date"] = pd.NaT

    df["death_flag"] = df["death_date"].notna()

    # ---------------------------------------------------------
    # Per-patient censor date
    #   patient_followup_end_i = min(non-null: death_date_i, study_admin_end)
    # Subjects who die before the admin end have their follow-up truncated.
    # study_admin_end is a scalar; death_date is per-patient.
    # ---------------------------------------------------------
    df["censor_date"] = (
        df["death_date"]
        .clip(upper=study_admin_end)       # if death_date > admin_end, cap at admin_end
        .fillna(study_admin_end)           # non-dead subjects: censor at admin_end
    )

    # Follow-up measured from actigraphy start date.
    df["follow_up_days"]  = (df["censor_date"] - df["wear_time_start"]).dt.days
    df["follow_up_years"] = df["follow_up_days"] / 365.25

    if verbose:
        n_dead = int(df["death_flag"].sum())
        died_in_followup = int(
            (
                df["death_flag"]
                & (df["death_date"] > df["wear_time_start"])
                & (df["death_date"] <= df["censor_date"])
            ).sum()
        )
        pct = round(100 * died_in_followup / len(df), 2)
        if death_cols:
            print(f"[DEATH] Total recorded deaths: {n_dead:,}")
            print(f"[DEATH] Deaths within follow-up window: {died_in_followup:,} ({pct}%)")
        else:
            print(
                "[DEATH] Warning: no p40000_* death-date columns found; "
                "competing-risk censoring disabled."
            )

    # ---------------------------------------------------------
    # Helper: extract first diagnosis date for a set of ICD-10 codes
    # ---------------------------------------------------------
    def _get_first_dx_date(row, codes_of_interest: set[str]) -> pd.Timestamp:
        if pd.isna(row["p41270"]):
            return pd.NaT

        codes = row["p41270"].split("|")
        dates = []

        for i, code in enumerate(codes):
            if code in codes_of_interest:
                date_col = f"p41280_a{i}"
                if date_col in row and pd.notna(row[date_col]):
                    dates.append(row[date_col])

        return min(dates) if dates else pd.NaT

    # ---------------------------------------------------------
    # Base disease flags + dates
    #
    # Merging strategy (applied to every disease):
    #   1. Extract first-occurrence date (UKBB Category 2410, data until 2025)
    #      if a field mapping exists in FIRST_OCCURRENCE_DATE.
    #   2. Extract HES ICD-10 date (p41270/p41280, data until ~2023).
    #   3. Assign dx_date = min(fo_date, hes_date), ignoring NaT.
    #
    # Why merge instead of choosing one source?
    # ------------------------------------------
    # First-occurrence fields have richer longitudinal coverage (to 2025) but
    # may miss cases not yet adjudicated. HES provides ICD-10-coded records up
    # to ~2023 and may capture earlier diagnoses. Taking the earliest date from
    # either source maximises sensitivity, extends the follow-up window for
    # dementia outcomes (which previously terminated at 2023-03-30 due to HES
    # ceiling), and ensures consistent treatment across diseases.
    #
    # DLB has no dedicated first-occurrence field in Category 2410 → HES only.
    # For PD and AD the first-occurrence fields already aggregate multiple
    # registries (including HES), so merging is redundant but harmless
    # (min will always select the first-occurrence date or earlier).
    # ---------------------------------------------------------
    _dx_source: Dict[str, str] = {}  # tracks which source(s) were used

    for disease, codes in outcomes_codes.items():
        d = disease.lower()
        codes_set = set(codes)

        dx_date_col = f"{d}_dx_date"
        flag_col = f"{d}_flag"

        # --- Source 1: First-occurrence date (if field mapping defined) ---
        fo_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        fo_prefixes: Optional[List[str]] = FIRST_OCCURRENCE_DATE.get(d)
        fo_used: Optional[str] = None

        if fo_prefixes is not None:
            fo_cols = [
                c for c in df.columns
                if any(c == p or c.startswith(f"{p}_") for p in fo_prefixes)
            ]
            if fo_cols:
                fo_date = _get_first_occurrence_date(df, fo_prefixes)
                fo_used = f"first_occurrence ({', '.join(fo_prefixes)})"
                if verbose:
                    n_fo = int(fo_date.notna().sum())
                    print(
                        f"[{d.upper()}] first-occurrence {fo_prefixes}: "
                        f"{n_fo:,} cases with date"
                    )
            else:
                if verbose:
                    print(
                        f"[{d.upper()}] first-occurrence {fo_prefixes} "
                        f"not found in columns — using HES only"
                    )

        # --- Source 2: HES ICD-10 row-wise extraction ---
        hes_date = pd.to_datetime(
            df.apply(
                _get_first_dx_date,
                axis=1,
                codes_of_interest=codes_set,
            ),
            errors="coerce",
        )
        if verbose:
            n_hes = int(hes_date.notna().sum())
            print(f"[{d.upper()}] HES p41270/p41280: {n_hes:,} cases with date")

        # --- Merge: earliest date from any available source ---
        # pd.concat min on axis=1 ignores NaT (treats as +inf), so a date
        # from either source is always preferred over NaT.
        df[dx_date_col] = pd.concat(
            [fo_date.rename("fo"), hes_date.rename("hes")], axis=1
        ).min(axis=1)

        df[flag_col] = df[dx_date_col].notna()

        # Source attribution for the diagnostic report
        if fo_used is not None:
            _dx_source[d] = f"merged ({fo_used} + hes p41270/p41280)"
        else:
            _dx_source[d] = "hes (p41270/p41280)"

        if verbose:
            n_merged = int(df[flag_col].sum())
            latest = df[dx_date_col].max()
            print(
                f"[{d.upper()}] merged total: {n_merged:,} cases "
                f"(latest dx: {latest.date() if pd.notna(latest) else 'N/A'})"
            )

    # ---------------------------------------------------------
    # Composite outcomes (flags)
    # ---------------------------------------------------------
    df["outcome_1a_pd_only"] = df["pd_flag"] & ~df["ad_flag"] & ~df["dem_flag"]
    df["outcome_1b_pd_ad"] = df["pd_flag"] & df["ad_flag"]
    df["outcome_2a_vasculardementia"] = df["dem_flag"] & ~df["pd_flag"] & ~df["ad_flag"]
    df["outcome_2b_pd_vasculardementia"] = df["pd_flag"] & df["dem_flag"] & ~df["ad_flag"]
    # df["outcome_3a_dlb_only"] = df["dlb_flag"] & ~df["pd_flag"]
    df["outcome_4a_ad_only"] = ~df["pd_flag"] & df["ad_flag"] & ~df["dem_flag"]

    # ---------------------------------------------------------
    # Composite outcome dates
    # ---------------------------------------------------------
    df["outcome_1a_pd_only_date"] = df["pd_dx_date"]
    df["outcome_1b_pd_ad_date"] = df[["pd_dx_date", "ad_dx_date"]].min(axis=1)
    df["outcome_2a_vasculardementia_date"] = df["dem_dx_date"]
    df["outcome_2b_pd_vasculardementia_date"] = df[["pd_dx_date", "dem_dx_date"]].min(axis=1)
    # df["outcome_3a_dlb_only_date"] = df["dlb_dx_date"]
    df["outcome_4a_ad_only_date"] = df["ad_dx_date"]

    # ---------------------------------------------------------
    # Survival builder
    # ---------------------------------------------------------
    def build_survival(
            df: pd.DataFrame,
            outcome_col: str,
            date_col: str,
            start_col: str = "wear_time_start",
            censor_col: str = "censor_date",
            death_date_col: str = "death_date",
    ) -> None:
        """
        Add survival columns in-place for a single outcome.

        surv_event encoding
        -------------------
        0  administratively censored (alive, no outcome by censor_date)
        1  incident case (outcome occurred within [start, censor])
        2  competing event (death without the outcome, within [start, censor])

        Prevalent cases (diagnosed before wear_time_start) receive NaN for
        surv_event and surv_time and are excluded from downstream analysis.

        Bug fixes vs prior version
        --------------------------
        * Incident definition now enforces date_col <= censor_col, so
          post-censor diagnoses are NOT counted as events.
        * Death-based competing censoring: subjects who die during follow-up
          before the outcome receive surv_event=2 rather than surv_event=0.
        """
        diagnosed    = col_dx(outcome_col)
        prevalent    = col_prevalent(outcome_col)
        incident     = col_incident(outcome_col)
        competing    = col_competing(outcome_col)
        tte          = col_tte_days(outcome_col)
        surv_time    = col_surv_time(outcome_col)
        surv_event   = col_surv_event(outcome_col)

        # --- diagnosed: flag is True AND a date is recorded ---
        df[diagnosed] = df[outcome_col] & df[date_col].notna()

        # --- prevalent: diagnosed BEFORE follow-up start ---
        df[prevalent] = df[diagnosed] & (df[date_col] < df[start_col])

        # --- incident: diagnosed ON OR AFTER start AND ON OR BEFORE censor ---
        # (post-censor diagnoses are excluded; they become competing/admin-censored)
        df[incident] = (
            df[diagnosed]
            & (df[date_col] >= df[start_col])
            & (df[date_col] <= df[censor_col])
        )

        # --- competing: death during follow-up without the outcome (default False) ---
        df[competing] = False

        # --- TTE from start to diagnosis date (NaN for prevalent) ---
        df[tte] = (df[date_col] - df[start_col]).dt.days
        df.loc[df[prevalent], tte] = np.nan

        # --- Initialise survival columns ---
        df[surv_event] = np.nan
        df[surv_time]  = np.nan

        # Event = 1: incident cases
        df.loc[df[incident], surv_event] = 1
        df.loc[df[incident], surv_time]  = df.loc[df[incident], tte]

        # Subjects not yet classified: not prevalent, not incident
        non_case = (~df[prevalent]) & (~df[incident])

        start_s  = df[start_col]
        censor_s = df[censor_col]
        censor_days = (censor_s - start_s).dt.days

        use_competing = (
            death_date_col in df.columns
            and df[death_date_col].notna().any()
        )

        if use_competing:
            death_s = df[death_date_col]
            # Death must occur: after wear start (strict >), within follow-up
            died_during = (
                non_case
                & death_s.notna()
                & (death_s > start_s)
                & (death_s <= censor_s)
            )
            df.loc[died_during, surv_event] = 2
            df.loc[died_during, surv_time]  = (
                death_s[died_during] - start_s[died_during]
            ).dt.days
            df.loc[died_during, competing]  = True

            admin_censored = non_case & ~died_during
        else:
            admin_censored = non_case

        # Event = 0: administratively censored
        df.loc[admin_censored, surv_event] = 0
        df.loc[admin_censored, surv_time]  = censor_days[admin_censored]

    # ---------------------------------------------------------
    # Survival for base diseases
    # ---------------------------------------------------------
    for disease in outcomes_codes.keys():
        d = disease.lower()
        build_survival(
            df=df,
            outcome_col=f"{d}_flag",
            date_col=f"{d}_dx_date",
        )

    # ---------------------------------------------------------
    # Survival for composite outcomes
    # ---------------------------------------------------------
    # outcome_5a_pd_med requires medication flags (Step 5) and is
    # handled separately by add_medication_confirmed_outcomes().
    _DEFERRED_OUTCOMES = {"outcome_5a_pd_med"}

    for oc in outcomes:
        if oc in _DEFERRED_OUTCOMES:
            continue
        build_survival(
            df=df,
            outcome_col=oc,
            date_col=f"{oc}_date",
        )

    def _report_outcomes(
            df: pd.DataFrame,
            base_outcomes: List[str],
            composite_outcomes: List[str],
            verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Generate an outcome report for base (PD/AD/DEM/DLB) and composite outcomes.

        Assumes the following columns already exist:
          - <outcome>_flag
          - <outcome>_dx_date
          - <outcome>_prevalent
          - <outcome>_incident
          - <outcome>_tte_days

        or for composites:
          - <outcome>
          - <outcome>_date
          - <outcome>_prevalent
          - <outcome>_incident
          - <outcome>_tte_days
        """

        rows = []

        # ------------------------
        # Base outcomes (pd, ad...)
        # build_survival() is called with outcome_col=f"{d}_flag", so the
        # generated column names carry the "_flag_" infix.
        # ------------------------
        for d in base_outcomes:
            flag_col     = f"{d}_flag"
            date_col     = f"{d}_dx_date"
            prev_col     = col_prevalent(f"{d}_flag")
            inc_col      = col_incident(f"{d}_flag")
            tte_col      = col_tte_days(f"{d}_flag")
            competing_col = col_competing(f"{d}_flag")

            if not all(c in df.columns for c in [flag_col, date_col, prev_col, inc_col]):
                continue

            n_diag = int(df[flag_col].sum())
            n_prev = int(df[prev_col].sum())
            n_inc  = int(df[inc_col].sum())
            n_comp = int(df[competing_col].sum()) if competing_col in df.columns else 0

            rows.append({
                "outcome": d,
                "type": "base",
                "source": _dx_source.get(d, "unknown"),
                "diagnosed_n": n_diag,
                "prevalent_n": n_prev,
                "incident_n": n_inc,
                "competing_n": n_comp,
                "incident_pct": round(100 * n_inc / len(df), 3),
                "median_tte_days": df.loc[df[inc_col], tte_col].median(),
                "earliest_dx": df[date_col].min(),
                "latest_dx": df[date_col].max(),
            })

        # ------------------------
        # Composite outcomes
        # ------------------------
        for o in composite_outcomes:
            date_col     = f"{o}_date"
            prev_col     = col_prevalent(o)
            inc_col      = col_incident(o)
            tte_col      = col_tte_days(o)
            competing_col = col_competing(o)

            if not all(c in df.columns for c in [o, date_col, prev_col, inc_col]):
                continue

            n_diag = int(df[o].sum())
            n_prev = int(df[prev_col].sum())
            n_inc  = int(df[inc_col].sum())
            n_comp = int(df[competing_col].sum()) if competing_col in df.columns else 0

            rows.append({
                "outcome": o,
                "type": "composite",
                "source": "mixed",
                "diagnosed_n": n_diag,
                "prevalent_n": n_prev,
                "incident_n": n_inc,
                "competing_n": n_comp,
                "incident_pct": round(100 * n_inc / len(df), 3),
                "median_tte_days": df.loc[df[inc_col], tte_col].median(),
                "earliest_dx": df[date_col].min(),
                "latest_dx": df[date_col].max(),
            })

        report_df = pd.DataFrame(rows).sort_values(
            ["type", "outcome"]
        )

        if verbose:
            print("\n" + "=" * 100)
            print("OUTCOME REPORT")
            print("=" * 100)
            print(tabulate(report_df, headers="keys", tablefmt="github", showindex=False))
            print("=" * 100)

        return report_df

    base_outcomes = [k.lower() for k in outcomes_codes.keys()]

    df_report = _report_outcomes(
        df=df,
        base_outcomes=base_outcomes,
        composite_outcomes=[oc for oc in outcomes if oc not in _DEFERRED_OUTCOMES],
    )

    # ---------------------------------------------------------
    # Censoring date diagnostic
    # Checks whether the latest recorded diagnoses are close to
    # the administrative censor_date.  A gap > 1 year may indicate
    # stale data or an incorrect censor_date parameter.
    # ---------------------------------------------------------
    def _check_censoring_date(
            report_df: pd.DataFrame,
            censor_date_val: pd.Timestamp,
            warn_threshold_days: int = 365,
    ) -> pd.DataFrame:
        """
        Per-outcome gap between latest recorded diagnosis and censor_date.

        Returns a DataFrame with columns:
            outcome, latest_dx, censor_date, gap_days, stale_flag
        stale_flag = True when gap_days > warn_threshold_days.
        """
        rows_diag = []
        for _, row in report_df.iterrows():
            latest_dx = row.get("latest_dx")
            if pd.isna(latest_dx) or latest_dx is None:
                gap = np.nan
                stale = False
            else:
                gap   = int((censor_date_val - pd.Timestamp(latest_dx)).days)
                stale = gap > warn_threshold_days

            if stale and verbose:
                print(
                    f"[WARN][CENSOR] '{row['outcome']}': latest_dx={latest_dx}, "
                    f"censor_date={censor_date_val.date()}, "
                    f"gap={gap} days (>{warn_threshold_days}). "
                    "Check data freshness or censor_date parameter."
                )
            rows_diag.append({
                "outcome":     row["outcome"],
                "latest_dx":   latest_dx,
                "censor_date": censor_date_val.date(),
                "gap_days":    gap,
                "stale_flag":  stale,
            })
        return pd.DataFrame(rows_diag)

    df_censor_diag = _check_censoring_date(
        report_df=df_report,
        censor_date_val=study_admin_end,
    )

    if verbose:
        print("\n" + "=" * 100)
        print("CENSORING DATE DIAGNOSTIC")
        print("=" * 100)
        print(tabulate(df_censor_diag, headers="keys", tablefmt="github", showindex=False))
        print("=" * 100)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        df_report.to_csv(save_dir / "outcome_summary.csv", index=False)
        df_censor_diag.to_csv(save_dir / "censor_diagnostic.csv", index=False)

        df.to_parquet(path_file, index=False)
        if verbose:
            print(f"Saved outcomes dataset")

    return df





# =============================================================================
# ACTIVE: Priority-based outcome flagging (2026-03-29)
# =============================================================================

def add_outcome_flags(
        df: pd.DataFrame,
        save_dir: Path | None = None,
        verbose: bool = True,
        overwrite: bool = True,
        censor_date: str | pd.Timestamp = "2025-11-30",
) -> pd.DataFrame:
    """
    Add outcome flags, first diagnosis dates, composite outcomes,
    and survival variables for UK Biobank.

    Data sources (priority-based per disease, defined in config.DISEASE_DATE_COLS):
      Primary : algo-defined UKB fields (p42xxx, adjudicated, coverage ~2024).
      Fallback: first-occurrence ICD-10 fields (p13xxx, coverage ~2025).
    Priority rule: use primary date if not NaT; else fallback.

    Base diseases: PD (pd), AD (ad), Vascular Dementia (dem).

    Composite outcomes:
      outcome_1a_pd_only            PD only (no AD, no DEM)
      outcome_1b_pd_ad              PD and AD
      outcome_2a_vasculardementia   DEM only (no PD, no AD)
      outcome_2b_pd_vasculardementia PD and DEM (no AD)
      outcome_4a_ad_only            AD only (no PD, no DEM)

    Survival variables per outcome:
      _diagnosed   bool   has the disease and a recorded date
      _prevalent   bool   diagnosed before wear_time_start
      _incident    bool   diagnosed within [start, censor_date]
      _competing   bool   died during follow-up without the outcome
      _tte_days    int    days from start to diagnosis (NaN for prevalent)
      _surv_time   float  days used in Cox/KM (NaN for prevalent)
      _surv_event  int    0=censored, 1=event, 2=competing event (death)

    Additional columns:
      death_date   Timestamp   earliest p40000_* date (NaT if no death recorded)
      death_flag   bool        True when death_date is not NaT

    Args:
        df: DataFrame with wear_time_start, DISEASE_DATE_COLS fields, p40000_* columns.
        save_dir: Directory for output parquet and CSV logs.
        verbose: Print diagnostic counts.
        overwrite: If False and output parquet exists, load and return it.
        censor_date: Administrative censoring ceiling (inclusive).

    Returns:
        DataFrame with all outcome, survival, and diagnosis columns appended.
    """
    path_file = save_dir.joinpath("2_neuro_exclude.parquet")
    if not overwrite and path_file.exists():
        df_report = pd.read_csv(save_dir / "outcome_summary.csv")
        df = pd.read_parquet(path_file)
        print("\n" + "=" * 100)
        print("OUTCOME REPORT")
        print("=" * 100)
        print(tabulate(df_report, headers="keys", tablefmt="github", showindex=False))
        print("=" * 100)
        return df

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------
    df = df.copy()
    df["wear_time_start"] = pd.to_datetime(df["wear_time_start"], errors="coerce")
    if isinstance(censor_date, str):
        censor_date = pd.to_datetime(censor_date, errors="coerce")

    # ---------------------------------------------------------
    # Study admin end: max date across all DISEASE_DATE_COLS columns,
    # capped at the configured ceiling.
    # ---------------------------------------------------------
    _all_outcome_date_cols = [
        col
        for cols in DISEASE_DATE_COLS.values()
        for col in (cols["primary"], cols["fallback"])
        if col in df.columns
    ]
    if _all_outcome_date_cols:
        data_max = pd.to_datetime(
            pd.concat(
                [pd.to_datetime(df[c], errors="coerce") for c in _all_outcome_date_cols],
                axis=1,
            ).max().max()
        )
        study_admin_end: pd.Timestamp = min(data_max, censor_date)
    else:
        study_admin_end = censor_date

    if verbose:
        print(f"[CENSOR] Study admin end (data max capped at config): {study_admin_end.date()}")

    # ---------------------------------------------------------
    # Death date — needed before per-patient censor computation.
    # ---------------------------------------------------------
    death_cols = sorted([c for c in df.columns if c.startswith("p40000")])
    if death_cols:
        df["death_date"] = pd.to_datetime(
            pd.concat(
                [pd.to_datetime(df[c], errors="coerce") for c in death_cols],
                axis=1,
            ).min(axis=1)
        )
    else:
        df["death_date"] = pd.NaT

    df["death_flag"] = df["death_date"].notna()

    # ---------------------------------------------------------
    # Per-patient censor date
    # ---------------------------------------------------------
    df["censor_date"] = (
        df["death_date"]
        .clip(upper=study_admin_end)
        .fillna(study_admin_end)
    )

    df["follow_up_days"] = (df["censor_date"] - df["wear_time_start"]).dt.days
    df["follow_up_years"] = df["follow_up_days"] / 365.25

    if verbose:
        n_dead = int(df["death_flag"].sum())
        died_in_followup = int(
            (
                df["death_flag"]
                & (df["death_date"] > df["wear_time_start"])
                & (df["death_date"] <= df["censor_date"])
            ).sum()
        )
        pct = round(100 * died_in_followup / len(df), 2)
        if death_cols:
            print(f"[DEATH] Total recorded deaths: {n_dead:,}")
            print(f"[DEATH] Deaths within follow-up window: {died_in_followup:,} ({pct}%)")
        else:
            print(
                "[DEATH] Warning: no p40000_* death-date columns found; "
                "competing-risk censoring disabled."
            )

    # ---------------------------------------------------------
    # Base disease flags + dates (priority-based extraction)
    #
    # For each disease defined in DISEASE_DATE_COLS:
    #   dx_date = primary if primary is not NaT, else fallback.
    # No ICD-10 HES scanning is performed here.
    # ---------------------------------------------------------
    _dx_source: Dict[str, str] = {}

    for d, cols in DISEASE_DATE_COLS.items():
        dx_date_col = f"{d}_dx_date"
        flag_col = f"{d}_flag"

        df[dx_date_col] = _priority_date(df, cols["primary"], cols["fallback"])
        df[flag_col] = df[dx_date_col].notna()

        if verbose:
            n_primary = int(
                pd.to_datetime(df.get(cols["primary"]), errors="coerce").notna().sum()
                if cols["primary"] in df.columns else 0
            )
            n_fallback_only = int(
                (
                    pd.to_datetime(df.get(cols["fallback"]), errors="coerce").notna()
                    & pd.to_datetime(df.get(cols["primary"]), errors="coerce").isna()
                ).sum()
                if cols["fallback"] in df.columns else 0
            )
            n_total = int(df[flag_col].sum())
            latest = df[dx_date_col].max()
            print(
                f"[{d.upper()}] primary ({cols['primary']}): {n_primary:,}  "
                f"fallback-only ({cols['fallback']}): {n_fallback_only:,}  "
                f"total: {n_total:,}  "
                f"(latest dx: {latest.date() if pd.notna(latest) else 'N/A'})"
            )
            _dx_source[d] = (
                f"priority: {cols['primary']} (primary) / {cols['fallback']} (fallback)"
            )

    # ---------------------------------------------------------
    # Composite outcome flags
    # ---------------------------------------------------------
    df["outcome_1a_pd_only"] = df["pd_flag"] & ~df["ad_flag"] & ~df["dem_flag"]
    df["outcome_1b_pd_ad"] = df["pd_flag"] & df["ad_flag"]
    df["outcome_2a_vasculardementia"] = df["dem_flag"] & ~df["pd_flag"] & ~df["ad_flag"]
    df["outcome_2b_pd_vasculardementia"] = df["pd_flag"] & df["dem_flag"] & ~df["ad_flag"]
    df["outcome_4a_ad_only"] = ~df["pd_flag"] & df["ad_flag"] & ~df["dem_flag"]

    # ---------------------------------------------------------
    # Composite outcome dates
    # ---------------------------------------------------------
    df["outcome_1a_pd_only_date"] = df["pd_dx_date"]
    df["outcome_1b_pd_ad_date"] = df[["pd_dx_date", "ad_dx_date"]].min(axis=1)
    df["outcome_2a_vasculardementia_date"] = df["dem_dx_date"]
    df["outcome_2b_pd_vasculardementia_date"] = df[["pd_dx_date", "dem_dx_date"]].min(axis=1)
    df["outcome_4a_ad_only_date"] = df["ad_dx_date"]

    # ---------------------------------------------------------
    # Survival builder
    # ---------------------------------------------------------
    def build_survival(
            df: pd.DataFrame,
            outcome_col: str,
            date_col: str,
            start_col: str = "wear_time_start",
            censor_col: str = "censor_date",
            death_date_col: str = "death_date",
    ) -> None:
        """
        Add survival columns in-place for a single outcome.

        surv_event encoding
        -------------------
        0  administratively censored (alive, no outcome by censor_date)
        1  incident case (outcome occurred within [start, censor])
        2  competing event (death without the outcome, within [start, censor])

        Prevalent cases (diagnosed before wear_time_start) receive NaN for
        surv_event and surv_time and are excluded from downstream analysis.
        """
        diagnosed  = col_dx(outcome_col)
        prevalent  = col_prevalent(outcome_col)
        incident   = col_incident(outcome_col)
        competing  = col_competing(outcome_col)
        tte        = col_tte_days(outcome_col)
        surv_time  = col_surv_time(outcome_col)
        surv_event = col_surv_event(outcome_col)

        df[diagnosed] = df[outcome_col] & df[date_col].notna()
        df[prevalent] = df[diagnosed] & (df[date_col] < df[start_col])
        df[incident] = (
            df[diagnosed]
            & (df[date_col] >= df[start_col])
            & (df[date_col] <= df[censor_col])
        )
        df[competing] = False
        df[tte] = (df[date_col] - df[start_col]).dt.days
        df.loc[df[prevalent], tte] = np.nan

        df[surv_event] = np.nan
        df[surv_time] = np.nan

        df.loc[df[incident], surv_event] = 1
        df.loc[df[incident], surv_time] = df.loc[df[incident], tte]

        non_case = (~df[prevalent]) & (~df[incident])
        start_s  = df[start_col]
        censor_s = df[censor_col]
        censor_days = (censor_s - start_s).dt.days

        use_competing = (
            death_date_col in df.columns
            and df[death_date_col].notna().any()
        )

        if use_competing:
            death_s = df[death_date_col]
            died_during = (
                non_case
                & death_s.notna()
                & (death_s > start_s)
                & (death_s <= censor_s)
            )
            df.loc[died_during, surv_event] = 2
            df.loc[died_during, surv_time] = (
                death_s[died_during] - start_s[died_during]
            ).dt.days
            df.loc[died_during, competing] = True
            admin_censored = non_case & ~died_during
        else:
            admin_censored = non_case

        df.loc[admin_censored, surv_event] = 0
        df.loc[admin_censored, surv_time] = censor_days[admin_censored]

    # ---------------------------------------------------------
    # Survival for base disease flags
    # ---------------------------------------------------------
    for d in DISEASE_DATE_COLS.keys():
        build_survival(
            df=df,
            outcome_col=f"{d}_flag",
            date_col=f"{d}_dx_date",
        )

    # ---------------------------------------------------------
    # Survival for composite outcomes
    # ---------------------------------------------------------
    for oc in outcomes:
        build_survival(
            df=df,
            outcome_col=oc,
            date_col=f"{oc}_date",
        )

    def _report_outcomes(
            df: pd.DataFrame,
            base_outcomes: List[str],
            composite_outcomes: List[str],
            verbose: bool = True,
    ) -> pd.DataFrame:
        """Generate outcome report for base (pd/ad/dem) and composite outcomes."""
        rows = []

        for d in base_outcomes:
            flag_col      = f"{d}_flag"
            date_col      = f"{d}_dx_date"
            prev_col      = col_prevalent(flag_col)
            inc_col       = col_incident(flag_col)
            tte_col       = col_tte_days(flag_col)
            competing_col = col_competing(flag_col)

            if not all(c in df.columns for c in [flag_col, date_col, prev_col, inc_col]):
                continue

            rows.append({
                "outcome":       d,
                "type":          "base",
                "source":        _dx_source.get(d, "unknown"),
                "diagnosed_n":   int(df[flag_col].sum()),
                "prevalent_n":   int(df[prev_col].sum()),
                "incident_n":    int(df[inc_col].sum()),
                "competing_n":   int(df[competing_col].sum()) if competing_col in df.columns else 0,
                "incident_pct":  round(100 * df[inc_col].sum() / len(df), 3),
                "median_tte_days": df.loc[df[inc_col], tte_col].median(),
                "earliest_dx":   df[date_col].min(),
                "latest_dx":     df[date_col].max(),
            })

        for o in composite_outcomes:
            date_col      = f"{o}_date"
            prev_col      = col_prevalent(o)
            inc_col       = col_incident(o)
            tte_col       = col_tte_days(o)
            competing_col = col_competing(o)

            if not all(c in df.columns for c in [o, date_col, prev_col, inc_col]):
                continue

            rows.append({
                "outcome":       o,
                "type":          "composite",
                "source":        "mixed",
                "diagnosed_n":   int(df[o].sum()),
                "prevalent_n":   int(df[prev_col].sum()),
                "incident_n":    int(df[inc_col].sum()),
                "competing_n":   int(df[competing_col].sum()) if competing_col in df.columns else 0,
                "incident_pct":  round(100 * df[inc_col].sum() / len(df), 3),
                "median_tte_days": df.loc[df[inc_col], tte_col].median(),
                "earliest_dx":   df[date_col].min(),
                "latest_dx":     df[date_col].max(),
            })

        report_df = pd.DataFrame(rows).sort_values(["type", "outcome"])

        if verbose:
            print("\n" + "=" * 100)
            print("OUTCOME REPORT")
            print("=" * 100)
            print(tabulate(report_df, headers="keys", tablefmt="github", showindex=False))
            print("=" * 100)

        return report_df

    df_report = _report_outcomes(
        df=df,
        base_outcomes=list(DISEASE_DATE_COLS.keys()),
        composite_outcomes=list(outcomes),
    )

    def _check_censoring_date(
            report_df: pd.DataFrame,
            censor_date_val: pd.Timestamp,
            warn_threshold_days: int = 365,
    ) -> pd.DataFrame:
        """Per-outcome gap between latest recorded diagnosis and censor_date."""
        rows_diag = []
        for _, row in report_df.iterrows():
            latest_dx = row.get("latest_dx")
            if pd.isna(latest_dx) or latest_dx is None:
                gap = np.nan
                stale = False
            else:
                gap   = int((censor_date_val - pd.Timestamp(latest_dx)).days)
                stale = gap > warn_threshold_days

            if stale and verbose:
                print(
                    f"[WARN][CENSOR] '{row['outcome']}': latest_dx={latest_dx}, "
                    f"censor_date={censor_date_val.date()}, "
                    f"gap={gap} days (>{warn_threshold_days}). "
                    "Check data freshness or censor_date parameter."
                )
            rows_diag.append({
                "outcome":     row["outcome"],
                "latest_dx":   latest_dx,
                "censor_date": censor_date_val.date(),
                "gap_days":    gap,
                "stale_flag":  stale,
            })
        return pd.DataFrame(rows_diag)

    df_censor_diag = _check_censoring_date(
        report_df=df_report,
        censor_date_val=study_admin_end,
    )

    if verbose:
        print("\n" + "=" * 100)
        print("CENSORING DATE DIAGNOSTIC")
        print("=" * 100)
        print(tabulate(df_censor_diag, headers="keys", tablefmt="github", showindex=False))
        print("=" * 100)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        df_report.to_csv(save_dir / "outcome_summary.csv", index=False)
        df_censor_diag.to_csv(save_dir / "censor_diagnostic.csv", index=False)
        df.to_parquet(path_file, index=False)
        if verbose:
            print(f"Saved outcomes dataset")

    return df


# =============================================================================
# LEGACY: medication-confirmed PD outcome (not called by active pipeline)
# Medication reporting is sparse in UKBB and does not add analytic value.
# Retained for reference.  Do not call; use add_outcome_flags() instead.
# =============================================================================
def _add_medication_confirmed_outcomes_legacy(
    df: pd.DataFrame,
    save_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add high-specificity PD outcome requiring both G20 diagnosis AND PD medication.

    Must be called AFTER ``add_medication_flags()`` has created the
    ``med_pd_medications`` and ``med_pd_medications_date`` columns.

    Creates outcome_5a_pd_med with all standard survival sub-columns:
        __dx, __prevalent, __incident, __competing,
        __tte_days, __surv_days, __surv_event

    Case definition
    ---------------
    outcome_5a_pd_med = pd_flag AND med_pd_medications AND NOT ad_flag AND NOT dem_flag

    Diagnosis date: pd_dx_date (the ICD-10/first-occurrence date, NOT the
    medication report date).  The medication is a confirmation filter only.

    Args:
        df: DataFrame with outcome flags (pd_flag, ad_flag, dem_flag) and
            medication flags (med_pd_medications, med_pd_medications_date).
        save_dir: Optional directory to save a validation log.
        verbose: Print diagnostic counts.

    Returns:
        DataFrame with outcome_5a_pd_med columns added.
    """
    required_cols = ["pd_flag", "ad_flag", "dem_flag", "med_pd_medications",
                     "wear_time_start", "censor_date", "pd_dx_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for outcome_5a_pd_med: {missing}. "
            "Ensure add_outcome_flags() and add_medication_flags() have run."
        )

    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Composite flag + date
    # ------------------------------------------------------------------
    df["outcome_5a_pd_med"] = (
        df["pd_flag"]
        & df["med_pd_medications"]
        & ~df["ad_flag"]
        & ~df["dem_flag"]
    )
    df["outcome_5a_pd_med_date"] = df["pd_dx_date"]

    # ------------------------------------------------------------------
    # 2. Survival sub-columns (reuses the same build_survival logic)
    # ------------------------------------------------------------------
    outcome = "outcome_5a_pd_med"
    date_col = "outcome_5a_pd_med_date"
    start_col = "wear_time_start"
    censor_col = "censor_date"
    death_date_col = "death_date"

    diagnosed = col_dx(outcome)
    prevalent = col_prevalent(outcome)
    incident = col_incident(outcome)
    competing = col_competing(outcome)
    tte = col_tte_days(outcome)
    surv_time = col_surv_time(outcome)
    surv_event = col_surv_event(outcome)

    # diagnosed: flag is True AND a date is recorded
    df[diagnosed] = df[outcome] & df[date_col].notna()

    # prevalent: diagnosed BEFORE follow-up start
    df[prevalent] = df[diagnosed] & (df[date_col] < df[start_col])

    # incident: diagnosed ON OR AFTER start AND ON OR BEFORE censor
    df[incident] = (
        df[diagnosed]
        & (df[date_col] >= df[start_col])
        & (df[date_col] <= df[censor_col])
    )

    # competing: death during follow-up without the outcome (default False)
    df[competing] = False

    # TTE from start to diagnosis date (NaN for prevalent)
    df[tte] = (df[date_col] - df[start_col]).dt.days
    df.loc[df[prevalent], tte] = np.nan

    # Initialise survival columns
    df[surv_event] = np.nan
    df[surv_time] = np.nan

    # Event = 1: incident cases
    df.loc[df[incident], surv_event] = 1
    df.loc[df[incident], surv_time] = df.loc[df[incident], tte]

    # Subjects not yet classified: not prevalent, not incident
    non_case = (~df[prevalent]) & (~df[incident])

    start_s = df[start_col]
    censor_s = df[censor_col]
    censor_days = (censor_s - start_s).dt.days

    use_competing = (
        death_date_col in df.columns
        and df[death_date_col].notna().any()
    )

    if use_competing:
        death_s = df[death_date_col]
        died_during = (
            non_case
            & death_s.notna()
            & (death_s > start_s)
            & (death_s <= censor_s)
        )
        df.loc[died_during, surv_event] = 2
        df.loc[died_during, surv_time] = (
            death_s[died_during] - start_s[died_during]
        ).dt.days
        df.loc[died_during, competing] = True

        admin_censored = non_case & ~died_during
    else:
        admin_censored = non_case

    # Event = 0: administratively censored
    df.loc[admin_censored, surv_event] = 0
    df.loc[admin_censored, surv_time] = censor_days[admin_censored]

    # ------------------------------------------------------------------
    # 3. Validation report
    # ------------------------------------------------------------------
    n_pd_only = int((df["pd_flag"] & ~df["ad_flag"] & ~df["dem_flag"]).sum())
    n_pd_med = int(df[outcome].sum())
    n_pd_no_med = n_pd_only - n_pd_med
    n_diag = int(df[diagnosed].sum())
    n_prev = int(df[prevalent].sum())
    n_inc = int(df[incident].sum())
    n_comp = int(df[competing].sum())

    if verbose:
        print("\n" + "=" * 80)
        print("OUTCOME 5a: PD + MEDICATION (HIGH-SPECIFICITY)")
        print("=" * 80)
        print(f"  PD-only cases (outcome_1a logic):   {n_pd_only:,}")
        print(f"  PD + medication (outcome_5a):       {n_pd_med:,}")
        print(f"  PD without medication (excluded):   {n_pd_no_med:,}")
        print(f"  Specificity gain (cases removed):   {n_pd_no_med:,} "
              f"({100 * n_pd_no_med / max(n_pd_only, 1):.1f}%)")
        print(f"  ---")
        print(f"  Diagnosed with date:                {n_diag:,}")
        print(f"  Prevalent:                          {n_prev:,}")
        print(f"  Incident:                           {n_inc:,}")
        print(f"  Competing (death):                  {n_comp:,}")
        print("=" * 80)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        validation = pd.DataFrame([{
            "outcome": outcome,
            "n_pd_only": n_pd_only,
            "n_pd_med": n_pd_med,
            "n_pd_no_med": n_pd_no_med,
            "pct_removed": round(100 * n_pd_no_med / max(n_pd_only, 1), 2),
            "diagnosed": n_diag,
            "prevalent": n_prev,
            "incident": n_inc,
            "competing": n_comp,
        }])
        validation.to_csv(save_dir / "outcome_5a_pd_med_validation.csv", index=False)

    return df


def add_outcome_flags_old_extraction(df: pd.DataFrame,
                      save_dir: Path = None,
                      verbose: bool = True,
                      censor_date: str = "2022-10-31") -> pd.DataFrame:
    """
    Adds outcome flags and survival analysis columns to a dataframe by analyzing ICD-10 diagnosis
    codes and their corresponding diagnosis dates. The function determines various outcomes, flags,
    diagnosis dates, and survival time for specified codes. It includes diagnosis categorization for
    Parkinson's Disease (PD), Alzheimer's Disease (AD), Dementia (DEM), and Dementia with Lewy Bodies (DLB).

    Adds PD/AD/dementia flags, diagnosis dates, composite outcomes,
    AND survival-analysis columns for each outcome:
        <outcome>_diagnosed
        <outcome>_prevalent
        <outcome>_incident
        <outcome>_TTE_days
        <outcome>_surv_time
        <outcome>_surv_event

    Survival rules:
        prevalent -> excluded (NaN time/event)
        incident  -> event=1, TTE
        control   -> event=0, censor_time (fixed date)

    :param df: Input dataframe containing participant data with ICD-10 diagnosis fields, dates,
        and wear time start information. Must include fields prefixed by "41270" for diagnosis codes
        and "41280" for their corresponding dates.
    :type df: pd.DataFrame

    :param save_dir: File path to which the summary table will be saved. If None, no table is saved.
        The summary table contains counts and statistics for each defined outcome.
    :type save_dir: Path | None

    :param verbose: If True, prints the summary table to the console. Defaults to True.
    :type verbose: bool

    :param censor_date: Censoring date used to define the end of the observation period.
        Should be a date string in "YYYY-MM-DD" format. Defaults to "2022-10-31".
    :type censor_date: str

    :return: Dataframe with added outcome flags, diagnosis dates, and survival analysis columns.
        Includes indicators for incident, prevalent, and unobserved cases, along with survival times
        and events for each outcome.
    :rtype: pd.DataFrame
    """
    print(f'Processing outcome flags....')
    df = df.copy()

    # ---------------------------------------------------------
    # ICD columns
    # ---------------------------------------------------------
    icd10_cols = [c for c in df.columns if "p41270" in c]
    icd10_date_cols = [c for c in df.columns if c.startswith("p41280")]
    df_icd = df[icd10_cols].astype(str)

    if not icd10_cols:
        raise ValueError("No 41270* diagnosis columns found.")
    if not icd10_date_cols:
        raise ValueError("No 41280* diagnosis date columns found.")

    # ICD code sets
    pd_codes  = outcomes_codes["PD"]
    ad_codes  = outcomes_codes["AD"]
    dem_codes = outcomes_codes["DEM"]
    dlb_codes = outcomes_codes["DLB"]
    # msa_codes = outcomes_codes["MSA"]
    # ms_codes  = outcomes_codes["MS"]

    df['censor_date'] = pd.to_datetime(censor_date)
    df['follow_up_days'] = (df['censor_date'] - df['wear_time_start']).dt.days
    df['follow_up_years'] = df['follow_up_days'] / 365.25

    # ---------------------------------------------------------
    # Individual ICD flags (lowercase)
    # ---------------------------------------------------------
    print(f'Searching outcome flags...')


    def add_icd10_flags(
            df: pd.DataFrame,
            icd10_cols: List[str],
            pd_codes: List[str],
            ad_codes: List[str],
            dem_codes: List[str],
            dlb_codes: List[str],
    ) -> pd.DataFrame:
        """
        Add ICD-10 disease flags (PD, AD, Dementia, DLB) using
        safe, vectorized, boundary-aware matching.

        Assumes ICD-10 columns contain pipe-separated codes.
        """

        def _build_icd_regex(codes: List[str]) -> re.Pattern:
            """
            Build a regex that matches full ICD-10 codes or valid prefixes
            at pipe boundaries.
            """
            escaped = [re.escape(c) for c in codes]
            pattern = rf"(?:^|\|)({'|'.join(escaped)})[A-Z0-9]*?(?:\||$)"
            return re.compile(pattern)

        def _compute_flag(codes: List[str]) -> pd.Series:
            """
            Vectorized ICD flag across multiple ICD columns.
            """
            regex = _build_icd_regex(codes)

            joined = (
                df[icd10_cols]
                .astype("string")
                .fillna("")
                .agg("|".join, axis=1)
            )

            return joined.str.contains(regex, regex=True).astype(int)

        # --------------------------------------------------
        # Apply flags
        # --------------------------------------------------
        df = df.copy()

        df["pd_flag"] = _compute_flag(pd_codes)
        df["ad_flag"] = _compute_flag(ad_codes)
        df["dem_flag"] = _compute_flag(dem_codes)
        df["dlb_flag"] = _compute_flag(dlb_codes)

        return df

    df = add_icd10_flags(
        df=df,
        icd10_cols=icd10_cols,
        pd_codes=pd_codes,
        ad_codes=ad_codes,
        dem_codes=dem_codes,
        dlb_codes=dlb_codes,
    )

    df["pd_flag"]  = df_icd.apply(lambda r: _contains_code(r, pd_codes), axis=1)
    df["ad_flag"]  = df_icd.apply(lambda r: _contains_code(r, ad_codes), axis=1)
    df["dem_flag"] = df_icd.apply(lambda r: _contains_code(r, dem_codes), axis=1)
    df["dlb_flag"] = df_icd.apply(lambda r: _contains_code(r, dlb_codes), axis=1)

    # sanity check
    for outcome in outcomes_codes.keys():
        col_flag = f"{outcome.lower()}_flag"
        print(f"\t{col_flag}: {df[col_flag].sum()}")


    # ---------------------------------------------------------
    # Diagnosis dates
    # ---------------------------------------------------------
    censor_date = pd.to_datetime(censor_date)
    df["wear_time_start"] = pd.to_datetime(df["wear_time_start"], errors="coerce")

    # if "PD_dx_date" not in df.columns:
    #     if "X42032.0.0" in df.columns:
    #         df["pd_dx_date"] = pd.to_datetime(df["X42032.0.0"], errors="coerce")
    #     else:
    #         df["pd_dx_date"] = _get_first_dx_date(df, pd_codes, icd10_cols, icd10_date_cols)
    # else:
    #     df.rename(columns={"PD_dx_date": "pd_dx_date"}, inplace=True)

    df["pd_dx_date"] = _get_first_dx_date(df, pd_codes, icd10_cols, icd10_date_cols)
    df["ad_dx_date"]  = _get_first_dx_date(df, ad_codes,  icd10_cols, icd10_date_cols)
    df["dem_dx_date"] = _get_first_dx_date(df, dem_codes, icd10_cols, icd10_date_cols)
    df["dlb_dx_date"] = _get_first_dx_date(df, dlb_codes, icd10_cols, icd10_date_cols)
    # df["msa_dx_date"] = _get_first_dx_date(df, msa_codes, icd10_cols, icd10_date_cols)
    # df["ms_dx_date"]  = _get_first_dx_date(df, ms_codes,  icd10_cols, icd10_date_cols)

    # ---------------------------------------------------------
    # Composite Outcomes
    # ---------------------------------------------------------
    df["outcome_1a_pd_only"] = df["pd_flag"] & ~df["ad_flag"] & ~df["dem_flag"]
    df["outcome_1b_pd_ad"] = df["pd_flag"] & df["ad_flag"]
    # df["outcome_1c_pd_dementia"] = df["pd_flag"] & df["dem_flag"]
    df["outcome_2a_vasculardementia"] = df["dem_flag"] & ~df["pd_flag"] & ~df["ad_flag"]
    df["outcome_2b_pd_vasculardementia"] = df["pd_flag"] & df["dem_flag"] & ~df["ad_flag"]

    df["outcome_3a_dlb_only"] = df["dlb_flag"] & ~df["pd_flag"]

    df["outcome_4a_ad_only"] = ~df["pd_flag"] & df["ad_flag"] & ~df["dem_flag"]


    # df["outcome_any_neurodegenerative"] = (
    #         df["pd_flag"] |
    #         df["ad_flag"] |
    #         df["dem_flag"] |
    #         df["dlb_flag"]
    # )
    #

    # sanity check
    for outcome in outcomes:
        print(f"\t{outcome}: {df[outcome].sum()}")



    # ---------------------------------------------------------
    # Composite outcome dates
    # ---------------------------------------------------------
    df["outcome_1a_pd_only_date"] = df["pd_dx_date"]
    df["outcome_1b_pd_ad_date"] = pd.concat([df["pd_dx_date"],
                                             df["ad_dx_date"]],
                                            axis=1).min(axis=1)
    # df["outcome_1c_pd_dementia_date"] = pd.concat([df["pd_dx_date"], df["dem_dx_date"]], axis=1).min(axis=1)
    df["outcome_2a_vasculardementia_date"] = df["dem_dx_date"]
    df["outcome_2b_pd_vasculardementia_date"] = pd.concat([df["pd_dx_date"],
                                                        df["dem_dx_date"]],
                                                       axis=1).min(axis=1)

    df["outcome_3a_dlb_only_date"] = df["dlb_dx_date"]

    df["outcome_4a_ad_only_date"] = df["ad_dx_date"]

    # Earliest diagnosis date among PD, AD, DEM, DLB
    # df["outcome_any_neurodegenerative_date"] = pd.concat([
    #     df["pd_dx_date"],
    #     df["ad_dx_date"],
    #     df["dem_dx_date"],
    #     df["dlb_dx_date"]
    # ], axis=1).min(axis=1)

    # ---------------------------------------------------------
    # Survival + prevalent/incident computation
    # ---------------------------------------------------------
    summary_rows = []

    for outcome in outcomes:
        outcome_l = outcome.lower()
        dx = f"{outcome_l}_date"

        # diagnosed
        diag = col_dx(outcome_l)
        df[diag] = df[outcome_l] & df[dx].notna()

        # prevalent
        prev = col_prevalent(outcome_l)
        df[prev] = df[diag] & (df[dx] < df["wear_time_start"])

        # incident
        inc = col_incident(outcome_l)
        df[inc] = df[diag] & (df[dx] >= df["wear_time_start"])

        # TTE
        tte = col_tte_days(outcome_l)
        df[tte] = (df[dx] - df["wear_time_start"]).dt.days
        df.loc[df[prev], tte] = np.nan

        # survival vars
        surv_time = col_surv_time(outcome_l)
        surv_event = col_surv_event(outcome_l)

        df[surv_time] = np.nan
        df[surv_event] = np.nan

        df.loc[df[inc], surv_event] = 1
        df.loc[df[inc], surv_time] = df.loc[df[inc], tte]

        # controls -> censored
        ctrl_mask = (~df[prev]) & (~df[inc])
        censor_days = (censor_date - df["wear_time_start"]).dt.days

        df.loc[ctrl_mask, surv_event] = 0
        df.loc[ctrl_mask, surv_time] = censor_days[ctrl_mask]

        summary_rows.append({
            "outcome": outcome_l,
            "diagnosed": int(df[diag].sum()),
            "prevalent": int(df[prev].sum()),
            "incident": int(df[inc].sum()),
            "incident_%": round(df[inc].mean() * 100, 3),
            "median_tte": df.loc[df[inc], tte].median(),
        })

    summary_df = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------
    # UNMAPPED PD/AD/DEM diagnostics
    # ---------------------------------------------------------
    composite_any = df[[o.lower() for o in outcomes]].sum(axis=1) > 0
    unmapped = (df["pd_flag"] | df["ad_flag"] | df["dem_flag"]) & (~composite_any)

    unmapped_df = pd.DataFrame({
        "category": [
            "pd only (no composite)",
            "ad only (no composite)",
            "dem only (no composite)",
            "pd + ad only (no composite)",
            "pd + dem only (no composite)",
            "ad + dem only (no composite)",
            "pd + ad + dem (no composite)",
        ],
        "count": [
            (df["pd_flag"] & ~df["ad_flag"] & ~df["dem_flag"] & unmapped).sum(),
            (df["ad_flag"] & ~df["pd_flag"] & ~df["dem_flag"] & unmapped).sum(),
            (df["dem_flag"] & ~df["pd_flag"] & ~df["ad_flag"] & unmapped).sum(),
            (df["pd_flag"] & df["ad_flag"] & ~df["dem_flag"] & unmapped).sum(),
            (df["pd_flag"] & df["dem_flag"] & ~df["ad_flag"] & unmapped).sum(),
            (df["ad_flag"] & df["dem_flag"] & ~df["pd_flag"] & unmapped).sum(),
            (df["pd_flag"] & df["ad_flag"] & df["dem_flag"] & unmapped).sum(),
        ]
    })

    # ---------------------------------------------------------
    # VERBOSE PRINT / SAVE
    # ---------------------------------------------------------
    if verbose:
        print("\n" + "=" * 100)
        print(" OUTCOME + SURVIVAL SUMMARY TABLE")
        print("=" * 100)
        print(tabulate(summary_df, headers="keys", tablefmt="github", showindex=False))
        print("=" * 100)

        print("\n" + "=" * 100)
        print(" UNMAPPED PD/AD/DEM DIAGNOSTICS")
        print("=" * 100)
        print(tabulate(unmapped_df, headers="keys", tablefmt="github", showindex=False))
        print("=" * 100 + "\n")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(Path(save_dir) / "outcome_summary.csv", index=False)
        unmapped_df.to_csv(Path(save_dir) / "unmapped_diagnostics.csv", index=False)
        df.to_parquet(Path(save_dir) / "1_outcome_flags.parquet", index=False)

    return df

