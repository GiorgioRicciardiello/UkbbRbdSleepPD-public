"""
Neuro-exclusion ICD-10 detection for removing neurologically confounded cases.
Creates:
    - neuro_exclude flag
"""
from tblib.decorators import return_error

# Large ICD-10 exclusion list
from config.config import neuro_exclusion_codes, outcomes
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import re
from typing import List
from pathlib import Path
from library.ehr_outcomes.utils import report_outcomes_by_flags


def add_neuro_exclusion(
    df: pd.DataFrame,
    verbose: bool = True,
        baseline_col: str = "wear_time_start",
        overwrite:bool = False,
        save_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Adds `neuro_exclude` flag for exclusionary neurological ICD-10 diagnoses
    present *before baseline* (wear_time_start).

    Logic mirrors outcome definition:
    - scans all ICD follow-ups (41270*)
    - finds earliest diagnosis date (41280*)
    - excludes only if dx_date < wear_time_start
    """

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

    path_out = save_dir / "2_neuro_exclude.parquet"

    print("Processing neurological exclusions...")
    if not overwrite and path_out.exists():
        print(f'Reading from existing file: {path_out}')
        df = pd.read_parquet(path_out)
        return df

    df = df.copy()

    # --------------------------------------------------
    # ICD columns (all follow-ups)
    # --------------------------------------------------
    icd10_cols = [c for c in df.columns if "p41270" in c]
    icd10_date_cols = [c for c in df.columns if c.startswith("p41280")]

    if not icd10_cols:
        raise ValueError("No ICD-10 diagnosis columns (p41270*) found.")
    if not icd10_date_cols:
        raise ValueError("No ICD-10 date columns (p41280*) found.")

    # --------------------------------------------------
    # Earliest neurological diagnosis date
    # --------------------------------------------------

    df['neuro_dx_date'] = pd.to_datetime(
        df.apply(
            _get_first_dx_date,
            axis=1,
            codes_of_interest=neuro_exclusion_codes,
        ),
        errors="coerce",
    )

    # --------------------------------------------------
    # Baseline exclusion rule (PREVALENT ONLY)
    # --------------------------------------------------
    df[baseline_col] = pd.to_datetime(
        df[baseline_col], errors="coerce"
    )

    df["neuro_exclude"] = (
        df["neuro_dx_date"].notna()
        & (df["neuro_dx_date"] < df[baseline_col])
    )

    if verbose:
        n_excl = int(df["neuro_exclude"].sum())
        print(f"Neurological exclusions at baseline: {n_excl}")



    report_outcomes_by_flags(df=df,
                             outcomes=outcomes,
                            verbose=verbose,
                             flags=["neuro_exclude"],
                             output_path=save_dir.joinpath("neuro_exclusion_report.csv") if save_dir else None)

    if save_dir:
        df.to_parquet(path_out, index=False)
    return df



def add_shift_worker(df: pd.DataFrame,
                     verbose: bool = True,
                    save_dir:Path=None,
                     ) -> pd.DataFrame:
    """
    Add shift work-related variables to a Pandas DataFrame by analyzing and
    aggregating relevant fields from UK Biobank data. This function creates
    multiple derived columns that categorize shift work exposure and frequency
    according to predefined criteria. It optionally reports descriptive
    statistics on shift work exposure by outcome.

    :param df: A Pandas DataFrame containing UK Biobank data. Expected to include
        relevant fields used for deriving shift work exposure variables
    :param verbose: A boolean indicating whether detailed logs and reports on the
        derived columns and their statistics should be printed. Default is True

    :return: A Pandas DataFrame with additional columns created to reflect shift
        work exposure and frequency metrics
    :rtype: pd.DataFrame
    """
    # FIRST SHIFT WORKER COLUMNS
    def derive_shift_work(
            df: pd.DataFrame,
            field: str = "p826",
            verbose: bool = True,
            drop_original: bool = True,
    ) -> pd.DataFrame:
        """
        Derive epidemiologically correct shift-work covariates from UK Biobank
        shift-related fields (e.g. 826 = shift work, 3426 = night shift).

        Expected UKBB coding:
            1  = Never / rarely
            2  = Sometimes
            3  = Usually
            4  = Always
           -1  = Do not know
           -3  = Prefer not to answer

        Derived variables (per event):
            - shift_worker_any_{event}   : binary (2?4 vs 1)
            - shift_work_freq_{event}    : ordinal (1?4, NaN otherwise)
            - shift_worker_high_{event}  : binary (3?4 vs 1?2)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        field : str
            Base UKBB field name (e.g. 'p826', 'p3426').
        verbose : bool
            Print summary statistics.
        drop_original : bool
            Drop original UKBB columns after derivation.

        Returns
        -------
        pd.DataFrame
        """

        # --------------------------------------------------
        # Identify all event-specific columns
        # --------------------------------------------------
        cols = [c for c in df.columns if c.startswith(f"{field}_i")]

        if not cols:
            raise KeyError(f"No columns found for base field '{field}'")

        created_cols: List[str] = []

        for col in cols:
            event = col.split("_i")[-1]
            x = pd.to_numeric(df[col], errors="coerce")

            valid = x.isin([1, 2, 3, 4])

            col_any = f"shift_any_i{event}_{field}"
            col_freq = f"shift_freq_i{event}_{field}"
            col_high = f"shift_high_i{event}_{field}"

            # Any exposure: sometimes / usually / always
            df[col_any] = np.where(
                valid,
                x.isin([2, 3, 4]).astype("float"),
                np.nan,
            )

            # Ordinal frequency
            df[col_freq] = x.where(valid)

            # High exposure: usually / always
            df[col_high] = np.where(
                valid,
                x.isin([3, 4]).astype("float"),
                np.nan,
            )

            created_cols.extend([col_any, col_freq, col_high])

            if verbose:
                print(f"[{field} | event i{event}]")
                print(f"  -> any shift work : {int(np.nansum(df[col_any])):,}")
                print(f"  -> high exposure  : {int(np.nansum(df[col_high])):,}")

        # --------------------------------------------------
        # Drop original UKBB columns
        # --------------------------------------------------
        if drop_original:
            df = df.drop(columns=cols)

        if verbose:
            print("\nCreated columns:")
            for c in created_cols:
                print(f"  - {c}")
            if drop_original:
                print(f"Dropped original columns: {cols}")

        return df

    # Shift work (field 826)
    df = derive_shift_work(
        df,
        field="p826",
        verbose=True,
    )

    # Night shift work (field 3426)
    df = derive_shift_work(
        df,
        field="p3426",
        verbose=True,
    )

    # SECOND SHIFT WORKER COLUMNS
    # def _collapse_shift_exposure(df, prefix, exposure_codes=(0, 1)):
    #     cols = [c for c in df.columns if c.startswith(prefix)]
    #     arr = df[cols].apply(pd.to_numeric, errors="coerce")
    #     return arr.isin(exposure_codes).any(axis=1)
    #
    # df['shift_worker_exposed'] = _collapse_shift_exposure(df=df,
    #                                                       prefix="p22650_a",
    #                                                       exposure_codes=(0, 1))

    def collapse_night_shift_22650(
            df: pd.DataFrame,
            prefix: str = "p22650_a",
            verbose: bool = True,
            drop_original: bool = True,
    ) -> pd.DataFrame:
        """
        Collapse UK Biobank Data-Field 22650 (Night shifts worked)
        across all jobs into epidemiologically valid covariates.

        Coding (Data-Coding 489):
            0 = Never
            1 = Sometimes
            2 = Usually
            3 = Always
           -1 = Do not know
           -3 = Prefer not to answer

        Derived variables:
            - night_shift_ever   : binary (?1 vs 0)
            - night_shift_max    : ordinal (0?3, NaN if no valid jobs)
            - night_shift_high   : binary (?2 vs 0?1)
        """

        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            raise KeyError(f"No columns found with prefix '{prefix}'")

        arr = df[cols].apply(pd.to_numeric, errors="coerce")

        valid = arr.isin([0, 1, 2, 3])
        arr_valid = arr.where(valid)

        # --------------------------------------------------
        # Ever exposed (PRIMARY)
        # --------------------------------------------------
        df["night_shift_ever"] = (
            arr_valid.ge(1).any(axis=1)
            .astype("float")
        )

        # --------------------------------------------------
        # Maximum lifetime exposure (ORDINAL)
        # --------------------------------------------------
        df["night_shift_max"] = arr_valid.max(axis=1)

        # --------------------------------------------------
        # High-intensity exposure (SENSITIVITY)
        # --------------------------------------------------
        df["night_shift_high"] = (
            arr_valid.ge(2).any(axis=1)
            .astype("float")
        )

        # Set all-missing rows to NaN
        all_missing = arr_valid.isna().all(axis=1)
        df.loc[all_missing, ["night_shift_ever",
                             "night_shift_max",
                             "night_shift_high"]] = np.nan

        if drop_original:
            df = df.drop(columns=cols)

        if verbose:
            print("Night shift exposure (Field 22650):")
            print(f"  -> ever exposed      : {int(np.nansum(df['night_shift_ever'])):,}")
            print(f"  -> high exposure     : {int(np.nansum(df['night_shift_high'])):,}")
            print(f"  -> max exposure dist :")
            print(df["night_shift_max"].value_counts(dropna=False).sort_index())

        return df

    df = collapse_night_shift_22650(
        df,
        prefix="p22650_a",
        verbose=True,
        drop_original=True,
    )
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    # REPORT
    col_shit = [col for col in df.columns if 'shift' in col]
    _ = report_outcomes_by_flags(df=df,
         outcomes=outcomes,
        verbose=verbose,
         flags=col_shit,
         output_path=save_dir.joinpath("shift_worker_report.csv") if save_dir else None)


    return df



def add_bad_actig_recording(
    df: pd.DataFrame,
    verbose: bool = True,
    save_dir: Path | None = None
) -> pd.DataFrame:
    """
    Adds a boolean column `acc_bad_quality` indicating poor accelerometer data quality.

    Exclusion criteria follow published UKBB accelerometry QC standards
    (Doherty et al. 2017; Walmsley et al. 2022):

    Flag as bad if ANY condition is met:
    - p90015 == 0  : insufficient wear time (< 3 valid days)
    - p90016 == 0  : calibration failure (physically invalid data)
    - p90002 in {1, 2} : data problem indicator (clipping or suspicious calibration)
    - p90180 > 0   : interrupted recording periods

    Excluded from criteria (non-standard, not used in published UKBB papers):
    - p90017: calibration *method* flag (0 = population-calibrated, 1 = own-data).
              Population calibration is the UKBB default and produces valid data;
              excluding p90017==0 would incorrectly reject the majority of participants.
    - p90018: daylight savings crossover. Affects ~4% of recordings but does not
              invalidate the accelerometry signal; not used as exclusion in any
              published UKBB accelerometry study.
    """

    # IMPORTANT: force pure boolean mask from the start
    bad_mask = pd.Series(False, index=df.index, dtype="bool")

    if "p90015" in df.columns:
        bad_mask |= df["p90015"].eq(0).fillna(False)

    if "p90016" in df.columns:
        bad_mask |= df["p90016"].eq(0).fillna(False)

    if "p90002" in df.columns:
        bad_mask |= df["p90002"].isin([1, 2]).fillna(False)

    if "p90180" in df.columns:
        bad_mask |= (
            pd.to_numeric(df["p90180"], errors="coerce")
            .fillna(0)
            .gt(0)
        )

    # Assign WITHOUT astype (already boolean)
    df["acc_bad_quality"] = bad_mask

    # ------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------
    report = report_outcomes_by_flags(
        df=df,
        outcomes=outcomes,
        flags=["acc_bad_quality"],
        verbose=verbose,
        output_path=(
            save_dir / "acc_bad_quality_report.csv"
            if save_dir else None
        ),
    )

    return df


def _exclusion_reporting(
    df: pd.DataFrame,
    outcomes: List[str],
    icd_cols: List[str],
    pattern: str,
    exclude_flag: str = "neuro_exclude",
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Generate standardized exclusion reporting tables.

    Outputs:
    --------
    1. cohort_flow      : participant-level exclusion flow
    2. icd_drivers      : ICD-10 prefix drivers of exclusion
    3. outcome_impact   : outcome-level case attrition

    Returns
    -------
    Dict[str, pd.DataFrame]
    """

    if exclude_flag not in df.columns:
        raise ValueError(f"Missing exclusion flag: {exclude_flag}")

    n_total = len(df)
    n_excl = int(df[exclude_flag].sum())

    # ================================================================
    # TABLE 1 ? COHORT FLOW
    # ================================================================
    cohort_flow = pd.DataFrame([
        {"Step": "Initial cohort", "N": n_total, "%": 100.0},
        {
            "Step": "Neurological exclusion",
            "N": n_excl,
            "%": round(n_excl / n_total * 100, 2)
        },
        {
            "Step": "Final analytic cohort",
            "N": n_total - n_excl,
            "%": round((n_total - n_excl) / n_total * 100, 2)
        }
    ])

    # ================================================================
    # TABLE 2 ? ICD DRIVERS OF EXCLUSION (FIXED)
    # ================================================================
    df_icd = df[icd_cols].astype(str)

    # Boolean mask: ICD entries that actually triggered exclusion
    matched_icd = df_icd.apply(
        lambda col: col.str.contains(pattern, regex=True),
        axis=0
    )

    icd_exploded = (
        df_icd
        .where(matched_icd)  # keep only exclusion-triggering ICDs
        .stack()
        .str.extract(r"^([A-Z]\d{2})")[0]
        .dropna()
    )

    icd_drivers = (
        icd_exploded
        .value_counts()
        .rename("N")
        .to_frame()
        .assign(percent=lambda x: (x["N"] / x["N"].sum() * 100).round(2))
        .reset_index()
        .rename(columns={"index": "ICD_prefix"})
        .sort_values("N", ascending=False)
    )

    # ================================================================
    # TABLE 3 ? OUTCOME-LEVEL IMPACT (CASES ONLY)
    # ================================================================
    rows = []
    for out in outcomes:
        total_cases = int(df[out].sum())
        excluded_cases = int(df.loc[df[exclude_flag] & df[out]].shape[0])

        rows.append({
            "Outcome": out,
            "Cases before exclusion": total_cases,
            "Cases excluded": excluded_cases,
            "% cases excluded": (
                round(excluded_cases / total_cases * 100, 2)
                if total_cases > 0 else 0.0
            ),
            "Cases after exclusion": total_cases - excluded_cases
        })

    outcome_impact = pd.DataFrame(rows)

    # ================================================================
    # SAVE
    # ================================================================
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        cohort_flow.to_csv(
            save_dir / "exclusion_cohort_flow.csv", index=False
        )
        icd_drivers.to_csv(
            save_dir / "exclusion_icd_drivers.csv", index=False
        )
        outcome_impact.to_csv(
            save_dir / "exclusion_outcome_impact.csv", index=False
        )

    # ================================================================
    # VERBOSE SUMMARY
    # ================================================================
    if verbose:
        print("\n" + "=" * 78)
        print("EXCLUSION REPORT SUMMARY")
        print("=" * 78)
        print(f"Total participants        : {n_total:,}")
        print(f"Excluded (neurological)   : {n_excl:,} ({n_excl / n_total:.2%})")
        print(f"Final analytic cohort     : {n_total - n_excl:,}")
        print("=" * 78 + "\n")

    return {
        "cohort_flow": cohort_flow,
        "icd_drivers": icd_drivers,
        "outcome_impact": outcome_impact
    }

