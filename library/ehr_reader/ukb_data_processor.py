"""
UK Biobank Data Processor

This module provides the UkbDataProcessor class for processing and transforming
extracted UK Biobank data.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from config.config import config
from library.ehr_outcomes.outcome_flags import add_outcome_flags
from library.ehr_outcomes.exclusion import add_neuro_exclusion, add_shift_worker, add_bad_actig_recording
from library.ehr_outcomes.controls import add_controls
from library.ehr_outcomes.covariates import add_covariates, merge_prodromal_markers, add_cognitive_latest_per_subject
from library.ehr_outcomes.identify_splits import add_data_split_flags
from library.ehr_outcomes.medications import add_medication_flags
from library.ehr_outcomes.age_groups import create_age_groups
from library.ehr_outcomes.consort_report import generate_ehr_consort_flow


class UkbDataProcessor:
    """
    Handles processing and transformation of UK Biobank data.

    This class filters subjects, applies processing pipelines (outcome flags,
    exclusions, controls, covariates, medication flags), and saves final datasets.

    Attributes:
        out_dir_logs: Directory for saving processing logs.
        censor_date: Censor date for outcome flags.
        data_sheet_dir: Directory containing the raw UKBB data-sheet CSVs used
                        by the medication-flag step.  None disables that step.
    """

    def __init__(
        self,
        out_dir_logs: Path,
        censor_date: str = "2025-11-1",
        data_sheet_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the UkbDataProcessor.

        Args:
            out_dir_logs: Directory for saving processing logs.
            censor_date: Censor date for outcome flags (default: "2025-11-1").
            data_sheet_dir: Directory containing the raw UKBB data-dictionary
                            and codings CSVs.  Pass None to skip medication flags.
        """
        self.out_dir_logs = out_dir_logs
        self.censor_date = censor_date
        self.data_sheet_dir = (
            data_sheet_dir
            if data_sheet_dir is not None
            else config.get("paths")["data_sheet"]["dir_input"]
        )
        self.out_dir_logs.mkdir(parents=True, exist_ok=True)

    def filter_subjects_with_actigraphy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only subjects with a valid actigraphy baseline.

        ``wear_time_start`` (mapped from p90003 by UkbDataExtractor) is the
        canonical indicator of actigraphy participation.

        Args:
            df: DataFrame to filter.

        Returns:
            Filtered DataFrame containing only subjects with
            ``wear_time_start`` not NaT.
        """
        print("\nFiltering for subjects with actigraphy data...")

        if 'wear_time_start' not in df.columns:
            raise KeyError(
                "'wear_time_start' column not found. "
                "Ensure UkbDataExtractor has renamed p90003 → wear_time_start "
                "before calling this method."
            )

        df['wear_time_start'] = pd.to_datetime(df['wear_time_start'], errors='coerce')
        n_before = len(df)
        df = df[df['wear_time_start'].notna()].copy()
        n_after = len(df)

        print(f"  Subjects with actigraphy (wear_time_start not NaT): {n_after:,}")
        print(f"  Subjects removed (no wear_time_start):               {n_before - n_after:,}")
        print(f"  [OK] Filtered from {n_before:,} to {n_after:,} subjects")

        return df

    @staticmethod
    def _get_max_diagnosis_date(
        df: pd.DataFrame,
        icd10_date_col_alias: str = "41262",
        censore_date: str = "2025-11-1",
    ) -> pd.Timestamp:
        """
        Log the maximum HES diagnosis date and return the configured censor date.

        The configured censor_date is authoritative — it must not be overridden
        by the HES coverage ceiling.  This function is diagnostic only.
        """
        censore_date_ts = pd.to_datetime(censore_date)

        date_cols = [c for c in df.columns if icd10_date_col_alias in c]
        if date_cols:
            max_date = pd.to_datetime(
                max(df[col].max() for col in date_cols), errors="coerce"
            )
            print(f"  Max HES diagnosis date (p{icd10_date_col_alias}): {max_date.date()}")
            print(f"  Configured censor date: {censore_date_ts.date()}")
            if max_date > censore_date_ts:
                print(
                    f"  Warning: HES data ({max_date.date()}) extends beyond "
                    f"censor date ({censore_date_ts.date()}) — events will be censored."
                )
            else:
                print(
                    f"  Info: HES coverage ends {max_date.date()}, "
                    f"{(censore_date_ts - max_date).days} days before censor date "
                    f"(first-occurrence fields may cover the gap)."
                )
        else:
            print(f"  Warning: no columns matching '{icd10_date_col_alias}' found.")

        return censore_date_ts

    def apply_processing_pipeline(self, df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
        """
        Apply the full processing pipeline: outcome flags, exclusions, controls, covariates.

        Args:
            df: Input DataFrame
            overwrite: Whether to overwrite existing intermediate outputs

        Returns:
            Processed DataFrame
        """
        print("\n--- Applying processing pipeline ---")

        df['wear_time_start'] = pd.to_datetime(df['wear_time_start'], errors='coerce')
        df['wear_time_end'] = pd.to_datetime(df['wear_time_end'], errors='coerce')

        n_null_wear = df['wear_time_start'].isna().sum()
        if n_null_wear > 0:
            raise ValueError(
                f"{n_null_wear} subjects have wear_time_start = NaT. "
                "Call filter_subjects_with_actigraphy() before apply_processing_pipeline()."
            )

        # 1. OUTCOME FLAGS
        print("\n1. Adding outcome flags...")
        df_diag = add_outcome_flags(
            df=df,
            verbose=True,
            save_dir=self.out_dir_logs / "outcome_flags",
            censor_date=self.censor_date,
            overwrite=overwrite
        )

        # 2. NEUROLOGICAL EXCLUSIONS
        print("\n2. Applying neurological exclusions...")
        df_diag_ex = add_neuro_exclusion(
            df=df_diag,
            overwrite=overwrite,
            save_dir=self.out_dir_logs / "exclusion"
        )

        print("\n2a. Applying shift worker exclusions...")
        df_diag_ex = add_shift_worker(
            df=df_diag_ex,
            save_dir=self.out_dir_logs / "exclusion",
            verbose=True
        )

        print("\n2b. Applying bad actigraphy exclusions...")
        df_diag_ex = add_bad_actig_recording(
            df=df_diag_ex,
            save_dir=self.out_dir_logs / "exclusion",
            verbose=True
        )

        print("\n2c. Data split flags [Deprecated — using ABK model]")

        # 3. CONTROL DEFINITION
        print("\n3. Adding controls...")
        df_diag_ex_ctrl = add_controls(
            df=df_diag_ex,
            save_dir=self.out_dir_logs / "controls",
            verbose=True,
            overwrite=overwrite
        )

        # 4. COVARIATES
        print("\n4. Adding covariates...")
        df_diag_ex_ctrl_cov = add_covariates(
            df=df_diag_ex_ctrl,
            save_dir=self.out_dir_logs / "covariates",
            overwrite=True,
            verbose=overwrite
        )

        # 4b. AGE GROUPS
        df_diag_ex_ctrl_cov = create_age_groups(
            df=df_diag_ex_ctrl_cov,
            age_col="cov_age_recruitment_21022"
        )

        # 4c. COGNITIVE LATEST PER SUBJECT
        # Selects latest available cognitive assessment per subject across instances.
        # Prints DEBUG table to terminal — validate units/ranges before analysis.
        print("\n4c. Selecting latest cognitive assessment per subject...")
        df_diag_ex_ctrl_cov = add_cognitive_latest_per_subject(
            df_diag_ex_ctrl_cov,
            verbose=True,
        )

        # 5. MEDICATION FLAGS
        if self.data_sheet_dir is not None:
            print("\n5. Adding medication flags...")
            df_diag_ex_ctrl_cov = add_medication_flags(
                df=df_diag_ex_ctrl_cov,
                data_sheet_dir=self.data_sheet_dir,
                col_prefix="med_",
                save_dir=self.out_dir_logs / "medications",
                overwrite=True,
                verbose=True,
            )
        else:
            print("\n5. Skipping medication flags (data_sheet_dir not provided).")

        # 6. PRODROMAL MARKERS (merge HES covariates + medication flags)
        print("\n6. Merging prodromal markers (HES + medication)...")
        df_diag_ex_ctrl_cov = merge_prodromal_markers(
            df=df_diag_ex_ctrl_cov,
            save_dir=self.out_dir_logs / "prodromal_markers",
            verbose=True,
        )

        # CONSORT / STROBE flow table
        from config.config import outcomes as _all_outcomes
        import warnings as _warnings
        try:
            generate_ehr_consort_flow(
                df=df_diag_ex_ctrl_cov,
                outcomes=_all_outcomes,
                save_dir=self.out_dir_logs / "consort",
                verbose=True,
            )
        except Exception as _exc:
            _warnings.warn(f"CONSORT flow table generation failed: {_exc}")

        return df_diag_ex_ctrl_cov

    def save_final_dataset(
        self,
        df: pd.DataFrame,
        out_dir: Path,
        filename_base: str = "ukb_new_data_processed"
    ) -> None:
        """
        Save the final processed dataset in Parquet format.

        Args:
            df: DataFrame to save
            out_dir: Output directory
            filename_base: Base filename (without extension)
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path_parquet = out_dir / f"{filename_base}.parquet"
        df.to_parquet(out_path_parquet, index=False)
        print(f"\n  [OK] Final processed dataset saved to: {out_path_parquet}")
        print(f"  Total rows: {len(df):,}")
        print(f"  Total columns: {len(df.columns)}")
