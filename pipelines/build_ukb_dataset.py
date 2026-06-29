"""
Build UK Biobank Dataset

This script orchestrates the complete pipeline for building the final UK Biobank dataset:
1. Map fields from the data dictionary to CSV files
2. Extract data from CSV files
3. Process data (filter, apply outcome flags, exclusions, controls, covariates)
4. Save final dataset

Author: Giorgio Ricciardello
Date: 2026-01-17
"""

from pathlib import Path
from config.config import config
import pandas as pd
from library.ehr_reader import (UkbFieldMapper,
                                UkbDataExtractor,
                                UkbDataProcessor,
                                UKBBFieldMetadataExtractor)
import re
import json
import numpy as np


def construct_data(tables_dir: Path,
                   out_dir: Path,
                   out_path_csv: Path, 
                   out_path_parquet: Path, 
                   ):
    """
    Construct the final dataset from raw UK Biobank data.
    :param tables_dir: 
    :param out_dir: 
    :param out_path_csv: 
    :param out_path_parquet: 
    :return: 
    """
    path_data_dict = tables_dir / "app45551_20251118060954.dataset.data_dictionary.csv"

    # CSV files mapping
    csv_files = {
        'additional_exposures': tables_dir / 'additional_exposures.txt',
        'assessment_center': tables_dir / 'assessment_center.txt',
        'biological_samples': tables_dir / 'biological_samples.txt',
        'health_outcomes': tables_dir / 'health_outcomes.txt',
        'online_followup': tables_dir / 'online_followup.txt',
        'population_characteristics': tables_dir / 'population_characteristics.txt',
    }

    print(f"  Data directory: {tables_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"  Data dictionary: {path_data_dict}")

    # --------------------------------------------------
    # 2. MAP FIELDS
    # --------------------------------------------------
    print("\n[2/5] Mapping fields to CSV files...")

    # Load data dictionary
    df_data_dictionary = pd.read_csv(path_data_dict, low_memory=False)

    # Initialize mapper
    mapper = UkbFieldMapper(df_data_dictionary, csv_files)

    # Get codes of interest
    field_groups, all_field_ids = mapper.get_codes_of_interest()
    field_groups_json = {k: sorted(v) for k, v in field_groups.items()}

    with open(out_dir.joinpath("data_fields_categories.json"), "w") as f:
        json.dump(field_groups_json, f, indent=4)
    # Map fields to CSV files
    df_matched = mapper.map_fields_to_csv(all_field_ids)

    print(f"\n  [OK] Mapped {len(df_matched)} fields across {df_matched['csv_name'].nunique()} CSV files")

    # --------------------------------------------------
    # 3. EXTRACT DATA
    # --------------------------------------------------
    print("\n[3/5] Extracting data from CSV files...")

    # Initialize extractor
    extractor = UkbDataExtractor(tables_dir, out_dir)

    # Extract all CSV data
    extracted_dfs = extractor.extract_all_csv_data(df_matched)

    # Merge extracted data
    df_merged = extractor.merge_extracted_data(extracted_dfs)
    print(f'\n Saving merged data to {out_path_csv} and {out_path_parquet}...')

    df_merged = _convert_data_types(df_merged)
    # keep only actigraphy records
    df_merged = df_merged.loc[df_merged['wear_time_start'].notna(), :]

    # remove subjects that opted out of the ukb dataset
    # Load withdrawn subject IDs
    withdraw_ids = pd.read_csv(tables_dir.joinpath("withdraw_subjects_ukb_20260310.csv")).values
    withdraw_ids = np.array(withdraw_ids).ravel()
    df_merged = df_merged.loc[~df_merged["eid"].isin(withdraw_ids)].copy()

    df_merged.reset_index(inplace=True, drop=True)
    # df_merged.to_csv(out_path_csv, index=False)
    df_merged.to_parquet(out_path_parquet)

    print(f"\n  [OK] Extracted and merged data: {len(df_merged):,} subjects, {len(df_merged.columns)} columns")
    return df_merged


def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely converts numeric-like object/string columns to numeric dtype,
    while preserving categorical/multi-select columns as string.

    Rules:
    - If column contains '|', ':', ',', '[' -> treat as categorical (string)
    - If column can be fully parsed as numeric -> convert to numeric
    - Otherwise -> keep as pandas nullable string
    """

    df = df.copy()

    forbidden_pattern = r"[|:,\[]"

    for col in df.columns:

        # Only inspect problematic types
        if df[col].dtype == "object" or str(df[col].dtype) == "string":

            # Normalize everything to pandas string first (prevents Arrow float/string mix)
            col_str = df[col].astype("string")

            # 1. If it contains multi-select patterns -> keep as string
            if col_str.str.contains(forbidden_pattern, regex=True, na=False).any():
                df[col] = col_str
                continue

            # 2. Try numeric conversion
            numeric_col = pd.to_numeric(col_str, errors="coerce")

            # Check if conversion is meaningful:
            # If non-null values remain after conversion
            original_non_null = col_str.notna().sum()
            numeric_non_null = numeric_col.notna().sum()

            # Accept conversion if ≥ 99 % of non-null values are numeric.
            # The remaining ≤ 1 % (sentinel strings, stray empties) become NaN.
            # Requiring 100 % is too strict for UKBB columns that mix valid
            # numerics with rare coded non-responses stored as text.
            threshold = 0.99
            pct_converted = (numeric_non_null / original_non_null) if original_non_null > 0 else 0.0
            if pct_converted >= threshold:
                df[col] = numeric_col
            else:
                # Genuinely mixed text -> keep as string
                df[col] = col_str

    return df

def ukbb_csv_to_pia_notation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UK Biobank CSV-style columns:
        41270-0.17 -> p41270_i0_a17

    Works for ICD codes (41270) and dates (41280),
    and any other array-style UKBB field.
    """

    def _rename(col: str) -> str:
        m = re.match(r"(\d+)-(\d+)\.(\d+)", col)
        if not m:
            return col

        field, instance, array = m.groups()
        return f"p{field}_i{instance}_a{array}"

    df = df.copy()
    df.rename(columns={c: _rename(c) for c in df.columns}, inplace=True)
    return df


def main(overwrite_ehr: bool = True, run_field_mapping: bool = False) -> None:
    """Build the UK Biobank EHR dataset from raw CSVs.

    Parameters
    ----------
    overwrite_ehr : bool
        Re-extract from raw CSVs even if cached parquet exists.
    run_field_mapping : bool
        Run UKB field metadata extraction (slow, rarely needed).
    """
    print("=" * 80)
    print("UK BIOBANK DATASET BUILDER")
    print("=" * 80)

    # --------------------------------------------------
    # 1. SETUP PATHS
    # --------------------------------------------------
    print("\n[1/5] Setting up paths...")
    dir_new_data = config.get('paths')['data_sheet']['dir_input']
    out_path_csv = config.get('paths')['data_sheet']['dir_csv']
    out_path_parquet = config.get('paths')['data_sheet']['dir_parquet']  # processed output
    # make sure the output directory exists
    out_dir = out_path_parquet.parents[0]  # 'UkbbRbdSleepPD/data/pp/data_sheet''
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw extract cache lives at a DIFFERENT path so save_final_dataset (processed)
    # never overwrites it.  If the raw cache is missing, extract from source CSVs.
    out_path_raw_parquet = out_dir / "ukb_raw_dataset.parquet"

    print("\n[1.5/5] Getting/Constructing Dataset...")

    if not overwrite_ehr and (out_path_csv.exists() and out_path_raw_parquet.exists()):
        print("\n[1.5/5] Getting Existing Dataset from Cache...")
        df_merged = pd.read_parquet(out_path_raw_parquet)
        # Re-apply type coercion: cached parquets may retain object-dtype columns
        # if they were built with a stricter version of _convert_data_types.
        df_merged = _convert_data_types(df_merged)
        print(f"\n  [OK] From Existing (raw cache): {len(df_merged):,} subjects, {len(df_merged.columns)} columns")

    else:

        print("\n[1.5/5] Constructing Dataset from Raw CSVs...")
        df_merged = construct_data(tables_dir=dir_new_data,
                                   out_dir=out_dir,
                                   out_path_csv=out_path_csv,
                                   out_path_parquet=out_path_raw_parquet, )

    if run_field_mapping:
        print("\n[*] UKB Field Mapping Data Extractor...")

        extractor = UKBBFieldMetadataExtractor(
            data_sheet_csv=config.get('paths')['data_sheet']['dir_csv'],
            output_csv=config.get('paths')['data_sheet']['formal_name_csv'],
            output_json=config.get('paths')['data_sheet']['formal_name_json'],
        )
        df_fields = extractor.run()

    # --------------------------------------------------
    # 4. PROCESS DATA
    # --------------------------------------------------
    print("\n[4/5] Processing data...")

    # Initialize processor
    processor = UkbDataProcessor(
        out_dir_logs=out_dir / "logs",
        censor_date="2025-11-30",
    )

    # Filter for subjects with actigraphy
    df_filtered = processor.filter_subjects_with_actigraphy(df_merged)

    # Apply processing pipeline
    df_final = processor.apply_processing_pipeline(df=df_filtered, overwrite=True)

    print(f"\n  [OK] Processing complete: {len(df_final):,} subjects in final dataset")

    # --------------------------------------------------
    # 5. SAVE FINAL DATASET
    # --------------------------------------------------
    print("\n[5/5] Saving final dataset...")

    processor.save_final_dataset(
        df=df_final,
        out_dir=out_dir,
        filename_base="ukb_final_dataset"
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Initial subjects extracted: {len(df_merged):,}")
    print(f"Subjects with actigraphy: {len(df_filtered):,}")
    print(f"Final processed subjects: {len(df_final):,}")
    print(f"\nOutput saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

