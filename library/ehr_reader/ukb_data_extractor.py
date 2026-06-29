"""
UK Biobank Data Extractor

This module provides the UkbDataExtractor class for extracting data from
UK Biobank CSV files using either Dask or Pandas.
"""

from typing import Dict, Set
from pathlib import Path
import pandas as pd
import dask.dataframe as dd


class UkbDataExtractor:
    """
    Handles data extraction from UK Biobank CSV files.

    This class extracts data from multiple CSV files, supporting both Dask
    (for faster parallel processing) and Pandas (as a fallback). It can also
    merge extracted dataframes and clean up temporary files.

    Attributes:
        dir_new_data: Directory containing the UK Biobank CSV data files
        out_dir: Output directory for temporary files
    """

    def __init__(self, dir_new_data: Path, out_dir: Path):
        """
        Initialize the UkbDataExtractor.

        Args:
            dir_new_data: Directory containing the UK Biobank CSV data files
            out_dir: Output directory for temporary files
        """
        self.dir_new_data = dir_new_data
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract_all_csv_data(
        self,
        df_matched: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract data from all CSV files.

        Args:
            df_matched: DataFrame with matched fields (columns: csv_name, column_name, is_match)

        Returns:
            Dictionary mapping CSV names to extracted DataFrames
        """
        print("\nExtracting data from CSV files...")
        extracted_dfs = {}

        for csv_name in df_matched['csv_name'].unique():
            csv_file = self.dir_new_data / f"UKB97043_{csv_name}_2025Nov.csv"
            field_ids = df_matched.loc[
                df_matched['csv_name'] == csv_name, 'column_name'
            ].tolist()
            field_ids = set(field_ids)

            if 'eid' not in field_ids:
                field_ids.add('eid')

            if not csv_file.exists():
                print(f"[WARN]  {csv_file} not found, skipping.")
                continue

            print(f"\n>> Processing {csv_name}")

            try:
                extracted_dfs[csv_name] = self._extract_csv_data_dask(
                    csv_name=csv_name,
                    csv_file=csv_file,
                    field_ids=field_ids,
                    blocksize="128MB",
                )
                print(f"  [OK] Dask succeeded for {csv_name}")

            except Exception as e:
                print(f"  [WARN] Dask failed for {csv_name}: {e}")
                print("  Falling back to pandas chunked extraction...")

                extracted_dfs[csv_name] = self._extract_csv_data_pandas(
                    csv_name=csv_name,
                    csv_file=csv_file,
                    field_ids=field_ids,
                )
                print(f"  [OK] Pandas fallback succeeded for {csv_name}")

        return extracted_dfs

    def merge_extracted_data(
        self,
        extracted_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge all extracted DataFrames on 'eid'.

        Args:
            extracted_dfs: Dictionary of DataFrames to merge

        Returns:
            Merged DataFrame
        """
        print("\nMerging all extracted data...")

        normalized = {}
        for name, df in extracted_dfs.items():
            df = df.copy()
            df["eid"] = df["eid"].astype("int64")
            normalized[name] = df

        df_merged = None

        for csv_name, df in normalized.items():
            if df_merged is None:
                df_merged = df
            else:
                df_merged = df_merged.merge(df, on="eid", how="outer")

            print(
                f"  After merging {csv_name}: "
                f"{len(df_merged):,} rows, {len(df_merged.columns)} columns"
            )

        print(
            f"\n[OK] Initial merge complete: "
            f"{len(df_merged):,} rows, {len(df_merged.columns)} columns"
        )

        # Rename field-53 visit dates and actigraphy wear-time markers to
        # human-readable names so the raw parquet is self-documenting.
        col_follow_up = {
            col: col.replace("p53", "follow_up_date")
            for col in df_merged.columns
            if col.startswith("p53_")
        }
        column_mapper = {"p90003": "wear_time_start", "p90011": "wear_time_end"}
        column_mapper.update(col_follow_up)
        df_merged = df_merged.rename(columns=column_mapper)

        if col_follow_up:
            print(f"  Renamed {len(col_follow_up)} p53_* -> follow_up_date_* columns")

        return df_merged

    def cleanup_temp_files(self, extracted_dfs: Dict[str, pd.DataFrame]) -> None:
        """
        Clean up temporary files created during extraction.

        Args:
            extracted_dfs: Dictionary of extracted DataFrames
        """
        print("\nCleaning up temporary files...")
        for csv_name in extracted_dfs.keys():
            temp_file = self.out_dir / f"temp_{csv_name}.csv"
            if temp_file.exists():
                temp_file.unlink()
                print(f"  Deleted {temp_file.name}")

    def _extract_csv_data_dask(
        self,
        csv_name: str,
        csv_file: Path,
        field_ids: Set[str],
        blocksize: str = "128MB",
    ) -> pd.DataFrame:
        """Extract data from a CSV using Dask for faster, parallel IO."""
        print(f"\nProcessing {csv_name} (Dask)...")

        dtype_map = {col: "string" for col in field_ids}

        ddf = dd.read_csv(
            csv_file,
            usecols=field_ids,
            dtype=dtype_map,
            blocksize=blocksize,
            assume_missing=True,
        )

        df = ddf.compute()
        print(f"  [OK] Extracted {len(df):,} rows, {len(df.columns)} columns")
        return df

    def _extract_csv_data_pandas(
        self,
        csv_name: str,
        csv_file: Path,
        field_ids: Set[str],
        chunksize: int = 100_000
    ) -> pd.DataFrame:
        """Extract data from a single CSV file using Pandas chunked reading."""
        print(f"\nProcessing {csv_name} (Pandas)...")
        print(f"  Fields to extract: {len(field_ids)}")

        first_write = True
        temp_out = self.out_dir / f"temp_{csv_name}.csv"

        for i, chunk in enumerate(pd.read_csv(
            csv_file,
            usecols=field_ids,
            chunksize=chunksize,
            low_memory=False
        )):
            chunk.to_csv(
                temp_out,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False

            if (i + 1) % 10 == 0:
                print(f"    Processed {(i + 1) * chunksize:,} rows...")

        df = pd.read_csv(temp_out)
        print(f"  [OK] Extracted {len(df):,} rows, {len(df.columns)} columns")
        return df
