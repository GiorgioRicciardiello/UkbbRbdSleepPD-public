"""
ABK Actigraphy Collection Module

Collects, cleans, merges, and reports on actigraphy batch extractions
(Gait, Sleep, RBD) from the ABK MATLAB model.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from tabulate import tabulate
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ID normalisation
# ---------------------------------------------------------------------------

def _normalize_ids(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """Strip .cwa suffix, parse visit_number and eid from ID column.

    Operates on a copy; never mutates the input frame.
    """
    df = df.copy()
    df[id_col] = (
        df[id_col]
        .astype(str)
        .str.replace(r"\.cwa$", "", regex=True, flags=re.IGNORECASE)
    )
    df["visit_number"] = (
        df[id_col]
        .str.extract(r"_(\d+)(?:_\d+)?$", expand=False)
        .astype("Int64")
    )
    df["eid"] = df[id_col].apply(lambda x: x.split("_")[0]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Batch specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _BatchSpec:
    """Per-batch file layout descriptor."""
    name: str
    dirname: str
    gait_file: Optional[str]   # None → no gait for this batch
    sleep_file: str
    rbd_file: str
    rbd_is_csv: bool = False   # first_batch RBD is a CSV, not parquet


# Ordered from earliest to latest — concat order determines deduplication
# priority (earlier batch = authoritative on (ID, Date) ties).
_BATCH_SPECS: tuple = (
    _BatchSpec(
        name="first",
        dirname="first_batch",
        gait_file="F_Gait_abk_merged.parquet",
        sleep_file="F_Sleep_abk.parquet",       # no _merged suffix
        rbd_file="RBD_Sleep_Score_all_abk.csv", # CSV, col: source_file
        rbd_is_csv=True,
    ),
    _BatchSpec(
        name="second",
        dirname="second_batch",
        gait_file="F_Gait_abk_merged.parquet",
        sleep_file="F_Sleep_abk_merged.parquet",
        rbd_file="RBD_Sleep_Score_all_abk_merged.parquet",
    ),
    _BatchSpec(
        name="third",
        dirname="third_batch",
        gait_file="F_Gait_abk_merged.parquet",
        sleep_file="F_Sleep_abk_merged.parquet",
        rbd_file="RBD_Sleep_Score_all_abk_merged.parquet",
    ),
    _BatchSpec(
        name="fourth",
        dirname="fourth_batch",
        gait_file="F_Gait_abk_merged.parquet",
        sleep_file="F_Sleep_abk_merged.parquet",
        rbd_file="RBD_Sleep_Score_all_abk_merged.parquet",
    ),
    _BatchSpec(
        name="data_remaining",
        dirname="DataRemaining",
        gait_file="F_Gait_abk_merged.parquet",
        sleep_file="F_Sleep_abk_merged.parquet",
        rbd_file="RBD_Sleep_Score_all_abk_merged.parquet",
    ),
    _BatchSpec(
        name="data_only_sleep_rbd",
        dirname="DataOnlySleepRBD",
        gait_file=None,                          # no gait in this batch
        sleep_file="F_Sleep_abk_merged.parquet",
        rbd_file="RBD_Sleep_Score_all_abk_merged.parquet",
    ),
    _BatchSpec(
        name="actig_stf_records",
        dirname="ActigStfRecords",
        gait_file="F_Gait_abk_merged.parquet",
        sleep_file="F_Sleep_abk_merged.parquet",
        rbd_file="RBD_Sleep_Score_all_abk_merged.parquet",
    ),
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class MergeReport:
    """Track merge statistics and leakage for reporting."""
    dataset_name: str
    before_cleaning: int = 0
    after_leakage: int = 0
    final: int = 0
    unique_ids_before: int = 0
    unique_ids_after: int = 0
    unique_ids_final: int = 0
    unique_nights_before: int = 0
    unique_nights_after: int = 0
    unique_nights_final: int = 0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "before_cleaning": self.before_cleaning,
            "after_leakage_removal": self.after_leakage,
            "final": self.final,
            "unique_ids_before": self.unique_ids_before,
            "unique_ids_after": self.unique_ids_after,
            "unique_ids_final": self.unique_ids_final,
            "unique_nights_before": self.unique_nights_before,
            "unique_nights_after": self.unique_nights_after,
            "unique_nights_final": self.unique_nights_final,
        }


class Report:
    """Consolidated report for all merge operations."""

    def __init__(self) -> None:
        self.merges: Dict[str, MergeReport] = {}

    def add(self, merge_report: MergeReport) -> None:
        """Add a merge report."""
        self.merges[merge_report.dataset_name] = merge_report

    def summary_df(self) -> pd.DataFrame:
        """Return summary as DataFrame."""
        data = [mr.to_dict() for mr in self.merges.values()]
        return pd.DataFrame(data)

    def print_report(self) -> None:
        """Print formatted report."""
        print("\n" + "=" * 100)
        print("  MERGE SUMMARY REPORT (all datasets, all stages)")
        print("=" * 100)

        rows = []
        for mr in self.merges.values():
            rows.append({
                "dataset": mr.dataset_name,
                "stage": "before_cleaning",
                "n_rows": mr.before_cleaning,
                "n_unique_ids": mr.unique_ids_before,
                "n_unique_nights": mr.unique_nights_before if mr.unique_nights_before > 0 else "N/A",
            })
            rows.append({
                "dataset": mr.dataset_name,
                "stage": "after_leakage_removal",
                "n_rows": mr.after_leakage,
                "n_unique_ids": mr.unique_ids_after,
                "n_unique_nights": mr.unique_nights_after if mr.unique_nights_after > 0 else "N/A",
            })
            rows.append({
                "dataset": mr.dataset_name,
                "stage": "final",
                "n_rows": mr.final,
                "n_unique_ids": mr.unique_ids_final,
                "n_unique_nights": mr.unique_nights_final if mr.unique_nights_final > 0 else "N/A",
            })

        print(tabulate(rows, headers="keys", tablefmt="grid"))
        print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class ABKCollector:
    """
    Collector for actigraphy batch extractions (Gait, Sleep, RBD).

    Handles merging, deduplication, and reporting across multiple batches
    and extraction iterations.
    """

    def __init__(self) -> None:
        """Initialize collector with default target map."""
        self.target_map = {
            "F_Gait_abk.csv": "Gait features",
            "F_Sleep_abk.csv": "Sleep features",
            "RBD_Sleep_Score_all_abk.csv": "RBD scores",
        }
        self.report = Report()
        self.merged_outputs: Dict[str, pd.DataFrame] = {}

    def get_targets_from_abk(
        self,
        path_abk: Path,
        target_map: Optional[Dict[str, str]] = None,
        output_dir: Optional[Path] = None,
        overwrite: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load merged target CSVs from batch directories.

        Parameters
        ----------
        path_abk : Path
            Root path containing batch_* subdirectories
        target_map : dict, optional
            Mapping {filename: description}. If None, uses default.
        output_dir : Path, optional
            Directory to cache merged outputs as parquet
        overwrite : bool
            Recompute even if parquet cache exists

        Returns
        -------
        dict
            {target_name: DataFrame}
        """
        if target_map is None:
            target_map = self.target_map

        if output_dir is None:
            output_dir = path_abk

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        for filename in target_map.keys():
            key = filename.replace(".csv", "")
            out_path = output_dir / f"{key}_merged.parquet"

            # Load cached parquet if available
            if out_path.exists() and not overwrite:
                print(f"[LOAD] {out_path.name}")
                outputs[key] = pd.read_parquet(out_path)
                continue

            print(f"[BUILD] Merging {filename}...")
            collector = []

            # Scan all batch_* directories
            for batch_dir in tqdm(path_abk.iterdir(), desc=f"Scanning {filename}", leave=False):
                if not (batch_dir.is_dir() and
                        batch_dir.name.startswith("batch_") and
                        "test" not in batch_dir.name):
                    continue

                f = batch_dir / filename
                if f.exists():
                    df = pd.read_csv(f, low_memory=False)
                    df["batch_id"] = batch_dir.name
                    collector.append(df)

            if collector:
                merged = pd.concat(collector, ignore_index=True)
                outputs[key] = merged
                merged.to_parquet(out_path, index=False)
                print(f"  -> Saved {out_path.name} ({merged.shape})")
            else:
                outputs[key] = pd.DataFrame()
                print(f"  -> No files found for {filename}")

        self.merged_outputs = outputs
        return outputs

    def load_file_counts_matrix(
        self,
        path_abk: Path,
        targets: Optional[Dict[str, pd.DataFrame]] = None,
        out_xlsx: Optional[Path] = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Build matrix of file counts per batch.

        Parameters
        ----------
        path_abk : Path
            Root ABK path
        targets : dict, optional
            Target dataframes. If None, uses self.merged_outputs
        out_xlsx : Path, optional
            Save matrix to Excel
        overwrite : bool
            Recompute even if file exists

        Returns
        -------
        pd.DataFrame
            File counts matrix
        """
        if targets is None:
            targets = self.merged_outputs

        if out_xlsx and out_xlsx.exists() and not overwrite:
            print(f"[LOAD] {out_xlsx.name}")
            return pd.read_excel(out_xlsx, index_col=0)

        # Count files per batch per target
        batches = sorted([
            d.name for d in path_abk.iterdir()
            if d.is_dir() and d.name.startswith("batch_")
        ])

        matrix_data = {}
        for batch in batches:
            batch_dir = path_abk / batch
            matrix_data[batch] = {}

            for filename in self.target_map.keys():
                csv_file = batch_dir / filename
                matrix_data[batch][filename] = 1 if csv_file.exists() else 0

        df_matrix = pd.DataFrame(matrix_data).T

        if out_xlsx:
            out_xlsx.parent.mkdir(parents=True, exist_ok=True)
            df_matrix.to_excel(out_xlsx)
            print(f"[SAVE] {out_xlsx.name}")

        return df_matrix

    def load_existing_batches(
        self,
        dir_batches: Path,
        dir_merged_output: Path,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame],
               Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load all known actigraphy batch directories.

        Iterates over ``_BATCH_SPECS`` in chronological order. Earlier batches
        are inserted first so that ``merge_batches`` retains the earliest
        occurrence when deduplicating on (ID, Date).

        Normalisation (``_normalize_ids``) is applied to every loaded frame,
        which strips the ``.cwa`` suffix, parses ``visit_number``, and sets
        ``eid``.

        Parameters
        ----------
        dir_batches : Path
            Root of extracted-features store (``actig_extracted_features/``).
        dir_merged_output : Path
            Destination for final merged files (not used here; kept for API
            compatibility).

        Returns
        -------
        tuple
            (dfs_gait, dfs_sleep, dfs_rbd, dfs_rbd_avg)
            Each is an ordered dict {batch_name: DataFrame}.
        """
        dfs_gait: Dict[str, pd.DataFrame] = {}
        dfs_sleep: Dict[str, pd.DataFrame] = {}
        dfs_rbd: Dict[str, pd.DataFrame] = {}
        dfs_rbd_avg: Dict[str, pd.DataFrame] = {}

        for spec in _BATCH_SPECS:
            batch_dir = dir_batches / spec.dirname
            if not batch_dir.exists():
                print(f"[SKIP] {spec.dirname} — directory not found")
                continue

            # --- Gait (subject-level, no Date column) ---
            if spec.gait_file is not None:
                gait_path = batch_dir / spec.gait_file
                if gait_path.exists():
                    df = pd.read_parquet(gait_path)
                    df["batch"] = spec.name
                    dfs_gait[spec.name] = _normalize_ids(df)
                else:
                    print(f"[WARN] {spec.dirname}: gait file missing ({spec.gait_file})")

            # --- Sleep (night-level, has Date column) ---
            sleep_path = batch_dir / spec.sleep_file
            if sleep_path.exists():
                df = pd.read_parquet(sleep_path)
                df["batch"] = spec.name
                dfs_sleep[spec.name] = _normalize_ids(df)
            else:
                print(f"[WARN] {spec.dirname}: sleep file missing ({spec.sleep_file})")

            # --- RBD scores (night-level, has Date column) ---
            rbd_path = batch_dir / spec.rbd_file
            if rbd_path.exists():
                df = (
                    pd.read_csv(rbd_path)
                    if spec.rbd_is_csv
                    else pd.read_parquet(rbd_path)
                )
                # First batch CSV uses 'source_file'; standardise to 'batch_id'
                if "source_file" in df.columns:
                    df = df.rename(columns={"source_file": "batch_id"})
                df["batch"] = spec.name
                dfs_rbd[spec.name] = _normalize_ids(df)
            else:
                print(f"[WARN] {spec.dirname}: rbd file missing ({spec.rbd_file})")

            # --- RBD subject-level averages (optional, not in all batches) ---
            rbd_avg_path = batch_dir / "RBD_Sleep_Score_avg_abk_merged.parquet"
            if rbd_avg_path.exists():
                df = pd.read_parquet(rbd_avg_path)
                df["batch"] = spec.name
                dfs_rbd_avg[spec.name] = df

        print(f"\n[BATCHES LOADED] Gait  : {list(dfs_gait.keys())}")
        print(f"[BATCHES LOADED] Sleep : {list(dfs_sleep.keys())}")
        print(f"[BATCHES LOADED] RBD   : {list(dfs_rbd.keys())}")
        print(f"[BATCHES LOADED] RBD avg: {list(dfs_rbd_avg.keys())}")

        return dfs_gait, dfs_sleep, dfs_rbd, dfs_rbd_avg

    def merge_batches(
        self,
        dfs: Dict[str, pd.DataFrame],
        dataset_name: str = "Dataset",
        id_col: str = "ID",
        date_col: Optional[str] = None,
        leakage_order: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Merge multiple batch dataframes with leakage detection.

        Parameters
        ----------
        dfs : dict
            {batch_name: DataFrame}
        dataset_name : str
            Name for reporting
        id_col : str
            Column name for entity ID
        date_col : str, optional
            Column for temporal deduplication.  If provided, uniqueness is
            enforced on (id_col, date_col) — appropriate for night-level data
            (Sleep, RBD).  If None, uniqueness is enforced on id_col alone —
            appropriate for subject-level data (Gait).
        leakage_order : tuple, optional
            Order to check for duplicates (earlier = priority)

        Returns
        -------
        tuple
            (merged_df, counts_df, unique_df)
        """
        if not dfs or all(df.empty for df in dfs.values()):
            print(f"[WARN] No data for {dataset_name}")
            mr = MergeReport(dataset_name)
            self.report.add(mr)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Concatenate all batches; dict insertion order = dedup priority
        all_dfs = [df for df in dfs.values() if not df.empty]
        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Track stats before cleaning
        n_rows_before = len(df_combined)
        n_ids_before = df_combined[id_col].nunique() if id_col in df_combined.columns else 0
        n_nights_before = (
            df_combined[date_col].nunique()
            if date_col and date_col in df_combined.columns
            else 0
        )

        # Deduplication: (ID, Date) for night-level data; ID only for gait
        dedup_subset = [id_col, date_col] if date_col and date_col in df_combined.columns else [id_col]
        df_combined = df_combined.drop_duplicates(subset=dedup_subset, keep="first")

        n_rows_after = len(df_combined)
        n_ids_after = df_combined[id_col].nunique() if id_col in df_combined.columns else 0
        n_nights_after = (
            df_combined[date_col].nunique()
            if date_col and date_col in df_combined.columns
            else 0
        )

        # Final cleanup: drop rows where ID is null
        df_final = df_combined.dropna(subset=[id_col])

        n_rows_final = len(df_final)
        n_ids_final = df_final[id_col].nunique() if id_col in df_final.columns else 0
        n_nights_final = (
            df_final[date_col].nunique()
            if date_col and date_col in df_final.columns
            else 0
        )

        # Build report
        mr = MergeReport(
            dataset_name=dataset_name,
            before_cleaning=n_rows_before,
            after_leakage=n_rows_after,
            final=n_rows_final,
            unique_ids_before=n_ids_before,
            unique_ids_after=n_ids_after,
            unique_ids_final=n_ids_final,
            unique_nights_before=n_nights_before,
            unique_nights_after=n_nights_after,
            unique_nights_final=n_nights_final,
        )
        self.report.add(mr)

        # Count unique entities per batch
        counts = pd.DataFrame({
            "batch": list(dfs.keys()),
            "n_rows": [len(df) for df in dfs.values()],
            "n_unique_ids": [
                df[id_col].nunique() if id_col in df.columns else 0
                for df in dfs.values()
            ],
        })

        # Unique per batch (before merge)
        unique = pd.DataFrame({
            "batch": list(dfs.keys()),
            "unique_ids": [
                set(df[id_col].dropna().unique()) if id_col in df.columns else set()
                for df in dfs.values()
            ],
        })

        print(f"\n[{dataset_name.upper()}] Merge Complete:")
        print(f"  Before: {n_rows_before:,} rows, {n_ids_before:,} unique IDs")
        print(f"  After leakage removal: {n_rows_after:,} rows, {n_ids_after:,} unique IDs")
        print(f"  Final: {n_rows_final:,} rows, {n_ids_final:,} unique IDs")

        return df_final, counts, unique
