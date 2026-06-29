"""
Run ABK Collection Pipeline
============================
Collects, cleans, and merges actigraphy batch extractions (Gait, Sleep, RBD)
using the ``ABKCollector`` class.

Usage:
    python notebook/run_abk_collection.py

Inputs are identical to ``collect_sleep_abk_metrics.py``.
At the end, a consolidated stage-by-stage report is printed via
``collector.report.print_report()``.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from config.config import config
from tabulate import tabulate
from library.abk_collection import ABKCollector
import re

# ============================================================================
# Batch-specific loaders  (thin wrappers around file I/O)
# ============================================================================
def _normalize(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """Normalize identifiers and extract visit number + eid."""
    df = df.copy()
    # Ensure string type
    df[id_col] = df[id_col].astype(str)

    # Remove .cwa suffix (case-insensitive)
    df[id_col] = df[id_col].str.replace(
        r"\.cwa$",
        "",
        regex=True,
        flags=re.IGNORECASE
    )
    # df[id_col] = df[id_col].astype(int)

    df["visit_number"] = (
        df[id_col]
        .str.extract(r"_(\d+)(?:_\d+)?$", expand=False)
        .astype("Int64")
    )

    df["eid"] = df[id_col].apply(lambda x: x.split("_")[0]).astype(int)
    return df



def first_batch_features(
    path_first_batch: Path,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load gait, sleep, and RBD features from the first batch."""

    # gait
    gait_first_parquet = path_first_batch / "F_Gait_abk_merged.parquet"
    rbd_scores_first_csv = path_first_batch / "RBD_Sleep_Score_all_abk.csv"

    if not gait_first_parquet.exists():
        if not gait_first_parquet.exists() or overwrite:
            gait_csvs = list(
                (path_first_batch / "gait_sparse").glob("F_Gait_abk*.csv")
            )
            df_gait_first = pd.concat(
                [pd.read_csv(f) for f in gait_csvs], ignore_index=True
            )
            df_gait_first["batch"] = "first"
            df_gait_first.to_parquet(gait_first_parquet, index=False)
    else:
        df_gait_first = pd.read_parquet(gait_first_parquet)

    # rbd scores
    if rbd_scores_first_csv.exists():
        df_rbd_first = pd.read_csv(rbd_scores_first_csv)
    else:
        df_rbd_first = pd.DataFrame()

    # sleep
    df_sleep_first = pd.read_parquet(path_first_batch / "F_Sleep_abk.parquet")
    df_sleep_first["batch"] = "first"
    df_rbd_first["batch"] = "first"

    df_gait_first = _normalize(df=df_gait_first, id_col='ID')
    df_sleep_first = _normalize(df=df_sleep_first, id_col='ID')
    df_rbd_first = _normalize(df=df_rbd_first, id_col='ID')

    return df_gait_first, df_sleep_first, df_rbd_first


def second_batch_features(
    path_second_batch: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-merged gait, sleep, and RBD features from the second batch."""
    df_gait = pd.read_parquet(path_second_batch / "F_Gait_abk_merged.parquet")
    # df_sleep = pd.read_parquet(path_second_batch / "F_Sleep_abk_merged.parquet")
    # df_rbd = pd.read_parquet(
    #     path_second_batch / "RBD_Sleep_Score_all_abk_merged.parquet"
    # )

    df_gait = _normalize(df=df_gait, id_col='ID')
    df_sleep, df_rbd = pd.DataFrame(), pd.DataFrame()
    # df_sleep = _normalize(df=df_sleep, id_col='ID')
    # df_rbd = _normalize(df=df_rbd, id_col='ID')
    return df_gait, df_sleep, df_rbd


def current_batch_features(
    merged_outputs: Dict[str, pd.DataFrame],
    batch_name: str = "current",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract gait, sleep, and RBD from the current batch merged outputs."""
    df_gait = merged_outputs["F_Gait_abk"].assign(batch=batch_name).drop(
        columns=["batch_id"]
    )
    df_sleep = merged_outputs["F_Sleep_abk"].copy()
    df_rbd = merged_outputs["RBD_Sleep_Score_all_abk"].copy()
    df_sleep["batch"] = batch_name

    df_gait = _normalize(df=df_gait, id_col='ID')
    df_sleep = _normalize(df=df_sleep, id_col='ID')
    df_rbd = _normalize(df=df_rbd, id_col='ID')
    return df_gait, df_sleep, df_rbd


# ============================================================================
# Main
# ============================================================================
def main(
    current_folder: str = "ActigStfRecords",
    overwrite: bool = False,
) -> None:
    """Collect, clean, and merge actigraphy batch extractions (Gait, Sleep, RBD).

    Parameters
    ----------
    current_folder : str
        Name of the current batch folder from the ABK model.
    overwrite : bool
        Overwrite existing merged outputs.
    """
    path_root = config.get("paths")["root"]
    path_abk = config.get("paths")["abk_matlab_out_dir"] / current_folder
    path_res = path_root / f"results/batch_extraction_report/{current_folder}"

    path_actig = config.get("paths")["actig_extracted"]["root"]
    path_output_abk_to_project = path_actig / current_folder

    # merged output destinations
    path_merged_final = config.get("paths")["actig_extracted"]["merged"]

    path_merged_gait = config.get("paths")["actig_extracted"]["merged_gait"]
    path_merged_sleep = config.get("paths")["actig_extracted"]["merged_sleep"]
    path_merged_rbd = config.get("paths")["actig_extracted"]["rbd_scores"]


    #%% ---- Instantiate collector ----
    collector = ABKCollector()
    # =================================================================
    # STEP 1 — transfer target from abk if available
    # =================================================================
    collector.get_targets_from_abk(
        path_abk=path_abk,
        target_map=None,  # set by constructor
        output_dir=path_output_abk_to_project,
        overwrite=overwrite,
    )
    # =================================================================
    # STEP 2 — File counts matrix (cached)
    # =================================================================
    df_matrix = collector.load_file_counts_matrix(
        path_abk=path_abk,
        targets=None,
        out_xlsx=path_res / "file_counts_matrix.xlsx",
        overwrite=overwrite,
    )

    # =================================================================
    # STEP 3 — Load previous batches
    # =================================================================
    (dfs_gait,
     dfs_sleep,
     dfs_rbd,
     dfs_rbd_avg) = collector.load_existing_batches(
                                    dir_batches=config.get("paths")["actig_extracted"]['root'],
                                    dir_merged_output=path_merged_final)


    # =================================================================
    # STEP 4 — Merge GAIT (no Date column → date_col=None)
    # # =================================================================
    df_gait_combined, gait_counts, gait_unique = collector.merge_batches(
        dfs_gait,
        dataset_name="Gait",
        id_col="ID",
        date_col=None,
        leakage_order=tuple(dfs_gait.keys()),
    )

    # Columns to evaluate (exclude ID and eid)
    cols_to_check = df_gait_combined.columns.difference(["ID", "eid", 'visit_number',
                                                         'dur_j, dur_w'])

    # For each ID → check if all values (across all rows & columns) are NaN
    ids_all_nan = (
        df_gait_combined
        .groupby("ID")[cols_to_check]
        .apply(lambda x: x.isna().all().all())
    )

    # Count how many IDs satisfy the condition
    n_ids_all_nan = ids_all_nan.sum()

    print(f"IDs with all rows entirely NaN (excluding ID/eid): {n_ids_all_nan}")

    # =================================================================
    # STEP 5 — Merge SLEEP (date_col="Date")
    # =================================================================
    df_sleep_combined, sleep_counts, sleep_unique = collector.merge_batches(
        dfs_sleep,
        dataset_name="Sleep",
        id_col="ID",
        date_col="Date",
        leakage_order=tuple(dfs_sleep.keys()),
    )

    # =================================================================
    # STEP 6 — Merge RBD SCORES (date_col="Date")
    # =================================================================
    df_rbd_combined, rbd_counts, rbd_unique = collector.merge_batches(
        dfs_rbd,
        dataset_name="RBD",
        id_col="ID",
        date_col="Date",
        leakage_order=tuple(dfs_rbd.keys()),
    )
    df_rbd_combined.columns = [col.lower() for col in df_rbd_combined.columns]
    df_gait_combined.columns = [col.lower() for col in df_gait_combined.columns]
    df_sleep_combined.columns = [col.lower() for col in df_sleep_combined.columns]
    # =================================================================
    # STEP 7 — Save cleaned datasets
    # =================================================================
    dfs = [
        (df_gait_combined, path_merged_gait),
        (df_sleep_combined, path_merged_sleep),
        (df_rbd_combined, path_merged_rbd),
    ]

    for df, path in dfs:
        df.to_parquet(path, index=False)

     # ================================================================================
    # | dataset   | stage                 |   n_rows |   n_unique_ids |   n_unique_nights |
    # |-----------|-----------------------|----------|----------------|-------------------|
    # | Gait      | before_cleaning       |   168569 |         168348 |               nan |
    # | Gait      | after_leakage_removal |   114250 |         114029 |               nan |
    # | Gait      | final                 |   114029 |         114029 |               nan |
    # | Sleep     | before_cleaning       |   777197 |         113624 |            777194 |
    # | Sleep     | after_leakage_removal |   777197 |         113624 |            777194 |
    # | Sleep     | final                 |   777194 |         113624 |            777194 |
    # | RBD       | before_cleaning       |   694066 |         108322 |            694064 |
    # | RBD       | after_leakage_removal |   694066 |         108322 |            694064 |
    # | RBD       | final                 |   694064 |         108322 |            694064 |
    # ================================================================================

    print(f"\nSaved -> {path_merged_gait}")
    print(f"Saved -> {path_merged_sleep}")
    print(f"Saved -> {path_merged_rbd}")

    # =================================================================
    # STEP 8 — FINAL REPORT  (all datasets, all stages)
    # =================================================================
    collector.report.print_report()

    # Optionally get as a DataFrame for further analysis:
    summary_df = collector.report.summary_df()
    # summary_df.to_excel(path_merged_gait.parent / "summary_report.xlsx")

    # get the ids in a frame
    ids_gait = set(df_gait_combined["id"].dropna().unique()) if not df_gait_combined.empty else set()
    ids_sleep = set(df_sleep_combined["id"].dropna().unique()) if not df_sleep_combined.empty else set()
    ids_rbd = set(df_rbd_combined["id"].dropna().unique()) if not df_rbd_combined.empty else set()

    all_ids = ids_gait | ids_sleep | ids_rbd

    df_ids = pd.DataFrame({
        "id": list(all_ids)
    })

    # ==========================================================
    # 1️⃣ LOCAL MODALITY COVERAGE
    # ==========================================================

    # Modality membership
    df_ids["in_gait"] = df_ids["id"].isin(ids_gait)
    df_ids["in_sleep"] = df_ids["id"].isin(ids_sleep)
    df_ids["in_rbd"] = df_ids["id"].isin(ids_rbd)

    df_ids["n_sources"] = (
        df_ids[["in_gait", "in_sleep", "in_rbd"]]
        .sum(axis=1)
    )

    df_ids["visit_number"] = (
        df_ids["id"]
        .str.extract(r"_(\d+)(?:_\d+)?$", expand=False)
        .astype("Int64")
    )

    # Save local report
    path_local_report = path_merged_gait.parent / "ids_local_modality_report.xlsx"
    df_ids.to_excel(path_local_report, index=False)

    print("\n=== LOCAL MODALITY SUMMARY ===")
    print(df_ids[["in_gait", "in_sleep", "in_rbd"]].sum())
    print(f"Complete (all three): {(df_ids['n_sources'] == 3).sum()}")

    # ==========================================================
    # 2️⃣ LOAD + NORMALIZE SERVER AUDIT
    # ==========================================================

    path_server_root = config.get('paths')['actig_extracted']['root']
    df_ids_server = pd.read_csv(path_server_root / "actigraphy_file_audit.csv")

    # Keep UKBB only before normalizing — non-UKBB IDs (e.g. SHAS*) cannot be cast to int eid
    df_ids_server = df_ids_server.loc[
        df_ids_server["file"].str.contains("90001", na=False)
    ].copy()

    df_ids_server = _normalize(df_ids_server, id_col='file')

    df_ids_server["id"] = df_ids_server["file"]

    # ==========================================================
    # 3️⃣ SERVER VS LOCAL COMPARISON
    # ==========================================================

    local_ids = set(df_ids["id"].dropna().unique())
    server_ids = set(df_ids_server["id"].dropna().unique())

    df_ids_server["in_local"] = df_ids_server["id"].isin(local_ids)

    missing_on_local = server_ids - local_ids
    missing_on_server = local_ids - server_ids

    print("\n=== SERVER RECONCILIATION ===")
    print(f"Total server UKBB IDs: {len(server_ids)}")
    print(f"Total local IDs: {len(local_ids)}")
    print(f"Missing locally: {len(missing_on_local)}")
    print(f"Missing on server: {len(missing_on_server)}")

    # ==========================================================
    # 4️⃣ ADD MODALITY FLAGS TO SERVER FRAME
    # ==========================================================

    df_ids_server["in_gait"] = df_ids_server["id"].isin(ids_gait)
    df_ids_server["in_sleep"] = df_ids_server["id"].isin(ids_sleep)
    df_ids_server["in_rbd"] = df_ids_server["id"].isin(ids_rbd)

    df_ids_server["n_sources"] = (
        df_ids_server[["in_gait", "in_sleep", "in_rbd"]]
        .sum(axis=1)
    )

    df_ids_server["has_all_three"] = df_ids_server["n_sources"] == 3

    print("\n=== SERVER MODALITY SUMMARY ===")
    print(df_ids_server[["in_gait", "in_sleep", "in_rbd"]].sum())
    print(f"Complete (all three): {df_ids_server['has_all_three'].sum()}")

    # ==========================================================
    # 5️⃣ SAVE ALL OUTPUTS
    # ==========================================================

    # A. Server full audit with modality flags
    df_ids_server.to_excel(
        path_server_root / "ids_server_full_audit.xlsx",
        index=False
    )

    # B. Missing locally
    pd.DataFrame({"ID_missing_locally": list(missing_on_local)}).to_excel(
        path_server_root / "ids_server_missing_locally.xlsx",
        index=False
    )

    # C. Missing on server
    pd.DataFrame({"ID_missing_on_server": list(missing_on_server)}).to_excel(
        path_server_root / "ids_local_missing_on_server.xlsx",
        index=False
    )


if __name__ == "__main__":
    main()

