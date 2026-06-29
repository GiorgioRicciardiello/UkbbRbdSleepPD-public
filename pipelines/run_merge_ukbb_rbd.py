"""
Risk-group pipeline management and path manipulation utility module.

This module contains utility functions for file path manipulation and a
comprehensive pipeline implementation for processing risk groups based
on validation handling and robust error control. It includes steps for
validation column detection, risk group computations, and output handling.

Functions
---------
- insert_after_data(path, folder)
- risk_group_pipeline(df_risk, config, outcomes, final_dir, thresholds_dir, rbd_col)
"""
import shutil
import pandas as pd
from config.config import config, outcomes
from library.column_registry import col_surv_time
from pathlib import Path
from typing import Dict, List, Optional
from library.risk.risk_groups import (
    # run_compute_risk_groups,  # DEPRECATED: per-outcome risk groups replaced by agnostic stratification
    run_compute_risk_group_rbd_only,
)
import traceback

def insert_after_data(path: Path, folder: str) -> Path:
    """
    We are testing the model with abk rbd scores and katarina rbd scores, therefore we are assigning specifics paths
    to each result
    :param path:
    :param folder:
    :return:
    """
    parts = list(path.parts)

    if "data" not in parts:
        raise ValueError(f"'data' not found in path: {path}")

    idx = parts.index("data") + 1
    new_parts = parts[:idx] + [folder] + parts[idx:]

    return Path(*new_parts)



def risk_group_pipeline(
    df_risk: pd.DataFrame,
    config: Dict,
    outcomes: List[str],
    final_dir: Path,
    thresholds_dir: Path,
    rbd_col: str = "rbd_prob_class1",
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Full risk-group pipeline with validation handling and robust error control.

    Parameters
    ----------
    df_risk : pd.DataFrame
    config : dict
    outcomes : list[str]
    final_dir : Path
    thresholds_dir : Path
    rbd_col : str

    Returns
    -------
    dict
        {
            "df_risk_val": DataFrame | None,
            "df_risk_all": DataFrame | None,
            "df_rbd_only_val": DataFrame | None,
            "df_rbd_only_all": DataFrame | None
        }
    """

    print("  >> Starting risk group pipeline")
    df_risk = df_risk.copy()

    # Ensure output directories exist
    final_dir.mkdir(parents=True, exist_ok=True)
    thresholds_dir.mkdir(parents=True, exist_ok=True)

    # Initialize outputs (prevents UnboundLocalError)
    df_risk_val = None
    df_risk_all = None
    df_rbd_only_val = None
    df_rbd_only_all = None

    # ---------------------------------------------
    # 1.  Build validation column
    # ---------------------------------------------
    try:
        val_cols = [c for c in df_risk.columns if c.startswith("val")]

        if val_cols:
            df_risk["val"] = df_risk[val_cols].max(axis=1)
            use_validation = True
            print("  [OK] Validation columns detected")
        else:
            df_risk["all"] = 1
            use_validation = False
            print("  [WARN] No validation columns found. Using full dataset.")

    except Exception:
        print("  [FAIL] Failed building validation column")
        traceback.print_exc()
        return {
            "df_risk_val": None,
            "df_risk_all": None,
            "df_rbd_only_val": None,
            "df_rbd_only_all": None,
        }

    # ---------------------------------------------
    # 2.  Risk groups using validation split
    # DEPRECATED: per-outcome risk groups replaced by agnostic stratification
    # ---------------------------------------------
    # if use_validation:
    #     try:
    #         print("  -> Computing risk groups (validation split)")
    #         df_risk_val = run_compute_risk_groups(
    #             df=df_risk,
    #             config=config,
    #             val_flags=val_flags,
    #             outcomes=outcomes,
    #             out_frame_dir=final_dir,
    #             out_thresholds_dir=thresholds_dir,
    #             file_name="ehr_diag_pd_rbd_val",
    #             rbd_col=rbd_col,
    #         )
    #     except Exception:
    #         print("  [FAIL] Failed in run_compute_risk_groups (validation)")
    #         traceback.print_exc()

    # ---------------------------------------------
    # 3.  Risk groups using all data
    # DEPRECATED: per-outcome risk groups replaced by agnostic stratification
    # ---------------------------------------------
    # try:
    #     print("  -> Computing risk groups (all data)")
    #
    #     df_all = df_risk.copy()
    #     val_cols = [c for c in df_all.columns if c.startswith("val")]
    #     df_all = df_all.drop(columns=val_cols, errors="ignore")
    #     df_all["all"] = 1
    #
    #     val_flags_all = {o: "all" for o in outcomes}
    #
    #     df_risk_all = run_compute_risk_groups(
    #         df=df_all,
    #         config=config,
    #         val_flags=val_flags_all,
    #         outcomes=outcomes,
    #         out_frame_dir=final_dir,
    #         out_thresholds_dir=thresholds_dir,
    #         file_name="ehr_diag_pd_rbd_all",
    #         rbd_col=rbd_col,
    #     )
    #
    # except Exception:
    #     print("  [FAIL] Failed in run_compute_risk_groups (all data)")
    #     traceback.print_exc()

    # ---------------------------------------------
    # 4.  RBD-only risk groups (validation)
    # ---------------------------------------------
    if use_validation:
        try:
            print("  -> Computing RBD-only groups (validation)")

            df_rbd_only_val = run_compute_risk_group_rbd_only(
                df=df_risk,
                config=config,
                outcomes=outcomes,
                out_thresholds_dir=thresholds_dir,
                val_col="val",
                out_frame_dir=final_dir,
                file_name="ehr_diag_pd_rbd_only_val",
                rbd_col=rbd_col,
            )

        except Exception:
            print("  [FAIL] Failed in run_compute_risk_group_rbd_only (validation)")
            traceback.print_exc()

    # ---------------------------------------------
    # 5.  RBD-only risk groups (all data)col_incident
    # ---------------------------------------------
    try:
        print("  -> Computing RBD-only groups (all data)")

        df_all_rbd = df_risk.copy()
        df_all_rbd["all"] = 1

        df_rbd_only_all = run_compute_risk_group_rbd_only(
            df=df_all_rbd,
            config=config,
            outcomes=outcomes,
            out_thresholds_dir=thresholds_dir,
            val_col="all",
            out_frame_dir=final_dir,
            file_name="ehr_diag_pd_rbd_only_all",
            rbd_col=rbd_col,
        )

    except Exception:
        print("  [FAIL] Failed in run_compute_risk_group_rbd_only (all data)")
        traceback.print_exc()

    # ---------------------------------------------
    # 6.  Final reporting
    # ---------------------------------------------
    print("  [OK] Risk group pipeline completed")

    if df_risk_val is not None:
        print(f"    * Validation rows: {len(df_risk_val):,}")

    if df_risk_all is not None:
        print(f"    * All-data rows: {len(df_risk_all):,}")

    if df_rbd_only_val is not None:
        print(f"    * RBD-only (val): {len(df_rbd_only_val):,}")

    if df_rbd_only_all is not None:
        print(f"    * RBD-only (all): {len(df_rbd_only_all):,}")

    print(f"    * Output directory: {final_dir}")

    return {
        "df_risk_val": df_risk_val,
        "df_risk_all": df_risk_all,
        "df_rbd_only_val": df_rbd_only_val,
        "df_rbd_only_all": df_rbd_only_all,
    }


def promote_abk_to_final(
    src_dataset_dir: Path,
    dst_dataset_dir: Path,
    src_thresholds_dir: Path,
    dst_thresholds_dir: Path,
) -> None:
    """
    Copy ABK pipeline outputs from the mode-specific subdirectory to the
    canonical final directories expected by downstream scripts.

    ABK is the production model.  Downstream scripts (run_cox_pipeline.py)
    read from the root directories defined in config; this step bridges the
    mode-namespaced write paths to those canonical locations.

    Source files are retained in the ABK subdirectory for traceability.
    Existing files in the destination are always overwritten.

    Parquet layout (flat):
        src_dataset_dir/*.parquet  ->  dst_dataset_dir/*.parquet

    Threshold layout (nested by file_name):
        src_thresholds_dir/<file_name>/*.json
            ->  dst_thresholds_dir/<file_name>/*.json
        copytree with dirs_exist_ok=True preserves the subdirectory
        structure that get_clean_risk_data expects.

    Parameters
    ----------
    src_dataset_dir : Path
        ABK parquet directory (e.g. data/pp/res_build_final_dataset/abk/).
    dst_dataset_dir : Path
        Canonical parquet directory (data/pp/res_build_final_dataset/).
    src_thresholds_dir : Path
        ABK threshold root (e.g. data/risk_thresholds/abk/).
    dst_thresholds_dir : Path
        Canonical threshold root (data/risk_thresholds/).
    """
    dst_dataset_dir.mkdir(parents=True, exist_ok=True)
    dst_thresholds_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Promoting ABK outputs to final directories ...")

    # ── Parquets (flat: one level deep) ──────────────────────────────────────
    parquet_files = list(src_dataset_dir.glob("*.parquet"))
    for src in parquet_files:
        dst = dst_dataset_dir / src.name
        shutil.copy2(src, dst)
        print(f"    [parquet] {src.name}  ->  {dst_dataset_dir}")

    # ── Thresholds (nested: abk/<file_name>/*.json) ───────────────────────────
    # copytree with dirs_exist_ok=True merges into dst without deleting existing
    # files not present in src (safe if other models wrote subdirs there).
    json_count = sum(1 for _ in src_thresholds_dir.rglob("*.json"))
    if src_thresholds_dir.exists() and json_count > 0:
        shutil.copytree(
            src_thresholds_dir,
            dst_thresholds_dir,
            dirs_exist_ok=True,
        )
        print(
            f"    [json]    {json_count} JSON file(s) (with subdirs)  "
            f"->  {dst_thresholds_dir}"
        )
    else:
        print(f"    [json]    No JSON files found in {src_thresholds_dir}")

    print(
        f"  [OK] Promoted {len(parquet_files)} parquet(s) and "
        f"{json_count} JSON(s) to final directories."
    )


def save_merge_consort_logs(
    df_ehr: pd.DataFrame,
    df_rbd_all: pd.DataFrame,
    path_gait: Path,
    dir_logs: Path,
    outcome_cols: List[str],
) -> None:
    """
    Save two CONSORT log files capturing the actigraphy–EHR merge pipeline.

    Outputs
    -------
    merge_diagnostics.csv
        Step-level subject counts from: sleep features → RBD scores → merge.
    no_rbd_diagnostic_breakdown.csv
        Per-outcome case breakdown for EHR subjects who had no RBD scores,
        identifying how many incident/prevalent cases are excluded for lack of
        actigraphy scoring.

    Parameters
    ----------
    df_ehr : pd.DataFrame
        Subject-level EHR dataset (output of build_ukb_dataset).
    df_rbd_all : pd.DataFrame
        All-visit RBD scores parquet (lowercased, with visit_number column).
    path_gait : Path
        Path to merged gait/sleep features parquet.
    dir_logs : Path
        Destination directory for log files.
    outcome_cols : list[str]
        Base outcome column names (e.g. 'outcome_1a_pd_only').
    """
    dir_logs.mkdir(parents=True, exist_ok=True)

    ehr_eids = set(df_ehr["eid"].unique())
    rbd_all_eids = set(df_rbd_all["eid"].unique())
    rbd_bl_eids = set(df_rbd_all.loc[df_rbd_all["visit_number"] == 0, "eid"].unique())

    # Sleep feature subjects (visit 0 only)
    try:
        df_gait = pd.read_parquet(path_gait)
        # Extract EID from night ID (format: eid_device_visit_night); keep visit 0 only
        id_parts = df_gait["id"].str.split("_")
        visit_mask = id_parts.str[2].astype(int) == 0
        sf_eids_v0 = set(id_parts.loc[visit_mask].str[0].astype(int))
        n_sleep_features = len(sf_eids_v0)
    except Exception:
        n_sleep_features = None
        sf_eids_v0 = set()

    # Merge set
    merged_eids = ehr_eids & rbd_bl_eids
    n_merged = len(merged_eids)

    # EHR without any RBD score
    no_rbd_eids = ehr_eids - rbd_all_eids
    n_no_rbd = len(no_rbd_eids)

    # RBD without EHR
    n_rbd_no_ehr = len(rbd_all_eids - ehr_eids)

    # Lost between sleep features and RBD scoring
    n_sf_no_rbd = len(sf_eids_v0 - rbd_bl_eids) if sf_eids_v0 else None

    diag_rows = [
        {"step": "sleep_features_v0",   "description": "Subjects with sleep features extracted (visit 0)", "n": n_sleep_features},
        {"step": "rbd_scores_v0",        "description": "Subjects with RBD probability scores (visit 0)",   "n": len(rbd_bl_eids)},
        {"step": "sf_to_rbd_lost",       "description": "Lost: sleep features without RBD score",           "n": n_sf_no_rbd},
        {"step": "ehr_total",            "description": "EHR subjects",                                     "n": len(ehr_eids)},
        {"step": "merged",               "description": "Retained after EHR x RBD inner join",              "n": n_merged},
        {"step": "ehr_no_rbd_lost",      "description": "Lost from EHR: no RBD score",                     "n": n_no_rbd},
        {"step": "rbd_no_ehr_lost",      "description": "Lost from RBD: no EHR match",                     "n": n_rbd_no_ehr},
    ]
    pd.DataFrame(diag_rows).to_csv(dir_logs / "merge_diagnostics.csv", index=False)
    print(f"  [LOG] merge_diagnostics.csv saved to {dir_logs}")

    # ── Diagnostic breakdown for EHR subjects without RBD scores ──────────────
    df_no_rbd = df_ehr[df_ehr["eid"].isin(no_rbd_eids)].copy()
    from config.config import outcomes_short_names
    outcome_labels = outcomes_short_names
    acc_bad_n = int(df_no_rbd["acc_bad_quality"].sum()) if "acc_bad_quality" in df_no_rbd.columns else None
    breakdown_rows = []
    for o, label in outcome_labels.items():
        if o not in df_no_rbd.columns:
            continue
        prev_col = f"{o}__prevalent"
        inc_col  = f"{o}__incident"
        breakdown_rows.append({
            "outcome":     o,
            "label":       label,
            "n_no_rbd":    n_no_rbd,
            "n_diagnosed": int(df_no_rbd[o].sum()),
            "n_prevalent": int(df_no_rbd[prev_col].sum()) if prev_col in df_no_rbd.columns else None,
            "n_incident":  int(df_no_rbd[inc_col].sum())  if inc_col  in df_no_rbd.columns else None,
            "n_acc_bad_quality": acc_bad_n,
            "pct_acc_bad_quality": round(100 * acc_bad_n / n_no_rbd, 2) if acc_bad_n and n_no_rbd else None,
        })
    pd.DataFrame(breakdown_rows).to_csv(dir_logs / "no_rbd_diagnostic_breakdown.csv", index=False)
    print(f"  [LOG] no_rbd_diagnostic_breakdown.csv saved to {dir_logs}")


def check_time_window(df_subj: pd.DataFrame) -> None:

    # 1. The censor date encoded in the parquet (all rows should be identical)
    print(df_subj['censor_date'].unique())  # → should show ['2025-11-30'] after re-run
    # if it shows '2022-10-31' → stale file

    # 2. Maximum follow-up in years for controls (the longest possible observation)
    surv_col = col_surv_time('outcome_1a_pd_only')  # in days
    print(df_subj[surv_col].max() / 365.25)  # → ~12.4 y expected with 2025 censor

    # 3. Distribution of follow-up (controls only, to exclude short incident TTE)
    ctrl_mask = df_subj['control'].fillna(False)
    print((df_subj.loc[ctrl_mask, surv_col] / 365.25).describe())

    # 4. Reconstruct expected max directly from the dates
    print((pd.Timestamp('2025-11-30') - df_subj['wear_time_start']).dt.days.max() / 365.25)

    # 5. Cross-check: confirm follow_up_years now matches
    print(df_subj['follow_up_years'].describe())

def main(overwrite: bool = True) -> None:
    """Merge EHR + RBD scores + gait, compute risk groups, promote to production.

    Parameters
    ----------
    overwrite : bool
        Regenerate parquets even if the output directory exists.
        Must be True whenever censor_date or EHR data has changed.
    """
    # get - results from build_ukbb_dataset
    path_ukbb_ehr = config.get('paths')['data_sheet']['dir_parquet']
    # get - results from run_abk_collection
    path_rbd_scores = config.get("paths")["actig_extracted"]["rbd_scores"]
    path_gait = config.get("paths")["actig_extracted"]["merged_gait"]
    # output - path to save results
    dir_our_dataset = config.get("pp")["final_dir"]
    dir_out_thresh = config.get("pp")["thresholds"]["root"]

    # Load EHR
    df_ehr = pd.read_parquet(path_ukbb_ehr)

    # Load RBD scores
    df_rbd = pd.read_parquet(path_rbd_scores)

    # Guard: empty RBD data
    if df_rbd.empty:
        print("\n[WARN] No RBD scores available. Skipping merge stage (cannot stratify without RBD).")
        return

    df_rbd.columns = [col.lower() for col in df_rbd.columns]
    df_rbd = df_rbd.rename(columns={"irbd_sleep_score": "abk_rbd_score"})
    df_rbd["visit_number"] = df_rbd["id"].str.split("_").str[2].astype(int)

    # Baseline RBD only
    df_rbd_bl = df_rbd[df_rbd["visit_number"] == 0]

    # Inner join: retains only subjects with both EHR and actigraphy data.
    # EIDs present in RBD but not EHR are dropped (different data releases /
    # withdrawn consent); EIDs in EHR without actigraphy are also dropped.
    n_rbd_unmatched = df_rbd_bl["eid"].nunique() - len(set(df_rbd_bl["eid"]) & set(df_ehr["eid"]))
    n_ehr_unmatched = df_ehr["eid"].nunique() - len(set(df_rbd_bl["eid"]) & set(df_ehr["eid"]))
    print(f"  Merge diagnostics — RBD subjects without EHR (dropped): {n_rbd_unmatched:,}")
    print(f"  Merge diagnostics — EHR subjects without actigraphy (dropped): {n_ehr_unmatched:,}")
    df_ehr_rbd = pd.merge(df_ehr, df_rbd_bl, validate='one_to_many', on="eid")
    print(f"  After merge: {df_ehr_rbd['eid'].nunique():,} subjects, {len(df_ehr_rbd):,} night rows")

    # Average RBD scores across the nights and merge
    df_subj = (
        df_ehr_rbd.groupby("eid", as_index=False)
        .agg(abk_rbd_score_mean=("abk_rbd_score", "mean"))
    )
    df_ehr_rbd_avg = pd.merge(
        left=df_ehr_rbd, right=df_subj, validate='many_to_one', on="eid",
    )

    # Merge gait arm_swing features — night-level join on 'id' (eid_device_visit_night).
    # df_gait has one row per recording window; unmatched nights get NaN arm_swing values.
    df_gait = pd.read_parquet(path_gait)
    arm_swing_cols = [c for c in df_gait.columns if c.startswith("arm_swing")]
    df_gait_arm = df_gait[['id'] + arm_swing_cols].copy()
    n_before = len(df_ehr_rbd_avg)
    df_ehr_rbd_avg = pd.merge(df_ehr_rbd_avg, df_gait_arm, on="id", how="left")
    gait_matched = df_ehr_rbd_avg[arm_swing_cols[0]].notna().sum() if arm_swing_cols else 0
    print(f"  Gait arm_swing columns merged: {arm_swing_cols}")
    print(f"  Gait coverage: {gait_matched:,} / {n_before:,} night rows ({gait_matched/n_before*100:.1f}%)")

    # Compute risk groups (ABK production model)
    print("  Computing risk groups ...")
    df_risk = df_ehr_rbd_avg.copy()
    rbd_col = "abk_rbd_score"
    dir_our_dataset_mode = insert_after_data(dir_our_dataset, "abk")
    dir_out_thresh_mode = insert_after_data(dir_out_thresh, "abk")

    if dir_our_dataset_mode.exists() and not overwrite:
        print(f"  Directory {dir_our_dataset_mode} already exists. Skipping (overwrite=False).")
    else:
        df_eval = df_risk[df_risk[rbd_col].notna()].reset_index(drop=True).copy()
        # Remove validation columns (not used in this dataset)
        col_vals = [col for col in df_eval.columns if col.startswith("val")]
        df_eval = df_eval.drop(columns=col_vals)
        df_eval['train_sleep'] = False  # no training on the ukbb

        risk_group_pipeline(
            df_risk=df_eval,
            config=config,
            outcomes=outcomes,
            final_dir=dir_our_dataset_mode,
            thresholds_dir=dir_out_thresh_mode,
            rbd_col=rbd_col,
        )

    # Save CONSORT merge logs
    save_merge_consort_logs(
        df_ehr=df_ehr,
        df_rbd_all=df_rbd,
        path_gait=Path(path_gait),
        dir_logs=Path("results/logs/final_consort"),
        outcome_cols=list(outcomes),
    )

    # Promote ABK outputs to canonical final directories
    promote_abk_to_final(
        src_dataset_dir=dir_our_dataset_mode,
        dst_dataset_dir=dir_our_dataset,
        src_thresholds_dir=dir_out_thresh_mode,
        dst_thresholds_dir=dir_out_thresh,
    )


if __name__ == "__main__":
    main()

