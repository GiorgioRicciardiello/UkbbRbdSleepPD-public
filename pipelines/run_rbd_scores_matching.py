"""
RBD Scores Matching — Entry Point
==================================
Generates RBD model predictions from sleep features, merges them with
ABK RBD scores on **(ID, Date)**, and runs agreement / disagreement
evaluation between the two scoring systems.

Why ``ID`` and not ``eid``?
    ``ID`` encodes ``{eid}_{visit}`` (e.g. ``1234567_0_12345``), preserving
    follow-up information.  ``eid`` alone conflates baseline and follow-up
    recordings, producing incorrect joins.

Inputs
------
- Sleep features   → ``config['paths']['actig_extracted']['merged_sleep']``
- ABK RBD scores   → ``config['paths']['actig_extracted']['rbd_scores']``
- EHR dataset      → ``config['paths']['data_sheet']['dir_parquet']``
- Pre-trained model → ``config['rar_rbd_models']['rar_sleep']``

Output
------
- ``config['pp']['rbd_scores']``  (parquet with both model predictions and
  ABK scores per night)

Usage
-----
    python notebook/run_rbd_scores_matching.py
"""

import warnings

import numpy as np
import pandas as pd

from config.config import features, config
from library.compute_rbd_scores import (
    load_model,
    harmonize_id,
    harmonize_date,
    report_predictions,
    plot_rbd_prob_vs_abk_score,
    plot_abk_score_distribution,
    plot_rbd_quadrant_ambulatory_disorder,
    plot_logit_score_fit,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ================================================================
    # 1  PATHS
    # ================================================================
    path_model_sleep    = config.get("rar_rbd_models")["rar_sleep"]
    path_ehr            = config.get("paths")["data_sheet"]["dir_parquet"]
    path_sleep_features = config.get("paths")["actig_extracted"]["merged_sleep"]
    path_abk_rbd_scores = config.get("paths")["actig_extracted"]["rbd_scores"]
    path_out            = config.get("pp")["rbd_scores"]

    # ================================================================
    # 2  LOAD DATA
    # ================================================================
    model_sleep = load_model(path_model_sleep)

    df_ehr = pd.read_parquet(path_ehr)
    ehr_eids = df_ehr.loc[df_ehr["wear_time_start"].notna(), "eid"]

    # Sleep features (one row per ID × Date night)
    df_sleep = pd.read_parquet(path_sleep_features)
    df_sleep = harmonize_id(df_sleep, id_col="ID")
    df_sleep = harmonize_date(df_sleep, date_col="Date")
    df_sleep["eid"] = df_sleep["eid"].astype(int)

    # Filter to subjects present in EHR with actigraphy wear-time
    df_sleep = df_sleep.loc[df_sleep["eid"].isin(ehr_eids)].reset_index(drop=True)

    # ABK RBD scores (one row per ID × Date night)
    df_abk = pd.read_parquet(path_abk_rbd_scores)
    df_abk = harmonize_id(df_abk, id_col="ID")
    df_abk = harmonize_date(df_abk, date_col="Date")

    print(f"\n  Sleep features : {df_sleep.shape}")
    print(f"  ABK RBD scores : {df_abk.shape}")

    # ================================================================
    # 3  INTEGRITY CHECKS — before prediction
    # ================================================================
    n_dup_sleep = df_sleep.duplicated(["ID", "Date"]).sum()
    n_dup_abk   = df_abk.duplicated(["ID", "Date"]).sum()

    print(f"\n  Duplicates (ID, Date) in sleep features : {n_dup_sleep}")
    print(f"  Duplicates (ID, Date) in ABK scores    : {n_dup_abk}")

    if n_dup_sleep > 0:
        print(f"  ⚠ Dropping {n_dup_sleep} duplicate (ID, Date) rows from sleep features")
        df_sleep = df_sleep.drop_duplicates(["ID", "Date"], keep="first").reset_index(drop=True)

    if n_dup_abk > 0:
        print(f"  ⚠ Dropping {n_dup_abk} duplicate (ID, Date) rows from ABK scores")
        df_abk = df_abk.drop_duplicates(["ID", "Date"], keep="first").reset_index(drop=True)

    # ================================================================
    # 4  GENERATE PREDICTIONS  (or load cached)
    # ================================================================
    if not path_out.exists():
        print("\n  Computing RBD predictions ...")

        # Prepare feature matrix
        df_features = df_sleep[features].replace([np.inf, -np.inf], np.nan).fillna(0)

        predictions = model_sleep.predict_proba(df_features)

        df_pred = pd.DataFrame({
            "ID":                df_sleep["ID"].values,
            "Date":              df_sleep["Date"].values,
            "eid":               df_sleep["eid"].values,
            "visit_number":      df_sleep["visit_number"].values,
            "PredictionClass_0": predictions[:, 0],
            "PredictionClass_1": predictions[:, 1],
            "prediction":        model_sleep.predict(df_features),
        })

        # ============================================================
        # 5  MERGE ABK RBD SCORES  →  on (ID, Date)
        # ============================================================
        abk_cols = ["ID", "Date", "iRBD_Sleep_Score"]
        abk_cols = [c for c in abk_cols if c in df_abk.columns]

        df_merged = pd.merge(
            left=df_pred,
            right=df_abk[abk_cols],
            on=["ID", "Date"],
            how="left",
        )

        # --- merge diagnostics ---
        n_total     = len(df_merged)
        n_matched   = df_merged["iRBD_Sleep_Score"].notna().sum() if "iRBD_Sleep_Score" in df_merged.columns else 0
        n_unmatched = n_total - n_matched
        pct_matched = (n_matched / n_total * 100) if n_total else 0

        print(f"\n  === MERGE DIAGNOSTICS (on ID, Date) ===")
        print(f"  Total prediction rows    : {n_total}")
        print(f"  Matched ABK score        : {n_matched}  ({pct_matched:.1f}%)")
        print(f"  Unmatched (ABK missing)  : {n_unmatched}")

        # ============================================================
        # 6  RENAME & CLEAN
        # ============================================================
        df_merged = df_merged.rename(columns={
            "prediction":        "rbd_bin",
            "PredictionClass_0": "rbd_prob_class0",
            "PredictionClass_1": "rbd_prob_class1",
            "iRBD_Sleep_Score":  "abk_rbd_score",
        })

        # lowercase all column names for downstream consistency
        df_merged.columns = [c.lower() for c in df_merged.columns]

        # ============================================================
        # 7  SAVE
        # ============================================================
        path_out.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_parquet(path_out, index=False)
        print(f"\n  ✅ Saved → {path_out}")

        # ============================================================
        # 8  REPORT
        # ============================================================
        report_predictions(
            df=df_merged,
            group_col="id",
            pred_col="rbd_bin",
            output_path=path_out.parent,
        )
    else:
        print(f"\n  Predictions already exist → loading from {path_out}")
        df_merged = pd.read_parquet(path_out)
        report_predictions(df=df_merged, group_col="id", pred_col="rbd_bin")

    # ================================================================
    # 9  FINAL SANITY CHECK
    # ================================================================
    print("\n" + "=" * 60)
    print("  FINAL OUTPUT SUMMARY")
    print("=" * 60)
    print(f"  Shape             : {df_merged.shape}")
    print(f"  Unique IDs        : {df_merged['id'].nunique()}")
    print(f"  Unique (ID, Date) : {df_merged[['id', 'date']].drop_duplicates().shape[0]}")
    if "abk_rbd_score" in df_merged.columns:
        print(f"  ABK score coverage : {df_merged['abk_rbd_score'].notna().mean():.1%}")
    print("=" * 60)

    # ================================================================
    # 10  EVALUATION — Agreement between ABK and model RBD scores
    # ================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION: ABK score vs Model prediction agreement")
    print("=" * 60)

    # Keep only rows where ABK score is available
    df_eval = df_merged.dropna(subset=["abk_rbd_score"])
    print(f"  Rows with ABK score : {len(df_eval)} / {len(df_merged)}")

    # Subject-level aggregation
    df_subj = (
        df_eval.groupby("id", as_index=False)
        .agg(
            rbd_prob_mean=("rbd_prob_class1", "mean"),
            abk_rbd_score_mean=("abk_rbd_score", "mean"),
            rbd_bin=("rbd_bin", "max"),   # ever-positive
        )
    )

    # Logit transform of mean probability
    eps = 1e-6
    p = df_subj["rbd_prob_mean"].clip(eps, 1 - eps)
    df_subj["rbd_logit"] = np.log(p / (1 - p))

    print(f"  Unique subjects with both scores : {len(df_subj)}")

    # ---- Plot 1: Probability vs ABK score scatter ----
    plot_rbd_prob_vs_abk_score(df_subj)

    # ---- Plot 2: ABK score distribution stratified by binary prediction ----
    plot_abk_score_distribution(df_subj)

    # ---- Plot 3: Quadrant agreement plot ----
    q_counts = plot_rbd_quadrant_ambulatory_disorder(
        df=df_subj,
        x_col="abk_rbd_score_mean",
        prob_col="rbd_prob_mean",
        bin_col="rbd_bin",
        use_logit=True,
        x_thr=0.0,
        prob_thr=0.4,
    )
    print("\n  Quadrant counts:")
    print(q_counts.to_string())

    # ---- Spearman correlation ----
    corr = df_subj[["rbd_logit", "abk_rbd_score_mean"]].corr(method="spearman")
    print(f"\n  Spearman correlation (logit vs ABK score):")
    print(corr.to_string())

    # ---- Plot 4: Logit vs ABK score polynomial fit ----
    fit_results = plot_logit_score_fit(
        df=df_subj,
        x_col="abk_rbd_score_mean",
        y_col="rbd_logit",
        subject_col="id",
    )
    print(f"\n  Fit results: MAE={fit_results['mae']:.3f}, "
          f"RMSE={fit_results['rmse']:.3f}, R²={fit_results['r2']:.3f}")
    print(f"  Pearson r={fit_results['pearson_r']:.3f} (p={fit_results['pearson_p']:.2e})")
    print(f"  Spearman r={fit_results['spearman_r']:.3f} (p={fit_results['spearman_p']:.2e})")
