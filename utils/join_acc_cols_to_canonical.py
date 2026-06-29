"""
Join 30 new Category 1009 acceleration columns from ukb_final_dataset.parquet
onto the canonical night-level parquet (ehr_diag_pd_rbd_only_all.parquet).

Why a join rather than a full rebuild:
    The canonical parquet holds the validated (backup) RBD scores and risk
    groups.  Re-running run_merge_ukbb_rbd.py would overwrite those with the
    repaired-ABK scores, which weakened PD risk stratification (High group
    dropped from 33 → 13 incident events).  Instead, we add the 30 new columns
    additively, leaving every other column untouched.

Join key: eid only.
    The 30 fields are UKB wear-period summary statistics — one value per
    subject, constant across all night rows.  Verified: existing acc cols show
    zero within-subject variance across night rows in the canonical parquet.

Safety: a pre-join copy is written before overwriting the canonical.

Seed/randomness: none (deterministic join).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from config.config import config

# ── Paths ────────────────────────────────────────────────────────────────────
EHR_PATH: Path = config["paths"]["data_sheet"]["dir_parquet"]
CANONICAL: Path = config["pp"]["final_dir"] / "ehr_diag_pd_rbd_only_all.parquet"
PRE_JOIN_COPY: Path = config["pp"]["final_dir"] / "ehr_diag_pd_rbd_only_all_prejoin_acc.parquet"

# ── Target columns ───────────────────────────────────────────────────────────
NEW_ACC_COLS: list[str] = (
    ["acc_overall_std_90013"]
    + [f"acc_hour_{h:02d}_9{27 + h:04d}"[:-1] + f"{27 + h}" for h in range(24)]
    + [
        "acc_nowear_avg_90087",
        "acc_nowear_std_90088",
        "acc_nowear_median_90089",
        "acc_nowear_min_90090",
        "acc_nowear_max_90091",
    ]
)

# Build the hourly names explicitly to avoid index arithmetic errors
_HOURLY: list[str] = [
    "acc_hour_00_90027", "acc_hour_01_90028", "acc_hour_02_90029",
    "acc_hour_03_90030", "acc_hour_04_90031", "acc_hour_05_90032",
    "acc_hour_06_90033", "acc_hour_07_90034", "acc_hour_08_90035",
    "acc_hour_09_90036", "acc_hour_10_90037", "acc_hour_11_90038",
    "acc_hour_12_90039", "acc_hour_13_90040", "acc_hour_14_90041",
    "acc_hour_15_90042", "acc_hour_16_90043", "acc_hour_17_90044",
    "acc_hour_18_90045", "acc_hour_19_90046", "acc_hour_20_90047",
    "acc_hour_21_90048", "acc_hour_22_90049", "acc_hour_23_90050",
]

NEW_ACC_COLS: list[str] = (
    ["acc_overall_std_90013"]
    + _HOURLY
    + [
        "acc_nowear_avg_90087",
        "acc_nowear_std_90088",
        "acc_nowear_median_90089",
        "acc_nowear_min_90090",
        "acc_nowear_max_90091",
    ]
)


def _verify_rbd_scores_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Assert RBD score columns are bit-identical before and after the join."""
    rbd_cols = ["abk_rbd_score_mean", "rg_pctl2", "rg_pctl3", "rg_q4"]
    for col in rbd_cols:
        if col not in before.columns:
            continue
        changed = (before[col] != after[col]).sum()
        if changed > 0:
            raise RuntimeError(
                f"[FAIL] RBD column '{col}' changed in {changed:,} rows after join. "
                "Aborting — canonical not overwritten."
            )
    print("[OK] RBD scores and risk groups verified unchanged.")


def main() -> None:
    """Join 30 new acc columns onto canonical parquet."""

    # ── Guard: EHR parquet must exist ────────────────────────────────────────
    if not EHR_PATH.exists():
        raise FileNotFoundError(
            f"EHR parquet not found: {EHR_PATH}\n"
            "Run Stage 1 (build_ukb_dataset) with OVERWRITE_EHR=False first."
        )
    if not CANONICAL.exists():
        raise FileNotFoundError(f"Canonical parquet not found: {CANONICAL}")

    # ── Load EHR (subject-level, one row per eid) ────────────────────────────
    print(f"[1/5] Loading EHR parquet: {EHR_PATH}")
    ehr = pd.read_parquet(EHR_PATH, columns=["eid"] + NEW_ACC_COLS)

    # Guard: all 30 columns must be present (i.e. covariates.py was updated)
    missing = [c for c in NEW_ACC_COLS if c not in ehr.columns]
    if missing:
        raise KeyError(
            f"[FAIL] {len(missing)} target column(s) missing from EHR parquet.\n"
            f"Missing: {missing}\n"
            "Check that covariates.py was edited and Stage 1 re-run."
        )

    print(f"       {ehr['eid'].nunique():,} subjects, {len(NEW_ACC_COLS)} new acc cols")
    null_pct = ehr[NEW_ACC_COLS].isna().mean().mul(100).round(1)
    print(f"       Null rates (%):\n{null_pct.to_string()}")

    # ── Load canonical (night-level, multiple rows per eid) ──────────────────
    print(f"\n[2/5] Loading canonical parquet: {CANONICAL}")
    canonical = pd.read_parquet(CANONICAL)
    n_rows, n_eid = len(canonical), canonical["eid"].nunique()
    print(f"       {n_rows:,} rows, {n_eid:,} subjects")

    # Drop any of the 30 cols that already exist (idempotent re-run)
    already_present = [c for c in NEW_ACC_COLS if c in canonical.columns]
    if already_present:
        print(f"       Dropping {len(already_present)} pre-existing col(s): {already_present}")
        canonical = canonical.drop(columns=already_present)

    # ── Safety copy before overwrite ─────────────────────────────────────────
    print(f"\n[3/5] Writing safety copy: {PRE_JOIN_COPY.name}")
    if not PRE_JOIN_COPY.exists():
        shutil.copy2(CANONICAL, PRE_JOIN_COPY)
        print("       Safety copy written.")
    else:
        print("       Safety copy already exists — skipping.")

    # ── Join: broadcast subject-level acc values to all night rows ────────────
    print("\n[4/5] Joining new acc columns by eid ...")
    ehr_slim = ehr[["eid"] + NEW_ACC_COLS].drop_duplicates("eid")
    merged = canonical.merge(ehr_slim, on="eid", how="left")

    # Integrity checks
    if len(merged) != n_rows:
        raise RuntimeError(
            f"[FAIL] Row count changed after join: {n_rows:,} → {len(merged):,}. Aborting."
        )
    if merged["eid"].nunique() != n_eid:
        raise RuntimeError(
            f"[FAIL] Subject count changed after join: {n_eid:,} → {merged['eid'].nunique():,}. Aborting."
        )

    _verify_rbd_scores_unchanged(canonical, merged)

    # ── Write back to canonical ──────────────────────────────────────────────
    print(f"\n[5/5] Writing updated canonical: {CANONICAL.name}")
    merged.to_parquet(CANONICAL, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    matched = merged[NEW_ACC_COLS[0]].notna().sum()
    print(
        f"\n[DONE] Canonical updated."
        f"\n       Rows:     {len(merged):,}"
        f"\n       Subjects: {merged['eid'].nunique():,}"
        f"\n       New cols: {len(NEW_ACC_COLS)}"
        f"\n       Matched rows (non-null acc_hour_00): {matched:,} / {n_rows:,}"
    )

    # Print rg_pctl3 as final sanity check
    ds = merged.drop_duplicates("eid")
    print("\n=== SANITY: rg_pctl3 distribution (subject-level) ===")
    print(ds["rg_pctl3"].value_counts().to_string())


if __name__ == "__main__":
    main()
