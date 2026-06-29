"""
Graft OLD (pre-ABKCollector-repair) RBD scores + risk groups onto the freshly
rebuilt temporal-window dataset.

Context
-------
The 2026-06-03 ABKCollector repair (commit d37f288, "load all 6 batches")
regenerated RBD_Sleep_Score_merged.parquet.  Those repaired scores spread PD
cases out of the high-risk group (top 1% captured only 13 incident PD vs 33
under the previous scores).  The previous (backup) scores give materially
stronger risk stratification and are the ones used for the analysis.

This script keeps the new temporal-window covariates (_bl/_fu/_delta/_post) but
restores the old per-subject ``abk_rbd_score_mean`` and the old risk-group
assignments (``rg_pctl2``/``rg_pctl3``/``rg_q4``) from the backup canonical
parquet, joined by ``eid``.  Old group labels are remapped to the current
``Low``/``Mid``/``High`` convention.

Reversibility
-------------
A pre-graft copy of the current canonical is written alongside before
overwriting.  Re-running ``pipelines/run_merge_ukbb_rbd.py`` would recompute the
repaired scores again — to make the old scores permanent, the old RBD score
source would need to be restored, or this graft re-applied.

Seed/randomness: none (deterministic join).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from config.config import config
from library.column_registry import col_incident

# ── Paths ────────────────────────────────────────────────────────────────────
FINAL_DIR: Path = config.get("pp")["final_dir"]
CANONICAL: Path = FINAL_DIR / "ehr_diag_pd_rbd_only_all.parquet"
BACKUP: Path = Path(
    "data/pp/backup/res_build_final_dataset/ehr_diag_pd_rbd_only_all.parquet"
)
PRE_GRAFT_COPY: Path = FINAL_DIR / "ehr_diag_pd_rbd_only_all_newabk_pregraft.parquet"

# Old group-label -> current convention.  rg_q4 labels are already identical.
RG3_RELABEL: dict[str, str] = {
    "Low (0,90%)": "Low",
    "Intermediate (90,99%)": "Mid",
    "High (99,100%)": "High",
}
RG2_RELABEL: dict[str, str] = {
    "Low (0,90%)": "Low",
    "High (90,100%)": "High",
}

GRAFT_COLS: list[str] = ["abk_rbd_score_mean", "rg_pctl2", "rg_pctl3", "rg_q4"]


def main() -> None:
    """Graft old RBD scores/groups onto the current canonical parquet."""
    if not CANONICAL.exists():
        raise FileNotFoundError(f"Current canonical not found: {CANONICAL}")
    if not BACKUP.exists():
        raise FileNotFoundError(f"Backup canonical not found: {BACKUP}")

    # ── Safety copy of the current (new-abk) canonical ───────────────────────
    if not PRE_GRAFT_COPY.exists():
        shutil.copy2(CANONICAL, PRE_GRAFT_COPY)
        print(f"[safety] Pre-graft copy written: {PRE_GRAFT_COPY.name}")
    else:
        print(f"[safety] Pre-graft copy already exists: {PRE_GRAFT_COPY.name}")

    # ── Load old subject-level scores + groups from backup ───────────────────
    old = (
        pd.read_parquet(BACKUP, columns=["eid", *GRAFT_COLS])
        .drop_duplicates("eid")
        .reset_index(drop=True)
    )
    old["rg_pctl3"] = old["rg_pctl3"].map(RG3_RELABEL).astype("object")
    old["rg_pctl2"] = old["rg_pctl2"].map(RG2_RELABEL).astype("object")
    # rg_q4 labels already match current convention; leave as-is.
    old = old.rename(columns={c: f"__old_{c}" for c in GRAFT_COLS})
    print(f"[load] backup subjects: {len(old):,}")

    # ── Load current canonical (night-level, all columns) ────────────────────
    cur = pd.read_parquet(CANONICAL)
    n_rows_before, n_eid_before = len(cur), cur["eid"].nunique()
    print(f"[load] current: {n_rows_before:,} rows, {n_eid_before:,} subjects")

    # ── Replace score/group columns with old values (join by eid) ────────────
    cur = cur.drop(columns=[c for c in GRAFT_COLS if c in cur.columns])
    cur = cur.merge(old, on="eid", how="left")
    for c in GRAFT_COLS:
        cur[c] = cur[f"__old_{c}"]
    cur = cur.drop(columns=[f"__old_{c}" for c in GRAFT_COLS])

    # ── Drop subjects with no old score (present in new cohort only) ─────────
    no_old = cur["abk_rbd_score_mean"].isna()
    n_dropped_rows = int(no_old.sum())
    n_dropped_eid = int(cur.loc[no_old, "eid"].nunique())
    cur = cur[~no_old].reset_index(drop=True)
    print(
        f"[drop] subjects without an old RBD score: {n_dropped_eid} "
        f"({n_dropped_rows:,} night rows)"
    )

    # ── Write back to canonical ──────────────────────────────────────────────
    cur.to_parquet(CANONICAL, index=False)
    print(f"[write] {CANONICAL.name}: {len(cur):,} rows, {cur['eid'].nunique():,} subjects")

    # ── Verify: PD incidence by restored risk group (subject-level) ──────────
    inc = col_incident("outcome_1a_pd_only")
    ds = cur.drop_duplicates("eid")
    g = (
        ds.groupby("rg_pctl3", dropna=False)
        .agg(n=("eid", "size"), pd_incident=(inc, "sum"))
    )
    g["pct"] = (g["pd_incident"] / g["n"] * 100).round(3)
    g = g.reindex([x for x in ["Low", "Mid", "High"] if x in g.index])
    print("\n=== VERIFY: PD incident by restored rg_pctl3 ===")
    print(g.to_string())
    print(f"total incident PD: {int(ds[inc].sum())}")


if __name__ == "__main__":
    main()
