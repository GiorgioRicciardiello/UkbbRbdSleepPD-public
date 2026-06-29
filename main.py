"""
UkbbRbdSleepPD — Main Pipeline Orchestrator
=============================================
Runs the full analysis pipeline in dependency order.
Toggle each stage with a boolean flag.

Pipeline stages
---------------
Stage 1: build_ukb_dataset      — Extract & process UKBB EHR from raw CSVs
Stage 2: run_abk_collection     — Collect & merge actigraphy batches (Gait, Sleep, RBD)
Stage 3: run_merge_ukbb_rbd     — Merge EHR + RBD + gait → risk groups → production parquet
Stage 4: run_table_one          — Table 1 baseline characteristics
Stage 5: run_cox_pipeline       — Cox prodromal analysis (Models 0–4)
Stage 6: sleep_phenotypes       — Sleep phenotype vs RBD score evaluation
Stage 7: sleep_temporal         — Temporal validation of RBD score reliability

Dependency graph
----------------
    [1] build_ukb_dataset ──┐
                            ├──▶ [3] run_merge_ukbb_rbd ──┬──▶ [4] table_one
    [2] run_abk_collection ─┘                              ├──▶ [5] cox_pipeline
                                                           ├──▶ [6] sleep_phenotypes
                                                           └──▶ [7] sleep_temporal

Stages 1 & 2 are independent (could run in parallel).
Stages 4–7 are independent of each other (depend only on 3).

Usage
----.
    python main.py
"""
import time

# ── Stage toggles ─────────────────────────────────────────────────────────────
RUN_BUILD_EHR: bool = True          # Stage 1: re-run processing to add 30 new acc cols (OVERWRITE_EHR=False → uses raw cache)
RUN_ABK_COLLECTION: bool = False    # Stage 2: collect actigraphy batches — data restored, skip
RUN_MERGE: bool = False              # Stage 3: do NOT run — would overwrite validated RBD scores
RUN_TABLE_ONE: bool = False          # Stage 4: Table 1 baseline characteristics
RUN_COX_PIPELINE: bool = False      # Stage 5: Cox prodromal analysis
RUN_SLEEP_PHENOTYPES: bool = False  # Stage 6: sleep phenotype stratification — DISABLED
RUN_SLEEP_TEMPORAL: bool = False    # Stage 7: temporal validation — DISABLED

# ── Stage overwrite flags ─────────────────────────────────────────────────────
OVERWRITE_EHR: bool = False          # Stage 1: False → reads ukb_raw_dataset.parquet cache (no CSV re-read)
OVERWRITE_MERGE: bool = False         # Stage 3: regenerate even if output exists


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(stage_num: int, total: int, name: str) -> None:
    """Print a stage header."""
    print(f"\n{'=' * 72}")
    print(f"  STAGE {stage_num}/{total}: {name}")
    print(f"{'=' * 72}\n")


def _footer(name: str, elapsed: float) -> None:
    """Print a stage footer with elapsed time."""
    print(f"\n  [{name}] completed in {elapsed:.1f}s")
    print("-" * 72)


def _skip(stage_num: int, name: str) -> None:
    """Print a skip message."""
    print(f"  [{stage_num}] {name} — SKIPPED")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Execute the full pipeline respecting stage toggles."""
    total = 7
    t_pipeline = time.time()

    print("\n" + "=" * 72)
    print("  UKBB RBD SLEEP PD — ANALYSIS PIPELINE")
    print("=" * 72)
    print(f"\n  Stages enabled: ", end="")
    flags = [
        (1, "EHR Build", RUN_BUILD_EHR),
        (2, "ABK Collection", RUN_ABK_COLLECTION),
        (3, "Merge", RUN_MERGE),
        (4, "Table One", RUN_TABLE_ONE),
        (5, "Cox Pipeline", RUN_COX_PIPELINE),
        (6, "Sleep Phenotypes", RUN_SLEEP_PHENOTYPES),
        (7, "Sleep Temporal", RUN_SLEEP_TEMPORAL),
    ]
    enabled = [f"[{n}] {name}" for n, name, on in flags if on]
    skipped = [f"[{n}] {name}" for n, name, on in flags if not on]
    print(", ".join(enabled) if enabled else "NONE")
    if skipped:
        print(f"  Stages skipped:  {', '.join(skipped)}")
    print()

    # ── Stage 1: Build UKBB EHR dataset ──────────────────────────────────
    if RUN_BUILD_EHR:
        _header(1, total, "BUILD UKBB EHR DATASET")
        t0 = time.time()
        from pipelines.build_ukb_dataset import main as build_ehr_main
        build_ehr_main(overwrite_ehr=OVERWRITE_EHR)
        _footer("Build EHR", time.time() - t0)
    else:
        _skip(1, "Build EHR")

    # ── Stage 2: Collect actigraphy batches ──────────────────────────────
    if RUN_ABK_COLLECTION:
        _header(2, total, "ABK ACTIGRAPHY COLLECTION")
        t0 = time.time()
        from pipelines.run_abk_collection import main as abk_main
        abk_main()
        _footer("ABK Collection", time.time() - t0)
    else:
        _skip(2, "ABK Collection")

    # ── Stage 3: Merge EHR + RBD -> production parquet ────────────────────
    if RUN_MERGE:
        _header(3, total, "MERGE EHR + RBD + GAIT -> RISK GROUPS")
        t0 = time.time()
        from pipelines.run_merge_ukbb_rbd import main as merge_main
        merge_main(overwrite=OVERWRITE_MERGE)
        _footer("Merge", time.time() - t0)
    else:
        _skip(3, "Merge")

    # ── Stage 4: Table 1 ─────────────────────────────────────────────────
    if RUN_TABLE_ONE:
        _header(4, total, "TABLE 1 — BASELINE CHARACTERISTICS")
        t0 = time.time()
        from library.table_one import main as table_one_main
        table_one_main()
        _footer("Table One", time.time() - t0)
    else:
        _skip(4, "Table One")

    # ── Stage 5: Cox prodromal pipeline ──────────────────────────────────
    if RUN_COX_PIPELINE:
        _header(5, total, "COX PRODROMAL ANALYSIS")
        t0 = time.time()
        from library.cox_prodromal.runner import run_prodromal_pipeline
        run_prodromal_pipeline()
        _footer("Cox Pipeline", time.time() - t0)
    else:

        _skip(5, "Cox Pipeline")

    # ── Stage 6: Sleep phenotype stratification ──────────────────────────
    if RUN_SLEEP_PHENOTYPES:
        _header(6, total, "SLEEP PHENOTYPE vs RBD SCORE EVALUATION")
        t0 = time.time()
        from analysis.sleep_phenotypes_rbd_scores import main as phenotype_main
        phenotype_main()
        _footer("Sleep Phenotypes", time.time() - t0)
    else:
        _skip(6, "Sleep Phenotypes")

    # ── Stage 7: Temporal validation ─────────────────────────────────────
    if RUN_SLEEP_TEMPORAL:
        _header(7, total, "TEMPORAL VALIDATION OF RBD SCORE RELIABILITY")
        t0 = time.time()
        from analysis.sleep_phenotypes_temporal_analysis import main as temporal_main
        temporal_main()
        _footer("Sleep Temporal", time.time() - t0)
    else:
        _skip(7, "Sleep Temporal")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_pipeline
    n_run = sum(1 for _, _, on in flags if on)
    print(f"\n{'=' * 72}")
    print(f"  PIPELINE COMPLETE — {n_run}/{total} stages executed "
          f"in {elapsed_total:.1f}s")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    run_pipeline()
