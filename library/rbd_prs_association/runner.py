"""
Runner: RBD–PRS biological strength analysis.

Execution order
───────────────
1. Load analytical dataset (PRS-complete subjects only).
2. Compute descriptive statistics.
3. Run statistical chain (Spearman → OLS → GAM) for full cohort and
   high-risk subgroup.
4. Save all tables and print console summary.
5. Generate all publication figures.

Usage
─────
    python -m src.rbd_prs_association.runner

from the project root directory with stats_env activated.
"""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows terminals (CP1252 cannot encode ─, ρ, →)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from library.rbd_prs_association.analysis import compute_descriptives, run_all_analyses
from library.rbd_prs_association.config import (
    HIGH_RISK_LABEL,
    RANDOM_SEED,
    RISK_GROUP_COL,
)
from library.rbd_prs_association.data_prep import load_analysis_dataset
from library.rbd_prs_association.plotting import generate_all_figures
from library.rbd_prs_association.reporting import save_all_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_rbd_prs_association(path_results: Path) -> None:
    """Orchestrate the full RBD–PRS association analysis.

    Parameters
    ----------
    path_results : Path
        Root output directory.  Sub-directories figures/ and tables/ are
        created here.  All downstream functions receive these paths
        explicitly — no global output paths are used.
    """
    # ── 1. Paths ─────────────────────────────────────────────────────────────
    out_dir_figs: Path = path_results / "figures"
    out_dir_tabs: Path = path_results / "tables"
    out_dir_figs.mkdir(parents=True, exist_ok=True)
    out_dir_tabs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("RBD–PRS BIOLOGICAL STRENGTH ANALYSIS")
    logger.info("Random seed: %d  |  Output root: %s", RANDOM_SEED, path_results)
    logger.info("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("[1/5] Loading analytical dataset (PRS-complete subjects) ...")
    df, active_covariates = load_analysis_dataset()
    n_total = df["id"].nunique() if "id" in df.columns else len(df)
    n_high = (df[RISK_GROUP_COL] == HIGH_RISK_LABEL).sum()
    logger.info("  N (analytical) = %d  |  High-risk (99th pctl) = %d", n_total, n_high)
    logger.info("  Active covariates (%d): %s", len(active_covariates), active_covariates)

    # ── 2. Descriptive statistics ────────────────────────────────────────────
    logger.info("[2/5] Computing descriptive statistics ...")
    descriptive_df = compute_descriptives(df, active_covariates)

    # ── 3. Statistical analyses ──────────────────────────────────────────────
    logger.info("[3/5] Running statistical chain (Spearman → OLS → GAM) ...")
    logger.info("  Note: permutation test uses %d iterations (seed=%d)", 10_000, RANDOM_SEED)
    results = run_all_analyses(df, active_covariates)
    logger.info(
        "  Completed: %d Spearman, %d OLS, %d GAM results",
        len(results.spearman), len(results.ols), len(results.gam),
    )

    # ── 4. Tables and console summary ───────────────────────────────────────
    logger.info("[4/5] Saving tables and printing summary ...")
    save_all_tables(results, descriptive_df, out_dir_tabs)

    # ── 5. Figures ───────────────────────────────────────────────────────────
    logger.info("[5/5] Generating publication figures ...")
    generate_all_figures(df, results, active_covariates, out_dir_figs)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("DONE in %.1f s", elapsed)
    logger.info("  Tables → %s", out_dir_tabs)
    logger.info("  Figures → %s", out_dir_figs)
    logger.info("=" * 60)


if __name__ == "__main__":
    # Read from canonical production directories — populated by run_merge_ukbb_rbd.py
    # which promotes ABK outputs from data/pp/res_build_final_dataset/abk/ to the
    # root paths here.  No mode subdirectory needed (mirrors run_cox_pipeline.py).
    from config.config import config
    path_results = config["results"]["root"] / "rbds_prs_assoc"
    path_results.mkdir(parents=True, exist_ok=True)
    run_rbd_prs_association(path_results)
