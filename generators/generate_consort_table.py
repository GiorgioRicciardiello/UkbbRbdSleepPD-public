"""
Generate Supplementary Table S2: CONSORT/STROBE cohort flow.

Design principle
----------------
Maximum granularity. Steps losing 0 subjects are reported because
absence of selection is as informative as presence for bias assessment.
No detail is collapsed. Source file is explicit for every row.

Two CSV outputs + one Excel workbook:
  TableA_cohort_flow.csv      — Linear attrition (all steps, incl. zero-loss flags)
  TableB_outcome_detail.csv   — Wide per-outcome table (all metrics merged)
  Supplementary_Table_S2_CONSORT.xlsx — Workbook with both sheets + ASCII tree

Source logs consumed
--------------------
EHR pipeline (data/pp/data_sheet/logs/):
  controls/control_summary.csv        — overall controls/cases split (N_total)
  outcome_flags/outcome_summary.csv   — per-outcome & base-disease counts
  outcome_flags/censor_diagnostic.csv — HES coverage vs censor date
  outcome_flags/outcome_5a_pd_med_validation.csv — medication-confirmed PD
  exclusion/neuro_exclusion_report.csv   — neuro excl among diagnosed cases
  exclusion/acc_bad_quality_report.csv   — actigraphy quality among cases
  exclusion/shift_worker_report.csv      — shift work among cases
  medications/medication_flags_log.csv   — medication family counts

ABK / RBD pipeline (results/logs/):
  final_consort/staging_consort_long.csv    — subject-level flow post-RBD merge
  consort_rbd/consort_rbd_risk_long.csv     — per-outcome eligible / incident (old run)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from textwrap import indent
from typing import Dict, List, Optional

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from config.config import outcomes as _cfg_outcomes, outcomes_formal_names as _cfg_formal


# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_EHR = Path("data/pp/data_sheet/logs")
LOGS_ABK = Path("results/logs")
OUT_DIR = Path("docs/publication/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = Path("data/pp/res_build_final_dataset/ehr_diag_pd_rbd_only_all.parquet")

# ── Labels (sourced from config.config — single source of truth) ───────────────
OUTCOME_LABELS: Dict[str, str] = _cfg_formal
BASE_LABELS: Dict[str, str] = {
    "pd":  "Parkinson's disease (base ascertainment)",
    "ad":  "Alzheimer's disease (base ascertainment)",
    "dem": "Dementia all-cause (base ascertainment)",
    "dlb": "DLB (base ascertainment)",
}

OUTCOMES = list(OUTCOME_LABELS.keys())


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read(path: Path, index_col: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Return CSV as DataFrame, or None with a warning if missing."""
    if not path.exists():
        warnings.warn(f"Log not found (skipped): {path}", stacklevel=2)
        return None
    return pd.read_csv(path, index_col=index_col)


def _pct(n: Optional[float], denom: Optional[float], decimals: int = 2) -> Optional[str]:
    if n is None or denom is None or denom == 0:
        return None
    return f"{100 * n / denom:.{decimals}f}%"


def _int(x) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _row(
    step_id: str,
    section: str,
    description: str,
    filter_type: str,
    n_before: Optional[int],
    n_excluded: Optional[int],
    n_retained: Optional[int],
    note: str,
    source: str,
) -> Dict:
    pct_excl = _pct(n_excluded, n_before) if n_before and n_excluded else None
    return {
        "step_id":     step_id,
        "section":     section,
        "description": description,
        "filter_type": filter_type,
        "n_before":    n_before,
        "n_excluded":  n_excluded,
        "pct_excluded": pct_excl,
        "n_retained":  n_retained,
        "note":        note,
        "source_file": source,
    }


# ── Runtime exclusion counts (mirrors get_clean_risk_data + make_subject_level) ─

def _compute_runtime_exclusions() -> Optional[Dict]:
    """
    Load the production parquet and replay the exact filtering sequence from
    ``get_clean_risk_data()`` and ``make_subject_level()`` in
    ``library/risk/risk_helpers.py``, capturing subject-level counts at each
    step.  Returns None if the parquet is missing.

    The steps mirror the Cox pipeline entry point in
    ``src/cox_prodromal/data_prep.py::load_prodromal_dataset()``:
      1. Load night-level parquet
      2. Exclude neuro_exclude != 0
      3. Exclude acc_bad_quality == True
      4. Exclude shift_any_i2_p3426 == 1
      5. Collapse to subject-level (groupby eid)
    """
    if not PARQUET_PATH.exists():
        warnings.warn(f"Parquet not found (runtime counts unavailable): {PARQUET_PATH}", stacklevel=2)
        return None

    df = pd.read_parquet(PARQUET_PATH, columns=[
        "eid", "neuro_exclude", "acc_bad_quality", "shift_any_i2_p3426",
    ])

    # Night-level total
    n_nights_total = len(df)
    n_subj_total = df["eid"].nunique()

    # Step 1: neuro_exclude == 0
    df = df[df["neuro_exclude"] == 0].copy()
    n_nights_post_neuro = len(df)
    n_subj_post_neuro = df["eid"].nunique()

    # Step 2: acc_bad_quality != True
    if "acc_bad_quality" in df.columns:
        n_nights_pre_acc = len(df)
        n_subj_pre_acc = df["eid"].nunique()
        df = df[df["acc_bad_quality"] != True].copy()
        n_nights_post_acc = len(df)
        n_subj_post_acc = df["eid"].nunique()
        n_subj_excl_acc = n_subj_pre_acc - n_subj_post_acc
    else:
        n_subj_pre_acc = n_subj_post_neuro
        n_subj_post_acc = n_subj_post_neuro
        n_subj_excl_acc = 0

    # Step 3: shift_any_i2_p3426 != 1
    if "shift_any_i2_p3426" in df.columns:
        n_subj_pre_ns = df["eid"].nunique()
        df = df[df["shift_any_i2_p3426"] != 1].copy()
        n_subj_post_ns = df["eid"].nunique()
        n_subj_excl_ns = n_subj_pre_ns - n_subj_post_ns
    else:
        n_subj_pre_ns = n_subj_post_acc
        n_subj_post_ns = n_subj_post_acc
        n_subj_excl_ns = 0

    # Subject-level (mirrors make_subject_level groupby)
    n_subj_final = df["eid"].nunique()

    return {
        "n_subj_total": n_subj_total,
        "n_subj_post_neuro": n_subj_post_neuro,
        "n_subj_pre_acc": n_subj_pre_acc,
        "n_subj_post_acc": n_subj_post_acc,
        "n_subj_excl_acc": n_subj_excl_acc,
        "n_subj_pre_ns": n_subj_pre_ns,
        "n_subj_post_ns": n_subj_post_ns,
        "n_subj_excl_ns": n_subj_excl_ns,
        "n_subj_final": n_subj_final,
    }


def _compute_table1_n(n_subj_final: Optional[int]) -> Optional[int]:
    """
    Compute the Table 1 cohort size after prevalent-PD exclusion.

    Mirrors src/table_one.py: after make_subject_level(), subjects with NaN in
    ``outcome_1a_pd_only_surv_time`` are dropped (prevalent PD at baseline).
    Only loads the columns needed to compute this count.
    Returns None if the parquet is missing or the surv_time column is absent.
    """
    if not PARQUET_PATH.exists():
        return None

    # Load only eid + surv_days for the primary outcome to compute subject-level count.
    # col_surv_time("outcome_1a_pd_only") returns "outcome_1a_pd_only__surv_days"
    # (double-underscore, days not time — see library/column_registry.py).
    surv_col = "outcome_1a_pd_only__surv_days"
    try:
        df = pd.read_parquet(PARQUET_PATH, columns=[
            "eid", "neuro_exclude", "acc_bad_quality", "shift_any_i2_p3426", surv_col,
        ])
    except Exception:
        return None

    # Replay runtime exclusions (same as _compute_runtime_exclusions)
    df = df[df["neuro_exclude"] == 0]
    if "acc_bad_quality" in df.columns:
        df = df[df["acc_bad_quality"] != True]
    if "shift_any_i2_p3426" in df.columns:
        df = df[df["shift_any_i2_p3426"] != 1]

    # Subject-level (first row per eid, matching make_subject_level)
    df_subj = df.groupby("eid", as_index=False).first()

    # Prevalent PD exclusion: drop NaN surv_time (diagnosed before actigraphy)
    df_subj = df_subj[df_subj[surv_col].notna()]

    return int(len(df_subj))


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE A — Linear attrition flow
# ═══════════════════════════════════════════════════════════════════════════════

def build_table_a() -> pd.DataFrame:  # noqa: C901 — intentionally long, sequential flow
    """
    Construct a linear attrition table covering every pipeline step.

    Rules:
    - Steps with n_excluded = 0 are retained; filter_type = 'informational flag'.
    - Every row carries the source_file so numbers are traceable.
    - Per-outcome cohort rows (S4.*) show outcome-level prevalent exclusion,
      not subject removal from the global cohort.
    """
    rows: List[Dict] = []

    # ── Compute runtime exclusion counts from production parquet ────────────
    rt = _compute_runtime_exclusions()

    # ── Load all source files once ─────────────────────────────────────────────
    df_ctrl      = _read(LOGS_EHR / "controls"     / "control_summary.csv")
    df_censor    = _read(LOGS_EHR / "outcome_flags" / "censor_diagnostic.csv")
    df_med_log   = _read(LOGS_EHR / "medications"  / "medication_flags_log.csv")
    df_outcome   = _read(LOGS_EHR / "outcome_flags" / "outcome_summary.csv")
    df_pd_med    = None  # outcome_5a_pd_med removed (medication-confirmed PD not used)
    df_neuro_ex  = _read(LOGS_EHR / "exclusion"    / "neuro_exclusion_report.csv", index_col=0)
    df_acc       = _read(LOGS_EHR / "exclusion"    / "acc_bad_quality_report.csv", index_col=0)
    df_shift     = _read(LOGS_EHR / "exclusion"    / "shift_worker_report.csv",    index_col=0)
    df_staging   = _read(LOGS_ABK / "final_consort" / "staging_consort_long.csv")
    df_rbd_flow  = _read(LOGS_ABK / "consort_rbd"  / "consort_rbd_risk_long.csv")
    df_merge_diag = _read(LOGS_ABK / "final_consort" / "merge_diagnostics.csv")
    df_no_rbd     = _read(LOGS_ABK / "final_consort" / "no_rbd_diagnostic_breakdown.csv")

    # ── Derived cohort sizes ───────────────────────────────────────────────────
    n_ehr: Optional[int] = None
    if df_ctrl is not None:
        n_ehr    = _int(df_ctrl["N_total"].iloc[0])
        n_ctrl   = _int(df_ctrl["N_controls"].iloc[0])
        n_cases  = _int(df_ctrl["N_cases"].iloc[0])
        pct_ctrl = float(df_ctrl["pct_controls"].iloc[0])
        pct_cas  = float(df_ctrl["pct_cases"].iloc[0])

    n_rbd = n_neuro = n_prev_excl = None
    if df_staging is not None:
        subj = df_staging[df_staging["Metric"] == "Subjects"].reset_index(drop=True)
        if len(subj) >= 1:
            n_rbd      = _int(subj.loc[0, "Before"])
            n_neuro    = _int(subj.loc[0, "After"])
            n_lost_neu = _int(subj.loc[0, "Lost"])
            pct_neu    = float(subj.loc[0, "% Lost"])
        if len(subj) >= 2:
            n_prev_excl = _int(subj.loc[1, "After"])
            n_lost_prev = _int(subj.loc[1, "Lost"])
            pct_prev    = float(subj.loc[1, "% Lost"])

    n_lost_rbd = (n_ehr - n_rbd) if n_ehr and n_rbd else None

    # ── Derived from merge diagnostics log ───────────────────────────────────
    n_sleep_features = n_rbd_scores = n_merged = None
    n_sf_to_rbd_lost = n_ehr_no_rbd = n_rbd_no_ehr = None
    if df_merge_diag is not None:
        def _mval(step: str) -> Optional[int]:
            row = df_merge_diag[df_merge_diag["step"] == step]
            return _int(row["n"].iloc[0]) if len(row) else None
        n_sleep_features  = _mval("sleep_features_v0")
        n_rbd_scores      = _mval("rbd_scores_v0")
        n_sf_to_rbd_lost  = _mval("sf_to_rbd_lost")
        n_merged          = _mval("merged")
        n_ehr_no_rbd      = _mval("ehr_no_rbd_lost")
        n_rbd_no_ehr      = _mval("rbd_no_ehr_lost")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION S0 — EHR pipeline (before RBD merge)
    # ══════════════════════════════════════════════════════════════════════════

    if df_ctrl is not None:
        rows.append(_row(
            step_id     = "S0.1",
            section     = "S0 — EHR pipeline (before RBD merge)",
            description = "Total UKBB participants processed through EHR pipeline",
            filter_type = "cohort definition",
            n_before    = None,
            n_excluded  = None,
            n_retained  = n_ehr,
            note        = (
                f"Controls (never diagnosed with any target disease): {n_ctrl:,} ({pct_ctrl:.2f}%). "
                f"Ever diagnosed with ≥1 target disease: {n_cases:,} ({pct_cas:.2f}%). "
                f"All subjects had a valid actigraphy wear_time_start (filter applied upstream)."
            ),
            source      = str(LOGS_EHR / "controls" / "control_summary.csv"),
        ))

    # S0.2 — Outcome ascertainment summary (base diseases)
    if df_outcome is not None:
        base_rows = df_outcome[df_outcome["outcome"].isin(BASE_LABELS.keys())]
        base_notes = "; ".join(
            f"{BASE_LABELS[r['outcome']]}: diagnosed={r['diagnosed_n']:,}, "
            f"prevalent={r['prevalent_n']:,}, incident={r['incident_n']:,}, "
            f"median TTE={r['median_tte_days']:.0f}d"
            for _, r in base_rows.iterrows()
            if r["outcome"] in BASE_LABELS
        )
        rows.append(_row(
            step_id     = "S0.2",
            section     = "S0 — EHR pipeline (before RBD merge)",
            description = "Base disease ascertainment (HES + first-occurrence fields)",
            filter_type = "informational (classification step; no subjects removed)",
            n_before    = n_ehr,
            n_excluded  = 0,
            n_retained  = n_ehr,
            note        = base_notes or "See outcome_summary.csv",
            source      = str(LOGS_EHR / "outcome_flags" / "outcome_summary.csv"),
        ))

    # S0.3 — Censor diagnostic
    if df_censor is not None:
        stale = df_censor[df_censor["stale_flag"].astype(str).str.lower() == "true"]
        stale_str = "; ".join(
            f"{r['outcome']} (last dx {r['latest_dx']}, gap {r['gap_days']}d before censor)"
            for _, r in stale.iterrows()
        )
        rows.append(_row(
            step_id     = "S0.3",
            section     = "S0 — EHR pipeline (before RBD merge)",
            description = "Censor date diagnostic: HES coverage vs administrative censor",
            filter_type = "informational (data quality check; no subjects removed)",
            n_before    = n_ehr,
            n_excluded  = 0,
            n_retained  = n_ehr,
            note        = (
                f"Administrative censor date: 2025-02-01. "
                f"Outcomes where last observed diagnosis precedes censor (stale HES coverage): "
                f"{stale_str or 'none'}. "
                f"DLB has no UKBB first-occurrence field; HES-only ascertainment ends 2023-03-21 "
                f"(683 days before censor). HR estimates for DLB are conservative."
            ),
            source      = str(LOGS_EHR / "outcome_flags" / "censor_diagnostic.csv"),
        ))

    # S0.4 — Medication flags
    if df_med_log is not None:
        med_str = "; ".join(
            f"{r['family']}={r['n_reported']:,}"
            for _, r in df_med_log.iterrows()
        )
        rows.append(_row(
            step_id     = "S0.4",
            section     = "S0 — EHR pipeline (before RBD merge)",
            description = "Prodromal marker: self-reported medication flags (p20003)",
            filter_type = "informational (ascertainment step; no subjects removed)",
            n_before    = n_ehr,
            n_excluded  = 0,
            n_retained  = n_ehr,
            note        = (
                f"Subjects with ≥1 medication in each family: {med_str}. "
                f"Medication flags are combined with HES ICD-10 flags in Step S0.5 to form "
                f"prodromal_{{marker}} binary variables (pre-baseline restriction applied)."
            ),
            source      = str(LOGS_EHR / "medications" / "medication_flags_log.csv"),
        ))

    # S0.5 — Medication-confirmed PD (outcome_5a)
    if df_pd_med is not None:
        r5a = df_pd_med.iloc[0]
        rows.append(_row(
            step_id     = "S0.5",
            section     = "S0 — EHR pipeline (before RBD merge)",
            description = "Medication-confirmed PD (outcome_5a_pd_med): sensitivity outcome",
            filter_type = "informational (secondary outcome definition; no subjects removed)",
            n_before    = n_ehr,
            n_excluded  = 0,
            n_retained  = n_ehr,
            note        = (
                f"Of {_int(r5a['n_pd_only']):,} PD-only cases, {_int(r5a['n_pd_med']):,} "
                f"({100 - float(r5a['pct_removed']):.2f}%) have concurrent PD medication "
                f"(medication-confirmed). {_int(r5a['n_pd_no_med']):,} have no medication record. "
                f"Medication-confirmed incident PD: {_int(r5a['incident']):,}; "
                f"prevalent: {_int(r5a['prevalent']):,}."
            ),
            source      = str(LOGS_EHR / "outcome_flags" / "outcome_5a_pd_med_validation.csv"),
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION S0.ACT — Actigraphy feature & ML scoring pipeline
    # These steps sit between EHR construction and the downstream RBD merge.
    # Source: results/logs/final_consort/merge_diagnostics.csv
    # ══════════════════════════════════════════════════════════════════════════

    # S0.6 — Sleep features extracted
    rows.append(_row(
        step_id     = "S0.6",
        section     = "S0 — Actigraphy feature & ML scoring pipeline",
        description = "Actigraphy sleep features extracted (run_abk_collection, visit 0)",
        filter_type = "cohort definition (subjects with ≥1 night of processed actigraphy features)",
        n_before    = None,
        n_excluded  = None,
        n_retained  = n_sleep_features,
        note        = (
            f"{n_sleep_features:,} subjects had sleep features (spectral, temporal, non-linear) "
            f"successfully computed from raw CWA actigraphy files at baseline visit (instance 2). "
            f"Features are the input to the ML RBD classification model."
        ) if n_sleep_features else "See merge_diagnostics.csv",
        source      = str(LOGS_ABK / "final_consort" / "merge_diagnostics.csv"),
    ))

    # S0.7 — RBD probability scores generated
    rows.append(_row(
        step_id     = "S0.7",
        section     = "S0 — Actigraphy feature & ML scoring pipeline",
        description = "RBD probability scores generated by ML model (valid nights, visit 0)",
        filter_type = "exclusion (subjects with insufficient valid nights for RBD scoring dropped)",
        n_before    = n_sleep_features,
        n_excluded  = n_sf_to_rbd_lost,
        n_retained  = n_rbd_scores,
        note        = (
            f"Lost {n_sf_to_rbd_lost:,} ({_pct(n_sf_to_rbd_lost, n_sleep_features)}) subjects "
            f"between feature extraction and RBD scoring. Causes: all scored epochs below quality "
            f"threshold, recording too short (<3 scorable nights), or device/data errors preventing "
            f"model inference. Retained {n_rbd_scores:,} subjects with ≥1 valid RBD probability score."
        ) if n_sf_to_rbd_lost and n_sleep_features else "See merge_diagnostics.csv",
        source      = str(LOGS_ABK / "final_consort" / "merge_diagnostics.csv"),
    ))

    # S0.8 — EHR × RBD inner join
    n_ehr_merge = n_ehr  # all EHR subjects enter the merge
    rows.append(_row(
        step_id     = "S0.8",
        section     = "S0 — Actigraphy feature & ML scoring pipeline",
        description = "EHR x RBD inner join: subjects retained in merged analytical dataset",
        filter_type = "exclusion (inner join; subjects without a match on either side dropped)",
        n_before    = n_ehr_merge,
        n_excluded  = n_ehr_no_rbd,
        n_retained  = n_merged,
        note        = (
            f"Inner join on participant ID (eid). "
            f"Lost {n_ehr_no_rbd:,} ({_pct(n_ehr_no_rbd, n_ehr_merge)}) EHR subjects with no RBD scores "
            f"(dominant cause: acc_bad_quality=True; see S0.9 for diagnostic breakdown). "
            f"Also dropped {n_rbd_no_ehr:,} RBD subjects without EHR match "
            f"(likely different data-release batches or withdrawn consent). "
            f"Retained {n_merged:,} subjects with both EHR covariates and actigraphy RBD scores."
        ) if n_ehr_no_rbd and n_ehr_merge else "See merge_diagnostics.csv",
        source      = str(LOGS_ABK / "final_consort" / "merge_diagnostics.csv"),
    ))

    # S0.9 — Diagnostic breakdown of EHR subjects excluded for lack of RBD scores
    if df_no_rbd is not None and n_ehr_no_rbd:
        _olabels = _cfg_formal
        acc_bad_n = _int(df_no_rbd["n_acc_bad_quality"].iloc[0])
        acc_bad_pct = df_no_rbd["pct_acc_bad_quality"].iloc[0]
        diag_strs = "; ".join(
            f"{_olabels.get(r['outcome'], r['outcome'])}: "
            f"diagnosed={_int(r['n_diagnosed'])}, "
            f"prevalent={_int(r['n_prevalent'])}, "
            f"incident={_int(r['n_incident'])}"
            for _, r in df_no_rbd.iterrows()
        )
        rows.append(_row(
            step_id     = "S0.9",
            section     = "S0 — Actigraphy feature & ML scoring pipeline",
            description = "Diagnostic profile of EHR subjects excluded for lack of RBD scores",
            filter_type = "informational (characterisation of excluded group; no additional removal)",
            n_before    = n_ehr_no_rbd,
            n_excluded  = 0,
            n_retained  = n_ehr_no_rbd,
            note        = (
                f"Of {n_ehr_no_rbd:,} EHR subjects without RBD scores: "
                f"{acc_bad_n:,} ({acc_bad_pct:.1f}%) had acc_bad_quality=True "
                f"(failed actigraphy quality criteria: wear time, calibration, or data problems), "
                f"confirming that poor recording quality is the dominant exclusion mechanism. "
                f"Outcome case distribution: {diag_strs}. "
                f"Incident cases in this excluded group represent a potential selection bias; "
                f"these subjects were not randomised to poor recording quality — cases with "
                f"prodromal motor symptoms may have had more recording artefacts."
            ),
            source      = str(LOGS_ABK / "final_consort" / "no_rbd_diagnostic_breakdown.csv"),
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION S1 — RBD integration (actigraphy + ML scoring)
    # ══════════════════════════════════════════════════════════════════════════

    rows.append(_row(
        step_id     = "S1.1",
        section     = "S1 — RBD integration (actigraphy ML scoring)",
        description = "Analytical dataset post-merge: subjects entering survival analysis",
        filter_type = "cohort definition (downstream of EHR x RBD merge at S0.8)",
        n_before    = n_merged,
        n_excluded  = 0,
        n_retained  = n_rbd,
        note        = (
            f"Starting analytical cohort: {n_rbd:,} subjects with matched EHR + RBD scores. "
            f"Corresponds to n_merged={n_merged:,} from S0.8 (values may differ slightly due to "
            f"staging_consort_long.csv being written at a different pipeline stage). "
            f"Subsequent steps (S2, S3) apply hard exclusions within this cohort."
        ) if n_rbd else "See staging_consort_long.csv",
        source      = str(LOGS_ABK / "final_consort" / "staging_consort_long.csv"),
    ))

    # S1.0 — Neurological exclusion (runtime replay; independent of staging CSV)
    rt_n_total     = rt["n_subj_total"]      if rt else None
    rt_n_post_neuro = rt["n_subj_post_neuro"] if rt else None
    rt_n_excl_neuro = (rt_n_total - rt_n_post_neuro) if (rt_n_total and rt_n_post_neuro) else None

    neuro_case_notes = ""
    if df_neuro_ex is not None:
        neuro_case_notes = "; ".join(
            f"{OUTCOME_LABELS.get(idx, idx)}: {int(r['n_neuro_exclude'])}/{int(r['n_outcome'])} "
            f"cases ({float(r['pct_neuro_exclude']):.2f}%)"
            for idx, r in df_neuro_ex.iterrows()
            if idx in OUTCOME_LABELS
        )

    rows.append(_row(
        step_id     = "S1.0",
        section     = "S1 — RBD integration (actigraphy ML scoring)",
        description = "Neurological exclusion (neuro_exclude != 0)",
        filter_type = "hard exclusion (applied in get_clean_risk_data(); pre-baseline neuro dx)",
        n_before    = rt_n_total,
        n_excluded  = rt_n_excl_neuro,
        n_retained  = rt_n_post_neuro,
        note        = (
            "neuro_exclude = True for subjects with ICD-10 neurological/neurodegenerative "
            "diagnoses (G10–G13, G21–G25, G31–G32, G35–G37, G40–G41, G47x, G04–G05, G93x, R56x) "
            "strictly BEFORE wear_time_start. Applied as the first filter in get_clean_risk_data() "
            "(library/risk/risk_helpers.py). "
            + (f"Cohort-level: {rt_n_excl_neuro:,} subjects excluded ({_pct(rt_n_excl_neuro, rt_n_total)}). "
               if rt_n_excl_neuro is not None and rt_n_total
               else "Cohort-level N: parquet not available. ")
            + (f"Among DIAGNOSED cases per outcome: {neuro_case_notes}." if neuro_case_notes else "")
        ),
        source      = f"{PARQUET_PATH} (runtime replay of get_clean_risk_data)",
    ))

    # S1.2 — Actigraphy quality exclusion (hard exclusion from analytical dataset)
    # Counts sourced from runtime replay of get_clean_risk_data() on production parquet.
    rt_n_excl_acc = rt["n_subj_excl_acc"] if rt else None
    rt_n_pre_acc  = rt["n_subj_pre_acc"] if rt else None
    rt_n_post_acc = rt["n_subj_post_acc"] if rt else None

    acc_notes = ""
    if df_acc is not None:
        acc_notes = "; ".join(
            f"{OUTCOME_LABELS.get(idx, idx)}: {int(r['n_acc_bad_quality'])}/{int(r['n_outcome'])} cases "
            f"({float(r['pct_acc_bad_quality']):.1f}%)"
            for idx, r in df_acc.iterrows()
            if idx in OUTCOME_LABELS
        )

    rows.append(_row(
        step_id     = "S1.2",
        section     = "S1 — RBD integration (actigraphy ML scoring)",
        description = "Actigraphy quality exclusion (acc_bad_quality = True)",
        filter_type = "hard exclusion (applied in get_clean_risk_data(); unreliable RBD scores)",
        n_before    = rt_n_pre_acc or n_neuro,
        n_excluded  = rt_n_excl_acc,
        n_retained  = rt_n_post_acc,
        note        = (
            "acc_bad_quality = True if ANY of: p90015=0 (wear time), p90016=0 (calibration), "
            "p90002 in {1,2} (device size), p90180>0 (recording problems). "
            "Exclusion applied in get_clean_risk_data() (library/risk/risk_helpers.py). "
            + (f"Cohort-level: {rt_n_excl_acc:,} subjects excluded ({_pct(rt_n_excl_acc, rt_n_pre_acc)}). "
               if rt_n_excl_acc is not None and rt_n_pre_acc
               else "Cohort-level N: parquet not available. ")
            + (f"Among DIAGNOSED cases per outcome: {acc_notes}." if acc_notes else "")
        ),
        source      = (
            f"{PARQUET_PATH} (runtime replay); "
            + str(LOGS_EHR / "exclusion" / "acc_bad_quality_report.csv")
        ),
    ))

    # S1.3 — Night-shift exclusion at actigraphy time (hard exclusion)
    # Counts sourced from runtime replay of get_clean_risk_data() on production parquet.
    rt_n_excl_ns = rt["n_subj_excl_ns"] if rt else None
    rt_n_pre_ns  = rt["n_subj_pre_ns"] if rt else None
    rt_n_post_ns = rt["n_subj_post_ns"] if rt else None

    ns_case_notes = ""
    if df_shift is not None:
        ns_i2_col = "n_shift_any_i2_p3426"
        if ns_i2_col in df_shift.columns:
            ns_case_notes = "; ".join(
                f"{OUTCOME_LABELS.get(idx, idx)}: {int(r[ns_i2_col])}/{int(r['n_outcome'])} cases "
                f"({float(r['pct_shift_any_i2_p3426']):.1f}%)"
                for idx, r in df_shift.iterrows()
                if idx in OUTCOME_LABELS
            )

    rows.append(_row(
        step_id     = "S1.3",
        section     = "S1 — RBD integration (actigraphy ML scoring)",
        description = "Night-shift exclusion at actigraphy time (shift_any_i2_p3426 = 1)",
        filter_type = "hard exclusion (applied in get_clean_risk_data(); disrupted circadian architecture)",
        n_before    = rt_n_pre_ns or rt_n_post_acc,
        n_excluded  = rt_n_excl_ns,
        n_retained  = rt_n_post_ns,
        note        = (
            "Excluded subjects reporting any night-shift work at instance 2 (p3426, imaging visit = "
            "actigraphy recording time, ~2014+). Night-shift disrupts circadian sleep architecture; "
            "actigraphy signals do not reflect physiological sleep, making RBD probability scores unreliable. "
            "Instance 0 (2006-2010) NOT used: shift status 8+ years before recording is not informative "
            "about recording-time sleep quality. Exclusion applied in get_clean_risk_data() "
            "(library/risk/risk_helpers.py). "
            + (f"Cohort-level: {rt_n_excl_ns:,} subjects excluded ({_pct(rt_n_excl_ns, rt_n_pre_ns)}). "
               if rt_n_excl_ns is not None and rt_n_pre_ns
               else "Cohort-level N: parquet not available. ")
            + (f"Among DIAGNOSED cases per outcome: {ns_case_notes}." if ns_case_notes else "")
        ),
        source      = (
            f"{PARQUET_PATH} (runtime replay); "
            + str(LOGS_EHR / "exclusion" / "shift_worker_report.csv")
        ),
    ))

    # S1.4 — Broader shift work flags (informational — not used for exclusion)
    if df_shift is not None:
        shift_col = "n_shift_any_i0_p826"
        if shift_col in df_shift.columns:
            shift_notes = "; ".join(
                f"{idx}: {int(r[shift_col])}/{int(r['n_outcome'])} ({float(r['pct_shift_any_i0_p826']):.1f}%)"
                for idx, r in df_shift.iterrows()
                if idx in OUTCOME_LABELS
            )
        else:
            shift_notes = "See shift_worker_report.csv"
        rows.append(_row(
            step_id     = "S1.4",
            section     = "S1 — RBD integration (actigraphy ML scoring)",
            description = "Broader shift-work flags (p826, p3426 i0–i3, p22650) — informational",
            filter_type = "informational flag (covariates only; no additional subjects removed here)",
            n_before    = n_rbd,
            n_excluded  = 0,
            n_retained  = n_rbd,
            note        = (
                "Shift work exposure additionally flagged from p826 (rotating/shift at main job) across "
                "4 assessment waves (i0–i3) and from p22650 (night shifts history, 0=Never→3=Always). "
                "These variables are retained as potential confounders for sensitivity analyses. "
                "No hard exclusion applied beyond S1.3 (i2 night shift). "
                f"Shift_any (i0, p826) among DIAGNOSED cases per outcome: {shift_notes}."
            ),
            source      = str(LOGS_EHR / "exclusion" / "shift_worker_report.csv"),
        ))

    # S1.5 — Final analytical cohort after all runtime exclusions
    rt_n_final = rt["n_subj_final"] if rt else None
    rt_n_total = rt["n_subj_total"] if rt else None
    rt_total_excl = (rt_n_total - rt_n_final) if rt_n_total and rt_n_final else None
    rows.append(_row(
        step_id     = "S1.5",
        section     = "S1 — RBD integration (actigraphy ML scoring)",
        description = "Final subject-level analytical cohort entering Cox pipeline",
        filter_type = "summary (result of S2.1 neuro + S1.2 acc_bad_quality + S1.3 night shift)",
        n_before    = rt_n_total,
        n_excluded  = rt_total_excl,
        n_retained  = rt_n_final,
        note        = (
            f"Subject-level cohort after all runtime exclusions applied by get_clean_risk_data() "
            f"+ make_subject_level(). This is the N entering build_survival_dataset_for_outcome() "
            f"in the Cox prodromal pipeline (src/cox_prodromal/data_prep.py). "
            f"Sequential filters: neuro_exclude -> acc_bad_quality -> shift_any_i2_p3426 -> "
            f"groupby(eid).first(). "
            + (f"Final N = {rt_n_final:,} subjects." if rt_n_final else "Parquet not available.")
        ),
        source      = f"{PARQUET_PATH} (runtime replay of get_clean_risk_data + make_subject_level)",
    ))

    # S1.6 — Table 1 cohort: prevalent PD exclusion applied to S1.5
    # Table One (src/table_one.py) drops subjects with NaN in outcome_1a_pd_only_surv_time
    # (i.e. prevalent PD at baseline) before stratifying by risk group.
    # These subjects cannot contribute to the primary-outcome survival analysis but may
    # still be incident for secondary outcomes (AD, dementia, DLB).
    # NaN surv_time means the subject was diagnosed before wear_time_start.
    rt_n_table1 = _compute_table1_n(rt_n_final)
    rt_n_prev_pd = (rt_n_final - rt_n_table1) if rt_n_final and rt_n_table1 else None
    rows.append(_row(
        step_id     = "S1.6",
        section     = "S1 — RBD integration (actigraphy ML scoring)",
        description = "Table 1 cohort: exclude prevalent PD (NaN surv_time for outcome_1a_pd_only)",
        filter_type = (
            "hard exclusion for Table 1 only; Cox pipeline handles prevalent exclusion "
            "per-outcome inside build_survival_dataset_for_outcome()"
        ),
        n_before    = rt_n_final,
        n_excluded  = rt_n_prev_pd,
        n_retained  = rt_n_table1,
        note        = (
            f"Table 1 (src/table_one.py) stratifies by RBD risk group using outcome_1a_pd_only "
            f"thresholds and drops subjects prevalent for PD before this stratification: "
            f"NaN in outcome_1a_pd_only_surv_time indicates diagnosis before actigraphy baseline. "
            + (f"Removed {rt_n_prev_pd:,} prevalent-PD subjects, retaining {rt_n_table1:,}. "
               if rt_n_prev_pd is not None else "Count unavailable (parquet missing). ")
            + "These subjects are NOT excluded from Cox per-outcome analyses for secondary outcomes "
            "(AD, dementia, DLB) — each outcome has its own prevalent filter in "
            "select_survival_dataset(). "
            "Note: risk group assignment (percentile thresholds) is not affected by prevalent status; "
            "all 88,115 subjects have a valid risk group. The NaN check is purely on surv_time."
        ),
        source      = f"{PARQUET_PATH} (runtime replay); src/table_one.py:718",
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION S2 — Neurological exclusion
    # ══════════════════════════════════════════════════════════════════════════

    if n_rbd and n_neuro and n_lost_neu is not None:
        # Per-outcome neuro excl impact among diagnosed
        neuro_case_notes = ""
        if df_neuro_ex is not None:
            neuro_case_notes = "; ".join(
                f"{OUTCOME_LABELS.get(idx, idx)}: {int(r['n_neuro_exclude'])}/{int(r['n_outcome'])} "
                f"cases ({float(r['pct_neuro_exclude']):.2f}%)"
                for idx, r in df_neuro_ex.iterrows()
                if idx in OUTCOME_LABELS
            )
        rows.append(_row(
            step_id     = "S2.1",
            section     = "S2 — Neurological exclusion",
            description = "Pre-baseline neurological exclusion (neuro_exclude = True)",
            filter_type = "hard exclusion (all analyses)",
            n_before    = n_rbd,
            n_excluded  = n_lost_neu,
            n_retained  = n_neuro,
            note        = (
                f"Excluded {n_lost_neu:,} ({pct_neu:.2f}%) subjects with neurological diagnoses "
                f"(ICD-10: G10–G13, G21–G25, G31–G32, G35–G37, G40–G41, G47x, G04–G05, G93x, R56x) "
                f"strictly BEFORE wear_time_start. Post-baseline diagnoses retained as potential incident outcomes. "
                f"Prevents pre-existing neuropathology from confounding the RBD–outcome association. "
                f"Impact on incident case counts per outcome: {neuro_case_notes}."
            ),
            source      = (
                f"{LOGS_ABK}/final_consort/staging_consort_long.csv; "
                f"{LOGS_EHR}/exclusion/neuro_exclusion_report.csv"
            ),
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION S3 — Global prevalent exclusion
    # ══════════════════════════════════════════════════════════════════════════

    if n_neuro and n_prev_excl is not None and n_lost_prev is not None:
        rows.append(_row(
            step_id     = "S3.1",
            section     = "S3 — Global prevalent exclusion",
            description = "Prevalent case exclusion (diagnosis predates actigraphy baseline)",
            filter_type = "hard exclusion (subjects prevalent for ≥1 outcome removed from cohort)",
            n_before    = n_neuro,
            n_excluded  = n_lost_prev,
            n_retained  = n_prev_excl,
            note        = (
                f"Excluded {n_lost_prev:,} ({pct_prev:.2f}%) subjects whose target disease "
                f"diagnosis date precedes wear_time_start. These subjects cannot contribute an "
                f"incident event and their baseline actigraphy is post-diagnosis (reverse causation risk). "
                f"Note: a subject prevalent for PD may still be incident for AD; the per-outcome prevalent "
                f"exclusion in S4.* is more granular (sum of per-outcome exclusions > {n_lost_prev:,} due to overlap)."
            ),
            source      = str(LOGS_ABK / "final_consort" / "staging_consort_long.csv"),
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION S4 — Per-outcome analytical cohorts
    # ══════════════════════════════════════════════════════════════════════════

    # Per-outcome eligible & incident from RBD flow log
    rbd_eligible: Dict[str, int] = {}
    rbd_incident_abk: Dict[str, int] = {}
    if df_rbd_flow is not None:
        for _, row in df_rbd_flow.iterrows():
            metric = str(row["Metric"])
            after  = _int(row["After"])
            if "eligible subjects" in metric:
                key = metric.replace(" eligible subjects", "").lower()
                rbd_eligible[key] = after
            elif "incident cases" in metric:
                key = metric.replace(" incident cases", "").lower()
                rbd_incident_abk[key] = after

    # Build per-outcome lookup from EHR outcome_summary (most granular)
    outcome_lookup: Dict[str, Dict] = {}
    if df_outcome is not None:
        for _, row in df_outcome.iterrows():
            outcome_lookup[str(row["outcome"])] = row.to_dict()

    for i, oc in enumerate(OUTCOMES, start=1):
        label = OUTCOME_LABELS[oc]
        oc_key_rbd = oc.upper()

        # Prefer runtime final cohort (post all exclusions) as eligible base;
        # fall back to RBD flow log, then staging log.
        rt_final = rt["n_subj_final"] if rt else None
        n_eligible = rbd_eligible.get(oc_key_rbd, rt_final or n_prev_excl)
        n_inc_abk  = rbd_incident_abk.get(oc_key_rbd)

        ol = outcome_lookup.get(oc, {})
        n_diag = _int(ol.get("diagnosed_n"))
        n_prev = _int(ol.get("prevalent_n"))
        n_inc  = _int(ol.get("incident_n"))
        n_comp = _int(ol.get("competing_n"))
        med_tte = ol.get("median_tte_days")
        inc_pct = ol.get("incident_pct")

        n_incident_display = n_inc or n_inc_abk

        rows.append(_row(
            step_id     = f"S4.{i}",
            section     = "S4 — Per-outcome analytical cohort",
            description = f"Analytical cohort: {label} ({oc})",
            filter_type = "outcome-specific prevalent exclusion (subjects not removed from other outcomes)",
            n_before    = n_eligible,
            n_excluded  = n_prev,
            n_retained  = (n_eligible - n_prev) if n_eligible and n_prev else None,
            note        = (
                f"Incident cases: {n_incident_display:,}. "
                f"Prevalent excluded: {n_prev:,}. "
                f"Diagnosed total (EHR log): {n_diag:,}. "
                f"Competing events: {n_comp:,}. "
                f"Incident rate: {inc_pct:.3f}%. "
                f"Median time-to-event: {med_tte:.0f} days."
                if all(x is not None for x in [n_incident_display, n_prev, n_diag, n_comp, med_tte, inc_pct])
                else f"Incident cases: {n_incident_display}; prevalent excluded: {n_prev}."
            ),
            source      = (
                f"{LOGS_EHR}/outcome_flags/outcome_summary.csv; "
                f"{LOGS_ABK}/consort_rbd/consort_rbd_risk_long.csv"
            ),
        ))

    return pd.DataFrame(rows, columns=[
        "step_id", "section", "description", "filter_type",
        "n_before", "n_excluded", "pct_excluded", "n_retained",
        "note", "source_file",
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE B — Per-outcome wide table
# ═══════════════════════════════════════════════════════════════════════════════

def build_table_b() -> pd.DataFrame:
    """
    One row per outcome. Every metric from every log file merged side by side.
    Nothing is dropped — missing values appear as NaN.
    """
    df_outcome  = _read(LOGS_EHR / "outcome_flags" / "outcome_summary.csv")
    df_censor   = _read(LOGS_EHR / "outcome_flags" / "censor_diagnostic.csv")
    df_neuro_ex = _read(LOGS_EHR / "exclusion"     / "neuro_exclusion_report.csv",  index_col=0)
    df_acc      = _read(LOGS_EHR / "exclusion"     / "acc_bad_quality_report.csv",  index_col=0)
    df_shift    = _read(LOGS_EHR / "exclusion"     / "shift_worker_report.csv",      index_col=0)
    df_rbd_flow = _read(LOGS_ABK / "consort_rbd"   / "consort_rbd_risk_long.csv")

    rows: List[Dict] = []

    # Per-outcome incident counts from ABK log
    rbd_eligible: Dict[str, Optional[int]] = {}
    rbd_incident: Dict[str, Optional[int]] = {}
    if df_rbd_flow is not None:
        for _, row in df_rbd_flow.iterrows():
            metric = str(row["Metric"])
            after  = _int(row["After"])
            if "eligible subjects" in metric:
                rbd_eligible[metric.replace(" eligible subjects", "").lower()] = after
            elif "incident cases" in metric:
                rbd_incident[metric.replace(" incident cases", "").lower()] = after

    # Censor lookup by outcome
    censor_lut: Dict[str, Dict] = {}
    if df_censor is not None:
        for _, row in df_censor.iterrows():
            censor_lut[str(row["outcome"])] = row.to_dict()

    for oc in OUTCOMES:
        label = OUTCOME_LABELS[oc]
        r: Dict = {
            "outcome":      oc,
            "outcome_label": label,
        }

        # — EHR outcome_summary —
        if df_outcome is not None:
            ol = df_outcome[df_outcome["outcome"] == oc]
            if not ol.empty:
                ol = ol.iloc[0]
                r["ascertainment_source"]  = ol.get("source", "")
                r["diagnosed_n_ehr"]       = _int(ol.get("diagnosed_n"))
                r["prevalent_n_ehr"]       = _int(ol.get("prevalent_n"))
                r["incident_n_ehr"]        = _int(ol.get("incident_n"))
                r["competing_n_ehr"]       = _int(ol.get("competing_n"))
                r["incident_pct_ehr"]      = ol.get("incident_pct")
                r["median_tte_days_ehr"]   = ol.get("median_tte_days")
                r["earliest_dx"]           = ol.get("earliest_dx")
                r["latest_dx"]             = ol.get("latest_dx")

        # — Censor diagnostic —
        cl = censor_lut.get(oc, {})
        r["censor_date"]        = cl.get("censor_date")
        r["gap_to_censor_days"] = cl.get("gap_days")
        r["stale_hes_flag"]     = cl.get("stale_flag")

        # — ABK flow (different pipeline run) —
        oc_key_rbd = oc.upper()
        r["n_eligible_abk"]    = rbd_eligible.get(oc_key_rbd)
        r["n_incident_abk"]    = rbd_incident.get(oc_key_rbd)

        # — Neuro exclusion impact among diagnosed cases —
        if df_neuro_ex is not None and oc in df_neuro_ex.index:
            ne = df_neuro_ex.loc[oc]
            r["n_cases_total"]              = _int(ne["n_outcome"])
            r["n_neuro_excl_among_cases"]   = _int(ne["n_neuro_exclude"])
            r["pct_neuro_excl_among_cases"] = float(ne["pct_neuro_exclude"])

        # — Actigraphy quality among diagnosed cases —
        if df_acc is not None and oc in df_acc.index:
            ac = df_acc.loc[oc]
            r["n_acc_bad_quality_among_cases"]  = _int(ac["n_acc_bad_quality"])
            r["pct_acc_bad_quality_among_cases"] = float(ac["pct_acc_bad_quality"])

        # — Shift work among diagnosed cases —
        if df_shift is not None and oc in df_shift.index:
            sh = df_shift.loc[oc]
            # Key shift metrics (i0 = baseline assessment)
            for col in [
                "n_shift_any_i0_p826",   "pct_shift_any_i0_p826",
                "n_shift_high_i0_p826",  "pct_shift_high_i0_p826",
                "n_night_shift_ever",    "pct_night_shift_ever",
                "n_night_shift_high",    "pct_night_shift_high",
            ]:
                if col in sh.index:
                    r[col] = sh[col]

        rows.append(r)

    df_b = pd.DataFrame(rows)
    return df_b


# ═══════════════════════════════════════════════════════════════════════════════
# ASCII CONSORT TREE
# ═══════════════════════════════════════════════════════════════════════════════

def build_ascii_tree(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    """
    Generate an ASCII CONSORT tree from pre-built table data.
    All numbers are derived from the tables — no re-reading of files.
    """

    def _get(step: str, col: str) -> str:
        rows = df_a[df_a["step_id"] == step]
        if rows.empty:
            return "N/A"
        v = rows.iloc[0][col]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "N/A"
        # Percentage strings (e.g. "5.81%") — return as-is
        if isinstance(v, str):
            return v
        try:
            return f"{int(v):,}"
        except (TypeError, ValueError):
            return str(v)

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  CONSORT / STROBE COHORT FLOW — UKB Actigraphy RBD Study")
    lines.append("=" * 72)

    lines.append("")
    lines.append(f"  +-----------------------------------------------------+")
    lines.append(f"  |  S0.1  EHR pipeline: all UKBB actigraphy subjects   |")
    lines.append(f"  |         N = {_get('S0.1','n_retained'):>10}                              |")
    lines.append(f"  +------------------------+----------------------------+")
    lines.append(f"                           | EXCLUDED: no valid RBD score")
    lines.append(f"                           |   n = {_get('S0.8','n_excluded'):>8} ({_get('S0.8','pct_excluded')})")
    lines.append(f"                           v")
    lines.append(f"  +-----------------------------------------------------+")
    lines.append(f"  |  S0.8  After EHR x RBD merge                        |")
    lines.append(f"  |         N = {_get('S0.8','n_retained'):>10}                              |")
    lines.append(f"  +------------------------+----------------------------+")
    lines.append(f"                           | EXCLUDED: pre-baseline neuro dx")
    lines.append(f"                           |   n = {_get('S1.0','n_excluded'):>8} ({_get('S1.0','pct_excluded')})")
    lines.append(f"                           v")
    lines.append(f"  +-----------------------------------------------------+")
    lines.append(f"  |  S1.0  After neurological exclusion                  |")
    lines.append(f"  |         N = {_get('S1.0','n_retained'):>10}                              |")
    lines.append(f"  +------------------------+----------------------------+")
    lines.append(f"                           | EXCLUDED: acc_bad_quality")
    lines.append(f"                           |   n = {_get('S1.2','n_excluded'):>8} ({_get('S1.2','pct_excluded')})")
    lines.append(f"                           v")
    lines.append(f"  +-----------------------------------------------------+")
    lines.append(f"  |  S1.2  After actigraphy quality exclusion            |")
    lines.append(f"  |         N = {_get('S1.2','n_retained'):>10}                              |")
    lines.append(f"  +------------------------+----------------------------+")
    lines.append(f"                           | EXCLUDED: night shift (i2)")
    lines.append(f"                           |   n = {_get('S1.3','n_excluded'):>8} ({_get('S1.3','pct_excluded')})")
    lines.append(f"                           v")
    lines.append(f"  +-----------------------------------------------------+")
    lines.append(f"  |  S1.3  After night-shift exclusion                   |")
    lines.append(f"  |         N = {_get('S1.3','n_retained'):>10}                              |")
    lines.append(f"  +------------------------+----------------------------+")
    lines.append(f"                           |")
    lines.append(f"                           v")
    lines.append(f"  +-----------------------------------------------------+")
    lines.append(f"  |  S1.5  Final cohort entering Cox pipeline            |")
    lines.append(f"  |         N = {_get('S1.5','n_retained'):>10}                              |")
    lines.append(f"  +------------------------+----------------------------+")
    lines.append(f"                           |  Each outcome: per-outcome")
    lines.append(f"                           |  prevalent exclusion (S4.*)")
    lines.append(f"                           v")

    lines.append("")
    lines.append("  Per-outcome analytical cohorts")
    lines.append("  " + "-" * 68)
    lines.append(
        f"  {'Outcome':<38} {'N eligible':>10}  {'Prevalent':>9}  {'Incident':>8}"
    )
    lines.append("  " + "-" * 68)

    for i, oc in enumerate(OUTCOMES, start=1):
        sid = f"S4.{i}"
        label = OUTCOME_LABELS[oc][:37]
        n_elig = _get(sid, "n_before")
        n_prev = _get(sid, "n_excluded")
        n_inc  = "N/A"
        # Get incident from table B
        b_row = df_b[df_b["outcome"] == oc]
        if not b_row.empty:
            v = b_row.iloc[0].get("incident_n_ehr")
            if pd.notna(v):
                n_inc = f"{int(v):,}"

        lines.append(f"  {label:<38} {n_elig:>10}  {n_prev:>9}  {n_inc:>8}")

    lines.append("  " + "-" * 68)
    lines.append("")
    lines.append("  Notes")
    lines.append("  -----")
    lines.append("  Exclusion order mirrors get_clean_risk_data() (library/risk/risk_helpers.py):")
    lines.append("    1. neuro_exclude == 0  (S1.0)")
    lines.append("    2. acc_bad_quality != True  (S1.2)")
    lines.append("    3. shift_any_i2_p3426 != 1  (S1.3)")
    lines.append("    4. groupby(eid).first() -> subject-level  (S1.5)")
    lines.append(f"  neuro excl  (S1.0): n = {_get('S1.0','n_excluded')} excluded.")
    lines.append(f"  acc_bad_quality (S1.2): n = {_get('S1.2','n_excluded')} excluded.")
    lines.append(f"  night shift i2 (S1.3): n = {_get('S1.3','n_excluded')} excluded.")
    lines.append("  broader shift-work (S1.4): flagged as covariates only, not excluded.")
    lines.append("  neuro exclusion: pre-baseline ICD-10 codes only.")
    lines.append("  prevalent exclusion: per-outcome (sum > global S3.1 due to overlap).")
    lines.append("  DLB: HES-only ascertainment (no UKBB first-occurrence field);")
    lines.append("    HES coverage ends 2023-03-21 (683 days before censor 2025-02-01).")
    lines.append("=" * 72)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    """Build all tables and write outputs."""
    print("Building Table A (linear attrition flow)...")
    df_a = build_table_a()

    print("Building Table B (per-outcome detail)...")
    df_b = build_table_b()

    print("Building ASCII CONSORT tree...")
    tree = build_ascii_tree(df_a, df_b)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    path_a = OUT_DIR / "TableA_cohort_flow.csv"
    path_b = OUT_DIR / "TableB_outcome_detail.csv"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    print(f"\nSaved: {path_a}")
    print(f"Saved: {path_b}")

    # ── Save Excel workbook ────────────────────────────────────────────────────
    # ── Save ASCII tree as plain text ────────────────────────────────────────
    path_tree = OUT_DIR / "CONSORT_tree.txt"
    path_tree.write_text(tree, encoding="utf-8")
    print(f"Saved: {path_tree}")

    # ── Save Excel workbook ────────────────────────────────────────────────────
    path_xlsx = OUT_DIR / "Supplementary_Table_S2_CONSORT.xlsx"
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        df_a.to_excel(writer, sheet_name="A - Cohort flow",      index=False)
        df_b.to_excel(writer, sheet_name="B - Outcome detail",   index=False)
        # Sanitize tree lines: prefix with space to prevent Excel interpreting
        # leading +, -, =, | as formula characters.
        safe_lines = [
            f" {line}" if line and line[0] in "+-=|@" else line
            for line in tree.splitlines()
        ]
        tree_df = pd.DataFrame({"CONSORT flow (text)": safe_lines})
        tree_df.to_excel(writer, sheet_name="CONSORT tree",       index=False)
    print(f"Saved: {path_xlsx}")

    # ── Print to console ───────────────────────────────────────────────────────
    sep = "=" * 72

    print(f"\n{sep}")
    print("  TABLE A — COMPLETE ATTRITION FLOW")
    print(sep)
    try:
        display_cols = ["step_id", "section", "description", "filter_type",
                        "n_before", "n_excluded", "pct_excluded", "n_retained"]
        print(df_a[display_cols].to_string(index=False).encode("ascii", errors="replace").decode())
    except Exception:
        print(f"  ({len(df_a)} rows)")

    print(f"\n{sep}")
    print("  TABLE B — PER-OUTCOME DETAIL")
    print(sep)
    try:
        display_cols_b = [
            "outcome_label",
            "diagnosed_n_ehr", "prevalent_n_ehr", "incident_n_ehr", "competing_n_ehr",
            "median_tte_days_ehr", "stale_hes_flag",
            "n_neuro_excl_among_cases", "pct_neuro_excl_among_cases",
            "n_acc_bad_quality_among_cases", "pct_acc_bad_quality_among_cases",
            "n_shift_any_i0_p826", "pct_shift_any_i0_p826",
            "n_night_shift_ever",
        ]
        display_cols_b = [c for c in display_cols_b if c in df_b.columns]
        print(df_b[display_cols_b].to_string(index=False).encode("ascii", errors="replace").decode())
    except Exception:
        print(f"  ({len(df_b)} rows)")

    print(tree.encode("ascii", errors="replace").decode())


if __name__ == "__main__":
    run()
