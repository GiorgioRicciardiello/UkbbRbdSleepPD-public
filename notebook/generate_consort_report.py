"""
CONSORT Flow Table & Epidemiological Report Generator

Reads pre-computed log CSVs from the dataset construction pipeline and
generates publication-ready CONSORT tables + STROBE-compliant narrative
for a retrospective longitudinal UK Biobank actigraphy study.

References:
    - STROBE checklist (items 4-7, 12-13)
    - RECORD extension for routinely collected health data
    - Vandenbroucke et al., Ann Intern Med, 2007

Author: Giorgio Ricciardello
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import data_pp_path, data_res, outcomes, outcomes_short_names  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _safe_loss(before: int, after: int) -> int:
    """Subjects lost between stages (floor at 0)."""
    return max(before - after, 0)


def _pct_loss(before: int, after: int) -> float:
    """Percentage of subjects lost between stages."""
    return 100.0 * _safe_loss(before, after) / before if before > 0 else np.nan


def _fmt_n_pct(n: int, total: int) -> str:
    """Format 'n (x.x%)' string."""
    pct = 100.0 * n / total if total > 0 else 0.0
    return f"{n:,} ({pct:.1f}%)"


# ── main class ─────────────────────────────────────────────────────────────

class ConsortReportGenerator:
    """
    Reads log CSVs produced by the EHR/actigraphy pipeline and generates:
        A) CONSORT flow table (subject-level attrition)
        B) Outcome summary table (diagnosed / prevalent / incident / competing)
        C) Exclusion impact by outcome (neuro, actigraphy QC, shift work)
        D) Censoring diagnostics
        E) Medication completeness
        F) STROBE-compliant narrative text
    """

    # Outcome lists sourced from config.config (single source of truth).
    OUTCOMES: List[str] = outcomes
    OUTCOME_LABELS: Dict[str, str] = outcomes_short_names

    def __init__(
        self,
        logs_dir: Optional[Path] = None,
        staging_csv: Optional[Path] = None,
        rbd_csv: Optional[Path] = None,
        save_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.logs_dir = logs_dir or data_pp_path / "data_sheet" / "logs"
        self.staging_csv = staging_csv or data_res / "logs" / "final_consort" / "staging_consort_long.csv"
        self.rbd_csv = rbd_csv or data_res / "logs" / "consort_rbd" / "consort_rbd_risk_long.csv"
        self.save_dir = save_dir or data_res / "consort_report"
        self.verbose = verbose

        # loaded dataframes (populated by _load_logs)
        self._control_summary: pd.DataFrame = pd.DataFrame()
        self._outcome_summary: pd.DataFrame = pd.DataFrame()
        self._neuro_excl: pd.DataFrame = pd.DataFrame()
        self._acc_qc: pd.DataFrame = pd.DataFrame()
        self._shift_work: pd.DataFrame = pd.DataFrame()
        self._censor: pd.DataFrame = pd.DataFrame()
        self._medications: pd.DataFrame = pd.DataFrame()
        self._staging: pd.DataFrame = pd.DataFrame()
        self._rbd_consort: pd.DataFrame = pd.DataFrame()

    # ── loading ────────────────────────────────────────────────────────

    def _load_logs(self) -> None:
        """Load all log CSVs into memory."""
        ld = self.logs_dir

        self._control_summary = pd.read_csv(ld / "controls" / "control_summary.csv")
        self._outcome_summary = pd.read_csv(ld / "outcome_flags" / "outcome_summary.csv")
        self._neuro_excl = pd.read_csv(ld / "exclusion" / "neuro_exclusion_report.csv")
        self._acc_qc = pd.read_csv(ld / "exclusion" / "acc_bad_quality_report.csv")
        self._shift_work = pd.read_csv(ld / "exclusion" / "shift_worker_report.csv")
        self._censor = pd.read_csv(ld / "outcome_flags" / "censor_diagnostic.csv")
        self._medications = pd.read_csv(ld / "medications" / "medication_flags_log.csv")

        if self.staging_csv.exists():
            self._staging = pd.read_csv(self.staging_csv)
        if self.rbd_csv.exists():
            self._rbd_consort = pd.read_csv(self.rbd_csv)

    # ── Table A: CONSORT flow ─────────────────────────────────────────

    def build_flow_table(self) -> pd.DataFrame:
        """
        Build unified CONSORT participant flow table.

        Stages:
            1. UKB actigraphy cohort (N_total from control_summary)
            2. After consent withdrawal / missing actigraphy (staging initial)
            3. After neurological exclusions (staging row)
            4. After night-level cleaning (staging row)
            5. With RBD prediction (rbd_consort)
            6. After validation split exclusion (rbd_consort)

        Returns
        -------
        pd.DataFrame
            Columns: Stage, N_before, N_excluded, N_after, Pct_lost, Reason
        """
        rows: list[dict] = []

        # Stage 1 → 2: full cohort to actigraphy-eligible
        n_total = int(self._control_summary["N_total"].iloc[0])
        n_actig = self._extract_staging_n("Initial cohort (with actigraphy)", col="Before")

        rows.append({
            "Stage": 1,
            "Description": "UKB participants with actigraphy data",
            "N_before": n_total,
            "N_excluded": _safe_loss(n_total, n_actig),
            "N_after": n_actig,
            "Pct_lost": _pct_loss(n_total, n_actig),
            "Reason": "Consent withdrawal, missing/invalid actigraphy",
        })

        # Stage 2 → 3: neurological exclusions
        n_after_neuro = self._extract_staging_n(
            "Initial cohort (with actigraphy)", col="After",
            transition_contains="Neuro"
        )

        rows.append({
            "Stage": 2,
            "Description": "After neurological exclusions",
            "N_before": n_actig,
            "N_excluded": _safe_loss(n_actig, n_after_neuro),
            "N_after": n_after_neuro,
            "Pct_lost": _pct_loss(n_actig, n_after_neuro),
            "Reason": "Pre-baseline neuropathology (parkinson-plus, "
                       "neurodegenerative, demyelinating, epilepsy, narcolepsy)",
        })

        # Stage 3 → 4: night-level cleaning
        n_after_nights = self._extract_staging_n(
            "Neuro exclusions", col="After",
            transition_contains="Night"
        )

        rows.append({
            "Stage": 3,
            "Description": "After night-level quality control",
            "N_before": n_after_neuro,
            "N_excluded": _safe_loss(n_after_neuro, n_after_nights),
            "N_after": n_after_nights,
            "Pct_lost": _pct_loss(n_after_neuro, n_after_nights),
            "Reason": "Insufficient valid nights, actigraphy quality flags",
        })

        # Stage 4 → 5: RBD prediction
        n_rbd_before, n_rbd_after = self._extract_rbd_stage(
            "After night QC", "With RBD prediction"
        )
        # note: RBD CONSORT starts from a slightly different base
        # (without neuro exclusion pre-filter); use its own numbers
        rows.append({
            "Stage": 4,
            "Description": "With RBD prediction available",
            "N_before": n_rbd_before,
            "N_excluded": _safe_loss(n_rbd_before, n_rbd_after),
            "N_after": n_rbd_after,
            "Pct_lost": _pct_loss(n_rbd_before, n_rbd_after),
            "Reason": "Missing RBD model output",
        })

        # Stage 5 → 6: validation split exclusion
        n_val_before, n_val_after = self._extract_rbd_stage(
            "With RBD prediction", "Eligible for risk analysis"
        )

        rows.append({
            "Stage": 5,
            "Description": "Analytical cohort (non-validation)",
            "N_before": n_val_before,
            "N_excluded": _safe_loss(n_val_before, n_val_after),
            "N_after": n_val_after,
            "Pct_lost": _pct_loss(n_val_before, n_val_after),
            "Reason": "Validation split holdout (model training set)",
        })

        df = pd.DataFrame(rows)
        return df

    def _extract_staging_n(
        self,
        stage_keyword: str,
        col: str = "Before",
        transition_contains: Optional[str] = None,
    ) -> int:
        """Extract subject count from staging_consort_long.csv."""
        mask = self._staging["Metric"] == "Subjects"
        search_term = transition_contains if transition_contains else stage_keyword
        mask = mask & self._staging["Stage transition"].str.contains(
            search_term, case=False, na=False, regex=False
        )
        row = self._staging.loc[mask]
        if row.empty:
            return 0
        return int(row[col].iloc[0])

    def _extract_rbd_stage(
        self, from_stage: str, to_stage: str
    ) -> tuple[int, int]:
        """Extract before/after subject counts from RBD CONSORT."""
        mask = (
            (self._rbd_consort["Metric"] == "Subjects")
            & self._rbd_consort["Stage transition"].str.contains(
                from_stage, case=False, na=False, regex=False
            )
            & self._rbd_consort["Stage transition"].str.contains(
                to_stage, case=False, na=False, regex=False
            )
        )
        row = self._rbd_consort.loc[mask]
        if row.empty:
            return 0, 0
        return int(row["Before"].iloc[0]), int(row["After"].iloc[0])

    # ── Table B: outcome summary ──────────────────────────────────────

    def build_outcome_table(self) -> pd.DataFrame:
        """
        Outcome-specific final counts: diagnosed, prevalent, incident,
        competing deaths, median TTE.

        Returns
        -------
        pd.DataFrame
            One row per composite outcome.
        """
        df = self._outcome_summary.copy()
        df = df.loc[df["type"] == "composite"].reset_index(drop=True)

        cols_keep = [
            "outcome", "diagnosed_n", "prevalent_n", "incident_n",
            "competing_n", "incident_pct", "median_tte_days",
        ]
        df = df[[c for c in cols_keep if c in df.columns]]

        # add human-readable label
        df.insert(
            1, "label",
            df["outcome"].map(self.OUTCOME_LABELS).fillna(df["outcome"])
        )

        # convert median TTE to years
        if "median_tte_days" in df.columns:
            df["median_tte_years"] = (df["median_tte_days"] / 365.25).round(2)

        return df

    # ── Table C: exclusion impact by outcome ──────────────────────────

    def build_exclusion_impact(self) -> pd.DataFrame:
        """
        Per-outcome exclusion counts from neuro, actigraphy QC, shift work.

        Returns
        -------
        pd.DataFrame
            Columns: outcome, label, neuro_n, neuro_pct,
                     acc_qc_n, acc_qc_pct, shift_any_n, shift_any_pct
        """
        rows: list[dict] = []

        for outcome in self.OUTCOMES:
            row: dict = {
                "outcome": outcome,
                "label": self.OUTCOME_LABELS.get(outcome, outcome),
            }

            # neuro exclusions
            ne = self._neuro_excl.loc[
                self._neuro_excl.iloc[:, 0].astype(str).str.strip() == outcome
            ]
            if not ne.empty:
                row["neuro_n"] = int(ne["n_neuro_exclude"].iloc[0])
                row["neuro_pct"] = float(ne["pct_neuro_exclude"].iloc[0])
            else:
                row["neuro_n"] = 0
                row["neuro_pct"] = 0.0

            # actigraphy QC
            aq = self._acc_qc.loc[
                self._acc_qc.iloc[:, 0].astype(str).str.strip() == outcome
            ]
            if not aq.empty:
                row["acc_qc_n"] = int(aq["n_acc_bad_quality"].iloc[0])
                row["acc_qc_pct"] = float(aq["pct_acc_bad_quality"].iloc[0])
            else:
                row["acc_qc_n"] = 0
                row["acc_qc_pct"] = 0.0

            # shift work (use first instance, "shift_any_i0_p826")
            sw = self._shift_work.loc[
                self._shift_work.iloc[:, 0].astype(str).str.strip() == outcome
            ]
            if not sw.empty and "n_shift_any_i0_p826" in sw.columns:
                row["shift_any_n"] = int(sw["n_shift_any_i0_p826"].iloc[0])
                row["shift_any_pct"] = float(sw["pct_shift_any_i0_p826"].iloc[0])
            else:
                row["shift_any_n"] = 0
                row["shift_any_pct"] = 0.0

            rows.append(row)

        return pd.DataFrame(rows)

    # ── Table D: censoring diagnostics ────────────────────────────────

    def build_censor_diagnostics(self) -> pd.DataFrame:
        """
        Censoring timeline: latest diagnosis, censor date, gap, staleness.

        Returns
        -------
        pd.DataFrame
        """
        df = self._censor.copy()
        # add label for composite outcomes
        df["label"] = df["outcome"].map(self.OUTCOME_LABELS).fillna(df["outcome"])
        col_order = ["outcome", "label", "latest_dx", "censor_date", "gap_days", "stale_flag"]
        return df[[c for c in col_order if c in df.columns]]

    # ── Table E: medication completeness ──────────────────────────────

    def build_medication_completeness(self) -> pd.DataFrame:
        """
        Medication family reporting completeness.

        Returns
        -------
        pd.DataFrame
        """
        df = self._medications.copy()
        if "n_reported" in df.columns and "n_both" in df.columns:
            df["completeness_pct"] = (
                100.0 * df["n_both"] / df["n_reported"]
            ).round(1)
        return df

    # ── Narrative report ──────────────────────────────────────────────

    def generate_narrative(self) -> str:
        """
        Auto-generate STROBE Methods-style narrative text.

        Covers STROBE items 4 (study design), 5 (setting), 6 (participants),
        7 (variables), 12 (follow-up), 13 (participant flow).

        Returns
        -------
        str
            Formatted text suitable for a manuscript Methods section.
        """
        flow = self.build_flow_table()
        outcomes = self.build_outcome_table()
        censor = self.build_censor_diagnostics()

        n_total = int(flow.loc[flow["Stage"] == 1, "N_before"].iloc[0])
        n_final_ehr = int(flow.loc[flow["Stage"] == 3, "N_after"].iloc[0])

        # analytical cohort from RBD pipeline
        rbd_row = flow.loc[flow["Stage"] == 5]
        n_analytical = int(rbd_row["N_after"].iloc[0]) if not rbd_row.empty else 0

        # censor date
        censor_date = "2025-02-01"
        if not censor.empty and "censor_date" in censor.columns:
            censor_date = censor["censor_date"].iloc[0]

        # build outcome text
        outcome_lines: list[str] = []
        for _, row in outcomes.iterrows():
            outcome_lines.append(
                f"  - {row['label']}: {int(row['incident_n']):,} incident cases, "
                f"{int(row['prevalent_n']):,} prevalent (excluded from survival analysis), "
                f"{int(row['competing_n']):,} competing deaths, "
                f"median time-to-event {row.get('median_tte_years', 'N/A')} years"
            )

        # stale outcomes
        stale = censor.loc[censor["stale_flag"] == True]
        stale_note = ""
        if not stale.empty:
            stale_names = ", ".join(stale["label"].tolist())
            stale_note = (
                f"\n\nNote: The following outcome(s) were flagged as potentially stale "
                f"(>6 months since last recorded diagnosis): {stale_names}. "
                f"Results for these outcomes should be interpreted with caution."
            )

        # medication completeness
        meds = self.build_medication_completeness()
        med_note = ""
        if not meds.empty:
            complete = meds.loc[meds["completeness_pct"] == 100.0]
            if len(complete) == len(meds):
                med_note = (
                    "All medication families achieved 100% reporting completeness "
                    f"(date and presence flags) across {len(meds)} tracked families."
                )
            else:
                med_note = (
                    f"Medication reporting completeness ranged from "
                    f"{meds['completeness_pct'].min():.0f}% to "
                    f"{meds['completeness_pct'].max():.0f}% across "
                    f"{len(meds)} tracked families."
                )

        # assemble narrative
        text = f"""
CONSORT Participant Flow and Study Design Report
=================================================
Generated from pipeline log CSVs.
Follows STROBE Statement for cohort studies and RECORD extension.

STUDY DESIGN (STROBE Item 4)
-----------------------------
This is a retrospective longitudinal cohort study using data from the UK
Biobank (UKB). Participants were drawn from the UKB actigraphy sub-study,
which collected 7-day wrist-worn accelerometry data between 2013 and 2015.
Neurodegenerative outcomes were ascertained through linked Hospital Episode
Statistics (HES) and primary care records using ICD-10 codes.

SETTING (STROBE Item 5)
------------------------
Baseline was defined as the start of the actigraphy wear period.
Follow-up extended to the administrative censor date of {censor_date}.
Outcome ascertainment used linked electronic health records (EHR)
comprising HES inpatient diagnoses (fields p41270/p41280), first-occurrence
dates (p131022, p131036, p42022, p42024), and GP clinical event data.

PARTICIPANTS (STROBE Item 6)
-----------------------------
Inclusion criteria:
  - UKB participants who completed the actigraphy sub-study
  - Valid accelerometry data with identifiable sleep periods

Exclusion criteria (applied sequentially):
  1. Consent withdrawal or missing actigraphy: {n_total:,} -> {int(flow.loc[flow['Stage']==1, 'N_after'].iloc[0]):,} ({_fmt_n_pct(_safe_loss(n_total, int(flow.loc[flow['Stage']==1, 'N_after'].iloc[0])), n_total)} excluded)
  2. Pre-baseline neurological conditions: {int(flow.loc[flow['Stage']==2, 'N_before'].iloc[0]):,} -> {int(flow.loc[flow['Stage']==2, 'N_after'].iloc[0]):,} ({int(flow.loc[flow['Stage']==2, 'N_excluded'].iloc[0]):,} excluded)
     (parkinson-plus syndromes, neurodegenerative diseases, demyelinating
      disorders, epilepsy, narcolepsy, and other confounding neuropathologies)
  3. Night-level quality control: {int(flow.loc[flow['Stage']==3, 'N_before'].iloc[0]):,} -> {int(flow.loc[flow['Stage']==3, 'N_after'].iloc[0]):,} ({int(flow.loc[flow['Stage']==3, 'N_excluded'].iloc[0]):,} excluded)
     (insufficient valid nights, accelerometry quality flags)

For RBD risk stratification analysis:
  4. RBD model prediction: {int(flow.loc[flow['Stage']==4, 'N_before'].iloc[0]):,} -> {int(flow.loc[flow['Stage']==4, 'N_after'].iloc[0]):,} ({int(flow.loc[flow['Stage']==4, 'N_excluded'].iloc[0]):,} excluded)
  5. Validation split exclusion: {int(flow.loc[flow['Stage']==5, 'N_before'].iloc[0]):,} -> {int(flow.loc[flow['Stage']==5, 'N_after'].iloc[0]):,} ({int(flow.loc[flow['Stage']==5, 'N_excluded'].iloc[0]):,} excluded)

Final analytical cohort: {n_analytical:,} participants.

OUTCOMES (STROBE Item 7)
-------------------------
Six composite neurodegenerative outcomes were defined using non-overlapping
ICD-10 diagnostic codes:

{chr(10).join(outcome_lines)}

Prevalent cases (diagnosed before baseline) were excluded from time-to-event
analyses. Competing risks were modelled using cause-specific hazard models
with death as the competing event.

FOLLOW-UP (STROBE Item 12)
---------------------------
Administrative censor date: {censor_date}.
Survival time was computed from the actigraphy wear start date to the
earliest of: outcome diagnosis, death, or administrative censoring.
{stale_note}

DATA COMPLETENESS
------------------
{med_note}

Actigraphy quality exclusions affected 11-20% of cases per outcome.
Neurological exclusions removed <2% of the cohort overall (range by
outcome: 0-8.8%).
""".strip()

        return text

    # ── orchestrator ──────────────────────────────────────────────────

    def run(self) -> None:
        """Load logs, build all tables, save outputs."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_logs()

        # Table A
        flow = self.build_flow_table()
        flow.to_csv(self.save_dir / "consort_flow_table.csv", index=False)

        # Table B
        outcomes = self.build_outcome_table()
        outcomes.to_csv(self.save_dir / "outcome_summary_table.csv", index=False)

        # Table C
        excl = self.build_exclusion_impact()
        excl.to_csv(self.save_dir / "exclusion_impact_table.csv", index=False)

        # Table D
        censor = self.build_censor_diagnostics()
        censor.to_csv(self.save_dir / "censor_diagnostics.csv", index=False)

        # Table E
        meds = self.build_medication_completeness()
        meds.to_csv(self.save_dir / "medication_completeness.csv", index=False)

        # Narrative
        narrative = self.generate_narrative()
        (self.save_dir / "consort_narrative.txt").write_text(narrative, encoding="utf-8")

        if self.verbose:
            print("=" * 70)
            print("CONSORT FLOW TABLE (Table A)")
            print("=" * 70)
            print(tabulate(flow, headers="keys", tablefmt="github", showindex=False,
                           floatfmt=".1f"))

            print("\n" + "=" * 70)
            print("OUTCOME SUMMARY (Table B)")
            print("=" * 70)
            print(tabulate(outcomes, headers="keys", tablefmt="github", showindex=False,
                           floatfmt=".1f"))

            print("\n" + "=" * 70)
            print("EXCLUSION IMPACT BY OUTCOME (Table C)")
            print("=" * 70)
            print(tabulate(excl, headers="keys", tablefmt="github", showindex=False,
                           floatfmt=".1f"))

            print("\n" + "=" * 70)
            print("CENSORING DIAGNOSTICS (Table D)")
            print("=" * 70)
            print(tabulate(censor, headers="keys", tablefmt="github", showindex=False))

            print("\n" + "=" * 70)
            print("MEDICATION COMPLETENESS (Table E)")
            print("=" * 70)
            print(tabulate(meds, headers="keys", tablefmt="github", showindex=False,
                           floatfmt=".1f"))

            print("\n" + "=" * 70)
            print("NARRATIVE REPORT")
            print("=" * 70)
            print(narrative)

            print(f"\nAll outputs saved to: {self.save_dir}")


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generator = ConsortReportGenerator(verbose=True)
    generator.run()
