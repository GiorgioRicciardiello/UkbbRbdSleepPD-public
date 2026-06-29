# Trail Making Test Columns

## Stage 1 — `trail_making_covariates()` output (24 columns, no temporal filter)

Each column follows the pattern `{metric}_{source}_{instance}` with `source ∈ {online, clinic}` and `instance ∈ {i0, i1, i2, i3}`.

| Column | UKB field(s) | Definition | What it captures |
|---|---|---|---|
| `tmt1_dur_online_i0` | p20156\_i0 | Trail-1 duration (seconds), online, instance 0 (2014–2015) | Processing speed / psychomotor speed — time to connect numbers 1→25 in sequence |
| `tmt1_dur_online_i1` | p20156\_i1 | Trail-1 duration (seconds), online, instance 1 (2021–2023) | Same as above, later assessment |
| `tmt1_dur_clinic_i2` | p6348\_i2 ÷ 10 | Trail-1 duration (seconds), in-clinic, instance 2 (2014–2020) | Same; unit-converted from deciseconds |
| `tmt1_dur_clinic_i3` | p6348\_i3 ÷ 10 | Trail-1 duration (seconds), in-clinic, instance 3 (2019+) | Same; unit-converted from deciseconds |
| `tmt2_dur_online_i0` | p20157\_i0 | Trail-2 duration (seconds), online, i0 | Cognitive flexibility + processing speed — time to connect 1→A→2→B…→13→L |
| `tmt2_dur_online_i1` | p20157\_i1 | Trail-2 duration (seconds), online, i1 | Same, later assessment |
| `tmt2_dur_clinic_i2` | p6350\_i2 ÷ 10 | Trail-2 duration (seconds), in-clinic, i2 | Same; unit-converted |
| `tmt2_dur_clinic_i3` | p6350\_i3 ÷ 10 | Trail-2 duration (seconds), in-clinic, i3 | Same; unit-converted |
| `tmt1_err_online_i0` | p20247\_i0 | Trail-1 errors, online, i0 | Attentional lapses on the numeric path |
| `tmt1_err_online_i1` | p20247\_i1 | Trail-1 errors, online, i1 | — |
| `tmt1_err_clinic_i2` | p6349\_i2 | Trail-1 errors, in-clinic, i2 | — |
| `tmt1_err_clinic_i3` | p6349\_i3 | Trail-1 errors, in-clinic, i3 | — |
| `tmt2_err_online_i0` | p20248\_i0 | Trail-2 errors, online, i0 | Errors on set-shifting task (more sensitive to executive dysfunction) |
| `tmt2_err_online_i1` | p20248\_i1 | Trail-2 errors, online, i1 | — |
| `tmt2_err_clinic_i2` | p6351\_i2 | Trail-2 errors, in-clinic, i2 | — |
| `tmt2_err_clinic_i3` | p6351\_i3 | Trail-2 errors, in-clinic, i3 | — |
| `tmt_ratio_online_i0` | p20157/p20156, i0 | TMT-B/A ratio, online, i0 | **Primary PD prodromal marker.** Scale-invariant index of executive function / set-shifting, independent of raw motor speed. Ratio ≥ 1.0 enforced; NaN if either duration is invalid |
| `tmt_ratio_online_i1` | p20157/p20156, i1 | TMT-B/A ratio, online, i1 | Same |
| `tmt_ratio_clinic_i2` | p6350/p6348, i2 | TMT-B/A ratio, in-clinic, i2 | Same |
| `tmt_ratio_clinic_i3` | p6350/p6348, i3 | TMT-B/A ratio, in-clinic, i3 | Same |
| `tmt_date_online_i0` | p20136\_i0 | Date of online assessment, i0 | Used by Stage 2 for temporal alignment to `wear_time_start` |
| `tmt_date_online_i1` | p20136\_i1 | Date of online assessment, i1 | — |
| `tmt_date_clinic_i2` | follow-up date i2 | Date of in-clinic assessment, i2 | — |
| `tmt_date_clinic_i3` | follow-up date i3 | Date of in-clinic assessment, i3 | — |

---

## Stage 2 — `select_tmt_baseline()` output (8 columns, temporal filter ±730 days of `wear_time_start`)

| Column | Definition | What it captures |
|---|---|---|
| `tmt1_dur_baseline` | Trail-1 duration (s) of selected instance | Motor/processing speed covariate for Cox model |
| `tmt2_dur_baseline` | Trail-2 duration (s) of selected instance | Executive + motor speed for Cox model |
| `tmt1_err_baseline` | Trail-1 errors of selected instance | Attentional accuracy, numeric path |
| `tmt2_err_baseline` | Trail-2 errors of selected instance | Set-shifting accuracy |
| `tmt_ratio_baseline` | TMT-B/A ratio of selected instance | **Cox covariate** — executive function marker; clinic preferred over online, minimum-lag tiebreak |
| `tmt_lag_days` | Days between `wear_time_start` and TMT assessment | Audit column — temporal proximity to actigraphy baseline |
| `tmt_source_baseline` | `"clinic_i2"`, `"clinic_i3"`, `"online_i0"`, `"online_i1"`, or NaN | Traceability — which instance was selected and why |
| `tmt_missing` | `True` if no valid baseline found within window | Missingness flag for sensitivity analyses |

---

## Python Dictionary

```python
TMT_COLUMNS: dict[str, dict] = {
    # ── Stage 1: Trail-1 duration ──────────────────────────────────────────
    "tmt1_dur_online_i0": {
        "ukb_field": "p20156_i0",
        "unit": "seconds",
        "source": "online",
        "instance": "i0",
        "window": "2014-2015",
        "definition": "Trail-1 duration, online instance 0",
        "captures": "Psychomotor speed — numeric path (TMT-A)",
    },
    "tmt1_dur_online_i1": {
        "ukb_field": "p20156_i1",
        "unit": "seconds",
        "source": "online",
        "instance": "i1",
        "window": "2021-2023",
        "definition": "Trail-1 duration, online instance 1",
        "captures": "Psychomotor speed — numeric path (TMT-A), later assessment",
    },
    "tmt1_dur_clinic_i2": {
        "ukb_field": "p6348_i2 / 10",
        "unit": "seconds (converted from deciseconds)",
        "source": "clinic",
        "instance": "i2",
        "window": "2014-2020",
        "definition": "Trail-1 duration, in-clinic instance 2",
        "captures": "Psychomotor speed — numeric path (TMT-A)",
    },
    "tmt1_dur_clinic_i3": {
        "ukb_field": "p6348_i3 / 10",
        "unit": "seconds (converted from deciseconds)",
        "source": "clinic",
        "instance": "i3",
        "window": "2019+",
        "definition": "Trail-1 duration, in-clinic instance 3",
        "captures": "Psychomotor speed — numeric path (TMT-A)",
    },
    # ── Stage 1: Trail-2 duration ──────────────────────────────────────────
    "tmt2_dur_online_i0": {
        "ukb_field": "p20157_i0",
        "unit": "seconds",
        "source": "online",
        "instance": "i0",
        "window": "2014-2015",
        "definition": "Trail-2 duration, online instance 0",
        "captures": "Cognitive flexibility + processing speed — alphanumeric path (TMT-B)",
    },
    "tmt2_dur_online_i1": {
        "ukb_field": "p20157_i1",
        "unit": "seconds",
        "source": "online",
        "instance": "i1",
        "window": "2021-2023",
        "definition": "Trail-2 duration, online instance 1",
        "captures": "Cognitive flexibility + processing speed — alphanumeric path (TMT-B)",
    },
    "tmt2_dur_clinic_i2": {
        "ukb_field": "p6350_i2 / 10",
        "unit": "seconds (converted from deciseconds)",
        "source": "clinic",
        "instance": "i2",
        "window": "2014-2020",
        "definition": "Trail-2 duration, in-clinic instance 2",
        "captures": "Cognitive flexibility + processing speed — alphanumeric path (TMT-B)",
    },
    "tmt2_dur_clinic_i3": {
        "ukb_field": "p6350_i3 / 10",
        "unit": "seconds (converted from deciseconds)",
        "source": "clinic",
        "instance": "i3",
        "window": "2019+",
        "definition": "Trail-2 duration, in-clinic instance 3",
        "captures": "Cognitive flexibility + processing speed — alphanumeric path (TMT-B)",
    },
    # ── Stage 1: Trail-1 errors ────────────────────────────────────────────
    "tmt1_err_online_i0": {
        "ukb_field": "p20247_i0",
        "unit": "count",
        "source": "online",
        "instance": "i0",
        "definition": "Trail-1 errors, online instance 0",
        "captures": "Attentional lapses on the numeric path",
    },
    "tmt1_err_online_i1": {
        "ukb_field": "p20247_i1",
        "unit": "count",
        "source": "online",
        "instance": "i1",
        "definition": "Trail-1 errors, online instance 1",
        "captures": "Attentional lapses on the numeric path",
    },
    "tmt1_err_clinic_i2": {
        "ukb_field": "p6349_i2",
        "unit": "count",
        "source": "clinic",
        "instance": "i2",
        "definition": "Trail-1 errors, in-clinic instance 2",
        "captures": "Attentional lapses on the numeric path",
    },
    "tmt1_err_clinic_i3": {
        "ukb_field": "p6349_i3",
        "unit": "count",
        "source": "clinic",
        "instance": "i3",
        "definition": "Trail-1 errors, in-clinic instance 3",
        "captures": "Attentional lapses on the numeric path",
    },
    # ── Stage 1: Trail-2 errors ────────────────────────────────────────────
    "tmt2_err_online_i0": {
        "ukb_field": "p20248_i0",
        "unit": "count",
        "source": "online",
        "instance": "i0",
        "definition": "Trail-2 errors, online instance 0",
        "captures": "Set-shifting errors — more sensitive to executive dysfunction",
    },
    "tmt2_err_online_i1": {
        "ukb_field": "p20248_i1",
        "unit": "count",
        "source": "online",
        "instance": "i1",
        "definition": "Trail-2 errors, online instance 1",
        "captures": "Set-shifting errors — more sensitive to executive dysfunction",
    },
    "tmt2_err_clinic_i2": {
        "ukb_field": "p6351_i2",
        "unit": "count",
        "source": "clinic",
        "instance": "i2",
        "definition": "Trail-2 errors, in-clinic instance 2",
        "captures": "Set-shifting errors — more sensitive to executive dysfunction",
    },
    "tmt2_err_clinic_i3": {
        "ukb_field": "p6351_i3",
        "unit": "count",
        "source": "clinic",
        "instance": "i3",
        "definition": "Trail-2 errors, in-clinic instance 3",
        "captures": "Set-shifting errors — more sensitive to executive dysfunction",
    },
    # ── Stage 1: TMT-B/A ratio ─────────────────────────────────────────────
    "tmt_ratio_online_i0": {
        "ukb_field": "p20157_i0 / p20156_i0",
        "unit": "dimensionless (ratio >= 1.0)",
        "source": "online",
        "instance": "i0",
        "definition": "TMT-B/A duration ratio, online instance 0",
        "captures": "Primary PD prodromal marker — executive function / set-shifting, scale-invariant across administration modes",
    },
    "tmt_ratio_online_i1": {
        "ukb_field": "p20157_i1 / p20156_i1",
        "unit": "dimensionless (ratio >= 1.0)",
        "source": "online",
        "instance": "i1",
        "definition": "TMT-B/A duration ratio, online instance 1",
        "captures": "Primary PD prodromal marker — executive function / set-shifting",
    },
    "tmt_ratio_clinic_i2": {
        "ukb_field": "p6350_i2 / p6348_i2",
        "unit": "dimensionless (ratio >= 1.0)",
        "source": "clinic",
        "instance": "i2",
        "definition": "TMT-B/A duration ratio, in-clinic instance 2",
        "captures": "Primary PD prodromal marker — executive function / set-shifting",
    },
    "tmt_ratio_clinic_i3": {
        "ukb_field": "p6350_i3 / p6348_i3",
        "unit": "dimensionless (ratio >= 1.0)",
        "source": "clinic",
        "instance": "i3",
        "definition": "TMT-B/A duration ratio, in-clinic instance 3",
        "captures": "Primary PD prodromal marker — executive function / set-shifting",
    },
    # ── Stage 1: Assessment dates ──────────────────────────────────────────
    "tmt_date_online_i0": {
        "ukb_field": "p20136_i0",
        "unit": "date",
        "source": "online",
        "instance": "i0",
        "definition": "Date of online TMT assessment, instance 0",
        "captures": "Temporal anchor for Stage 2 baseline selection",
    },
    "tmt_date_online_i1": {
        "ukb_field": "p20136_i1",
        "unit": "date",
        "source": "online",
        "instance": "i1",
        "definition": "Date of online TMT assessment, instance 1",
        "captures": "Temporal anchor for Stage 2 baseline selection",
    },
    "tmt_date_clinic_i2": {
        "ukb_field": "follow_up_date_i2",
        "unit": "date",
        "source": "clinic",
        "instance": "i2",
        "definition": "Date of in-clinic TMT assessment, instance 2",
        "captures": "Temporal anchor for Stage 2 baseline selection",
    },
    "tmt_date_clinic_i3": {
        "ukb_field": "follow_up_date_i3",
        "unit": "date",
        "source": "clinic",
        "instance": "i3",
        "definition": "Date of in-clinic TMT assessment, instance 3",
        "captures": "Temporal anchor for Stage 2 baseline selection",
    },
    # ── Stage 2: Baseline columns (select_tmt_baseline) ───────────────────
    "tmt1_dur_baseline": {
        "ukb_field": "derived",
        "unit": "seconds",
        "source": "selected instance",
        "definition": "Trail-1 duration of baseline-aligned instance",
        "captures": "Psychomotor speed covariate for Cox model",
    },
    "tmt2_dur_baseline": {
        "ukb_field": "derived",
        "unit": "seconds",
        "source": "selected instance",
        "definition": "Trail-2 duration of baseline-aligned instance",
        "captures": "Executive + motor speed covariate for Cox model",
    },
    "tmt1_err_baseline": {
        "ukb_field": "derived",
        "unit": "count",
        "source": "selected instance",
        "definition": "Trail-1 errors of baseline-aligned instance",
        "captures": "Attentional accuracy on numeric path",
    },
    "tmt2_err_baseline": {
        "ukb_field": "derived",
        "unit": "count",
        "source": "selected instance",
        "definition": "Trail-2 errors of baseline-aligned instance",
        "captures": "Set-shifting accuracy",
    },
    "tmt_ratio_baseline": {
        "ukb_field": "derived",
        "unit": "dimensionless (ratio >= 1.0)",
        "source": "selected instance",
        "definition": "TMT-B/A ratio of baseline-aligned instance",
        "captures": "Primary Cox covariate — executive function; clinic prioritised over online, minimum temporal lag tiebreak",
    },
    "tmt_lag_days": {
        "ukb_field": "derived",
        "unit": "days",
        "source": "derived",
        "definition": "Days between wear_time_start and selected TMT assessment date",
        "captures": "Audit — temporal proximity to actigraphy baseline",
    },
    "tmt_source_baseline": {
        "ukb_field": "derived",
        "unit": "categorical",
        "source": "derived",
        "definition": "Instance identifier of selected baseline measurement",
        "captures": "Traceability — one of: clinic_i2, clinic_i3, online_i0, online_i1, or NaN",
    },
    "tmt_missing": {
        "ukb_field": "derived",
        "unit": "bool",
        "source": "derived",
        "definition": "True if no valid baseline found within ±730 days of wear_time_start",
        "captures": "Missingness flag for sensitivity analyses",
    },
}
```
