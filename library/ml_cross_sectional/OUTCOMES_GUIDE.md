# Outcomes Configuration Guide

## Overview

Outcome definitions are **centralized and transparent** in `outcomes.py`. Each outcome specifies:
- The binary classification target column
- The associated event date column (for time-to-event analysis)
- A human-readable description

This makes outcomes easy to modify, swap, and document.

---

## Available Outcomes

| Outcome Name | Column | Description |
|--------------|--------|-------------|
| `pd_only` | `outcome_1a_pd_only` | PD only (no AD or DLB) |
| `pd_ad` | `outcome_1b_pd_ad` | PD or AD |
| `ad_only` | `outcome_4a_ad_only` | AD only |
| `other_dementia` | `outcome_2a_otherdementia` | Other dementia (excl. PD, AD, DLB) |
| `pd_other_dementia` | `outcome_2b_pd_otherdementia` | PD or other dementia (excl. AD, DLB) |
| `dlb_only` | `outcome_3a_dlb_only` | DLB only |

---

## How to Change the Outcome

### 1. **Change the Default (in `outcomes.py`)**

```python
DEFAULT_OUTCOME: Final[str] = "pd_ad"  # Switch from "pd_only" to "pd_ad"
```

### 2. **Specify Outcome in RunConfig (in `pipeline.py` or `run_ml_cross_sectional.py`)**

```python
cfg = RunConfig(
    outcome_name="ad_only",  # Use AD instead of PD
    n_outer=5,
    n_inner=3,
    n_trials=50,
)
```

### 3. **Specify Outcome When Loading Data**

```python
from library.ml_cross_sectional.pipeline import build_xy, load_data

df = load_data("ehr_diag_pd_rbd_only_all")
X, y = build_xy(df, outcome_name="pd_ad")  # Explicit outcome
```

---

## How to Add a New Outcome

Edit `outcomes.py` and add a new entry to `OUTCOMES`:

```python
OUTCOMES: Final[dict[str, OutcomeConfig]] = {
    # ... existing outcomes ...
    "custom_outcome": OutcomeConfig(
        outcome_col="my_custom_outcome_column",
        event_date_col="my_custom_outcome_date",
        description="My custom disease outcome definition",
    ),
}
```

Then use it:

```python
X, y = build_xy(df, outcome_name="custom_outcome")
```

---

## Data Flow

```
outcomes.py (OUTCOMES registry)
    ↓
RunConfig(outcome_name="pd_only")
    ↓
build_xy(df, outcome_name="pd_only")
    ↓
convert_to_cross_sectional(df, outcome_name="pd_only")
    ↓
get_outcome_config("pd_only") → OutcomeConfig
    ↓
Extract columns: outcome_col, event_date_col
    ↓
Cross-sectional dataframe with selected outcome
```

---

## Key Functions

### List Available Outcomes

```python
from library.ml_cross_sectional.outcomes import list_outcomes

outcomes_dict = list_outcomes()
for name, description in outcomes_dict.items():
    print(f"{name:20s} → {description}")
```

**Output:**
```
pd_only              → PD (Parkinson's Disease) only — diagnosed with PD, no AD or DLB
pd_ad                → PD or AD — diagnosed with either Parkinson's or Alzheimer's disease
ad_only              → AD only — diagnosed with Alzheimer's disease only
...
```

### Get Outcome Configuration

```python
from library.ml_cross_sectional.outcomes import get_outcome_config

cfg = get_outcome_config("pd_ad")
print(f"Outcome column: {cfg.outcome_col}")
print(f"Date column: {cfg.event_date_col}")
print(f"Description: {cfg.description}")
```

---

## Example: Running Multiple Outcomes

```python
from library.ml_cross_sectional.pipeline import RunConfig, run_all_feature_sets, load_data, build_xy

outcomes_to_test = ["pd_only", "pd_ad", "ad_only"]

for outcome_name in outcomes_to_test:
    print(f"\n{'='*60}")
    print(f"Running models for outcome: {outcome_name}")
    print('='*60)
    
    cfg = RunConfig(
        outcome_name=outcome_name,
        n_outer=5,
        n_inner=3,
    )
    
    df = load_data(cfg.file_name)
    X, y = build_xy(df, outcome_name=outcome_name)
    
    results = run_all_feature_sets(X, y, cfg)
    print(f"Completed {outcome_name}: {len(results)} models trained")
```

---

## Benefits of Explicit Outcome Definition

✅ **Transparency** — Exact column names and descriptions visible  
✅ **Reproducibility** — Same outcome name always uses same columns  
✅ **Maintainability** — Add/modify outcomes in one place  
✅ **Consistency** — All models use the same outcome definition  
✅ **Alignment** — Matches the screening model's explicit feature approach  

---

## Technical Notes

- Outcome names are **case-sensitive** (e.g., `"pd_only"`, not `"PD_Only"`)
- Event date columns are required for time-to-event analysis
- Missing outcome values are dropped during preprocessing
- All outcomes are binary (0 = control/negative, 1 = case/positive)
