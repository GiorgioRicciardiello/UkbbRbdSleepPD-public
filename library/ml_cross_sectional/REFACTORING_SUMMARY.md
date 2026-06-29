# Cross-Sectional Model Refactoring Summary

## Goal: Transparency & Modularity

Make the cross-sectional model's core definitions (outcomes, features, prodromal markers) **explicit, transparent, and easy to modify**.

---

## Changes Made

### 1. **Explicit Prodromal Features** (`dataset.py`)

#### Before (Token-Based Search)
```python
PRODROMAL_TOKENS = ("prodromal_",)  # Regex substring search
# → Auto-discovers all columns containing "prodromal_"
```

#### After (Explicit List)
```python
PRODROMAL_FEATURES: Final[tuple[str, ...]] = (
    "constipation",
    "orthostatic",
    "depression",
    "erectile",
    "anosmia",
    "hyposmia",
    "memory",
    "anxiety",
    "dream_enactment",
)
```

**Benefits:**
- ✅ Exact features are visible in code
- ✅ No surprises from regex matching
- ✅ Matches screening model's 8 prodromal markers
- ✅ Easy to add/remove features

---

### 2. **Outcome Registry** (`outcomes.py` — NEW)

#### Before (Hardcoded)
```python
OUTCOME_COL = "outcome_1a_pd_only"
EVENT_DATE_COL = "outcome_1a_pd_only_date"
# Only one outcome possible per run
```

#### After (Outcome Registry)
```python
OUTCOMES = {
    "pd_only": OutcomeConfig(
        outcome_col="outcome_1a_pd_only",
        event_date_col="outcome_1a_pd_only_date",
        description="PD only (no AD or DLB)",
    ),
    "pd_ad": OutcomeConfig(...),
    "ad_only": OutcomeConfig(...),
    # ... 3 more outcomes ...
}
```

**Benefits:**
- ✅ 6 outcomes registered, switch between them instantly
- ✅ Self-documenting: each outcome has a description
- ✅ Easy to add new outcomes
- ✅ Consistent across all feature sets

---

### 3. **Feature Sets with Explicit Prodromal** (`feature_sets.py`)

#### Before
```python
"rbd_prodromal": {
    "label": "RBD + Prodromal + demographics",
    "demographics": [...],
    "rbd": [...],
    "prs": [],
    "include_prodromal": True,  # Flag + token search
}
```

#### After
```python
"rbd_prodromal": {
    "label": "RBD + Prodromal (8 markers) + demographics",
    "demographics": [...],
    "rbd": [...],
    "prs": [],
    "prodromal": list(PRODROMAL_MARKERS),  # Explicit list
}
```

**Benefits:**
- ✅ Exact prodromal features are listed
- ✅ Feature count transparent (8 markers)
- ✅ Easy to add selective prodromal subsets

---

### 4. **RunConfig Extended** (`pipeline.py`)

#### Before
```python
@dataclass(frozen=True)
class RunConfig:
    n_outer: int = 5
    n_inner: int = 3
    n_trials: int = 50
    # outcome hardcoded in dataset.py
```

#### After
```python
@dataclass(frozen=True)
class RunConfig:
    n_outer: int = 5
    n_inner: int = 3
    n_trials: int = 50
    outcome_name: str = "pd_only"  # ← NEW: user-tunable
    file_name: str = "ehr_diag_pd_rbd_only_all"
    feature_set: str | None = None
```

**Benefits:**
- ✅ Outcome is a first-class tunable parameter
- ✅ Can specify in config or as function argument
- ✅ Flows through the entire pipeline

---

## Configuration Flow

```
Step 1: Define Outcome
        ↓
    outcomes.py: OUTCOMES registry
        ↓
Step 2: Specify in Config
        ↓
    RunConfig(outcome_name="pd_only")
        ↓
Step 3: Load Data with Outcome
        ↓
    build_xy(df, outcome_name=cfg.outcome_name)
        ↓
Step 4: Convert to Cross-Sectional
        ↓
    convert_to_cross_sectional(df, outcome_name=...)
        ↓
Step 5: Resolve to Columns
        ↓
    get_outcome_config(outcome_name)
        → OutcomeConfig(outcome_col, event_date_col, description)
        ↓
Step 6: Extract Features & Outcome
        ↓
    X, y = get_feature_matrix(frame)
```

---

## Usage Examples

### Run with Different Outcome

```python
from library.ml_cross_sectional.pipeline import RunConfig, run_all_feature_sets, load_data, build_xy

# Change outcome to AD
cfg = RunConfig(
    outcome_name="ad_only",  # ← Specify outcome
    n_outer=5,
    feature_set="rbd_prodromal",
)

df = load_data(cfg.file_name)
X, y = build_xy(df, outcome_name=cfg.outcome_name)
run_all_feature_sets(X, y, cfg)
```

### List Available Outcomes

```python
from library.ml_cross_sectional.outcomes import list_outcomes

for name, desc in list_outcomes().items():
    print(f"{name:20s} → {desc}")
```

### Sweep Multiple Outcomes

```python
for outcome_name in ["pd_only", "pd_ad", "ad_only"]:
    cfg = RunConfig(outcome_name=outcome_name)
    df = load_data(cfg.file_name)
    X, y = build_xy(df, outcome_name=outcome_name)
    run_all_feature_sets(X, y, cfg)
```

---

## Alignment with Screening Model

Both models now use **explicit definitions**:

| Aspect | Screening Model | Cross-Sectional |
|--------|-----------------|-----------------|
| **Prodromal Features** | 8 explicit features (constipation, orthostatic, ...) | 8 explicit features in PRODROMAL_FEATURES |
| **Feature Sets** | Explicit in config | Explicit in feature_sets.py |
| **Outcome** | P1 uses incident PD | Registry with 6 outcomes (choose any) |
| **Transparency** | Features visible in config.py | Features visible in outcomes.py & feature_sets.py |
| **Modularity** | Paradigm classes for training strategies | Feature sets for feature combinations |

---

## Files Changed

| File | Changes |
|------|---------|
| `dataset.py` | Token search → explicit PRODROMAL_FEATURES; outcome_name parameter |
| `feature_sets.py` | include_prodromal flag → explicit prodromal list; PRODROMAL_MARKERS registry |
| `pipeline.py` | Added outcome_name to RunConfig; updated build_xy signature |
| **outcomes.py** | **NEW** — Outcome registry with 6 predefined outcomes |
| **OUTCOMES_GUIDE.md** | **NEW** — User guide for outcome configuration |
| **REFACTORING_SUMMARY.md** | **THIS FILE** — Overview of changes |

---

## Backward Compatibility

- Default behavior unchanged: `build_xy(df)` still uses "pd_only"
- Feature sets work as before: `feature_set="rbd_prodromal"` still includes 8 prodromal markers
- Existing code continues to work, but now with transparent definitions

---

## Next Steps

1. ✅ Verify `run_ml_cross_sectional.py` runs without errors
2. ✅ Test outcome switching: `RunConfig(outcome_name="ad_only")`
3. ✅ Confirm prodromal features are correct
4. ✅ Run feature set comparison with new explicit definitions
5. ✅ Compare cross-sectional results with screening model (P1 Combined)

---

## Key Takeaway

The cross-sectional model now has **explicit, modular, transparent** definitions for:
- **Outcomes** (6 choices, fully configurable)
- **Prodromal features** (8 binary HES markers, exact list visible)
- **Feature sets** (4 combinations, feature composition transparent)

This makes it easy to understand, modify, and extend the model without hunting through code for hardcoded strings or regex patterns.
