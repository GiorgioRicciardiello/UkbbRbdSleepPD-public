# Delta Analysis: Cognitive & Prodromal by RBD Group

## Overview

This analysis suite examines whether cognitive scores decline and prodromal symptoms accumulate over time (~10 years), and whether RBD risk groups stratify these trajectories in controls (non-converters).

**Research Question:**
> Do cognitive scores decline AND prodromal burden increase over 10 years (i0 → i2)? Does RBD risk group stratify both trajectories?

**Population:** Controls only (N=93,254 for prodromal; N=11,990–34,272 for cognitive pairs)

---

## Running the Analysis

### Quick Start
```bash
cd notebook/
python run_delta_analysis.py
```

This runs both cognitive and prodromal analyses sequentially.

### Individual Analyses

**Cognitive Delta Only:**
```bash
python analyze_cognitive_delta_by_rbd.py
```

**Prodromal Delta Only:**
```bash
python analyze_prodromal_delta_by_rbd.py
```

---

## What Each Script Does

### `analyze_cognitive_delta_by_rbd.py`

**Input:** Controls with cognitive measurements at baseline (i0) and follow-up (i2)

**Pipeline:**
1. Load controls from parquet
2. **Compute deltas:** i2_score - i0_score for 5 cognitive variables
3. **Generate audit tables:**
   - `audit_cognitive_delta_overall.csv` — Coverage and distribution per variable
   - `audit_cognitive_delta_by_rbd_group.csv` — Median delta by RBD group
4. **Residualize:** Remove age + sex effects from delta scores
5. **Kruskal-Wallis:** Test if RBD group stratifies residualized delta (per variable)
6. **Dunn post-hoc:** Pairwise comparisons (Bonferroni-corrected)
7. **FDR correction:** Benjamini-Hochberg across 5 variables
8. **Generate results:**
   - `results_cognitive_delta_kruskal_wallis.csv` — KW H, p, epsilon², FDR-p
   - `results_cognitive_delta_pairwise.csv` — Pairwise z-scores and p-values
9. **Generate figures:** Violin+box plots per variable (saved to `figures/`)

**Variables Analyzed:**
- Reaction Time (N=34,272 pairs)
- Fluid Intelligence (N=11,990 pairs)
- TMT-A Duration (N=17,392 pairs)
- TMT-B Duration (N=17,136 pairs)
- Log TMT-B/A Ratio (N=17,040 pairs)

**Expected Outcome:**
- High-RBD shows MORE NEGATIVE delta (steeper decline)
- If p < 0.05 (FDR-corrected): RBD stratifies cognitive decline

---

### `analyze_prodromal_delta_by_rbd.py`

**Input:** All controls (prodromal data complete for all)

**Pipeline:**
1. Load controls from parquet
2. **Compute burden:**
   - Baseline burden: count of prodromal markers at i0
   - Incident burden: count of NEW markers acquired post-baseline
   - Delta = incident burden (new markers over ~10 years)
3. **Generate audit tables:**
   - `audit_prodromal_burden_by_rbd_group.csv` — Burden by RBD group
   - `audit_prodromal_markers_overall.csv` — Prevalence of each marker
   - `audit_prodromal_markers_by_rbd_group.csv` — Marker prevalence by RBD group
4. **Residualize:** Remove age + sex effects from burden delta
5. **Kruskal-Wallis:** Test if RBD group stratifies residualized burden delta
6. **Chi-square:** Individual markers (incident vs. RBD group)
7. **FDR correction:** Benjamini-Hochberg across burden + 8 markers
8. **Generate results:**
   - `results_prodromal_burden_kruskal_wallis.csv` — KW test on burden delta
   - `results_prodromal_markers_chisquare.csv` — Chi-square results per marker
9. **Generate figures:** Bar chart of incident marker % by RBD group

**Markers Analyzed (8 total):**
- Constipation (3.28% incident overall, 5.55% in High-RBD)
- Depression (1.30% incident overall, 2.01% in High-RBD)
- Anxiety (0.06% incident — sparse)
- Orthostatic hypotension (0.74% incident, 2.24% in High-RBD)
- Erectile dysfunction (0.17% incident — sparse)
- Dream enactment (0.00% — absent)
- Anosmia (0.01% incident — sparse)
- Hyposmia (0.00% — absent)

**Expected Outcome:**
- High-RBD acquires MORE new prodromal markers
- If p < 0.05 (FDR-corrected): RBD stratifies prodromal accumulation
- Sparse markers unlikely to show significant stratification

---

## Output Structure

```
results/
├── cognitive_delta_analysis/
│   ├── audit_cognitive_delta_overall.csv
│   ├── audit_cognitive_delta_by_rbd_group.csv
│   ├── results_cognitive_delta_kruskal_wallis.csv
│   ├── results_cognitive_delta_pairwise.csv
│   └── figures/
│       ├── delta_reaction_time_(ms).png
│       ├── delta_fluid_intelligence.png
│       ├── delta_tmt-a_duration_(s).png
│       ├── delta_tmt-b_duration_(s).png
│       └── delta_log_tmt-b/a_ratio.png
│
└── prodromal_delta_analysis/
    ├── audit_prodromal_burden_by_rbd_group.csv
    ├── audit_prodromal_markers_overall.csv
    ├── audit_prodromal_markers_by_rbd_group.csv
    ├── results_prodromal_burden_kruskal_wallis.csv
    ├── results_prodromal_markers_chisquare.csv
    └── figures/
        └── prodromal_markers_by_rbd.png
```

---

## Including Audit Tables in Report

**Best Practice:** Include audit tables as supplementary material to demonstrate:
1. **Data coverage:** How many subjects had both i0 and i2 measurements?
2. **Data quality:** How many outliers? What's the distribution?
3. **RBD group composition:** Are sample sizes balanced?
4. **Baseline characteristics:** Median burden at baseline by RBD group

**Example sections:**

### Supplementary Table S1: Cognitive Delta — Audit
[Include: `audit_cognitive_delta_overall.csv` + `audit_cognitive_delta_by_rbd_group.csv`]

Shows N per variable, median delta, follow-up time consistency, and outlier rates.

### Supplementary Table S2: Prodromal Delta — Audit
[Include: `audit_prodromal_burden_by_rbd_group.csv` + `audit_prodromal_markers_overall.csv`]

Shows baseline vs. incident burden, % acquiring new markers by RBD group.

### Main Results Table 1: Cognitive Delta — Kruskal-Wallis
[Include: `results_cognitive_delta_kruskal_wallis.csv`]

Shows H-statistic, p-value, FDR-corrected p, epsilon-squared (effect size).

### Main Results Table 2: Prodromal Delta — Chi-Square
[Include: `results_prodromal_markers_chisquare.csv`]

Shows chi-square statistic, p-value, FDR-corrected p, % with incident marker by RBD group.

---

## Interpreting Results

### Best-Case Scenario
- **Cognitive:** Multiple variables show FDR-p < 0.05, with High-RBD declining faster
- **Prodromal:** Burden delta shows KW p < 0.05, constipation/depression/orthostatic show chi-square p < 0.05

→ **Interpretation:** "RBD risk comprehensively stratifies both cognitive decline and prodromal progression in non-converters."

### Mixed Scenario
- **Cognitive:** RT shows FDR-p < 0.05, FI does not
- **Prodromal:** Burden delta significant, but only 1–2 markers

→ **Interpretation:** "RBD stratifies processing speed decline and constipation accumulation, but effects are variable across domains."

### Null Scenario
- **Cognitive:** All FDR-p > 0.05
- **Prodromal:** All chi-square p > 0.05

→ **Interpretation:** "Cognitive decline and prodromal accumulation over 10 years are not stratified by RBD risk in non-converters, despite RBD's strong prognostic value for incident PD."

---

## Notes

1. **TMT Paradigm Mismatch:** Baseline TMT is online (self-administered), follow-up is clinic (paper-pencil). Interpret TMT results cautiously.
2. **Sparse Markers:** Anxiety, ED, anosmia, hyposmia have very low incident rates (<0.2%); unlikely to reach FDR significance.
3. **Multiple Testing:** FDR correction applied across all tests (5 cognitive variables + 1 burden + 8 markers = 14 total). This is conservative.
4. **Follow-up Time:** Median ~11 years, range 10–13 years. Consistent across RBD groups.

---

## References

- **Kruskal-Wallis:** Non-parametric ANOVA-equivalent test for group differences
- **Dunn test:** Post-hoc pairwise comparisons after Kruskal-Wallis
- **Epsilon-squared:** Effect size for Kruskal-Wallis (analogous to eta-squared)
- **Chi-square:** Test of association between categorical variables (incident marker × RBD group)
- **Benjamini-Hochberg FDR:** Controls false discovery rate at alpha=0.05 across multiple tests
