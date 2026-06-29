# Comprehensive Delta Analysis Plan: Cognitive & Prodromal by RBD Group

## Overall Research Question
**Over 10 years (i0 → i2), do cognitive scores decline AND prodromal burden increase? Does RBD risk group stratify both trajectories?**

Hypothesis: High-RBD controls show:
1. Steeper cognitive decline
2. More incident prodromal markers
3. Both effects combined indicate RBD captures prodromal neurodegeneration progression even in non-converters

---

## PART A: COGNITIVE DELTA ANALYSIS

### Population
Controls with BOTH i0 and i2 cognitive measurements:
- Reaction Time: 34,272 pairs (36.8%)
- Fluid Intelligence: 11,990 pairs (12.9%)
- TMT-A Duration: ~10k–30k pairs
- TMT-B Duration: ~10k–30k pairs
- TMT Ratio (log): ~10k–30k pairs

**5 cognitive variables analyzed**

### Method
1. **Compute delta:** cog_*_fu - cog_*_bl (change over ~10 years)
2. **Residualize delta** on age_at_i0 + sex
3. **Test:** Kruskal-Wallis on residualized delta across RBD groups (Low/Intermediate/High)
4. **Post-hoc:** Dunn pairwise + Bonferroni within each variable
5. **Multiple testing correction:** Benjamini-Hochberg FDR across 5 variables

### Expected Pattern
- High-RBD > Intermediate-RBD > Low-RBD in magnitude of decline (more negative delta)

### Output
- Table: Median delta ± MAD by RBD group, KW p, FDR p, epsilon²
- Table: Pairwise Dunn p-values (Bonferroni-corrected)
- Figure: Violin+box (5 panels) by RBD group

---

## PART B: PRODROMAL DELTA ANALYSIS

### Population & Data Structure

**All controls in final dataset:** N=93,254

**Prodromal markers available:**
- prodromal_constipation_bl, prodromal_constipation_post
- prodromal_depression_bl, prodromal_depression_post
- prodromal_anxiety_bl, prodromal_anxiety_post
- prodromal_orthostatic_bl, prodromal_orthostatic_post
- prodromal_erectile_dysfunction_bl, prodromal_erectile_dysfunction_post
- prodromal_dream_enactment_bl, prodromal_dream_enactment_post
- prodromal_anosmia_bl, prodromal_anosmia_post
- prodromal_hyposmia_bl, prodromal_hyposmia_post

(8 markers total, pre-baseline + post-baseline flags)

### Computation of Prodromal Delta

**Baseline prodromal burden (at i0):**
```
prodromal_burden_bl = sum of 8 binary flags at baseline
Range: 0–8
```

**Incident prodromal burden (post-baseline, i0 → i2):**
```
prodromal_burden_post = sum of 8 binary flags for NEW markers acquired post-baseline
Range: 0–8
(Only counts markers where baseline = 0; excludes pre-existing)
```

**Prodromal delta (change in burden):**
```
delta_prodromal_burden = prodromal_burden_post
(How many NEW prodromal markers developed over ~10 years)
```

### Adjustment

**Residualize delta_prodromal_burden on:**
- age_at_i0 (older subjects may accumulate prodromals faster)
- sex (sex-specific prodromal patterns, e.g., ED in males)

### Statistical Analysis

**Two approaches (both reported):**

#### Approach 1: Categorical (RBD group × Prodromal delta)
- **Test:** Kruskal-Wallis on residualized delta_prodromal_burden across RBD groups
- **Post-hoc:** Dunn pairwise + Bonferroni
- **Effect size:** Epsilon²
- **Question:** Does RBD group stratify how many NEW prodromals develop?

#### Approach 2: Individual marker analysis (RBD group × Incident marker)
- For each of 8 prodromal markers:
  - **Test:** Chi-square or Fisher's exact (binary: marker_post = yes/no by RBD group)
  - **Effect:** Odds ratio of incident marker by RBD group
  - **Multiple testing:** FDR across 8 markers
- **Question:** Which specific prodromals are RBD-stratified?

### Expected Pattern
- **High-RBD:** Higher incident prodromal burden (more new markers over 10 years)
- **Intermediate-RBD:** Intermediate
- **Low-RBD:** Lower incident prodromal burden

---

## PART C: INTEGRATED RESULTS & INTERPRETATION

### Combined Table: Cognitive + Prodromal Delta Summary

| Domain | Variable | N | RBD_Low_median | RBD_Int_median | RBD_High_median | KW_p | FDR_p | Pattern |
|---|---|---|---|---|---|---|---|---|
| **Cognitive** | RT delta (residual) | 34,272 | 0.0 | -0.15 | -0.45 | p₁ | FDR₁ | Monotone ↓ / Mixed / Null |
| | FI delta (residual) | 11,990 | 0.0 | -0.20 | -0.60 | p₂ | FDR₂ | ... |
| | TMT-A delta (residual) | ~15k | ... | ... | ... | p₃ | FDR₃ | ... |
| | TMT-B delta (residual) | ~15k | ... | ... | ... | p₄ | FDR₄ | ... |
| | TMT Ratio delta (residual) | ~15k | ... | ... | ... | p₅ | FDR₅ | ... |
| **Prodromal** | Burden delta (residual) | 93,254 | 0.0 | +0.15 | +0.35 | p₆ | FDR₆ | Monotone ↑ / Mixed / Null |
| | Constipation (incident %) | 93,254 | 2.1% | 2.8% | 3.5% | OR₁ | FDR₇ | ... |
| | Depression (incident %) | 93,254 | 1.2% | 1.5% | 1.9% | OR₂ | FDR₈ | ... |
| | ... (6 more markers) | ... | ... | ... | ... | ... | ... | ... |

### Figure 1: Cognitive Delta by RBD Group
5 violin+box panels (RT, FI, TMT-A, TMT-B, TMT Ratio)

### Figure 2: Prodromal Burden Delta by RBD Group
Single violin+box panel (total incident prodromal count)

### Figure 3: Individual Prodromal Markers by RBD Group
8 bar plots (one per marker, showing % with incident flag by RBD group)

---

## INTERPRETATION FRAMEWORK

### Scenario 1: Both Cognitive AND Prodromal Show RBD Stratification ✅
- Cognitive: High-RBD declines faster
- Prodromal: High-RBD accumulates more markers
- **Interpretation:** "RBD risk comprehensively stratifies both cognitive decline and prodromal progression over 10 years. RBD score captures an integrated prodromal phenotype combining motor sleep (RBD) + cognitive + non-motor (prodromal) domains."

### Scenario 2: Only Cognitive Shows Stratification
- Cognitive: High-RBD declines faster ✓
- Prodromal: No RBD stratification ✗
- **Interpretation:** "RBD risk stratifies cognitive decline but not prodromal accumulation. Suggests RBD is a marker of cognitive aging rather than broader prodromal progression."

### Scenario 3: Only Prodromal Shows Stratification
- Cognitive: No RBD stratification ✗
- Prodromal: High-RBD accumulates more markers ✓
- **Interpretation:** "RBD risk stratifies prodromal non-motor symptom progression but not cognitive decline. Suggests RBD captures autonomic/GI/mood vulnerability independent of cognition."

### Scenario 4: No Stratification in Either Domain
- Cognitive: No RBD stratification ✗
- Prodromal: No RBD stratification ✗
- **Interpretation:** "RBD risk predicts incident PD but does not stratify cognitive decline or prodromal accumulation in non-converters. RBD may be a direct motor marker independent of general neurodegeneration progression."

---

## Multiple Testing Strategy

**Total tests:** 5 cognitive + 1 prodromal burden + 8 individual markers = **14 statistical tests**

**FDR applied across all 14 tests** (unified Benjamini-Hochberg correction)

Rationale: Both cognitive and prodromal analyses address the same research question (RBD stratification of neurodegeneration over time), so multiple testing correction should be integrated.

---

## Sample Size & Power

### Cognitive Delta
- RT: 34,272 pairs — **very high power** for small effects
- FI: 11,990 pairs — **good power** for medium effects
- TMT: ~15k pairs — **good power** for medium effects

### Prodromal Delta
- Full cohort: 93,254 — **excellent power** for RBD group stratification
- Individual markers: baseline prevalence varies (constipation 2.5%, depression 6%, etc.)
  - High prevalence → good power
  - Low prevalence → may be underpowered

---

## Robustness & Sensitivity Checks

### Cognitive Delta
1. Winsorize extreme deltas (±3 SD) and re-run
2. Stratify by age quartile at i0 (do older subjects show different patterns?)
3. Stratify by follow-up time quartile (do longer/shorter follow-ups differ?)

### Prodromal Delta
1. Sensitivity to outliers: Exclude subjects with >4 new prodromals (rare outliers)
2. Separate analysis for prevalent vs. incident prodromals (do baseline-present markers differ from new?)

---

## Deliverables

### Main Outputs
1. **Cognitive delta table:** Median delta by RBD group, KW p, FDR p, ε² (5 variables)
2. **Cognitive delta figure:** 5-panel violin+box plot
3. **Prodromal burden delta table:** Median delta by RBD group, KW p, FDR p, ε²
4. **Prodromal burden delta figure:** Single violin+box plot
5. **Individual marker table:** Incident % by RBD group + chi-square p + FDR p (8 markers)
6. **Individual marker figure:** 8-panel bar plot
7. **Integrated interpretation:** 2–3 page summary connecting cognitive + prodromal findings

### Supplementary Outputs
1. Dropout/completion rates by RBD group (for cognitive delta sample)
2. Sensitivity analysis results (winsorization, stratification)
3. Effect sizes (epsilon², OR with 95% CI)

---

## Scientific Narrative

**If results show both cognitive + prodromal stratification:**

"Over a 10-year period, controls stratified by baseline RBD risk show differential trajectories of both cognitive decline and prodromal accumulation. High-RBD individuals exhibit steeper cognitive decline (X% faster than Low-RBD) and accumulate more prodromal symptoms (Y more markers on average). These findings suggest that actigraphy-derived RBD probability score captures an integrated prodromal phenotype—combining sleep motor abnormalities, cognitive aging, and non-motor symptom emergence—that progresses towards PD even in individuals who have not yet converted. This supports RBD as a quantitative biomarker of prodromal severity, not merely an isolated sleep symptom."

---

## Timeline & Execution

1. Run cognitive delta audit (how many pairs per variable)
2. Run prodromal delta audit (prevalence and incident rates by RBD group)
3. Implement cognitive Kruskal-Wallis pipeline
4. Implement prodromal chi-square pipeline
5. Generate all tables and figures
6. Write interpretation summary
