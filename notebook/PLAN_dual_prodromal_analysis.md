# Plan: Dual Prodromal Analysis (Prevalence + Incident)

## Research Questions

**Q1: Prevalence** — Among controls with valid actigraphy, are high-RBD subjects MORE LIKELY to **HAVE** prodromal symptoms at follow-up?

**Q2: Incident** — Among controls with valid actigraphy, are high-RBD subjects MORE LIKELY to **DEVELOP** prodromal symptoms?

Both questions test RBD-driven stratification but capture different clinical signals:
- Q1 detects overall disease burden at follow-up (prevalence)
- Q2 detects progression rate (incidence = new-onset cases)

---

## Data Structure

**Baseline (_bl suffix):** Prodromal markers at baseline (i0)
**Follow-up (_post suffix):** Prodromal markers at latest follow-up

**Markers (8 total):**
- constipation, depression, anxiety, orthostatic, erectile_dysfunction, dream_enactment, anosmia, hyposmia

**Cohort:** Controls only (control=True) + valid actigraphy

**Outcome variables:**
- `prodromal_{marker}_bl` — binary flag at baseline
- `prodromal_{marker}_post` — binary flag at follow-up

---

## Analysis: Q1 (Prevalence at Follow-up)

### Prevalence burden
```
prevalence_burden_post = sum of {marker}_post flags (0–8)
```

**Test:** Kruskal-Wallis on burden_post across RBD groups
- Residualize on age + sex first
- Effect size: epsilon²
- Post-hoc: Dunn pairwise

### Individual markers (prevalence)
For each marker:
- **Denominator:** All subjects in group (n_total)
- **Numerator:** Those with marker_post = 1
- **% with marker = n_with_marker / n_total × 100**

**Test:** Chi-square test per marker by RBD group
- Effect: relative risk (RR)
- Multiple test correction: FDR across 8 markers + 1 burden = 9 tests

---

## Analysis: Q2 (Incident Cases)

### Incident burden
```
incident_burden = sum of {marker}_post where {marker}_bl = 0
= NEW markers developed post-baseline
Range: 0–8
```

**Test:** Kruskal-Wallis on incident_burden across RBD groups
- Residualize on age + sex
- Effect size: epsilon²
- Post-hoc: Dunn pairwise

### Individual markers (incident)
For each marker:
- **At-risk population:** Those with marker_bl = 0
- **Incident event:** marker_post = 1 AND marker_bl = 0
- **Incident % = n_incident / n_at_risk × 100**

**Test:** Chi-square test per marker by RBD group
- Denominator: n_at_risk (not total cohort)
- Effect: relative risk (RR) of developing symptom
- Multiple test correction: FDR across 8 markers + 1 burden = 9 tests

---

## Output Files

### Results (CSVs)

**Q1 (Prevalence):**
- `results_prodromal_prevalence_burden_kruskal_wallis.csv`
  - Columns: rbd_group, n, median_burden, kw_h, kw_p, fdr_p, epsilon2
- `results_prodromal_prevalence_markers_chisquare.csv`
  - Columns: marker, chi2, p, fdr_p, pct_low, pct_mid, pct_high, rr_high_vs_low

**Q2 (Incident):**
- `results_prodromal_incident_burden_kruskal_wallis.csv`
  - Columns: rbd_group, n, median_incident_burden, kw_h, kw_p, fdr_p, epsilon2
- `results_prodromal_incident_markers_chisquare.csv`
  - Columns: marker, n_at_risk, chi2, p, fdr_p, pct_incident_low, pct_incident_mid, pct_incident_high, rr_high_vs_low

### Audit Tables

- `audit_prodromal_prevalence_by_rbd_group.csv`
  - Per-group: n, n_with_data, median_burden_bl, median_burden_post, pct_with_any_post
- `audit_prodromal_incident_by_rbd_group.csv`
  - Per-group: n, n_at_risk (those with bl=0), median_incident_burden, pct_developing_any

### Figures

- `prodromal_prevalence_burden_by_rbd.png` — Violin+box of prevalence burden by RBD group
- `prodromal_incident_burden_by_rbd.png` — Violin+box of incident burden by RBD group
- `prodromal_prevalence_markers_by_rbd.png` — Bar chart: % with marker at follow-up
- `prodromal_incident_markers_by_rbd.png` — Bar chart: % who developed marker

---

## Script Design: `analyze_prodromal_prevalence_and_incident.py`

**Structure:**
```
1. Load controls + filter valid actigraphy
2. Compute prevalence burden (sum of _post flags)
3. Compute incident burden (sum of _post where _bl=0)
4. AUDIT section
   - Audit prevalence by RBD group
   - Audit incident by RBD group
   - Audit individual markers (both Q1 and Q2)
5. RESIDUALIZATION
   - Residualize prevalence_burden on age + sex
   - Residualize incident_burden on age + sex
6. Q1: PREVALENCE ANALYSIS
   - Kruskal-Wallis on prevalence_burden
   - Chi-square per marker (denominators = total per group)
   - FDR correction (9 tests)
7. Q2: INCIDENT ANALYSIS
   - Kruskal-Wallis on incident_burden
   - Chi-square per marker (denominators = at-risk per group)
   - FDR correction (9 tests)
8. GENERATE OUTPUTS
   - All 4 results CSVs
   - All 2 audit CSVs
   - All 4 figures
9. SUMMARY STATS (for interpretation MD)
   - Key findings for Q1
   - Key findings for Q2
```

---

## Interpretation Markdown: `prodromal_prevalence_and_incident_interpretation.md`

**Structure:**

### Q1: Prevalence at Follow-up
- Results table (burden + individual markers)
- Figure: bar charts for both burden and markers
- Interpretation: "High-RBD subjects are X% more likely to HAVE prodromal symptoms at follow-up..."

### Q2: Incident Cases
- Results table (incident burden + incident markers)
- Figure: bar charts for incident-specific rates
- Interpretation: "High-RBD subjects are X% more likely to DEVELOP new prodromal symptoms..."

### Integrated Interpretation
- Comparison: prevalence vs incident findings
- What do they tell us about RBD's predictive role?
- Clinical implications

---

## Unresolved Questions

- [ ] Should FDR be applied separately per question (9 tests each) or jointly (18 tests total)?
  - **Decision needed:** Unified correction treats both as same research domain (both test RBD stratification); separate keeps Q1/Q2 independent
  
- [ ] For incident analysis, exclude people with baseline=1 entirely, or include them with incident=0?
  - **Decision needed:** Cleaner to restrict at-risk denominator to baseline=0 only (what % of unaffected develop?)

---

## Execution Steps

1. ✅ Create unified script with both analysis pipelines
2. ✅ Compute audit tables for transparency
3. ✅ Run Kruskal-Wallis + chi-square for Q1
4. ✅ Run Kruskal-Wallis + chi-square for Q2
5. ✅ Generate all results CSVs and figures
6. ✅ Create interpretation MD with both Q1 and Q2 sections
7. ✅ Keep existing `delta_analysis_interpretation.md` untouched (cognitive + prevalence results)
8. ✅ New MD is standalone for the dual prodromal analysis
