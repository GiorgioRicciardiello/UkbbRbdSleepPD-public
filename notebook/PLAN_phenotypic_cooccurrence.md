# Phenotypic Co-occurrence Analysis: Plan

**Goal:** Map prodromal marker combinations in high-RBD controls to identify coherent phenotypic signatures.

**Research Question:** Among high-RBD controls, what % have isolated vs clustered prodromal markers? Are motor-autonomic markers (constipation+orthostatic) a coherent syndrome?

---

## Analysis Strategy

### 1. Single & Pairwise Co-occurrence
- **Single markers:** % with each marker (prevalence at follow-up)
- **Pairwise clusters:** 
  - Autonomic pair: constipation + orthostatic
  - Motor-autonomic + mood: (constipation OR orthostatic) + depression
  - Cognitive-motor: reaction time × (constipation OR orthostatic)
- **Domain groups:**
  - Autonomic domain: {constipation, orthostatic, erectile dysfunction}
  - Mood domain: {depression, anxiety}
  - Olfactory domain: {anosmia, hyposmia}
  - Cognitive: reaction time (from prior data)

### 2. Multi-marker Phenotypes
- **Isolation:** % with 0, 1, 2, 3+ markers
- **Phenotypic clusters:** 
  - Autonomic-dominant: constipation OR orthostatic, AND NO mood/olfactory
  - Mood-dominant: depression OR anxiety, AND NO autonomic
  - Mixed: 2+ domains represented
  - Silent: 0 markers

### 3. Stratification by RBD Group
- Compare phenotype prevalence across Low/Mid/High RBD
- Statistical tests: Chi-square for each phenotype × RBD group
- Effect sizes: Phi coefficient, Cramér's V

### 4. Outputs
**Tables:**
- Single marker prevalence by RBD group
- Pairwise co-occurrence (crosstabs)
- Multi-marker phenotype frequencies
- Domain clustering patterns

**Figures:**
- Heatmap: marker co-occurrence matrix (High-RBD vs Low-RBD)
- Stacked bar chart: phenotype distribution by RBD
- Venn diagram: autonomic vs mood vs olfactory overlap

**CSVs:**
- Single_marker_prevalence.csv
- Pairwise_cooccurrence_crosstabs.csv
- Phenotypic_clusters_frequency.csv
- Domain_dominance_by_rbd.csv

---

## Data Input

**Source:** `data/pp/res_build_final_dataset/merged_dataset_final.parquet`

**Filters:**
- control==True
- train_sleep==False (exclude training set)
- neuro_exclude==0

**Prodromal Markers (at follow-up, _post suffix):**
1. constipation_post
2. depression_post
3. anxiety_post
4. orthostatic_post
5. erectile_dysfunction_post
6. dream_enactment_post
7. anosmia_post
8. hyposmia_post

**RBD Groups:** From rg_pctl3 (Low/Mid/High)

---

## Implementation Notes

**Design Decisions:**
- Use **follow-up (post) prevalence**, not incident (allows cross-sectional phenotyping)
- **No adjustment** for age/sex (want raw phenotypes; adjust if FDR-corrected p-values needed)
- **Sparse markers (anosmia, hyposmia, dream_enactment):** Report but note low power
- **Fisher exact** for small cell counts (<5)

**Output Path:** `notebook/results/phenotypic_cooccurrence_analysis/`

---

## Expected Findings

**Hypothesis:** 
- High-RBD will show **autonomic clustering** (constipation + orthostatic together)
- **Motor-autonomic-only phenotype** more common in high-RBD (no mood/olfactory)
- **Mood-only phenotype** more common in low-RBD (mood-independent)

**Clinical Implication:** 
- Identifies whether high-RBD defines a coherent **autonomic syndrome** vs scattered prodromality
- Guides phenotype-specific interventions (e.g., autonomic support for high-RBD + orthostatic+constipation)

---

## Unresolved Questions

1. Should we include **baseline (pre) data** to show phenotype stability/change?
2. Should we apply **chi-square with FDR** to phenotype × RBD associations, or just descriptive?
3. Should we stratify by **age/sex** (e.g., do ED patterns differ by sex)?
4. Should **cognitive reaction time** be included (requires linking to different dataset with lower coverage)?

---

## Timeline

1. **Script generation:** analyze_phenotypic_cooccurrence.py (~400 lines)
2. **Execution:** Load data → compute tables → generate figures
3. **Interpretation:** Write 200-line phenotypic_cooccurrence_interpretation.md
4. **Integration:** Add to CLINICAL_UTILITY_QUESTIONS_MAP as answered question
