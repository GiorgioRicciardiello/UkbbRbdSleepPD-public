# Reporting Guideline: Output File → Model → Paper Extraction

**Pipeline entry point:** `src/cox_prodromal/runner.py`
**Results directory:** `results/cox_prodromal_abk_<timestamp>/`
**Report subdirectory:** `results/.../report/` (table-numbered CSVs mirroring the xlsx files)

---

## Primary Analysis (all outcomes unless noted)

| File | Model | Scientific Question | Extract for Paper |
|---|---|---|---|
| `rbd_only_cox.xlsx` | **Model A-i** — categorical 3-group RBD | Is RBD score associated with outcome? | HR (Intermediate vs Low, High vs Low), 95% CI, FDR-p — main effect table |
| `rbd_continuous.xlsx` | **Model A-ii** — continuous RBD per SD | Is dose-response linear and monotone? | HR per SD, 95% CI — report as sensitivity in text |
| `rbd_threshold_stability.xlsx` | **Model A-iii** — alternative cutoffs 5/10/15% | Is the HR sensitive to the 90th-percentile threshold choice? | HR at each cutoff — supplementary table |
| `baseline_cox_HRs.xlsx` | **Model B** — prodromal-only | Is each prodromal marker independently associated with outcome? | HR + FDR-adj p per prodromal variable — baseline association table |
| `additive_cox.xlsx` | **Model C** — RBD + prodromal additive | After conditioning on RBD, does the prodromal marker retain independent association? | HR for prodromal term + compare attenuation vs Model B (quantifies shared latent pathway) |
| `interaction_cox.xlsx` | **Model D** — multiplicative interaction | Does the prodromal marker's effect differ by RBD risk group? | Interaction term HR + p — report if significant; indicates multiplicative effect modification |
| `additive_interaction.xlsx` | **Model C** 4-group Cox + bootstrap | Is there biological synergy beyond additivity (Rothman RERI)? | RERI, AP, SI with bootstrap 95% CI — additive interaction section |
| `poisson_reri_sensitivity.xlsx` | Poisson RERI sensitivity | Is RERI robust in sparse cells? | RERI from Poisson — sensitivity paragraph |
| `competing_risk_cox.xlsx` | **Model E-iii** — cause-specific Cox | Does the RBD–PD association hold when competing events are explicitly handled? | Cause-specific HR (primary outcome only) — compare to Model A in text |
| `competing_risk_cif_vs_km.xlsx` | **Model E-ii** — Aalen-Johansen CIF vs KM | How much does KM overestimate cumulative incidence? | KM − AJ bias at 5 and 10 years — note in methods/sensitivity |
| `model_f_rbd_prs_stratified_interaction.xlsx` | **Model F** — RBD strata (3-group) × PRS_PD interaction (primary outcome only) | Does genetic PD risk (PRS_PD) amplify the RBD–outcome association multiplicatively across risk strata? | Stratified PRS HRs per 1-SD (Low, Mid, High RBD groups) with 95% bootstrap CI; full model summary with interaction term; interaction p-value — primary outcome only |

---

## Discrimination & Calibration (primary outcome `outcome_1a_pd_only` only)

| File | Component | Scientific Question | Extract for Paper |
|---|---|---|---|
| `c_index.xlsx` | Harrell C-index per model | Does adding RBD improve discrimination beyond covariates? | C-index for Models A, B, C; ΔC (model − null) |
| `discrimination_summary.xlsx` | Bootstrap ΔC, NRI, IDI | Does RBD statistically improve risk reclassification? | ΔC with 95% CI + p; NRI (events + non-events); IDI — discrimination table |

---

## Non-linearity & Dose-Response

| File | Component | Scientific Question | Extract for Paper |
|---|---|---|---|
| `spline_cox.xlsx` | Natural cubic spline Cox on continuous prodromal markers | Is the prodromal–outcome relationship linear? | LRT p for non-linearity per marker — footnote or supplementary |
| `rbd_spline.xlsx` | RBD dose-response spline (df=4, primary outcome) | Is there a threshold effect in the RBD–PD relationship? | LRT overall association + LRT non-linearity; reference = cohort median |
| `rbd_spline_hr_curve.xlsx` | Grid of HR(r) with delta-method 95% CI | Source data for spline figure | Use directly for Figure (spline HR curve + KDE panel) |
| `rbd_spline_model_data.xlsx` | Subject-level data used for spline fit | Reproducibility audit | Not reported; retain for archive |

---

## Model Fit & Proportional Hazards Diagnostics

| File | Component | Scientific Question | Extract for Paper |
|---|---|---|---|
| `model_fit_summary.xlsx` | AIC, BIC, LRT per model | Does the full model fit better than the null? | ΔAIC Model C vs null — supplementary |
| `ph_diagnostics.xlsx` | Schoenfeld residual test per covariate | Is the PH assumption satisfied? | Flag covariates with p < 0.05 — methods / supplement |
| `ph_violation_summary.xlsx` | Aggregated PH violations across outcomes | Which covariates systematically violate PH? | Summary statement in methods |
| `ph_time_interaction_sensitivity.xlsx` | Time-varying HR at t = 2, 5, 10 yr for PH violators | Does any PH violation materially change estimates? | HR(t) for violating covariates — reassurance paragraph (primary outcome only) |

---

## Sensitivity Analyses

| File | Component | Extract for Paper |
|---|---|---|
| `lag_sensitivity.xlsx` | 2-year reverse-causality lag (events within 2 yr excluded) | HR with vs without lag — sensitivity paragraph |
| `sensitivity_hes_active.xlsx` | HES coverage restriction (gap ≤ 4 yr to wear date) | HR in high-coverage subsample — addresses non-differential misclassification |
| `age_stratified_sensitivity.xlsx` | Replication within age strata (< 60 vs ≥ 60 at recruitment) | HRs by stratum — test for age-based effect modification (primary outcome only) |
| `km_logrank_summary.xlsx` | KM log-rank test per RBD risk group | Log-rank p per outcome — supports KM figures |
| `absolute_risks.xlsx` | KM-based cumulative incidence at 5 and 10 yr per risk group | Absolute risk by group — Figure 2 source data |
| `screening_metrics.xlsx` | Sensitivity, Specificity, PPV, NPV at 90/95/99th %ile × 5/10 yr | Screening performance section; PPV most clinically relevant |

---

## Per-Outcome Subdirectories

Each outcome subfolder (`outcome_1a_pd_only/`, `outcome_4a_ad_only/`, etc.) contains:
- KM cumulative incidence figures by RBD risk group
- Spline HR figure (primary outcome only)

---

## Mediation Analysis (`mediation/` subdirectory, primary outcome only)

**Source:** `src/rbd_prodromal_mediation/` — optional stage triggered by `RUN_MEDIATION = True`.
**Location:** `results/cox_prodromal_abk_<timestamp>/mediation/`

### Association tables (Models 1a, 1b, 1b-3g)

| File | Model | Scientific Question | Extract for Paper |
|---|---|---|---|
| `interpretation_A/assoc_linear.xlsx` | **Model 1a** — OLS: rbd_score_z ~ binary prodromal + covariates | Do binary prodromal markers predict the continuous RBD score? | β, HC3 SE, 95% CI, p, p_FDR, partial R² per marker — supplementary association table |
| `interpretation_C/assoc_linear.xlsx` | **Model 1a** — OLS: rbd_score_z ~ cognitive marker + covariates | Do cognitive impairments predict RBD score (a-path direction)? | β, HC3 SE, 95% CI, p, p_FDR, partial R² per marker |
| `interpretation_A/assoc_logistic.xlsx` | **Model 1b** — Logistic: binary high-RBD (≥p99) ~ binary prodromal | Does prodromal burden predict high-risk RBD group membership? | OR, 95% CI, p, p_FDR, Nagelkerke R², AUC per marker |
| `interpretation_C/assoc_logistic.xlsx` | **Model 1b** — Logistic: binary high-RBD ~ cognitive marker | Do cognitive deficits predict high-risk RBD membership? | OR, 95% CI, p, p_FDR, AUC per marker |
| `interpretation_A/assoc_logistic_3g.xlsx` | **Model 1b-3g** — Multinomial logit: 3-group RBD ~ binary prodromal | Does the association differ for Intermediate vs High-risk groups specifically? | OR for Intermediate vs Low and High vs Low, CI, p, p_FDR per marker |
| `interpretation_C/assoc_logistic_3g.xlsx` | **Model 1b-3g** — Multinomial logit: 3-group RBD ~ cognitive marker | Same question for cognitive markers | OR per contrast, CI, p, p_FDR |

### Mediation tables (Baron & Kenny)

| File | Component | Scientific Question | Extract for Paper |
|---|---|---|---|
| `interpretation_A/mediation_steps.xlsx` | c-path, a-path, b-path, c'-path — point estimates | Does the RBD score mediate the binary prodromal → PD relationship? | HR_c, β_a, HR_b, HR_c', PM%, inconsistency flag, p_c_fdr, p_a_fdr per marker |
| `interpretation_C/mediation_steps.xlsx` | Same for cognitive markers | Does RBD score mediate cognitive impairment → PD? | Same columns |
| `interpretation_A/mediation_indirect.xlsx` | Bootstrap HR_indirect, PM% with 95% CI | What fraction of the prodromal marker → PD association is mediated by RBD? | HR_indirect [95% CI], PM% [95% CI], n_converged per marker |
| `interpretation_C/mediation_indirect.xlsx` | Same for cognitive markers | Same question | Same columns |
| `interpretation_A/mediation_model_perf.xlsx` | C-index, AIC, BIC, LRT for c-path vs joint Cox | Does adding RBD improve discrimination of the prodromal-only model? | ΔC (joint − c-path) per marker — discrimination supplementary |
| `interpretation_A/mediation_feasibility.xlsx` | N, events per marker | Which variables had sufficient data for mediation analysis? | n_complete_case, events, feasible flag |
| `interpretation_A/supplementary_3g_bpath.xlsx` | Categorical 3-group b-path — HR for Intermediate vs Low and High vs Low | Does the High-risk group specifically drive the mediated pathway? | HR (High vs Low), attenuation of direct effect — supplementary |
| `interpretation_C/temporal_filter_log.xlsx` | Per cognitive variable: N nulled by temporal filter, N retained | How many cognitive measurements were excluded as post-baseline? | n_nulled, pct_loss per variable — data quality note in Methods |

### Combined report tables

| File | Contents | Extract for Paper |
|---|---|---|
| `report/mediation_summary_A.xlsx` | One row per binary prodromal marker: c-path HR, direct effect HR, HR_indirect [bootstrap 95% CI], PM% [95% CI], inconsistent flag | Primary mediation results table for Interpretation A |
| `report/mediation_summary_C.xlsx` | Same for cognitive markers | Primary mediation results table for Interpretation C |
| `report/model_performance_summary.xlsx` | C-index comparison: c-path vs joint model, ΔC, AIC per prodromal variable | Model discrimination gain from mediator — supplementary discrimination table |

---

## Writing Priority

### Main text tables
1. `baseline_cox_HRs` — prodromal marker associations (Model B), all outcomes
2. `rbd_only_cox` — RBD risk group HRs (Model A-i), primary + secondary outcomes
3. `additive_cox` — joint model (Model C), attenuation of prodromal HRs vs Model B
4. `additive_interaction` — RERI / AP / SI for biological interaction
5. `discrimination_summary` — ΔC, NRI, IDI
6. `report/mediation_summary_A` + `mediation_summary_C` — PM% and HR_indirect for each marker (if mediation is included in main text)

### Main text figures
- KM curves (source: `absolute_risks` + `km_logrank_summary`)
- RBD dose-response spline HR curve (source: `rbd_spline_hr_curve`)

### Supplementary
- `rbd_threshold_stability`, `lag_sensitivity`, `sensitivity_hes_active`
- `age_stratified_sensitivity`, `ph_diagnostics`, `ph_time_interaction_sensitivity`
- `competing_risk_*`, `screening_metrics`, `poisson_reri_sensitivity`
- `mediation/interpretation_A/assoc_linear`, `assoc_logistic`, `assoc_logistic_3g`
- `mediation/interpretation_C/assoc_linear`, `assoc_logistic`, `temporal_filter_log`
- `mediation/interpretation_A/supplementary_3g_bpath`
- `mediation/report/model_performance_summary`

---

## Model Comparison Logic (for Results narrative)

| Comparison | Inference | Decision criterion |
|---|---|---|
| A vs C | Does the prodromal marker add information beyond RBD? | β_P in Model C significant after FDR |
| B vs C | Does RBD add information beyond the prodromal marker? | β_R in Model C significant |
| B vs D | Does the prodromal HR differ across RBD strata? | β_RP in Model D significant (multiplicative modification) |
| C vs D | Does the interaction term improve model fit? | LRT p < 0.05 |
| A vs E | Does the RBD–outcome association persist after competing events? | Cause-specific HR from Model E consistent with Model A |
| B vs B_lag | Are prodromal associations robust to reverse-causality? | HR direction and magnitude stable after 2-yr lag |

### Mediation-specific comparisons

| Comparison | Inference | Decision criterion |
|---|---|---|
| c-path vs joint | Does conditioning on RBD attenuate the prodromal–PD association? | HR_c' < HR_c; attenuation quantified as (HR_c − HR_c') / (HR_c − 1) |
| PM% | What fraction of the prodromal–PD pathway is captured by RBD? | Bootstrap 95% CI excludes 0; interpret 20–50% as partial mediation, >50% as substantial |
| Inconsistent mediation | Is the indirect effect in the opposite direction from the total effect? | `inconsistent_mediation = True`; report in text; can occur when a-path and total effect have opposite signs |
| 3g supplementary | Does the High-risk RBD group specifically drive mediation? | HR (High vs Low) substantially larger than HR (Intermediate vs Low) in the joint model |
