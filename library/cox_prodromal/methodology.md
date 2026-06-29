# Methodology: Cox Prodromal Analysis Pipeline

**Source library:** `src/cox_prodromal/`
**Primary outcome:** Incident Parkinson's disease (`outcome_1a_pd_only`)
**RBD score source:** ABK actigraphy model (`abk_rbd_score_mean`)

---

## 1. Study Design and Time Origin

All survival analyses use **actigraphy wear date** (`wear_time_start`) as the time origin — not UKBB recruitment. This is defined in `data_prep.py::build_survival_dataset_for_outcome()` via `select_survival_dataset(time_unit="years")`.

**Analytic cohort per outcome:**
Subjects included are those with `{outcome}_incident == True` (cases) or `control == True` (controls). Prevalent cases — diagnosed before `wear_time_start` — are excluded at the survival dataset construction step.

**Administrative censoring:**
`censor_date = min(death_date, 2025-11-30)` per subject, defined in `outcome_flags.py`. Follow-up time is measured in years from `wear_time_start` to `censor_date`.

**Global exclusions applied in `get_clean_risk_data()`:**
- `neuro_exclude != 0`: subjects with prevalent neurological disease at baseline
- `train_sleep == True`: not applied for the ABK model (all UKBB subjects have `train_sleep = False`)

---

## 2. Outcomes

Seven outcomes are analysed (defined in `cox_config.py::OUTCOMES`):

| Identifier | Definition |
|---|---|
| `outcome_1a_pd_only` | PD (G20), no AD, no DEM — **primary** |
| `outcome_1b_pd_ad` | PD ∩ AD |
| `outcome_2a_otherdementia` | DEM, no PD, no AD |
| `outcome_2b_pd_otherdementia` | PD ∩ DEM, no AD |
| `outcome_3a_dlb_only` | DLB (G31.8), no PD |
| `outcome_4a_ad_only` | AD (G30), no PD, no DEM |
| `outcome_5a_pd_med` | PD defined by medication evidence only (no ICD-10 requirement) |

`outcome_5a_pd_med` provides a sensitivity outcome for cases captured by antiparkinsonian drug prescriptions but lacking a coded G20 diagnosis; it is expected to show a weaker RBD association than `outcome_1a_pd_only`.

**Survival encoding** (`outcome_flags.py::build_survival()`):

$$
\text{surv\_event}_i =
\begin{cases}
1 & \text{if incident case: } T_{\text{dx},i} \in [\text{wear\_start}_i,\ \text{censor}_i] \\
2 & \text{if competing event (death without outcome): } T_{\text{death},i} \in (\text{wear\_start}_i,\ \text{censor}_i] \\
0 & \text{otherwise (administratively censored)}
\end{cases}
$$

$$
\text{surv\_time}_i =
\begin{cases}
T_{\text{dx},i} - \text{wear\_start}_i & \text{if event} = 1 \\
T_{\text{death},i} - \text{wear\_start}_i & \text{if event} = 2 \\
\text{censor}_i - \text{wear\_start}_i & \text{if event} = 0
\end{cases}
$$

Prevalent cases receive `surv_event = NaN` and `surv_time = NaN` and are excluded from all models.

---

## 3. RBD Exposure Variable

The RBD exposure is derived from the **ABK actigraphy model**, which assigns a nightly RBD probability score. Subject-level scores are computed in `run_merge_ukbb_rbd.py`:

$$
\text{abk\_rbd\_score\_mean}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \text{abk\_rbd\_score}_{i,n}
$$

where $N_i$ is the number of valid actigraphy nights for subject $i$.

Two risk group encodings are used (defined in `cox_config.py::METHODS`):

| Method | Column suffix | Groups |
|---|---|---|
| `percentile_3g` | `risk_group_mean_3g` | Low (0–90th %ile) / Intermediate (90–99th) / High (99–100th) — **primary** |
| `percentile_2g` | `risk_group_mean_2g` | Low (0–90th %ile) / High (90–100th %ile) |

The three-tier encoding (`percentile_3g`) is the primary stratification used in all main analyses (Models A–D, figures, and tables), consistent with the publication terminology: **Low** (0–90th percentile, blue), **Intermediate** (90th–99th, orange), **High** (99th–100th, red).

Thresholds are computed on the full dataset (no validation split for UKBB) and stored as JSON files in `data/risk_thresholds/`.

**Continuous per-SD encoding** (Model A-ii): the score is standardised to $Z = (X - \bar{X}) / s_X$ before model fitting (`model_rbd.py::fit_rbd_continuous_per_sd()`).

---

## 4. Prodromal Marker Variables

### 4.1 Continuous Cognitive Markers (`cox_config.py::PRODROMAL_VARS`)

All measured at UKBB recruitment visit (suffix `_bl`, formerly `_i0`) or imaging
visit (suffix `_fu`, formerly `_i2`):

| Column | Label |
|---|---|
| `cog_fluid_intelligence_bl` | Fluid Intelligence Score |
| `cog_react_time_bl` | Reaction Time (ms) |
| `cov_fi_questions_attempted_20128_bl` | FI Questions Attempted |
| `cog_numeric_memory_bl` | Numeric Memory |
| `trail_making_errors_trail1_i2` | Trail Making Errors (Trail 1) |
| `cog_pairs_matching_bl` | Pairs Matching Status |

### 4.2 Binary Prodromal Markers (`cox_config.py::PRODROMAL_BINARY_VARS`)

Derived from merged HES ICD-10 records + self-reported medication evidence. Ascertainment window: all records prior to `wear_time_start`.

| Column | Label |
|---|---|
| `prodromal_constipation_bl` | Constipation |
| `prodromal_depression_bl` | Depression |
| `prodromal_anxiety_bl` | Anxiety |
| `prodromal_orthostatic_bl` | Orthostatic Hypotension |
| `prodromal_erectile_dysfunction_bl` | Erectile Dysfunction |
| `prodromal_dream_enactment_bl` | Dream Enactment |
| `prodromal_anosmia_bl` | Anosmia |
| `prodromal_hyposmia_bl` | Hyposmia |

**Minimum prevalence filter** (`data_prep.py::filter_active_variables()`): binary variables with fewer than **30 positive cases** in the analytic cohort are excluded from analysis.

---

## 5. Adjustment Covariates

**Base covariates** (`cox_config.py::BASE_COVARIATES`), included in all models:

$$
\mathbf{X} = [\text{age\_at\_recruitment},\ \text{sex},\ \text{BMI}]
$$

**Lifestyle covariates** (`data_prep.py::prepare_lifestyle_covariates()`): smoking status (`cov_smoking`) and alcohol status (`cov_alcohol`) are constructed by taking the first non-null, non-negative value across visits $i_0, i_1, i_2, i_3$. Added to $\mathbf{X}$ only when at least one observation is available.

---

## 6. Causal Framework (DAG)

The assumed causal structure (`cox_config.py::CAUSAL_DAG_NODES`, `CAUSAL_DAG_EDGES`) is:

```
latent_pd_pathology ──► rbd_score
latent_pd_pathology ──► prodromal_marker
latent_pd_pathology ──► incident_pd
age  ──► {rbd_score, prodromal_marker, incident_pd}
sex  ──► {rbd_score, prodromal_marker, incident_pd}
bmi  ──► {rbd_score, prodromal_marker, incident_pd}
```

**Key assumption:** There is no direct edge between `rbd_score` and `prodromal_marker`. Their association is assumed to be entirely explained by their shared latent pathology. This implies that in Model C (additive), adjustment for one does not introduce collider bias on the other.

---

## 7. Model Hierarchy and Comparison Design

Five Cox models (labelled **Model A–E** in reports; internally M0–M4 in code) are fitted for each (outcome × prodromal variable × RBD method) combination.

### 7.0 Overview

| Label | Code tag | Script | Formula (compact) | Scientific question |
|-------|----------|--------|-------------------|---------------------|
| **Model A** | M0 | `model_rbd.py` | h₀(t) exp(β_R R + β_X X) | Is RBD score associated with outcome, independent of confounders? |
| **Model B** | M1 | `model_baseline.py` | h₀(t) exp(β_P P + β_X X) | Is the prodromal marker associated with outcome, independent of confounders? |
| **Model C** | M2 | `model_additive.py` | h₀(t) exp(β_R R + β_P P + β_X X) | Do RBD and the prodromal marker have **independent** additive contributions? |
| **Model D** | M3 | `model_interaction.py` | h₀(t) exp(β_R R + β_P P + β_RP(R×P) + β_X X) | Is there **multiplicative effect modification** between RBD and the prodromal marker? |
| **Model E** | M4 | `model_competing.py` | Aalen-Johansen CIF + cause-specific Cox | Are results materially changed when competing events are accounted for? |

### 7.1 Pairwise Model Comparisons

Each pair of models addresses a specific inference question:

| Comparison | Question | Decision criterion |
|------------|----------|-------------------|
| **A vs C** | Does the prodromal marker add independent information beyond RBD? | β_P in Model C significant after FDR; HR_P direction consistent |
| **B vs C** | Does RBD add independent information beyond the prodromal marker? | β_R in Model C significant; HR_R direction consistent |
| **B vs D** | Does the prodromal marker's hazard ratio differ across RBD risk strata? | β_RP in Model D significant (p < 0.05); confirms multiplicative modification |
| **C vs D** | Does adding the interaction term improve model fit? | LRT Λ = −2(ℓ_C − ℓ_D) ~ χ²_{df_RP}; p < 0.05 |
| **A vs E** | Does the RBD–outcome association persist after censoring competing events? | Cause-specific HR from Model E consistent with Model A HR |
| **B vs B_lag** | Are prodromal associations robust to 2-year reverse-causality lag? | HR_lag direction and magnitude stable relative to HR_primary |

### 7.2 Model A — RBD-Only (`model_rbd.py`)

**A-i. Categorical (primary):**

$$
h(t \mid R, \mathbf{X}) = h_0(t)\, \exp\!\bigl(\boldsymbol{\beta}_R^\top \mathbf{R} + \boldsymbol{\beta}_X^\top \mathbf{X}\bigr)
$$

where $\mathbf{R}$ is a dummy-encoded vector for the RBD risk group (reference: Low), with the first category dropped (`drop_first=True`). For the primary method (`percentile_3g`), this yields two dummy columns: Intermediate vs Low, and High vs Low.

**A-ii. Continuous per-SD:**

$$
h(t \mid Z_R, \mathbf{X}) = h_0(t)\, \exp\!\bigl(\beta_R\, Z_R + \boldsymbol{\beta}_X^\top \mathbf{X}\bigr)
$$

where $Z_R = (R - \bar{R}) / s_R$. The reported HR is $\exp(\hat{\beta}_R)$, interpreted as the hazard ratio per one standard deviation increase in the RBD score.

**A-iii. Threshold stability** (`fit_rbd_threshold_stability()`): Model A-i is repeated at three alternative percentile cutoffs $p \in \{5\%, 10\%, 15\%\}$. At each cutoff, subjects with $R \geq (100-p)\text{th percentile}$ are labelled High.

### 7.3 Model B — Prodromal-Only (`model_baseline.py`)

$$
h(t \mid P, \mathbf{X}) = h_0(t)\, \exp\!\bigl(\boldsymbol{\beta}_P^\top \mathbf{P} + \boldsymbol{\beta}_X^\top \mathbf{X}\bigr)
$$

$\mathbf{P}$ is the prodromal variable. If categorical or binary, it is dummy-encoded with the lowest-risk category as reference. If continuous, it enters as a linear term.

Estimates the **total association** between a prodromal marker and the outcome without conditioning on RBD. This is the primary estimate of prodromal marker utility reported in the baseline Cox HR table.

### 7.4 Model C — Additive Combined (`model_additive.py`)

$$
h(t \mid R, P, \mathbf{X}) = h_0(t)\, \exp\!\bigl(\boldsymbol{\beta}_R^\top \mathbf{R} + \boldsymbol{\beta}_P^\top \mathbf{P} + \boldsymbol{\beta}_X^\top \mathbf{X}\bigr)
$$

Both $\mathbf{R}$ and $\mathbf{P}$ are dummy-encoded. Reference categories are selected as the lowest-risk label (containing "low", "never", or "no" in the label string) via `categorical_ref.py`. No interaction term.

**Interpretation:** If $\hat{\beta}_P$ in Model C attenuates toward zero compared with Model B, RBD and the prodromal marker share variance explained by the latent pathology (consistent with the DAG). If $\hat{\beta}_P$ is unchanged, the association is independent of the RBD pathway. The comparison B → C directly quantifies how much of the prodromal–outcome association is explained by the shared latent neurodegeneration signal captured in the RBD score.

### 7.5 Model D — Multiplicative Interaction (`model_interaction.py`)

$$
h(t \mid R, P, \mathbf{X}) = h_0(t)\, \exp\!\bigl(\boldsymbol{\beta}_R^\top \mathbf{R} + \boldsymbol{\beta}_P^\top \mathbf{P} + \boldsymbol{\beta}_{RP}^\top (\mathbf{R} \otimes \mathbf{P}) + \boldsymbol{\beta}_X^\top \mathbf{X}\bigr)
$$

where $\mathbf{R} \otimes \mathbf{P}$ denotes all pairwise products of the dummy columns. For the primary three-tier RBD encoding and binary prodromal, this yields two interaction columns: Intermediate×Yes and High×Yes.

The interaction coefficients $\hat{\boldsymbol{\beta}}_{RP}$ test **multiplicative** effect modification: whether the hazard ratio for the prodromal marker differs between RBD groups. A significant positive interaction implies the combined hazard exceeds the product of the individual hazard ratios. Note: multiplicative interaction on the HR scale is distinct from additive (biological) interaction tested via RERI (Section 12).

### 7.6 Model E — Competing Risk (`model_competing.py`)

**E-i. Aalen-Johansen cumulative incidence function (CIF):**

The CIF for the primary event $k=1$ in the presence of competing event $k=2$ is:

$$
F_1(t) = \int_0^t S(u^-)\, d\Lambda_1(u)
$$

where $S(t) = \exp\!\left(-\sum_k \Lambda_k(t)\right)$ is the overall survival function and $\Lambda_k(t)$ is the cause-specific cumulative hazard for event $k$, estimated non-parametrically by the Aalen-Johansen estimator (`lifelines.AalenJohansenFitter`).

**Multi-state event encoding** (`encode_competing_events()`):

$$
\delta_i =
\begin{cases}
1 & \text{primary event} \\
2 & \text{competing event (earlier of: cross-diagnosis, death)} \\
0 & \text{censored}
\end{cases}
$$

When both a primary and a competing event are recorded, the event with the earlier time is used.

**Competing outcomes per primary outcome** (`cox_config.py::COMPETING_OUTCOMES`):

| Primary | Competing events |
|---|---|
| `outcome_1a_pd_only` | `outcome_2a_otherdementia`, `outcome_4a_ad_only`, death |
| `outcome_4a_ad_only` | `outcome_1a_pd_only`, `outcome_2a_otherdementia`, death |
| `outcome_2a_otherdementia` | `outcome_1a_pd_only`, `outcome_4a_ad_only`, death |
| `outcome_5a_pd_med` | `outcome_2a_otherdementia`, `outcome_4a_ad_only`, death |

**Pipeline scope:** In the current implementation, Model E is run for `PRIMARY_OUTCOME` only (`runner.py:890–931`). `COMPETING_OUTCOMES` defines four primary outcomes with competing event lists, but the runner adds the additional guard `outcome == PRIMARY_OUTCOME`. The remaining three outcomes (`outcome_4a_ad_only`, `outcome_2a_otherdementia`, `outcome_5a_pd_med`) have competing event mappings defined but do not execute the Aalen-Johansen or cause-specific Cox analyses in the automated pipeline.

**E-ii. CIF vs 1-KM comparison** (`compare_cif_vs_km()`):
The standard Kaplan-Meier estimator overestimates cumulative incidence when competing events exist because it treats them as non-informative censoring:

$$
\text{bias} = \widehat{F}_1^{\text{KM}}(t) - \widehat{F}_1^{\text{AJ}}(t)
$$

where $\widehat{F}_1^{\text{KM}}(t) = 1 - \hat{S}^{\text{KM}}(t)$. This difference is reported at $t \in \{5, 10\}$ years.

**E-iii. Cause-specific Cox** (`fit_cause_specific_cox()`):
Competing events are treated as non-informative censoring (standard Cox likelihood). This is identical in form to Model A-i but makes the competing-risk interpretation explicit. It estimates the cause-specific hazard ratio, not the sub-distribution hazard ratio.

---

## 8. Model Fitting Details

All Cox models are fitted using `lifelines.CoxPHFitter` with:

- **Robust (sandwich) variance estimator** (`robust=True`): standard errors are computed via the Lin-Wei sandwich estimator, which is consistent under mild misspecification of the baseline hazard and heteroscedasticity.
- **Ridge penalisation** (`penalizer=0.01`): a L2 penalty $\frac{\lambda}{2}\|\boldsymbol{\beta}\|^2$ is added to the log partial-likelihood to improve numerical stability in the presence of near-collinear dummy variables.
- **Minimum events threshold** = 5: models with fewer than 5 events in the complete case are skipped.

**Model fit metrics** extracted per model: partial AIC (`AIC_partial_` — the correct lifelines attribute for semi-parametric Cox models; `AIC_` raises an error), BIC, log-likelihood, LRT statistic vs null, $\Delta$AIC, $\Delta$BIC, C-index.

---

## 9. Proportional Hazards Assumption

### 9.1 Schoenfeld residual test

Tested via **Schoenfeld residuals** using `lifelines.statistics.proportional_hazard_test` with rank-transformed time (`time_transform="rank"`) — equivalent to a Grambsch-Therneau test (`diagnostics.py::run_ph_test()`).

For each covariate $j$ in a fitted model, the test statistic is:

$$
\chi^2_j = \frac{\hat{r}_j^\top \mathbf{W} \hat{r}_j}{\hat{\sigma}^2_j}
$$

where $\hat{r}_j$ are the scaled Schoenfeld residuals for covariate $j$ and $\mathbf{W}$ is a weight matrix based on the rank-transformed event times. A p-value $< 0.05$ is flagged as a violation.

### 9.2 Time-covariate interaction sensitivity for PH violators (`model_time_varying.py`)

**Pipeline scope:** Applied to `PRIMARY_OUTCOME` only (`runner.py:761–772`).

For covariates where the Schoenfeld test rejects PH ($p < 0.05$), the model is refit with a `covariate × log(time)` interaction term following Therneau & Grambsch (2000, §6.3):

$$
h(t) = h_0(t)\, \exp\!\bigl(\beta_j x_j + \gamma_j\, x_j \log(t + \varepsilon) + \boldsymbol{\beta}_{-j}^\top \mathbf{X}_{-j}\bigr)
$$

The time-varying HR at time $t$ is:

$$
\text{HR}(t) = \exp\!\bigl(\hat{\beta}_j + \hat{\gamma}_j \log(t)\bigr)
$$

Results are reported at $t \in \{2, 5, 10\}$ years. If $\hat{\gamma}_j$ is non-significant ($p \geq 0.05$), the PH violation is deemed inconsequential. If significant, the time-varying HRs at the three time points are reported alongside the original static HR.

---

## 10. Multiple Testing Correction

Across all prodromal variables within each outcome, Benjamini-Hochberg false discovery rate (FDR) correction is applied to the primary p-values (`diagnostics.py::apply_fdr()`):

$$
p_{\text{adj},\,(k)} = \min\!\left(1,\ \frac{m}{k}\, p_{(k)}\right)
$$

where $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$ are the ordered raw p-values and $k$ is the rank. Implemented via `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.

---

## 11. Spline Models (`splines.py`, `rbd_spline_analysis.py`)

### 11.1 Non-linearity assessment for continuous prodromal markers

**Pipeline scope:** Restricted to `PRIMARY_OUTCOME` and continuous (non-binary) prodromal markers only (`runner.py:548`).

A natural cubic spline basis is constructed using `patsy.dmatrix("cr(x, df=4) - 1")` with **4 degrees of freedom**. The spline Cox model is:

$$
h(t \mid f(P), \mathbf{X}) = h_0(t)\, \exp\!\bigl(\boldsymbol{\gamma}^\top \mathbf{s}(P) + \boldsymbol{\beta}_X^\top \mathbf{X}\bigr)
$$

where $\mathbf{s}(P) = [s_1(P), s_2(P), s_3(P), s_4(P)]^\top$ are the natural cubic spline basis functions.

Non-linearity is assessed by likelihood-ratio test (LRT) against the linear model:

$$
\Lambda = -2\bigl(\ell_{\text{linear}} - \ell_{\text{spline}}\bigr) \sim \chi^2_{k-1}
$$

where $k = 4$ degrees of freedom, so $\chi^2_3$.

### 11.2 RBD dose-response spline analysis (`rbd_spline_analysis.py`)

A dedicated sub-analysis for the primary outcome (`outcome_1a_pd_only`) fits a natural cubic spline (df=4) Cox model on the continuous RBD probability score and generates a two-panel publication figure.

**Reference point:** the cohort median RBD score (post-prevalent-case exclusion), not the minimum. All HR estimates are relative to this reference.

$$
\text{HR}(r) = \exp\!\bigl(\boldsymbol{\gamma}^\top [\mathbf{s}(r) - \mathbf{s}(r_{\text{med}})]\bigr)
$$

Pointwise 95% confidence intervals are computed via the **delta method**:

$$
\text{Var}\!\left[\log \widehat{\text{HR}}(r)\right] = [\mathbf{s}(r) - \mathbf{s}(r_{\text{med}})]^\top\, \hat{\mathbf{V}}_{\boldsymbol{\gamma}}\, [\mathbf{s}(r) - \mathbf{s}(r_{\text{med}})]
$$

where $\hat{\mathbf{V}}_{\boldsymbol{\gamma}}$ is the variance-covariance submatrix for the spline coefficients, extracted from `cph.variance_matrix_`. The HR curve is evaluated on a 300-point grid over the observed range of RBD scores (`N_GRID = 300`).

**Two formal LRT tests are reported:**

| Test | Null | Alternative | df |
|---|---|---|---|
| Overall association | Covariate-only (no RBD) | Spline RBD | 4 |
| Non-linearity | Linear RBD | Spline RBD | 3 |

Knot positions from the training data are preserved via `patsy.build_design_matrices(design_info, ...)` when evaluating the curve on the grid, avoiding the single-point basis failure that occurs with plain `dmatrix`.

**Figure output (two panels):**
- **Panel A:** Spline HR curve with delta-method 95% CI ribbon; categorical HR point estimates from the M0 pipeline results overlaid; threshold lines at the 90th and 99th percentile.
- **Panel B:** Kernel density estimate of RBD scores for cases vs non-cases.

---

## 12. Additive Interaction Analysis (`additive_interaction.py`, `additive_interaction_poisson.py`)

Additive interaction tests departure from **additivity of absolute risks**, corresponding to biological interaction in the Rothman sense.

### 12.1 Four-group Cox model

A single Cox model is fitted with three binary indicator variables using $R=0, P=0$ as the reference:

$$
h(t) = h_0(t)\, \exp\!\bigl(\beta_{10}\, \mathbb{1}[R=1,P=0] + \beta_{01}\, \mathbb{1}[R=0,P=1] + \beta_{11}\, \mathbb{1}[R=1,P=1] + \boldsymbol{\beta}_X^\top \mathbf{X}\bigr)
$$

Yielding hazard ratios:

$$
\text{HR}_{10} = e^{\hat\beta_{10}},\quad \text{HR}_{01} = e^{\hat\beta_{01}},\quad \text{HR}_{11} = e^{\hat\beta_{11}},\quad \text{HR}_{00} = 1
$$

### 12.2 Measures of additive interaction

**Relative Excess Risk due to Interaction (RERI):**

$$
\text{RERI} = \text{HR}_{11} - \text{HR}_{10} - \text{HR}_{01} + 1
$$

Null value: $\text{RERI} = 0$. Positive values indicate super-additivity (synergy).

**Attributable Proportion (AP):**

$$
\text{AP} = \frac{\text{RERI}}{\text{HR}_{11}}
$$

Null value: $\text{AP} = 0$.

**Synergy Index (S):**

$$
S = \frac{\text{HR}_{11} - 1}{(\text{HR}_{10} - 1) + (\text{HR}_{01} - 1)}
$$

Null value: $S = 1$.

### 12.3 Bootstrap confidence intervals

Because RERI is a nonlinear function of the hazard ratios, closed-form variance is not available. Percentile bootstrap confidence intervals are computed with $B = 1000$ resamples (`seed=42`, `numpy.random.default_rng`):

1. Compute point estimates $\widehat{\text{RERI}}, \widehat{\text{AP}}, \hat{S}$ from the original data.
2. For each bootstrap resample $b = 1, \ldots, B$: draw $n$ subjects with replacement, fit the four-group Cox model, compute $\text{RERI}^{(b)}, \text{AP}^{(b)}, S^{(b)}$.
3. Report $[\text{2.5th percentile},\ \text{97.5th percentile}]$ of the bootstrap distribution.

Samples are discarded if the four-group Cox fails to converge. A warning is issued if fewer than 50% of bootstrap samples converge.

### 12.4 Poisson RERI sensitivity analysis (`additive_interaction_poisson.py`)

As a sensitivity check for cells with sparse events, RERI is also estimated via **Poisson regression with log(time) offset** (Zou, 2004; Am J Epidemiol). Under the rare disease assumption (PD prevalence ≈ 0.4%), the incidence rate ratio (IRR) from Poisson regression approximates the hazard ratio:

$$
\log\, E[\text{events}_i] = \log(t_i) + \beta_0 + \beta_{10}\, g_{10} + \beta_{01}\, g_{01} + \beta_{11}\, g_{11} + \boldsymbol{\beta}_X^\top \mathbf{X}
$$

IRRs are exponentiated coefficients; $\text{RERI} = \text{IRR}_{11} - \text{IRR}_{10} - \text{IRR}_{01} + 1$. Wald 95% CIs for each IRR are computed via the delta method. A `sparse_cell_warning` flag is set when any cell has fewer than 10 events.

---

## 13. Discrimination Metrics (`discrimination.py`)

**Pipeline scope:** All discrimination metrics (delta-C, NRI, IDI) are computed for `PRIMARY_OUTCOME` only (`runner.py:775`).

### 13.1 Harrell's C-index

The concordance index measures the probability that a randomly selected case has a higher predicted hazard than a randomly selected control:

$$
C = \frac{\#\{(i,j): T_i < T_j,\ \delta_i = 1,\ \hat{h}_i > \hat{h}_j\}}{\#\{(i,j): T_i < T_j,\ \delta_i = 1\}}
$$

Extracted directly from `CoxPHFitter.concordance_index_`. The **incremental C-index** is:

$$
\Delta C = C_{\text{full}} - C_{\text{null}}
$$

where $C_{\text{null}}$ is the C-index of the covariates-only model (age, sex, BMI ± lifestyle).

### 13.2 Bootstrap $\Delta C$ test

Statistical significance of $\Delta C$ is assessed via bootstrap (`bootstrap_delta_c_test()`, $B=1000$, `seed=42`):

$$
p = \Pr(\Delta C^{(b)} \leq 0) = \frac{1}{B}\sum_{b=1}^B \mathbb{1}[\Delta C^{(b)} \leq 0]
$$

Bootstrap percentile 95% CI: $[\text{2.5th},\ \text{97.5th}]$ percentile of $\{\Delta C^{(b)}\}$.

### 13.3 Net Reclassification Improvement (NRI)

Category-based NRI (Pencina et al., 2008) at a risk threshold $\tau$:

$$
\text{NRI} = \underbrace{\frac{N_{\uparrow|\text{event}} - N_{\downarrow|\text{event}}}{N_{\text{event}}}}_{\text{NRI}_{\text{events}}} + \underbrace{\frac{N_{\downarrow|\text{non-event}} - N_{\uparrow|\text{non-event}}}{N_{\text{non-event}}}}_{\text{NRI}_{\text{non-events}}}
$$

where $\uparrow / \downarrow$ indicate reclassification to higher/lower risk category between the null and full model. The threshold $\tau$ is set to the **median predicted partial hazard of the full model** (`runner.py:847`). Standard error:

$$
\text{SE}(\text{NRI}) = \sqrt{\frac{N_{\uparrow|\text{event}} + N_{\downarrow|\text{event}}}{N_{\text{event}}^2} + \frac{N_{\uparrow|\text{non-event}} + N_{\downarrow|\text{non-event}}}{N_{\text{non-event}}^2}}
$$

Two-sided z-test: $z = \text{NRI}/\text{SE}$.

### 13.4 Integrated Discrimination Improvement (IDI)

Continuous analog of NRI (Pencina et al., 2008), requiring no threshold:

$$
\text{IDI} = \underbrace{(\bar{p}_{\text{new},\text{event}} - \bar{p}_{\text{new},\text{non-event}})}_{\text{IS}_{\text{new}}} - \underbrace{(\bar{p}_{\text{old},\text{event}} - \bar{p}_{\text{old},\text{non-event}})}_{\text{IS}_{\text{old}}}
$$

where $\bar{p}_{*,\text{event}}$ and $\bar{p}_{*,\text{non-event}}$ are mean predicted partial hazards for events and non-events respectively. Approximate variance:

$$
\text{Var}(\text{IDI}) \approx \frac{\text{Var}(\hat{p}_{\text{new}}|\text{event})}{n_{\text{event}}} + \frac{\text{Var}(\hat{p}_{\text{new}}|\text{non-event})}{n_{\text{non-event}}} + \frac{\text{Var}(\hat{p}_{\text{old}}|\text{event})}{n_{\text{event}}} + \frac{\text{Var}(\hat{p}_{\text{old}}|\text{non-event})}{n_{\text{non-event}}}
$$

---

## 14. Calibration (`calibration.py`)

**Pipeline scope:** Calibration metrics are computed for `PRIMARY_OUTCOME` only, for the full model (RBD + covariates) vs covariates-only (`runner.py:870–885`).

Calibration quantifies whether predicted risks agree with observed event rates. Two estimands are reported, both at a specified time horizon $t^*$. Binary outcome is defined as: event within $t^*$ years.

### 14.1 Calibration slope

Logistic regression of the binary outcome on $\log(\hat{h}_i)$ (log predicted partial hazard):

$$
\text{logit}\, P(\text{event within } t^*) = \alpha + \beta \log(\hat{h}_i)
$$

- Ideal: $\beta = 1.0$
- $\beta > 1$: underfitting (predicted risks too compressed toward the mean)
- $\beta < 1$: overfitting (predicted risks too extreme)

Returns: slope, intercept, slope SE, slope p-value.

### 14.2 Calibration-in-the-large

Comparison of mean predicted risk to observed event rate:

$$
\text{O/E ratio} = \frac{\bar{\hat{h}}}{\bar{y}_{t^*}}
$$

where $\bar{y}_{t^*}$ is the observed proportion with event within $t^*$. Ideal O/E = 1.0.

### 14.3 Decile calibration plot

Subjects are divided into deciles of predicted risk. Mean predicted risk vs observed event rate is plotted per decile with an ideal 45-degree reference line. Deviations from the diagonal indicate systematic miscalibration within risk strata.

**Implementation status:** The decile calibration plot is defined in `calibration.py` but is not currently called in the automated pipeline (`runner.py`). Only `calibration_slope` and `calibration_in_the_large` are invoked at runtime.

---

## 15. Screening Performance Metrics (`screening_metrics.py`)

Diagnostic accuracy of the RBD probability score as a continuous screening test is quantified at fixed percentile thresholds. "Screen positive" is defined as $\hat{r}_i \geq \tau_p$ where $\tau_p = F_R^{-1}(p/100)$.

Binary truth: event within time horizon $t^*$ (subjects censored before $t^*$ without an event are excluded from the denominator — they are not evaluable).

For each (percentile threshold $p$, time horizon $t^*$) pair:

$$
\text{Sensitivity} = \frac{TP}{TP + FN}, \quad
\text{Specificity} = \frac{TN}{TN + FP}
$$

$$
\text{PPV} = \frac{TP}{TP + FP}, \quad
\text{NPV} = \frac{TN}{TN + FN}
$$

**Wilson score 95% CIs** are used for all four proportions (Brown et al., Am Statistician, 2001), which are recommended for small numerators and provide better coverage than Wald intervals near 0 and 1.

Default thresholds: $p \in \{90, 95, 99\}$ (`SCREENING_PERCENTILES`).
Default time horizons: $t^* \in \{5, 10\}$ years (`SCREENING_TIME_HORIZONS`).

---

## 16. Sensitivity Analyses

### 16.1 Reverse-causality lag filter (`data_prep.py::apply_lag_filter()`)

**Pipeline scope:** Applied to `PRIMARY_OUTCOME` only (`runner.py:566`).

To address the possibility that prodromal markers or actigraphy features reflect subclinical disease already present at baseline, events occurring within **2 years** of `wear_time_start` are excluded:

$$
\text{include}_i = \mathbb{1}[\delta_i = 0] + \mathbb{1}[\delta_i = 1 \text{ and } T_i > 2]
$$

The main analysis uses `LAG_YEARS = 2.0`. Results with and without the lag filter are compared.

### 16.2 Threshold stability analysis (`model_rbd.py::fit_rbd_threshold_stability()`)

Model A-i is refitted at three alternative top-percentile cutoffs $p \in \{5\%, 10\%, 15\%\}$ to verify that the HR for High vs Low RBD is not driven by the specific 90th-percentile threshold. For each cutoff, the threshold value is:

$$
\tau_p = F_R^{-1}(1 - p/100)
$$

where $F_R^{-1}$ is the empirical quantile function of the RBD score in the analytic sample.

### 16.3 HES coverage sensitivity (`data_prep.py::build_availability_table()`)

For HES-derived binary prodromal variables, the quality of the "unexposed" label depends on hospital contact frequency. A subject with no HES record for a prodromal condition could be a true negative or an unascertained positive.

The **HES gap** is defined as the number of years between the subject's last pre-baseline HES record and `wear_time_start`. Subjects with gap $> 4$ years (`HES_GAP_THRESHOLD_YEARS`) have potentially sparse HES coverage. The sensitivity analysis restricts to subjects with gap $\leq 4$ years to test whether HR estimates are inflated by misclassified unexposed subjects.

### 16.4 Age-stratified analysis (`cox_config.py::AGE_STRATA`)

**Pipeline scope:** Applied to `PRIMARY_OUTCOME` only (`runner.py:419`).

All models are replicated within two age strata defined at recruitment:

| Stratum | Age range | Rationale |
|---|---|---|
| Younger | ≤60 years | Lower background PD risk; test whether RBD association is age-independent |
| Older | >60 years | Higher background incidence; larger absolute risks |

A three-group split was considered but rejected because the youngest tertile (<50 years) would have insufficient events, reducing statistical power without adding scientific value.

---

## 17. Analysis Parameters Summary

| Parameter | Value | Source |
|---|---|---|
| Primary outcome | `outcome_1a_pd_only` | `cox_config.py::PRIMARY_OUTCOME` |
| Primary RBD method | `percentile_3g` | `cox_config.py::PRIMARY_METHOD` |
| Minimum events for model | 5 | `cox_config.py::MIN_EVENTS_FOR_MODEL` |
| Minimum binary prevalence | 30 | `cox_config.py::MIN_PREVALENCE_FOR_BINARY` |
| Lag filter (years) | 2.0 | `cox_config.py::LAG_YEARS` |
| Spline degrees of freedom | 4 | `cox_config.py::SPLINE_DF` |
| Ridge penaliser (L2) | 0.01 | `cox_config.py::RIDGE_PENALIZER` |
| CIF timepoints (years) | 5, 10 | `cox_config.py::ABSOLUTE_RISK_TIMEPOINTS` |
| Bootstrap resamples | 1000 | `cox_config.py::BOOTSTRAP_N` |
| Bootstrap seed | 42 | `cox_config.py::BOOTSTRAP_SEED` |
| HES gap threshold (years) | 4.0 | `cox_config.py::HES_GAP_THRESHOLD_YEARS` |
| Screening percentile thresholds | 90, 95, 99 | `cox_config.py::SCREENING_PERCENTILES` |
| Screening time horizons (years) | 5, 10 | `cox_config.py::SCREENING_TIME_HORIZONS` |
| Age strata (years) | (0, 60], (60, 200] | `cox_config.py::AGE_STRATA` |
| Spline HR grid points | 300 | `rbd_spline_analysis.py::N_GRID` |
| FDR method | Benjamini-Hochberg | `diagnostics.py::apply_fdr()` |
| Variance estimator | Robust (Lin-Wei sandwich) | `CoxPHFitter(robust=True)` |
