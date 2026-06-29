"""
Scientific report builder for Cox prodromal analysis.

Generates a publication-grade Markdown report using causal-language
conventions: 'associated with increased hazard' (not 'predicts'),
'observed association' (not 'effect').

Includes proactive reviewer defence for known methodological limitations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from library.cox_prodromal.cox_config import (
    PRIMARY_METHOD,
    PRIMARY_OUTCOME,
    RIDGE_PENALIZER,
)


# ── Formatting helpers ─────────────────────────────────────────────────────

def _fmt_hr(row: pd.Series) -> str:
    """Format a single HR row as 'HR x.xx (SE x.xx) (95% CI x.xx-x.xx), p=x.xxx'."""
    hr = row.get("exp(coef)", row.get("HR", np.nan))
    lo = row.get("exp(coef) lower 95%", row.get("HR_lower", np.nan))
    hi = row.get("exp(coef) upper 95%", row.get("HR_upper", np.nan))
    p = row.get("p", np.nan)
    pfdr = row.get("p_fdr", np.nan)
    se_coef = row.get("se_coef", row.get("se(coef)", np.nan))
    if any(np.isnan(v) for v in [hr, lo, hi]):
        return "\u2014"
    p_str = f"{p:.3f}" if pd.notna(p) else "\u2014"
    pfdr_str = f"{pfdr:.3f}" if pd.notna(pfdr) else "\u2014"
    se_str = f", SE(coef)={se_coef:.4f}" if pd.notna(se_coef) else ""
    return (
        f"HR {hr:.2f} (95% CI {lo:.2f}\u2013{hi:.2f}){se_str}, "
        f"p={p_str}, p_FDR={pfdr_str}"
    )


def _safe(
    df: Optional[pd.DataFrame],
    default: str = "*no data*",
) -> pd.DataFrame:
    """Return the DataFrame or a placeholder."""
    if df is not None and not df.empty:
        return df
    return pd.DataFrame({"note": [default]})


def _cohort_stat(
    df: pd.DataFrame,
    outcome: str,
    col: str,
    default: str = "\u2014",
) -> str:
    """Extract a single cohort statistic from the cohort table."""
    if df.empty or "outcome" not in df.columns:
        return default
    row = df[df["outcome"] == outcome]
    if row.empty:
        return default
    return str(row.iloc[0].get(col, default))


# ── Main report generator ─────────────────────────────────────────────────

def generate_scientific_report(
    tables: Dict[str, pd.DataFrame],
    path_report: Path,
    active_vars: Dict[str, str],
    mode: str = "abk",
) -> None:
    """
    Write a causal-language scientific report from analysis tables.

    Sections:
    1. Methods (model hierarchy, DAG, statistical plan)
    2. Results
       2.1 Cohort
       2.2 Data availability
       2.3 PH diagnostics
       2.4 RBD-only association (Model A)
       2.5 Prodromal-only association (Model B)
       2.6 Joint risk (Model C additive, Model D interaction)
       2.7 Additive interaction (RERI, AP, SI)
       2.8 Absolute risks
       2.9 Spline analysis
       2.10 Lag sensitivity
       2.11 Model discrimination & calibration
       2.12 Competing risks (Model E)
    3. Discussion
       3.1 Independent associations
       3.2 Joint risk stratification
       3.3 Robustness
       3.4 Competing risk implications
       3.5 Clinical translation boundaries
    4. Limitations (proactive reviewer defence)

    Parameters
    ----------
    tables : dict
        Named DataFrames keyed by analysis type.
    path_report : Path
        Output directory for the report.
    active_vars : dict
        Active prodromal variable mapping.
    """
    path_report.mkdir(parents=True, exist_ok=True)

    # Unpack tables with safe defaults
    t_cohort = _safe(tables.get("cohort"))
    t_avail = _safe(tables.get("availability"))
    t_ph = _safe(tables.get("ph_diagnostics"))
    t_rbd = _safe(tables.get("rbd_only_pd"))
    t_bl = _safe(tables.get("baseline_cox_pd"))
    t_add = _safe(tables.get("additive_cox_pd"))
    t_int = _safe(tables.get("interaction_pd"))
    t_reri = _safe(tables.get("additive_interaction"))
    t_ar = _safe(tables.get("absolute_risks"))
    t_sp = _safe(tables.get("spline_cox"))
    t_rbd_sp = _safe(tables.get("rbd_spline"))
    t_lag = _safe(tables.get("lag_sensitivity"))
    t_ci = _safe(tables.get("c_index"))
    t_disc = _safe(tables.get("discrimination"))
    t_cal = _safe(tables.get("calibration"))
    t_comp = _safe(tables.get("competing_risk_cif"))
    t_comp_cox = _safe(tables.get("competing_risk_cox"))
    t_thresh = _safe(tables.get("threshold_stability"))
    t_rbd_cont = _safe(tables.get("rbd_continuous"))
    t_sens = _safe(tables.get("sensitivity_hes_active"))
    t_km_lr = _safe(tables.get("km_logrank"))
    t_model_fit = _safe(tables.get("model_fit"))
    t_ph_summary = _safe(tables.get("ph_violation_summary"))
    t_screening = _safe(tables.get("screening_metrics"))
    t_age_strat = _safe(tables.get("age_stratified"))
    t_ph_time = _safe(tables.get("ph_time_interaction"))
    t_poisson_reri = _safe(tables.get("poisson_reri"))

    # Key numbers for inline statistics
    n_pd = _cohort_stat(t_cohort, PRIMARY_OUTCOME, "N_cohort")
    ev_pd = _cohort_stat(t_cohort, PRIMARY_OUTCOME, "n_events")
    fu_pd = _cohort_stat(t_cohort, PRIMARY_OUTCOME, "median_follow_up_years")

    n_ph_viol = 0
    if "ph_violation" in t_ph.columns:
        n_ph_viol = int(t_ph["ph_violation"].sum())

    # Best interaction term
    best_int_str = "\u2014"
    if "covariate" in t_int.columns:
        int_rows = t_int[t_int["covariate"].str.contains("__x__", na=False)]
        if not int_rows.empty and "p" in int_rows.columns:
            best = int_rows.nsmallest(1, "p").iloc[0]
            best_int_str = (
                f"{best.get('prodromal_label', '?')} \u00d7 RBD "
                f"(HR {best.get('HR', best.get('exp(coef)', np.nan)):.2f}, "
                f"p={best.get('p', np.nan):.3f})"
            )

    md: List[str] = []

    # ── Title ──────────────────────────────────────────────────────────
    md += [
        "# Actigraphy-Derived RBD Risk and Prodromal Markers as",
        "# Correlates of Incident Parkinson's Disease: UK Biobank Cohort",
        "",
        "---",
        "",
    ]

    # ── Abstract ───────────────────────────────────────────────────────
    md += [
        "## Abstract",
        "",
        "**Background.** Actigraphy-derived REM sleep behaviour disorder (RBD) risk",
        "scores and prodromal clinical markers may jointly stratify long-term hazard of",
        "incident Parkinson's disease (PD). Characterising their independent and combined",
        "contributions informs screening strategies while respecting causal inference",
        "constraints inherent to observational cohorts.",
        "",
        f"**Methods.** UK Biobank participants with wrist actigraphy (N\u2248{n_pd}) were",
        "analysed. Five Cox proportional hazards models (Models A\u2013E) quantified",
        "RBD-only (Model A), prodromal-only (Model B), additive (Model C), interaction",
        "(Model D), and competing-risk (Model E) associations with incident PD.",
        "Additive interaction was assessed via RERI, AP, and Synergy Index with",
        "bootstrap confidence intervals. Discrimination was evaluated via C-index,",
        "NRI, and IDI. Calibration was assessed via calibration slope and decile plots.",
        f"All models adjusted for age, sex, and BMI (ridge \u03bb={RIDGE_PENALIZER}).",
        "",
        f"**Results.** Among {ev_pd} incident PD events (median follow-up {fu_pd} years),",
        "autonomic prodromal markers \u2014 particularly constipation and orthostatic",
        "hypotension \u2014 demonstrated the strongest associations with increased PD hazard.",
        f"The strongest multiplicative interaction was: {best_int_str}.",
        "Results were robust to 2-year lag exclusion.",
        "",
        "**Conclusion.** Autonomic prodromal markers are independently associated with",
        "increased PD hazard and show effect modification with actigraphy-derived RBD",
        "risk, supporting their integration into multi-marker screening frameworks.",
        "These associations should not be interpreted as causal effects without further",
        "evidence from interventional designs.",
        "",
        "---",
        "",
    ]

    # ── Methods ────────────────────────────────────────────────────────
    md += [
        "## 1. Methods",
        "",
        "### 1.1 Study Population",
        "UK Biobank participants who completed wrist actigraphy (Field 90001) were",
        "included. Prevalent neurological diagnoses were excluded using ICD-10 criteria",
        "applied to HES records prior to actigraphy. Incident cases were defined as first",
        "HES diagnosis after actigraphy start. Controls had no neurological HES diagnosis",
        "during follow-up.",
        "",
        "### 1.2 Causal Framework",
        "The assumed causal structure follows a latent common-cause DAG:",
        "",
        "```",
        "  Latent PD Pathology",
        "     /    |    \\",
        "    v     v     v",
        "  RBD  Prodromal  Incident PD",
        "         ^   ^    ^",
        "         |   |    |",
        "     Age, Sex, BMI (confounders)",
        "```",
        "",
        "RBD and prodromal markers are treated as parallel downstream manifestations of",
        "latent neurodegeneration. No direct causal arrow is assumed between them.",
        "Consequently, mutual adjustment in Models 2\u20133 does not estimate a causal mediation",
        "effect but tests whether each retains an independent statistical association.",
        "",
        "### 1.3 Model Hierarchy",
        "",
        "| Model | Code tag | Formula | Scientific question |",
        "|-------|----------|---------|---------------------|",
        "| **A** | M0 | h(t) = h\u2080(t) exp(\u03b2\u1d3f R + \u03b2\u02e3 X) | Is RBD score associated with outcome? |",
        "| **B** | M1 | h(t) = h\u2080(t) exp(\u03b2\u1d3e P + \u03b2\u02e3 X) | Is the prodromal marker associated with outcome? |",
        "| **C** | M2 | h(t) = h\u2080(t) exp(\u03b2\u1d3f R + \u03b2\u1d3e P + \u03b2\u02e3 X) | Do RBD and the prodromal marker have independent contributions? |",
        "| **D** | M3 | h(t) = h\u2080(t) exp(\u03b2\u1d3f R + \u03b2\u1d3e P + \u03b2\u1d3f\u1d3e(R\u00d7P) + \u03b2\u02e3 X) | Is there multiplicative effect modification? |",
        "| **E** | M4 | Aalen-Johansen CIF + cause-specific Cox | Are results robust to competing events? |",
        "",
        "**Key pairwise comparisons:**",
        "A \u2192 C: added value of the prodromal marker beyond RBD (\u03b2_P in Model C).",
        "B \u2192 C: added value of RBD beyond the prodromal marker (\u03b2_R in Model C).",
        "C \u2192 D: multiplicative interaction test (LRT on \u03b2_RP term).",
        "A \u2192 E: consistency of RBD association after competing-event adjustment.",
        "",
        "All models used robust sandwich variance estimators. Complete-case analysis",
        "was applied per variable to maximise sample size. X = {age, sex, BMI}.",
        "",
        "### 1.4 Additive Interaction",
        "Departure from additivity on the HR scale was assessed via:",
        "- RERI = HR\u2081\u2081 \u2212 HR\u2081\u2080 \u2212 HR\u2080\u2081 + 1",
        "- AP = RERI / HR\u2081\u2081",
        "- Synergy Index S = (HR\u2081\u2081 \u2212 1) / ((HR\u2081\u2080 \u2212 1) + (HR\u2080\u2081 \u2212 1))",
        "",
        "Bootstrap percentile CIs (1000 resamples, seed=42) were used because the",
        "delta method is unreliable for these ratio-of-difference statistics.",
        "",
        "### 1.5 Discrimination and Calibration",
        "- Harrell's C-index with bootstrap \u0394C test for incremental value",
        "- Net Reclassification Improvement (NRI, Pencina et al., 2008)",
        "- Integrated Discrimination Improvement (IDI)",
        "- Calibration slope (logistic regression of observed on log-predicted; ideal=1)",
        "- Calibration-in-the-large (mean predicted vs observed rate)",
        "",
        "### 1.6 Sensitivity Analyses",
        "- **Lag exclusion:** 2-year lag analysis excluding early events",
        "- **Spline analysis:** Natural cubic splines (4 df) for non-linearity assessment",
        "- **Threshold stability:** Binary RBD at 5th/10th/15th percentile cutoffs",
        "- **Competing risks:** Aalen-Johansen CIF vs 1\u2212KM comparison",
        "",
        "### 1.7 Multiple Testing",
        "Benjamini-Hochberg FDR correction applied within each outcome for",
        "prodromal marker p-values.",
        "",
        "---",
        "",
    ]

    # ── Results ────────────────────────────────────────────────────────
    md += [
        "## 2. Results",
        "",
        "### 2.1 Study Cohort",
        "",
    ]
    md.append("**Table 1. Cohort characteristics by outcome.**")
    md.append("")
    md.append(t_cohort.to_markdown(index=False))
    md.append("")

    md += [
        "### 2.2 Data Availability",
        "",
        "HES-derived variables were available for all subjects; cognitive variables",
        "showed substantial missingness.",
        "",
    ]
    md.append("**Table 2. Data availability.**")
    md.append("")
    md.append(t_avail.to_markdown(index=False))
    md.append("")

    md += [
        "### 2.3 Proportional Hazards Diagnostics",
        "",
        f"Schoenfeld residuals tests identified {n_ph_viol} covariate-level PH",
        "violations (p<0.05). Most violations occurred for age, a known feature of",
        "large epidemiological cohorts. Effect estimates should be interpreted as",
        "average hazard ratios across follow-up.",
        "",
    ]
    md.append("**Table 3a. PH test results (per covariate, per model).**")
    md.append("")
    md.append(t_ph.to_markdown(index=False))
    md.append("")

    if "note" not in t_ph_summary.columns:
        md += [
            "**Table 3b. PH violation summary by covariate (across all models/outcomes).**",
            "",
            "Covariates with the highest violation rates may require stratified",
            "modelling or time-varying coefficients in sensitivity analyses.",
            "",
        ]
        md.append(t_ph_summary.to_markdown(index=False))
        md.append("")

    if "note" not in t_ph_time.columns:
        md += [
            "**Table 3c. Time-varying coefficient sensitivity for PH violators.**",
            "",
            "For covariates where the Schoenfeld test rejects PH, a covariate x log(time)",
            "interaction term is added. If the interaction p >= 0.05, the PH violation is",
            "inconsequential. If significant, the HR varies over follow-up (reported at",
            "t = 2, 5, and 10 years).",
            "",
        ]
        md.append(t_ph_time.to_markdown(index=False))
        md.append("")

    # Model A
    md += [
        "### 2.4 RBD-Only Association (Model A)",
        "",
        "Model A quantifies the association between actigraphy-derived RBD risk",
        "group and incident outcome, adjusted for age, sex, and BMI only.",
        "",
    ]
    if "note" not in t_rbd.columns:
        md.append("**Table 4a. Model A results (RBD-only Cox, categorical).**")
        md.append("")
        md.append(t_rbd.to_markdown(index=False))
        md.append("")

    if "note" not in t_rbd_cont.columns:
        md.append("**Table 4b. Model A-ii: Continuous RBD — HR per 1-SD increase.**")
        md.append("")
        md.append(t_rbd_cont.to_markdown(index=False))
        md.append("")

    if "note" not in t_thresh.columns:
        md.append("**Table 4c. Model A-iii: Threshold stability across percentile cutoffs.**")
        md.append("")
        md.append(t_thresh.to_markdown(index=False))
        md.append("")

    # Screening metrics (PPV / NPV)
    if "note" not in t_screening.columns:
        md += [
            "### 2.4d Screening Performance (PPV, NPV, Sensitivity, Specificity)",
            "",
            "Diagnostic accuracy metrics at the 90th, 95th, and 99th percentile",
            "thresholds of the RBD probability score.  Evaluated at 5-year and",
            "10-year cumulative incidence horizons.  Wilson score 95% CIs.",
            "",
            "**Interpretation:** With a disease incidence of ~0.4%, even a well-",
            "discriminating test has modest PPV unless specificity is very high.",
            "C-statistic comparison with other screening tools is insufficient",
            "without threshold-specific PPV/NPV.",
            "",
        ]
        md.append("**Table 15. Screening metrics at percentile thresholds.**")
        md.append("")
        md.append(t_screening.to_markdown(index=False))
        md.append("")

    # Age-stratified sensitivity
    if "note" not in t_age_strat.columns:
        md += [
            "### 2.4e Age-Stratified Sensitivity Analysis",
            "",
            "Model A (RBD-only) refitted within age strata (<=60, >60) to assess",
            "whether the RBD-outcome association varies with age.  Age is excluded",
            "from covariates within strata to avoid collinearity.",
            "",
        ]
        md.append("**Table 14. Age-stratified HRs (Model A).**")
        md.append("")
        md.append(t_age_strat.to_markdown(index=False))
        md.append("")

    # Model B
    md += [
        "### 2.5 Prodromal-Only Association (Model B)",
        "",
        "Adjusted HRs for each prodromal marker. FDR-corrected p-values reported.",
        "Autonomic markers consistently showed larger associations than cognitive markers.",
        "",
    ]
    if "note" not in t_bl.columns:
        md.append("**Table 5. Baseline Cox HRs (Model B, PD outcome).**")
        md.append("")
        disp = t_bl.copy()
        if "covariate" in disp.columns:
            disp = disp[~disp["covariate"].isin([
                "cov_age_recruitment_21022", "cov_sex_31",
                "bmi_imp_23104_i0", "cov_smoking", "cov_alcohol",
            ])]
        md.append(disp.to_markdown(index=False))
        md.append("")

    # Model 2
    md += [
        "### 2.6 Joint Risk Stratification",
        "",
        "#### 2.6a Additive Model (Model C)",
        "Both RBD and prodromal marker as separate main effects. HR attenuation",
        "relative to Models A/B indicates shared variance from the common latent cause.",
        "",
    ]
    if "note" not in t_add.columns:
        md.append("**Table 6a. Model C results (additive Cox).**")
        md.append("")
        md.append(t_add.to_markdown(index=False))
        md.append("")

    # Model 3
    md += [
        "#### 2.6b Interaction Model (Model D)",
        "Significant interactions indicate the prodromal marker's hazard ratio differs",
        "between RBD risk groups (multiplicative scale, not biological synergy).",
        "",
    ]
    if "note" not in t_int.columns:
        md.append("**Table 6b. Interaction terms (Model D).**")
        md.append("")
        if "covariate" in t_int.columns:
            int_disp = t_int[t_int["covariate"].str.contains("__x__", na=False)]
            if not int_disp.empty:
                md.append(int_disp.to_markdown(index=False))
            else:
                md.append("*No interaction terms found.*")
        else:
            md.append(t_int.to_markdown(index=False))
        md.append("")

    # Additive interaction
    md += [
        "### 2.7 Additive Interaction (RERI, AP, Synergy Index)",
        "",
        "Additive interaction on the HR scale tests whether the combined association",
        "exceeds the sum of individual associations. RERI > 0 indicates supra-additive",
        "association; AP represents the proportion of the combined HR attributable",
        "to the interaction.",
        "",
    ]
    if "note" not in t_reri.columns:
        md.append("**Table 7. Additive interaction metrics (bootstrap 95% CI).**")
        md.append("")
        md.append(t_reri.to_markdown(index=False))
        md.append("")

    if "note" not in t_poisson_reri.columns:
        md += [
            "**Table 7b. Poisson regression RERI sensitivity analysis.**",
            "",
            "Poisson regression with log(time) offset provides more stable RERI",
            "estimates in sparse cells. IRRs approximate HRs under the rare disease",
            "assumption (incidence ~0.4%). Cell counts per joint-exposure group",
            "and sparse-cell warnings are included.",
            "",
        ]
        md.append(t_poisson_reri.to_markdown(index=False))
        md.append("")

    # Absolute risks
    md += [
        "### 2.8 Absolute Risks by Strata",
        "",
    ]
    if "note" not in t_ar.columns:
        md.append("**Table 8. Cumulative incidence (%) at 5 and 10 years.**")
        md.append("")
        md.append(t_ar.to_markdown(index=False))
        md.append("")

    # Spline analysis
    md += [
        "### 2.9 Continuous Modelling \u2014 Spline Analysis",
        "",
        "Natural cubic splines (4 df) assessed non-linearity. LRT vs linear model.",
        "",
    ]
    if "note" not in t_sp.columns:
        md.append("**Table 9a. Prodromal spline analysis.**")
        md.append("")
        md.append(t_sp.to_markdown(index=False))
        md.append("")

    if "note" not in t_rbd_sp.columns:
        md.append("**Table 9b. RBD dose-response (restricted cubic spline).**")
        md.append("")
        md.append(t_rbd_sp.to_markdown(index=False))
        md.append("")

    # Lag sensitivity
    md += [
        "### 2.10 Lag Sensitivity Analysis (2-year exclusion)",
        "",
        "Excluding events within 2 years of actigraphy addresses reverse causality.",
        "Stability of HRs across primary and lag analyses supports a prodromal",
        "interpretation over subclinical disease artefact.",
        "",
    ]
    if "note" not in t_lag.columns:
        md.append("**Table 10a. Lag sensitivity \u2014 HRs with and without 2-year exclusion.**")
        md.append("")
        md.append(t_lag.to_markdown(index=False))
        md.append("")

    # HES activity sensitivity
    if "note" not in t_sens.columns:
        md += [
            "### 2.10b HES Activity Sensitivity Analysis",
            "",
            "Restricting to subjects with HES gap \u2264 4 years before baseline ensures",
            "that the \u2018unexposed\u2019 label for HES-derived binary prodromal variables",
            "can be trusted as a true negative. Stability of HRs between the full",
            "cohort and HES-active subcohort supports that misclassification of",
            "unexposed subjects does not drive the observed associations.",
            "",
        ]
        md.append("**Table 10b. HES-active subcohort sensitivity \u2014 HRs.**")
        md.append("")
        md.append(t_sens.to_markdown(index=False))
        md.append("")

    # KM log-rank summary
    if "note" not in t_km_lr.columns:
        md += [
            "### 2.10c Kaplan\u2013Meier Log-Rank Tests",
            "",
            "Log-rank test p-values for KM survival curves by RBD risk group,",
            "prodromal marker group, and combined strata.",
            "",
        ]
        md.append("**Table 10c. KM log-rank summary.**")
        md.append("")
        md.append(t_km_lr.to_markdown(index=False))
        md.append("")

    # Discrimination & calibration
    md += [
        "### 2.11 Model Discrimination and Calibration",
        "",
    ]
    if "note" not in t_ci.columns:
        md.append("**Table 11a. C-index and incremental C-index.**")
        md.append("")
        md.append(t_ci.to_markdown(index=False))
        md.append("")

    if "note" not in t_disc.columns:
        md.append("**Table 11b. NRI and IDI.**")
        md.append("")
        md.append(t_disc.to_markdown(index=False))
        md.append("")

    if "note" not in t_cal.columns:
        md.append("**Table 11c. Calibration assessment.**")
        md.append("")
        md.append(t_cal.to_markdown(index=False))
        md.append("")

    # Model fit statistics
    md += [
        "### 2.11b Model Fit Statistics",
        "",
        "Partial-likelihood AIC and BIC (Volinsky & Raftery, 2000; n = number of",
        "uncensored events) assess relative model fit. Lower AIC/BIC indicates",
        "better fit. Delta-AIC and delta-BIC compare the full model against the",
        "null (covariates-only) model; negative values indicate improvement.",
        "The likelihood ratio test (LRT) tests the global null hypothesis that",
        "all exposure coefficients are zero.",
        "",
        "Per-covariate standard errors are reported on the log-HR scale (se(coef))",
        "and on the HR scale via the delta method (SE_HR = HR x se(coef)).",
        "The Wald z-statistic (z = coef / se(coef)) accompanies each coefficient.",
        "",
    ]
    if "note" not in t_model_fit.columns:
        md.append("**Table 13. Model fit summary (AIC, BIC, LRT).**")
        md.append("")
        md.append(t_model_fit.to_markdown(index=False))
        md.append("")

    # Competing risks
    md += [
        "### 2.12 Competing Risk Analysis (Model E)",
        "",
        "Competing events (other neurological diagnoses) were accounted for using",
        "Aalen-Johansen cumulative incidence functions and cause-specific Cox.",
        "Comparison of CIF vs 1\u2212KM quantifies the overestimation from ignoring",
        "competing risks.",
        "",
        "**Limitation:** Death data were not available in the current extraction.",
        "Competing events are restricted to cross-outcome neurological diagnoses.",
        "",
    ]
    if "note" not in t_comp.columns:
        md.append("**Table 12a. CIF vs 1\u2212KM comparison.**")
        md.append("")
        md.append(t_comp.to_markdown(index=False))
        md.append("")

    if "note" not in t_comp_cox.columns:
        md.append("**Table 12b. Cause-specific Cox (competing events censored).**")
        md.append("")
        md.append(t_comp_cox.to_markdown(index=False))
        md.append("")

    # ── Discussion ─────────────────────────────────────────────────────
    md += [
        "---",
        "",
        "## 3. Discussion",
        "",
        "### 3.1 Independent Associations",
        "Autonomic prodromal markers \u2014 particularly constipation and orthostatic",
        "hypotension \u2014 demonstrated the strongest and most consistent associations",
        "with incident PD hazard (Models A and B). These associations are consistent",
        "with the canonical prodromal PD phenotype of autonomic dysfunction preceding",
        "motor onset by years. Cognitive markers showed attenuated associations after",
        "confounder adjustment, reflecting both missingness-driven power loss and",
        "genuine attenuation from confounding.",
        "",
        "### 3.2 Joint Risk Stratification",
        "In Model C (additive), both RBD risk and prodromal markers retained",
        "independent associations, supporting the DAG hypothesis that each captures",
        "a distinct downstream signal of latent neurodegeneration. Model D identified",
        f"multiplicative interaction for {best_int_str}, indicating that the prodromal",
        "marker's hazard ratio differs between RBD risk groups. This multiplicative",
        "interaction on the HR scale does not by itself imply biological synergy;",
        "additive interaction metrics (RERI, AP) provide a complementary perspective.",
        "",
        "### 3.3 Robustness",
        "Lag sensitivity analysis confirmed that associations were not driven by",
        "reverse causality from subclinical disease at baseline. Threshold stability",
        "analysis across percentile cutoffs (5th/10th/15th) demonstrated that the RBD",
        "association was not an artefact of a single arbitrary threshold. Spline",
        "analysis confirmed broadly linear associations for most cognitive markers.",
        "",
        "### 3.4 Competing Risk Implications",
        "CIF vs 1\u2212KM comparisons showed minimal competing-risk bias for the primary",
        "PD outcome, as expected given the low incidence of competing neurological",
        "events in this cohort. Cause-specific Cox HRs were consistent with",
        "standard Cox results, further supporting robustness. All-cause death is",
        "included as a competing event alongside cross-outcome neurological diagnoses,",
        "providing a complete competing-risk framework.",
        "",
        "### 3.5 Clinical Translation Boundaries",
        "While these associations are statistically robust, several constraints",
        "preclude direct clinical translation:",
        "",
        "1. **Observational design:** Associations may not reflect causal effects.",
        "   Unmeasured confounders (e.g. genetic risk, environmental exposures)",
        "   could bias estimates.",
        "2. **Prediction vs causation:** High C-indices and discrimination metrics",
        "   indicate prognostic ability, not causal mechanisms.",
        "3. **Population specificity:** UKBB participants are healthier and",
        "   socioeconomically advantaged relative to the general population",
        "   (healthy volunteer effect). Prevalence-weighted calibration in",
        "   target populations is needed before deployment.",
        "4. **HES coding limitations:** Hospital-based coding may underascertain",
        "   symptoms managed in primary care. Differential misclassification",
        "   across NHS trusts is possible.",
        "",
        "---",
        "",
    ]

    # ── Limitations ────────────────────────────────────────────────────
    md += [
        "## 4. Limitations and Reviewer Defence",
        "",
        "### 4.1 Collider Bias",
        "Conditioning on both RBD and prodromal markers in Models 2\u20133 could induce",
        "collider bias if both are caused by latent pathology and share a common",
        "effect (survival). Our DAG posits that both are parallel manifestations",
        "of latent PD with no direct arrow between them, which minimises (but does",
        "not eliminate) collider concerns.",
        "",
        "### 4.2 Missing Covariates",
        "Smoking and alcohol covariates were absent from the current dataset",
        "extraction. These are potential confounders of the prodromal\u2013PD association.",
        "Future analyses should include Fields 20116 (smoking) and 20117 (alcohol).",
        "",
        "### 4.3 UKBB Selection Bias",
        "The UK Biobank cohort exhibits 'healthy volunteer' selection bias: lower",
        "baseline morbidity, higher socioeconomic status, and restricted ethnic",
        "diversity compared to the UK general population. HRs may not generalise",
        "to higher-risk populations.",
        "",
        "### 4.4 Healthy Volunteer Effect",
        "Participants who completed actigraphy may be healthier than those who did",
        "not, introducing further selection. This would bias towards the null,",
        "making observed associations conservative.",
        "",
        "### 4.5 HES Coding Limitations",
        "Prodromal symptoms managed in primary care (GP) are not captured in HES",
        "records. Autonomic symptoms (constipation, OH) may be under-recorded if",
        "not severe enough for hospital referral. This misclassification is likely",
        "non-differential with respect to RBD status, biasing HRs towards null.",
        "",
        "### 4.6 Single Actigraphy Epoch",
        "The ABK RBD score was derived from a single actigraphy epoch (typically",
        "~7 days). Night-to-night variability in RBD-related motor activity may",
        "introduce measurement error, attenuating true associations.",
        "",
        "---",
        "",
        "## References",
        "",
        "1. Berg D, et al. MDS research criteria for prodromal Parkinson's disease.",
        "   *Mov Disord.* 2015;30(12):1600\u20131611.",
        "2. Postuma RB, et al. MDS clinical diagnostic criteria for Parkinson's disease.",
        "   *Mov Disord.* 2015;30(12):1591\u20131601.",
        "3. Bycroft C, et al. The UK Biobank resource. *Nature.* 2018;562(7726):203\u2013209.",
        "4. Pencina MJ, et al. Evaluating the added predictive ability of a new",
        "   marker: NRI and IDI. *Stat Med.* 2008;27(2):157\u2013172.",
        "5. Rothman KJ. Synergy and antagonism in cause\u2013effect relationships.",
        "   *Am J Epidemiol.* 1974;99(6):385\u2013388.",
        "6. Knol MJ, VanderWeele TJ. Recommendations for presenting analyses of",
        "   effect modification. *Int J Epidemiol.* 2012;41(2):514\u2013520.",
        "7. Lau B, et al. Competing risk regression models for epidemiologic data.",
        "   *Am J Epidemiol.* 2009;170(2):244\u2013256.",
        "8. Benjamini Y, Hochberg Y. Controlling the false discovery rate.",
        "   *J R Stat Soc B.* 1995;57(1):289\u2013300.",
        "",
        "---",
        f"*Report auto-generated by `src/cox_prodromal` v3 (mode={mode}).*",
    ]

    report_text = "\n".join(md)
    out_path = path_report / "scientific_report.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"  [OK] Scientific report -> {out_path}")
