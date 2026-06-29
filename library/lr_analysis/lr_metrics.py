"""
Likelihood ratio metrics for the actigraphy RBD z-score.

Provides:
- LR+/LR- at a given z-score threshold (2x2 table, log-normal CI)
- LR profile across a threshold grid
- Youden-optimal threshold
- Unadjusted and adjusted logistic OR (continuous z-score predictor)
- Sex-stratified LR at a given threshold
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_curve
from statsmodels.formula.api import logit as sm_logit

from library.lr_analysis.config import (
    CONFOUNDERS,
    FEMALE_CODE,
    MALE_CODE,
    MIN_CELL_COUNT,
    PRODROMAL_VIABLE,
    RBD_ZSCORE_COL,
    SEX_COL,
    ZSCORE_THRESHOLD_GRID,
)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LRResult:
    """LR+/LR- at a single threshold, with 95% CIs and 2x2 cells."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    sensitivity: float
    sensitivity_ci: tuple[float, float]
    specificity: float
    specificity_ci: tuple[float, float]
    lr_pos: float
    lr_pos_ci: tuple[float, float]
    lr_neg: float
    lr_neg_ci: tuple[float, float]
    n_cases: int
    n_controls: int
    stratum: str = "overall"  # "overall", "male", "female"
    stable: bool = True       # False if any cell < MIN_CELL_COUNT

    def to_dict(self) -> dict[str, Any]:
        """Return as flat dict, expanding CI tuples."""
        d = asdict(self)
        d["sensitivity_lci"] = self.sensitivity_ci[0]
        d["sensitivity_uci"] = self.sensitivity_ci[1]
        d["specificity_lci"] = self.specificity_ci[0]
        d["specificity_uci"] = self.specificity_ci[1]
        d["lr_pos_lci"] = self.lr_pos_ci[0]
        d["lr_pos_uci"] = self.lr_pos_ci[1]
        d["lr_neg_lci"] = self.lr_neg_ci[0]
        d["lr_neg_uci"] = self.lr_neg_ci[1]
        del d["sensitivity_ci"], d["specificity_ci"]
        del d["lr_pos_ci"], d["lr_neg_ci"]
        return d


@dataclass(frozen=True)
class LogisticORResult:
    """OR per 1 SD increase in RBD z-score from logistic regression."""

    or_estimate: float
    or_lci: float
    or_uci: float
    p_value: float
    model_type: str   # "unadjusted" or "adjusted"
    n: int
    n_cases: int
    converged: bool


@dataclass(frozen=True)
class EmpiricalMarkerLR:
    """Empirically computed LR for one prodromal marker."""

    col: str
    label: str
    lr_pos: float
    lr_pos_ci: tuple[float, float]
    lr_neg: float
    lr_neg_ci: tuple[float, float]
    tp: int
    fp: int
    fn: int
    tn: int
    stable: bool


@dataclass(frozen=True)
class RBDInteractionResult:
    """RBD × predictor interaction test (LRT)."""

    variable: str
    label: str
    cohort_name: str
    reduced_llf: float          # Log-likelihood of reduced model
    full_llf: float             # Log-likelihood of full model
    lrt_stat: float             # -2(llf_reduced - llf_full)
    lrt_df: int                 # Degrees of freedom (always 2 for RBD Low/Mid/High)
    lrt_p: float                # Chi-squared test p-value
    interaction_ors: dict[str, float]      # {"Mid": or_mid, "High": or_high}
    interaction_lcis: dict[str, float]
    interaction_ucis: dict[str, float]
    interaction_ps: dict[str, float]
    main_g_or: float            # OR of main effect of G in full model
    main_g_lci: float
    main_g_uci: float
    main_g_p: float
    n_total: int
    n_cases: int
    n_controls: int
    converged_reduced: bool
    converged_full: bool

    def to_dict(self) -> dict[str, Any]:
        """Return as flat dict for CSV export."""
        return {
            "variable": self.variable,
            "label": self.label,
            "cohort": self.cohort_name,
            "lrt_stat": round(self.lrt_stat, 6),
            "lrt_df": self.lrt_df,
            "lrt_p": round(self.lrt_p, 6),
            "interaction_or_Mid": round(self.interaction_ors.get("Mid", float("nan")), 6),
            "interaction_lci_Mid": round(self.interaction_lcis.get("Mid", float("nan")), 6),
            "interaction_uci_Mid": round(self.interaction_ucis.get("Mid", float("nan")), 6),
            "interaction_p_Mid": round(self.interaction_ps.get("Mid", float("nan")), 6),
            "interaction_or_High": round(self.interaction_ors.get("High", float("nan")), 6),
            "interaction_lci_High": round(self.interaction_lcis.get("High", float("nan")), 6),
            "interaction_uci_High": round(self.interaction_ucis.get("High", float("nan")), 6),
            "interaction_p_High": round(self.interaction_ps.get("High", float("nan")), 6),
            "main_g_or": round(self.main_g_or, 6),
            "main_g_lci": round(self.main_g_lci, 6),
            "main_g_uci": round(self.main_g_uci, 6),
            "main_g_p": round(self.main_g_p, 6),
            "n_total": self.n_total,
            "n_cases": self.n_cases,
            "n_controls": self.n_controls,
            "converged_reduced": self.converged_reduced,
            "converged_full": self.converged_full,
        }


# ── CI helpers ────────────────────────────────────────────────────────────────

def _wilson_ci(count: int, nobs: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    from statsmodels.stats.proportion import proportion_confint
    if nobs == 0:
        return float("nan"), float("nan")
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def _lr_lognormal_ci(
    tp: int, fp: int, fn: int, tn: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Log-normal 95% CI for LR+ and LR-.

    SE(ln LR+) = sqrt(FP/(TP*(TP+FP)) + FN/(TN*(TN+FN)))
    SE(ln LR-) = sqrt(FN/(TP*(FN+TP)) + TN/(FP*(FP+TN)))
    """
    ci_pos = (float("nan"), float("nan"))
    ci_neg = (float("nan"), float("nan"))

    if tp > 0 and (tp + fp) > 0 and fn >= 0 and (tn + fn) > 0:
        se_pos = np.sqrt(fp / (tp * (tp + fp)) + fn / (tn * (tn + fn)))
        ln_lr_pos = np.log(tp / (tp + fn)) - np.log(fp / (fp + tn))
        ci_pos = (
            float(np.exp(ln_lr_pos - 1.96 * se_pos)),
            float(np.exp(ln_lr_pos + 1.96 * se_pos)),
        )

    if (fn + tp) > 0 and fn > 0 and (fp + tn) > 0:
        se_neg = np.sqrt(fn / (tp * (fn + tp)) + tn / (fp * (fp + tn))) if tp > 0 and fp > 0 else float("nan")
        ln_lr_neg = np.log(fn / (fn + tp)) - np.log(tn / (tn + fp))
        if np.isfinite(se_neg):
            ci_neg = (
                float(np.exp(ln_lr_neg - 1.96 * se_neg)),
                float(np.exp(ln_lr_neg + 1.96 * se_neg)),
            )

    return ci_pos, ci_neg


# ── Core 2x2 LR computation ───────────────────────────────────────────────────

def _lr_from_2x2(
    is_case: np.ndarray,
    test_pos: np.ndarray,
    threshold: float,
    stratum: str = "overall",
) -> LRResult:
    """Compute LR+/LR- from a 2x2 table.

    Parameters
    ----------
    is_case : np.ndarray[bool]
        Incident PD indicator.
    test_pos : np.ndarray[bool]
        Test-positive indicator (rbd_zscore >= threshold).
    threshold : float
        Z-score threshold used to define test_pos.
    stratum : str
        Label for stratified analyses ("overall", "male", "female").

    Returns
    -------
    LRResult
    """
    tp = int(np.sum(test_pos & is_case))
    fp = int(np.sum(test_pos & ~is_case))
    fn = int(np.sum(~test_pos & is_case))
    tn = int(np.sum(~test_pos & ~is_case))
    n_cases = int(is_case.sum())
    n_controls = int((~is_case).sum())

    stable = all(c >= MIN_CELL_COUNT for c in [tp, fp, fn, tn])
    if not stable:
        warnings.warn(
            f"[{stratum}] threshold={threshold:.2f}: one or more 2x2 cells < "
            f"{MIN_CELL_COUNT} (TP={tp}, FP={fp}, FN={fn}, TN={tn}). "
            "LR CI is unreliable.",
            UserWarning,
            stacklevel=3,
        )

    sens = tp / n_cases if n_cases > 0 else float("nan")
    spec = tn / n_controls if n_controls > 0 else float("nan")

    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("nan")

    ci_pos, ci_neg = _lr_lognormal_ci(tp, fp, fn, tn)
    sens_ci = _wilson_ci(tp, n_cases)
    spec_ci = _wilson_ci(tn, n_controls)

    return LRResult(
        threshold=threshold,
        tp=tp, fp=fp, fn=fn, tn=tn,
        sensitivity=round(sens, 6),
        sensitivity_ci=sens_ci,
        specificity=round(spec, 6),
        specificity_ci=spec_ci,
        lr_pos=round(lr_pos, 6),
        lr_pos_ci=ci_pos,
        lr_neg=round(lr_neg, 6),
        lr_neg_ci=ci_neg,
        n_cases=n_cases,
        n_controls=n_controls,
        stratum=stratum,
        stable=stable,
    )


# ── Youden threshold ──────────────────────────────────────────────────────────

def compute_youden_threshold(
    rbd_zscore: np.ndarray,
    is_case: np.ndarray,
) -> float:
    """Return the z-score threshold maximising Youden's J = sensitivity + specificity - 1.

    Parameters
    ----------
    rbd_zscore : np.ndarray
        Continuous RBD z-score.
    is_case : np.ndarray[bool]
        Incident PD indicator.

    Returns
    -------
    float
        Optimal threshold on the z-score scale.
    """
    fpr, tpr, thresholds = roc_curve(is_case.astype(int), rbd_zscore)
    j = tpr - fpr
    idx = int(np.argmax(j))
    t = float(thresholds[idx])
    # roc_curve may return +inf as the first threshold — clamp to data range
    if not np.isfinite(t):
        t = float(np.nanmax(rbd_zscore))
    return t


# ── LR at a single threshold ─────────────────────────────────────────────────

def compute_lr_at_threshold(
    df: pd.DataFrame,
    is_case: pd.Series,
    threshold: float,
    zscore_col: str = RBD_ZSCORE_COL,
    stratum: str = "overall",
) -> LRResult:
    """Compute LR+/LR- at a single z-score threshold.

    Parameters
    ----------
    df : pd.DataFrame
    is_case : pd.Series[bool]
    threshold : float
        Z-score cut-off. Subjects >= threshold are "test positive".
    zscore_col : str
    stratum : str
        Label for output ("overall", "male", "female").
    """
    valid = df[zscore_col].notna()
    zs = df.loc[valid, zscore_col].values
    ic = is_case[valid].values
    test_pos = zs >= threshold
    return _lr_from_2x2(ic, test_pos, threshold=threshold, stratum=stratum)


# ── LR profile over threshold grid ───────────────────────────────────────────

def compute_lr_profile(
    df: pd.DataFrame,
    is_case: pd.Series,
    thresholds: list[float] | None = None,
    zscore_col: str = RBD_ZSCORE_COL,
) -> pd.DataFrame:
    """Compute LR+/LR- at each threshold in the grid.

    Parameters
    ----------
    df : pd.DataFrame
    is_case : pd.Series[bool]
    thresholds : list[float], optional
        Z-score thresholds. Defaults to ZSCORE_THRESHOLD_GRID.
    zscore_col : str

    Returns
    -------
    pd.DataFrame
        One row per threshold, columns from LRResult.to_dict().
    """
    thresholds = thresholds or ZSCORE_THRESHOLD_GRID
    rows = [
        compute_lr_at_threshold(df, is_case, t, zscore_col=zscore_col).to_dict()
        for t in thresholds
    ]
    return pd.DataFrame(rows)


# ── Sex-stratified LR ─────────────────────────────────────────────────────────

def compute_sex_stratified_lr(
    df: pd.DataFrame,
    is_case: pd.Series,
    threshold: float,
    zscore_col: str = RBD_ZSCORE_COL,
) -> list[LRResult]:
    """Compute LR+/LR- separately for males and females at a given threshold.

    Parameters
    ----------
    df : pd.DataFrame
    is_case : pd.Series[bool]
    threshold : float
        Z-score cut-off.
    zscore_col : str

    Returns
    -------
    list[LRResult]
        Two elements: [female_result, male_result].
    """
    results = []
    for sex_val, label in [(FEMALE_CODE, "female"), (MALE_CODE, "male")]:
        sex_mask = df[SEX_COL] == sex_val
        results.append(
            compute_lr_at_threshold(
                df[sex_mask],
                is_case[sex_mask],
                threshold=threshold,
                zscore_col=zscore_col,
                stratum=label,
            )
        )
    return results


# ── Logistic regression OR (continuous z-score) ───────────────────────────────

def compute_logistic_or(
    df: pd.DataFrame,
    is_case: pd.Series,
    zscore_col: str = RBD_ZSCORE_COL,
    adjusted: bool = False,
) -> LogisticORResult:
    """Logistic regression OR per 1 SD increase in RBD z-score.

    Because the z-score is normalised by the control SD, 1 unit = 1 SD of
    the control RBD distribution. The OR is directly interpretable as the
    multiplicative change in PD odds per 1 control-SD increase in RBD score.

    Parameters
    ----------
    df : pd.DataFrame
    is_case : pd.Series[bool]
    zscore_col : str
    adjusted : bool
        If True, include AGE_COL, SEX_COL, BMI_COL as confounders.

    Returns
    -------
    LogisticORResult
    """
    model_type = "adjusted" if adjusted else "unadjusted"
    outcome_col = "is_case_int"

    work = df[[zscore_col] + (CONFOUNDERS if adjusted else [])].copy()
    work[outcome_col] = is_case.astype(int).values

    # Drop rows with any missing value in the model columns.
    work = work.dropna()
    n = len(work)
    n_cases = int(work[outcome_col].sum())

    confounders_str = ""
    if adjusted:
        available = [c for c in CONFOUNDERS if c in work.columns]
        confounders_str = " + " + " + ".join(available) if available else ""

    formula = f"{outcome_col} ~ {zscore_col}{confounders_str}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm_logit(formula, data=work).fit(
                disp=False, maxiter=200, method="bfgs"
            )
        converged = bool(model.mle_retvals.get("converged", True))
        coef = float(model.params[zscore_col])
        ci = model.conf_int().loc[zscore_col].values
        p_val = float(model.pvalues[zscore_col])
        return LogisticORResult(
            or_estimate=round(float(np.exp(coef)), 6),
            or_lci=round(float(np.exp(ci[0])), 6),
            or_uci=round(float(np.exp(ci[1])), 6),
            p_value=round(p_val, 6),
            model_type=model_type,
            n=n,
            n_cases=n_cases,
            converged=converged,
        )
    except Exception as exc:
        warnings.warn(
            f"Logistic regression ({model_type}) failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return LogisticORResult(
            or_estimate=float("nan"), or_lci=float("nan"), or_uci=float("nan"),
            p_value=float("nan"), model_type=model_type,
            n=n, n_cases=n_cases, converged=False,
        )


# ── Empirical prodromal marker LRs (used in C1) ───────────────────────────────

def compute_empirical_marker_lrs(
    df: pd.DataFrame,
    is_case: pd.Series,
    marker_cols: list[str] | None = None,
) -> list[EmpiricalMarkerLR]:
    """Compute empirical LR+/LR- for each viable prodromal marker.

    Markers are binary (0/1). Test-positive = marker == 1.

    Parameters
    ----------
    df : pd.DataFrame
    is_case : pd.Series[bool]
    marker_cols : list[str], optional
        Columns to evaluate. Defaults to PRODROMAL_VIABLE.

    Returns
    -------
    list[EmpiricalMarkerLR]
        One entry per marker column.
    """
    from library.lr_analysis.config import PRODROMAL_LABELS
    marker_cols = marker_cols or PRODROMAL_VIABLE
    results = []

    for col in marker_cols:
        if col not in df.columns:
            warnings.warn(f"Marker column '{col}' not in dataframe.", UserWarning, stacklevel=2)
            continue

        label = PRODROMAL_LABELS.get(col, col)
        valid = df[col].notna()
        marker_vals = df.loc[valid, col].astype(int).values
        ic = is_case[valid].values

        tp = int(np.sum((marker_vals == 1) & ic))
        fp = int(np.sum((marker_vals == 1) & ~ic))
        fn = int(np.sum((marker_vals == 0) & ic))
        tn = int(np.sum((marker_vals == 0) & ~ic))

        stable = all(c >= MIN_CELL_COUNT for c in [tp, fp, fn, tn])
        if not stable:
            warnings.warn(
                f"Prodromal marker '{col}': sparse cells "
                f"(TP={tp}, FP={fp}, FN={fn}, TN={tn}).",
                UserWarning, stacklevel=2,
            )

        n_cases = tp + fn
        n_controls = fp + tn
        sens = tp / n_cases if n_cases > 0 else float("nan")
        spec = tn / n_controls if n_controls > 0 else float("nan")
        lr_pos = sens / (1 - spec) if np.isfinite(spec) and (1 - spec) > 0 else float("nan")
        lr_neg = (1 - sens) / spec if np.isfinite(sens) and spec > 0 else float("nan")

        ci_pos, ci_neg = _lr_lognormal_ci(tp, fp, fn, tn)

        results.append(EmpiricalMarkerLR(
            col=col,
            label=label,
            lr_pos=round(lr_pos, 4),
            lr_pos_ci=ci_pos,
            lr_neg=round(lr_neg, 4),
            lr_neg_ci=ci_neg,
            tp=tp, fp=fp, fn=fn, tn=tn,
            stable=stable,
        ))

    return results


# ── Cohort-aware OR computation (for cognitive/TMT/genetic analyses) ───────────

def compute_logistic_or_cohort(
    df: pd.DataFrame,
    is_case: pd.Series,
    predictor_col: str,
    cohort_name: str,
    adjusted: bool = False,
) -> tuple[LogisticORResult, dict[str, int]]:
    """Compute logistic OR for a single predictor in cohort-specific context.

    Automatically determines adjusters based on cohort_name and applies
    complete-case filtering for the predictor + adjusters.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-set frame (already filtered to cohort).
    is_case : pd.Series[bool]
        Incident PD indicator, aligned to df.
    predictor_col : str
        Name of predictor column.
    cohort_name : str
        One of "cognitive", "tmt", "genetic", "mds_standard".
    adjusted : bool
        If True, adjust for cohort-specific confounders.
        If False, return crude model.

    Returns
    -------
    result : LogisticORResult
        OR estimate with CI and p-value.
    cohort_stats : dict
        Keys: n_total, n_cases, n_controls, pct_complete.
    """
    from library.lr_analysis.config import AGE_COL, BMI_COL, PC_COLS, SEX_COL

    # Determine adjusters by cohort
    if cohort_name == "genetic":
        base_adjusters = [AGE_COL, SEX_COL]
        additional_adjusters = PC_COLS
    else:  # cognitive, tmt, mds_standard
        base_adjusters = [AGE_COL, SEX_COL, BMI_COL]
        additional_adjusters = []

    adjusters = base_adjusters if adjusted else []
    adjusters_for_model = base_adjusters + additional_adjusters if adjusted else []

    # Identify all columns needed
    required = [predictor_col] + adjusters_for_model
    complete_mask = df[required].notna().all(axis=1)
    n_complete = int(complete_mask.sum())
    n_missing = len(df) - n_complete
    pct_complete = 100.0 * n_complete / len(df) if len(df) > 0 else 0.0

    # Filter to complete cases
    work = df[required][complete_mask].copy()
    work["is_case_int"] = is_case[complete_mask].astype(int).values
    n = len(work)
    n_cases = int(work["is_case_int"].sum())
    n_controls = n - n_cases

    # Build formula
    formula = f"is_case_int ~ {predictor_col}"
    if adjusted:
        adjusters_str = " + ".join(adjusters_for_model)
        formula = f"is_case_int ~ {predictor_col} + {adjusters_str}"

    # Fit model
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm_logit(formula, data=work).fit(
                disp=False, maxiter=200, method="bfgs"
            )
        converged = bool(model.mle_retvals.get("converged", True))
        coef = float(model.params[predictor_col])
        ci = model.conf_int().loc[predictor_col].values
        p_val = float(model.pvalues[predictor_col])

        result = LogisticORResult(
            or_estimate=round(float(np.exp(coef)), 6),
            or_lci=round(float(np.exp(ci[0])), 6),
            or_uci=round(float(np.exp(ci[1])), 6),
            p_value=round(p_val, 6),
            model_type="adjusted" if adjusted else "unadjusted",
            n=n,
            n_cases=n_cases,
            converged=converged,
        )
    except Exception as exc:
        warnings.warn(
            f"Logistic regression failed for {predictor_col}: {exc}",
            UserWarning,
            stacklevel=2,
        )
        result = LogisticORResult(
            or_estimate=float("nan"),
            or_lci=float("nan"),
            or_uci=float("nan"),
            p_value=float("nan"),
            model_type="adjusted" if adjusted else "unadjusted",
            n=n,
            n_cases=n_cases,
            converged=False,
        )

    cohort_stats = {
        "n_total": n_complete,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "pct_complete": round(pct_complete, 2),
    }

    return result, cohort_stats


def compute_logistic_or_multivariate(
    df: pd.DataFrame,
    is_case: pd.Series,
    predictor_cols: list[str],
    cohort_name: str,
    adjusted: bool = False,
) -> tuple[dict[str, LogisticORResult], dict[str, int]]:
    """Fit single logistic model with multiple predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-set frame (already filtered to cohort).
    is_case : pd.Series[bool]
        Incident PD indicator.
    predictor_cols : list[str]
        Columns to include as predictors.
    cohort_name : str
        One of "cognitive", "tmt", "genetic", "mds_standard".
    adjusted : bool
        If True, include cohort-specific confounders.

    Returns
    -------
    results : dict[str, LogisticORResult]
        Maps predictor_col -> LogisticORResult.
    cohort_stats : dict
        Sample completeness stats.
    """
    from library.lr_analysis.config import AGE_COL, BMI_COL, PC_COLS, SEX_COL

    # Determine adjusters by cohort
    if cohort_name == "genetic":
        adjusters = [AGE_COL, SEX_COL] + PC_COLS if adjusted else []
    else:
        adjusters = [AGE_COL, SEX_COL, BMI_COL] if adjusted else []

    required = predictor_cols + adjusters
    complete_mask = df[required].notna().all(axis=1)
    n_complete = int(complete_mask.sum())

    work = df[required][complete_mask].copy()
    work["is_case_int"] = is_case[complete_mask].astype(int).values
    n = len(work)
    n_cases = int(work["is_case_int"].sum())
    n_controls = n - n_cases

    # Build formula
    pred_str = " + ".join(predictor_cols)
    formula = f"is_case_int ~ {pred_str}"
    if adjusted:
        adj_str = " + ".join(adjusters)
        formula = f"is_case_int ~ {pred_str} + {adj_str}"

    results = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm_logit(formula, data=work).fit(
                disp=False, maxiter=200, method="bfgs"
            )
        converged = bool(model.mle_retvals.get("converged", True))

        for col in predictor_cols:
            coef = float(model.params[col])
            ci = model.conf_int().loc[col].values
            p_val = float(model.pvalues[col])
            results[col] = LogisticORResult(
                or_estimate=round(float(np.exp(coef)), 6),
                or_lci=round(float(np.exp(ci[0])), 6),
                or_uci=round(float(np.exp(ci[1])), 6),
                p_value=round(p_val, 6),
                model_type="adjusted" if adjusted else "unadjusted",
                n=n,
                n_cases=n_cases,
                converged=converged,
            )
    except Exception as exc:
        warnings.warn(
            f"Multivariate logistic regression failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        for col in predictor_cols:
            results[col] = LogisticORResult(
                or_estimate=float("nan"),
                or_lci=float("nan"),
                or_uci=float("nan"),
                p_value=float("nan"),
                model_type="adjusted" if adjusted else "unadjusted",
                n=n,
                n_cases=n_cases,
                converged=False,
            )

    cohort_stats = {
        "n_total": n_complete,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "pct_complete": round(100.0 * n_complete / len(df), 2) if len(df) > 0 else 0.0,
    }

    return results, cohort_stats


# ── LR profile stratified by age groups ────────────────────────────────────────

def compute_lr_profile_by_age(
    df: pd.DataFrame,
    is_case: pd.Series,
    predictor_col: str,
    age_col: str = "cov_age_recruitment_21022",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute LR+ and LR- at each threshold, stratified by age groups.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis frame.
    is_case : pd.Series[bool]
        Incident PD indicator.
    predictor_col : str
        Z-score predictor column.
    age_col : str
        Age column for stratification.
    thresholds : list[float], optional
        Z-score thresholds. Defaults to ZSCORE_THRESHOLD_GRID.

    Returns
    -------
    pd.DataFrame
        One row per (threshold, age_group) combination with LR+, LR-, etc.
    """
    from library.lr_analysis.config import ZSCORE_THRESHOLD_GRID

    thresholds = thresholds or ZSCORE_THRESHOLD_GRID

    # Define age groups (5-year bands)
    age_bands = [50, 55, 60, 65, 70, 75, 80]
    age_group_map = {}
    for i, band in enumerate(age_bands[:-1]):
        age_group_map[f"{band}-{age_bands[i+1]-1}"] = (band, age_bands[i+1])
    age_group_map["80+"] = (80, 120)

    rows = []
    for age_label, (age_min, age_max) in age_group_map.items():
        age_mask = (df[age_col] >= age_min) & (df[age_col] < age_max)
        if age_mask.sum() < 10:  # Skip small groups
            continue

        for threshold in thresholds:
            test_pos = df.loc[age_mask, predictor_col] >= threshold
            ic = is_case[age_mask].values

            tp = int(np.sum(test_pos & ic))
            fp = int(np.sum(test_pos & ~ic))
            fn = int(np.sum(~test_pos & ic))
            tn = int(np.sum(~test_pos & ~ic))

            n_cases = tp + fn
            n_controls = fp + tn
            if n_cases == 0 or n_controls == 0:
                continue

            sens = tp / n_cases if n_cases > 0 else float("nan")
            spec = tn / n_controls if n_controls > 0 else float("nan")

            lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("nan")
            lr_neg = (1 - sens) / spec if spec > 0 else float("nan")

            ci_pos, ci_neg = _lr_lognormal_ci(tp, fp, fn, tn)

            rows.append({
                "threshold": threshold,
                "age_group": age_label,
                "n_total": tp + fp + fn + tn,
                "n_cases": n_cases,
                "n_controls": n_controls,
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "lr_pos": round(lr_pos, 4),
                "lr_pos_lci": round(ci_pos[0], 4),
                "lr_pos_uci": round(ci_pos[1], 4),
                "lr_neg": round(lr_neg, 4),
                "lr_neg_lci": round(ci_neg[0], 4),
                "lr_neg_uci": round(ci_neg[1], 4),
            })

    return pd.DataFrame(rows)


# ── LR profile stratified by RBD tertiles ──────────────────────────────────────

def compute_lr_profile_by_rbd_strata(
    df: pd.DataFrame,
    is_case: pd.Series,
    predictor_col: str,
    rbd_strata_col: str = "rbd_risk_group_mean_3g",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute LR+ and LR- at each threshold, stratified by RBD tertiles.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis frame (must contain rbd_strata_col).
    is_case : pd.Series[bool]
        Incident PD indicator.
    predictor_col : str
        Z-score predictor column.
    rbd_strata_col : str
        RBD risk group column (categorical: Low/Mid/High).
    thresholds : list[float], optional
        Z-score thresholds.

    Returns
    -------
    pd.DataFrame
        One row per (threshold, rbd_stratum) combination.
    """
    from library.lr_analysis.config import ZSCORE_THRESHOLD_GRID

    thresholds = thresholds or ZSCORE_THRESHOLD_GRID

    rbd_groups = ["Low", "Mid", "High"]
    rows = []

    for rbd_stratum in rbd_groups:
        rbd_mask = df[rbd_strata_col] == rbd_stratum
        if rbd_mask.sum() < 10:
            continue

        for threshold in thresholds:
            test_pos = df.loc[rbd_mask, predictor_col] >= threshold
            ic = is_case[rbd_mask].values

            tp = int(np.sum(test_pos & ic))
            fp = int(np.sum(test_pos & ~ic))
            fn = int(np.sum(~test_pos & ic))
            tn = int(np.sum(~test_pos & ~ic))

            n_cases = tp + fn
            n_controls = fp + tn
            if n_cases == 0 or n_controls == 0:
                continue

            sens = tp / n_cases if n_cases > 0 else float("nan")
            spec = tn / n_controls if n_controls > 0 else float("nan")

            lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("nan")
            lr_neg = (1 - sens) / spec if spec > 0 else float("nan")

            ci_pos, ci_neg = _lr_lognormal_ci(tp, fp, fn, tn)

            rows.append({
                "threshold": threshold,
                "rbd_stratum": rbd_stratum,
                "n_total": tp + fp + fn + tn,
                "n_cases": n_cases,
                "n_controls": n_controls,
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "lr_pos": round(lr_pos, 4),
                "lr_pos_lci": round(ci_pos[0], 4),
                "lr_pos_uci": round(ci_pos[1], 4),
                "lr_neg": round(lr_neg, 4),
                "lr_neg_lci": round(ci_neg[0], 4),
                "lr_neg_uci": round(ci_neg[1], 4),
            })

    return pd.DataFrame(rows)


def compute_rbd_interaction_test(
    df: pd.DataFrame,
    is_case: pd.Series,
    predictor_col: str,
    rbd_col: str,
    confounders: list[str],
    label: str,
    cohort_name: str,
) -> RBDInteractionResult:
    """Test RBD × predictor interaction via likelihood ratio test.

    Models:
    - M1 (reduced): PD ~ C(rbd_col) + predictor + confounders
    - M2 (full): PD ~ C(rbd_col) + predictor + C(rbd_col):predictor + confounders

    Args:
        df: DataFrame with outcome, predictor, rbd_col, and confounders.
        is_case: Boolean series indicating incident PD (True = case, False = control).
        predictor_col: Column name for continuous predictor (will be z-scored).
        rbd_col: Column name for RBD categorical (should be Low/Mid/High).
        confounders: List of confounder column names.
        label: Human-readable label for predictor.
        cohort_name: Name of cohort (for reporting).

    Returns:
        RBDInteractionResult with LRT stat, interaction ORs, and main effect OR.
    """
    required_cols = [predictor_col, rbd_col] + confounders
    complete_mask = df[required_cols].notna().all(axis=1)
    n_complete = int(complete_mask.sum())

    work = df[required_cols].copy()
    work = work[complete_mask].copy()
    work["is_case_int"] = is_case[complete_mask].astype(int).values

    n = len(work)
    n_cases = int(work["is_case_int"].sum())
    n_controls = n - n_cases

    # Z-score predictor using control distribution only (no leakage)
    control_mask = work["is_case_int"] == 0
    if control_mask.sum() > 1:
        mu = float(work.loc[control_mask, predictor_col].mean())
        sigma = float(work.loc[control_mask, predictor_col].std())
        if sigma == 0:
            sigma = 1.0
        work["predictor_z"] = (work[predictor_col] - mu) / sigma
    else:
        work["predictor_z"] = (work[predictor_col] - work[predictor_col].mean()) / (
            work[predictor_col].std() or 1.0
        )

    # Build formula strings
    confounders_str = " + ".join(confounders)
    formula_reduced = f"is_case_int ~ C({rbd_col}) + predictor_z + {confounders_str}"
    formula_full = f"is_case_int ~ C({rbd_col}) + predictor_z + C({rbd_col}):predictor_z + {confounders_str}"

    # Fit models
    reduced_result = None
    full_result = None
    converged_reduced = False
    converged_full = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reduced_model = sm_logit(formula_reduced, data=work).fit(
                disp=False, maxiter=400, method="bfgs"
            )
        converged_reduced = bool(reduced_model.mle_retvals.get("converged", True))
        reduced_result = reduced_model
    except Exception as exc:
        warnings.warn(
            f"Reduced model ({label}) failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return RBDInteractionResult(
            variable=predictor_col,
            label=label,
            cohort_name=cohort_name,
            reduced_llf=float("nan"),
            full_llf=float("nan"),
            lrt_stat=float("nan"),
            lrt_df=2,
            lrt_p=float("nan"),
            interaction_ors={"Mid": float("nan"), "High": float("nan")},
            interaction_lcis={"Mid": float("nan"), "High": float("nan")},
            interaction_ucis={"Mid": float("nan"), "High": float("nan")},
            interaction_ps={"Mid": float("nan"), "High": float("nan")},
            main_g_or=float("nan"),
            main_g_lci=float("nan"),
            main_g_uci=float("nan"),
            main_g_p=float("nan"),
            n_total=n,
            n_cases=n_cases,
            n_controls=n_controls,
            converged_reduced=converged_reduced,
            converged_full=converged_full,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = sm_logit(formula_full, data=work).fit(
                disp=False, maxiter=400, method="bfgs"
            )
        converged_full = bool(full_model.mle_retvals.get("converged", True))
        full_result = full_model
    except Exception as exc:
        warnings.warn(
            f"Full model ({label}) failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return RBDInteractionResult(
            variable=predictor_col,
            label=label,
            cohort_name=cohort_name,
            reduced_llf=float(reduced_result.llf) if reduced_result else float("nan"),
            full_llf=float("nan"),
            lrt_stat=float("nan"),
            lrt_df=2,
            lrt_p=float("nan"),
            interaction_ors={"Mid": float("nan"), "High": float("nan")},
            interaction_lcis={"Mid": float("nan"), "High": float("nan")},
            interaction_ucis={"Mid": float("nan"), "High": float("nan")},
            interaction_ps={"Mid": float("nan"), "High": float("nan")},
            main_g_or=float("nan"),
            main_g_lci=float("nan"),
            main_g_uci=float("nan"),
            main_g_p=float("nan"),
            n_total=n,
            n_cases=n_cases,
            n_controls=n_controls,
            converged_reduced=converged_reduced,
            converged_full=converged_full,
        )

    # Compute LRT: stat = -2 * (llf_reduced - llf_full)
    llf_reduced = float(reduced_result.llf)
    llf_full = float(full_result.llf)
    lrt_stat = -2 * (llf_reduced - llf_full)
    lrt_p = float(scipy_stats.chi2.sf(lrt_stat, df=2))

    # Extract interaction ORs for Mid and High levels
    interaction_ors = {}
    interaction_lcis = {}
    interaction_ucis = {}
    interaction_ps = {}

    for level in ["Mid", "High"]:
        term_name = f"C({rbd_col})[T.{level}]:predictor_z"
        try:
            if term_name in full_result.params.index:
                coef = float(full_result.params[term_name])
                ci_row = full_result.conf_int().loc[term_name]
                p_val = float(full_result.pvalues[term_name])

                interaction_ors[level] = float(np.exp(coef))
                interaction_lcis[level] = float(np.exp(ci_row[0]))
                interaction_ucis[level] = float(np.exp(ci_row[1]))
                interaction_ps[level] = p_val
            else:
                interaction_ors[level] = float("nan")
                interaction_lcis[level] = float("nan")
                interaction_ucis[level] = float("nan")
                interaction_ps[level] = float("nan")
        except Exception:
            interaction_ors[level] = float("nan")
            interaction_lcis[level] = float("nan")
            interaction_ucis[level] = float("nan")
            interaction_ps[level] = float("nan")

    # Extract main effect of predictor_z in full model
    main_g_or = float("nan")
    main_g_lci = float("nan")
    main_g_uci = float("nan")
    main_g_p = float("nan")
    try:
        if "predictor_z" in full_result.params.index:
            coef = float(full_result.params["predictor_z"])
            ci_row = full_result.conf_int().loc["predictor_z"]
            p_val = float(full_result.pvalues["predictor_z"])

            main_g_or = float(np.exp(coef))
            main_g_lci = float(np.exp(ci_row[0]))
            main_g_uci = float(np.exp(ci_row[1]))
            main_g_p = p_val
    except Exception:
        pass

    return RBDInteractionResult(
        variable=predictor_col,
        label=label,
        cohort_name=cohort_name,
        reduced_llf=llf_reduced,
        full_llf=llf_full,
        lrt_stat=lrt_stat,
        lrt_df=2,
        lrt_p=lrt_p,
        interaction_ors=interaction_ors,
        interaction_lcis=interaction_lcis,
        interaction_ucis=interaction_ucis,
        interaction_ps=interaction_ps,
        main_g_or=main_g_or,
        main_g_lci=main_g_lci,
        main_g_uci=main_g_uci,
        main_g_p=main_g_p,
        n_total=n,
        n_cases=n_cases,
        n_controls=n_controls,
        converged_reduced=converged_reduced,
        converged_full=converged_full,
    )
