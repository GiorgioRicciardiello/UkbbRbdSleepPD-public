"""
features.py
===========

Build the feature matrix for ml_cross_sectional and supply a leakage-safe
imputer that is fit *inside* each cross-validation fold.

Time-to-event encoding (mathematical rationale)
-----------------------------------------------
``time_to_event`` is the number of days between ``wear_time_start`` and
``outcome_1a_pd_only_date``. It is defined for cases only. The encoding
choice for controls is non-trivial because they are right-censored:

* Setting controls to total follow-up *inflates* them beyond the case
  range and conflates the censoring mechanism with the biological signal.
* Setting controls to NaN and imputing with the case mean tells the model
  "this control is an average converter", inverting the signal.

We instead set controls to NaN and let ``ImputerPipeline`` fill them with
the **95th percentile of the case distribution computed on the training
fold**. This treats controls as "very late converters or non-converters",
keeping them within the case range while preserving the early-vs-late
discrimination signal. The fill value is recomputed per fold to avoid
leakage.

Cases with negative days (PD diagnosed before wear-time start, i.e.
prevalent cases) are clipped to 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from .dataset import CrossSectionalFrame

#: Name of the engineered time-to-event column.
TIME_TO_EVENT_COL: Final[str] = "time_to_event_log"


# --- Time-to-event -----------------------------------------------------------

def build_time_to_event(
    df: pd.DataFrame,
    outcome_col: str,
    wear_start_col: str,
    event_date_col: str,
) -> pd.Series:
    """
    Compute the time-to-event feature for cases (NaN for controls).

    Parameters
    ----------
    df :
        Cross-sectional dataframe (output of ``convert_to_cross_sectional``).
    outcome_col :
        Binary outcome column name (1 = case, 0 = control).
    wear_start_col, event_date_col :
        Date columns. ``event_date_col`` is only meaningful for cases.

    Returns
    -------
    pd.Series
        ``log1p(days)`` for cases, ``NaN`` for controls. Length matches
        ``len(df)`` and the index is preserved.

    Notes
    -----
    * For cases the value is ``log(1 + max(0, days))``.
    * Prevalent PD (negative days) are clipped to 0 → log1p(0) = 0.
    """
    days = (df[event_date_col] - df[wear_start_col]).dt.days
    is_case = df[outcome_col].astype(int) == 1
    out = pd.Series(np.nan, index=df.index, dtype="float64", name=TIME_TO_EVENT_COL)
    case_days = days.where(is_case).clip(lower=0)
    out.loc[is_case] = np.log1p(case_days.loc[is_case].astype(float))
    return out


# --- Feature matrix ----------------------------------------------------------

def get_feature_matrix(
    frame: CrossSectionalFrame,
    include_tte: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the (X, y) tuple for modelling.

    Parameters
    ----------
    frame :
        Output of ``dataset.convert_to_cross_sectional``.
    include_tte :
        Whether to append the engineered ``time_to_event_log`` column.
        Set to ``False`` when the caller drops TTE at training time (e.g.
        P1CombinedTrainer with ``tte_strategy="exclude"``) so that cohort
        stats and distribution reports reflect the actual feature set used.
        Default is ``True`` for backward compatibility with NestedCVTrainer.

    Returns
    -------
    X :
        Feature matrix. When ``include_tte=True``, includes the engineered
        ``time_to_event_log`` column; otherwise only ``frame.feature_cols``.
        Continuous and categorical columns are kept as-is. Imputation is the
        responsibility of ``ImputerPipeline`` and must happen inside the CV
        loop.
    y :
        Binary outcome series, dtype ``int``.
    """
    df = frame.df
    X = df.loc[:, list(frame.feature_cols)].copy()

    if include_tte:
        tte = build_time_to_event(
            df=df,
            outcome_col=frame.outcome_col,
            wear_start_col=frame.wear_start_col,
            event_date_col=frame.event_date_col,
        )
        X[TIME_TO_EVENT_COL] = tte

    y = df[frame.outcome_col].astype(int)
    return X, y


# --- Leakage-safe imputer ----------------------------------------------------

#: Allowed time-to-event imputation strategies for ``ImputerPipeline``.
#:
#: * ``constant_p95``   : fill controls with a single value = 95th percentile
#:                       of case distribution (training fold). Creates a point
#:                       mass at the fill value → trees will exploit this.
#: * ``jittered_q3_max``: draw each control independently from a uniform
#:                       distribution on [Q3, max] of the training-fold case
#:                       distribution. Gives controls realistic spread so the
#:                       feature cannot be exploited as a label indicator.
TTEStrategy = str  # Literal["constant_p95", "jittered_q3_max"]


@dataclass
class ImputerPipeline:
    """
    Per-fold imputer with configurable time-to-event handling.

    Parameters
    ----------
    tte_strategy :
        One of ``"constant_p95"`` or ``"jittered_q3_max"``. See module-level
        ``TTEStrategy`` for the rationale of each.
    random_state :
        Seed used by ``jittered_q3_max``. Required for reproducibility — the
        same training fold must always yield identical control imputations.
    enabled :
        If ``False``, no imputation is performed and missing values are left as-is.
        Default is ``True`` for backward compatibility.

    Notes
    -----
    * Numeric columns: median (computed on the training fold).
    * Categorical columns: mode.
    * ``time_to_event_log`` handled per ``tte_strategy``.
    * When ``enabled=False``, all transform operations return the input unchanged.
    """

    tte_strategy: TTEStrategy = "constant_p95"
    random_state: int = 42
    time_to_event_col: str = TIME_TO_EVENT_COL
    enabled: bool = True

    median_: dict[str, float] = field(default_factory=dict)
    mode_: dict[str, object] = field(default_factory=dict)
    tte_case_q3_: float = 0.0
    tte_case_max_: float = 0.0
    tte_constant_: float = 0.0
    fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ImputerPipeline":
        """
        Learn imputation values from the training fold only.

        Parameters
        ----------
        X :
            Training-fold feature matrix.
        y :
            Training-fold outcome (used to identify cases for the TTE fill).
        """
        if not self.enabled:
            self.fitted_ = True
            return self

        if self.tte_strategy not in ("constant_p95", "jittered_q3_max"):
            raise ValueError(f"Unknown tte_strategy: {self.tte_strategy!r}")

        self.median_ = {}
        self.mode_ = {}
        for col in X.columns:
            if col == self.time_to_event_col:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                self.median_[col] = float(X[col].median())
            else:
                mode_vals = X[col].mode(dropna=True)
                self.mode_[col] = mode_vals.iloc[0] if len(mode_vals) else None

        if self.time_to_event_col in X.columns:
            case_mask = (y.astype(int) == 1)
            case_tte = X.loc[case_mask, self.time_to_event_col].dropna()
            if len(case_tte) > 0:
                self.tte_constant_ = float(np.percentile(case_tte, 95.0))
                self.tte_case_q3_ = float(np.percentile(case_tte, 75.0))
                self.tte_case_max_ = float(case_tte.max())
            else:
                self.tte_constant_ = 0.0
                self.tte_case_q3_ = 0.0
                self.tte_case_max_ = 0.0

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned imputations to *X* (returns a copy)."""
        if not self.fitted_:
            raise RuntimeError("ImputerPipeline.transform called before fit.")
        if not self.enabled:
            return X.copy()
        out = X.copy()
        for col, val in self.median_.items():
            if col in out.columns:
                out[col] = out[col].fillna(val)
        for col, val in self.mode_.items():
            if col in out.columns and val is not None:
                out[col] = out[col].fillna(val)

        if self.time_to_event_col in out.columns:
            out[self.time_to_event_col] = self._fill_tte(
                out[self.time_to_event_col]
            )

        return out.astype("float64")

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Convenience: ``fit(X, y)`` then ``transform(X)``."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fill_tte(self, col: pd.Series) -> pd.Series:
        """Return *col* with NaNs filled according to ``tte_strategy``."""
        out = col.copy()
        nan_mask = out.isna()
        n_nan = int(nan_mask.sum())
        if n_nan == 0:
            return out

        if self.tte_strategy == "constant_p95":
            out.loc[nan_mask] = self.tte_constant_
            return out

        # jittered_q3_max: uniform draws on [Q3, max] of training cases.
        lo = self.tte_case_q3_
        hi = self.tte_case_max_
        if hi <= lo:
            out.loc[nan_mask] = lo  # degenerate range → point mass
            return out
        rng = np.random.default_rng(self.random_state)
        draws = rng.uniform(low=lo, high=hi, size=n_nan)
        out.loc[nan_mask] = draws
        return out
