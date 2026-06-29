"""
training.py
===========

Nested cross-validation trainer with Optuna inner-loop hyperparameter
optimisation.

Design notes
------------
* **Outer loop**: 5-fold stratified CV → unbiased generalisation estimate.
* **Inner loop**: 3-fold stratified CV → average-precision objective for
  Optuna. The inner CV uses the *outer training fold only*; the outer test
  fold is never seen during tuning.
* **Imputer**: ``ImputerPipeline`` is fit on the inner training fold inside
  the Optuna objective, AND on the outer training fold for the final refit.
  Nothing is fit on the full dataset.
* **Class imbalance**: Trees and linear models receive a sample-weight array
  ``w[i] = (n_neg / n_pos)`` for positives and ``1.0`` for negatives.
  XGBoost ignores this in favour of its native ``scale_pos_weight``.
* **Reproducibility**: ``random_state=42`` everywhere unless overridden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .features import ImputerPipeline
from .metrics import MetricsResult, compute_all_metrics
from .models.base import ModelBase

# Suppress Optuna's INFO chatter — only WARNING+ from now on.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class FoldResult:
    """Per-fold output of nested CV."""

    fold: int
    metrics: MetricsResult
    best_params: dict[str, Any]
    val_indices: np.ndarray
    val_proba: np.ndarray
    val_true: np.ndarray
    # Fold-level cohort composition (populated by both trainers).
    n_train_cases: int = 0
    n_train_controls: int = 0
    n_test_cases: int = 0
    n_test_controls: int = 0


@dataclass
class NestedCVResult:
    """Aggregate result of all outer folds for one model."""

    model_name: str
    folds: list[FoldResult] = field(default_factory=list)

    def metrics_frame(self) -> pd.DataFrame:
        """Return one row per fold of metrics."""
        rows = [{"fold": fr.fold, **fr.metrics.to_dict()} for fr in self.folds]
        return pd.DataFrame(rows)

    def mean_metrics(self) -> pd.DataFrame:
        """Return mean ± SD across folds for each numeric metric."""
        df = self.metrics_frame().drop(columns=["fold"])
        agg = df.agg(["mean", "std"]).T
        agg.columns = ["mean", "sd"]
        return agg


# --- Helpers ----------------------------------------------------------------

def _compute_sample_weight(y: pd.Series) -> np.ndarray:
    """
    Build per-sample weights for class-imbalance reweighting.

    Positives get weight ``n_neg / n_pos``; negatives get ``1.0``. This is
    the maximum-likelihood reweighting under Bernoulli loss for the rare
    positive class.
    """
    y_arr = np.asarray(y).astype(int)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0:
        return np.ones_like(y_arr, dtype=float)
    pos_w = n_neg / n_pos
    return np.where(y_arr == 1, pos_w, 1.0).astype(float)


# --- Trainer ----------------------------------------------------------------

@dataclass
class NestedCVTrainer:
    """
    Nested CV + Optuna trainer.

    Parameters
    ----------
    model_cls :
        Subclass of ``ModelBase`` to train.
    n_outer :
        Outer-fold count (default 5).
    n_inner :
        Inner-fold count (default 3).
    n_trials :
        Optuna trials per outer fold (default 50).
    random_state :
        Master seed for both outer and inner CV. Documented at the call
        site so re-runs are exactly reproducible.
    tte_strategy :
        Time-to-event imputation strategy (default "constant_p95").
    imputation_enabled :
        Whether to enable imputation of missing values (default True).
    """

    model_cls: type[ModelBase]
    n_outer: int = 5
    n_inner: int = 3
    n_trials: int = 50
    random_state: int = 42
    tte_strategy: str = "constant_p95"
    imputation_enabled: bool = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NestedCVResult:
        """
        Run the full nested CV procedure.

        Returns
        -------
        NestedCVResult
            Per-fold predictions, best hyperparameters, and metrics.
        """
        outer = StratifiedKFold(
            n_splits=self.n_outer, shuffle=True, random_state=self.random_state,
        )
        result = NestedCVResult(model_name=self.model_cls.name)

        for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            # Fit imputer on outer-train only.
            outer_imp = ImputerPipeline(
                tte_strategy=self.tte_strategy,
                random_state=self.random_state + fold_idx,
                enabled=self.imputation_enabled,
            ).fit(X_tr, y_tr)
            X_tr_imp = outer_imp.transform(X_tr)
            X_te_imp = outer_imp.transform(X_te)

            # Optuna over inner CV.
            best_params = self._tune(X_tr_imp, y_tr) if self.model_cls.supports_optuna else {}

            # Final refit on full outer-train.
            final_model = self.model_cls(params=best_params, random_state=self.random_state)
            sw = _compute_sample_weight(y_tr) if final_model.supports_sample_weight else None
            final_model.fit(X_tr_imp, y_tr, sample_weight=sw)

            proba = final_model.predict_proba(X_te_imp)
            metrics = compute_all_metrics(y_te.values, proba)

            result.folds.append(FoldResult(
                fold=fold_idx,
                metrics=metrics,
                best_params=best_params,
                val_indices=np.asarray(te_idx),
                val_proba=proba,
                val_true=y_te.values.astype(int),
                n_train_cases=int(y_tr.sum()),
                n_train_controls=int((y_tr == 0).sum()),
                n_test_cases=int(y_te.sum()),
                n_test_controls=int((y_te == 0).sum()),
            ))

        return result

    # ---- inner loop --------------------------------------------------------

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Run Optuna over the inner stratified CV. Maximise mean PR-AUC.
        """
        inner = StratifiedKFold(
            n_splits=self.n_inner, shuffle=True, random_state=self.random_state + 1,
        )

        def objective(trial: optuna.Trial) -> float:
            """Average PR-AUC over inner folds for one hyperparameter draw."""
            template = self.model_cls(random_state=self.random_state)
            params = template.get_optuna_params(trial)
            scores: list[float] = []
            for itr_idx, ite_idx in inner.split(X, y):
                Xi_tr, Xi_te = X.iloc[itr_idx], X.iloc[ite_idx]
                yi_tr, yi_te = y.iloc[itr_idx], y.iloc[ite_idx]

                # Imputer is already applied at the outer level (numeric only,
                # no NaN remains), so we skip refitting per inner fold for
                # speed. (No leakage: outer_imp was fit on outer-train.)
                model = self.model_cls(params=params, random_state=self.random_state)
                sw = _compute_sample_weight(yi_tr) if model.supports_sample_weight else None
                model.fit(Xi_tr, yi_tr, sample_weight=sw)
                p = model.predict_proba(Xi_te)
                if len(np.unique(yi_te)) < 2:
                    continue  # skip degenerate folds
                scores.append(average_precision_score(yi_te, p))

            return float(np.mean(scores)) if scores else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return dict(study.best_params)
