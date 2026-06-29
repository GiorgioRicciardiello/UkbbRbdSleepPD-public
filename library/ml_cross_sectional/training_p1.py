"""
training_p1.py
==============

P1 Combined training paradigm for ml_cross_sectional.

Training strategy
-----------------
Mirrors the P1 Combined paradigm from ``src/screening/paradigms/p1_combined.py``:

* **Cases** = incident + prevalent positives (both are treated as y=1).
* **Controls** = 1:N random matching per outer fold from the valid control pool
  (subjects where ``control == True``).
* **Outer CV** = StratifiedKFold stratified on ``y_incident`` (not on the
  combined y_label). Prevalent cases contribute noise if stratified on y_label
  because they represent a fully-expressed disease signal.
* **Test set** = incident cases + controls only. Prevalent cases are excluded
  from the test fold evaluation (they are not good prognostic targets).
* **Inner CV / Optuna** = maximise mean PR-AUC across inner folds, identical
  to NestedCVTrainer.
* **Imputer** = ImputerPipeline fit on the matched training set (fold-local,
  no leakage into the test fold).

Result type
-----------
Returns ``NestedCVResult`` — identical to the standard trainer, so all
downstream reporting, storage, and plotting code is re-used without changes.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .features import ImputerPipeline, TIME_TO_EVENT_COL
from .matching import match_controls, split_cases_controls
from .metrics import compute_all_metrics
from .models.base import ModelBase
from .training import FoldResult, NestedCVResult, _compute_sample_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class P1CombinedConfig:
    """
    User-tunable knobs for a P1 Combined training run.

    Parameters
    ----------
    n_outer :
        Number of outer CV folds (default 5).
    n_inner :
        Number of inner CV folds for Optuna (default 3).
    n_trials :
        Optuna trials per outer fold (default 50).
    random_state :
        Master seed. Documented so re-runs are reproducible.
    controls_per_case :
        1:N matching ratio (default 10). Reduce if control pool is small.
    outcome_name :
        Registered outcome name (see outcomes.py, default "pd_only").
    feature_set :
        Feature set name (see feature_sets.py, default None = rbd_alone).
    tte_strategy :
        Time-to-event imputation strategy ("exclude" or "jittered_q3_max").
    file_name :
        Parquet file name passed to get_clean_risk_data.
    """

    n_outer: int = 5
    n_inner: int = 3
    n_trials: int = 50
    random_state: int = 42
    controls_per_case: int = 10
    outcome_name: str = "pd_only"
    feature_set: str | None = None
    tte_strategy: str = "exclude"
    file_name: str = "ehr_diag_pd_rbd_only_all"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class P1CombinedTrainer:
    """
    Nested CV trainer using the P1 Combined paradigm.

    Parameters
    ----------
    model_cls :
        Subclass of ``ModelBase`` to train.
    cfg :
        Run configuration (P1CombinedConfig).
    """

    model_cls: type[ModelBase]
    cfg: P1CombinedConfig

    def fit(
        self,
        X: pd.DataFrame,
        y_label: pd.Series,
        y_incident: pd.Series,
        y_prevalent: pd.Series,
        y_control: pd.Series,
    ) -> NestedCVResult:
        """
        Run the full P1 Combined nested CV procedure.

        Parameters
        ----------
        X :
            Pre-imputation feature matrix (output of get_feature_matrix or
            similar), indexed 0…N-1.
        y_label :
            Combined binary outcome (y_incident | y_prevalent). Used for
            evaluation only (not for fold stratification).
        y_incident :
            Binary series; 1 = incident case. Used for outer CV stratification
            and test-fold filtering.
        y_prevalent :
            Binary series; 1 = prevalent case. Combined with y_incident for
            training, excluded from evaluation.
        y_control :
            Boolean series; True = valid control (outcome-agnostic ``control``
            column). Defines the eligible matching pool.

        Returns
        -------
        NestedCVResult
            Per-fold predictions, best hyperparameters, and metrics.
            Identical structure to NestedCVTrainer output.
        """
        # Stratify outer CV on incident labels (not combined y_label).
        # Prevalent cases contribute as "0" for stratification, which keeps
        # fold class distributions consistent with the incident case signal.
        outer = StratifiedKFold(
            n_splits=self.cfg.n_outer,
            shuffle=True,
            random_state=self.cfg.random_state,
        )
        result = NestedCVResult(model_name=self.model_cls.name)

        # Resolve TTE mode: "exclude" drops the TTE column; other values pass
        # directly to ImputerPipeline as the imputation strategy.
        if self.cfg.tte_strategy == "exclude":
            if TIME_TO_EVENT_COL in X.columns:
                X = X.drop(columns=[TIME_TO_EVENT_COL])
            _imputer_tte_strategy = "jittered_q3_max"  # irrelevant; column absent
        else:
            _imputer_tte_strategy = self.cfg.tte_strategy

        for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X, y_incident)):
            fold_rng = np.random.default_rng(self.cfg.random_state + fold_idx)

            # --- Training fold: case:control matching -----------------------
            y_inc_tr = y_incident.iloc[tr_idx].reset_index(drop=True)
            y_prev_tr = y_prevalent.iloc[tr_idx].reset_index(drop=True)
            y_ctrl_tr = y_control.iloc[tr_idx].reset_index(drop=True)

            case_idx_local, ctrl_idx_local = split_cases_controls(
                y_inc_tr, y_prev_tr, y_ctrl_tr,
            )
            matched_ctrl_local = match_controls(
                case_idx_local, ctrl_idx_local, self.cfg.controls_per_case, fold_rng,
            )

            # Build global training indices (case + matched controls).
            tr_local_selected = np.concatenate([case_idx_local, matched_ctrl_local])
            tr_global_selected = tr_idx[tr_local_selected]

            X_tr = X.iloc[tr_global_selected].reset_index(drop=True)
            # y for training: 1 for all cases (incident + prevalent), 0 for controls.
            y_tr_arr = np.zeros(len(tr_local_selected), dtype=int)
            y_tr_arr[: len(case_idx_local)] = 1
            y_tr = pd.Series(y_tr_arr, name=y_label.name or "y")

            n_train_cases = int(len(case_idx_local))
            n_train_controls = int(len(matched_ctrl_local))

            # --- Test fold: incident cases + controls only -------------------
            y_inc_te = y_incident.iloc[te_idx]
            y_ctrl_te = y_control.iloc[te_idx]
            test_mask = (y_inc_te.values.astype(bool) | y_ctrl_te.values.astype(bool))
            te_global_filtered = te_idx[test_mask]

            X_te = X.iloc[te_global_filtered].reset_index(drop=True)
            y_te = y_incident.iloc[te_global_filtered].astype(int).reset_index(drop=True)

            n_test_cases = int(y_te.sum())
            n_test_controls = int((y_te == 0).sum())

            if len(np.unique(y_te)) < 2:
                warnings.warn(
                    f"Fold {fold_idx} test set is single-class after filtering. "
                    "Skipping fold.",
                    stacklevel=2,
                )
                continue

            logger.info(
                "Fold %d | train: %d cases + %d controls | "
                "test: %d incident + %d controls",
                fold_idx, n_train_cases, n_train_controls,
                n_test_cases, n_test_controls,
            )

            # --- Imputer: fit on matched training set only ------------------
            # Import IMPUTATION_ENABLED from pipeline module
            from .pipeline import IMPUTATION_ENABLED
            outer_imp = ImputerPipeline(
                tte_strategy=_imputer_tte_strategy,
                random_state=self.cfg.random_state + fold_idx,
                enabled=IMPUTATION_ENABLED,
            ).fit(X_tr, y_tr)
            X_tr_imp = outer_imp.transform(X_tr)
            X_te_imp = outer_imp.transform(X_te)

            # --- Optuna inner tuning ----------------------------------------
            best_params = (
                self._tune(X_tr_imp, y_tr)
                if self.model_cls.supports_optuna
                else {}
            )

            # --- Final refit on full matched training set -------------------
            final_model = self.model_cls(
                params=best_params, random_state=self.cfg.random_state,
            )
            sw = _compute_sample_weight(y_tr) if final_model.supports_sample_weight else None
            final_model.fit(X_tr_imp, y_tr, sample_weight=sw)

            proba = final_model.predict_proba(X_te_imp)
            metrics = compute_all_metrics(y_te.values, proba)

            result.folds.append(FoldResult(
                fold=fold_idx,
                metrics=metrics,
                best_params=best_params,
                val_indices=np.asarray(te_global_filtered),
                val_proba=proba,
                val_true=y_te.values.astype(int),
                n_train_cases=n_train_cases,
                n_train_controls=n_train_controls,
                n_test_cases=n_test_cases,
                n_test_controls=n_test_controls,
            ))

        return result

    # -----------------------------------------------------------------------
    # Inner loop (identical to NestedCVTrainer._tune)
    # -----------------------------------------------------------------------

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Run Optuna over the inner stratified CV. Maximise mean PR-AUC.

        Inner CV uses pre-imputed data (outer imputer was fit on outer-train
        only, so no leakage). This trades minor distributional precision for
        a 3× speed-up in the inner loop.
        """
        inner = StratifiedKFold(
            n_splits=self.cfg.n_inner,
            shuffle=True,
            random_state=self.cfg.random_state + 1,
        )

        def objective(trial: optuna.Trial) -> float:
            template = self.model_cls(random_state=self.cfg.random_state)
            params = template.get_optuna_params(trial)
            scores: list[float] = []
            for itr_idx, ite_idx in inner.split(X, y):
                Xi_tr, Xi_te = X.iloc[itr_idx], X.iloc[ite_idx]
                yi_tr, yi_te = y.iloc[itr_idx], y.iloc[ite_idx]
                model = self.model_cls(params=params, random_state=self.cfg.random_state)
                sw = _compute_sample_weight(yi_tr) if model.supports_sample_weight else None
                model.fit(Xi_tr, yi_tr, sample_weight=sw)
                p = model.predict_proba(Xi_te)
                if len(np.unique(yi_te)) < 2:
                    continue
                scores.append(average_precision_score(yi_te, p))
            return float(np.mean(scores)) if scores else 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.random_state),
        )
        study.optimize(objective, n_trials=self.cfg.n_trials, show_progress_bar=False)
        return dict(study.best_params)
