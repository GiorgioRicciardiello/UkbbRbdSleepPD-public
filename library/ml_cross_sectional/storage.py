"""
storage.py
==========

Persist results from one model run to ``results/ml_cross_sectional/{run_id}/``.

Layout per run
--------------
::

    results/ml_cross_sectional/<model>_<timestamp>/
        cohort_stats.json
        feature_stats.csv
        metrics_per_fold.csv
        mean_metrics.csv
        best_hyperparams.json
        confusion_matrices.json
        shap_summary.csv
        permutation_importance.csv
        predictions_per_fold.csv
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .explainability import PermutationResult, ShapResult
from .metrics import CohortStats
from .report.distribution import feature_distribution_by_class, fold_composition_table, oof_distribution_summary
from .training import NestedCVResult


@dataclass
class ResultsWriter:
    """Writes one model's results into a timestamped subdirectory."""

    base_dir: Path
    model_name: str
    timestamp: str = ""
    run_dir: Path = Path()

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.model_name}_{self.timestamp}"
        self.run_dir = Path(self.base_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Individual writers
    # ------------------------------------------------------------------

    def write_cohort_stats(self, cohort: CohortStats) -> None:
        """Cohort summary → ``cohort_stats.json`` + ``feature_stats.csv``."""
        meta: dict = {
            "n_subjects": cohort.n_subjects,
            "n_cases": cohort.n_cases,
            "n_controls": cohort.n_controls,
            "prevalence": cohort.prevalence,
        }
        if cohort.n_incident is not None:
            meta["n_incident"] = cohort.n_incident
        if cohort.n_prevalent is not None:
            meta["n_prevalent"] = cohort.n_prevalent
        (self.run_dir / "cohort_stats.json").write_text(json.dumps(meta, indent=2))
        cohort.feature_stats.to_csv(self.run_dir / "feature_stats.csv")

    def write_nested_cv(self, result: NestedCVResult) -> None:
        """Per-fold metrics, mean metrics, best params, predictions."""
        result.metrics_frame().to_csv(self.run_dir / "metrics_per_fold.csv", index=False)
        result.mean_metrics().to_csv(self.run_dir / "mean_metrics.csv")

        # Best hyperparams per fold.
        params = {f"fold_{fr.fold}": fr.best_params for fr in result.folds}
        (self.run_dir / "best_hyperparams.json").write_text(json.dumps(params, indent=2, default=str))

        # Confusion matrices per fold.
        cms = {
            f"fold_{fr.fold}": {
                "tp": fr.metrics.tp,
                "fp": fr.metrics.fp,
                "tn": fr.metrics.tn,
                "fn": fr.metrics.fn,
                "threshold": fr.metrics.threshold,
            }
            for fr in result.folds
        }
        (self.run_dir / "confusion_matrices.json").write_text(json.dumps(cms, indent=2))

        # Long-format predictions for downstream calibration / ROC plots.
        rows: list[dict[str, Any]] = []
        for fr in result.folds:
            for idx, p, t in zip(fr.val_indices, fr.val_proba, fr.val_true):
                rows.append({"fold": fr.fold, "row_index": int(idx),
                             "y_true": int(t), "y_pred_proba": float(p)})
        pd.DataFrame(rows).to_csv(self.run_dir / "predictions_per_fold.csv", index=False)

    def write_shap(self, shap_result: ShapResult) -> None:
        """
        Write SHAP artifacts.

        * ``shap_summary.csv``     — mean |SHAP| per feature.
        * ``shap_values.npy``      — raw (n_samples × n_features) array.
        * ``shap_X_eval.csv``      — raw feature matrix SHAP was computed on.
        """
        df = shap_result.feature_importance.copy()
        df["available"] = shap_result.available
        df["note"] = shap_result.note
        df.to_csv(self.run_dir / "shap_summary.csv", index=False)

        if shap_result.available and shap_result.shap_values is not None:
            np.save(self.run_dir / "shap_values.npy", shap_result.shap_values)
        if shap_result.available and shap_result.X_eval is not None:
            shap_result.X_eval.to_csv(self.run_dir / "shap_X_eval.csv", index=False)

    def write_permutation(self, perm: PermutationResult) -> None:
        """Permutation importance CSV."""
        perm.importance.to_csv(self.run_dir / "permutation_importance.csv", index=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def write_distribution_reports(
        self,
        result: NestedCVResult,
        X: "pd.DataFrame",
        y: "pd.Series",
    ) -> None:
        """
        Write distribution reports alongside the standard artifacts.

        Files written:
        * ``feature_distribution_by_class.csv`` — mean/SD/missing per feature
          stratified by case (y=1) vs control (y=0), plus SMD.
        * ``fold_composition.csv`` — per-fold n_train/n_test cases/controls.
        * ``oof_distribution.csv`` — OOF calibration summary.
        """
        feat_dist = feature_distribution_by_class(X, y)
        feat_dist.to_csv(self.run_dir / "feature_distribution_by_class.csv")

        fold_comp = fold_composition_table(result)
        fold_comp.to_csv(self.run_dir / "fold_composition.csv", index=False)

        oof = oof_distribution_summary(result)
        oof.to_csv(self.run_dir / "oof_distribution.csv", index=False)

    def save_all(
        self,
        result: NestedCVResult,
        cohort: CohortStats,
        shap_result: ShapResult | None = None,
        perm: PermutationResult | None = None,
        X: "pd.DataFrame | None" = None,
        y: "pd.Series | None" = None,
    ) -> Path:
        """Write everything in one call. Returns the run directory."""
        self.write_cohort_stats(cohort)
        self.write_nested_cv(result)
        if shap_result is not None:
            self.write_shap(shap_result)
        if perm is not None:
            self.write_permutation(perm)
        if X is not None and y is not None:
            self.write_distribution_reports(result, X, y)
        return self.run_dir
