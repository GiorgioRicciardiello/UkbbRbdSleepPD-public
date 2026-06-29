"""
plots.py
========

Figure generation for ml_cross_sectional:

* ``plot_shap_importance_ci``  — bar chart of mean |SHAP| per feature with
  bootstrap 95% confidence intervals.
* ``plot_shap_interaction``    — scatter of ``shap[feature]`` vs feature
  value, coloured by an interaction feature (e.g. RBD x PRS_pd).
* ``plot_all_for_run``         — convenience wrapper called from the
  pipeline after each model fit.

The CIs are computed by bootstrap: we resample rows of the per-sample
SHAP matrix with replacement and recompute the mean absolute SHAP per
feature for each bootstrap draw. The 2.5 / 97.5 percentiles across
bootstrap draws give the CI. This reflects uncertainty in the summary
statistic across the held-out sample, NOT model uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend for batch runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .explainability import ShapResult


# --- Bootstrap helpers -------------------------------------------------------

def bootstrap_mean_abs_shap(
    shap_values: np.ndarray,
    n_boot: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap the mean |SHAP| per feature.

    Parameters
    ----------
    shap_values :
        Array of shape ``(n_samples, n_features)``.
    n_boot :
        Number of bootstrap resamples.
    random_state :
        Seed for reproducibility.

    Returns
    -------
    mean :
        Point estimate (mean of ``|shap_values|`` across rows), shape
        ``(n_features,)``.
    lo, hi :
        2.5 and 97.5 percentiles across bootstrap draws, same shape.
    """
    rng = np.random.default_rng(random_state)
    n, p = shap_values.shape
    abs_sv = np.abs(shap_values)
    mean = abs_sv.mean(axis=0)

    boot = np.empty((n_boot, p), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = abs_sv[idx].mean(axis=0)

    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return mean, lo, hi


# --- Plot: feature importance with CI ----------------------------------------

def plot_shap_importance_ci(
    shap_result: ShapResult,
    out_path: Path,
    top_n: int | None = None,
    n_boot: int = 1000,
    random_state: int = 42,
    title: str | None = None,
) -> Path | None:
    """
    Horizontal bar chart of mean |SHAP| with bootstrap 95% CI.

    Returns ``None`` (and does not create the file) if SHAP values are
    unavailable for this run.
    """
    if not shap_result.available or shap_result.shap_values is None or shap_result.X_eval is None:
        return None

    sv = shap_result.shap_values
    features = list(shap_result.X_eval.columns)
    mean, lo, hi = bootstrap_mean_abs_shap(sv, n_boot=n_boot, random_state=random_state)

    order = np.argsort(mean)  # ascending → largest at top of horizontal bar
    if top_n is not None:
        order = order[-top_n:]

    feat_sorted = [features[i] for i in order]
    mean_sorted = mean[order]
    # Clip to 0: with small test folds the bootstrap CI can be asymmetric so
    # lo > mean for some features, which matplotlib barh/errorbar rejects.
    err_lo = np.maximum(0.0, mean_sorted - lo[order])
    err_hi = np.maximum(0.0, hi[order] - mean_sorted)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(feat_sorted) + 1.5)))
    y_pos = np.arange(len(feat_sorted))
    ax.barh(y_pos, mean_sorted, xerr=[err_lo, err_hi],
            color="#4C72B0", edgecolor="black", alpha=0.85,
            error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1.0})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_sorted)
    ax.set_xlabel("Mean |SHAP value|  (95% bootstrap CI)")
    ax.set_title(title or "Feature importance (SHAP, bootstrap 95% CI)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- Plot: SHAP interaction --------------------------------------------------

def plot_shap_interaction(
    shap_result: ShapResult,
    feature: str,
    interaction_feature: str,
    out_path: Path,
    title: str | None = None,
) -> Path | None:
    """
    SHAP "dependence" scatter plot.

    * x-axis: raw value of ``feature``
    * y-axis: SHAP value for ``feature``
    * colour: raw value of ``interaction_feature``

    This visualises the marginal effect of ``feature`` on the model output,
    modulated by ``interaction_feature``. Vertical spread at a given x is
    the interaction signal.
    """
    if not shap_result.available or shap_result.shap_values is None or shap_result.X_eval is None:
        return None

    X = shap_result.X_eval
    sv = shap_result.shap_values
    if feature not in X.columns:
        return None
    if interaction_feature not in X.columns:
        return None

    f_idx = list(X.columns).index(feature)
    x_vals = X[feature].values.astype(float)
    y_vals = sv[:, f_idx]
    c_vals = X[interaction_feature].values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="viridis",
                    s=18, alpha=0.75, edgecolor="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(interaction_feature)
    ax.axhline(0.0, color="#999999", lw=0.8, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(title or f"SHAP dependence: {feature} (colored by {interaction_feature})")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- Convenience: generate all plots for one run ----------------------------

#: Interaction pairs we always compute when both features are present.
DEFAULT_INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("abk_rbd_score_mean", "prs_score_pd"),
    ("abk_rbd_score_mean", "cov_age_recruitment_21022"),
)


def plot_all_for_run(
    shap_result: ShapResult,
    run_dir: Path,
    interaction_pairs: Sequence[tuple[str, str]] = DEFAULT_INTERACTION_PAIRS,
) -> dict[str, Path]:
    """
    Generate the standard figure set for one model run. Figures are saved
    under ``run_dir/figures/``.

    Parameters
    ----------
    shap_result :
        Populated ``ShapResult`` (SHAP values required).
    run_dir :
        Output directory (the per-model run dir).
    interaction_pairs :
        Iterable of ``(feature, interaction_feature)`` pairs.

    Returns
    -------
    dict
        ``{figure_name: path}`` for each figure that was actually written.
    """
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    ci_path = plot_shap_importance_ci(
        shap_result, fig_dir / "shap_importance_ci.png",
        title="SHAP feature importance (bootstrap 95% CI)",
    )
    if ci_path is not None:
        written["importance_ci"] = ci_path

    for feat, inter in interaction_pairs:
        safe = f"shap_interaction_{feat}__x__{inter}.png"
        p = plot_shap_interaction(
            shap_result, feature=feat, interaction_feature=inter,
            out_path=fig_dir / safe,
        )
        if p is not None:
            written[f"{feat}__x__{inter}"] = p

    return written
