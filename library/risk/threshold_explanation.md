# Threshold Calculation: Outcome-Dependent vs. Outcome-Agnostic

## Summary
There is a **computational difference** depending on the method used.

*   **Distribution-Based Methods (Percentile, Quartile):** The outcome labels (Case vs Control) are **NOT computed** to determine the threshold values. The thresholds depend only on the distribution of predicted probabilities within the defined validation cohort.
*   **Performance-Based Methods (ROC, PR, F1, Survival):** The outcome labels **ARE essential**. The thresholds are calculated to optimize a specific metric (e.g., sensitivity/specificity) by comparing predictions against the actual outcomes.

---

## Detailed Breakdown

### 1. Data-Driven / Distribution-Based Methods
**Methods:** `Percentile (2g/3g)`, `Quartile`

These methods determine thresholds based on the statistical distribution of the predicted probabilities (e.g., "Top 10% of risk scores").

*   **How outcomes are used:**
    *   **Computation:** The actual disease status (Outcome=1 vs 0) is **ignored** during the calculation of the threshold value.
    *   **Validation Cohort:** The code filters for the "Validation Set" (`val_flag == True`). If your validation flags differ between outcomes (e.g., `val_pd` vs `val_dlb`), the underlying population changes, which *will* slightly shift the distribution and thus the thresholds. However, if the validation set is constant, the thresholds will be identical across all outcomes.
*   **Computational Difference:** None relative to the label. The calculation is purely `np.percentile(probabilities, 90)`.

### 2. Outcome-Optimized / Performance-Based Methods
**Methods:** `ROC`, `PR` (Precision-Recall), `F1`, `Survival`

These methods explicitly search for a threshold that maximizes separation or prediction accuracy for a specific disease.

*   **How outcomes are used:**
    *   **Computation:** The algorithms iterate through possible thresholds to find the "optimal" one by comparing **Predicted Risk vs. Actual Outcome**.
        *   *ROC:* Maximizes Youden's Index (Sensitivity + Specificity - 1).
        *   *F1:* Maximizes the harmonic mean of Precision and Recall.
    *   **Effect:** The threshold is heavily dependent on the prevalence and "easiness" of classifying that specific outcome. A threshold optimized for Parkinson's Disease (PD) will likely be different from one optimized for Lewy Body Dementia (DLB).
*   **Computational Difference:** Significant. These require running an optimization loop (e.g., `roc_curve`) that takes both the probabilities scores `y_score` and the ground truth labels `y_true` as input.

### Comparison Table

| Feature | Percentile / Quartile | ROC / PR / F1 |
| :--- | :--- | :--- |
| **Input Data** | Probability Scores only | Probability Scores + **Outcome Labels** |
| **Goal** | Segment population (e.g. "Top 10%") | Maximize Accuracy / Separation |
| **Outcome Influence** | **None** (unless validation set changes) | **Direct** (defines the threshold) |
| **Outcome Agnostic?** | Yes | No |

## Practical Implication for your Code

If you run the **Outcome-Agnostic** function (`run_compute_risk_group_rbd_only`):
*   It computes thresholds (p90, p99) on the *entire* specified reference set.
*   It generates a single set of thresholds applicable to *any* outcome analysis.

If you run the **Outcome-Specific** function (`run_compute_risk_groups` -> `_compute_risk_groups_percentile...`):
*   It technically re-calculates the p90/p99 for *each* outcome loop.
*   **If the validation flags are identical**, the result is computationally identical to the agnostic version.
*   **If the validation flags differ** (e.g., different exclusion criteria for PD vs DLB), you will get slightly different "90th percentile" values for each outcome.
