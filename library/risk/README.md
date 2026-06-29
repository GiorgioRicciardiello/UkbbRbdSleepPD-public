# Risk Group Stratification Methods

This module (`library/risk`) implements methods to stratify subjects into "Risk Groups" for neurodegenerative outcomes (PD, Dementia, etc.) based on their RBD (REM Sleep Behavior Disorder) probability scores.

## Overview of Approaches

There are two primary approaches to defining these risk groups:

1.  **Outcome-Specific**: Thresholds are optimized to predict a specific clinical diagnosis (e.g., Parkinson's Disease).
2.  **Outcome-Agnostic (RBD-Only)**: Thresholds are based solely on the statistical distribution of RBD probabilities in the population, irrespective of clinical outcomes.

---

## 1. Outcome-Specific Methods
*Implemented in `run_compute_risk_groups`*

These methods define "High Risk" by finding the optimal cut-off on the RBD probability curve that best discriminates between cases (e.g., PD Patients) and controls.

### Thresholding Techniques
*   **ROC (Receiver Operating Characteristic)**: Uses **Youden’s J statistic** (Sensitivity + Specificity - 1) to find the optimal balance between true positives and false positives.
*   **PR (Precision-Recall)**: Optimizes the F-measure (harmonic mean of Precision and Recall) specifically from the Precision-Recall curve. Often better for imbalanced datasets.
*   **F1-Score**: Explicitly maximizes the F1 score across all potential thresholds.
*   **Survival (Log-Rank)**: (*Experimental*) Selected to maximize the difference in survival curves between groups.
*   **Fixed Percentiles**:
    *   **2-Group**: High Risk = Top 10% (90th percentile) of the validation cases.
    *   **3-Group**: Low (<90%), Intermediate (90-99%), High (>99%).

### Usage Variations
*   **With Validation Split** (`ehr_diag_pd_rbd_val`): Thresholds are learned **only** from the designated validation fold (e.g., `val_pd == 1`). These thresholds are then applied to the rest of the data. This avoids circularity/leakage.
*   **All Data** (`ehr_diag_pd_rbd_all`): Thresholds are calculated using **all subjects**. This provides the "best possible" fit for descriptive purposes but includes training data (risk of overfitting).

---

## 2. Outcome-Agnostic (RBD-Only) Methods
*Implemented in `run_compute_risk_group_rbd_only`*

These methods do not "know" about PD or Dementia diagnoses. They simply categorize subjects based on how high their RBD probability is relative to the population.

### Thresholding Techniques
*   **Percentile (2 Groups)**: High Risk defined as the top 10% of the reference population.
*   **Percentile (3 Groups)**:
    *   **Low**: < 90th percentile
    *   **Intermediate**: 90th – 99th percentile
    *   **High**: > 99th percentile
*   **Quartiles**: Splits the population into 4 equal bins (Q1, Q2, Q3, Q4).

### Usage Variations
*   **Reference = Validation Set** (`ehr_diag_pd_rbd_only_val`): The 90th/99th percentile values are derived from the *validation set* distribution and applied to everyone.
*   **Reference = All Data** (`ehr_diag_pd_rbd_only_all`): The percentiles are calculated from the *entire dataset*.

---

## Folder Output Structure

The pipeline generates risk groups and saves them in `data/risk_thresholds/` (or similar configured location), organized by variation:

| Directory Name | Strategy | Data Used for Thresholds |
| :--- | :--- | :--- |
| `ehr_diag_pd_rbd_val` | Outcome-Specific | Validation Fold Only |
| `ehr_diag_pd_rbd_all` | Outcome-Specific | Entire Dataset |
| `ehr_diag_pd_rbd_only_val` | RBD-Only (Distribution) | Validation Fold Only |
| `ehr_diag_pd_rbd_only_all` | RBD-Only (Distribution) | Entire Dataset |

Each directory contains:
*   **`collection.json`**: A master file containing all thresholds for all methods.
*   **`risk_*.json`**: Individual metadata files for each method/metric.
*   **`*.parquet`**: (Optional) The dataframe containing the assigned risk groups.
