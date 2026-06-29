# Focal Loss Implementation Notes

## Current Status

**Focal loss is currently DISABLED** due to implementation issues. This document outlines the problem and the correct approach for future work.

## The Problem with Post-Hoc Retraining

The initial implementation attempted focal loss retraining AFTER inner CV hyperparameter tuning:

```python
# INCORRECT APPROACH (causes overfitting):
best_model = inner_cv_select_best_hyperparams()  # Tuned for standard BCE loss
y_pred_train = best_model.predict_proba(X_train)
focal_weights = compute_focal_weights(y_pred_train)
best_model.refit(X_train, y_train, sample_weight=focal_weights)  # Retrains with different objective!
```

**Why this fails**:
1. Inner CV tunes hyperparameters (max_depth, learning_rate, etc.) to minimize PR-AUC of standard BCE loss
2. Retraining with focal loss weights changes the objective function → selected hyperparams no longer optimal
3. Model overfits to the retraining data (same training set, new objective)
4. Test performance collapses (ROC-AUC drops from 0.83 to 0.20)

**Evidence from run 20260415_160000**:
```
Standard (baseline):  ROC-AUC = 0.828 ± 0.030
Focal (post-hoc):     ROC-AUC = 0.220 ± 0.040  ← Massive collapse!
```

## Correct Implementation Approaches

### Approach 1: Focal Loss in XGBoost Objective (Recommended)

Integrate focal loss as XGBoost's custom objective function during inner CV training:

```python
from xgboost import XGBClassifier
from src.screening.focal_loss import focal_loss_objective, focal_loss_metric

xgb = XGBClassifier(
    objective=focal_loss_objective,  # Custom objective
    custom_metric=focal_loss_metric,
    eval_metric="aucpr",
    tree_method="hist",
)

# Inner CV will now tune hyperparameters to minimize focal loss
search = RandomizedSearchCV(
    estimator=xgb,
    cv=inner_cv,
    scoring="focal_loss",  # Use focal loss for inner CV selection
)
```

**Advantages**:
- Hyperparameters optimized for focal loss objective
- No data leakage (inner CV uses focal loss throughout)
- Proper probabilistic calibration

**Challenges**:
- XGBoost custom objectives must return (gradients, hessians)
- Requires careful numerical stability (log, exp operations)
- Testing needed to ensure convergence

### Approach 2: Focal Loss via Sample Weights (Simpler)

Apply focal loss weights DURING inner CV, not after:

```python
# Before inner CV, compute focal loss weights
focal_weights = compute_focal_weights(y_train, y_pred_initial)

# Use these weights throughout inner CV
search = RandomizedSearchCV(...).fit(
    X_train, y_train,
    sample_weight=focal_weights  # Fixed throughout inner CV
)
```

**Advantages**:
- Simpler to implement
- Uses XGBoost's standard BCE loss with weighted samples
- No custom objective needed

**Challenges**:
- Requires pre-trained model to compute initial y_pred (circular dependency)
- May still have slight objective mismatch

### Approach 3: Two-Stage Training

Stage 1: Standard inner CV to select hyperparams  
Stage 2: Use selected hyperparams with focal loss objective on full training set:

```python
# Stage 1: Select hyperparams with standard loss
best_hyperparams = inner_cv_search(X_train, y_train, loss="standard")

# Stage 2: Train final model with focal loss + selected hyperparams
final_model = XGBClassifier(
    objective=focal_loss_objective,
    **best_hyperparams,
)
final_model.fit(X_train, y_train)
```

**Advantages**:
- Hyperparams selected for generalizable loss
- Final model uses focal loss for hard-example focus
- No retraining overfitting

**Challenges**:
- Hyperparams may not be optimal for focal loss
- Adds complexity to pipeline

## Recommended Path Forward

**Priority 1**: Implement Approach 1 (Focal Loss Custom Objective)
- Most principled solution
- Requires implementing focal_loss_objective with proper gradients/hessians
- Expected ROI: +0.01–0.03 ROC-AUC

**Priority 2**: Fallback to Approach 3 (Two-Stage)
- Simpler than Approach 1
- Better than current disabled state
- Expected ROI: +0.005–0.015 ROC-AUC

**Not Recommended**: Approach 2 (Sample Weights in Inner CV)
- Circular dependency issue
- Difficult to implement cleanly

## Implementation Checklist for Approach 1

- [ ] Implement proper `focal_loss_objective(y_pred, y_true)` returning (gradients, hessians)
- [ ] Handle numerical stability (clipping, log operations)
- [ ] Validate gradients/hessians with finite differences
- [ ] Create `focal_loss_metric` for XGBoost evaluation
- [ ] Test on synthetic data (ensure convergence)
- [ ] Test on actual data (verify improvement)
- [ ] Benchmark against baseline (target: +0.02 ROC-AUC)
- [ ] Document configuration parameters (alpha, gamma)
- [ ] Add to config.py with toggle flag

## References

- Lin et al. (2017). "Focal Loss for Dense Object Detection" (CVPR)
  - Original focal loss paper
  - Uses classification loss: FL(p_t) = -α(1-p_t)^γ log(p_t)

- XGBoost Custom Objectives
  - https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
  - Requires: logit output, return gradients and hessians

- SKLearn Custom Loss Metrics
  - https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object
