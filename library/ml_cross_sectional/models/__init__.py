"""
ml_cross_sectional.models
=========================

Concrete model wrappers around scikit-learn / XGBoost. Each model subclasses
``ModelBase`` and exposes a ``get_optuna_params`` method that defines its
hyperparameter search space. Add a new model by:

1. Creating ``new_model.py`` with a subclass of ``ModelBase``.
2. Importing it here and appending to ``ALL_MODELS``.
"""
from __future__ import annotations

from .base import ModelBase
from .elasticnet import ElasticNetModel
from .logistic import LogisticModel
from .random_forest import RandomForestModel
from .svm_model import SVMModel
from .xgboost_model import XGBoostModel

#: Registry of available models. ``pipeline.py`` iterates over this list.
ALL_MODELS: tuple[type[ModelBase], ...] = (
    LogisticModel,
    ElasticNetModel,
    XGBoostModel,
    SVMModel,
    RandomForestModel,
)

__all__ = [
    "ModelBase",
    "LogisticModel",
    "ElasticNetModel",
    "XGBoostModel",
    "SVMModel",
    "RandomForestModel",
    "ALL_MODELS",
]
