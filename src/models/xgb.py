"""XGBoost baseline (separate classifier for direction, regressor for return).

XGBoost has no native dual-target API, and the marginal cost of two models is
trivial — both fit in seconds on ~7.7k rows × 15 features. So we keep the two
heads as two independent estimators sharing the same flattened-feature input.

Inputs to ``.fit`` / ``.predict`` are a 2D array of shape ``(n_samples,
n_features)``. Use ``df[feature_cols].to_numpy()`` rather than the 3D
sequences ``src.data.make_sequences`` produces — the lag features in the
config (``lag_close_1/2``, ``lag_return_1/2``) already encode the short-term
history XGB cares about.
"""
from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier, XGBRegressor


def build_xgb_classifier(hparams: dict[str, Any]) -> XGBClassifier:
    """Direction head: predicts P(target_direction == 1)."""
    return XGBClassifier(
        n_estimators=int(hparams["n_estimators"]),
        max_depth=int(hparams["max_depth"]),
        learning_rate=float(hparams["learning_rate"]),
        subsample=float(hparams["subsample"]),
        colsample_bytree=float(hparams["colsample_bytree"]),
        random_state=int(hparams["random_state"]),
        eval_metric="logloss",
        tree_method="hist",
    )


def build_xgb_regressor(hparams: dict[str, Any]) -> XGBRegressor:
    """Return head: predicts the next-day pct change directly."""
    return XGBRegressor(
        n_estimators=int(hparams["n_estimators"]),
        max_depth=int(hparams["max_depth"]),
        learning_rate=float(hparams["learning_rate"]),
        subsample=float(hparams["subsample"]),
        colsample_bytree=float(hparams["colsample_bytree"]),
        random_state=int(hparams["random_state"]),
        eval_metric="rmse",
        tree_method="hist",
    )
