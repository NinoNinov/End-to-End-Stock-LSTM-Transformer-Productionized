"""Pydantic response models for the stock prediction API.

Phase 5: only response schemas. There is no request body — predictions are
parameter-less (the ticker is in the URL) and the server pulls the latest
cached data itself.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class _Resp(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PredictionResponse(_Resp):
    """Response for GET /predict/{ticker}.

    `expected_return_pct` is in *percent units* (e.g. 0.32 means +0.32%) for
    direct display; multiply by 0.01 to get the raw fraction the model emits.
    `implied_close = last_close * (1 + expected_return_pct/100)` and is
    display-only.
    """

    ticker: str
    ticker_key: str
    display_name: str
    last_bar_date: str
    last_close: float
    predicted_for_date: str
    direction: Literal["up", "down"]
    direction_prob: float
    direction_threshold: float
    expected_return_pct: float
    implied_close: float
    model_kind: Literal["transformer", "xgb"]
    last_train_date: Optional[str] = None
    holdout_accuracy: float
    holdout_naive_accuracy: float
    holdout_return_mae: float


class TickerMetadata(_Resp):
    """Per-ticker block returned by GET /metadata."""

    ticker_key: str
    ticker_symbol: str
    display_name: str
    model_kind: Literal["transformer", "xgb"]
    seq_len: Optional[int] = None
    feature_order: list[str]
    holdout_metrics: dict
    train_window: dict
    last_train_date: Optional[str] = None
    stl_outlier_removal_applied: Optional[bool] = None
    honesty_gate: Optional[dict] = None


class HistoryRow(_Resp):
    """One past prediction (populated by Phase 7's tools/log_prediction.py)."""

    predicted_at: str
    predicted_for_date: str
    direction: Literal["up", "down"]
    direction_prob: float
    expected_return_pct: float
    implied_close: float
    model_kind: Literal["transformer", "xgb"]
    last_train_date: Optional[str] = None
    realized_close: Optional[float] = None
    realized_return_pct: Optional[float] = None
    was_correct: Optional[bool] = None
