"""Training orchestration: fits a model, evaluates on holdout, writes artifacts.

Implementation lands in Phase 4.
"""
from __future__ import annotations

from typing import Literal


def train_ticker(
    ticker_key: str, model_kind: Literal["transformer", "xgb"]
) -> dict:
    raise NotImplementedError("Phase 4")
