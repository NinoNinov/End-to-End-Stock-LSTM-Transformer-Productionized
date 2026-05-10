"""Data layer: fetch, cache, feature engineering, holdout split, normalizer.

Implementations land in Phase 2. This file currently defines only signatures
so downstream phases can import-test their wiring against this contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fetch_history(ticker: str, years: int) -> pd.DataFrame:
    raise NotImplementedError("Phase 2")


def update_cache(ticker: str, cache_path: Path) -> int:
    raise NotImplementedError("Phase 2")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Phase 2")


def stl_outlier_mask(df: pd.DataFrame, sigma: float = 3.0) -> pd.Series:
    raise NotImplementedError("Phase 2")


def split_train_val_test(
    df: pd.DataFrame, holdout_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raise NotImplementedError("Phase 2")


def fit_normalizer(train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    raise NotImplementedError("Phase 2")


def apply_normalizer(
    df: pd.DataFrame, normalizer: dict, feature_cols: list[str]
) -> pd.DataFrame:
    raise NotImplementedError("Phase 2")


def make_sequences(
    df: pd.DataFrame, feature_cols: list[str], seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError("Phase 2")
