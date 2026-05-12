"""Data layer: fetch, cache, feature engineering, holdout split, normalizer.

Implements the contract defined in PLAN.md Phase 2. Faithfully ports the
preprocessing logic from the original research notebook
(``research/Deep Learning for Stock Price Prediction with LSTM and Transformers.py``,
lines 332-390) and adds a leakage-free normalizer (fit on train only).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# `statsmodels` is heavy (pulls in scipy + OpenBLAS); only imported lazily inside
# `stl_outlier_mask` so the hot path (fetch / features / split / sequences)
# doesn't pay for it.

logger = logging.getLogger(__name__)


_YF_DROP = ["Dividends", "Stock Splits", "Capital Gains"]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Strip yfinance extras, drop tz, normalize to date, sort, dedupe."""
    keep = [c for c in df.columns if c not in _YF_DROP]
    df = df[keep].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_history(ticker: str, years: int) -> pd.DataFrame:
    """Pull `years` of daily OHLCV from yfinance and return a clean dataframe."""
    raw = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    return _normalize_ohlcv(raw)


def update_cache(ticker: str, cache_path: Path, years: int = 35) -> int:
    """Append any new rows from yfinance into the parquet at cache_path.

    Idempotent: if the cache is up to date, returns 0 and leaves the file
    untouched. On an empty cache, performs a full backfill of `years` years.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        full = fetch_history(ticker, years)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(cache_path)
        return len(full)

    existing = pd.read_parquet(cache_path)
    last_date = existing.index.max()
    start = (last_date + pd.Timedelta(days=1)).date()
    raw = yf.Ticker(ticker).history(start=start.isoformat(), auto_adjust=True)
    if raw.empty:
        return 0
    new = _normalize_ohlcv(raw)
    new = new[new.index > last_date]
    if new.empty:
        return 0
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(cache_path)
    return len(new)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def engineer_features(
    df: pd.DataFrame, ma_window: int = 20, *, include_targets: bool = True
) -> pd.DataFrame:
    """Add the 15 model features (+ 2 target columns when training). Drops NaN rows.

    Ports the notebook's `preprocess_data` with three deliberate changes:
      1. Moving-average window is `ma_window` (default 20) instead of the
         notebook's `va_window=100` — the notebook column is named SMA_20
         but computed a 100-day MA. We honour the column name.
      2. Spread features use Close as denominator (matches PLAN.md), not
         Open/High as in the notebook. Both are reasonable; we follow the
         contract.
      3. Adds two target columns: `target_return` (next-day pct change) and
         `target_direction` (1 if up, 0 if flat/down). Replaces the
         notebook's absolute-price regression target.

    Set `include_targets=False` for inference: the most recent row has no
    known next-day return, but it IS the row whose features we want to feed
    the model. Skipping the target columns means dropna() only trims the
    rolling-window warmup at the head and keeps the latest bar.
    """
    out = df.copy()

    out["Volume_M"] = out["Volume"] / 1_000_000.0

    out["SMA_20"] = out["Close"].rolling(window=ma_window).mean()
    out["EMA_20"] = out["Close"].ewm(span=ma_window, adjust=False).mean()
    out["RSI"] = _compute_rsi(out["Close"])

    out.index = pd.to_datetime(out.index)
    out["day_of_week"] = out.index.dayofweek

    out["lag_close_1"] = out["Close"].shift(1)
    out["lag_close_2"] = out["Close"].shift(2)
    daily_return = out["Close"].pct_change()
    out["lag_return_1"] = daily_return.shift(1)
    out["lag_return_2"] = daily_return.shift(2)

    out["Spread_OC"] = (out["Open"] - out["Close"]) / out["Close"]
    out["Spread_HL"] = (out["High"] - out["Low"]) / out["Close"]

    keep_cols = [
        "Close", "Open", "High", "Low", "Volume_M",
        "SMA_20", "EMA_20", "RSI",
        "lag_close_1", "lag_close_2", "lag_return_1", "lag_return_2",
        "Spread_OC", "Spread_HL", "day_of_week",
    ]
    if include_targets:
        out["target_return"] = out["Close"].pct_change().shift(-1)
        out["target_direction"] = (out["target_return"] > 0).astype("int8")
        keep_cols = keep_cols + ["target_return", "target_direction"]

    out = out[keep_cols]
    out = out.dropna()
    return out


def stl_outlier_mask(df: pd.DataFrame, sigma: float = 3.0) -> pd.Series:
    """Boolean Series, True where Close residual exceeds `sigma * std(residual)`.

    Caller decides whether to drop these rows (driven by config flag
    `apply_stl_outlier_removal`). STL needs a regular-frequency series, so
    Close is reindexed to business-day frequency and forward-filled for the
    decomposition only — the returned mask is reindexed back to the input.
    """
    from statsmodels.tsa.seasonal import STL  # lazy: see module-level note

    series = df["Close"].asfreq("B").ffill()
    stl = STL(series, seasonal=5, robust=True)
    residual = stl.fit().resid
    threshold = sigma * float(np.std(residual))
    mask = (residual.abs() > threshold).reindex(df.index, fill_value=False)
    return mask


def split_train_val_test(
    df: pd.DataFrame, holdout_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve off the last `holdout_days` as test, then split the rest 90/10.

    Chronological split — never shuffle a time series. The holdout is the
    only data the model has truly never seen; honest evaluation depends on
    not touching it during training, validation, normalization, or feature
    selection.
    """
    if len(df) <= holdout_days + 1:
        raise ValueError(
            f"Not enough rows ({len(df)}) for a {holdout_days}-day holdout"
        )
    test = df.iloc[-holdout_days:]
    pre = df.iloc[:-holdout_days]
    train_end = int(len(pre) * 0.9)
    train = pre.iloc[:train_end]
    val = pre.iloc[train_end:]
    return train, val, test


def fit_normalizer(train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute per-feature mean and std on TRAIN rows only.

    Returns a JSON-serializable {col: {"mean": float, "std": float}} dict so
    it can sit alongside the model artifact in metadata.json.
    """
    norm: dict = {}
    for col in feature_cols:
        series = train_df[col].astype("float64")
        std = float(series.std(ddof=0))
        if std == 0:
            std = 1.0  # constant column: don't blow up at apply time
        norm[col] = {"mean": float(series.mean()), "std": std}
    return norm


def apply_normalizer(
    df: pd.DataFrame, normalizer: dict, feature_cols: list[str]
) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        mean = normalizer[col]["mean"]
        std = normalizer[col]["std"]
        out[col] = (out[col] - mean) / std
    return out


def make_sequences(
    df: pd.DataFrame, feature_cols: list[str], seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window of length `seq_len` across rows; pair each window with
    its terminal-row targets. Output shapes: X (n, seq_len, n_features),
    y_return (n,), y_direction (n,)."""
    if len(df) < seq_len:
        raise ValueError(
            f"Not enough rows ({len(df)}) to build a single seq_len={seq_len} window"
        )
    feats = df[feature_cols].to_numpy(dtype="float32")
    y_ret = df["target_return"].to_numpy(dtype="float32")
    y_dir = df["target_direction"].to_numpy(dtype="float32")
    windows = np.lib.stride_tricks.sliding_window_view(
        feats, window_shape=seq_len, axis=0
    )  # shape: (n, n_features, seq_len)
    X = windows.transpose(0, 2, 1).copy()  # -> (n, seq_len, n_features)
    return X, y_ret[seq_len - 1:], y_dir[seq_len - 1:]
