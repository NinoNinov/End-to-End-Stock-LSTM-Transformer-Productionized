"""Inference: load production artifacts, predict next-day direction + return.

The artifacts written by Phase 4 (`artifacts/{ticker_key}_production.{keras|joblib}`
plus `..._production_metadata.json`) carry everything inference needs:
feature order, train-fit normalizer, seq_len, calibrated direction threshold,
and the holdout metrics we surface to the UI for transparency.

A `ModelHandle` is loaded once at process start (FastAPI lifespan) and
reused across requests; `predict(ticker_key, handle)` is the per-request
hot path.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import TICKERS
from src.data import apply_normalizer, engineer_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class ModelHandle:
    """Loaded production model + metadata + mtime stamp for hot-reload checks."""

    ticker_key: str
    model_kind: Literal["transformer", "xgb"]
    model: Any                          # tf.keras.Model OR (XGBClassifier, XGBRegressor) tuple
    metadata: dict
    artifact_path: Path
    artifact_mtime: float


def _artifact_paths(ticker_key: str) -> tuple[Path, Path, Path]:
    """Return (keras_path, joblib_path, metadata_path) — only one of the first two exists."""
    base = ARTIFACT_DIR / f"{ticker_key}_production"
    return base.with_suffix(".keras"), base.with_suffix(".joblib"), Path(f"{base}_metadata.json")


def load_handle(ticker_key: str) -> ModelHandle:
    """Load production artifact + sibling metadata.json for one ticker."""
    keras_path, joblib_path, meta_path = _artifact_paths(ticker_key)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing production metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = metadata.get("model_kind")

    if kind == "transformer":
        if not keras_path.exists():
            raise FileNotFoundError(f"Missing production model: {keras_path}")
        # Lazy import — TF is heavy and we only want to pay for it once per worker.
        from tensorflow import keras  # type: ignore

        model = keras.models.load_model(keras_path, compile=False)
        artifact_path = keras_path
    elif kind == "xgb":
        if not joblib_path.exists():
            raise FileNotFoundError(f"Missing production model: {joblib_path}")
        import joblib

        # XGB metadata stores two estimators side-by-side; train.py serialized them
        # as a (clf, reg) tuple under the same joblib file (Phase 4 convention).
        model = joblib.load(joblib_path)
        artifact_path = joblib_path
    else:
        raise ValueError(f"Unknown model_kind in metadata: {kind!r}")

    return ModelHandle(
        ticker_key=ticker_key,
        model_kind=kind,
        model=model,
        metadata=metadata,
        artifact_path=artifact_path,
        artifact_mtime=artifact_path.stat().st_mtime,
    )


def maybe_reload(handle: ModelHandle) -> ModelHandle:
    """Reload the model in place if its artifact has changed on disk.

    Used by FastAPI on each /predict/* call so a fresh weekly retrain is
    picked up without restarting the server. mtime comparison is enough for
    a single-writer (retrain script) / single-reader (API) setup.
    """
    try:
        current_mtime = handle.artifact_path.stat().st_mtime
    except FileNotFoundError:
        return handle
    if current_mtime > handle.artifact_mtime:
        logger.info("Artifact for %s changed on disk; reloading", handle.ticker_key)
        return load_handle(handle.ticker_key)
    return handle


def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    """Return the next business day after `d` (rough proxy for next trading session).

    Doesn't honour US market holidays — close enough for a display field.
    The model itself doesn't depend on this; it only feeds the UI's
    `predicted_for_date`.
    """
    return (d + pd.tseries.offsets.BDay(1)).normalize()


def predict(ticker_key: str, handle: ModelHandle | None = None) -> dict:
    """Predict next-day direction + return for `ticker_key`.

    Reads the cached parquet, engineers features (without targets — the
    most recent bar's target is unknown precisely because it's what we
    want to predict), normalizes using the saved train-fit normalizer,
    and runs the production model. Returns a JSON-serializable dict
    matching the API's `PredictionResponse` schema.

    `handle` is optional; if not provided, the artifact is loaded fresh
    on each call (slow — only do this from one-off scripts).
    """
    if ticker_key not in TICKERS:
        raise KeyError(f"Unknown ticker key: {ticker_key}")

    if handle is None:
        handle = load_handle(ticker_key)

    meta = handle.metadata
    cfg = TICKERS[ticker_key]
    feature_cols = meta["feature_order"]
    normalizer = meta["normalizer"]
    seq_len = meta.get("seq_len")  # None for xgb

    cache_path = DATA_DIR / f"{ticker_key}.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached data for {ticker_key} at {cache_path}. "
            f"Run `python -m tools.fetch_market_data` first."
        )
    raw = pd.read_parquet(cache_path)

    feats = engineer_features(raw, include_targets=False)
    if len(feats) == 0:
        raise RuntimeError(f"engineer_features produced no rows for {ticker_key}")

    last_close = float(feats["Close"].iloc[-1])
    last_bar_date = pd.Timestamp(feats.index[-1])

    normed = apply_normalizer(feats, normalizer, feature_cols)

    if handle.model_kind == "transformer":
        if len(normed) < seq_len:
            raise RuntimeError(
                f"Need at least {seq_len} rows for inference; got {len(normed)}"
            )
        window = normed[feature_cols].to_numpy(dtype="float32")[-seq_len:]
        X = window[np.newaxis, ...]  # (1, seq_len, n_features)
        out = handle.model.predict(X, verbose=0)
        # Keras' functional model with named outputs returns a dict on .predict
        # when called this way; gracefully handle the list fallback too.
        if isinstance(out, dict):
            p_up = float(np.asarray(out["direction"]).ravel()[0])
            ret_pred = float(np.asarray(out["return"]).ravel()[0])
        else:
            p_up = float(np.asarray(out[0]).ravel()[0])
            ret_pred = float(np.asarray(out[1]).ravel()[0])
    elif handle.model_kind == "xgb":
        # Phase 4 serializes a {"classifier": ..., "regressor": ...} dict per joblib.
        row = normed[feature_cols].to_numpy(dtype="float32")[-1:].copy()
        if isinstance(handle.model, dict):
            clf = handle.model["classifier"]
            reg = handle.model["regressor"]
        else:  # tuple fallback
            clf, reg = handle.model
        p_up = float(clf.predict_proba(row)[0, 1])
        ret_pred = float(reg.predict(row)[0])
    else:
        raise ValueError(f"Unknown model_kind: {handle.model_kind!r}")

    threshold = float(meta["holdout_metrics"].get("direction_threshold", 0.5))
    direction = "up" if p_up >= threshold else "down"

    implied_close = last_close * (1.0 + ret_pred)

    holdout = meta["holdout_metrics"]
    return {
        "ticker": cfg.ticker,
        "ticker_key": ticker_key,
        "display_name": cfg.display_name,
        "last_bar_date": last_bar_date.date().isoformat(),
        "last_close": round(last_close, 4),
        "predicted_for_date": _next_trading_day(last_bar_date).date().isoformat(),
        "direction": direction,
        "direction_prob": round(p_up, 4),
        "direction_threshold": round(threshold, 4),
        "expected_return_pct": round(ret_pred * 100.0, 4),  # percent units for UI
        "implied_close": round(implied_close, 4),
        "model_kind": handle.model_kind,
        "last_train_date": meta.get("last_train_date"),
        "holdout_accuracy": round(
            float(holdout.get("direction_accuracy_at_0_5", 0.0)), 4
        ),
        "holdout_naive_accuracy": round(
            float(holdout.get("naive_always_up_accuracy", 0.0)), 4
        ),
        "holdout_return_mae": round(float(holdout.get("return_mae", 0.0)), 6),
    }
