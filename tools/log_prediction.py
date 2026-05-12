"""CLI: append today's predictions and backfill yesterday's outcomes.

Implements PLAN.md Phase 7 (daily logger):

1. For every ticker in ``src.config.TICKERS``:
   - Call ``src.inference.predict`` against the current production handle.
   - Append one row to ``data/predictions_{ticker_key}.parquet`` describing
     what the model said and when it said it.
2. Before the append, **backfill** any prior row whose realized outcome is
   now knowable — i.e. the cached market parquet has a bar dated after the
   row's ``last_bar_date_at_prediction``. We fill ``realized_*`` columns
   for every row that became resolvable since the last run.

The script is **idempotent per predicted_for_date**: re-running on the same
trading day appends nothing new. Re-running after a fresh data fetch (which
moved ``last_bar_date`` forward) writes a fresh prediction row, because the
new ``predicted_for_date`` differs.

Run::

    C:\\venvs\\stock-predictor\\.venv\\Scripts\\python.exe -m tools.log_prediction
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TICKERS  # noqa: E402  (path setup must come first)
from src.inference import load_handle, predict  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"

# Columns persisted per row. Order is significant for parquet readability.
ROW_COLUMNS = [
    "predicted_at",
    "predicted_for_date",
    "last_bar_date_at_prediction",
    "last_close_at_prediction",
    "direction",
    "direction_prob",
    "direction_threshold",
    "expected_return_pct",
    "implied_close",
    "model_kind",
    "last_train_date",
    # backfill (null at append, filled when the next trading day's bar lands)
    "realized_date",
    "realized_close",
    "realized_return_pct",
    "was_correct",
]


def _build_row(pred: dict) -> dict:
    """Map a ``src.inference.predict`` response into a parquet-friendly row."""
    return {
        "predicted_at": pd.Timestamp.now(tz="UTC"),
        "predicted_for_date": pd.Timestamp(pred["predicted_for_date"]),
        "last_bar_date_at_prediction": pd.Timestamp(pred["last_bar_date"]),
        "last_close_at_prediction": float(pred["last_close"]),
        "direction": str(pred["direction"]),
        "direction_prob": float(pred["direction_prob"]),
        "direction_threshold": float(pred["direction_threshold"]),
        "expected_return_pct": float(pred["expected_return_pct"]),
        "implied_close": float(pred["implied_close"]),
        "model_kind": str(pred["model_kind"]),
        "last_train_date": str(pred.get("last_train_date") or ""),
        "realized_date": pd.NaT,
        "realized_close": None,
        "realized_return_pct": None,
        "was_correct": None,
    }


def _empty_log() -> pd.DataFrame:
    """Empty DataFrame with the canonical column order + dtypes."""
    df = pd.DataFrame(columns=ROW_COLUMNS)
    # Explicit dtypes so an empty parquet roundtrip preserves them.
    df["predicted_at"] = pd.to_datetime(df["predicted_at"], utc=True)
    df["predicted_for_date"] = pd.to_datetime(df["predicted_for_date"])
    df["last_bar_date_at_prediction"] = pd.to_datetime(
        df["last_bar_date_at_prediction"]
    )
    df["realized_date"] = pd.to_datetime(df["realized_date"])
    return df


def _backfill(log: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Resolve realized outcomes for any row whose next-bar has now landed.

    For each row missing ``realized_close``, find the first market bar dated
    strictly after ``last_bar_date_at_prediction``. If such a bar exists in
    the cached parquet, write the realized fields. Rows whose next bar
    hasn't been fetched yet are left untouched and will be retried on the
    next run.

    Returns (updated_log, num_rows_backfilled).
    """
    if log.empty:
        return log, 0
    pending = log["realized_close"].isna()
    if not pending.any():
        return log, 0

    # Build a sorted index of available market dates once.
    market_dates = market.index.sort_values()
    backfilled = 0

    for idx in log.index[pending]:
        last_bar = log.at[idx, "last_bar_date_at_prediction"]
        if pd.isna(last_bar):
            continue
        # First market bar strictly after the prediction's reference date.
        pos = market_dates.searchsorted(last_bar, side="right")
        if pos >= len(market_dates):
            continue  # next bar hasn't been fetched yet
        realized_date = pd.Timestamp(market_dates[pos])
        realized_close = float(market.at[realized_date, "Close"])
        last_close = float(log.at[idx, "last_close_at_prediction"])
        if last_close <= 0:
            continue
        realized_return_pct = (realized_close / last_close - 1.0) * 100.0
        direction_predicted = str(log.at[idx, "direction"])
        was_correct = (direction_predicted == "up") == (realized_return_pct > 0)

        log.at[idx, "realized_date"] = realized_date
        log.at[idx, "realized_close"] = round(realized_close, 4)
        log.at[idx, "realized_return_pct"] = round(realized_return_pct, 4)
        log.at[idx, "was_correct"] = bool(was_correct)
        backfilled += 1

    return log, backfilled


def _append_idempotent(
    log: pd.DataFrame, row: dict
) -> tuple[pd.DataFrame, bool]:
    """Append ``row`` unless a row with the same predicted_for_date already exists.

    Same-day re-runs of the logger are a non-event; only a fresh data fetch
    (which moves ``predicted_for_date`` forward) produces a new row.
    """
    target = row["predicted_for_date"]
    if not log.empty and (log["predicted_for_date"] == target).any():
        return log, False
    new = pd.DataFrame([row], columns=ROW_COLUMNS)
    # Align dtypes on the appendee to silence pandas' future-warning churn.
    new["predicted_at"] = pd.to_datetime(new["predicted_at"], utc=True)
    new["predicted_for_date"] = pd.to_datetime(new["predicted_for_date"])
    new["last_bar_date_at_prediction"] = pd.to_datetime(
        new["last_bar_date_at_prediction"]
    )
    new["realized_date"] = pd.to_datetime(new["realized_date"])
    return pd.concat([log, new], ignore_index=True), True


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet via tmp+rename so a crash mid-write can't corrupt the log."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def log_one(ticker_key: str) -> dict:
    """Log a prediction for one ticker; backfill resolvable prior rows.

    Returns a small summary dict useful for the CLI/journal output.
    """
    market_path = DATA_DIR / f"{ticker_key}.parquet"
    log_path = DATA_DIR / f"predictions_{ticker_key}.parquet"
    if not market_path.exists():
        raise FileNotFoundError(
            f"No cached market data for {ticker_key} at {market_path}. "
            f"Run `python -m tools.fetch_market_data` first."
        )

    handle = load_handle(ticker_key)
    pred = predict(ticker_key, handle)

    log = pd.read_parquet(log_path) if log_path.exists() else _empty_log()

    market = pd.read_parquet(market_path)
    log, n_backfilled = _backfill(log, market)

    row = _build_row(pred)
    log, appended = _append_idempotent(log, row)

    _atomic_write_parquet(log, log_path)

    return {
        "ticker_key": ticker_key,
        "appended": appended,
        "backfilled": n_backfilled,
        "predicted_for_date": str(row["predicted_for_date"].date()),
        "direction": row["direction"],
        "direction_prob": row["direction_prob"],
        "rows_total": int(len(log)),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logging.info("log_prediction started at %s (UTC)", started)
    failures = 0
    for key in TICKERS:
        try:
            summary = log_one(key)
        except Exception as exc:
            logging.exception("%s: log_prediction failed (%s)", key, exc)
            failures += 1
            continue
        logging.info(
            "%s: appended=%s backfilled=%d for=%s dir=%s p=%.4f rows=%d",
            key,
            summary["appended"],
            summary["backfilled"],
            summary["predicted_for_date"],
            summary["direction"],
            summary["direction_prob"],
            summary["rows_total"],
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
