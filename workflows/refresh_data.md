# Workflow: Refresh market data

## Objective
Keep `data/{ticker_key}.parquet` up to date with the latest daily OHLCV from yfinance for every ticker in `src/config.py::TICKERS`. The downstream prediction and retrain pipelines read these parquet files directly — if the cache is stale, every prediction is stale.

## Inputs
- `src/config.py::TICKERS` — which tickers to refresh and how many years of history to backfill on first run.
- Network access to yfinance (no API key required, but rate limits apply).

## Tools to use
- `tools/fetch_market_data.py` — CLI entry point. Iterates over `TICKERS`, calls `src.data.update_cache` per ticker.

## How to invoke
```powershell
& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m tools.fetch_market_data
```
On the Contabo VM (Phase 8) this will be wrapped by a systemd timer.

## Expected outputs
- One parquet per ticker at `data/{ticker_key}.parquet` — the **config key**, not the yfinance symbol (so `data/SPX.parquet`, not `data/^GSPC.parquet`).
- One log line per ticker: `MSFT: cache up to YYYY-MM-DD, +N new rows`.
- Empty cache → full backfill of `cfg.history_years` years (~9k rows for 35y of MSFT or ^GSPC).
- Non-empty cache → fetches only `(last_cached_date, today]`; appends, dedupes, sorts.

## Edge cases
- **No new rows** (weekend, holiday, second run on the same day): logs `+0 new rows`, exits in <2s per ticker, parquet untouched.
- **yfinance failure** (network, rate limit, transient 5xx): logged at ERROR level, that ticker is skipped, the existing cache is left untouched. The CLI exits non-zero so a wrapping systemd unit can record the failure. Other tickers still attempt to refresh.
- **Index ticker** (`^GSPC`): yfinance reports `Volume=0` for indexes; `engineer_features` propagates this to `Volume_M=0`, which is fine — it just means the model sees a constant feature for that ticker.
- **Pre-market / mid-session**: yfinance excludes the in-progress bar, so we never persist a partial day. No extra logic needed.
- **Stock split / dividend**: `yfinance.history(auto_adjust=True)` retroactively adjusts historical Close prices. This is desired (price series stays continuous) but means a fresh full backfill on a split day will produce slightly different historical prices than an incremental fetch. Acceptable for a daily-cadence model — we're never comparing pre- vs post-split absolute prices in the loss.

## Cadence
Final cron timing is decided in Phase 7 / installed in Phase 8. Default: daily 22:00 local time (~03:00 UTC), comfortably after the 16:00 ET US close + the typical yfinance ingestion lag.

## What this workflow is *not* responsible for
- Feature engineering (that's `src.data.engineer_features`, called by `train.py` and `inference.py`).
- Model retraining (separate workflow: `weekly_retrain.md`).
- Logging predictions (separate tool: `tools/log_prediction.py`).

The refresh's only job is: parquet on disk reflects yfinance's latest closed bar. Everything else reads from that parquet.
