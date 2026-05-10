# Workflow: Refresh market data

> Filled in during Phase 2.

## Objective
Keep `data/{ticker}.parquet` up to date with the latest daily OHLCV from yfinance for every ticker in `src/config.py::TICKERS`.

## Inputs
- `src/config.py::TICKERS` (which tickers, how many years of history)

## Tools to use
- `tools/fetch_market_data.py` (calls `src.data.update_cache` per ticker)

## Expected outputs
- One parquet per ticker at `data/{ticker_key}.parquet` (uses the config key, not the yfinance symbol — so `data/SPX.parquet`, not `data/^GSPC.parquet`)
- One log line per ticker: `MSFT: cache up to YYYY-MM-DD, +N new rows`

## Edge cases
- Empty cache: full backfill (history_years from config)
- No new rows: idempotent, exits in <2s per ticker
- yfinance flakiness: retry once, then surface the error (don't write a partial file)

## Cadence
Decided in Phase 7. Default: daily 22:00 local time (after US close + yfinance lag).
