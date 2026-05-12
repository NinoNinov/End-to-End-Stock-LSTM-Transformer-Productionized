# Workflow: Daily prediction logging + weekly retrain

This SOP covers both Phase 7 automated jobs. They share an artifact ecosystem (`data/`, `artifacts/`) and the same operational discipline (atomic writes, .prev rollback slots), so they're documented together.

## Objective
1. **Daily**: append one prediction row per ticker and backfill the previous day's outcome — produces the `/history/{ticker}` data the frontend renders.
2. **Weekly**: refit production weights on the latest cached data and atomically swap the artifact. The API picks up the new file via mtime check (Phase 5) — no restart.

## Inputs
- Up-to-date `data/{ticker_key}.parquet` from the `refresh_data` workflow (must run first on the daily cadence).
- For retrain: a current `artifacts/{ticker_key}_production_metadata.json` — its `model_kind` and `stl_outlier_removal_applied` fields drive what gets refit. Phase 4's bake-off populates these initially.

## Tools to use
- `tools/log_prediction.py` — daily logger. Calls `src.inference.predict` per ticker, appends one row to `data/predictions_{ticker_key}.parquet`, backfills resolvable prior rows.
- `tools/retrain_all.py` — weekly retrainer. Reads each ticker's production `model_kind`, calls `src.train.train_ticker` with the same config, atomic-swaps the production artifact, rotates the previous version to `.prev.*`.

## How to invoke

```powershell
# Daily (manual during dev; Windows Task Scheduler or systemd timer in prod):
& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m tools.fetch_market_data
& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m tools.log_prediction

# Weekly retrain (all tickers, full swap):
& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m tools.retrain_all

# Single ticker (debug):
& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m tools.retrain_all --ticker MSFT

# Smoke test — train but don't promote:
& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m tools.retrain_all --no-swap
```

## Expected outputs

### `log_prediction`
- New row appended to `data/predictions_{ticker_key}.parquet` with these columns: `predicted_at`, `predicted_for_date`, `last_bar_date_at_prediction`, `last_close_at_prediction`, `direction`, `direction_prob`, `direction_threshold`, `expected_return_pct`, `implied_close`, `model_kind`, `last_train_date`, `realized_date`, `realized_close`, `realized_return_pct`, `was_correct`.
- The five backfill columns (`realized_*`, `was_correct`) are null at append time; the **next** run that has access to the next trading day's bar fills them in.
- One log line per ticker, e.g. `MSFT: appended=True backfilled=1 for=2026-05-12 dir=up p=0.5218 rows=42`.

### `retrain_all`
- Updated `artifacts/{ticker_key}_production.{keras|joblib}` + matching metadata sidecar — both fresh-written, mtime = now.
- Previous artifact rotated to `artifacts/{ticker_key}_production.prev.{ext}` (and `..._production_metadata.prev.json`).
- API picks up the new artifact on the next request via Phase 5's `maybe_reload`.

## Cadence rationale

- **`fetch_market_data`**: daily 22:00 local (~03:00 UTC). Comfortably after the 16:00 ET US close + yfinance ingestion lag.
- **`log_prediction`**: daily 22:15 local. After fetch — predictions log against fresh bars.
- **`retrain_all`**: weekly Sunday 03:00 local. Off-hours, no contention with daily jobs.

**Why weekly retrain is sufficient.** Data freshness comes from the daily fetch, not the model weights. On ~8,800 training bars, adding 5 new bars before refit changes weights negligibly. Daily retrains are cheap (~10 min CPU) but add run-to-run prediction variance from random init — not worth it unless monitoring shows real Friday-vs-Monday accuracy drift.

**When to escalate to daily**: after ≥4 weeks of logs, compute `holdout_accuracy_by_dow`. If Friday accuracy is materially worse than Monday on logged predictions (not on the static holdout — that's frozen), the model is stale by week's end and the cadence should change.

## Idempotency

- `log_prediction` is idempotent per `predicted_for_date`. Re-running within the same trading day appends nothing new. A fresh `fetch_market_data` that moved `last_bar_date` forward makes the next `log_prediction` write a new row, because `predicted_for_date` advances.
- `retrain_all` is **not** idempotent: every run produces fresh weights and a fresh atomic swap. The previous-run artifact is preserved as `.prev.*` so a single re-run never destroys more than one prior version of history.

## Atomic write discipline

Both tools follow the same pattern for any file the API might read concurrently:

1. Write fresh bytes to a sibling `.tmp` path.
2. `fsync` the tmp file (forces OS to flush bytes before the rename swaps it in).
3. `os.replace(tmp, dst)` — atomic on the same volume, on both POSIX and Windows.

For `retrain_all`, the existing production file is `os.replace`'d to `.prev.{ext}` *before* the fresh write begins. The API's `maybe_reload` tolerates the brief window where the production file is missing — it falls back to the in-memory model.

## Rollback procedure

If a swap landed bad weights (degraded holdout, NaN predictions, etc.):

```powershell
# Stop nothing — the API stays running. The mtime check will pick up the rolled-back file on its next request.
Set-Location "$env:PROJECT\artifacts"
Move-Item MSFT_production.keras MSFT_production.bad.keras
Move-Item MSFT_production_metadata.json MSFT_production_metadata.bad.json
Move-Item MSFT_production.prev.keras MSFT_production.keras
Move-Item MSFT_production_metadata.prev.json MSFT_production_metadata.json
```

On Linux (Phase 8) the same flow with `mv`. The next `/predict/MSFT` call sees a fresh mtime, reloads the rolled-back file, and serves the previous version. The `.bad.*` files stay for forensic inspection.

## Edge cases

- **Stale data, fresh prediction**: `log_prediction` will still produce a row, but `predicted_for_date` won't have advanced from yesterday's run → idempotency dedup kicks in and nothing appends. Fix by running `fetch_market_data` first.
- **Training fails mid-retrain**: `train_ticker` raises, the `_rotate_to_prev` step never runs, the existing production artifact is untouched. The CLI exits non-zero so a wrapping unit (systemd / Task Scheduler) can record the failure.
- **Holdout metrics regress badly vs previous run**: the swap still proceeds (we accept run-to-run noise on this dataset — Phase 4 documented the ~5pp AUC swing from re-seeding alone). Manual rollback if the regression looks structural (e.g. NaN predictions, accuracy below naive baseline by >5pp).
- **Disk full during atomic write**: the `.tmp` write fails before `os.replace`. The production file is unchanged; the failed `.tmp` is left behind for inspection.
- **API holds an old in-memory model for the full request after a swap**: by design. `maybe_reload` happens at the *start* of each request, so an in-flight request sees the model it started with. New requests see the new model.

## What this workflow is *not* responsible for

- Fetching market data (separate workflow: `refresh_data.md`).
- Re-running the Phase 4 bake-off (that's a human-supervised decision; weekly retrain stays on the production family chosen there).
- Deciding cadence on the VM (Phase 8 installs the timers; this file documents the *rationale*, the units file documents the *minutes*).
