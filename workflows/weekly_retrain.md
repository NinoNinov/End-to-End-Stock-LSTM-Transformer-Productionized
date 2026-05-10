# Workflow: Weekly retrain

> Filled in during Phase 7.

## Objective
Refit the production model for every ticker on the latest cached data and atomically swap the artifact, without restarting the API.

## Inputs
- Up-to-date `data/{ticker_key}.parquet` from `refresh_data` workflow
- Current production model_kind per ticker (recorded in `artifacts/{ticker_key}_production_metadata.json`)

## Tools to use
- `tools/retrain_all.py` (calls `src.train.train_ticker` for each ticker's production model_kind)

## Expected outputs
- Updated `artifacts/{ticker_key}_production.{keras|joblib}` + matching metadata
- Previous artifact preserved as `artifacts/{ticker_key}_production.prev.*` for rollback
- API picks up the new artifact on the next request via mtime check (Phase 5)

## Edge cases
- Training fails → keep the previous artifact, exit non-zero, surface to journald
- Holdout metrics regress badly vs previous run → still swap (we accept noise) but flag in the log; manual rollback if needed via `mv` of `.prev.*` back to `production.*`

## Cadence rationale
Weekly is sufficient because data freshness comes from the daily fetch, not the model weights. On ~8,800 training bars, +5 new bars before refit changes weights negligibly. Escalate to daily only if `holdout_accuracy_by_dow` shows Friday materially worse than Monday after 4 weeks of logs.
