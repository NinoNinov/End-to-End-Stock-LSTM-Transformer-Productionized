# Workflow: Serve predictions

> Filled in during Phase 5.

## Objective
Serve next-trading-day predictions (direction probability + expected return %) for each configured ticker via a FastAPI service with a static HTML frontend.

## Inputs
- Production artifacts at `artifacts/{ticker_key}_production.{keras|joblib}` + metadata
- Latest `data/{ticker_key}.parquet`
- Prediction history at `data/predictions_{ticker_key}.parquet` (for `/history`)

## Tools to use
- `api/main.py` (FastAPI app, lifespan-loaded models)
- `src/inference.py::predict` (single-ticker prediction)

## Endpoints
- `GET /predict/{ticker}` → next-day prediction JSON
- `GET /metadata` → per-ticker config + holdout metrics
- `GET /history/{ticker}` → recent logged predictions with realized outcomes
- `GET /` → static UI

## Edge cases
- Stale data (last bar > 2 trading days old) → still serve, but include a `data_freshness_warning` in the response
- Model file changed on disk since lifespan load → mtime check reloads it on next request
