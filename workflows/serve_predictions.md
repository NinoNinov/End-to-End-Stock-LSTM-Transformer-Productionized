# Workflow: Serve predictions

## Objective
Serve next-trading-day predictions (direction probability + expected return %)
for each configured ticker via a FastAPI service. Phase 5 ships the API only;
Phase 6 will add the static HTML frontend.

## Inputs
- Production artifacts at `artifacts/{ticker_key}_production.{keras|joblib}` plus
  the sibling `_production_metadata.json` (written by `run_bakeoff` in Phase 4).
- Latest `data/{ticker_key}.parquet` (refreshed by `tools.fetch_market_data`).
- Optional: prediction history at `data/predictions_{ticker_key}.parquet`
  (populated by `tools.log_prediction` in Phase 7).

## Tools used
- `api/main.py` — FastAPI app. Lifespan loader eagerly imports every ticker's
  production model so the first /predict request doesn't pay the TF cold-start
  cost (~15s on this box).
- `src/inference.py::predict(ticker_key, handle)` — single-ticker prediction.
- `src/inference.py::load_handle` / `maybe_reload` — model loading + mtime-based
  hot reload.

## How to run
```powershell
& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m uvicorn api.main:app `
  --host 127.0.0.1 --port 8771 --log-level info
```
- The retail-analytics project on this machine already uses 8765; **pick a
  different port (e.g. 8771) for local dev** to avoid the "ready" probe falsely
  succeeding against the wrong server.
- `--reload` is fine for development. Phase 8 systemd will run without it.

## Endpoints
| Path | Method | Returns |
|---|---|---|
| `/` | GET | `index.html` if present (Phase 6+), otherwise a JSON status block. |
| `/predict/{ticker_key}` | GET | Next-day prediction for `MSFT` or `SPX`. |
| `/metadata` | GET | Per-ticker holdout metrics, normalizer, train window, honesty-gate verdict. |
| `/history/{ticker_key}` | GET | Recent logged predictions (empty list until Phase 7). `?limit=N` to override default 30. |
| `/static/...` | GET | Static assets (Phase 6). |

## Edge cases / things learned
- **Port collision on dev box.** The retail-analytics FastAPI app on this
  machine binds 8765 by default and *also* responds to `/metadata`; an "is the
  server up?" probe that only checks for HTTP 200 will report ready even when
  ours failed to bind. Always confirm with `curl /predict/MSFT` (404 → wrong
  app on the port).
- **Heavy TF imports happen at lifespan, not at request time.** First boot
  takes ~15s on this CPU box. Don't kill the server during startup.
- **`shutil.copy2` preserves mtime.** When the bake-off promotes a model whose
  *source* artifact hasn't changed, `*_production.keras`'s mtime doesn't move
  and `maybe_reload` won't fire. Phase 7's retrain script uses `os.replace`
  after a fresh write, which forces a new mtime — this is the supported path
  for "hot reload on retrain."
- **Stale data.** No automatic warning yet (the response always reflects the
  latest bar in `data/{ticker}.parquet`); Phase 7's logger will surface
  freshness in the history table.
- **Missing artifacts are fatal at startup.** The lifespan loader raises if any
  ticker's production model is absent. Run `python -m tools.train_model
  --ticker MSFT --all` (and `--ticker SPX --all`) first.

## Verification
```bash
curl -s http://127.0.0.1:8771/predict/MSFT | python -m json.tool
curl -s http://127.0.0.1:8771/predict/SPX  | python -m json.tool
curl -s http://127.0.0.1:8771/metadata     | python -m json.tool
curl -s http://127.0.0.1:8771/history/MSFT     # → []
curl -s -w "%{http_code}\n" http://127.0.0.1:8771/predict/NOPE  # → 404
```
