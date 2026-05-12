"""FastAPI inference service for the stock prediction project (Phase 5).

Run locally:
    uvicorn api.main:app --reload

Endpoints:
    GET /predict/{ticker_key}   - next-day direction + return for one ticker
    GET /metadata               - per-ticker config + holdout metrics for all tickers
    GET /history/{ticker_key}   - past predictions (empty until Phase 7 starts logging)
    GET /                       - static index.html (added in Phase 6)
    GET /static/*               - other static assets

Each /predict/{ticker_key} call does an mtime check on the production
artifact; a fresh retrain (Phase 7) is picked up without restarting the
server.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import TICKERS
from src.inference import ModelHandle, load_handle, maybe_reload, predict as run_predict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load every ticker's production model + metadata at startup.

    A missing artifact is fatal: refusing to start beats serving 500s for a
    misconfigured ticker. The retrain script (Phase 7) writes fresh files
    atomically, so the in-process handle never sees a half-written model.
    """
    handles: dict[str, ModelHandle] = {}
    for ticker_key in TICKERS:
        try:
            handles[ticker_key] = load_handle(ticker_key)
            logger.info(
                "Loaded %s production model (%s, trained %s)",
                ticker_key,
                handles[ticker_key].model_kind,
                handles[ticker_key].metadata.get("last_train_date"),
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Cannot start API: production artifact missing for {ticker_key}. "
                f"Run `python -m tools.train_model --ticker {ticker_key} --all` first. "
                f"({e})"
            ) from e
    app.state.handles = handles
    yield
    app.state.handles = {}


app = FastAPI(title="Stock Prediction (MSFT + SPX)", lifespan=lifespan)


def _bakeoff_for(ticker_key: str) -> dict | None:
    path = ARTIFACTS_DIR / f"bakeoff_{ticker_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@app.get("/predict/{ticker_key}")
def get_prediction(ticker_key: str) -> dict[str, Any]:
    if ticker_key not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker_key}")

    handle = app.state.handles.get(ticker_key)
    if handle is None:
        raise HTTPException(
            status_code=503, detail=f"No loaded model for {ticker_key}"
        )

    # Hot-reload check: Phase 7's retrain script atomically replaces the
    # artifact, bumping its mtime; pick that up here without a restart.
    refreshed = maybe_reload(handle)
    if refreshed is not handle:
        app.state.handles[ticker_key] = refreshed
        handle = refreshed

    try:
        return run_predict(ticker_key, handle)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed for %s", ticker_key)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    """Per-ticker metadata blocks for the frontend."""
    out: dict[str, Any] = {}
    for ticker_key, cfg in TICKERS.items():
        handle = app.state.handles.get(ticker_key)
        if handle is None:
            continue
        meta = handle.metadata
        bakeoff = _bakeoff_for(ticker_key)
        out[ticker_key] = {
            "ticker_key": ticker_key,
            "ticker_symbol": cfg.ticker,
            "display_name": cfg.display_name,
            "model_kind": meta.get("model_kind"),
            "seq_len": meta.get("seq_len"),
            "feature_order": meta.get("feature_order"),
            "holdout_metrics": meta.get("holdout_metrics"),
            "train_window": meta.get("train_window"),
            "last_train_date": meta.get("last_train_date"),
            "stl_outlier_removal_applied": meta.get("stl_outlier_removal_applied"),
            "honesty_gate": (bakeoff or {}).get("honesty_gate"),
        }
    return out


@app.get("/history/{ticker_key}")
def get_history(ticker_key: str, limit: int = 30) -> list[dict[str, Any]]:
    """Recent prediction history. Empty list until Phase 7 starts logging."""
    if ticker_key not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker_key}")

    path = DATA_DIR / f"predictions_{ticker_key}.parquet"
    if not path.exists():
        return []

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.exception("Failed reading history for %s", ticker_key)
        raise HTTPException(status_code=500, detail=f"History read failed: {e}") from e

    if df.empty:
        return []

    if "predicted_at" in df.columns:
        df = df.sort_values("predicted_at", ascending=False)
    df = df.head(max(1, int(limit)))
    # Stringify dates/timestamps for JSON. NaT becomes NaN inside the resulting
    # string-typed column, so we have to coerce *after* the formatting pass —
    # `df.where(df.notnull(), ...)` on a string-dtype column leaks NaN through
    # to JSON and produces invalid output (Phase 7 finding).
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    # Final scrub: object-dtype columns can still hold float('nan') sentinels
    # that survived the where() above (StringDtype quirk). Replace defensively.
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v):
                row[k] = None
    return records


# Static frontend (Phase 6 will populate). Mount LAST so it doesn't shadow API routes.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            {
                "service": "stock-prediction",
                "phase": 5,
                "note": "frontend is built in Phase 6; use /predict/{MSFT,SPX} and /metadata for now",
                "tickers": list(TICKERS.keys()),
            }
        )
    return FileResponse(index)
