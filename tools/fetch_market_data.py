"""CLI: refresh per-ticker parquet caches from yfinance.

For every entry in ``src.config.TICKERS``, calls ``src.data.update_cache``
and writes to ``data/{ticker_key}.parquet`` (config key, not the yfinance
symbol — so ``data/SPX.parquet``, not ``data/^GSPC.parquet``).

Idempotent: re-running on the same day prints "+0 new rows" and exits in
under a couple of seconds per ticker.

Run::

    C:\\venvs\\stock-predictor\\.venv\\Scripts\\python.exe -m tools.fetch_market_data
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TICKERS  # noqa: E402  (path setup must come first)
from src.data import update_cache  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for key, cfg in TICKERS.items():
        cache_path = DATA_DIR / f"{key}.parquet"
        try:
            new_rows = update_cache(cfg.ticker, cache_path, years=cfg.history_years)
        except Exception as exc:  # network errors, ratelimits, etc.
            logging.error("%s: fetch failed (%s) — leaving cache untouched", key, exc)
            failures += 1
            continue
        last_date = pd.read_parquet(cache_path).index.max().date()
        logging.info("%s: cache up to %s, +%d new rows", key, last_date, new_rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
