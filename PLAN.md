# Plan: Productionize Stock Prediction Project (MSFT + SPX, dual output, self-retraining)

> **For future Claude sessions:** read **Phase 0** first. It contains the project intent, key decisions, and pointers to reference code so you can resume work in a cold session without losing context.

---

## Phase 0 — Session Bootstrap (read first, every session)

### What this project is
A research-grade Jupyter notebook (`Deep Learning for Stock Price Prediction with LSTM and Transformers.ipynb`) that compares LSTM (2-yr MSFT) vs encoder-only Transformer (35-yr MSFT) for next-day price regression. Training data ends **2024-12-30** (>1 year stale as of 2026-05-08). Known issues: denormalization bug at notebook lines 1634–1636, no inference path, no API, no automation, single ticker (MSFT).

### What we're transforming it into
A live web app that:
- Predicts **both direction probability AND expected return %** for the **next trading session**
- Covers **SPX (`^GSPC`) and MSFT** as launch tickers — one model per ticker, same architecture
- **Retrains itself weekly** on fresh data
- Runs **first on localhost**, then on the user's **Contabo VM** where retraining and data refresh happen via systemd timers
- Serves as a **portfolio/CV piece** demonstrating end-to-end ML productionization (not a revenue SaaS — no auth, no billing, no rate-limiting)

### Key design decisions already made (do not relitigate)
1. **Drop LSTM** for the live product. Keep one model family (Transformer) for simpler retrain/monitor/explain. LSTM stays in the original notebook as the scientific artifact.
2. **Multi-ticker via config blocks** from day one. Identical architecture per ticker.
3. **Add an XGBoost baseline** in Phase 3 — on ~8,800 daily samples it often matches/beats a Transformer. Whichever wins on holdout per ticker goes into production; the loser stays as a documented baseline.
4. **Most preprocessing is preserved** from the scientific project (faithful productionization is the CV story). Two specific changes only:
   - **Fix data leakage**: fit z-score normalizer on **train only**, then `.transform()` on val/test/inference. The notebook fits on the full dataset (lines ~370–410) — this is leakage and needs fixing.
   - **STL outlier removal becomes a config flag**, not always-on. 3-sigma removal of "outliers" in financial returns may be removing exactly the events (crashes, FOMC, gaps) the model needs to learn. A/B it on holdout in Phase 4.
5. **Predict return %, not absolute price.** Returns are scale-clean — this naturally sidesteps the denormalization bug. Expected price is a *display-only* derived value at inference time: `yesterday_close * (1 + predicted_return)`.
6. **Weekly retrain is sufficient.** Inputs are refreshed daily (data is never >1 trading day stale); only model *weights* lag by up to a week. On 8,800 training bars, adding 5 more before refit changes weights negligibly. Daily retraining is cheap and feasible (~10 min CPU) but adds run-to-run prediction variance from random init — not worth it unless monitoring shows Friday accuracy is measurably worse than Monday accuracy.
7. **No Docker for v1.** Contabo VM with Python venv + systemd + nginx is sufficient and easier to debug. Containerization is a documented "future improvement."

### User's environment context
- OS: Windows 10 (development), Linux on Contabo (production target)
- The user is on a credit budget — work compactly, don't burn tokens on rediscovery
- The user follows the **WAT framework** (`workflows/` + `tools/` + `src/`, see `CLAUDE.md` in the project root)

### Reference projects to mirror (paths exist on user's machine)

**Current project (the one being transformed):**
- `c:\Users\Nino\iCloudDrive\Projects\ML and DL\DL for Stock Price Prediction with LSTM and Transformers\`
- Key file: `Deep Learning for Stock Price Prediction with LSTM and Transformers.py` (the converted notebook — read this for current preprocessing/model logic)
- Notebook line refs used throughout this plan are from this `.py` file

**Production-pattern reference (mirror its layout):**
- `C:\Users\Nino\iCloudDrive\Projects\ML and DL\ML for Retail Analytics\`
- Mirror: `src/config.py` (config-as-code), `api/main.py` (FastAPI lifespan loader), `api/static/index.html` (vanilla JS frontend, dynamic forms from `/metadata`), `artifacts/{name}_metadata.json` sibling pattern

### Where to save state across sessions
- Plan progress and per-phase notes: this file (`C:\Users\Nino\.claude\plans\i-want-to-transform-tender-rabbit.md`) — at the end of each phase, append a short "Phase N completed YYYY-MM-DD: <key facts/gotchas discovered>" log under that phase
- Code lives in the project directory; data + artifacts live under `data/` and `artifacts/` inside the project

---

## Phase 1 — Scaffolding (no logic yet, just the skeleton)

**Goal:** Create the empty WAT-style structure so subsequent phases can fill it in. No model code, no data fetching — just directories, `config.py`, and `requirements.txt`.

**Inputs:** Phase 0 context only.

**Deliverables:**
```
<project_root>/
├── src/
│   ├── __init__.py
│   ├── config.py          # SEE BELOW — needs full content
│   ├── data.py            # empty stub: defines function signatures only
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py # empty stub
│   │   └── xgb.py         # empty stub
│   ├── train.py           # empty stub
│   └── inference.py       # empty stub
├── tools/
│   ├── fetch_market_data.py    # empty: just `if __name__ == "__main__": print("not implemented")`
│   ├── train_model.py
│   ├── predict.py
│   ├── retrain_all.py
│   └── log_prediction.py
├── workflows/
│   ├── refresh_data.md
│   ├── weekly_retrain.md
│   └── serve_predictions.md
├── api/
│   ├── __init__.py
│   ├── main.py            # empty stub
│   ├── schemas.py         # empty stub
│   └── static/            # empty dir
├── artifacts/             # empty dir, gitkeep
├── data/                  # empty dir, gitkeep
├── research/              # MOVE the original notebook + .py here for archival
├── requirements.txt
└── .gitignore             # ignore data/, artifacts/, .venv/, __pycache__/
```

**`src/config.py` content (this is the contract every other phase reads):**
```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TickerConfig:
    ticker: str                           # yfinance symbol, e.g. "MSFT" or "^GSPC"
    display_name: str                     # for UI, e.g. "S&P 500"
    history_years: int = 35
    seq_len: int = 60
    feature_columns: list[str] = field(default_factory=lambda: [
        "Close", "Open", "High", "Low", "Volume_M",
        "SMA_20", "EMA_20", "RSI",
        "lag_close_1", "lag_close_2", "lag_return_1", "lag_return_2",
        "Spread_OC", "Spread_HL", "day_of_week",
    ])
    holdout_days: int = 252               # ~1 trading year held out for honest eval
    apply_stl_outlier_removal: bool = False  # default OFF; flip to A/B in Phase 4
    transformer_hparams: dict = field(default_factory=lambda: dict(
        num_blocks=3, num_heads=4, key_dim=32, ffn_units=64,
        dense_units=32, dropout=0.1, lr=1e-3,
        epochs=30, batch_size=32,
    ))
    xgb_hparams: dict = field(default_factory=lambda: dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    ))

TICKERS: dict[str, TickerConfig] = {
    "MSFT":   TickerConfig(ticker="MSFT",   display_name="Microsoft"),
    "SPX":    TickerConfig(ticker="^GSPC",  display_name="S&P 500"),
}

RANDOM_STATE = 42
```

**`requirements.txt`:**
```
yfinance>=0.2
pandas
numpy
scikit-learn
xgboost
tensorflow>=2.15
fastapi
uvicorn[standard]
pydantic>=2
joblib
pyarrow            # parquet
statsmodels        # STL
```

**Verification:**
- `python -c "from src.config import TICKERS; print(list(TICKERS))"` outputs `['MSFT', 'SPX']`
- `pip install -r requirements.txt` succeeds in a fresh venv
- `tree -L 2` (or PowerShell `Get-ChildItem -Recurse -Depth 2`) matches the deliverable layout

**Out of scope for this phase:**
- Any model logic
- Any data fetching
- Any tests

---

## Phase 2 — Data layer (fetch, cache, features, holdout)

**Goal:** A complete, deterministic data pipeline that any later phase can call. Faithfully preserves the scientific project's feature engineering — only fixes the leakage bug.

**Inputs:** Phase 1 scaffolding. Notebook source at `research/Deep Learning for Stock Price Prediction with LSTM and Transformers.py` (lines 327–437 contain the preprocessing/feature code to port).

**Deliverables:**

**`src/data.py`** — implement these functions:
```python
def fetch_history(ticker: str, years: int) -> pd.DataFrame
    # uses yfinance Ticker(ticker).history(period=f"{years}y")
    # returns OHLCV with DatetimeIndex

def update_cache(ticker: str, cache_path: Path) -> int
    # idempotent: reads existing parquet, fetches only the gap from last cached date
    # to today via yfinance, appends, returns number of new rows added

def engineer_features(df: pd.DataFrame) -> pd.DataFrame
    # PORTS notebook lines 327-363 verbatim (with renames for clarity):
    #   Volume_M = Volume / 1e6
    #   SMA_20, EMA_20, RSI(14), lag_close_1/2, lag_return_1/2,
    #   Spread_OC = (Open-Close)/Close, Spread_HL = (High-Low)/Close,
    #   day_of_week
    # ALSO computes target columns:
    #   target_return = Close.pct_change().shift(-1)   # next-day return
    #   target_direction = (target_return > 0).astype(int)
    # Drops the last row (no target available) and rows with NaN from rolling features

def stl_outlier_mask(df: pd.DataFrame, sigma: float = 3.0) -> pd.Series
    # PORTS notebook lines ~380-437: STL decomposition on Close, mask residuals > 3*std
    # returns boolean Series; caller decides whether to apply (config.apply_stl_outlier_removal)

def split_train_val_test(df: pd.DataFrame, holdout_days: int) -> tuple[df, df, df]
    # last `holdout_days` rows = test (holdout), never seen until evaluation
    # of remainder: 90% train, 10% val (chronological, no shuffling)
    # returns (train, val, test)

def fit_normalizer(train_df: pd.DataFrame, feature_cols: list[str]) -> dict
    # FIX FROM SCIENTIFIC PROJECT: fit on TRAIN ONLY, not full dataset
    # returns {col: (mean, std)} dict; serializable to JSON

def apply_normalizer(df: pd.DataFrame, normalizer: dict, feature_cols: list[str]) -> pd.DataFrame

def make_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # returns (X, y_return, y_direction) where X has shape (n, seq_len, n_features)
```

**`tools/fetch_market_data.py`** — CLI entry point:
```python
# Iterates over TICKERS in config, calls update_cache for each
# Logs: "MSFT: cache up to 2026-05-08, +1 new row"
# Writes to data/{ticker}.parquet (use the config key, not the yfinance symbol —
# so it's data/SPX.parquet not data/^GSPC.parquet)
```

**`workflows/refresh_data.md`** — short SOP per CLAUDE.md WAT spec, describes the daily refresh purpose and the cron timing decided in Phase 7.

**Holdout discipline:**
- The last 252 trading days are the holdout. Every training run carves them off **before** any normalizer fit, before any sequence construction. Mirrors `ML for Retail Analytics/samples/holdout_*.parquet` pattern.

**Why we keep the scientific preprocessing:**
- The 11-feature set (price, OHLC, volume, technicals, lags, spreads, day-of-week) is sensible, validated, and faithful to the research origin — important for the CV narrative
- Sequence length 60 is reasonable for daily data (~3 trading months of context) and was tuned in the original work
- STL detrending is preserved; only the *outlier removal* on top of it becomes opt-in, since removing fat-tail return events likely hurts generalization

**Verification:**
- `python -m tools.fetch_market_data` creates `data/MSFT.parquet` and `data/SPX.parquet`
- Run twice — second invocation logs "no new rows" and exits in <2s per ticker
- `python -c "from src.data import engineer_features; import pandas as pd; df = pd.read_parquet('data/MSFT.parquet'); print(engineer_features(df).columns.tolist())"` shows all 15 feature + 2 target columns
- Train/val/test split is chronologically ordered (no leakage) and holdout = last 252 rows

---

## Phase 3 — Models (Transformer dual-head + XGBoost baseline)

**Goal:** Two trainable model classes, both predicting `(direction_logit, return_value)` from the same feature set.

**Inputs:** Phase 2 data layer.

**Deliverables:**

**`src/models/transformer.py`** — encoder-only Transformer, **dual output head**:
```python
def build_transformer(seq_len: int, n_features: int, hparams: dict) -> tf.keras.Model
    # PORTS notebook lines 1141-1182: positional encoding (Embedding) + 3 transformer blocks
    # (MultiHeadAttention → Dropout → LayerNorm → FFN), GlobalAveragePooling1D, Dense(32)
    # REPLACES the original (y_output, x_output) dual loss with:
    #   direction_head: Dense(1, activation='sigmoid', name='direction')
    #   return_head:    Dense(1, activation='linear',  name='return')
    # Compile with:
    #   loss = {'direction': 'binary_crossentropy', 'return': 'mse'}
    #   loss_weights = {'direction': 1.0, 'return': 1.0}  # tune later if needed
    #   metrics = {'direction': ['accuracy', tf.keras.metrics.AUC()],
    #              'return': ['mae']}
```

**`src/models/xgb.py`** — XGBoost baseline. No sequencing — operates on the **flattened latest row** (current features + lag features already encode short history):
```python
def build_xgb_classifier(hparams: dict) -> xgboost.XGBClassifier  # for direction
def build_xgb_regressor(hparams: dict) -> xgboost.XGBRegressor    # for return
# Note: two separate XGB models per ticker (one for each target).
# Reason: XGB doesn't support dual-target natively, and the marginal complexity
# of two models is trivial here. Both train in seconds.
```

**Why both:** On ~8,800 daily samples with engineered tabular features, XGB is a strong baseline that often beats deep models. Phase 4 picks the winner per ticker on holdout. Shipping the simpler model when it's actually better is a stronger CV signal than defaulting to "fancy = transformer."

**Verification:**
- `python -c "from src.models.transformer import build_transformer; m = build_transformer(60, 15, {'num_blocks':3,'num_heads':4,'key_dim':32,'ffn_units':64,'dense_units':32,'dropout':0.1,'lr':1e-3}); m.summary()"` prints a model with two output layers named `direction` and `return`
- `python -c "from src.models.xgb import build_xgb_classifier, build_xgb_regressor; print(type(build_xgb_classifier({'n_estimators':10})).__name__)"` shows `XGBClassifier`

**Out of scope:** Training loops (Phase 4). This phase just builds model objects.

---

## Phase 4 — Training pipeline + bake-off

**Goal:** Per ticker (MSFT, SPX), train Transformer + XGB, evaluate on the 252-day holdout, declare a winner, write artifacts and metadata.

**Inputs:** Phases 2 & 3.

**Deliverables:**

**`src/train.py`** — orchestration:
```python
def train_ticker(ticker_key: str, model_kind: Literal['transformer', 'xgb']) -> dict
    # 1. Load data/{ticker_key}.parquet, engineer_features, split, fit_normalizer
    # 2. Optionally apply STL outlier removal (per config flag)
    # 3. Build model, fit on train, validate on val
    # 4. Evaluate on holdout: returns dict of metrics:
    #      direction_accuracy, direction_logloss, direction_auc,
    #      return_rmse, return_mae,
    #      naive_always_up_accuracy,    # baseline: predict up always
    #      naive_yesterday_return_rmse, # baseline: tomorrow = today's return
    # 5. Save:
    #      artifacts/{ticker_key}_{model_kind}.{keras|joblib}
    #      artifacts/{ticker_key}_{model_kind}_metadata.json with:
    #        feature_order, normalizer (mean/std per col),
    #        seq_len, holdout_metrics, train_window (start/end dates),
    #        model_kind, hparams, last_train_date (UTC ISO),
    #        sklearn/xgb/tf versions
    # 6. Return metrics dict
```

**`tools/train_model.py`** — CLI: `python -m tools.train_model --ticker MSFT --model transformer`

**Bake-off rule (per ticker):**
- Train both transformer and xgb
- Whichever has **higher direction_accuracy** on holdout wins (tie-broken by lower return_mae)
- Symlink (or copy on Windows) the winner: `artifacts/{ticker_key}_production.{keras|joblib}` + `..._production_metadata.json`
- Write `artifacts/bakeoff_{ticker_key}.json` recording both models' metrics for transparency

**STL outlier A/B (in this phase):**
- Train transformer once with `apply_stl_outlier_removal=False` and once with `True`
- Compare holdout accuracy; the winning setting becomes the default in `config.py` going forward
- Document the result in this plan's append log

**Honesty gate:**
If the production model for a ticker doesn't beat **both** naive baselines by ≥3 percentage points on direction accuracy, do not proceed to API/UI for that ticker. Surface this explicitly. The CV story is "honest evaluation found X — here's what I'd try next" not "shipped despite no signal."

**Verification:**
- `python -m tools.train_model --ticker MSFT --model transformer` writes 2 files in `artifacts/`, takes <30 min on CPU
- `cat artifacts/MSFT_transformer_metadata.json` shows `holdout_metrics.direction_accuracy` between 0 and 1 and `naive_always_up_accuracy` for comparison
- After running both models for both tickers, `artifacts/bakeoff_MSFT.json` and `artifacts/bakeoff_SPX.json` exist with both models' metrics

---

## Phase 5 — FastAPI inference service

**Goal:** A running localhost server that serves predictions from the production artifacts.

**Inputs:** Phase 4 artifacts. Reference: `ML for Retail Analytics/api/main.py`.

**Deliverables:**

**`src/inference.py`**:
```python
def predict(ticker_key: str) -> dict
    # 1. Load latest data/{ticker_key}.parquet
    # 2. Engineer features on the WHOLE history (cheap, deterministic)
    # 3. Take the last seq_len rows, normalize using saved normalizer from metadata.json
    # 4. Run model.predict(...)
    # 5. Build response:
    #    {
    #       "ticker": "MSFT",
    #       "predicted_for_date": "2026-05-09",   # next trading day
    #       "direction": "up" | "down",
    #       "direction_prob": 0.58,
    #       "expected_return_pct": 0.32,
    #       "implied_close": 412.34,              # last_close * (1 + return)
    #       "model_kind": "transformer",
    #       "last_train_date": "2026-05-03",
    #       "holdout_accuracy": 0.547,
    #    }
```

**`api/main.py`** — FastAPI:
- Lifespan context manager preloads all production models + metadata (mirrors retail project's `lifespan` in `api/main.py`)
- `GET /predict/{ticker}` → calls `inference.predict`
- `GET /metadata` → returns per-ticker config + holdout metrics (frontend reads this)
- `GET /history/{ticker}` → reads `data/predictions_{ticker_key}.parquet` from Phase 7 (returns empty list until Phase 7 lands)
- Mounts `api/static/` at `/`
- Hot-reload check: on each `/predict/*` call, compare mtime of artifact file vs loaded model; reload if newer (so a fresh retrain shows up without restart)

**`api/schemas.py`** — Pydantic response models matching the dict above.

**Verification:**
- `uvicorn api.main:app --reload` starts on `localhost:8000`
- `curl localhost:8000/predict/MSFT` returns JSON with all fields populated
- `curl localhost:8000/metadata` returns both tickers' metrics

---

## Phase 6 — Static HTML frontend

**Goal:** A single-page UI showing both tickers' predictions, served by FastAPI.

**Inputs:** Phase 5 endpoints. Reference: `ML for Retail Analytics/api/static/index.html`.

**Deliverables:**

**`api/static/index.html`** — vanilla JS, no framework:
- Header: project title + GitHub link + disclaimer ("Informational only, not financial advice")
- Ticker tabs (MSFT / SPX) — clicking switches the active card
- Active card displays:
  - Big number 1: **direction probability** ("58% probability up tomorrow") with up/down arrow
  - Big number 2: **expected return** ("+0.32%")
  - Big number 3: **implied close** ("$412.34") with delta from yesterday
  - "Prediction for **2026-05-09** • Model last trained **2026-05-03** (5 days ago)"
- Holdout performance block (read from `/metadata`):
  - direction_accuracy, baseline accuracy, return_mae
  - small text: "Model has been right X% of the time on the last 252 trading days"
- Recent predictions table (last 30 days from `/history/{ticker}`): predicted_at, predicted_for, direction_prob, realized_return, was_correct ✓/✗
- Tiny line chart (Chart.js via CDN, or hand-rolled SVG): cumulative correct vs incorrect over the last 30 predictions

**Verification:**
- Open `http://localhost:8000/` in a browser
- Both ticker tabs work, switching between them refreshes the card without a full page reload
- Holdout metrics block populates from `/metadata`
- History table is empty (until Phase 7 starts logging)

---

## Phase 7 — Daily prediction logging + weekly retraining (still localhost)

**Goal:** Two automated jobs running on localhost via Windows Task Scheduler (or just manual cron-style invocation during dev). This proves the loop end-to-end before moving to the VM.

**Inputs:** Phases 2, 4, 5.

**Deliverables:**

**`tools/log_prediction.py`**:
- For each ticker: call `inference.predict`, append a row to `data/predictions_{ticker_key}.parquet`
- Schema: `predicted_at`, `predicted_for_date`, `direction`, `direction_prob`, `expected_return_pct`, `implied_close`, `model_kind`, `last_train_date`
- On the next day's run, also backfill `realized_close`, `realized_return_pct`, `was_correct` for yesterday's prediction

**`tools/retrain_all.py`**:
- For each ticker, call `train.train_ticker` for the production model_kind
- Atomic artifact swap: write to `artifacts/{ticker}_production.tmp`, fsync, rename
- The API picks up the new artifact via the mtime check from Phase 5 — no restart needed

**`workflows/weekly_retrain.md`** — SOP describing the cadence, what to inspect on failure, how to roll back to the previous artifact (kept as `artifacts/{ticker}_production.prev.*` automatically).

**Cadence (defaults; document the rationale, not just the times):**
- `fetch_market_data.py`: daily 22:00 local time (after US close + yfinance lag)
- `log_prediction.py`: daily 22:15 local time (after fetch)
- `retrain_all.py`: weekly Sunday 03:00 local time
- Document in the workflow markdown: "Weekly retrain is sufficient because data freshness comes from the daily fetch, not the model weights. Monitor `holdout_accuracy_by_dow` after 4 weeks of logs — escalate to daily retrain only if Friday accuracy is materially worse than Monday."

**Verification (after a week of running):**
- `data/predictions_MSFT.parquet` has 5 rows, one per trading day
- A `realized_*` column is populated for the first 4 (yesterday's prediction has its outcome)
- `data/predictions_MSFT.parquet` shows model_kind staying constant unless a Sunday retrain swapped the production winner
- The frontend `/history/MSFT` table shows real entries
- Manually trigger `retrain_all.py` — `last_train_date` in `/metadata` advances on next API hit

---

## Phase 8 — Deployment to Contabo VM

**Goal:** Move everything from localhost to the user's Contabo Linux VM. Same code, same flow, now publicly accessible with HTTPS and self-running.

**Inputs:** All previous phases working on localhost.

**Deliverables:**

1. **VM provisioning** (one-time, doc in `workflows/deploy.md`):
   - SSH access, non-root user with sudo
   - `apt install python3.11 python3.11-venv python3-pip nginx certbot python3-certbot-nginx git`
   - Clone repo to `/srv/stock-predictor/`
   - `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
   - Set env via `/etc/default/stock-predictor` (TZ, paths)

2. **systemd units** (in `deploy/systemd/`, install to `/etc/systemd/system/`):
   - `stock-predictor.service` — runs `uvicorn api.main:app --host 127.0.0.1 --port 8000` under non-root user, `Restart=on-failure`
   - `stock-predictor-fetch.service` + `.timer` — runs `tools/fetch_market_data.py` daily at 22:00 UTC
   - `stock-predictor-log.service` + `.timer` — runs `tools/log_prediction.py` daily at 22:15 UTC
   - `stock-predictor-retrain.service` + `.timer` — runs `tools/retrain_all.py` weekly Sunday 03:00 UTC
   - All log to journald (no extra log infra)

3. **nginx** (`deploy/nginx/stock-predictor.conf` → `/etc/nginx/sites-available/`):
   - Reverse proxy `https://<your-domain>/` → `http://127.0.0.1:8000`
   - certbot for TLS (Let's Encrypt)
   - Basic gzip + cache headers for `/static`

4. **First-boot training:** SSH in, manually run `python -m tools.fetch_market_data && python -m tools.train_model --ticker MSFT && python -m tools.train_model --ticker SPX` to populate `data/` and `artifacts/`. Then enable the timers.

5. **Smoke test:** `curl https://<domain>/predict/MSFT` returns valid JSON. Browser shows the UI over HTTPS.

**Verification:**
- `systemctl status stock-predictor` is active (running)
- `systemctl list-timers | grep stock` shows three active timers with NEXT firing times
- `journalctl -u stock-predictor-fetch -n 50` shows the last fetch log
- After one week unattended: data, predictions, and artifacts have all updated without intervention
- The site is reachable at the public URL with valid TLS

**Out of scope:** Auth, billing, rate-limiting, monitoring stack (Prometheus/Grafana), Docker, CI/CD. journald logs are sufficient for v1.

---

## Phase 9 — Polish for CV / portfolio

**Goal:** Make the repo presentable to a hiring manager who has 90 seconds.

**Deliverables:**

**`README.md`** at project root:
1. **One-line pitch** + screenshot of the live UI
2. **Live URL** (the deployed site)
3. **Architecture diagram** (Mermaid is fine — no image hosting): `yfinance → daily cron → parquet cache → weekly retrain cron → keras/xgb artifact → FastAPI → static UI`
4. **Honest holdout metrics table** for both tickers, with naive baselines included. If the model didn't beat baseline, say so explicitly and explain the next iteration.
5. **What I'd do differently with more time** section — drift detection, sentiment features, ensemble, true Bayesian uncertainty intervals, Docker, CI
6. Link to `research/` notebook (the LSTM-vs-Transformer scientific origin)
7. Setup instructions (clone, venv, requirements.txt, run)

**`docs/journey.md`** — a short blog-post-style write-up (~600–1000 words) covering:
- Starting point: research notebook with stale data and a denormalization bug
- Decisions: why drop LSTM, why dual output, why XGB baseline, why weekly retrain
- The data-leakage fix and why it mattered
- The honesty gate (how the bake-off was decided)
- Deployment minimalism (systemd not k8s)

This file is the part recruiters actually read. Optimize for clarity over polish.

**Verification:**
- A reader who has never seen the project can, in 5 minutes, understand what it does, how it works, and how good it actually is
- The deployed URL renders correctly on mobile and desktop
- All claimed metrics in the README match `artifacts/*_metadata.json`

---

## Per-phase append log (Claude updates this at end of each phase)

> Append a brief entry per phase as it's completed: `### Phase N — completed YYYY-MM-DD\n- key facts/decisions made during execution\n- gotchas discovered\n- anything future phases should know`

### Phase 1 — completed 2026-05-10
- Scaffolding created exactly per spec. Layout verified via Glob; matches deliverable tree.
- Original notebook + .py archived at `research/Deep Learning for Stock Price Prediction with LSTM and Transformers.{ipynb,py}`. **All notebook line refs in this plan (327–437 preprocessing, 1141–1182 transformer model, ~370–410 normalizer leakage, 1634–1636 denormalization bug) now point inside `research/`, not the project root.**
- `src/config.py` import verified: `TICKERS = ['MSFT', 'SPX']`, MSFT.ticker = `MSFT`, SPX.ticker = `^GSPC`, 15 feature columns, RANDOM_STATE = 42.
- All Python stubs raise `NotImplementedError("Phase N")` so a future phase that wires them prematurely fails loudly with the right pointer.
- `requirements.txt` written but **NOT installed** — the spec made install a verification step, not a deliverable. Phase 2 should `pip install -r requirements.txt` in a venv before running anything.
- `.gitignore` ignores `data/` and `artifacts/` but keeps `.gitkeep` markers so the layout survives a fresh clone.
- Project is **not yet a git repo** (CLAUDE.md says `Is a git repository: false`). Whoever does `git init` first should commit the scaffolding before Phase 2 starts adding generated data.
- Workflow markdowns are skeletons with the structure CLAUDE.md WAT spec calls for (Objective, Inputs, Tools, Outputs, Edge cases, Cadence) — to be filled in during their respective phases (2, 5, 7).

---

## Cross-phase scope reminders (do not violate)

- **Don't add features to v1**: no auth, no Stripe, no rate-limiting, no Docker, no Kubernetes, no Sentry, no Prometheus, no multi-step forecasting, no sentiment/RSS, no tickers beyond MSFT and SPX, no mobile app
- **Don't relitigate Phase 0 decisions** — if you think one is wrong, raise it explicitly with the user before changing course
- **Don't write new code where existing patterns from `ML for Retail Analytics` apply** — mirror the reference paths called out in each phase
- **Don't claim performance you haven't measured** — the holdout metrics in metadata.json and README must come from actual evaluation runs, not estimates
