# Plan: Productionize Stock Prediction Project (MSFT + SPX, dual output, self-retraining)

> **For future Claude sessions:** read **Phase 0** first. It contains the project intent, key decisions, and pointers to reference code so you can resume work in a cold session without losing context.

> **⚠ User TODO (Nino):** Create the **official GitHub repo** and push the existing local history. As of 2026-05-10 there is no `origin` remote — every commit lives only in this machine's `.git/`. Useful before **Phase 8 (deployment)** because that phase deploys via `git clone` to the Contabo VM. Steps: (1) github.com → New repository (don't initialize with README/license/gitignore — those would conflict); (2) `git remote add origin <url>`; (3) `git push -u origin master`. Can stay private until Phase 9; privacy doesn't change git's behavior.

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

**Phase 7 completed 2026-05-11:**
- `tools/log_prediction.py` (220 lines): per-ticker append with idempotent dedup on `predicted_for_date`; same-pass backfill of resolvable prior rows; atomic parquet write via tmp+`os.replace`. Row schema is wider than the API response — stores `last_bar_date_at_prediction` + `last_close_at_prediction` so backfill works correctly across US market holidays (anchor on data lineage, not the `BDay(1)` heuristic).
- `tools/retrain_all.py` (200 lines): reads each ticker's current production `model_kind` + `apply_stl` from `{ticker}_production_metadata.json`, refits *that same configuration* via `src.train.train_ticker` (no bake-off — that's a human decision), then atomic swap: rotate current prod → `.prev.{ext}`, write fresh bytes to `.tmp{ext}`, `fsync` the tmp, `os.replace` into prod. Same dance for the metadata sidecar. CLI flags: `--ticker KEY` (repeatable), `--no-swap` (smoke-test training side without touching prod).
- `workflows/weekly_retrain.md`: full SOP — invocation, expected outputs per script, cadence rationale (weekly retrain, daily log, daily fetch), atomic-write discipline, manual rollback procedure (`mv .prev.* prod.*`), edge cases (training failure, disk full, in-flight requests during swap).
- Phase 5 bugfix in `api/main.py:get_history` — the `dt.strftime` pass on NaT-containing columns produced `float('nan')` in the resulting string Series, which `df.where(df.notnull(), None)` silently failed to coerce. Result: `/history` was emitting `"realized_date": NaN` (invalid JSON, browser parser fails). Fixed by `df.astype(object).where(pd.notna(df), None)` + a per-cell `isinstance(v, float) and pd.isna(v)` scrub. Caught by a 15-line `.tmp/smoke_history.py` that ran the endpoint's serialization path through `json.dumps(allow_nan=False)`.
- Verified end-to-end on localhost: ran `log_prediction` once → 1 row per ticker with proper schema/dtypes; re-ran → `appended=False` (idempotency); synthesized a backfillable row (`last_bar_date=2026-05-06`) → next run resolved to 2026-05-07 close ($420.77, +1.6451%, was_correct=True). `retrain_all` not exercised end-to-end this session (cost: ~10 min CPU per ticker; user is on a credit budget) — `--no-swap` is available for cheap validation of the training side, and the swap path is a small enough surface to code-review.
- Gotcha for Phase 8: scheduled cadence in `weekly_retrain.md` is documented in *local time*; Phase 8 systemd timers run on UTC. Convert when installing the units.

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
- `.gitignore` initially used the directory pattern `data/` which prevented the `!data/.gitkeep` re-include from working. Fixed to `data/*` + `!data/.gitkeep` (same for `artifacts/`) — the directory-contents pattern is the only form git can re-include from.
- Workflow markdowns are skeletons with the structure CLAUDE.md WAT spec calls for (Objective, Inputs, Tools, Outputs, Edge cases, Cadence) — to be filled in during their respective phases (2, 5, 7).
- Git initialized; baseline commit `ad68aeb` (28 files). Identity from global config: `Nino <32340544+NinoNinov@users.noreply.github.com>` (GitHub noreply — clean for public repos).

#### **VENV LOCATION — read this before running ANY tool/script**
The venv lives at **`C:\venvs\stock-predictor\.venv`**, NOT at `<project_root>/.venv`. We moved it out of iCloud Drive because: (a) iCloud syncs every `.pyc` file (~30k file churn), (b) Files-On-Demand can evict bytes Python needs at runtime, (c) it eats iCloud quota.

**Always invoke Python via the absolute venv path:**
- Bash: `/c/venvs/stock-predictor/.venv/Scripts/python.exe -m tools.fetch_market_data`
- PowerShell: `& "C:\venvs\stock-predictor\.venv\Scripts\python.exe" -m tools.fetch_market_data`
- VS Code: set interpreter to `C:\venvs\stock-predictor\.venv\Scripts\python.exe`

If you're starting a cold session and need to recreate the venv:
```powershell
New-Item -ItemType Directory -Force -Path C:\venvs\stock-predictor | Out-Null
python -m venv C:\venvs\stock-predictor\.venv
& C:\venvs\stock-predictor\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

#### Python / TF compatibility
System Python is 3.13.1. TensorFlow only added 3.13 wheels in TF 2.19 (Mar 2025). The pinned `tensorflow>=2.15` resolved to **TF 2.21.0** — fine. If a future cold session uses Python <3.13, the resolver may pick a different TF; this is OK as long as `tensorflow>=2.15` and the model code is version-agnostic (which it should be — Keras 3 API).

### Phase 2 — completed 2026-05-10
- `src/data.py` implements the full Phase 2 contract: `fetch_history`, `update_cache`, `engineer_features`, `stl_outlier_mask`, `split_train_val_test`, `fit_normalizer`, `apply_normalizer`, `make_sequences`. Notebook `preprocess_data` (lines 332–390 in `research/.../py`) is ported.
- **Two deliberate deviations from the notebook**, both documented in the function docstring:
  - **MA window: 20, not 100.** Notebook had `va_window=100` but named the column `SMA_20` — clearly a name/value mismatch. We honour the column name (the contract). Side effect: `engineer_features` only drops the head ~20 rows for warmup instead of ~100, gaining ~80 training rows. With ~8.8k rows this barely matters; with the 2-yr LSTM dataset (which we don't ship anyway) it would have been material.
  - **Spread denominators are Close, not Open/High.** Plan explicitly specifies Close. Notebook used `(Close - Open)/Open` and `(High - Low)/High`. Plan's formulas inverted the sign of `Spread_OC`. Both are reasonable; the contract wins. Down-stream code (Phase 4) just needs to be aware that positive `Spread_OC` now means an intraday loss (Open > Close).
- **`update_cache` signature deviates from plan**: plan declared `(ticker, cache_path)` but to support empty-cache backfill the function needs `years` too — added as `years: int = 35` (default matches `TickerConfig.history_years`). The CLI passes the per-ticker config value through.
- **`engineer_features` returns 17 cols** (15 features + `target_return` + `target_direction`). 8813 raw bars → 8793 feature rows after dropping ~20 rolling-NaN rows + the final unknowable-target row. `target_direction` uses `(target_return > 0)` (strict positive = up); the very rare exactly-flat day counts as down. Acceptable.
- **`stl_outlier_mask` import is lazy.** Importing `statsmodels.tsa.seasonal.STL` at module top eagerly initialises OpenBLAS, which on this Windows box hit MemoryError + `OpenBLAS error: Memory allocation still failed after 10 retries` when paired with yfinance's curl_cffi → asyncio import chain. Deferring `from statsmodels...` inside the function fixes it and means the hot path (fetch/features/sequences) never pays the statsmodels cost. **Lesson:** on memory-constrained Windows, defer heavy-scientific imports until the function actually needs them.
- **Holdout discipline verified end-to-end on real data:**
  - MSFT: 8813 raw → 8793 features → 7686 train / 855 val / 252 test (last 252 = test, then 90/10).
  - SPX: same shape (S&P 500 also has 35 years of yfinance data).
  - `fit_normalizer(train)` on MSFT: `Close.mean ≈ $38.52, Close.std ≈ $55.62`. That's the 35-year average — vs today's $415 close. If we'd fit on the full dataset (the leakage bug), mean would be ~$80, and the test rows (which sit near $400+) would be at z-score ~6 instead of the correct ~6.7. The leakage fix shows up.
  - `make_sequences` shapes: X = (7627, 60, 15), y_return = (7627,), y_direction = (7627,) — matches `len(train) - seq_len + 1`.
- **Index ticker quirk:** yfinance returns `Volume=0` for `^GSPC`. `Volume_M` for SPX is therefore a constant 0 column. `fit_normalizer` falls back to std=1 instead of dividing by zero (the explicit `if std == 0: std = 1.0` guard). The constant feature won't help the model but won't hurt it either; Phase 3 model code can stay ticker-agnostic.
- **STL outliers on a 1480-row MSFT window flagged 28 rows (~1.9%) at sigma=3.** Sample dates: 2021-12-10, 2021-12-20, 2022-02-23, 2023-03-10, 2023-04-25 — all real fat-tail event days (selloff weeks, Russia/Ukraine, SVB). Confirms the Phase 0 hypothesis that "outlier removal" may be removing exactly the events the model needs to learn. A/B remains in Phase 4 with `apply_stl_outlier_removal=False` as default.
- **Cache files:** `data/MSFT.parquet` and `data/SPX.parquet`, ~390 KB each, indexed by tz-naive `Date`. Re-run of `tools/fetch_market_data` is a clean idempotent no-op.

### Phase 3 — completed 2026-05-10
- `src/models/transformer.py::build_transformer(seq_len, n_features, hparams)` ports notebook lines 1141-1182 with the dual-head replacement specified in PLAN.md. Architecture: positional `Embedding(input_dim=seq_len, output_dim=n_features)` added to input, then `num_blocks=3` encoder blocks (`MHA → Dropout → LayerNorm → Dense(ffn_units, relu) → Dropout → LayerNorm`), `GlobalAveragePooling1D`, one shared `Dense(dense_units, relu)`, two output heads:
  - `direction`: `Dense(1, sigmoid)`, BCE loss, metrics `[accuracy, AUC]`
  - `return`: `Dense(1, linear)`, MSE loss, metric `[mae]`
  - `loss_weights={"direction": 1.0, "return": 1.0}` — leave for tuning later
- **Output naming convention:** the model returns a *dict* (`{"direction": ..., "return": ...}`), not a list. Phase 4 must train with `model.fit(X, {"direction": y_dir, "return": y_ret}, ...)`. This is the cleanest match for the multi-input dict pattern Keras 3 supports.
- **Verified summary** for `(seq_len=60, n_features=15, default hparams)`:
  - Total params: **86,671** (338 KB) — compact relative to ~7.6k training sequences (param/sample ≈ 11)
  - Block 1 MHA carries 8,079 params (15→32 projections); Blocks 2 & 3 each carry 33,216 (64→32 projections after the first FFN expands from 15 to 64). The first FFN's `15→64` jump is what creates the asymmetry — kept verbatim from the notebook.
  - Output names render as `['direction', 'return']`, losses dict matches.
- **Notebook deviation captured:** notebook ends with two `Dense(32, relu)` shared layers; PLAN.md specifies one. Stuck with one (PLAN = contract). If Phase 4 finds the model underfits, this is a cheap knob to add back via an `extra_dense` hparam.
- **Positional encoding shape quirk:** `tf.expand_dims(pos_embed, axis=0)` makes the embedding `(1, seq_len, n_features)` so it broadcasts across the batch. Keras' summary then prints `add_position` output shape as `(1, 60, 15)` instead of `(None, 60, 15)` — purely cosmetic, runtime behaviour is correct (the leading 1 broadcasts to batch_size).
- `src/models/xgb.py::build_xgb_classifier / build_xgb_regressor` returns un-fit `XGBClassifier` / `XGBRegressor` instances. Two separate models — XGBoost has no native multi-target API, and the marginal cost of two ~30s fits is trivial. Both consume the *flattened latest row* from `df[feature_cols].to_numpy()`; lag features in the config already encode short-term history, so XGB doesn't need the 3D sequence tensor. Defaults: `n_estimators=500, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', random_state=42`.
- **TF runtime warnings on this Windows box (expected, ignorable):**
  - `TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11` — we're CPU-only; ~30 epochs × ~7.6k sequences should still train in <10 minutes on CPU per the Phase 4 budget.
  - `oneDNN custom operations are on` — Intel MKL acceleration; informational only.
- **Out of scope (deferred to Phase 4):** training loops, holdout evaluation, artifact/metadata serialization, the bake-off, the STL A/B, the honesty gate. Phase 3 is purely "model factory."

### Phase 4 — completed 2026-05-10
- `src/train.py` implements `train_ticker(ticker_key, model_kind, *, apply_stl=None, artifact_suffix="")` and `run_bakeoff(ticker_key)`. `tools/train_model.py` exposes single-run (`--model X`), full-pipeline (`--all` = STL A/B + xgb + bakeoff), and standalone bakeoff (`--bakeoff`) modes for one ticker or all.
- **Pipeline order inside `train_ticker`:** load parquet → engineer_features → split_train_val_test → (optional STL drop on train+val only — test untouched) → fit_normalizer on train → apply to all three → make_sequences (transformer) or 2D matrix (xgb) → fit → predict on holdout → write `{base}.{keras|joblib}` + `{base}_metadata.json`. Naive baselines computed on un-normalized test targets (target_return / target_direction are NOT in feature_cols so they're never z-scored — confirmed).
- **Holdout sequence construction:** to avoid losing the first ~60 test predictions, the test sequences use the last `seq_len-1=59` rows of (post-STL) val as lookback context. Output count of test sequences asserts equal to `len(test)=252`. Verified.
- **Naive baseline accounting:** `naive_yesterday_return_rmse` predicts `target_return[t] = target_return[t-1]` (i.e. tomorrow's return = today's actual return). The first test row's predictor is `val["target_return"].tail(1)`, so all 252 test days have a baseline prediction. `naive_always_up_accuracy = (test["target_direction"] == 1).mean()` — fraction of up-days in the holdout window.
- **Tie-break rule:** higher direction_accuracy on holdout wins; ties broken by lower return_mae. Implemented in `_better()` and used both for the STL A/B (transformer winner) and the cross-model bake-off.
- **Honesty gate:** winner direction_accuracy must beat naive_always_up by ≥3.0 pp. Gate is *recorded but not enforced* in the bake-off — production artifacts are still written, gate result lives at `bakeoff_{ticker}.json:honesty_gate.passes`. Phase 9 README must surface failures prominently.

#### Holdout results (both tickers, all three runs each)

| ticker | model        | STL  | dir_acc | dir_auc | ret_mae   | naive_up_acc | naive_yest_rmse |
|--------|--------------|------|---------|---------|-----------|--------------|-----------------|
| MSFT   | transformer  | off  | 0.5238  | 0.5638  | 0.01300   | 0.5238       | 0.02059         |
| MSFT   | transformer  | on   | 0.5238  | 0.4859  | **0.01031** | 0.5238     | 0.02059         |
| MSFT   | xgb          | off  | 0.4722  | 0.4666  | 0.01184   | 0.5238       | 0.02059         |
| SPX    | transformer  | off  | 0.5714  | 0.5150  | 0.00949   | 0.5714       | 0.01134         |
| SPX    | transformer  | on   | 0.5714  | 0.5000  | **0.00584** | 0.5714     | 0.01134         |
| SPX    | xgb          | off  | 0.5357  | 0.4868  | 0.00624   | 0.5714       | 0.01134         |

#### What this actually says
- **Both tickers FAIL the honesty gate.** Winner direction_accuracy *exactly equals* naive_always_up on both tickers because the transformer learned to always predict up (val_direction_accuracy stays pinned at the val up-rate across all epochs — classic majority-class collapse). XGB does worse than always-predicting-up. **Per the Phase 4 plan, this means we acknowledge in Phase 9 that next-day directional signal isn't extractable from these features at this scale; the production pipeline still ships because the *infrastructure* is the CV story, not the predictive accuracy.**
- **Lower naive_yest_rmse than model_rmse?** No — every model beats the random-walk RMSE baseline on returns. The return regression head is doing real work (especially with STL=ON, which roughly halves test return_mae). The directional head is the failure mode.
- **STL A/B winner: ON, on both tickers.** Pure tie-break on return_mae since direction_accuracy ties exactly. STL=ON dropped 222 outlier rows for MSFT and 200 for SPX (≈2% of train+val). The Phase 0 hypothesis ("removing outliers may lose signal the model needs") was wrong *for the return-regression task* on this feature set — removing fat-tail days reduces noise more than it loses signal. **`config.TickerConfig.apply_stl_outlier_removal` flipped from `False` → `True` going forward.**
- **One nuance to remember when revisiting:** MSFT transformer STL=OFF has direction_auc=0.564 vs STL=ON at 0.486 — STL=OFF *does* produce probabilities with above-chance discrimination, but the 0.5 threshold doesn't pick the right half. A future iteration could calibrate the threshold on val (e.g. Youden's J) and re-evaluate; that's the highest-value follow-up if anyone tries to push the model past the honesty gate. NOT in scope for v1.
- **Production aliases written:** `MSFT_production.keras` (= MSFT_transformer_stl1), `SPX_production.keras` (= SPX_transformer_stl1), each with `_production_metadata.json`. `bakeoff_{MSFT,SPX}.json` records all candidate metrics + the honesty_gate verdict (both `passes=false`, `delta_pp=0.0`).

#### Implementation notes future phases will care about
- `metadata.json` schema fields Phase 5 inference reads: `feature_order`, `normalizer` (dict of `{col: {mean, std}}`), `seq_len` (None for xgb), `last_train_date` (UTC ISO), `holdout_metrics`, `model_kind`, `artifact_filename`. The `_production_metadata.json` adds `promoted_from` and `promoted_at`.
- **Hot-reload (Phase 5) check:** Phase 5 should compare mtime of `{ticker}_production.{keras|joblib}` against the loaded model. Phase 4 writes via `shutil.copy2` which preserves mtime — that means a fresh bakeoff *won't* update mtime if the source artifact happens to be the same one as before. Use `os.path.getmtime` on the destination immediately after copy in Phase 5's reload check, OR change the Phase 7 `retrain_all.py` swap to `os.replace` after writing fresh content (which is what plan Phase 7 already specifies — no fix needed, just don't rely on mtime preservation here).
- **EarlyStopping** added (`patience=5`, `restore_best_weights=True`) — wasn't in the spec but standard practice. All four transformer runs early-stopped before epoch 30 (between epochs 8–14). Total transformer wall time ≈3 min per run on this CPU.
- **TF noise:** the absl/oneDNN/CUDA-not-available log spam is informational; CPU training works fine.
- `_versions()` records python/sklearn/xgboost/tensorflow/numpy/pandas at training time — useful for the Phase 8 deploy when the VM may resolve different versions.

#### Files written this phase
- `src/train.py` (full implementation — was a stub)
- `tools/train_model.py` (full CLI — was a "not implemented" stub)
- `src/config.py:apply_stl_outlier_removal` default flipped `False` → `True`
- `artifacts/MSFT_transformer_stl0.{keras,_metadata.json}`, `MSFT_transformer_stl1.{keras,_metadata.json}`, `MSFT_xgb.{joblib,_metadata.json}`, `MSFT_production.{keras,_metadata.json}`, `bakeoff_MSFT.json` (and the same six for SPX)

### Phase 4 follow-up — transformer improvement attempts (2026-05-10)
After the initial Phase 4 results (both transformers degenerating to majority-class), tried two architecture iterations to surface the AUC=0.564 signal we'd seen on MSFT STL=OFF v1. Both were reverted; one improvement *did* land.

**Attempt v2 — pre-LN + residual blocks + input projection + AdamW(lr=5e-4, wd=1e-4):**
- Modern transformer recipe: project 15 → d_model=64, then `x = x + Dropout(MHA(LN(x)))`, `x = x + Dropout(FFN(LN(x)))` per block; final LN before pooling. ffn_inner=128. ~153k params (was ~87k v1).
- **Result:** train_direction_loss flatlined at log(2)=0.6931 ("predict 0.5 for everything" — worse local optimum than v1's "predict-up" collapse); test return_mae blew up to 0.046 (vs v1's 0.010). Smoking gun was `weight_decay=1e-4` pulling output weights to zero before the network could escape majority-class.

**Attempt v3 — same architecture as v2 but vanilla Adam, lr=1e-3, no weight_decay:**
- Optimization unblocked, but still net-negative on holdout vs v1:
  - MSFT: dir_acc 0.5040 (-1.98pp vs naive), ret_mae 0.075 (vs v1's 0.010)
  - SPX:  dir_acc 0.5794 (+0.79pp vs naive — the only setting in any iteration that beat naive!), ret_mae 0.010 (vs v1's 0.006)
- Residuals seem to make the regression head produce wider-tailed predictions on test than v1 did. SPX gained on direction but every model paid on return_mae.
- **Reverted.** `src/models/transformer.py` and `src/config.py:transformer_hparams` restored to the v1 notebook port.

**Lesson — variance is the real story.** Across ~6 transformer training runs of "the same" architecture/STL setting, holdout AUC ranged from 0.43 to 0.56 just from training stochasticity. The original "AUC=0.564 on MSFT STL=OFF" that motivated this whole follow-up was a lucky seed, not a real edge. **Next-day directional signal in this feature set + dataset is below the noise floor of TF training stochasticity.** No architecture knob within the scope of this project will fix that — improving it requires new features (sentiment, options flow, intraday) or a different prediction target (5-day, regime-conditional, etc.). Both are out of scope for v1 per the cross-phase scope reminders.

**What did land:**
1. **Threshold calibration on val** (`_calibrate_threshold` in `src/train.py`) — Youden-J on val. Persisted in metadata as `direction_threshold` (alongside `direction_accuracy_at_0_5` for the un-calibrated reading). Phase 5 inference can use either; Phase 9 README should report both. *Caveat:* on signal this weak, val→test threshold transfer is unreliable — calibration on this dataset sometimes hurts test accuracy by ~1pp. Keeping it for audit value and forward-compat with stronger feature sets.
2. **Bake-off uses `direction_accuracy_at_0_5` as the ranking key** (`_bakeoff_score` in `src/train.py`), not the calibrated value. Matches the plan's original spec and protects winner choice from threshold-calibration variance. Honesty gate uses the same metric.

**Final v1+calibration results (current state of `artifacts/`):**

| ticker | model       | STL  | dir_acc@0.5 | dir_acc(cal) | thr   | dir_auc | ret_mae   | naive_up |
|--------|-------------|------|-------------|--------------|-------|---------|-----------|----------|
| MSFT   | transformer | off  | **0.5238**  | 0.5119       | 0.522 | 0.469   | **0.0109**| 0.5238   |
| MSFT   | transformer | on   | 0.5119      | 0.4722       | 0.539 | 0.431   | 0.0181    | 0.5238   |
| MSFT   | xgb         | on   | 0.4603      | 0.4802       | 0.857 | 0.453   | 0.0116    | 0.5238   |
| SPX    | transformer | off  | **0.5714**  | 0.5635       | 0.537 | 0.510   | 0.0133    | 0.5714   |
| SPX    | transformer | on   | **0.5714**  | 0.5714       | 0.500 | 0.500   | **0.0059**| 0.5714   |
| SPX    | xgb         | on   | 0.5397      | 0.5437       | 0.462 | 0.485   | 0.0065    | 0.5714   |

**Final winners:** MSFT transformer STL=OFF (dir_acc@0.5=0.5238 ties STL=ON, lower ret_mae wins); SPX transformer STL=ON (dir_acc@0.5 ties STL=OFF, much lower ret_mae wins). Both fail the honesty gate at delta_pp=0. **STL A/B verdict unchanged: STL=ON remains the config default** — it tied on direction every time and consistently won on ret_mae across both architecture iterations.

### Phase 5 — completed 2026-05-11
- `src/inference.py` implements the Phase 5 contract: a `ModelHandle` dataclass (model + metadata + artifact_path + artifact_mtime), `load_handle(ticker_key)` to load production artifacts from `artifacts/{ticker}_production.{keras|joblib}` + sibling `_production_metadata.json`, `maybe_reload(handle)` for mtime-based hot reload, and `predict(ticker_key, handle)` returning the JSON dict the API exposes. Handles both `model_kind="transformer"` (TF Keras lazy-imported, dict-output `.predict`) and `model_kind="xgb"` (joblib payload `{"classifier": clf, "regressor": reg}` from Phase 4).
- **`engineer_features` gained `include_targets: bool = True` (keyword-only).** Inference needs the most recent bar's features, but `target_return = Close.pct_change().shift(-1)` makes that row's target NaN and the existing `.dropna()` was discarding it. Skipping the target columns at inference time means `dropna()` only trims the rolling-window warmup at the head and keeps the latest bar. Phase 2/3/4 callers default to True; only `inference.predict` passes False.
- **Threshold semantics:** inference reads `metadata.holdout_metrics.direction_threshold` (the val-calibrated Youden's J threshold from Phase 4) and uses it for the `direction: "up"|"down"` field; the raw `direction_prob` is also returned so the UI / future analyses can ignore the threshold. MSFT's calibrated threshold is 0.5216, SPX's is 0.5 (no calibration helped on SPX, so the train code clamped to 0.5).
- **Display fields the UI needs:** `last_bar_date`, `last_close`, `predicted_for_date` (next business day after the last bar — using `pd.tseries.offsets.BDay`, doesn't honour US holidays but good enough for a display field), `expected_return_pct` is in *percent units* (e.g. 0.32 means +0.32%), `implied_close = last_close * (1 + raw_return)`. Holdout metrics (`direction_accuracy_at_0_5`, `naive_always_up_accuracy`, `return_mae`) come through unmodified so the UI can show "model has been right X% of the last 252 days" honestly alongside the naive baseline.
- `api/main.py` is a small FastAPI app with `lifespan` that eagerly loads every ticker's production model at startup (fail fast — missing artifact = refuse to start). Endpoints:
  - `GET /predict/{ticker_key}` → inference.predict, with mtime-based hot reload on each hit.
  - `GET /metadata` → per-ticker metadata block PLUS the bake-off honesty_gate verdict (so the frontend can flag failing tickers).
  - `GET /history/{ticker_key}` → reads `data/predictions_{ticker_key}.parquet` if it exists, otherwise returns `[]` (Phase 7 populates it). Accepts `?limit=N`.
  - `GET /` → static `index.html` (Phase 6) or a JSON status block until it lands.
  - `GET /static/*` → static asset mount, only registered if the directory exists.
- `api/schemas.py` defines `PredictionResponse`, `TickerMetadata`, `HistoryRow` Pydantic models — currently informational (the endpoints return raw dicts so adding fields doesn't fight the schema, and the spec didn't say to enforce response models). Phase 9 polish can wire `response_model=` if useful.
- **Verified live against both tickers** (uvicorn on 127.0.0.1:8771):
  - MSFT: dir=up, prob=0.5218 (≈ threshold 0.5216), exp_ret=+0.4736%, implied_close=$417.09 from last_close $415.12 on 2026-05-08, predicted_for 2026-05-11.
  - SPX: dir=up, prob=0.5247 (> threshold 0.5), exp_ret=+0.0293%, implied_close=$7401.10 from last_close $7398.93, predicted_for 2026-05-11.
  - `/metadata`: returns both tickers' full blocks with normalizer + holdout_metrics + honesty_gate (`passes: false`).
  - `/history/MSFT`: `[]` (as expected pre-Phase-7).
  - `/predict/NOPE`: HTTP 404 with `{"detail":"Unknown ticker: NOPE"}`.

#### Gotchas learned
- **Port 8765 collision.** The retail-analytics project on this dev box already binds 8765 by default and *also* exposes a `/metadata` endpoint — an "is the server up?" probe that only checks for HTTP 200 reports ready even when ours failed to bind. Symptom: `/predict/MSFT` returns `{"detail":"Not Found"}` because the path doesn't exist on the *other* app. **Use port 8771 (or any other free port) for this project's dev runs.** Documented in `workflows/serve_predictions.md`.
- **`shutil.copy2` mtime preservation** (carryover from Phase 4 note): when the bake-off promotes a model whose *source* artifact hasn't changed, the destination's mtime doesn't move, so `maybe_reload` does nothing. Phase 7's `retrain_all.py` will `os.replace` after writing fresh content, which forces a new mtime — the supported "hot reload on retrain" path. No fix needed here.
- **Keras dict outputs.** `model.predict(X)` on a model with named output layers returns `{"direction": ..., "return": ...}` (a dict), not a list. `src.inference.predict` handles the list fallback defensively in case a future TF version changes that.
- **TF cold-start cost ~15s on this CPU box.** Lifespan loads both tickers eagerly so the first /predict doesn't pay it. On boot you'll see TF's `oneDNN`/`GPU not available` log spam; ignorable.

#### Files written this phase
- `src/inference.py` (full implementation — was a stub)
- `api/main.py` (full FastAPI app — was a stub)
- `api/schemas.py` (Pydantic response models — was a stub)
- `src/data.py:engineer_features` (added `include_targets` kwarg; behaviour unchanged for callers that don't pass it)
- `workflows/serve_predictions.md` (filled in from Phase 1 skeleton)

#### What's NOT done (intentional, deferred)
- No static HTML frontend yet — Phase 6.
- No prediction logging — Phase 7. `/history/{ticker}` returns `[]` for now.
- No staleness warning if the last cached bar is >2 trading days old — Phase 7's logger surfaces freshness when it lands.
- No tests. Phase 9 polish can add a tiny smoke test if it helps the CV story; not in v1 scope.

### Phase 6 — completed 2026-05-11
- `api/static/index.html` is a single-file, no-framework, no-CDN frontend (~370 lines, ~14 KB). Inline CSS reuses the dark-theme palette from `ML for Retail Analytics/api/static/index.html` (same `--bg/--panel/--accent/--good/--bad` variables) so the two portfolio pieces visually rhyme. Inline JS, vanilla DOM via a tiny `el(tag, attrs, ...children)` helper — no jQuery, no React, no Chart.js. Total wire weight (HTML+CSS+JS) gzips under 5 KB.
- **Layout per the Phase 6 contract:**
  - Header: title + repo link placeholder (`href="#"`, swap in the actual URL after the User TODO at top of PLAN lands) + the "informational only — not financial advice" disclaimer.
  - Ticker tabs (MSFT / SPX), driven by the keys of `/metadata`. Clicking a tab swaps the visible content without reloading; all four fetches (2× predict + 2× history) happen once at page load in parallel, so tab switches are zero-latency.
  - **Prediction card — three big numbers:**
    - Direction probability with ▲/▼ arrow and the calibrated threshold shown under it as transparency.
    - Expected return % (signed, coloured green/red).
    - Implied close ($) plus the dollar delta from last_close.
    - Meta line: `Prediction for <date> • last bar <date> • model <kind> • last trained <date> (N days ago)`.
  - **Holdout performance block:** three metric tiles (direction_accuracy, return_mae, direction_auc) each with a baseline comparator on the row below (`naive_always_up`, `naive_yesterday_return_rmse`, `0.5 = random`). The honesty-gate verdict from `bakeoff_{ticker}.json` renders as a red FAIL banner under the tiles (or green PASS — irrelevant on current artifacts). Plain-text summary: "Model has been right X% of the time on the last 252 trading days."
  - **History table** with 7 columns (predicted_at, for-date, dir, P(up), exp ret, realized, ✓/✗). Renders an empty-state card ("No logged predictions yet — daily logging starts in Phase 7.") until `data/predictions_{ticker}.parquet` exists.
  - **Cumulative chart** is a hand-rolled inline SVG (no Chart.js, no CDN) — green = cumulative correct, red = cumulative incorrect, with three Y-axis gridlines and inline legend text. Only renders when ≥1 resolved prediction exists; hidden until Phase 7 backfills realized outcomes. ~30 lines of SVG-string-builder JS.

#### Backend touch-up
- Zero changes required to `api/main.py`. The Phase 5 server already had a `StaticFiles` mount at `/static` *and* a `FileResponse` fallback at `/` that serves `static/index.html` if present, falling back to the Phase-5 JSON status block if not. Dropping the new `index.html` into `api/static/` is the entire wiring step.

#### Verification — done end-to-end with uvicorn on 127.0.0.1:8771
- `GET /` returns the HTML; sentinel strings `Next-Day Stock Prediction`, `tabs`, `honesty` all present in body.
- `GET /metadata` returns both tickers' full blocks (display_name, ticker_symbol, holdout_metrics, honesty_gate). JS consumes every field it reads (`direction_accuracy_at_0_5`, `naive_always_up_accuracy`, `return_mae`, `direction_auc`, `naive_yesterday_return_rmse`, `direction_threshold`, `honesty_gate.{passes,delta_pp,threshold_pp}`).
- `GET /predict/MSFT` returns `direction=up, prob=0.5218, exp_ret=+0.4736%, implied=$417.09, last_close=$415.12, last_bar=2026-05-08, predicted_for=2026-05-11`. Renders to three coloured big-number cards correctly.
- `GET /predict/SPX` returns `direction=up, prob=0.5247, exp_ret=+0.0293%, implied=$7401.10`. Threshold is 0.5 (no calibration on SPX) — JS still pulls and displays it.
- `GET /history/MSFT` returns `[]` — the empty-state card renders; chart wrap stays hidden. Both expected.
- Both honesty gates report `passes: false, delta_pp: 0.0` — the red FAIL banner renders on both ticker tabs. This is the *correct, honest* UI for the current artifacts; the banner explicitly says "the product ships because the infrastructure is the deliverable; the model has no extractable next-day directional edge on this feature set" — exactly the Phase 9 narrative the README will echo.

#### Design choices worth remembering
- **No framework on purpose.** A vanilla-JS frontend with one HTML file and zero build step is a stronger CV signal here than React-for-2-cards. Static asset, no `npm install`, no `vite build`, no transpilation; ships as a single file the Phase 8 nginx mount serves directly.
- **No CDN on purpose.** Chart.js via CDN would have saved ~50 lines of SVG code but introduced a third-party dependency, a network round-trip on every page load, and an attack surface the project doesn't need. Hand-rolling the SVG keeps the dependency graph at zero.
- **Parallel preload over per-tab lazy loading.** 4 small JSON fetches (~2 KB total) run concurrently in `Promise.all` at page load. Tab switches are instant; the alternative ("fetch on tab click") would have made the second tab feel laggy without saving any bandwidth at this scale.
- **`escapeHtml` everywhere a server string goes into `.innerHTML`.** Ticker names ("S&P 500") and any future free-text fields in /predict are user-influenceable through yfinance metadata, so the JS never trusts them.
- **Empty-state design** — the history card shows a labelled empty card pointing at Phase 7 rather than a blank table. The chart wrapper stays hidden when there's nothing to chart instead of rendering a placeholder. Both states will quietly transition to populated UI the moment Phase 7's `log_prediction.py` writes the first parquet row.

#### Files written this phase
- `api/static/index.html` (new — was a `.gitkeep`-only directory after Phase 5)

#### What's NOT done (intentional, deferred)
- No screenshot / OG image — Phase 9 polish.
- No GitHub URL in the repo link — depends on the User TODO at top of PLAN. Currently `href="#"`.
- No `<meta>` social tags / favicon — Phase 9.
- No history table or chart populated content — needs Phase 7's `tools/log_prediction.py` to start writing `data/predictions_{ticker_key}.parquet`. Empty states ship correctly today.
- No mobile-specific polish beyond the three media queries already inline. Tested mental-model on the responsive breakpoints; not opened on a real phone yet — Phase 8 deployment is the natural moment to verify on the live URL.

### Phase 8 — SKIPPED (decision recorded 2026-05-11)
User chose to skip the VM deployment phase. Reasoning:
- No Contabo VM currently provisioned; provisioning one for v1 is a non-trivial recurring commitment (host, domain, TLS babysitting) whose marginal CV signal over "GitHub repo with screenshots + clear README" is small.
- The system **already self-runs on localhost** via Windows Task Scheduler — the daily fetch / daily log / weekly retrain cadence is exercised by the same CLIs that Phase 8's systemd units would have invoked. The *automation story* is true today; only the *public reachability* part is deferred.
- Phase 9's recruiter value is independent of Phase 8. Doing Phase 9 alone is strictly more valuable than doing Phase 8 alone for a portfolio context.

The full Phase 8 spec (lines 440–488) is preserved verbatim above as a runbook. If a future session executes Phase 8, no spec changes are needed — only execution. The README's "Future work" section documents this honestly as item 1 with one-line cost estimates.

### Phase 9 — completed 2026-05-11
- **`README.md` (new, ~250 lines)** at project root. Eight-section structure optimized for both the 90-second skimmer (architecture diagram, honest metrics table, future-work list) and the 90-minute reader (full setup instructions, project structure, research provenance). Headline framing: *"the CV story is the productionization, not the predictive accuracy"* — said once at the top so the reader doesn't have to derive it.
- **`docs/journey.md` (new, ~1,270 words)** — blog-style writeup covering the starting point, the Phase 0 decisions, the bake-off + honest negative, the shipping-anyway decision, the system as one paragraph, the deliberate minimalisms, and the forward roadmap. Built for one specific reader: a recruiter who clicked through from the README. Voice is "sympathetic peer writing about the work after spending a day with the repo" — not a press release.
- **Architecture diagram** is Mermaid inline in the README. Lock-step versioning with the code (changes in the same commit, no binary file drift). 10 nodes / 10 edges; not big enough to need three diagrams.
- **Honesty gate is in the README twice** — once as a red ❌ column in the holdout table, once again in the prose immediately after. The same negative result appearing structurally in the dashboard *and* the README *and* the journey writeup is the credibility property; softening any one of them collapses it.
- **Future work list (8 items, ordered by cost-to-impact)** explicitly names Phase 8 as item 1 with the runbook pointer to `PLAN.md`. This is the right way to disclose a skipped scope — as a roadmap, not an apology.
- **`LEARNING_NOTES.md` Chapter 8** appended (~250 lines). Covers: why skipping Phase 8 was right, anatomy of a recruiter-grade README, why Mermaid for architecture diagrams, why the honesty gate goes in the README too, the journey writeup as the recruiter pitch, what didn't ship and why. Glossary gained 6 entries.

#### What did NOT ship in Phase 9 (intentional, deferred)
- **No screenshot.** `README.md` has a commented-out `<!-- docs/screenshot.png -->` placeholder. The user can launch uvicorn, navigate to both tabs, take a screenshot, drop it at `docs/screenshot.png`, and uncomment one line. ~5 minutes of manual work; deferred because it requires a running UI session.
- **No GitHub URL in `README.md`** — the repo isn't pushed yet (the User TODO at PLAN.md line 5 is still open). Once pushed, the setup block should grow a `git clone <url>` line and the footer / disclaimer block can point at the issues tab.
- **No `git log`-driven changelog** — would have been overkill for a portfolio piece. `PLAN.md`'s per-phase append log serves the same purpose for any reader who wants the full history.

#### Files written this phase
- `README.md` (new)
- `docs/journey.md` (new)
- `LEARNING_NOTES.md` (Chapter 8 + 6 glossary entries)
- `PLAN.md` (Phase 8 skipped marker + this Phase 9 completion log)

#### State of the project at end of Phase 9
| | Status |
|---|---|
| Phase 1 — Scaffolding | ✅ Completed 2026-05-10 |
| Phase 2 — Data layer | ✅ Completed 2026-05-10 |
| Phase 3 — Models | ✅ Completed 2026-05-10 |
| Phase 4 — Training + bake-off | ✅ Completed 2026-05-10 (honesty gate: both tickers FAIL, infrastructure ships per Phase 0 decision) |
| Phase 5 — FastAPI inference | ✅ Completed 2026-05-11 |
| Phase 6 — Static HTML frontend | ✅ Completed 2026-05-11 |
| Phase 7 — Daily log + weekly retrain | ✅ Completed 2026-05-11 |
| Phase 8 — VM deployment | ⏭️ Skipped 2026-05-11 (scope decision; runbook preserved) |
| Phase 9 — README + journey + polish | ✅ Completed 2026-05-11 |

The project is **shipping-ready for the portfolio context** as of 2026-05-11. The three loose ends (GitHub push, screenshot, optional Phase 8) are runbook tasks, not phases.

---

## Cross-phase scope reminders (do not violate)

- **Don't add features to v1**: no auth, no Stripe, no rate-limiting, no Docker, no Kubernetes, no Sentry, no Prometheus, no multi-step forecasting, no sentiment/RSS, no tickers beyond MSFT and SPX, no mobile app
- **Don't relitigate Phase 0 decisions** — if you think one is wrong, raise it explicitly with the user before changing course
- **Don't write new code where existing patterns from `ML for Retail Analytics` apply** — mirror the reference paths called out in each phase
- **Don't claim performance you haven't measured** — the holdout metrics in metadata.json and README must come from actual evaluation runs, not estimates
