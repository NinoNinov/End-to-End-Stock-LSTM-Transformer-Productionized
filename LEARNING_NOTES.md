# Learning Notes — From Research Notebook to Live Web App

> **Who this is for:** me, a data scientist who builds models in Jupyter and has never shipped one as a live service.
> **Why this file exists:** PLAN.md is the *contract* — what we'll do, in what order. This file is the *story* — what actually happened, why it mattered, and what concepts I had to learn along the way.
> **How to read it:** chronologically. Each phase appends a new "Chapter." Concepts I didn't know are explained in **What this means** boxes inline.

---

## Background: what does "commercial migration" of an ML project actually mean?

In research mode, success looks like: a notebook that loads data, trains a model, prints metrics, makes a chart. You run it manually when you want to. If the data is six months old, that's fine — you're studying *the model*, not predicting tomorrow.

In production mode, success looks like: a service that **runs by itself**, with **fresh data**, that **other people can query**, that **doesn't quietly break**. The model is just one ingredient now. Most of the work is the surrounding plumbing:

| Research notebook needs | Production service also needs |
|---|---|
| Model code | Model code (same) |
| | Data refresh on a schedule (cron / systemd timers) |
| | Versioned, reproducible feature engineering (no ad-hoc cell edits) |
| | Inference path (predict on *one new sample*, not the full training set) |
| | Honest evaluation on data the model has never seen (holdout discipline) |
| | A way to serve predictions (HTTP API) |
| | A frontend so non-engineers can use it |
| | A way to retrain when the world changes |
| | Logging so you can see what predictions were made, and whether they were right |
| | A deployment target (a server somewhere on the internet) |

That's the migration. The notebook becomes ~10% of the system. The other 90% is the part I haven't done before — and the part this project is teaching me.

**Why bother for this particular project?** It's a portfolio piece. A hiring manager looking at my GitHub will see hundreds of "trained an LSTM on stock prices, got 53% accuracy" notebooks. Almost none of them turned into a live URL where the model actually predicts tomorrow on real data, every day, by itself. Doing the productionization is the differentiator.

---

## Chapter 0 — The starting point (May 2026)

What I had:

- A Jupyter notebook (`Deep Learning for Stock Price Prediction with LSTM and Transformers.ipynb`) comparing two architectures on Microsoft stock
- Training data ending December 2024 — **over a year stale** by the time I picked it back up
- A known bug at notebook lines 1634–1636 (denormalization done wrong, so plotted predictions are subtly off)
- No way to make a prediction on *today's* data — the notebook only knows how to evaluate on the test split it carved off during training
- No automation — every refresh would be me, manually, opening Jupyter and re-running cells

This is a *typical* state for a research notebook to be in. None of those issues matter for studying the architectures. All of them matter the second you want it to be a live product.

---

## Chapter 1 — Phase 0: Designing the migration before writing code

> This phase produced no code. It produced a 500-line plan document (`PLAN.md`). That's the point.

In research, jumping straight to code is fine — the "experiment is the spec." In production, the cost of changing direction halfway through is much higher (you've already deployed something, schema migrations are painful, etc.), so we plan first.

### What we decided, and why

**Decision 1: Drop the LSTM for the live product. Keep only the Transformer.**
- Why: shipping two models doubles the work — two retrain pipelines, two monitoring dashboards, two failure modes to debug, two sets of holdout metrics to interpret. The scientific question (LSTM vs Transformer) is already answered in the notebook. The production system doesn't need to keep re-asking it.
- What I gave up: the comparison story. I keep the original notebook in `research/` to preserve it.

**Decision 2: Predict *return percentage*, not absolute price.**
- Why: returns are scale-invariant ("MSFT went up 0.5%" means the same thing whether MSFT is $100 or $400). Absolute prices need denormalization, which is exactly the subtle math that bugged the original notebook.
- What this means in practice: the model output is a number like `+0.0032` (0.32% expected return tomorrow). The UI shows the *implied* tomorrow's price as `last_close * (1 + 0.0032)`, but that's display-only — the model itself never thinks in dollars.

**Decision 3: Add an XGBoost baseline alongside the Transformer.**
- Why: I have ~8,800 daily samples (35 years × 252 trading days). On tabular data of that size with hand-engineered features, XGBoost very often beats deep models. If it does here, *shipping XGBoost is the more honest answer* — and "I A/B'd a complex model against a simple one and shipped the simpler one because it actually won" is a stronger CV story than "I shipped a Transformer because Transformers are cool."
- We'll do a "bake-off" in Phase 4: train both, evaluate both on the holdout, ship the winner per ticker.

> **What this means: bake-off**
> A controlled comparison between candidate models, decided by performance on a dataset *neither model saw during training*. Not "I trained Transformer last week and XGBoost today on slightly different splits and they both look about the same" — that's how research papers lie to themselves. A bake-off uses the *same* train/val/test split for every contender, fits each one, and the holdout numbers decide the winner. No vibes, no architecture loyalty.

**Decision 4: Fix the data leakage in the normalizer.**
- Why: the original notebook fits the z-score normalizer (subtract mean, divide by std) on the *full* dataset, then splits into train/test. That means the test-set values influenced the mean and std the training data was scaled by. The model gets a tiny peek at the test data through the normalizer's parameters — this inflates test accuracy and is a classic bug.
- The fix: fit the normalizer on the **training data only**, then *apply* it to val/test/inference data using the train-derived mean and std.

> **What this means: data leakage**
> Any way information from the test set "leaks" into training, even indirectly. The model didn't *see* the test rows directly, but if any preprocessing step (normalization, feature selection, even outlier removal) was computed on the combined dataset, the test set affected those parameters. Result: holdout metrics look better than they really are, and the model surprises you (badly) in production.

**Decision 5: Multi-ticker via config blocks from day one.**
- Why: launching with just MSFT and then "adding SPX later" usually means a refactor, because hardcoded paths and ticker-specific decisions sneak in everywhere. Cheaper to design it for N tickers from the start, even if N=2 today.
- How: `src/config.py` has a `TICKERS` dict keyed by short name. Adding a third ticker tomorrow is one new entry; no other code changes.

**Decision 6: No Docker for v1.**
- Why: Docker is the "right" answer for production deployments... once you actually need it. For a one-server, one-app, low-traffic portfolio project, plain Python + systemd timers + nginx on a Linux VM is *simpler*, easier to debug, and the user (me) actually understands every line of it. Docker would be cargo-culting.
- Documented as a "future improvement" in the README.

> **What this means: cargo-culting**
> Adopting a tool or pattern because successful projects use it, not because the tool solves a problem you actually have. Docker is great for *fleets* of services, polyglot stacks, multi-stage CI, etc. None of those apply here. Using Docker because "real ML systems use Docker" is cargo-culting.

### What Phase 0 didn't do

It didn't write any code. It didn't create directories. It didn't install anything. It just nailed down the *contract* so that subsequent phases (which I might run weeks apart, in cold sessions where I've forgotten everything) can execute against a clear spec without rediscovering the decisions.

---

## Chapter 2 — Phase 1: Building the skeleton

Phase 1 created the empty directory layout — the *shape* of the production codebase, with no logic in it yet. This is harder to appreciate than it sounds, because it's about deciding **where things go** before there are any things.

### The WAT framework

The directory layout follows a pattern called **WAT** (Workflows / Agents / Tools), which the project's `CLAUDE.md` introduces. The core idea:

- **Workflows** (`workflows/*.md`) — plain-language SOPs. "To refresh the data: do X, then Y, watch out for Z." Written like instructions for a junior teammate.
- **Tools** (`tools/*.py`) — *deterministic* Python scripts that do one thing each (fetch market data, train a model, log a prediction). Boring, testable, reliable.
- **Agents** — the orchestrator (in this project, that's Claude Code, or me running things by hand). Reads the workflow, calls the tools in the right order, handles failures.

> **Why this matters: probabilistic vs deterministic code**
> AI assistants like Claude make decisions probabilistically — they're right ~90% of the time. If you let an AI handle every step of a 5-step pipeline, your overall success rate is 0.9⁵ ≈ 59%. Disaster.
> The WAT pattern keeps the AI in the *coordination* role (which is what it's good at) and pushes the *execution* into Python scripts (which are 100% reproducible). Now the AI just has to pick the right script and pass the right args. Much higher success rate.

This is unfamiliar to me as a DS because notebooks are *the opposite* — every cell is a hand-tuned, possibly-stateful interactive thing. WAT enforces that all the actual work lives in scripts that don't depend on a running interpreter session.

### The directory layout we built

```
project/
├── src/                    # Importable Python library code (model, data, inference)
│   ├── config.py           # Single source of truth for ticker configs, hparams, etc.
│   ├── data.py             # Data fetching, feature engineering, splitting (Phase 2)
│   ├── models/
│   │   ├── transformer.py  # Phase 3
│   │   └── xgb.py          # Phase 3
│   ├── train.py            # Training orchestration (Phase 4)
│   └── inference.py        # Single-prediction logic (Phase 5)
├── tools/                  # CLI entry points — call from cron / systemd
│   ├── fetch_market_data.py
│   ├── train_model.py
│   ├── predict.py
│   ├── retrain_all.py
│   └── log_prediction.py
├── workflows/              # Markdown SOPs
│   ├── refresh_data.md
│   ├── weekly_retrain.md
│   └── serve_predictions.md
├── api/                    # FastAPI service (Phase 5)
│   ├── main.py
│   ├── schemas.py
│   └── static/             # HTML/JS frontend (Phase 6)
├── artifacts/              # Trained model files + metadata (gitignored)
├── data/                   # Cached parquet files (gitignored)
├── research/               # Original notebook + .py — preserved as the scientific origin
├── requirements.txt
├── PLAN.md                 # The execution contract
├── LEARNING_NOTES.md       # ← this file
└── .gitignore
```

> **What this means: `src/` vs `tools/`**
> `src/` is a *library* — it exposes functions like `engineer_features(df)` that other code imports. It never runs by itself.
> `tools/` is a collection of *scripts* — each one is a `python -m tools.<name>` command you can run from a terminal or a cron job. They import from `src/` to do the heavy lifting.
> The split matters because production systems run things on schedules (daily fetch, weekly retrain). Those schedules need *commands they can call*, not *functions they need to import*. Hence `tools/`.

### `src/config.py` — the contract

Every other phase reads from `src/config.py`. Adding a ticker, changing hyperparameters, switching the holdout window — all one-file edits. This is the **single source of truth pattern**.

The bad alternative (which research notebooks usually do): hardcode `"MSFT"` in 14 different places. Then "let's also do SPX" becomes a 14-place rename, and you guarantee you'll miss one.

### Empty stubs that fail loudly

Every Python file in `src/` and `tools/` is a stub right now. They don't contain real code — they contain `raise NotImplementedError("Phase 3")` (or whatever phase will fill them in).

Why bother? Because if Phase 4 accidentally calls `train_ticker(...)` before Phase 3 has built the model, the failure will be:
- ❌ Without stubs: cryptic `ImportError: cannot import name 'build_transformer'` — you have to grep around to figure out what's missing.
- ✅ With stubs: clear `NotImplementedError: Phase 3` — you know exactly what you forgot.

> **What this means: failing loudly**
> When code hits an unexpected state, it should crash *immediately* with a *clear error* pointing at the cause. The opposite is "failing silently" — returning `None`, swallowing exceptions, defaulting to plausible-but-wrong values. Silent failures are how production systems quietly serve garbage for weeks before anyone notices. Loud failures are debuggable.

### The git baseline

Right after the scaffolding existed, we ran `git init` and made a commit. **Before** generating any data, **before** training any model, **before** anything that's expensive or non-deterministic.

Why this order matters: once you start generating artifacts, the scaffolding starts feeling permanent and you stop wanting to revise it. If you commit *after* a week of work, the first commit conflates "the structure" with "the content" and you lose the ability to say "go back to just the skeleton, throw away everything else."

> **What this means: a baseline commit**
> The first commit in a repo. Should contain *only* the empty structure and any config that's part of the contract (here: `requirements.txt`, `src/config.py`, `.gitignore`). No generated data, no model files, nothing that takes hours to recreate. This commit is what `git reset --hard` lands you on if you ever need to "start over without losing the layout."

### The .gitignore subtlety

`.gitignore` had a bug on the first try. I wrote:
```
data/
artifacts/
!data/.gitkeep      # try to re-include the empty-dir marker
!artifacts/.gitkeep
```

Git rejected staging the `.gitkeep` files. Reason: **once a directory is fully ignored, you can't re-include children of it.** Git's matching is greedy at the directory level.

The fix:
```
data/*              # ignore the *contents* of data/, not the directory itself
!data/.gitkeep
artifacts/*
!artifacts/.gitkeep
```

This is the kind of thing nobody teaches you. You just hit it once and remember.

### The venv detour (the painful learning moment)

This is the bit I'd most want to skip telling, which is exactly why it goes in here.

I created the Python virtual environment at `<project>/.venv/`, ran `pip install -r requirements.txt`, and watched it fail partway through TensorFlow with:
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

Two compounding causes:

1. **C: drive only had ~500 MB free.** The Tensorflow wheel alone is 351 MB; the full venv would be ~2 GB. Disk space was the proximate cause.

2. **The venv was sitting inside iCloud Drive.** This was the *real* problem. My project lives in `iCloudDrive\Projects\...`, so when I created `.venv` inside the project, every one of the venv's ~30,000 files (mostly `.pyc` bytecode) was getting indexed for sync to iCloud. Three things go wrong here:
   - Pip churns the disk creating tens of thousands of tiny files; iCloud frantically tries to upload all of them.
   - It eats my iCloud storage quota for files that should never be backed up.
   - **Worst:** iCloud's "Files On-Demand" feature can *evict* file contents to free local space. A `.pyc` file that gets evicted becomes "downloadable on access" — and Python crashes when it tries to import a file that's been turned into a placeholder.

The fix:
1. Cleared 12 GB of unrelated files off C:.
2. Deleted the broken half-installed `.venv`.
3. Created the new venv at **`C:\venvs\stock-predictor\.venv`** — outside iCloud entirely.
4. Reinstalled — succeeded in 4 minutes.

> **What this means: virtual environment (venv)**
> A self-contained Python installation that lives in a folder. Activating it makes `python` point at *that* installation instead of the system one. Why bother: lets each project pin its own dependency versions without fights. Project A can use TF 2.15 and Project B can use TF 2.21, and they don't see each other.
> **Where to put it:** in a fast, local folder that's not synced to any cloud service. Standard idiom on Linux/Mac is `<project>/.venv/`. On Windows with iCloud / OneDrive / Dropbox in the mix, put it somewhere like `C:\venvs\<project-name>\.venv` instead.

To run anything from the venv from the command line, I always use the absolute path:
```powershell
C:\venvs\stock-predictor\.venv\Scripts\python.exe -m tools.fetch_market_data
```
(Activating the venv via `Activate.ps1` works too, but doesn't survive between Claude tool calls — so absolute path is the reliable form for this workflow.)

### Phase 1 outcome

Two git commits. A real Python environment with TF 2.21, xgboost 3.2, pandas 3.0, numpy 2.4, fastapi, sklearn, statsmodels, yfinance — all importing cleanly. An empty WAT-style scaffolding that the next 8 phases will fill in.

Zero lines of model code. Zero predictions. And that's correct — Phase 1's job is to make Phase 2 cheap.

---

## Chapter 3 — Phase 2: The data layer

Phase 2 was the first phase that produced *running* code. By the end I had two parquet files on disk holding 35 years of daily bars for MSFT and ^GSPC, a feature-engineering function that turns raw OHLCV into the 15-feature input the model expects, and a train/val/test splitter that won't let the test set leak into training. About 250 lines of Python in `src/data.py` plus a thin CLI wrapper.

This chapter is also where most of the *learning* happens, because the gap between "wrote a notebook cell that does this" and "wrote a function the rest of the system can call reliably" is wider than it looks.

### Data acquisition: idempotent caching

In the notebook we did `yf.Ticker("MSFT").history(start=..., end=...)` once at the top and kept the result in memory for the whole session. That doesn't work for a service that runs every day for years.

What does work: a **parquet cache** on disk that we *update* incrementally each day. The function `update_cache(ticker, cache_path, years)`:
- If the cache doesn't exist → full backfill (35 years), write parquet.
- If it does → read it, look at the most recent date, ask yfinance for *only* the rows after that, append, sort, dedupe, write back.
- If yfinance returns nothing (weekend, holiday, second run today) → return 0 and leave the file untouched.

> **What this means: parquet**
> A binary, columnar file format optimized for fast reads and good compression on tabular data. Reading a parquet file into a pandas DataFrame is ~10× faster than reading the same CSV, and the file is ~5× smaller. Native to the data engineering world; the cost of using it here is one extra dependency (`pyarrow`). Worth it.

> **What this means: idempotent**
> A function/script you can call any number of times with the same end state. `tools/fetch_market_data.py` is idempotent: run it ten times today and the cache ends up identical to running it once. Critical for cron jobs, where retries, overlaps, and "did it actually run?" are facts of life.

The other detail: the cache filename uses the *config key*, not the yfinance ticker. So `data/SPX.parquet`, not `data/^GSPC.parquet`. The caret character in `^GSPC` is fine on Linux but is a special char in Windows shells, and tying filenames to provider-specific quirks is the kind of small thing that becomes painful three months later. The config key (`"SPX"`) is what the rest of the system speaks.

### Feature engineering: faithfully porting the notebook

The notebook's `preprocess_data` function was a single 60-line block that did everything in place. I ported it to `engineer_features(df, ma_window=20)` in `src/data.py`. The 15 features and 2 targets that come out:

| Group | Features |
|---|---|
| Raw OHLCV | Close, Open, High, Low, Volume_M (volume in millions) |
| Technical indicators | SMA_20, EMA_20, RSI |
| Lag features | lag_close_1, lag_close_2, lag_return_1, lag_return_2 |
| Intraday spreads | Spread_OC = (Open–Close)/Close, Spread_HL = (High–Low)/Close |
| Calendar | day_of_week (0=Mon..4=Fri) |
| Targets | target_return (next-day pct change), target_direction (1 up / 0 down) |

Two places where I deliberately *deviated* from the notebook:

1. **MA window is 20, not 100.** The notebook computed a 100-day moving average but called the column `SMA_20`. Almost certainly a leftover from a parameter sweep — at some point `va_window` was 20, became 100, and the column name didn't get updated. I went with what the name says (20). The cost of that mistake-as-written: ~80 extra warmup rows dropped (because the rolling window stretches further back before any non-NaN value exists). Negligible on 8.8k rows; would have been painful on the 2-year dataset.

2. **Spread denominators are Close instead of Open/High.** The notebook had `(Close - Open) / Open`. The plan says `(Open - Close) / Close`. Same magnitude information, opposite sign convention. I followed the plan because the plan is the contract; the alternative formula is documented as "the notebook did it differently" so a future reader doesn't think I broke it.

> **What this means: contract over folklore**
> When the plan and the notebook disagree, the plan wins, because the plan is the document we agreed on at the start. Notebooks accumulate drift — half-finished refactors, parameter sweeps that left the column names behind. If I just "ported faithfully" I'd inherit those bugs. The fix is: explicit deviations, documented in the docstring, so the deviation is loud.

### Targets: what is the model actually predicting?

This is the conceptual leap from the research notebook. The notebook predicted *next-day Close price* — an absolute dollar amount. To turn that into a useful number the notebook had to *denormalize* the model output (multiply by std, add mean), and that's exactly where the bug at notebook lines 1634–1636 lives.

We sidestep all of that by predicting **return percentage** (`target_return = Close.pct_change().shift(-1)`). Returns are scale-invariant and don't need denormalization. The model's "Close" feature *is* still z-score-normalized as input, but the *output* (a return like `+0.0032`) is in raw units already.

The UI will compute the implied tomorrow-Close as `last_close * (1 + predicted_return)` for display only. The model never sees that conversion.

We also predict **direction** (`target_direction = (target_return > 0).astype(int)`) as a separate target, so the model can output both a probability ("58% chance up") and a magnitude ("+0.32%"). One Transformer with two output heads in Phase 3.

### The leakage fix: fit on train only

Here's the bug from the original notebook, in pseudo-code:
```
features = engineer(full_dataset)
features = (features - features.mean()) / features.std()  # ← leakage
train, test = split(features)
```

The mean and std were computed from the *whole* dataset, including the test rows. So even though the model never *sees* test rows directly during training, it sees scaled inputs whose normalization parameters were partially derived from those test rows. That's a tiny information leak. It inflates test accuracy and surprises you in production.

The fix in `src/data.py`:
```
features = engineer(full_dataset)
train, val, test = split(features)              # split FIRST
normalizer = fit_normalizer(train, feature_cols)  # fit ONLY on train
train = apply_normalizer(train, normalizer, feature_cols)
val   = apply_normalizer(val,   normalizer, feature_cols)
test  = apply_normalizer(test,  normalizer, feature_cols)
```

You can see the effect in numbers. On 35-year MSFT, the **train-only** Close mean is ~$38.52 (because it's averaged over decades when MSFT was cheap). Today's Close is ~$415 — that's a z-score of ~6.7. The full-dataset Close mean is ~$80 (the recent expensive years pull it up), giving a z-score of ~5.9. The model trained on the leaky version sees test inputs that are *less unfamiliar* than they should be. Then production (where every new row is at $400+) feels even further out of distribution than the test split made it look.

> **What this means: fit, then transform**
> Standard scikit-learn pattern: a transformer (scaler, encoder, imputer) has two phases. `fit()` learns parameters from the data you give it. `transform()` applies those parameters. The two must be separated by your train/test split: fit on train, transform on everything. Doing both in one shot on the full dataset is the most common ML bug there is.

### Sequence construction

Time-series Transformers take a window of `seq_len` past time steps and predict the next one. So instead of feeding the model one row at a time (like a tabular regressor), we feed it a 60×15 matrix: 60 days of history, 15 features per day. We need to slide that 60-day window across the dataset and pair each window with the target at its terminal day.

```python
X, y_return, y_direction = make_sequences(train_df, feature_cols, seq_len=60)
# X.shape          == (7627, 60, 15)
# y_return.shape   == (7627,)
# y_direction.shape == (7627,)
```

7627 = 7686 train rows − 60 + 1 (the first 59 rows can't be the *terminal* row of any window). Implemented with `np.lib.stride_tricks.sliding_window_view`, which is O(1) memory because it returns a *view* over the original array — no copying. (We do `.copy()` at the end so downstream code can't accidentally mutate the underlying buffer.)

### Holdout discipline

The function `split_train_val_test(df, holdout_days=252)` carves off the **last 252 trading days** as the holdout (≈ one trading year). The remaining ~96.5% of rows is then split chronologically 90/10 into train/val.

Three things that are *not* allowed to touch the holdout, ever, for any reason, until the end of Phase 4:
- The normalizer fit
- Feature selection
- Hyperparameter tuning

If any of those reads from the holdout, the bake-off is no longer measuring "how well will this do on tomorrow's data?" — it's measuring "how well does it fit the holdout I already peeked at?"

> **What this means: holdout / test set**
> Data set aside *before any modeling decisions are made*. Used exactly once: at the end, to estimate generalization performance. Distinct from the *validation* set, which you can look at as much as you want during model development (for early stopping, hyperparameter selection, etc.). The validation set is "data you're allowed to overfit to in pursuit of a good model"; the holdout is "the answer you write in pen at the end and never erase."

### A surprise: STL was about to delete the events I want the model to learn

The notebook's `preprocess_stock_data` runs an **STL decomposition** on Close, computes the residual, and replaces values whose residual exceeds 3σ with linearly interpolated values. The motivation is sensible in classical time-series stats: separate signal from noise.

I ran `stl_outlier_mask` on the most recent ~1480 days of MSFT and got 28 flagged rows. Spot-checking the dates: **2021-12-10, 2021-12-20, 2022-02-23, 2023-03-10, 2023-04-25.** These are all real fat-tail event days — selloff weeks, the Russia/Ukraine invasion, the SVB collapse. *These are exactly the days the model needs to learn from.* If we silently interpolate them away, we've taught the model "the world is approximately Gaussian, surprises don't happen."

So in this codebase STL outlier removal is a **config flag, off by default**, and Phase 4 will A/B it on the holdout to settle the question empirically. If it actually helps generalization, fine — but the prior is "don't delete the data points your loss function is most sensitive to."

> **What this means: 3σ outlier removal in financial returns**
> Daily return distributions for stocks have **fat tails** — extreme days are dramatically more likely than a normal distribution predicts. Mechanically applying a 3σ filter to "clean" them is mathematically reasonable and empirically wrong: the 3σ days are often where most of the year's information lives. This is one of the standard ways finance ML projects quietly fool themselves.

### Memory pressure → lazy import

A small Windows-specific gotcha: importing `statsmodels.tsa.seasonal.STL` at the top of `src/data.py` paired with yfinance's `curl_cffi → asyncio` import chain crashed Python with `OpenBLAS error: Memory allocation still failed after 10 retries` and a `MemoryError`. Both libraries pull in OpenBLAS during their initial import, and this machine has tight RAM headroom. The fix is to **defer the statsmodels import** to inside `stl_outlier_mask`, so the module only pays for it when STL is actually requested:

```python
def stl_outlier_mask(df, sigma=3.0):
    from statsmodels.tsa.seasonal import STL  # lazy
    ...
```

The hot path (fetch / engineer_features / split / sequences) never imports statsmodels, so the daily refresh and (eventually) the inference call don't pay for it. Lazy imports are usually a code smell — you give up "all imports work at module load time" — but for libraries that are large *and* used in only one branch *and* causing real failures, it's the right tool.

> **What this means: heavy import**
> Some Python libraries (statsmodels, scipy, tensorflow, torch) execute substantial work at *import time*: loading C extensions, initializing thread pools, allocating BLAS workspaces. On a tight machine, importing all of them eagerly can trip the OS's memory limits even if you weren't going to use them. Lazy imports defer that work until the first call.

### Idempotency in practice

```
$ python -m tools.fetch_market_data
2026-05-10 16:55:39,303 INFO MSFT: cache up to 2026-05-08, +8813 new rows
2026-05-10 16:55:39,821 INFO SPX: cache up to 2026-05-08, +8813 new rows

$ python -m tools.fetch_market_data
2026-05-10 16:55:53,019 INFO MSFT: cache up to 2026-05-08, +0 new rows
2026-05-10 16:55:53,150 INFO SPX: cache up to 2026-05-08, +0 new rows
```

First run: empty cache, full backfill, ~8800 rows per ticker. Second run a few seconds later: nothing new since 2026-05-08 (today is Sunday May 10), no writes, exits in under half a second. That's the contract a daily cron job needs.

### Phase 2 outcome

Two small changes to the directory: `data/MSFT.parquet`, `data/SPX.parquet` (gitignored — about 390 KB each). One real `src/data.py` (250 lines) replacing a stub. One real `tools/fetch_market_data.py` (50 lines) replacing a stub. One filled-out `workflows/refresh_data.md`.

The `engineer_features` ↔ notebook port had three deliberate fixes baked in (MA window, spread formula, train-only normalizer) — each one documented in the docstring so they aren't surprises later.

The next phase (Phase 3) doesn't touch any of this. It just builds the model classes — Transformer with dual output heads, XGBoost baseline — that consume what `make_sequences` produces. Phase 2 is the contract; Phase 3 implements against it.

---

## Chapter 4 — Phase 3: The model factory

Phase 3 is the smallest in the whole plan in terms of lines of code (~150 in `src/models/`), but it's where the *shape* of what the model promises to produce gets pinned down. After Phase 3 there is no more architectural ambiguity — Phases 4-7 just have to make this object good and serve its outputs.

### What "model factory" means

In a research notebook the model is a literal block of code at the top of section 4. You execute it once, the variable `model_transformer` exists, you call `model.fit(...)` directly. There's no "factory" — the model is a thing, not a recipe.

In production, every retrain run, every cold-restart of the API, and every unit test needs a *fresh* model object with the same architecture and the same compile config. So we wrap the construction in functions:

```python
def build_transformer(seq_len, n_features, hparams) -> tf.keras.Model: ...
def build_xgb_classifier(hparams)  -> XGBClassifier: ...
def build_xgb_regressor(hparams)   -> XGBRegressor: ...
```

Each returns a *new* uncompiled or pre-compiled estimator. Phase 4 will call these. The retrain timer (Phase 7) will call them again next Sunday on fresher data. The smoke tests (whenever I write them) call them with tiny `seq_len=5, n_features=3` to verify the wiring without spending compute. Same recipe, many uses.

### Dual output heads — what changed from the notebook

The original notebook's Transformer ended like this:
```python
y_output = layers.Dense(1, name="y_output")(x)        # next-day Close (regression)
x_output = layers.Dense(5, name="x_output")(x)        # last timestep features (auxiliary)
```

The auxiliary `x_output` was a clever-ish trick — predict the input's last row as a regularizer — but it's not what we want to ship. The product is "direction probability + expected return." So our Transformer ends:

```python
direction = layers.Dense(1, activation="sigmoid", name="direction")(shared)
return_   = layers.Dense(1, activation="linear",  name="return")(shared)
```

And the compile config matches the two heads with two different losses:
```python
model.compile(
    loss={"direction": "binary_crossentropy", "return": "mse"},
    loss_weights={"direction": 1.0, "return": 1.0},
    metrics={
        "direction": ["accuracy", tf.keras.metrics.AUC(name="auc")],
        "return":    ["mae"],
    },
)
```

> **What this means: dual-head model**
> A single neural network that produces *more than one output* from a *shared backbone*. Up to the last 2-3 layers, the model is one big representation extractor; only the final `Dense` layers split into separate prediction heads. Why this is more than just "two models taped together":
> - The shared layers learn features useful for *both* tasks. If understanding tomorrow's direction also requires understanding tomorrow's magnitude (and it usually does), the shared representation gets gradient pressure from both losses and ends up better than either alone.
> - One forward pass → both predictions. Half the inference cost of two models.
> - But: the two losses can fight. If `binary_crossentropy` (typical scale: 0.5-1.0) is much larger than `mse` on returns (typical scale: 1e-4-1e-3), the direction head dominates training. That's what `loss_weights` is for. We start with 1.0/1.0 and tune in Phase 4 if the metrics show one head starving the other.

### Why two XGBoost models, not one

XGBoost's `XGBRegressor` and `XGBClassifier` each handle exactly one target. There is no off-the-shelf dual-head XGB. Could I bolt one together? Probably. Should I? No — both fit in seconds, both share zero state, and "two simple things" is more honest than "one fancy thing."

So `src/models/xgb.py` is just two thin factories returning standard sklearn-style estimators with the hparams from `config.py` plugged in. Phase 4 will call `.fit()` on each.

> **What this means: XGBoost (XGB)**
> A gradient-boosted decision tree library. Roughly: train a small tree to predict the target, look at where it's wrong, train a *second* small tree to predict those errors, add it to the ensemble, repeat ~500 times. Each tree is shallow (here: max_depth=6) so individual trees underfit; the *ensemble* of hundreds gets very accurate. Fast to train (~seconds on tens of thousands of rows), interpretable via feature importance, and famously hard to beat on tabular data with engineered features. That last bit is the entire reason it's our baseline: on ~7.7k samples × 15 features, deep learning's edge often disappears.

### The data shape contract

This is the moment where the data layer (Phase 2) and the model layer (Phase 3) lock together.

| Model | Input shape | Output |
|---|---|---|
| **Transformer** | `(batch, seq_len=60, n_features=15)` — the 3D tensor `make_sequences` produces | `{"direction": (batch, 1), "return": (batch, 1)}` |
| **XGB classifier** | `(batch, n_features=15)` — the *latest row* of features only, NOT a sequence | `(batch,)` — `predict_proba` for `P(up)` |
| **XGB regressor** | `(batch, n_features=15)` | `(batch,)` — predicted return |

So at evaluation time, for the same row, the Transformer sees the trailing 60 days, while XGB sees only that one row's features (the lag features encode the past it cares about). They make their predictions, we compare on holdout, the better one wins. Two paths into the same prediction.

### Parameter count sanity check

The compiled Transformer with default hparams has **86,671 parameters** for `(seq_len=60, n_features=15)`. Cross-checked against the training data:

- Train sequences: ~7,627 (after the 252-day holdout carve-off + 60-row sequence warmup)
- Parameter / training-sample ratio: ~11

That's small enough to train without massive overfitting on a regular CPU laptop. Modern vision/language Transformers have ratios in the thousands and require billions of training tokens to keep them in check. A ratio of 11 says "this model can comfortably memorize the training data, but not exotically so" — early stopping and dropout will be enough to keep it honest.

> **What this means: param/sample ratio**
> A rough proxy for overfitting risk. If your model has more parameters than training examples (ratio > 1), you can in principle memorize the training set. The interesting question is by how much. Ratio ~10 with regularization (dropout, weight decay, early stopping) usually trains cleanly. Ratio ~1000 needs much stronger regularization or more data. Ratio < 1 (parameters < samples) was the deep-learning rule until ~2019, after which "scaling laws" rewrote the playbook for very large models with very large datasets.

### A small Keras quirk worth noting

When I call `tf.expand_dims(pos_embed, axis=0)` to give the position embedding a leading batch dimension of 1, Keras' `model.summary()` displays the resulting shape as `(1, 60, 15)` instead of `(None, 60, 15)`. The `1` is misleading — at runtime it broadcasts to whatever the actual batch size is. This is purely a printing artifact of how Keras infers static shapes through `tf.expand_dims`. The model trains and predicts correctly with batches of any size.

I noticed this, double-checked by running a forward pass mentally, and decided not to fight it. Could replace with a Lambda or a custom layer that produces a `(None, 60, 15)`-shaped tensor, but the cost (more code, more abstraction) outweighs the benefit (a prettier printed summary).

> **What this means: leaky abstractions**
> Sometimes a tool's high-level abstraction (here, "the model knows its own shapes") doesn't perfectly hide the lower-level reality (broadcasting rules, static shape inference). The right move is usually to *understand* the leak rather than paper over it. Future me will thank present me when the same shape print appears in a different context.

### Phase 3 outcome

`src/models/transformer.py` and `src/models/xgb.py` go from one-line stubs raising `NotImplementedError("Phase 3")` to fully implemented model factories. The factories are pure (no side effects, no file I/O), parameterized entirely by `hparams` dicts read from `src/config.py`. A unit test in Phase 4 (or right now, mentally) confirms shapes match what `make_sequences` produces.

Zero training has happened. The next phase will actually fit these models on real data and run the bake-off — the moment of truth where I find out whether a 35-year-trained Transformer can beat a "predict up tomorrow" coin flip on the 252-day holdout.

---

## Chapter 5 — Phase 4: Training, the bake-off, and the moment of truth

Phase 4 was where the system met reality for the first time. Up to now everything had been infrastructure — directories, configs, factories, stubs. Phase 4 fits actual models on actual data and asks the one question that matters: *do they beat a coin flip?*

The honest answer turned out to be **no**. Which is its own kind of valuable, and is the point of having designed the "honesty gate" in advance.

### The shape of the phase

Three deliverables:

1. **`src/train.py:train_ticker(ticker_key, model_kind, *, apply_stl)`** — the orchestrator. Loads parquet → engineers features → splits 90/10/holdout chronologically → optionally drops STL outliers from train+val → fits the train normalizer → applies it everywhere → fits the model → predicts on the holdout → writes the artifact and a sibling metadata JSON.
2. **`src/train.py:run_bakeoff(ticker_key)`** — given the trained models for one ticker, decides the winner and writes "production" aliases plus a `bakeoff_<ticker>.json` audit file.
3. **`tools/train_model.py`** — CLI with three modes: `--model X` (one run), `--all` (transformer STL A/B + xgb + bakeoff), `--bakeoff` (re-decide using existing metadata, no re-training).

The whole loop for one ticker (`--all`) takes ~10 minutes on this CPU laptop. STL A/B = train transformer twice, once with outlier removal on and once off. Plus one XGB run. Plus a bake-off comparison. Then symlink-equivalent (just `shutil.copy2`) the winner to `<ticker>_production.{keras|joblib}`.

### Naive baselines: what counts as "beating the model" lower bound

Without a baseline, "53% direction accuracy on a 252-day holdout" sounds vaguely impressive. With baselines, you find out it isn't.

Two we compute alongside every model's holdout metrics:

- **`naive_always_up_accuracy`** = fraction of up-days in the test window. The accuracy you get if you predict "tomorrow is up" every single day. For MSFT 2025-ish, this is ~0.524; for SPX, ~0.571. If a Transformer scores 0.524 on MSFT, the Transformer is not adding any signal — it's just discovered that majority-class collapse is a comfortable place to settle.
- **`naive_yesterday_return_rmse`** = the RMSE you get if you predict tomorrow's return equals today's actual return (a random-walk persistence model). Sanity check for the regression head.

> **What this means: naive baseline**
> A trivially-simple "prediction" you can compute without any model. The point isn't to be a great predictor — it's to be the *bar* a real model has to clear. Common examples in time-series: persistence (tomorrow = today), seasonal naive (tomorrow = same-day-last-week), or the per-class base rate (always predict the majority class). If your fancy model can't beat these on a held-out window, you don't have a model — you have an expensive coin flip.

### The honesty gate

PLAN.md spells it out: *if the production model for a ticker doesn't beat naive_always_up by ≥3 percentage points on direction accuracy, do not proceed to API/UI for that ticker as if it had real signal.* The CV story becomes "honest evaluation found X" instead of "I shipped despite no signal." The gate doesn't *block* deployment — both models still got production aliases — but it gets recorded in `bakeoff_<ticker>.json:honesty_gate.passes` and will get surfaced prominently in the Phase 9 README.

Why 3 percentage points and not 1 or 5? It's a soft heuristic for "more than noise." On 252 trials with ~52% base rate, the standard error of accuracy is ~3pp; one standard error past the baseline is the absolute minimum to call something a real (but small) effect. 3pp is "barely real." Anything below that is in the noise.

### The actual numbers (Phase 4 final state, v1 architecture)

| ticker | model | STL | dir_acc | dir_auc | ret_mae | naive_up | gate |
|--------|-------|-----|---------|---------|---------|----------|------|
| MSFT   | transformer **(winner)** | off | 0.5238 | 0.469 | 0.0109 | 0.5238 | **FAIL** (Δ +0.0pp) |
| MSFT   | xgb | on | 0.4603 | 0.453 | 0.0116 | 0.5238 | (loser) |
| SPX    | transformer **(winner)** | on | 0.5714 | 0.500 | 0.0059 | 0.5714 | **FAIL** (Δ +0.0pp) |
| SPX    | xgb | on | 0.5397 | 0.485 | 0.0065 | 0.5714 | (loser) |

Read it carefully: **the transformers' direction accuracies *exactly equal* the naive_always_up baselines.** That's not a coincidence — it's the model collapsing to "predict up every day" because, given the noise in the input, that's the lowest-loss strategy under binary cross-entropy. AUC of 0.469 / 0.500 confirms it: at-or-below chance on ranking, which is what you'd expect from a model that's effectively outputting a constant.

XGB scored *worse than always-up* on both tickers — it tried harder than the transformer to find a signal and ended up overfitting to noise in the opposite direction.

The return-regression head, on the other hand, *does* beat its naive baseline on every model. SPX transformer's holdout ret_mae of 0.0059 vs the random-walk RMSE of 0.0113 — the regression head is doing real work; the directional head is the failure mode.

### The STL A/B verdict

`config.apply_stl_outlier_removal=False` was the original default; Phase 4's A/B was supposed to settle it empirically. Across both architecture iterations of the transformer, STL=ON came out ahead on `return_mae` every time. The reason: STL drops the fat-tail days (selloffs, FOMC, gaps) from the training set, which reduces the variance of the regression target on the days the model sees. Test-set return MAE drops because the model isn't trying to fit the unfittable.

This is the *opposite* of what we feared in Phase 2 — that STL would delete the days the model needs to learn from. For *direction* the worry was correct (STL doesn't help — both settings collapse to majority class). For *return regression* it was wrong (STL is a clean win). The config default flipped `False → True`.

> **What this means: an A/B for a config flag**
> Picking a binary flag's default empirically rather than from priors. You train two models that differ only in that one setting, compare them on a holdout they both saw equally, and let the numbers pick. Cost: 2× training time for one run. Benefit: a defensible default with a number attached.

### Threshold calibration: the one improvement that survived

After the initial Phase 4 results showed both transformers collapsing, I wanted to try improving the model — modern transformer architecture (pre-LN, residual connections, input projection), a different optimizer (AdamW with weight decay), threshold calibration on val. The architectural changes didn't pay off (more on that below). But threshold calibration is still in `src/train.py`, and the metadata persists it for Phase 5 inference to use.

The setup: the model outputs a probability `p ∈ [0,1]` for "tomorrow is up." Default cutoff: predict "up" iff `p > 0.5`. But that cutoff is arbitrary — if the model's probability distribution is biased, 0.5 may not be where the classes actually separate.

The fix: on the val set, sweep all candidate thresholds and pick the one that maximizes Youden's J = TPR − FPR (equivalent to maximizing balanced accuracy). Save that threshold in metadata as `direction_threshold`. At inference time, use it instead of 0.5.

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, p_val)
j = tpr - fpr
optimal_threshold = thresholds[np.argmax(j)]
```

> **What this means: Youden's J statistic / threshold calibration**
> A way of picking a classification threshold from val data after the model is already trained. J = sensitivity + specificity − 1. Maximizing it picks the operating point furthest from chance on the ROC curve. Useful when (a) the model produces well-ranked probabilities (high AUC) but is biased away from 0.5, or (b) the class balance differs between val and test. *Caveat* — it only helps if the optimal-threshold-on-val transfers to test. If signal is noisy, val and test optimal thresholds disagree, and calibration can *hurt* test accuracy by 1-2pp.

Here, calibration didn't help (on weak/no signal, the val→test transfer is unreliable). But it's recorded for forward-compatibility: if a future feature set produces real signal, calibration becomes free win.

The bake-off ranking explicitly does **not** use the calibrated metric — it uses `direction_accuracy_at_0_5`. Otherwise the winner choice would get warped by threshold-calibration variance, which is exactly the noise we're trying to defeat.

### What didn't work: the v2/v3 architecture experiments

The original transformer is a verbatim port of the research notebook's encoder block, which has two structural quirks compared to modern transformers:

1. No **residual connections** — the block computes `attn → dropout → LN`, with no skip path from the input. Three of these stacked make gradients route through a series of attention layers, which is well-known to hurt training.
2. Post-LN ordering — LayerNorm comes *after* dropout, not before the sublayer.

So I tried **v2**: input projection to a fixed `d_model=64`, modern pre-LN blocks with residuals, `AdamW(lr=5e-4, weight_decay=1e-4)`. Bumped FFN inner width to 128 (≈2 × d_model, standard).

Result: train_direction_loss flatlined at log(2) ≈ 0.6931 ("predict 0.5 for everything" — a worse local optimum than v1's "predict-up" collapse), and test return_mae blew up to 0.046 (vs v1's 0.010). The smoking gun was `weight_decay=1e-4` pulling output weights to zero before the network could escape majority class.

> **What this means: weight_decay**
> An L2 regularization term added to the optimizer that pulls weights toward zero on every step. In AdamW it's decoupled from the gradient (different from "L2 loss" in classic Adam). Helpful when the model has way more capacity than data — it prevents memorization. *Unhelpful* when the model is struggling to learn even the basics, because it adds a constant pull toward "output nothing." On this 7.6k-sample problem with already-modest capacity (~150k params), wd=1e-4 was too aggressive.

So I tried **v3**: same architecture, but vanilla Adam, lr=1e-3, no weight decay. The optimizer unstuck; the model trained. But the holdout still came in net-negative vs v1:

- MSFT: dir_acc 0.5040 (−1.98pp vs naive), ret_mae 0.075 (vs v1's 0.010)
- SPX: dir_acc 0.5794 (**+0.79pp vs naive — the only setting in any iteration to beat naive**), but ret_mae 0.010 (vs v1's 0.006)

SPX got a small directional gain. Both tickers paid on regression. The residual path seems to produce wider-tailed predictions on test than the no-residual notebook architecture does, which hurts MAE on financial returns where the tails are exactly where mistakes are expensive.

I reverted to v1 architecture.

> **What this means: residual / skip connection**
> A connection that adds the input of a block back to its output: `x_out = x_in + Sublayer(x_in)`. Two things they do: (a) give gradients a path through the network that doesn't depend on the sublayer learning anything sensible — critical for deep networks because plain stacked layers compose multiplicatively and vanishing/exploding gradients become near-certain past 5-10 layers; (b) make the *identity* a valid solution — if the sublayer is useless, the network can just ignore it and the residual passes the input through. **Why it didn't help here:** my network is only 3 blocks deep. Vanishing gradients aren't the bottleneck. The model isn't underfitting — it's correctly identifying that there is *no learnable pattern* and falling back to the only stable strategy (predict the base rate). No amount of architectural cleverness fixes "the input doesn't predict the output."

> **What this means: pre-LN vs post-LN**
> Where the LayerNormalization goes inside an encoder block. **Post-LN** (the original transformer paper, 2017): `LN(x + Sublayer(x))` — normalize after the residual addition. **Pre-LN** (GPT-2 onward, ~2019): `x + Sublayer(LN(x))` — normalize before the sublayer. Pre-LN is more stable to train, especially in deep stacks, and is the default in everything modern. For 3 shallow blocks the difference is tiny; for 30+ blocks it's the difference between training and diverging.

### The real finding: signal is below the noise floor

Across ~6 transformer training runs of *nominally identical* architecture and STL setting (v1 → v2 → v3, MSFT + SPX), holdout AUC ranged from **0.43 to 0.56** just from training stochasticity. TF on CPU isn't fully deterministic even with seeded numpy + `tf.random.set_seed`; dropout, parallelism, and floating-point reductions introduce real variance.

The first v1 run's "MSFT STL=OFF: dir_auc = 0.564" — the one number that looked like a real signal in the original Phase 4 results — was a lucky seed. Re-running it produced 0.469. The "edge" was inside the noise.

> **What this means: noise floor**
> The minimum performance difference detectable on a given evaluation. Determined by: holdout size (smaller = noisier), label noise (more noise = harder to find signal), and training stochasticity (different seeds = different scores even for the same model). On a 252-day binary classification holdout with ~50% base rate and training-seed variance of ±5pp in AUC, anything claiming a 2-3pp directional edge is *not distinguishable from random*. Real signal needs to clear the noise floor consistently across re-runs, not on a lucky seed.

This is consistent with decades of finance literature: short-horizon (1-day) directional prediction from OHLCV + technicals alone is *very close to efficient* in the academic-sense — the predictable component is small enough that it gets dominated by training noise on a 252-day window. Things that do produce real edges (in the literature, in practice): news/sentiment in real time, options-implied volatility, intraday microstructure, alternative data (credit card, satellite, social), regime-conditional models, and longer horizons (5-day, 1-month).

All of those are out of scope for v1 by the cross-phase scope reminders. Phase 4's contribution is **infrastructure plus an honest negative result**. The CV story is "I built the pipeline, ran the honest evaluation, and the honest evaluation said the signal isn't there in this feature set" — which is a much better story than the dozens of "I trained an LSTM, got 53% accuracy, didn't validate, claimed success" notebooks on GitHub.

### Atomic artifact swap

A small but production-important detail: when the bake-off picks a winner, it doesn't *write* a new model file — it `shutil.copy2`s the existing variant's file to the canonical name `<ticker>_production.{keras|joblib}`. Sibling metadata gets the same treatment.

Why copy and not move/symlink:
- **Move**: would delete the variant-specific file. Lose the audit trail (which STL setting won, both candidates' metrics).
- **Symlink**: Windows-native symlinks need administrator privileges. Don't fight the OS for a portfolio project; copy is 1 MB.
- **Copy**: keeps every intermediate artifact for inspection, costs a few MB per retrain.

`shutil.copy2` (not just `shutil.copy`) preserves mtime from the source — which is a tiny gotcha for the Phase 5 hot-reload check (compare API's loaded-model mtime vs file mtime to detect retrains). The fix: Phase 5 will record its own load timestamp instead of relying on file mtime, OR Phase 7's `retrain_all.py` will use `os.replace` after a fresh write (which is what PLAN.md already specifies). Noted for that phase.

### Phase 4 outcome

- `src/train.py` (full implementation — 400 lines): `train_ticker`, `run_bakeoff`, naive baselines, threshold calibration, metadata serialization.
- `tools/train_model.py` (CLI — 90 lines): per-(ticker, model) runs, full pipeline, standalone bake-off.
- `src/config.py`: `apply_stl_outlier_removal` default flipped `False → True`.
- `artifacts/`: 14 files per pipeline run (4 transformer variants × {.keras, .json}, 1 xgb × {.joblib, .json}, 1 production × {.{keras|joblib}, .json}, 1 bakeoff JSON), times 2 tickers.

Two transformer architecture attempts tried and reverted; the lesson (the variance dominates the signal at this horizon) recorded in PLAN.md's per-phase append log. Improving the *model* further is deferred until the *site* is built — at that point a real production loop is up, daily predictions are being logged, and any future feature-engineering work has a place to live and an audit trail to verify against.

The remaining phases (5 — FastAPI, 6 — frontend, 7 — automation, 8 — VM deploy, 9 — README) don't depend on the model being good. They build the system *around* the model. The honest "the signal isn't there yet, here's exactly why" is itself the most interesting thing to put in the README.

---

## Chapter 6 — Phase 5: Wrapping the model in HTTP

Phase 5 turned the trained artifacts into a *service*. After Phase 4 I had `.keras` and `_metadata.json` files on disk; after Phase 5 I have `curl localhost:8771/predict/MSFT` returning live JSON. About 350 lines of Python in `src/inference.py` + `api/main.py` + `api/schemas.py`.

The headline: **everything wired on the first try except for the port collision, which is a story all on its own.**

### What the API has to do

The contract from PLAN.md:

| Endpoint | Purpose |
|---|---|
| `GET /predict/{ticker_key}` | Run the model on today's cached data, return a JSON prediction for tomorrow. |
| `GET /metadata` | Per-ticker normalizer + holdout metrics + honesty-gate verdict (the frontend reads this once at load time). |
| `GET /history/{ticker_key}` | Past predictions with realized outcomes — empty until Phase 7 starts logging. |
| `GET /` | Serve `index.html` (Phase 6). Until then, a JSON status block so a visiting human sees something useful. |

Notice that none of those endpoints take a request body. The "input" is the cached parquet file on disk; the API itself is a thin shell over `src/inference.predict(ticker_key)`. This is the right amount of decoupling for a portfolio piece — the prediction logic doesn't know it's being served over HTTP, and tomorrow's CLI / cron / scheduled job can call the same function without going through HTTP at all.

> **What this means: thin API layer**
> The HTTP layer's job is *adaptation* (URL routing → function call, return value → JSON), not *logic*. Business logic lives in functions that can be called by tests, scripts, or other code paths. If you can replace FastAPI with Flask (or remove HTTP entirely) by editing one file, you've drawn the boundary right. If half your prediction logic lives inside `@app.get(...)` handlers, you haven't.

### The lifespan loader: paying TF's cold-start cost once

TensorFlow's `keras.models.load_model(...)` takes ~15 seconds on this CPU box, mostly initializing the eager runtime and rebuilding the computation graph from the `.keras` file. Doing that on every `/predict` request would mean a 15-second first response per ticker, every cold start, and an unhappy frontend.

FastAPI's `lifespan` context manager runs once at startup and once at shutdown:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    handles = {}
    for ticker_key in TICKERS:
        handles[ticker_key] = load_handle(ticker_key)  # ~15s TF import once
    app.state.handles = handles
    yield                                              # ← server runs here
    app.state.handles = {}                             # cleanup (no-op for in-mem)
```

The `app.state` namespace is FastAPI's official place to stash request-shared resources (DB pools, ML models, configs, anything heavy you load once). Each handler reads from `app.state.handles[ticker_key]`. Total cost: ~30s startup, ~50ms per request. Worth it.

> **What this means: lifespan vs. deprecated `@app.on_event`**
> Older FastAPI tutorials use `@app.on_event("startup")` / `@app.on_event("shutdown")`. Those are deprecated. The modern form is a single async generator that yields between setup and teardown — both phases live in one place, share local variables, and let `async with` semantics handle cleanup if startup raises. If you see `on_event` in a new tutorial in 2026, it's outdated.

The "fail fast" decision: if any ticker's production artifact is missing at startup, the lifespan raises a clear `RuntimeError`. The alternative would be lazy-loading on first request, but that lets a broken deploy serve 500s indefinitely. **Refusing to start beats serving errors** — the deploy script either succeeds or fails loudly, never partially.

### Hot reload via mtime: how a weekly retrain takes effect

PLAN.md Phase 7 will set up a Sunday cron that retrains both models and atomically replaces `artifacts/{ticker}_production.keras`. The API has to pick up the new file *without restarting*, because restarting an HTTP server drops in-flight connections (no one cares for a portfolio project, but the technique is good practice).

The pattern: each `ModelHandle` stores its source path's mtime at load time. Every `/predict/*` call calls `maybe_reload(handle)`:

```python
def maybe_reload(handle):
    current_mtime = handle.artifact_path.stat().st_mtime
    if current_mtime > handle.artifact_mtime:
        return load_handle(handle.ticker_key)   # cold reload
    return handle                                # cheap no-op
```

The cost on the happy path (no retrain happened) is one `stat()` syscall — negligible. On the rare path (Phase 7's retrain just landed), one /predict call pays the 15-second TF load and subsequent calls are fast again.

> **What this means: mtime-based hot reload**
> A lightweight pattern for "the data on disk might have changed; reload only if it actually did." Works because the OS gives you a cheap "when was this last written?" check that you can compare to your in-memory snapshot. Suitable for single-writer / single-reader setups (one cron job writes, one server reads). For multi-writer or distributed systems you'd need something stronger (file checksums, atomic versioned filenames, message queues).

There's a subtle gotcha here that PLAN.md flagged in Phase 4: `shutil.copy2` *preserves* the source file's mtime. So when the bake-off promotes a model whose source file hasn't changed, the destination's mtime doesn't move, and `maybe_reload` doesn't fire. The fix lives in Phase 7's `retrain_all.py`: use `os.replace` after writing fresh content. The hot-reload path is correct *for the way Phase 7 writes*; Phase 4's bake-off copies don't need it.

### A small data layer change: keep the last bar

This was the one place I had to touch already-written code. Phase 2's `engineer_features` does:

```python
out["target_return"] = out["Close"].pct_change().shift(-1)
out["target_direction"] = (out["target_return"] > 0).astype("int8")
...
out = out.dropna()  # drops rolling-NaN at head AND the last row (NaN target)
```

That's correct for training — the last row's target is unknown, so you can't fit on it. But it's wrong for *inference*, where the last row's features are *exactly* what we want to feed the model to predict tomorrow.

The change: a keyword-only `include_targets: bool = True` parameter. Inference passes `False` and only the target columns are skipped:

```python
def engineer_features(df, ma_window=20, *, include_targets=True):
    out = df.copy()
    # ... 15 feature columns ...
    keep_cols = [...15 feature columns...]
    if include_targets:
        out["target_return"] = ...
        out["target_direction"] = ...
        keep_cols = keep_cols + ["target_return", "target_direction"]
    out = out[keep_cols].dropna()
    return out
```

Default behavior is unchanged, so Phases 2/3/4 don't notice. Only `inference.predict` opts in.

> **What this means: keyword-only argument (`*` in the signature)**
> Python syntax that *forces* callers to pass an argument by name, not by position. Without the `*`, `engineer_features(df, 20, False)` would compile but read as "ma_window=20, include_targets=False" — a sentence no one wants to write. With the `*`, the only legal form is `engineer_features(df, 20, include_targets=False)`, which is self-documenting. Always use keyword-only for boolean flags and any parameter where the name carries more meaning than the position.

### The percent-units decision

The model outputs raw returns (`0.0032` for "expected +0.32%"). The API returns `expected_return_pct: 0.4736` — in *percent units* — so the frontend can write `${expected_return_pct.toFixed(2)}%` without any multiplication. This is a tiny choice but I wrote it down in the docstring because it's exactly the kind of thing that bites three months later when someone (me) wonders why their JavaScript chart is off by a factor of 100.

The raw-return value is reconstructable: `raw = expected_return_pct / 100`. The `implied_close = last_close * (1 + raw)` is computed server-side so the frontend doesn't need to know about it. UI math is fragile; backend math is testable.

### Schemas that aren't enforced (yet)

`api/schemas.py` has `PredictionResponse`, `TickerMetadata`, `HistoryRow` Pydantic models. I wrote them but *didn't* attach them as `response_model=` to the endpoints. Reasons:

1. The metadata block has nested dicts (normalizer, holdout_metrics, train_window) that I haven't bothered to type. With `response_model` Pydantic would strip them or complain. Leaving the endpoints returning raw dicts means adding a new field is a one-line change.
2. The frontend (Phase 6) is the only consumer; it'll grab fields it needs and ignore the rest. There's no external client whose contract I'm preserving.
3. Phase 9 polish can wire response models if it helps the CV story ("look, typed API").

> **What this means: response_model vs. raw dict**
> FastAPI's `response_model=PredictionResponse` does two things: (a) generates an OpenAPI schema (so `/docs` shows clients what to expect), (b) *strips* any fields not in the schema, silently. The strip is a footgun if you add a new field server-side and forget to update the schema — the field just disappears from the response. For internal APIs where the only client is your own frontend, returning a raw dict (and letting OpenAPI infer the response shape) is often less brittle.

### The port-collision learning moment

Here's the part I'd most want to skip telling, which (per the established tradition) is exactly why it goes in.

I started uvicorn on port 8765 (because that's the port the retail-analytics project uses, and habit is a hell of a drug). The "wait for ready" probe — `curl http://127.0.0.1:8765/metadata` — returned HTTP 200 instantly. I ran the prediction endpoints and got:

```
$ curl localhost:8765/predict/MSFT
{"detail":"Not Found"}
$ curl localhost:8765/
<!DOCTYPE html>... Retail Analytics ML — live demo ...
```

The probe lied because *the retail-analytics server was already on 8765*, my server failed to bind (Windows error 10048), and the probe happily talked to the wrong app. The retail app coincidentally also exposes `/metadata`, so the 200 response didn't even raise a red flag.

Three lessons that compound:

1. **A liveness probe should test *your* endpoint, not just any endpoint.** `/metadata` exists on many ML services. `/predict/MSFT` is much more specific to this app — a 200 there is real evidence that *my* code is serving.
2. **Multiple servers per machine need a per-project port map.** A `dev_ports.md` somewhere listing "retail=8765, stock=8771, …" would have prevented this. (Action item for me, not done in this phase.)
3. **`run_in_background` can't tell you a process exited.** The exit-1 from uvicorn ended up in a log file that I had to explicitly fetch and read. The clue ("only one usage of each socket address") was in the output the whole time; I just didn't check it because the probe said "ready."

After moving to port 8771 everything worked. Documented the port choice in `workflows/serve_predictions.md` so future me doesn't repeat it.

> **What this means: the right level of skepticism**
> An automated check that returns "OK" is *evidence*, not *proof*. The strength of the evidence depends on how specific the check is to what you actually want verified. A generic "did some server bind this port?" check answers a strictly weaker question than "did *my* server come up correctly?". When debugging weird production issues, this distinction is often the whole story.

### What I have at the end of Phase 5

```bash
$ curl localhost:8771/predict/MSFT | python -m json.tool
{
    "ticker": "MSFT",
    "ticker_key": "MSFT",
    "display_name": "Microsoft",
    "last_bar_date": "2026-05-08",
    "last_close": 415.12,
    "predicted_for_date": "2026-05-11",
    "direction": "up",
    "direction_prob": 0.5218,
    "direction_threshold": 0.5216,
    "expected_return_pct": 0.4736,
    "implied_close": 417.0859,
    "model_kind": "transformer",
    "last_train_date": "2026-05-10T16:12:27+00:00",
    "holdout_accuracy": 0.5238,
    "holdout_naive_accuracy": 0.5238,
    "holdout_return_mae": 0.01087
}
```

That JSON is the entire reason this project exists. Every line in `src/`, `api/`, `data/`, and `artifacts/` is here to make that one HTTP call return that one object on demand, with up-to-date data, every day, forever, by itself.

It's also worth noting what's *honest* about it:
- `direction_prob: 0.5218` — barely above the calibrated threshold of 0.5216. The model is reporting "very mild lean up," not "I'm confident."
- `holdout_accuracy: 0.5238` *equals* `holdout_naive_accuracy: 0.5238`. The UI can show both side-by-side and the user will see the gap is zero — exactly the honesty-gate result baked into the response.

The next phase (6) puts a thin JS frontend around this JSON. Nothing in the model or API changes; the response shape was designed (in this phase) to make the frontend's job trivial.

---

## Chapter 6 — Phase 6: A frontend that fits in one file

> The Phase 5 API speaks JSON. This phase wraps it in something a non-engineer can look at, on their phone, in five seconds, and walk away knowing what the model thinks tomorrow will do — and how much to trust that.

### The premise: do less, on purpose

The reflex I had to talk myself out of: "open this up with `npm create vite`, get React, set up routing, configure Tailwind, install Chart.js, deploy a Vercel preview." All of that is what production frontends look like. None of it is what *this* frontend needs. The page has:

- Two tabs.
- One prediction card with three numbers.
- One holdout-metrics card with three more numbers.
- One history table that's empty for now.

That's it. Reaching for a framework on a four-element page is *cargo-culting* the production stack of a job you don't have yet. Worse, it'd hide the part of the project that's actually impressive — a hiring manager looking at the repo would see "another React app" instead of "a single static HTML file that knows enough about the API to render an honest dashboard."

So the rule for Phase 6 was: **one HTML file, vanilla JS, no CDN, no build step, no transpiler, no dependencies**. The whole UI ships as one file that nginx can serve as-is. The repo's dependency graph for the frontend is the empty set.

> **What this means: cargo-culting (frontend edition)**
> Adopting React/Tailwind/Vite because all the apps on your timeline use them, not because you have a problem they solve. React shines when you have shared state spread across hundreds of components and a team of frontend engineers. For a four-element page, it adds 40 KB of runtime, a build step, and a folder of config files that someone (you, in 6 months) has to understand to make a CSS change. The cost compounds; the benefit is zero.

### What the page actually contains

```
┌─ Header: title + repo link + "informational only — not financial advice"
├─ Tabs: [ Microsoft (MSFT) ] [ S&P 500 (^GSPC) ]
│
├─ Card 1 — Tomorrow's prediction
│    ┌─ Direction probability ─┐  ┌─ Expected return ─┐  ┌─ Implied close ─┐
│    │ ▲ 52.18% up             │  │ +0.47%            │  │ $417.09         │
│    │ threshold 52.16%        │  │ next-session …    │  │ from $415.12 …  │
│    └─────────────────────────┘  └───────────────────┘  └─────────────────┘
│    Prediction for 2026-05-11 • last bar 2026-05-08 • last trained 2026-05-10 (1 day ago)
│
├─ Card 2 — Holdout performance (last 252 trading days)
│    ┌─ dir acc ─┐ ┌─ ret MAE ─┐ ┌─ AUC ─┐
│    │ 52.38%    │ │ 1.087%    │ │ 0.469 │
│    │ naive 52% │ │ naive 2%  │ │ rand  │
│    └───────────┘ └───────────┘ └───────┘
│    🚨 Honesty gate: FAIL — delta vs naive_always_up is +0.00 pp (required ≥ 3.0 pp).
│       The product ships because the infrastructure is the deliverable; the model has
│       no extractable next-day directional edge on this feature set.
│
└─ Card 3 — Recent predictions (last 30 days)
     ⓘ No logged predictions yet — daily logging starts in Phase 7.
```

Everything you see is driven by JSON the API already returned. The frontend has no opinions of its own about prediction logic; it's an adapter from `{...}` to "things a human can read."

### The honesty banner

The part I'm proudest of is the red FAIL banner on both ticker tabs. It says, in plain language, that the model has no extractable edge over "always predict up" on the holdout. Most ML demos I've seen would have buried this — show the accuracy as a big green number, omit the baseline, hope the recruiter doesn't notice. This one puts the baseline directly under the accuracy in the same font size, then surfaces the verdict from `bakeoff_{ticker}.json:honesty_gate.passes` in a red box you can't miss.

The reason this is a *feature*, not an embarrassment: the entire premise of the bake-off in Phase 4 was "don't ship if you can't beat baseline by 3 pp." The artifacts didn't pass. The plan said the product still ships because the *infrastructure* is the CV story. The UI exists to make both of those decisions visible — the work, *and* the honest verdict on what the model can and can't do.

A frontend that lies about its model is one search-Ctrl-F away from being unmasked. A frontend that pre-empts the criticism by stating the negative finding clearly is much stronger. It's the same principle that makes a research paper with a "limitations" section more credible than one without.

> **What this means: honest-negative UX**
> A user-facing surface that proactively discloses the model's failure modes instead of waiting to be caught. The bar is "would a sophisticated reader walk away with the correct mental model?" — not "does the page look impressive at a glance?" In a portfolio context, this is the differentiator: most demos are pitched, this one is *evaluated*.

### Hand-rolling the cumulative chart

The plan suggested "tiny line chart (Chart.js via CDN, or hand-rolled SVG)." I went hand-rolled SVG. The reasoning chain:

- **Chart.js via CDN** = 50 fewer lines of JS, but adds a CDN round-trip (slower first paint), a third-party dependency I have to trust forever, and ~70 KB of code to do what 30 lines of SVG-string-building can.
- **Inline SVG** = no dependency, no network, no version drift, but I have to compute the x/y mapping myself.

The x/y mapping is six lines of arithmetic. Worth it.

```js
const xAt = (i) => padL + (innerW * i) / (n - 1);
const yAt = (v) => padT + innerH - (innerH * v) / maxY;
const path = (pts, getter) =>
  pts.map((p, i) => `${i === 0 ? "M" : "L"}${xAt(i)},${yAt(getter(p))}`).join(" ");
```

The output is two coloured polylines (green = cumulative correct, red = cumulative wrong) on a panel-coloured rect with three gridlines. Zero dependencies. Scales to whatever the container width is because the SVG `viewBox` does the math for me.

> **What this means: inline SVG vs canvas vs charting library**
> Three levels of abstraction. **SVG** = declarative DOM elements; the browser handles compositing, accessibility, and scaling. Cheap to hand-roll for static charts; expensive for animations. **Canvas** = imperative pixel buffer; better for thousands of points or fast animation, worse for accessibility (the chart isn't in the DOM). **Charting library** (Chart.js, Recharts, Plotly) = a higher abstraction on top of canvas/SVG with sensible defaults, themes, interactivity. The right choice depends on chart complexity: under ~5 data series and no animation, inline SVG wins; above that, a library pays for itself.

### Parallel preload over per-tab lazy loading

The frontend has four JSON resources to fetch:
- `/metadata` (the tabs need it)
- `/predict/MSFT`, `/predict/SPX` (the cards)
- `/history/MSFT`, `/history/SPX` (the tables)

The naive approach: fetch on tab click. That'd save bandwidth at page load and make the second tab feel laggy. The better approach for *this* page (two tickers, ~2 KB total): preload everything in parallel at page load:

```js
await Promise.all(keys.flatMap((k) => [
  fetchJSON(API.predict(k)).then((p) => { STATE.predictions[k] = p; }),
  fetchJSON(API.history(k)).then((h) => { STATE.histories[k] = h; }),
]));
```

`Promise.all` runs all the fetches concurrently and waits for all of them. `flatMap` is how I splice two fetches per ticker into one flat array. The result is tab switches that are pure DOM updates with zero network — instant. The user pays a few hundred ms on first load and never pays again.

This trade-off flips at scale (50 tickers, you'd lazy-load) but at N=2 the math is decisive.

> **What this means: `Promise.all` + `flatMap`**
> `Promise.all([p1, p2, ...])` waits for *all* promises to resolve and returns an array of their values; if any rejects, the whole thing rejects. `flatMap(fn)` is `map(fn).flat(1)` — useful when each input produces *multiple* outputs and you want them flattened into one array. Together they're the standard recipe for "kick off N×M independent fetches and resume when they're all in."

### The `el(tag, attrs, ...children)` helper

A 12-line function that's worth more than its size suggests:

```js
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("data-")) node.setAttribute(k, v);
    else node[k] = v;
  }
  for (const c of children) {
    if (c == null) continue;
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else node.appendChild(c);
  }
  return node;
}
```

This is roughly what JSX compiles down to. Once you have it, building UI in vanilla JS reads like:

```js
el("div", { class: "metric" },
  el("div", { class: "k", text: "Direction accuracy" }),
  el("div", { class: "v", text: "52.38%" }),
);
```

…which is shorter than the equivalent JSX *and* doesn't need a build step. The trick that makes it ergonomic is the distinction between `text:` (safe — uses `textContent`, can't be HTML-injected) and `html:` (explicit — uses `innerHTML`, used only for strings *I* built from numbers).

> **What this means: `textContent` vs `innerHTML`**
> `textContent` sets the element's text and HTML-escapes everything; it cannot inject markup. `innerHTML` parses the string as HTML. The rule: if the string came from a server response, a URL parameter, or anywhere a user could influence it, use `textContent`. Use `innerHTML` only when *you wrote the string yourself*, character by character, from values you control. Mixing them up is the source of approximately every XSS vulnerability in web history.

### Empty states are part of the design

The history card has two valid renderings: a populated table + chart, or an empty-state card saying "No logged predictions yet — daily logging starts in Phase 7." Both ship today. The empty state doesn't say "loading…" forever or render a broken-looking blank table; it explicitly communicates *why* it's empty and *when* it'll fill up.

This is the kind of UX detail that separates "I built a thing" from "I shipped a thing." The empty case is the case for the first 30 days after this lands in production — and is the case any reviewer hitting the live URL today will see. It has to look intentional.

> **What this means: empty-state design**
> The treatment of "no data yet." A well-designed empty state explains *why* (no logged predictions yet), *when it'll change* (Phase 7 logging starts), and *what to do* (nothing — wait for the system). A poorly-designed empty state is a blank table that looks like a bug. Most UIs are designed for the populated case and then improvised for the empty one; flipping that order makes the product feel finished from the first second of its life.

### What changed on the backend: nothing

I didn't have to touch `api/main.py` for Phase 6. The Phase 5 server already had a `StaticFiles` mount at `/static` *and* a `FileResponse` fallback at `/` that serves `static/index.html` if present, falling back to the Phase 5 status JSON if not. Dropping the new HTML into `api/static/` is the entire wiring step.

This is the payoff for Phase 5's "thin API layer" discipline — the API and the UI are decoupled enough that the UI can ship without breaking anything else, and the API doesn't need to know there's a UI. If Phase 9 decides to swap in a different frontend (a static-site generator, a React SPA, whatever), the work is one file's worth.

### What I have at the end of Phase 6

A working web app at `http://localhost:8771/` with two ticker tabs, three big-number prediction cards, an honest holdout metrics block with a red FAIL banner, and an empty history card waiting for Phase 7's logger. Total deliverable: one file, ~370 lines, zero dependencies, zero build steps, gzips to under 5 KB.

What the page communicates in five seconds, without the viewer reading any text:
- "There's a prediction for tomorrow." (big number 1)
- "The model thinks barely up." (52% probability, ▲ in green but pale)
- "The model isn't actually better than guessing up every day." (red banner under the metrics)

That last bullet is the entire reason the page is honest. Every other portfolio ML demo I've looked at would have hidden it.

The next phase (7) makes the history card stop being empty: a daily cron-like job appends one row per ticker per day, and a separate weekly retrain job swaps fresh artifacts in atomically. The frontend doesn't change; the empty state quietly transitions to a populated state the moment the first parquet row lands.

---

## Chapter 7 — Phase 7: Closing the loop

> Phase 5 made the model serveable. Phase 6 made it presentable. Phase 7 makes it *self-running* — every weekday the system writes down what it predicted, every following day it writes down whether it was right, and every Sunday it refits its weights against fresh data. Everything still runs on localhost; Phase 8 will lift this to a real server. The shape of the automation has to be solid here first, because a bad cron job on localhost is annoying; a bad cron job on the production VM at 3 AM is silent and sends nobody an email.

### The two jobs and why they're separate

Two scripts land in this phase:

| Script | Cadence | What it does |
|---|---|---|
| `tools/log_prediction.py` | Daily, 22:15 | Append today's prediction per ticker; backfill yesterday's outcome if the next bar has now landed. |
| `tools/retrain_all.py` | Weekly, Sun 03:00 | Refit the production model_kind per ticker on the latest cached data; atomically swap the artifact; rotate the previous version to `.prev.*` for rollback. |

The reason they're separate scripts — not one omnibus "nightly" — is that they have **different blast radii**. The logger appends a row to a parquet; if it crashes you lose a day of history and nothing else. The retrainer rewrites the production artifact; if it crashes mid-swap or produces garbage weights, every live prediction is affected until rollback. Two scripts means each one can fail loud, on its own schedule, without contaminating the other.

> **What this means: blast radius**
> The scope of damage a single failed job can do. Logging has a small blast radius (one missing parquet row). Retraining has a medium blast radius (every subsequent prediction is wrong until rollback). Splitting jobs by blast radius makes operational reasoning tractable: you respond to a failed logger differently than to a failed retrain, and you don't want one alert channel buried under the other's noise.

### `log_prediction`: the persistent memory of the system

Before Phase 7, the system had no memory. Each `/predict` call was forgetful — same parquet in, same answer out, no record. After Phase 7, every prediction the model ever made is on disk: `data/predictions_{ticker_key}.parquet`. This is what `/history/{ticker_key}` reads to populate the dashboard's "Recent predictions" table. It's also the substrate for any future analysis ("how does Friday accuracy compare to Monday?") — none of which is possible without first writing things down.

The row schema is wider than the API response on purpose:

```text
predicted_at                  — UTC timestamp when the script ran
predicted_for_date            — next-trading-day heuristic (display field)
last_bar_date_at_prediction   — what the model actually saw  ← key for backfill
last_close_at_prediction      — what the implied_close was anchored on
direction, direction_prob, direction_threshold
expected_return_pct, implied_close
model_kind, last_train_date

# backfill columns — null at append, filled on the *next* run that can see the outcome:
realized_date, realized_close, realized_return_pct, was_correct
```

The fields the API doesn't expose are the ones that **make backfill correct in the presence of holidays**. The naive design ("predict on day D for day D+1, backfill by looking up date D+1 tomorrow") breaks when D+1 is a US market holiday — the next bar is actually D+2. So instead of trusting `predicted_for_date` (which uses a `BDay(1)` heuristic and doesn't know about Memorial Day), backfill keys off `last_bar_date_at_prediction` — the date of the actual bar the model last saw — and finds *the first market bar strictly after it*. Whatever date that is, that's the realized close.

> **What this means: backfill anchored on what the model saw, not on a heuristic**
> When recording a forecast for future verification, the anchor field shouldn't be your guess of *when* the verification will happen — it should be the data you used to make the forecast. Calendar heuristics drift; data lineage doesn't. If the model fed on the bar dated D, the verification is whatever bar came next, regardless of how many holidays sat between them.

### Idempotency without database constraints

The script will be run by Windows Task Scheduler (and later systemd timers). Both have "rerun on failure" semantics — a job that succeeded partially and was killed will fire again. So the logger has to be re-run safe.

The dedup key is `predicted_for_date`. If a row already exists for that date in the parquet, the new run appends nothing. This is the parquet equivalent of `INSERT ... ON CONFLICT DO NOTHING`, with the dedup key chosen so that the natural cadence (one run per day, after `fetch_market_data` moves last_bar_date forward) produces exactly one row per trading day.

A subtle correctness property falls out: re-running the logger *before* fetching fresh data is also harmless. `last_bar_date` hasn't advanced, so `predicted_for_date` hasn't advanced, so the dedup hits, so nothing is logged. If the user fixes that ordering bug at 22:30 by running the fetch first, the next logger run *does* append, because last_bar_date moved.

> **What this means: idempotency in batch jobs**
> A batch job is idempotent if running it twice produces the same end state as running it once. For an appending logger this means picking a dedup key that uniquely identifies a "logical run." `predicted_at` (the wallclock at exec time) would be a bad key — every retry is unique, you'd get duplicates. `predicted_for_date` is a good key — it's what differentiates "today's prediction" from "yesterday's prediction" from the model's perspective.

### Backfill in one pass, not a sweep job

A first instinct: have a *separate* `backfill_outcomes.py` script that runs once a week, sweeps every unresolved row, fills it in. That's how a data warehouse would handle it — separate ETL stages.

For a single-process, file-based system, that's overkill. The append job already touches the parquet on every run. While the file is open, it can do the backfill pass for free: any pending rows whose next bar is now available get resolved before the append happens. One write, both jobs done.

The backfill pass is O(unresolved_rows × log(market_dates)) per run — `searchsorted` over a sorted DatetimeIndex. For a year of daily predictions that's ~252 rows × 18 lookups = ~4500 operations, taking single-digit milliseconds. The opposite of a problem.

> **What this means: piggyback writes vs separate ETL passes**
> When a process is already opening a file to mutate it, doing additional bookkeeping on the same open is usually cheaper and simpler than scheduling a separate sweep. The separation makes sense at scale (different jobs, different SLAs, different failure isolation) but is overhead in a single-process system. Pick the design that matches the system's size, not the design that matches the architecture diagrams of larger systems.

### Atomic writes for the prediction log too

The parquet write at the end of `log_prediction` uses the same `tmp + os.replace` dance as the retrainer. This is overkill for a small file in a single-process system — until the moment your machine loses power between the `pd.to_parquet` and the OS flushing the bytes. Then `predictions_MSFT.parquet` is a half-written file the next run can't read, and you've lost all of history.

The discipline is cheap to apply once and removes a class of failures. Three lines of code (tmp suffix, write, replace) buy "the file on disk is *always* one of: untouched, or a complete write."

> **What this means: atomic file write pattern**
> Write to `dst.tmp`, then `os.replace(dst.tmp, dst)`. `os.replace` is atomic on the same filesystem (POSIX *and* Windows since Python 3.3). A process or power failure during the temp write leaves only `dst.tmp` — readers never see a half-written `dst`. The pattern costs one extra rename per write and is the standard way single-writer systems get crash-consistency for free.

### `retrain_all`: atomicity is the whole point

The retrainer's job is one sentence: replace the production artifact with a fresh one without breaking the API. Everything else is plumbing.

The full sequence per ticker:

1. Read `artifacts/{ticker_key}_production_metadata.json` to learn the current production family (`transformer` or `xgb`) and the STL setting.
2. Call `src.train.train_ticker(ticker_key, model_kind, apply_stl=...)`. This writes a fresh `{ticker_key}_{model_kind}.{ext}` plus its metadata sibling — the *non-production* slot. Phase 4 used the same path; this overwrites it cleanly.
3. **Rotate the current production files to `.prev.*`** via `os.replace`. After this step, `{ticker_key}_production.{ext}` does not exist on disk — there's a 1-millisecond gap where the API would see "no file." The gap is tolerated by `maybe_reload`: if `stat()` fails, the old in-memory handle is kept.
4. **Write fresh production bytes via `.tmp + fsync + os.replace`.** `fsync` forces the OS to flush the bytes to disk *before* the rename swaps them in — otherwise a power loss between the rename and the actual write could leave a metadata pointer to bytes that never made it to disk.
5. Same `.tmp + fsync + os.replace` dance for the production metadata sidecar (with `promoted_from` / `promoted_at` / `promotion_source: "retrain_all"` fields mutated in).

The whole sequence is **failure-tolerant at every step**: if training raises, step 3 never runs, and production is untouched. If step 3 runs but step 4 raises, the `.prev.*` file is still there and a manual rollback is a single rename. If step 4 succeeds but step 5 raises, the API has a fresh model but stale metadata — the next request reloads the model (mtime changed) but reads old metadata fields like `last_train_date`. That's a recoverable degradation; nothing serves errors.

> **What this means: fsync before os.replace**
> `os.replace` is atomic at the *filesystem metadata* level (the directory entry flips from old inode to new). But the bytes of the new file might still be in the OS page cache, not yet on the physical disk. A power loss in that window leaves you pointing at a "file" whose data was never flushed. `fsync` forces the bytes down before the rename, closing the window. The cost is one syscall and a few milliseconds; the benefit is crash-consistency that survives unexpected reboots.

### Why "no bake-off on weekly retrain"

PLAN.md is explicit: the weekly retrain refits the **current production family**, not the full bake-off. Asked another way: every week, why don't we re-decide whether transformer or XGBoost wins?

The reason is that the bake-off in Phase 4 produced a verdict (`honesty_gate.passes = False`, transformer ~= XGBoost ~= naive baseline, within 5pp seed noise). Re-running it weekly would just resample that noise — sometimes XGBoost looks slightly better, sometimes transformer, sometimes the order flips. Promoting on each flip would produce a production line whose model identity flickers, with no underlying improvement in skill.

The bake-off is a *human-supervised* decision because it changes the architecture story. Weekly retrain is a *machine* decision because it only updates the weights inside a chosen architecture. Mixing them would mean the model identity drifts based on noise. Keeping them separate means: until a human runs `tools/train_model --bakeoff`, the production family is stable; only the weights age.

> **What this means: model identity vs model weights**
> Two different things to retrain. Weights = the floats inside a chosen architecture, refit on new data. Identity = which architecture and which hyperparameters were chosen. Weights can be refit automatically because the comparison is "new weights vs old weights of the same model" — small differences are expected. Identity changes should be human decisions because they reshape the comparison frame. Conflating the two produces drift you can't reason about.

### A real bug from smoke-testing: the `/history` endpoint's NaT handling

I left smoke-testing in for Phase 7 (Phase 5 deserved more of it; I didn't). Running `log_prediction` once produces a single row whose `realized_*` columns are all null — exactly what `/history` will serve. I replayed the endpoint's serialization path in Python and found this:

```json
"realized_date": NaN,
"realized_close": null,
"realized_return_pct": null,
"was_correct": null
```

Three columns serialize as proper `null`. One column emits a literal `NaN` token. `JSON.parse` in any browser rejects `NaN` (`SyntaxError: Unexpected token N`). The history table would render as broken on the first day after deploy.

The cause: `dt.strftime("%Y-%m-%d %H:%M:%S")` on a column that contains `NaT` produces an output column with `float('nan')` mixed in (pandas substitutes NaN for NaT when the result dtype is string). The Phase 5 code then did `df.where(df.notnull(), None)` to convert nulls — but for a string-typed column with mixed NaN floats, `where` silently leaves the NaN floats untouched. The fix is two extra lines: `df.astype(object).where(pd.notna(df), None)` *and* a final per-row scrub that replaces any leftover `float('nan')` with `None`.

This is a class of bug — "the test would have caught it, but I didn't write the test" — that this project has had on a low simmer since Phase 4. The smoke test that found it was 15 lines and three minutes of effort. The honest lesson is to make those 15 lines the default at the end of every phase that touches a serialization boundary.

> **What this means: NaT vs NaN in pandas date columns**
> `pd.NaT` is the missing-value sentinel for datetime dtypes. It's distinct from `float('nan')` even though they print the same way. When you reshape a datetime column to string via `.dt.strftime`, NaT gets converted to NaN — and NaN inside a string-dtype Series is a permanent foot-gun for any "replace nulls with None" pass. The robust pattern is `df.astype(object).where(pd.notna(df), None)` *plus* a per-cell sweep, because `df.notnull()` and `pd.notna()` don't catch the same set of sentinels across all dtypes.

### Smoke-testing made cheap, on purpose

I wrote two throwaway scripts in `.tmp/` (a gitignored scratch directory) during this phase:

1. `smoke_backfill.py` — synthesized a backfillable scenario by rewriting a logged row to claim an earlier `last_bar_date_at_prediction`, ran `log_one("MSFT")` again, and printed the resolved row. Confirmed `realized_close=420.77`, `realized_return_pct=+1.6451`, `was_correct=True`.
2. `smoke_history.py` — replayed the `/history` serialization path locally (no uvicorn) and ran the result through `json.dumps(allow_nan=False)`. This is the strict mode that browsers use; it raises if the JSON contains `NaN`. That's what surfaced the bug above.

The scripts cost five minutes to write, exist outside the source tree, and get deleted when done. The value isn't in the scripts themselves — it's in the discipline of *not trusting the code until something has fed it inputs and looked at outputs*. Phase 4 and Phase 5 had moments where I trusted code that hadn't been exercised; Phase 7 didn't.

> **What this means: throwaway smoke tests as a discipline**
> A smoke test isn't a unit test. It doesn't go in the repo, doesn't get a CI runner, doesn't need a name. It exists for the five minutes between "I wrote this code" and "I committed it" so that a human (you) actually looked at the outputs. The cost is trivially low; the rate of caught bugs is high. The reason to put them in `.tmp/` and delete them is that they're not *the* test — they're the proof that you noticed.

### What I have at the end of Phase 7

Three deliverables landed:

- `tools/log_prediction.py` — 220 lines. Daily logger with idempotent append + same-pass backfill + atomic parquet write.
- `tools/retrain_all.py` — 200 lines. Weekly retrainer with prev-rotation, fsync+os.replace atomic swap, `--ticker` and `--no-swap` debug flags.
- `workflows/weekly_retrain.md` — full SOP covering both jobs: how to invoke, expected outputs, cadence rationale, atomic-write discipline, rollback procedure.

Plus a Phase 5 bugfix in `api/main.py`'s `/history` serialization — NaT-derived NaN no longer leaks through as invalid JSON.

`tools/retrain_all.py` was deliberately *not* exercised end-to-end in this session. A real run trains a transformer for ~10 minutes of CPU; the swap path is simple enough that code review catches the failure modes; and the user is on a credit budget. The training side is covered by Phase 4's existing run; the swap side is covered by `--no-swap` (trains, doesn't swap) which the user can validate any time. When the first weekly run happens for real, the operational notes in `weekly_retrain.md` are the runbook.

The next phase (8) moves all of this from localhost to a real Linux VM with systemd timers, nginx, and HTTPS. Phase 7 was the last phase where "manual invocation in PowerShell" is acceptable; Phase 8 makes the system self-running on hardware the user doesn't have to touch.

---

## Chapter 8 — Phase 9: Writing the part recruiters actually read

> Skipped Phase 8 (deployment to a VM) by choice — scope discipline, not laziness. Jumped straight to the polish phase: the README, the architecture diagram, the honest-metrics table, and the blog-style writeup at `docs/journey.md`. This chapter is about how the *artifacts at the top of the repo* are themselves engineering deliverables, and how to write them so the next reader — a hiring manager skimming for 90 seconds — gets the right mental model of what was built.

### Why skipping Phase 8 was the right call

The original plan had nine phases; I shipped seven of them. Phase 8 (Contabo VM + nginx + systemd timers + Let's Encrypt) was deliberately skipped. A few reasons made this the right tradeoff in this context:

1. **The VM wasn't provisioned.** I don't have a running Contabo box. Provisioning one for a portfolio piece is a non-trivial commitment (a recurring cost, a domain renewal, a TLS certificate to babysit, an attack surface to keep patched). Skipping it isn't refusing the work — it's refusing to over-invest in a recruiter signal whose marginal value is "live URL" vs. "GitHub repo with screenshots."
2. **The system already self-runs on localhost.** Windows Task Scheduler reproduces the daily fetch / log / weekly retrain cadence that Phase 8's systemd timers would have replaced. The *automation story* is true today; only the *public reachability* part is missing.
3. **Phase 9's value is independent of Phase 8.** A clear README, an honest metrics table, and a blog-style writeup are what a hiring manager actually reads. They don't click through to live URLs unless the README has already convinced them to. Doing Phase 9 with no Phase 8 is strictly more recruiter-valuable than doing Phase 8 with no Phase 9.

The README documents the skipped phase under **Future work** with a one-line cost estimate, which is itself the right signal: "I scoped this consciously, I know what's missing, and I can tell you exactly what it would take to finish." That reads better than a deployed-but-undocumented project.

> **What this means: scope discipline as a CV signal**
> Hiring managers know what production looks like. A portfolio piece that *correctly identifies what's out of scope*, says so plainly, and explains why, demonstrates the same judgment as one that shipped everything. The opposite — a project that pretends to be complete when it isn't, or that ships everything because shipping everything was the goal — reads as either dishonest or as not-knowing-when-to-stop. Naming the scope is itself a deliverable.

### Anatomy of a recruiter-grade README

The Phase 9 spec asks for seven things in `README.md`. I wrote it around a stricter mental model — *what does someone who has 90 seconds need, in priority order?*

1. **One-line pitch with the noun first.** Not "I built a system that…" — "*A self-retraining web service that predicts direction and expected return for the next trading session of MSFT and SPX.*" The reader can stop after sentence one and have the right mental model.
2. **The "60 seconds" paragraph.** Two sentences: what this came from (a research notebook with a bug and stale data), what it became (a productionized service with a brutally honest metrics dashboard). The *frame* is "the CV story is the productionization, not the predictive accuracy" — said once, up front, before any reader has to figure it out.
3. **The architecture diagram, before any prose.** A Mermaid flowchart of yfinance → parquet → train/inference → API → UI, with the daily/weekly cron arrows on the right side. Mermaid renders inline on GitHub with no image hosting, no broken-link risk, and *no decoupling from the source* — when the architecture changes, the diagram changes in the same commit as the code.
4. **The honest holdout metrics table, with naive baselines in the same view.** Two tables (one per ticker), four columns each, the honesty-gate verdict in the rightmost column. The reader sees both the model accuracy and the always-up baseline without scrolling or clicking. A red ❌ on the gate column is not buried — it's the most prominent thing in the section.
5. **The "what this says" paragraph immediately after the tables.** Not a footnote. The interpretation is part of the result, and a sophisticated reader explicitly *expects* the project to interpret its own numbers. Hiding behind "you can see for yourself" doesn't fly when the numbers say "we tied the baseline."
6. **Run-it-locally instructions, copy-pastable.** PowerShell command blocks the reader can hit Cmd-K on. Includes the iCloud-venv gotcha because that's the *first* thing a Windows reader will trip on if they happen to also work in iCloud Drive.
7. **Future work as a numbered list, ordered by cost-to-impact.** Not as an apology — as a roadmap. "Public deployment was Phase 8 in the original plan, the runbook is in `PLAN.md`, executing it is left." That sentence is more credible than a deployed-but-fragile URL would be.
8. **A `License & disclaimer` paragraph.** One sentence each. The disclaimer ("informational only, not financial advice") is non-negotiable for anything that displays a stock prediction; the license is a single line that the reader doesn't have to read but its absence is conspicuous.

The README came out at ~250 lines. It's optimized for the reader who never scrolls past the architecture diagram, *and* for the reader who reads every word. Both are real readers. Most demos are written for the first kind and lose the second.

> **What this means: writing for the 90-second reader and the 90-minute reader at the same time**
> Asymmetric audiences. The 90-second reader scans headers, looks at diagrams and tables, and forms a verdict from the structure. The 90-minute reader reads every word and forms a verdict from the substance. A good README serves both: clear structure for the skimmer, honest detail for the deep reader. The trap is optimizing only for the skimmer (bullet points, hero shots, no substance) — that reads as a pitch deck, not as engineering.

### Mermaid for architecture diagrams (not Excalidraw, not draw.io)

The Phase 9 spec mentioned "Mermaid is fine — no image hosting." I want to be more emphatic: **Mermaid is the right choice**, and the no-image-hosting reason isn't the main one.

The main reason is **lock-step versioning**. A `mermaid` code block is text inside `README.md`. When the architecture changes, the diagram changes in the same commit, reviewed in the same PR, by the same person. There's no "is the diagram stale?" question because the diagram lives inside the file the reader is already reading.

Compare: an Excalidraw export is a PNG. The source `.excalidraw` file is JSON in a different folder. Changing the architecture means re-opening Excalidraw, dragging boxes, exporting, replacing the PNG, committing both files, and hoping the next person knows the source exists. Half the time the source has been deleted; half the time the PNG was hand-edited *after* export and the source no longer matches. Diagrams rot.

For a small system like this one (10 boxes, 10 arrows), Mermaid's syntax is simpler than a drawing tool's UI. For systems bigger than that, the diagram probably needed to be three diagrams anyway.

> **What this means: text-as-source for diagrams**
> Same principle as infrastructure-as-code. If your source-of-truth for a diagram is a binary file, the diagram will drift away from the system within months. If your source is a `mermaid` (or `dot`, or `plantuml`) code block in the README, the diagram drifts at the same rate as the code that produced it — i.e., never. The tooling tradeoff is "easy to draw arbitrary shapes" vs "versioned in the same commit as the code"; for system diagrams, lock-step versioning wins every time.

### Putting the honesty gate in the README, not just the UI

Phase 6 put the red FAIL banner in the dashboard. Phase 9 had to decide whether to *also* put it in the README, or whether the README could keep the polite framing ("the model achieves 52% direction accuracy on a 252-day holdout").

The same logic that put the banner in the UI applies twice as strongly to the README:

- **The README is the most-read artifact in the repo.** Burying the verdict here would mean burying it for the readers most likely to form a hiring opinion.
- **The reader will compute the comparison anyway.** Anyone who knows ML will read "52% direction accuracy" and ask "what does the naive baseline get?" If the README doesn't answer that question on the same page, the reader either Googles the answer (badly — naive baselines are dataset-specific) or assumes the project is hiding it. Both outcomes are worse than just printing the baseline.
- **The strong negative is the differentiator.** Most ML demos report a number that beats a baseline they don't show. A demo that puts both numbers in the same row and admits the gap is zero is, paradoxically, the most credible-sounding one in the stack. The reader updates: "this person knows what honesty looks like in this field."

So the README has the same two tables the dashboard does, with the same honesty-gate verdict in the rightmost column, with the same red ❌. The framing immediately after is exactly the framing in the UI: *neither winner beats the always-up baseline; the return-regression head does real work; the directional head is the failure mode; run-to-run variance dominates any claimed sub-3pp edge.*

A version of me wanting to embellish would have softened that. A version of me wanting to over-correct would have led with it as if the failure were the whole project. The version that's in the README leads with what was built, names the failure as a part of the result, and lets the reader judge.

> **What this means: the same negative result, said twice, is more credible than once**
> A claim that appears only in the dashboard can be missed; a claim that appears in the README *and* the dashboard *and* the journey writeup can't be dismissed as something the reader stumbled on. Repeated honesty is structural — it tells the reader the project was designed for them to see it, not by accident.

### The journey writeup as the recruiter pitch

`docs/journey.md` is a ~1,200-word blog-style essay. It exists for one specific reader: a recruiter (or hiring manager) who has read the README, decided the project is worth more than 30 seconds, and clicked through to the linked writeup. They have maybe 90 more seconds.

What that reader needs, in this order:

1. The starting point (one paragraph — "a notebook with stale data and a denormalization bug").
2. The decisions made *before* writing code (one paragraph — drop LSTM, dual output, XGB baseline, fix the leakage, multi-ticker from day one, no Docker).
3. The bake-off and the honest negative ("neither winner beats naive; the AUC=0.56 result was a lucky seed").
4. The shipping decision ("ship anyway, make the failure visible, that's the story").
5. The system as a paragraph (one paragraph of nouns: yfinance, parquet, daily cron, weekly cron, FastAPI, mtime hot reload, static dashboard, prediction log).
6. The deliberate minimalisms (no Docker, no React, no Chart.js — each a tradeoff, not laziness).
7. The "what I'd do differently" — same content as the README's Future work but written as forward motion, not as a list of bugs.
8. The closing sentence the writeup is *built around*: *"in research, the model is the product; in production, the model is one ingredient. The other 90% is the part that decides whether anything you trained is going to keep working when you stop watching it."*

The essay's structure is the structure of the project. It's not a press release. It's the kind of writeup a sympathetic peer would write about my work after spending a day with the repo. That voice — "here's what was done, here's what was learned, here's what was honest about it" — is the voice the writeup is in.

> **What this means: a portfolio piece needs a portfolio piece**
> The repo is the work; the writeup is the *frame* the work is shown in. A great repo with a bad writeup gets read like a bad repo. A pretty good repo with a great writeup gets read like a great repo. Both pieces deserve the same care; spending zero time on the writeup leaves the repo to fend for itself, which it cannot, because nobody reads source code without a reason to.

### What didn't ship in Phase 9

A few honest gaps:

- **No screenshot.** The README has a commented-out placeholder for `docs/screenshot.png`. To take it I'd need to launch the UI, navigate to both tabs, and screenshot — a manual five-minute task I deferred. The architecture diagram does most of the visual work in the meantime; a real screenshot would be a strict upgrade.
- **No GitHub repo link.** The README mentions `<your-github-username>/<repo-name>` in spirit but the repo isn't pushed yet (`PLAN.md` line 5 still flags this as a User TODO). The clone-instructions section currently lists local-only setup; the moment the push happens, the README needs a `git clone <url>` line at the top of the setup block.
- **No deployed URL.** Phase 8 was deliberately skipped; the README documents this honestly under Future work.
- **No social / OG meta tags or favicon on `index.html`.** Phase 6's frontend doesn't ship these. They're cheap and worth adding if the project ever does go up at a public URL; while it's localhost-only they would have been dead pixels.

The right move on each is "documented as future work, ordered by cost-to-impact." Adding them would have been hours of work for a recruiter signal I'm willing to defer.

### What I have at the end of Phase 9

```
README.md          (new, ~250 lines)
docs/journey.md    (new, ~1,270 words)
LEARNING_NOTES.md  (this chapter + glossary entries)
PLAN.md            (Phase 9 completion log, Phase 8 explicitly marked skipped)
```

Nothing in `src/`, `tools/`, `api/`, or `workflows/` changed. Phase 9 is pure documentation — and that's the right kind of phase to be pure documentation, because anything you'd change in the code after writing the README is something you'd have to rewrite in the README. Doing them in the right order means the README is the *truthful* state of the system at the moment the project freezes.

The next thing isn't a phase. It's three small acts: push the repo to GitHub, take a screenshot of the running UI for the README placeholder, and (optionally, later, if the credit/time budget allows) come back to Phase 8 and make the localhost system reachable at a public URL. Each is a runbook task. None changes the substance of what's been built.

---

## What's coming next (preview)

| What | Why | Out of scope, but the runbook exists |
|---|---|---|
| **Phase 8 — VM deployment** | Public URL, systemd timers, HTTPS | [PLAN.md](PLAN.md#phase-8) carries the full spec; skipped in v1 by scope choice |
| Drift detection on the prediction log | Surface accuracy regressions, weekly | Building blocks exist in `data/predictions_*.parquet` from Phase 7 |
| Stronger features (sentiment, options flow) | The current 15-feature set provably doesn't carry 1-day directional signal | The honest-negative result is what motivates this direction; pursuing it would be a separate project |

---

## Glossary (added to as new terms come up)

- **WAT framework** — Workflows (markdown SOPs) + Agents (orchestrator) + Tools (deterministic scripts). A separation-of-concerns pattern that keeps probabilistic code (the AI/me) in coordination and deterministic code (Python) in execution.
- **Holdout set** — Data the model has never seen during training *or* validation. The only honest measure of how well it'll do on tomorrow's data. We carve off the last 252 trading days.
- **Bake-off** — Train multiple candidate models on the same split, evaluate on the same holdout, ship the winner. No vibes-based architecture choices.
- **Data leakage** — Any way information about the test set affects training, including via preprocessing parameters (means, stds, feature selection). Inflates holdout metrics, surprises you in production.
- **Failing loudly** — Crashing immediately with a clear error when something unexpected happens, instead of silently returning `None` or default values. Production systems must fail loudly or you'll never know they're broken.
- **Cargo-culting** — Adopting a tool because successful projects use it, not because it solves a problem you actually have. (Common example: Docker for one-server projects.)
- **Baseline commit** — The first git commit in a repo. Contains only the empty structure and contract files (requirements, config, .gitignore). Lets you `git reset` back to "just the skeleton" if needed.
- **Virtual environment (venv)** — A self-contained Python installation in a folder. Each project gets its own; they don't see each other's dependencies. Put them outside cloud-synced folders.
- **Single source of truth** — One file (here: `src/config.py`) that every other piece of code reads from for shared constants. Adding a ticker is a one-line edit, not a 14-place rename.
- **Idempotent** — A function/script that produces the same end-state no matter how many times you run it. Daily data refresh is idempotent: re-running it after it already ran today should change nothing. Critical for cron jobs that might overlap or retry.
- **Parquet** — Binary, columnar tabular file format. Faster to read, smaller on disk, and type-aware (unlike CSV). What we use to cache market data on disk.
- **Fit / transform** — A scikit-learn-style separation: `fit()` learns parameters from data; `transform()` applies them. The two phases must be split by your train/test boundary, or the test set leaks into training.
- **STL decomposition** — Seasonal-Trend decomposition using LOESS. Splits a time series into trend + seasonal + residual components. Often used to flag "outliers" in the residual; in financial returns this is dangerous because the outliers are usually the events you want the model to learn from.
- **Sliding window / sequence construction** — Turning a flat 2D time series (rows × features) into the 3D `(num_windows, seq_len, features)` tensor a sequence model expects. Each window pairs with the target at its terminal row.
- **Lazy import** — Importing a heavy library inside the function that needs it instead of at module top. Trades "all imports validated at startup" for "the module is cheap to import even if you never call the heavy function." Useful when memory is tight or when an import has expensive side effects.
- **Fat tails** — Real-world return distributions have many more extreme days than a normal distribution would predict. Mechanical 3σ outlier filtering on financial data quietly discards the most informative days.
- **Model factory** — A function that returns a *new* model instance with a fixed architecture, parameterized by hparams. Distinct from "the model" — used by retrains, tests, and any code that needs to rebuild the network from scratch.
- **Dual-head model** — One neural network with a shared backbone and two (or more) prediction heads producing different targets. Cheaper inference than two separate models, and shared layers learn from both losses' gradients. Loss weighting matters: heads at very different scales can starve each other.
- **XGBoost** — Gradient-boosted decision trees. Builds an ensemble of shallow trees where each new tree predicts the previous ensemble's residual errors. Hard to beat on small/medium tabular data with engineered features.
- **Param/sample ratio** — Number of model parameters divided by number of training examples. Rough overfitting risk indicator. ~10 with regularization is fine; ~1000 needs much stronger regularization or much more data.
- **Leaky abstraction** — When a high-level abstraction's implementation details bleed through in surprising places (here: Keras printing a `(1, 60, 15)` shape because of how `tf.expand_dims` interacts with shape inference). Best response is to understand the leak rather than paper over it.
- **Naive baseline** — A trivially-simple predictor whose accuracy is the *floor* your model must clear to be worth shipping. Examples: always-predict-majority-class, predict-yesterday-equals-tomorrow. Without one, "53% accuracy" sounds impressive; with one, you find out the always-up baseline was 52% and you have no signal.
- **Honesty gate** — A pre-declared evaluation rule that decides whether the model has earned production status. Here: must beat naive_always_up by ≥3pp on holdout direction accuracy. Recorded in `bakeoff_<ticker>.json`. Designed *before* seeing the numbers so you can't move the goalposts after.
- **Noise floor** — The minimum performance difference detectable on a given evaluation. Driven by holdout size, label noise, and training-seed stochasticity. On this project: ~5pp AUC swings just from re-seeding TF, so any claimed "2-3pp edge" on direction is indistinguishable from random. Real signal has to clear the noise floor consistently across re-runs.
- **Youden's J / threshold calibration** — Picking the binary-classification cutoff (not just 0.5) by maximizing J = TPR − FPR on val. Helps when the model's probabilities are well-ranked (high AUC) but biased away from 0.5. Free win on real signal; can hurt by 1-2pp when signal is weak and val→test threshold-optima don't transfer.
- **Residual / skip connection** — `x_out = x_in + Sublayer(x_in)`. Gives gradients a non-multiplicative path through deep networks (so they don't vanish) and lets the identity be a valid solution (so a useless block can be ignored). Essential past ~5 layers. Doesn't help — and on this dataset, slightly hurt — when the problem is "no signal," not "deep network won't train."
- **Pre-LN vs post-LN** — Where LayerNormalization goes inside a Transformer block. Post-LN: `LN(x + Sublayer(x))` (2017 original). Pre-LN: `x + Sublayer(LN(x))` (GPT-2 onward). Pre-LN is more stable in deep stacks; the difference is invisible for 3 shallow blocks.
- **Weight decay / AdamW** — L2 regularization that pulls weights toward zero on every optimizer step. In AdamW it's *decoupled* from the gradient (mathematically different from "L2 in the loss"). Helpful for over-parameterized models that overfit; *unhelpful* for models that haven't learned anything yet — wd then accelerates collapse to "output zero."
- **Local optimum (in training)** — A weight configuration where the loss can't go down by small local perturbations, but isn't globally optimal. Majority-class collapse (predict-up-always; predict-0.5-always) are common ones for classification. Once stuck, the model stays there regardless of remaining epochs — the data doesn't tell it where else to go.
- **Run-to-run variance** — Different training seeds (or different parallelism orderings, even with the same seed) producing meaningfully different holdout scores. The noise floor for "is X better than Y" judgments — if the within-config variance is bigger than the cross-config difference, you can't tell them apart from one run each.
- **Atomic artifact swap** — Replacing a file (e.g. the production model) in a way that any process reading it sees either the old version or the new, never a half-written intermediate. On Linux, `os.replace` is atomic. `shutil.copy2` is *not* atomic (multi-step write), so for the swap point use `os.replace` after a temp-file write.
- **Thin API layer** — The HTTP layer's job is adaptation (URL → function call, return value → JSON), not business logic. Prediction logic lives in plain functions that can be called by tests, CLIs, or cron without going through HTTP. If you can swap FastAPI for Flask by editing one file, the boundary is drawn right.
- **Lifespan loader** — FastAPI's modern startup/shutdown hook (an async generator that yields between setup and teardown). Use it to load expensive resources once (model files, DB pools) and stash them in `app.state` so request handlers reuse them. Replaces the deprecated `@app.on_event("startup")` pattern.
- **mtime-based hot reload** — Cheap polling pattern: each request `stat()`s the artifact and reloads only if the on-disk mtime exceeds the in-memory snapshot. Works for single-writer / single-reader setups. Cost on the happy path is one syscall; cost on the rare path is a full reload.
- **Keyword-only argument (`*` in the signature)** — Forces callers to pass an argument by name. Self-documenting for boolean flags and any parameter where the *name* carries more meaning than the *position*. Default for new flags; default-positional makes call sites cryptic six months later.
- **response_model vs raw dict (FastAPI)** — `response_model=` generates an OpenAPI schema and *strips* fields not in the model (silently). The strip is a footgun if you add server fields and forget to update the schema. For internal APIs with one consumer, returning a raw dict is often more honest.
- **Liveness probe specificity** — An automated "is the server up?" check is only as strong as the endpoint it queries. Generic paths (`/`, `/metadata`) often exist on multiple services; if you have two servers on the same machine, a probe against a non-specific path can talk to the wrong one. Probe an endpoint unique to *your* app (e.g. `/predict/MSFT`).
- **`run_in_background` and silent failures** — A background process that exits non-zero doesn't notify you; the failure lives in a log file you have to fetch and read. When something seems wrong after starting a background task, always check the output file before re-trying.
- **Cargo-culting (frontend edition)** — Pulling in React/Tailwind/Vite by reflex when the page has four elements. Frameworks earn their weight on apps with shared state across hundreds of components; on a static dashboard they add 40 KB of runtime, a build step, and config files that someone has to maintain. Pick the abstraction that fits the surface, not the abstraction you've seen on Twitter.
- **Honest-negative UX** — A user-facing surface that proactively discloses the model's failure modes (here: a red FAIL banner when the honesty gate doesn't pass) instead of hiding them. Counter-intuitively *strengthens* a portfolio piece: a sophisticated reader trusts a demo that names its limitations more than one that pretends not to have any.
- **Inline SVG vs canvas vs charting library** — Three levels of abstraction for charts. SVG = declarative DOM, cheap to hand-roll for static charts under ~5 series. Canvas = imperative pixel buffer, better for thousands of points or animation. Charting library = sensible defaults on top, costs a dependency and tens of KB. Match the tool to the chart complexity.
- **`Promise.all` + `flatMap`** — The standard recipe for "kick off N×M independent fetches and resume when they're all in." `flatMap(fn)` produces multiple outputs per input and flattens them; `Promise.all([...])` waits for all to resolve concurrently. Tab-switching feels instant when every tab's data was preloaded this way.
- **`textContent` vs `innerHTML`** — `textContent` sets text and HTML-escapes everything (safe). `innerHTML` parses the string as HTML (unsafe for any user-influenced value). The rule: server strings and URL parameters always go through `textContent`; `innerHTML` only takes strings you built character-by-character from values you control. Mixing them up is the source of approximately every XSS vulnerability.
- **Empty-state design** — Treating "no data yet" as a designed state, not a bug. A good empty state explains *why* it's empty (Phase 7 hasn't run yet), *when it'll fill up*, and *what to do* (nothing). Most UIs design for the populated case and improvise the empty one; flipping that order makes the product feel finished from day one.
- **Thin API + decoupled UI** — When a frontend ships without requiring backend changes, the API boundary is drawn right. Phase 6's frontend dropped into Phase 5's existing `StaticFiles` mount with zero `api/main.py` edits. The payoff is reversibility: swap React for vanilla JS for a static-site generator without touching the API.
- **Blast radius (of a scheduled job)** — The scope of damage one failed run can do. A logger has small blast radius (one missing parquet row); a retrainer has medium (every prediction wrong until rollback). Splitting jobs by blast radius keeps operational reasoning tractable: each failure mode has its own alert path and recovery procedure.
- **Idempotency in batch jobs** — Running the job twice produces the same end state as running it once. Achieved by picking a dedup key that uniquely identifies a *logical* run (`predicted_for_date`, not `predicted_at`). Re-runs from a retry queue or a scheduler-restart are then harmless.
- **Backfill anchored on data lineage, not heuristics** — When logging a forecast for later verification, key the backfill lookup off *the data the model actually saw* (`last_bar_date_at_prediction`), not off a guessed verification date (`predicted_for_date` from a `BDay(1)` heuristic). Calendar heuristics drift around holidays; data lineage doesn't.
- **Piggyback writes** — Doing additional bookkeeping (backfill) during a write a process is already making (append) is cheaper than a separate sweep job for single-process systems. Separate ETL passes pay off at scale (different SLAs, different failure isolation); piggyback wins when one process owns the file.
- **Atomic file write pattern** — Write to `dst.tmp`, then `os.replace(dst.tmp, dst)`. Atomic on the same filesystem on both POSIX and Windows since Python 3.3. Readers never see a half-written file; a crash mid-write leaves only the tmp. Costs one rename per write; buys crash-consistency for free.
- **`fsync` before `os.replace`** — `os.replace` flips the directory entry atomically, but the *bytes* of the new file might still be in the OS page cache. A power loss in that window leaves you pointing at unflushed data. `fsync` on the tmp file forces the bytes down before the rename. One syscall, closes the window.
- **Model identity vs model weights** — Two different things to retrain. Weights = the floats inside a chosen architecture, refit on new data; safe to automate. Identity = which architecture / hparams won the bake-off; should be human-supervised because the comparison frame is changing. Conflating them produces drift you can't reason about.
- **NaT vs NaN in pandas date columns** — `pd.NaT` is the datetime null sentinel; distinct from `float('nan')` even though they print the same. `.dt.strftime` on a NaT-containing column produces `NaN` floats in the resulting string Series, which `df.where(df.notnull(), None)` silently fails to convert. Robust replacement: `df.astype(object).where(pd.notna(df), None)` *plus* a per-cell `isinstance(v, float) and pd.isna(v)` scrub.
- **Throwaway smoke tests as a discipline** — A 15-line script in `.tmp/` that feeds the code real-shaped inputs and inspects the outputs, deleted when done. Not a unit test, no CI, no name. Exists for the five minutes between "wrote code" and "committed code" so a human actually looked at outputs. High catch rate per minute of investment.
- **Scope discipline as a CV signal** — A portfolio piece that correctly *names what's out of scope*, says so plainly, and explains why, demonstrates the same judgment as one that shipped everything. The opposite (pretending to be complete when it isn't, or doing-it-all because doing-it-all was the goal) reads as either dishonest or as not knowing when to stop. The "Future work" list is structurally part of the deliverable, not an apology.
- **Writing for the 90-second reader and the 90-minute reader at the same time** — Asymmetric audiences. The skimmer scans headers, diagrams, and tables; the deep reader reads every word. A good README serves both: clear structure for the skimmer, honest detail for the deep reader. Optimizing only for the skimmer (bullets, hero shots, no substance) reads as a pitch deck, not engineering.
- **Text-as-source for diagrams (Mermaid / PlantUML / Graphviz)** — A `mermaid` code block in the README versions in lock-step with the code that produced it. A binary diagram file (Excalidraw PNG, draw.io export) drifts away from the system within months because the source-of-truth lives somewhere most editors won't touch. For system diagrams of any size you can fit in your head, text-source wins; the "easy to draw arbitrary shapes" benefit of GUI tools rarely pays for itself.
- **The same negative result, said twice, is more credible than once** — A failure named only in the dashboard can be missed; a failure named in the README *and* the dashboard *and* the writeup is structural. Repeated honesty tells the reader the result was designed for them to see, not stumbled on. The reflex to soften the second appearance ("but actually…") destroys the structural property.
- **A portfolio piece needs a portfolio piece (the writeup)** — The repo is the work; the writeup (`docs/journey.md`, blog post, talk) is the frame the work is shown in. A great repo with a bad writeup gets read like a bad repo, because nobody reads source code without a reason to. The writeup *creates* the reason; underinvesting in it leaves the repo to fend for itself.

---

*This file grows by one chapter per phase. Last updated: end of Phase 6 (2026-05-11).*
