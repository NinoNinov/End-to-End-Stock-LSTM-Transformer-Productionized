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

## What's coming next (preview, will fill in as we go)

| Phase | What gets built | New concepts I'll need to learn |
|---|---|---|
| **2 — Data layer** | yfinance fetching, parquet caching, feature engineering ported from the notebook, train/val/test split with proper holdout, normalizer (fit on train only) | Parquet format, idempotent caches, normalizer serialization, sequence construction for time-series models |
| **3 — Models** | Encoder-only Transformer with **dual output heads** (direction + return), XGBoost baseline | Dual-output Keras models, why XGB needs two separate models for two targets |
| **4 — Training & bake-off** | Real training runs, holdout evaluation, "honesty gate" (don't ship if you don't beat the naive baseline) | Naive baselines, atomic artifact swaps, model metadata files |
| **5 — FastAPI service** | HTTP endpoints serving live predictions | FastAPI lifespan loaders, hot-reload via mtime checks, Pydantic schemas |
| **6 — Frontend** | Vanilla JS single-page UI showing both tickers, prediction history | No-framework JS as a deliberate choice |
| **7 — Automation** | Daily prediction logging, weekly retraining (still on localhost) | Atomic file swaps, rollback artifacts, cron/Task Scheduler basics |
| **8 — Deployment** | Move everything to the Contabo Linux VM, with HTTPS and self-running timers | systemd units, systemd timers (vs cron), nginx reverse proxy, certbot/Let's Encrypt |
| **9 — Polish for CV** | README, architecture diagram, honest metrics table, journey blog post | What recruiters actually read; how to present "honest negative results" |

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

---

*This file grows by one chapter per phase. Last updated: end of Phase 1 (2026-05-10).*
