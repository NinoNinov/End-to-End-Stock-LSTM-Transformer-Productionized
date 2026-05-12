# From research notebook to a (locally) self-running service

This is the story of turning a stock-prediction Jupyter notebook into a live, self-retraining web service — and discovering, in the process, that the most interesting thing about the result is what it admits it cannot do.

## The starting point

I had a notebook from a research project: `Deep Learning for Stock Price Prediction with LSTM and Transformers.ipynb`. It compared two architectures on Microsoft's daily closing price, reported test accuracy in the low fifties, and produced some pretty charts. Useful enough as a study — but in the way these notebooks always are. Training data ended December 2024 and was over a year stale by the time I picked it back up. There was a denormalization bug at notebook lines 1634–1636 that made one of the plots subtly wrong. There was no way to predict on *today's* data — the only "inference" path was re-evaluating the same train/test split it had carved off during training. And there was no automation — every refresh would be me, manually re-running cells.

This is a normal state for a research notebook to be in. *None of these issues matter for studying the architectures.* All of them matter the moment you want a live product. The point of this project was to do the second thing.

## The contract before the code

Before writing any production code I wrote a ~700-line plan document. It's still in the repo as [`PLAN.md`](../PLAN.md). The phases were sequential and self-contained — partly because I was on a credit budget for AI assistance and needed each phase to resume cleanly in a cold session, and partly because I genuinely didn't want to relitigate decisions halfway through. Six key calls were made up-front, and almost none of them moved during execution:

1. **Drop the LSTM** from the live product. Keep one model family for simpler retrain/monitor/explain. LSTM stays in `research/` as the scientific artifact.
2. **Predict return percentage, not absolute price.** Returns are scale-clean — they sidestep the denormalization bug entirely. The dashboard shows an *implied close* (`last_close × (1 + predicted_return)`) but that's display arithmetic, not learned output.
3. **Add an XGBoost baseline alongside the Transformer**, and run a real bake-off. On ~8,800 daily samples of tabular data, XGBoost often beats deep models. If it did here, shipping XGBoost would be the more honest answer.
4. **Fix the data leakage** in the normalizer. The notebook fit z-score on the *full* dataset, then split into train/test. The test set leaks into training through the normalizer's parameters. The fix is one keyword change but it materially affects the holdout numbers.
5. **Multi-ticker from day one** via config blocks. Adding SPX later would have meant a refactor; designing it for N tickers up front meant adding one cost ~1 line.
6. **No Docker for v1.** Plain venv + systemd + nginx is simpler to debug on a one-server portfolio project. Containers would be cargo-culting.

## The bake-off

Phase 4 trained both architectures, on both tickers, with two STL outlier-removal settings A/B'd for the transformer — six runs in total. The same train/val/test split for every contender, the same 252-day holdout window, the same naive baselines computed alongside. A **pre-declared honesty gate** said: the production winner must beat naive_always_up by ≥3 percentage points to count as "real signal."

Neither winner did. On both tickers, the production transformer tied naive_always_up exactly. The bake-off file records this verdict with `passes: false, delta_pp: 0.0`.

I tried to fix it. Two transformer rearchitectures (pre-LN + residual + AdamW, then just pre-LN + residual) — both worse. The lesson, recorded in [LEARNING_NOTES.md](../LEARNING_NOTES.md), is that **training-stochasticity variance dominates the signal at this horizon**. Across six runs of "the same" config the holdout AUC ranged from 0.43 to 0.56 just from re-seeding TensorFlow. The "AUC=0.56 result" that motivated all of this was a lucky seed, not a real edge.

So the question changed. The question wasn't "how do I make the model good?" — it was "what do I do when I can prove the model isn't good?"

## Shipping the honest negative

The answer this project picked: **ship the system anyway**, and make the failure visible. The dashboard renders a red **FAIL** banner under the holdout metrics on every ticker tab. It reads, in plain language: *the product ships because the infrastructure is the deliverable; the model has no extractable next-day directional edge on this feature set.* Both holdout accuracy and the naive baseline are shown side-by-side, in the same font size, with their delta. The user can verify the gap is zero without reading any text.

The reflex would be to bury that — show the accuracy as a big green number, hope nobody notices the baseline. Every other portfolio ML demo I've looked at does exactly this. The honest version is more credible. A sophisticated reader walks away with the correct mental model: this is a working productionization pipeline, the predictive signal isn't there yet, and the system would *immediately surface* any future feature engineering that did move the gate. That's a stronger CV story than a fake green number is.

## What the productionization actually does

Stripped to one paragraph: yfinance feeds a daily parquet cache, daily-cron jobs append a logged prediction with a same-pass backfill of yesterday's outcome, a weekly cron refits the production weights and atomically swaps the artifact via `tmp + fsync + os.replace`, and a FastAPI server reads everything with an `mtime`-based hot reload so a Sunday retrain becomes visible without a server restart. There's a static HTML dashboard with two ticker tabs, a hand-rolled SVG cumulative-correctness chart, an empty-state design for the history table that gracefully transitions to populated as logs accumulate, and a single-file no-framework frontend that gzips under 5 KB.

The deliberate minimalisms — no Docker, no React, no Chart.js CDN, no auth, no monitoring stack — aren't laziness. Each was a documented tradeoff between "what production stacks look like" and "what *this* production stack needs." A one-server, two-ticker, no-traffic portfolio piece doesn't need Kubernetes. Choosing the right level of abstraction for the size of the problem is itself a skill the project is trying to demonstrate.

## What I'd do differently with more time

In order of cost-to-impact: deploy to a VM with HTTPS and systemd timers (the runbook exists, only execution is left); add sentiment and options-flow features to give the model something the OHLCV-only set doesn't have; broaden the prediction horizon (5-day is hugely easier than 1-day); wire a drift-detection job on the now-accumulating prediction log; replace the point probability with a calibrated credible interval. Ensembling Transformer + XGB is the cheapest possible gain; Docker and CI are the cheapest possible polish.

## What the project taught me

The shortest version: in research, the model *is* the product; in production, the model is one ingredient. The other 90% — the data refresh, the inference path, the holdout discipline, the API, the frontend, the retrain loop, the prediction log, the runbooks — is *also* engineering, and it's the part that decides whether anything you trained is going to keep working when you stop watching it. Hiring managers see hundreds of "trained an LSTM, got 53% test accuracy" notebooks. Almost none of them turn into a live URL where the model actually predicts tomorrow, every day, by itself, and tells the user honestly how much to trust the answer.

That's the project. It's not impressive because the model is good. It's impressive because the productionization is real, the failures are visible, and the system would have caught any of them — if there had been any to catch.
