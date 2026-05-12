"""Training orchestration: fit a model, evaluate on holdout, write artifacts.

Implements PLAN.md Phase 4:
- ``train_ticker(ticker_key, model_kind, *, apply_stl=None, artifact_suffix="")``
  trains a single (ticker, model_kind), evaluates on the 252-day holdout,
  computes naive baselines, and writes
  ``artifacts/{ticker_key}_{model_kind}{suffix}.{keras|joblib}`` + a
  ``..._metadata.json`` sibling carrying everything Phase 5 inference needs
  (normalizer, feature order, seq_len, holdout metrics, hparams, versions).
- ``run_bakeoff(ticker_key)`` reads the per-model metadata, picks a winner
  (higher direction_accuracy; tie → lower return_mae), copies it to
  ``artifacts/{ticker_key}_production.{keras|joblib}`` + metadata, evaluates
  the honesty gate (winner must beat naive_always_up by ≥3pp), and writes
  ``artifacts/bakeoff_{ticker_key}.json``.

The CLI (``tools.train_model``) wires both together, including the
transformer STL outlier-removal A/B that runs both ``apply_stl_outlier_removal``
settings and lets the higher-accuracy variant represent "transformer" in the
bake-off.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, TICKERS
from src.data import (
    apply_normalizer,
    engineer_features,
    fit_normalizer,
    make_sequences,
    split_train_val_test,
    stl_outlier_mask,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ---- internals ------------------------------------------------------------


def _set_seeds() -> None:
    """Seed numpy + python hash. TF seed is set lazily (see _set_tf_seed)."""
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def _set_tf_seed() -> None:
    import tensorflow as tf

    tf.random.set_seed(RANDOM_STATE)


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("sklearn", "xgboost", "tensorflow", "numpy", "pandas"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return out


def _drop_stl_outliers(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Drop STL outlier rows from train+val (test stays untouched).

    The STL fit sees train+val combined so it has enough context for a stable
    decomposition; the test split is never inspected here, which keeps the
    holdout honest.
    """
    combined = pd.concat([train, val])
    mask = stl_outlier_mask(combined)
    n_dropped = int(mask.sum())
    train_mask = mask.reindex(train.index, fill_value=False)
    val_mask = mask.reindex(val.index, fill_value=False)
    return train.loc[~train_mask], val.loc[~val_mask], n_dropped


def _calibrate_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability threshold that maximises Youden's J on val.

    Youden's J = TPR − FPR; equivalent to the threshold that maximises
    *balanced* accuracy. Persisted in metadata so Phase 5 inference uses the
    same threshold the training run used to compute holdout direction_accuracy
    — important when AUC > 0.5 but the natural model output is biased
    (e.g. tilted toward the majority class).
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    # Edge case: all probs identical → roc_curve returns trivial curve; fall back to 0.5.
    if np.allclose(y_prob, y_prob[0]):
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    optimal = thresholds[int(np.argmax(j))]
    # roc_curve can return inf for the first threshold (sklearn quirk); clamp into [0,1].
    if not np.isfinite(optimal):
        optimal = 0.5
    return float(np.clip(optimal, 0.01, 0.99))


def _holdout_metrics(
    y_dir_true: np.ndarray,
    y_dir_prob: np.ndarray,
    y_ret_true: np.ndarray,
    y_ret_pred: np.ndarray,
    *,
    naive_up_acc: float,
    naive_yest_rmse: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    y_dir_prob = np.asarray(y_dir_prob, dtype="float64")
    y_dir_class = (y_dir_prob > threshold).astype(int)
    y_dir_true = np.asarray(y_dir_true).astype(int)
    y_ret_pred = np.asarray(y_ret_pred, dtype="float64")
    y_ret_true = np.asarray(y_ret_true, dtype="float64")
    # Direction accuracy at 0.5 is reported alongside the calibrated metric so
    # the bake-off log shows whether the calibration actually helped.
    y_dir_class_default = (y_dir_prob > 0.5).astype(int)
    return {
        "direction_accuracy": float(accuracy_score(y_dir_true, y_dir_class)),
        "direction_accuracy_at_0_5": float(
            accuracy_score(y_dir_true, y_dir_class_default)
        ),
        "direction_threshold": float(threshold),
        "direction_logloss": float(
            log_loss(y_dir_true, np.clip(y_dir_prob, 1e-7, 1 - 1e-7))
        ),
        "direction_auc": float(roc_auc_score(y_dir_true, y_dir_prob)),
        "return_rmse": float(np.sqrt(np.mean((y_ret_pred - y_ret_true) ** 2))),
        "return_mae": float(np.mean(np.abs(y_ret_pred - y_ret_true))),
        "naive_always_up_accuracy": float(naive_up_acc),
        "naive_yesterday_return_rmse": float(naive_yest_rmse),
    }


def _naive_baselines(
    val: pd.DataFrame, test: pd.DataFrame
) -> tuple[float, float]:
    """Compute the two naive baselines on the (un-normalized) test targets.

    - always-up: predict direction=1 every day → accuracy is the up-rate.
    - yesterday-return RW: predict tomorrow's return = today's actual return.
      Today's return at row t equals ``target_return[t-1]``. We pull the last
      val row's target_return as the predictor for the first test row, so the
      baseline has all 252 predictions (no NaN drop).
    """
    naive_up = float((test["target_direction"] == 1).mean())
    yest = (
        pd.concat([val["target_return"].tail(1), test["target_return"]])
        .shift(1)
        .iloc[1:]
        .to_numpy(dtype="float64")
    )
    actual = test["target_return"].to_numpy(dtype="float64")
    naive_yest_rmse = float(np.sqrt(np.mean((yest - actual) ** 2)))
    return naive_up, naive_yest_rmse


# ---- public ---------------------------------------------------------------


def train_ticker(
    ticker_key: str,
    model_kind: Literal["transformer", "xgb"],
    *,
    apply_stl: bool | None = None,
    artifact_suffix: str = "",
) -> dict[str, Any]:
    """Train one (ticker, model_kind) combination end-to-end.

    Parameters
    ----------
    ticker_key
        Key into ``src.config.TICKERS`` (e.g. ``"MSFT"``, ``"SPX"``).
    model_kind
        ``"transformer"`` or ``"xgb"``.
    apply_stl
        ``None`` → use the per-ticker config default. ``True`` / ``False`` to
        override (used for the STL A/B in Phase 4).
    artifact_suffix
        Appended to the artifact filename, before the extension. Used by the
        STL A/B (``"_stl0"``, ``"_stl1"``) so both variants can coexist on
        disk for the bake-off to inspect.

    Returns the metadata dict that was written to ``..._metadata.json``.
    """
    if ticker_key not in TICKERS:
        raise KeyError(
            f"Unknown ticker_key {ticker_key!r}; known: {list(TICKERS)}"
        )
    cfg = TICKERS[ticker_key]
    if apply_stl is None:
        apply_stl = bool(cfg.apply_stl_outlier_removal)

    _set_seeds()

    parquet_path = DATA_DIR / f"{ticker_key}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing {parquet_path}; run `python -m tools.fetch_market_data` first"
        )
    raw = pd.read_parquet(parquet_path)
    df = engineer_features(raw)

    train, val, test = split_train_val_test(df, cfg.holdout_days)

    n_outliers = 0
    if apply_stl:
        train, val, n_outliers = _drop_stl_outliers(train, val)
        logger.info(
            "[%s/%s] STL dropped %d outlier rows (train+val) before fit",
            ticker_key,
            model_kind,
            n_outliers,
        )

    norm = fit_normalizer(train, cfg.feature_columns)
    train_n = apply_normalizer(train, norm, cfg.feature_columns)
    val_n = apply_normalizer(val, norm, cfg.feature_columns)
    test_n = apply_normalizer(test, norm, cfg.feature_columns)

    naive_up, naive_yest_rmse = _naive_baselines(val, test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_base = f"{ticker_key}_{model_kind}{artifact_suffix}"

    epochs_run: int | None = None

    if model_kind == "transformer":
        _set_tf_seed()
        import tensorflow as tf  # noqa: F401  (warm imports for callbacks)
        from tensorflow.keras.callbacks import EarlyStopping

        from src.models.transformer import build_transformer

        X_train, y_ret_train, y_dir_train = make_sequences(
            train_n, cfg.feature_columns, cfg.seq_len
        )
        X_val, y_ret_val, y_dir_val = make_sequences(
            val_n, cfg.feature_columns, cfg.seq_len
        )

        ctx_n = cfg.seq_len - 1
        if len(val_n) < ctx_n:
            raise ValueError(
                f"val has only {len(val_n)} rows after STL; need >= {ctx_n} for test context"
            )
        test_with_ctx = pd.concat([val_n.iloc[-ctx_n:], test_n])
        X_test, y_ret_test, y_dir_test = make_sequences(
            test_with_ctx, cfg.feature_columns, cfg.seq_len
        )
        assert len(X_test) == len(test_n), (
            f"test sequence count {len(X_test)} != test rows {len(test_n)}"
        )

        hp = cfg.transformer_hparams
        model = build_transformer(cfg.seq_len, len(cfg.feature_columns), hp)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )
        ]
        history = model.fit(
            X_train,
            {"direction": y_dir_train, "return": y_ret_train},
            validation_data=(
                X_val,
                {"direction": y_dir_val, "return": y_ret_val},
            ),
            epochs=int(hp["epochs"]),
            batch_size=int(hp["batch_size"]),
            callbacks=callbacks,
            verbose=2,
        )
        epochs_run = len(history.history["loss"])

        val_preds = model.predict(X_val, verbose=0)
        val_dir_prob = np.asarray(val_preds["direction"]).reshape(-1)
        threshold = _calibrate_threshold(y_dir_val, val_dir_prob)

        preds = model.predict(X_test, verbose=0)
        y_dir_prob = np.asarray(preds["direction"]).reshape(-1)
        y_ret_pred = np.asarray(preds["return"]).reshape(-1)

        model_path = ARTIFACTS_DIR / f"{artifact_base}.keras"
        model.save(model_path)
        hparams_used: dict[str, Any] = dict(hp)

        y_dir_eval = y_dir_test
        y_ret_eval = y_ret_test
    elif model_kind == "xgb":
        from src.models.xgb import build_xgb_classifier, build_xgb_regressor

        X_train_2d = train_n[cfg.feature_columns].to_numpy(dtype="float32")
        y_dir_train = train["target_direction"].to_numpy(dtype="int8")
        y_ret_train = train["target_return"].to_numpy(dtype="float32")
        X_val_2d = val_n[cfg.feature_columns].to_numpy(dtype="float32")
        y_dir_val = val["target_direction"].to_numpy(dtype="int8")
        y_ret_val = val["target_return"].to_numpy(dtype="float32")
        X_test_2d = test_n[cfg.feature_columns].to_numpy(dtype="float32")

        clf = build_xgb_classifier(cfg.xgb_hparams)
        reg = build_xgb_regressor(cfg.xgb_hparams)
        clf.fit(
            X_train_2d,
            y_dir_train,
            eval_set=[(X_val_2d, y_dir_val)],
            verbose=False,
        )
        reg.fit(
            X_train_2d,
            y_ret_train,
            eval_set=[(X_val_2d, y_ret_val)],
            verbose=False,
        )

        val_dir_prob = clf.predict_proba(X_val_2d)[:, 1]
        threshold = _calibrate_threshold(y_dir_val, val_dir_prob)

        y_dir_prob = clf.predict_proba(X_test_2d)[:, 1]
        y_ret_pred = reg.predict(X_test_2d)

        model_path = ARTIFACTS_DIR / f"{artifact_base}.joblib"
        joblib.dump({"classifier": clf, "regressor": reg}, model_path)
        hparams_used = dict(cfg.xgb_hparams)

        y_dir_eval = test["target_direction"].to_numpy(dtype="int8")
        y_ret_eval = test["target_return"].to_numpy(dtype="float32")
    else:
        raise ValueError(f"Unknown model_kind {model_kind!r}")

    metrics = _holdout_metrics(
        y_dir_eval,
        y_dir_prob,
        y_ret_eval,
        y_ret_pred,
        naive_up_acc=naive_up,
        naive_yest_rmse=naive_yest_rmse,
        threshold=threshold,
    )

    metadata: dict[str, Any] = {
        "ticker_key": ticker_key,
        "ticker_symbol": cfg.ticker,
        "model_kind": model_kind,
        "feature_order": list(cfg.feature_columns),
        "normalizer": norm,
        "seq_len": cfg.seq_len if model_kind == "transformer" else None,
        "holdout_metrics": metrics,
        "train_window": {
            "start": train.index.min().date().isoformat(),
            "end": train.index.max().date().isoformat(),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        },
        "hparams": hparams_used,
        "stl_outlier_removal_applied": bool(apply_stl),
        "stl_outliers_dropped": int(n_outliers),
        "epochs_run": epochs_run,
        "last_train_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "artifact_filename": model_path.name,
    }
    metadata_path = ARTIFACTS_DIR / f"{artifact_base}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info(
        "[%s/%s] holdout: dir_acc=%.4f (thr=%.3f, @0.5=%.4f, auc=%.3f) vs naive=%.4f | "
        "ret_mae=%.5f vs naive_rmse=%.5f | wrote %s",
        ticker_key,
        model_kind,
        metrics["direction_accuracy"],
        metrics["direction_threshold"],
        metrics["direction_accuracy_at_0_5"],
        metrics["direction_auc"],
        metrics["naive_always_up_accuracy"],
        metrics["return_mae"],
        metrics["naive_yesterday_return_rmse"],
        model_path.name,
    )

    return metadata


def _bakeoff_score(metrics: dict) -> tuple[float, float]:
    """The bake-off ranking key.

    Uses ``direction_accuracy_at_0_5`` (default 0.5 threshold) — *not* the
    calibrated ``direction_accuracy`` — so the ranking is consistent with how
    PLAN.md specifies the comparison and isn't shifted by val→test threshold
    variance. The calibrated accuracy is still recorded in metadata for
    Phase 5 inference and the bake-off audit log.

    Returns (acc, -mae) so ``max()`` gives the higher-accuracy / lower-mae
    candidate (tie-break per plan).
    """
    acc = metrics.get("direction_accuracy_at_0_5", metrics["direction_accuracy"])
    return float(acc), -float(metrics["return_mae"])


def _better(meta_a: dict, meta_b: dict) -> dict:
    """Pick the higher-direction-accuracy metadata; tie → lower return_mae."""
    return meta_a if _bakeoff_score(meta_a["holdout_metrics"]) >= _bakeoff_score(
        meta_b["holdout_metrics"]
    ) else meta_b


def run_bakeoff(ticker_key: str) -> dict[str, Any]:
    """Compare trained models for a ticker, declare winner, write production aliases.

    Looks for, in order of precedence:
      - Transformer STL A/B pair: ``{ticker}_transformer_stl0_metadata.json``
        and ``..._stl1_metadata.json`` (use the better one as transformer
        candidate; record both in the bake-off file).
      - Canonical: ``{ticker}_transformer_metadata.json`` (used if A/B pair
        missing).
      - ``{ticker}_xgb_metadata.json``.

    The winner's artifact is copied to ``artifacts/{ticker}_production.{ext}``
    and its metadata to ``..._production_metadata.json``. The honesty gate
    (winner direction_accuracy must beat naive_always_up by ≥3 pp) is recorded
    but does not block writing — surfacing the failure is the point.
    """
    if ticker_key not in TICKERS:
        raise KeyError(ticker_key)

    candidates: dict[str, dict] = {}
    stl_ab: dict[str, dict] | None = None

    stl0 = ARTIFACTS_DIR / f"{ticker_key}_transformer_stl0_metadata.json"
    stl1 = ARTIFACTS_DIR / f"{ticker_key}_transformer_stl1_metadata.json"
    canonical = ARTIFACTS_DIR / f"{ticker_key}_transformer_metadata.json"

    if stl0.exists() and stl1.exists():
        m0 = json.loads(stl0.read_text())
        m1 = json.loads(stl1.read_text())
        stl_ab = {"stl_off": m0["holdout_metrics"], "stl_on": m1["holdout_metrics"]}
        candidates["transformer"] = _better(m0, m1)
    elif canonical.exists():
        candidates["transformer"] = json.loads(canonical.read_text())

    xgb_meta_path = ARTIFACTS_DIR / f"{ticker_key}_xgb_metadata.json"
    if xgb_meta_path.exists():
        candidates["xgb"] = json.loads(xgb_meta_path.read_text())

    if not candidates:
        raise FileNotFoundError(
            f"No trained model metadata found for {ticker_key} in {ARTIFACTS_DIR}"
        )

    winner_kind = max(
        candidates,
        key=lambda k: _bakeoff_score(candidates[k]["holdout_metrics"]),
    )
    winner_meta = candidates[winner_kind]

    src_artifact = ARTIFACTS_DIR / winner_meta["artifact_filename"]
    if not src_artifact.exists():
        raise FileNotFoundError(
            f"Winner metadata points to missing artifact {src_artifact}"
        )
    ext = src_artifact.suffix
    dst_artifact = ARTIFACTS_DIR / f"{ticker_key}_production{ext}"
    shutil.copy2(src_artifact, dst_artifact)
    prod_meta = dict(winner_meta)
    prod_meta["promoted_from"] = winner_meta["artifact_filename"]
    prod_meta["artifact_filename"] = dst_artifact.name
    prod_meta["promoted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (ARTIFACTS_DIR / f"{ticker_key}_production_metadata.json").write_text(
        json.dumps(prod_meta, indent=2, default=str)
    )

    m = winner_meta["holdout_metrics"]
    # Honesty gate uses the same 0.5-threshold accuracy as the bake-off ranking.
    gate_acc = m.get("direction_accuracy_at_0_5", m["direction_accuracy"])
    delta_pp = (gate_acc - m["naive_always_up_accuracy"]) * 100.0

    bakeoff: dict[str, Any] = {
        "ticker_key": ticker_key,
        "winner": winner_kind,
        "winner_artifact": winner_meta["artifact_filename"],
        "winner_metrics": m,
        "candidates": {
            k: v["holdout_metrics"] for k, v in candidates.items()
        },
        "honesty_gate": {
            "naive_always_up_accuracy": m["naive_always_up_accuracy"],
            "winner_direction_accuracy_at_0_5": gate_acc,
            "winner_direction_accuracy_calibrated": m["direction_accuracy"],
            "delta_pp": delta_pp,
            "threshold_pp": 3.0,
            "passes": bool(delta_pp >= 3.0),
        },
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if stl_ab is not None:
        bakeoff["transformer_stl_ab"] = stl_ab
        if winner_kind == "transformer":
            bakeoff["winner_stl_setting"] = winner_meta.get(
                "stl_outlier_removal_applied"
            )

    bakeoff_path = ARTIFACTS_DIR / f"bakeoff_{ticker_key}.json"
    bakeoff_path.write_text(json.dumps(bakeoff, indent=2, default=str))

    logger.info(
        "[%s] bakeoff winner: %s (dir_acc=%.4f, naive=%.4f, delta=%+.2fpp, gate=%s)",
        ticker_key,
        winner_kind,
        m["direction_accuracy"],
        m["naive_always_up_accuracy"],
        delta_pp,
        "PASS" if bakeoff["honesty_gate"]["passes"] else "FAIL",
    )
    return bakeoff
