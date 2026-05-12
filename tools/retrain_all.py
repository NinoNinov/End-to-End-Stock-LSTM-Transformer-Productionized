"""CLI: weekly retrain of all configured tickers + atomic artifact swap.

Implements PLAN.md Phase 7 (retrain loop):

For every ticker in ``src.config.TICKERS``:
1. Read ``artifacts/{ticker_key}_production_metadata.json`` to discover
   which model family is currently in production (``transformer`` or ``xgb``)
   and whether STL outlier removal was applied. We retrain *that same
   configuration* rather than re-running the full bake-off; the bake-off is
   a Phase 4 / human-supervised choice, not a weekly automated one.
2. Call ``src.train.train_ticker`` to fit fresh weights on the latest cached
   data.
3. Atomically swap the production artifact:
     - rotate current ``{key}_production.{ext}`` → ``{key}_production.prev.{ext}``
     - rotate current ``..._production_metadata.json`` → ``..._production_metadata.prev.json``
     - write fresh model bytes to ``{key}_production.tmp.{ext}`` + fsync,
       then ``os.replace`` to the production path (atomic on the same volume)
     - same temp+rename dance for the metadata sidecar

The API's mtime-based hot-reload (`src.inference.maybe_reload`) picks up
the new artifact on the next request — no server restart.

Manual rollback (documented in ``workflows/weekly_retrain.md``)::

    cd artifacts/
    mv MSFT_production.prev.keras            MSFT_production.keras
    mv MSFT_production_metadata.prev.json    MSFT_production_metadata.json

Run::

    C:\\venvs\\stock-predictor\\.venv\\Scripts\\python.exe -m tools.retrain_all
    # or one ticker only:
    C:\\venvs\\stock-predictor\\.venv\\Scripts\\python.exe -m tools.retrain_all --ticker MSFT
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TICKERS  # noqa: E402  (path setup must come first)
from src.train import train_ticker  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

logger = logging.getLogger(__name__)


def _production_paths(ticker_key: str, ext: str) -> dict[str, Path]:
    """Group the file paths involved in the swap so the code reads top-to-bottom."""
    base = ARTIFACTS_DIR / f"{ticker_key}_production"
    return {
        "prod_model": base.with_suffix(ext),
        "prev_model": ARTIFACTS_DIR / f"{ticker_key}_production.prev{ext}",
        "tmp_model": ARTIFACTS_DIR / f"{ticker_key}_production.tmp{ext}",
        "prod_meta": ARTIFACTS_DIR / f"{ticker_key}_production_metadata.json",
        "prev_meta": ARTIFACTS_DIR / f"{ticker_key}_production_metadata.prev.json",
        "tmp_meta": ARTIFACTS_DIR / f"{ticker_key}_production_metadata.tmp.json",
    }


def _read_production_kind(ticker_key: str) -> tuple[str, bool]:
    """Return (model_kind, apply_stl) from the current production metadata.

    Raises if no production model exists — by Phase 7 we expect Phase 4 to
    have already run the bake-off and promoted a winner.
    """
    meta_path = ARTIFACTS_DIR / f"{ticker_key}_production_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No production metadata for {ticker_key} at {meta_path}. "
            f"Run the Phase 4 bake-off first: "
            f"`python -m tools.train_model --ticker {ticker_key} --all`."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = meta.get("model_kind")
    if kind not in {"transformer", "xgb"}:
        raise ValueError(
            f"Unexpected model_kind={kind!r} in {meta_path}; cannot retrain."
        )
    return kind, bool(meta.get("stl_outlier_removal_applied", False))


def _fsync_replace(src: Path, dst: Path) -> None:
    """``os.replace(src, dst)`` with an fsync on src first.

    fsync forces the OS to flush the file's bytes to disk before the rename
    swaps it in. Without it, a power loss in the swap window could leave a
    metadata pointer to bytes that never made it to disk.
    """
    fd = os.open(src, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(src, dst)


def _rotate_to_prev(paths: dict[str, Path]) -> None:
    """Move current prod files to ``.prev`` slots (atomic, overwriting)."""
    if paths["prod_model"].exists():
        os.replace(paths["prod_model"], paths["prev_model"])
    if paths["prod_meta"].exists():
        os.replace(paths["prod_meta"], paths["prev_meta"])


def _swap_in_fresh(
    ticker_key: str,
    paths: dict[str, Path],
    fresh_model_path: Path,
    fresh_meta_path: Path,
) -> dict:
    """Write tmp files + atomically replace into production. Returns prod metadata."""
    # Model bytes → tmp → rename. Copying via read_bytes/write_bytes keeps the
    # source file (used by future inspect/diff workflows) intact and makes the
    # destination's mtime "now" (so the API's mtime check fires reliably,
    # sidestepping the shutil.copy2 timestamp-preservation gotcha noted in
    # Phase 4 outcomes).
    paths["tmp_model"].write_bytes(fresh_model_path.read_bytes())
    _fsync_replace(paths["tmp_model"], paths["prod_model"])

    fresh_meta = json.loads(fresh_meta_path.read_text(encoding="utf-8"))
    prod_meta = dict(fresh_meta)
    prod_meta["promoted_from"] = fresh_model_path.name
    prod_meta["artifact_filename"] = paths["prod_model"].name
    prod_meta["promoted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prod_meta["promotion_source"] = "retrain_all"

    paths["tmp_meta"].write_text(json.dumps(prod_meta, indent=2, default=str))
    _fsync_replace(paths["tmp_meta"], paths["prod_meta"])
    return prod_meta


def retrain_one(ticker_key: str, *, swap: bool = True) -> dict:
    """Retrain one ticker against its current production model_kind.

    Parameters
    ----------
    ticker_key
        Key into ``src.config.TICKERS``.
    swap
        If ``False``, train but skip the atomic production swap. Useful for
        smoke-testing the training side without touching the live artifact.
    """
    if ticker_key not in TICKERS:
        raise KeyError(ticker_key)

    model_kind, apply_stl = _read_production_kind(ticker_key)
    ext = ".keras" if model_kind == "transformer" else ".joblib"
    paths = _production_paths(ticker_key, ext)

    logger.info(
        "[%s] retrain start: kind=%s apply_stl=%s",
        ticker_key, model_kind, apply_stl,
    )
    train_ticker(ticker_key, model_kind, apply_stl=apply_stl)

    fresh_model = ARTIFACTS_DIR / f"{ticker_key}_{model_kind}{ext}"
    fresh_meta = ARTIFACTS_DIR / f"{ticker_key}_{model_kind}_metadata.json"
    if not fresh_model.exists() or not fresh_meta.exists():
        raise RuntimeError(
            f"[{ticker_key}] training claimed success but {fresh_model.name} "
            f"or its metadata sidecar is missing"
        )

    if not swap:
        logger.info("[%s] --no-swap: leaving production artifact untouched", ticker_key)
        return {"ticker_key": ticker_key, "swapped": False, "model_kind": model_kind}

    _rotate_to_prev(paths)
    prod_meta = _swap_in_fresh(ticker_key, paths, fresh_model, fresh_meta)

    holdout = prod_meta.get("holdout_metrics", {})
    logger.info(
        "[%s] retrain swap done: dir_acc=%.4f naive=%.4f mae=%.5f → %s",
        ticker_key,
        holdout.get("direction_accuracy", float("nan")),
        holdout.get("naive_always_up_accuracy", float("nan")),
        holdout.get("return_mae", float("nan")),
        paths["prod_model"].name,
    )
    return {
        "ticker_key": ticker_key,
        "swapped": True,
        "model_kind": model_kind,
        "holdout": holdout,
        "prev_model": paths["prev_model"].name,
        "prev_meta": paths["prev_meta"].name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly retrain + atomic swap.")
    parser.add_argument(
        "--ticker",
        action="append",
        choices=sorted(TICKERS.keys()),
        help="Restrict to one ticker (repeatable). Default: all configured tickers.",
    )
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="Train but do not promote to production (smoke-test mode).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    targets = args.ticker if args.ticker else list(TICKERS.keys())

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logger.info("retrain_all started at %s (UTC) for %s", started, targets)
    failures = 0
    for key in targets:
        try:
            retrain_one(key, swap=not args.no_swap)
        except Exception as exc:
            logger.exception("%s: retrain failed (%s) — production artifact unchanged", key, exc)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
