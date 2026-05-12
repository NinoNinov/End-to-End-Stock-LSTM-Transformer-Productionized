"""CLI: train a single (ticker, model_kind) and write artifacts.

Usage::

    # single training run
    python -m tools.train_model --ticker MSFT --model transformer
    python -m tools.train_model --ticker SPX  --model xgb

    # full Phase 4 pipeline for one ticker (STL A/B for transformer + xgb + bakeoff)
    python -m tools.train_model --ticker MSFT --all

    # bake-off using whatever metadata already exists in artifacts/
    python -m tools.train_model --ticker MSFT --bakeoff

    # everything for every ticker
    python -m tools.train_model --ticker all --all

The ``--stl`` flag overrides the per-ticker config default. ``auto`` (the
default) uses ``TickerConfig.apply_stl_outlier_removal``. ``--all`` ignores
``--stl`` because it runs the A/B explicitly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import TICKERS
from src.train import run_bakeoff, train_ticker


def _parse_stl(value: str) -> bool | None:
    s = value.strip().lower()
    if s in ("none", "auto", ""):
        return None
    if s in ("0", "false", "off", "no"):
        return False
    if s in ("1", "true", "on", "yes"):
        return True
    raise argparse.ArgumentTypeError(f"invalid --stl value {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools.train_model")
    p.add_argument(
        "--ticker",
        required=True,
        choices=list(TICKERS) + ["all"],
        help="ticker key, or 'all' to iterate over every configured ticker",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=["transformer", "xgb"])
    g.add_argument(
        "--all",
        dest="full_pipeline",
        action="store_true",
        help="STL A/B for transformer + train xgb + run bake-off",
    )
    g.add_argument(
        "--bakeoff",
        action="store_true",
        help="run bake-off using existing trained metadata only",
    )
    p.add_argument(
        "--stl",
        type=_parse_stl,
        default=None,
        help="auto (config default) | true | false; ignored with --all",
    )
    return p


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)
    tickers = list(TICKERS) if args.ticker == "all" else [args.ticker]

    for tk in tickers:
        if args.full_pipeline:
            logging.info("[%s] STL A/B transformer (off)", tk)
            train_ticker(tk, "transformer", apply_stl=False, artifact_suffix="_stl0")
            logging.info("[%s] STL A/B transformer (on)", tk)
            train_ticker(tk, "transformer", apply_stl=True, artifact_suffix="_stl1")
            logging.info("[%s] xgb", tk)
            train_ticker(tk, "xgb")
            logging.info("[%s] running bake-off", tk)
            _print(run_bakeoff(tk))
        elif args.bakeoff:
            _print(run_bakeoff(tk))
        else:
            meta = train_ticker(tk, args.model, apply_stl=args.stl)
            _print(
                {
                    "ticker": tk,
                    "model": args.model,
                    "stl_applied": meta["stl_outlier_removal_applied"],
                    "metrics": meta["holdout_metrics"],
                    "artifact": meta["artifact_filename"],
                    "epochs_run": meta["epochs_run"],
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
