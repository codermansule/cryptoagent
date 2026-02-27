"""
Model Training Pipeline
Downloads historical OHLCV data, builds features, trains the LightGBM
directional classifier with walk-forward cross-validation, and saves
the model to models/lgbm_classifier.joblib.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --symbol BTC-USDC --timeframe 15m --candles 3000
    python scripts/train_models.py --force-retrain --horizon 6 --threshold 0.006
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.exchanges.blofin import BloFinAdapter

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def fetch_candles(
    symbol: str,
    timeframe: str,
    n_candles: int,
) -> pd.DataFrame:
    """Fetch historical OHLCV from BloFin and return as a DataFrame."""
    exchange = BloFinAdapter()
    await exchange.connect()

    all_candles = []
    since: int | None = None
    batch_size = 300

    print(f"  Fetching {n_candles} candles for {symbol}/{timeframe} ...")

    while len(all_candles) < n_candles:
        batch = await exchange.get_candles(symbol, timeframe, limit=batch_size, since=since)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[0].timestamp - 1
        print(f"    {len(all_candles)}/{n_candles} candles fetched", end="\r")
        await asyncio.sleep(0.15)

    await exchange.disconnect()

    if not all_candles:
        raise RuntimeError(f"No candles returned for {symbol}/{timeframe}")

    # Sort ascending by timestamp
    all_candles.sort(key=lambda c: c.timestamp)

    df = pd.DataFrame([{
        "open":   c.open,
        "high":   c.high,
        "low":    c.low,
        "close":  c.close,
        "volume": c.volume,
    } for c in all_candles])

    print(f"\n  Fetched {len(df)} candles.")
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    model_path: Path,
    horizon: int,
    threshold: float,
    n_splits: int,
    drop_flat: bool,
) -> dict:
    from src.signals.ml_models.classifier import DirectionalClassifier

    print(f"\n  Training classifier on {len(df)} bars  "
          f"(horizon={horizon}, threshold={threshold:.3%}, "
          f"walk-forward splits={n_splits}) ...")

    clf = DirectionalClassifier(
        model_path=model_path,
        horizon=horizon,
        threshold=threshold,
    )

    if n_splits > 1:
        fold_metrics = clf.walk_forward_train(df, n_splits=n_splits, drop_flat=drop_flat)
        for fm in fold_metrics:
            print(f"    Fold {fm['fold']}/{n_splits}  "
                  f"val_acc={fm['val_accuracy']:.3f}  "
                  f"n_train={fm['n_train']}  n_val={fm['n_val']}")
        avg_acc = sum(f["val_accuracy"] for f in fold_metrics) / len(fold_metrics) if fold_metrics else 0
        metrics = {"avg_val_accuracy": avg_acc, "folds": fold_metrics}
    else:
        metrics = clf.train(df, drop_flat=drop_flat, verbose=True)

    # Feature importance
    imp = clf.feature_importance(top_n=15)
    print("\n  Top 15 features by LightGBM gain:")
    for fname, val in imp.items():
        bar = "#" * int(val / imp.max() * 30) if imp.max() > 0 else ""
        print(f"    {fname:<35} {bar} {val:.1f}")

    print(f"\n  Model saved → {model_path}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    configure_logging()
    settings = get_settings()

    symbols    = [args.symbol]    if args.symbol    else settings.symbols
    timeframes = [args.timeframe] if args.timeframe else [settings.primary_timeframe]
    n_candles  = args.candles
    horizon    = args.horizon
    threshold  = args.threshold
    n_splits   = args.splits
    drop_flat  = args.drop_flat

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    all_results = []

    for symbol in symbols:
        for timeframe in timeframes:
            slug = f"{symbol.replace('-', '_').lower()}_{timeframe}"
            model_path = model_dir / f"lgbm_{slug}.joblib"

            # Also keep a default model (used by agent.py)
            default_path = model_dir / "lgbm_classifier.joblib"

            print(f"\n{'='*60}")
            print(f"  Symbol: {symbol}  |  Timeframe: {timeframe}")
            print(f"  Model:  {model_path}")
            print(f"{'='*60}")

            if model_path.exists() and not args.force_retrain:
                print(f"  Model already exists. Use --force-retrain to retrain.")
                continue

            try:
                df = await fetch_candles(symbol, timeframe, n_candles)
                metrics = train(df, model_path, horizon, threshold, n_splits, drop_flat)
                all_results.append({"symbol": symbol, "tf": timeframe, **metrics})

                # Copy first trained model as the default
                if not default_path.exists() or args.force_retrain:
                    import shutil
                    shutil.copy2(model_path, default_path)
                    print(f"  Default model updated → {default_path}")

            except Exception as exc:
                logger.error("Training failed for %s/%s: %s", symbol, timeframe, exc, exc_info=True)
                print(f"\n  ERROR: {exc}")

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("  TRAINING SUMMARY")
        print(f"{'='*60}")
        for r in all_results:
            acc = r.get("val_accuracy") or r.get("avg_val_accuracy") or 0
            print(f"  {r['symbol']}/{r['tf']}  val_accuracy={acc:.3f}")
        print(f"\n  Models saved in: {model_dir.resolve()}")
        print("  Done.")
    else:
        print("\n  No models were trained (all already exist or all failed).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CryptoAgent LightGBM directional classifier"
    )
    p.add_argument("--symbol",        type=str,   default=None,
                   help="Single symbol to train on (default: all from settings)")
    p.add_argument("--timeframe",     type=str,   default=None,
                   help="Single timeframe (default: primary_timeframe from settings)")
    p.add_argument("--candles",       type=int,   default=3000,
                   help="Number of historical candles to fetch (default: 3000)")
    p.add_argument("--horizon",       type=int,   default=4,
                   help="Forward bars for label generation (default: 4)")
    p.add_argument("--threshold",     type=float, default=0.005,
                   help="Min return to generate a directional label (default: 0.005 = 0.5%%)")
    p.add_argument("--splits",        type=int,   default=5,
                   help="Walk-forward CV splits (default: 5; use 1 for single train/val)")
    p.add_argument("--drop-flat",     action="store_true",
                   help="Exclude flat (0) labels from training")
    p.add_argument("--force-retrain", action="store_true",
                   help="Retrain even if model file already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
