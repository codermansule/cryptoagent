"""
Model Training Pipeline
Downloads historical OHLCV data, builds features, trains the LightGBM
directional classifier with walk-forward cross-validation, and saves
the model to models/lgbm_classifier.joblib.

Also supports training the LSTM sequence model (--train-lstm).

Usage:
    # 1 year of 15-min data (BTC/ETH/SOL), 10 walk-forward folds
    python scripts/train_models.py --candles 35040 --splits 10 --force-retrain

    # Single symbol, LightGBM + LSTM
    python scripts/train_models.py --symbol BTC-USDC --timeframe 15m --candles 35040 --train-lstm

    # Quick test run
    python scripts/train_models.py --candles 3000 --splits 5
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
        "ts":     c.timestamp,   # ms epoch — used by build_features() for time-of-day features
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
    run_shap: bool = False,
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

    # Feature importance (gain-based)
    imp = clf.feature_importance(top_n=15)
    print("\n  Top 15 features by LightGBM gain:")
    for fname, val in imp.items():
        bar = "#" * int(val / imp.max() * 30) if imp.max() > 0 else ""
        print(f"    {fname:<35} {bar} {val:.1f}")

    # Optional SHAP analysis
    if run_shap:
        _run_shap_analysis(clf, df)

    print(f"\n  Model saved -> {model_path}")
    return metrics


def _run_shap_analysis(clf, df: pd.DataFrame, n_samples: int = 500) -> None:
    """Compute SHAP values and print top/bottom features by mean |SHAP|."""
    try:
        import shap
    except ImportError:
        print("\n  SHAP not installed — skipping (pip install shap)")
        return

    try:
        from src.data.preprocessing.features import build_features
        feat_df = build_features(df)
        if feat_df is None or feat_df.empty:
            print("\n  SHAP skipped: feature build returned empty DataFrame")
            return

        # Align feature names with model
        feature_names = clf.feature_names
        available = [f for f in feature_names if f in feat_df.columns]
        if not available:
            print("\n  SHAP skipped: no feature overlap between model and data")
            return

        X = feat_df[available].dropna()
        X = X.iloc[-min(n_samples, len(X)):]

        explainer = shap.TreeExplainer(clf.model.booster_)
        shap_values = explainer.shap_values(X)

        # For multi-class: shap_values is list of arrays (one per class)
        # Take class 1 (LONG) vs rest, or abs mean across all classes
        if isinstance(shap_values, list):
            mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_series = pd.Series(mean_abs_shap, index=available).sort_values(ascending=False)
        top    = shap_series.head(10)
        bottom = shap_series.tail(10)

        print(f"\n  SHAP Analysis (top 10 by mean |SHAP| on {len(X)} samples):")
        for fname, val in top.items():
            bar = "#" * int(val / top.max() * 25) if top.max() > 0 else ""
            print(f"    {fname:<35} {bar} {val:.4f}")

        print(f"\n  SHAP — bottom 10 (noise candidates — consider dropping):")
        for fname, val in bottom.items():
            print(f"    {fname:<35} {val:.4f}")

    except Exception as exc:
        print(f"\n  SHAP analysis failed: {exc}")


def train_lstm(
    df: pd.DataFrame,
    model_path: Path,
    horizon: int,
    threshold: float,
    n_splits: int,
    epochs: int,
) -> dict:
    """Train the LSTM sequence model, optionally with walk-forward CV."""
    try:
        from src.signals.ml_models.lstm import LSTMClassifier
    except ImportError as exc:
        print(f"\n  LSTM skipped: {exc}")
        return {}

    print(f"\n  Training LSTM on {len(df)} bars  "
          f"(horizon={horizon}, threshold={threshold:.3%}, "
          f"epochs={epochs}, walk-forward splits={n_splits}) ...")

    # Pass model_path=None so we do NOT load an existing model before training.
    # Loading an old model sets _feature_names from the old checkpoint, which
    # causes shape mismatches when the feature set has changed (e.g. 69->70 features).
    # We set clf.model_path after init so the final trained model saves there.
    clf = LSTMClassifier(
        model_path=None,
        horizon=horizon,
        threshold=threshold,
    )
    clf.model_path = model_path   # save destination after training

    if n_splits > 1:
        fold_metrics = clf.walk_forward_train(df, n_splits=n_splits, epochs=epochs)
        for fm in fold_metrics:
            print(f"    Fold {fm['fold']}/{n_splits}  val_acc={fm.get('val_accuracy', 0):.3f}")
        avg_acc = sum(f.get("val_accuracy", 0) for f in fold_metrics) / max(len(fold_metrics), 1)
        metrics = {"avg_val_accuracy": avg_acc, "folds": fold_metrics}
    else:
        metrics = clf.train(df, epochs=epochs, verbose=True)

    print(f"\n  LSTM model saved -> {model_path}")
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
            model_path     = model_dir / f"lgbm_{slug}.joblib"
            lstm_path      = model_dir / f"lstm_{slug}.pt"

            # Also keep default models (used by agent.py)
            default_lgbm_path = model_dir / "lgbm_classifier.joblib"
            default_lstm_path = model_dir / "lstm.pt"

            print(f"\n{'='*60}")
            print(f"  Symbol: {symbol}  |  Timeframe: {timeframe}  |  Candles: {n_candles}")
            print(f"  LightGBM model: {model_path}")
            if args.train_lstm:
                print(f"  LSTM model:     {lstm_path}")
            print(f"{'='*60}")

            lgbm_skip = model_path.exists() and not args.force_retrain
            lstm_skip = lstm_path.exists() and not args.force_retrain

            if lgbm_skip and (not args.train_lstm or lstm_skip):
                print(f"  Models already exist. Use --force-retrain to retrain.")
                continue

            try:
                df = await fetch_candles(symbol, timeframe, n_candles)

                # ── LightGBM training
                if not lgbm_skip:
                    metrics = train(df, model_path, horizon, threshold, n_splits, drop_flat,
                                    run_shap=args.shap)
                    all_results.append({"symbol": symbol, "tf": timeframe, "model": "lgbm", **metrics})

                    if not default_lgbm_path.exists() or args.force_retrain:
                        import shutil
                        shutil.copy2(model_path, default_lgbm_path)
                        print(f"  Default LightGBM model updated -> {default_lgbm_path}")
                else:
                    print(f"  LightGBM model exists — skipping (use --force-retrain to override)")

                # ── LSTM training
                if args.train_lstm and not lstm_skip:
                    lstm_metrics = train_lstm(
                        df, lstm_path, horizon, threshold,
                        n_splits=min(n_splits, 3),   # LSTM is slower; cap at 3 folds
                        epochs=args.lstm_epochs,
                    )
                    all_results.append({"symbol": symbol, "tf": timeframe, "model": "lstm", **lstm_metrics})

                    if not default_lstm_path.exists() or args.force_retrain:
                        import shutil
                        shutil.copy2(lstm_path, default_lstm_path)
                        print(f"  Default LSTM model updated -> {default_lstm_path}")
                elif args.train_lstm:
                    print(f"  LSTM model exists — skipping (use --force-retrain to override)")

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
            model_tag = r.get("model", "lgbm")
            print(f"  [{model_tag.upper()}] {r['symbol']}/{r['tf']}  val_accuracy={acc:.3f}")
        print(f"\n  Models saved in: {model_dir.resolve()}")
        print("  Done.")
    else:
        print("\n  No models were trained (all already exist or all failed).")
        print("  Tip: use --force-retrain to retrain existing models.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CryptoAgent LightGBM directional classifier"
    )
    p.add_argument("--symbol",        type=str,   default=None,
                   help="Single symbol to train on (default: all from settings)")
    p.add_argument("--timeframe",     type=str,   default=None,
                   help="Single timeframe (default: primary_timeframe from settings)")
    p.add_argument("--candles",       type=int,   default=10000,
                   help="Number of historical candles to fetch (default: 10000; "
                        "use 35040 for ~1 year of 15m data)")
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
    p.add_argument("--train-lstm",   action="store_true",
                   help="Also train the LSTM sequence model alongside LightGBM")
    p.add_argument("--lstm-epochs",  type=int,   default=30,
                   help="LSTM training epochs (default: 30)")
    p.add_argument("--shap",         action="store_true",
                   help="Run SHAP feature importance analysis after LightGBM training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
