"""
Parameter Grid Search Optimizer for CryptoAgent

Efficiently finds optimal (confidence threshold, ATR SL multiplier) by
precomputing signals once and resimulating across all combinations in O(n) each.

Strategy:
  - Precomputes 1h / 4h signals ONCE  (independent of conf/atr_mult)
  - Precomputes LTF raw signals with min_confidence=1 ONCE
  - For each (conf, atr_mult): filters + recomputes sl/tp + simulates in O(n)

Grid defaults:
  - Confidence thresholds: [25, 30, 35, 40, 45, 50, 55, 60]%
  - ATR SL multipliers:    [1.5, 2.0, 2.5, 3.0]
  - RR ratio:              2.0 (fixed; override with --rr)
  - OOS split:             25% out-of-sample (last quarter of data)

Usage:
    python scripts/optimize_params.py --symbol BTC-USDC --candles 3000
    python scripts/optimize_params.py --symbol BTC-USDC --timeframe 5m --candles 3000
    python scripts/optimize_params.py --symbol ETH-USDC --from-csv data.csv
    python scripts/optimize_params.py --symbol BTC-USDC --conf 30 40 50 --atr 1.5 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd

from scripts.backtest import (
    resample_ohlcv,
    _precompute_tf_signals,
    simulate,
    compute_metrics,
    fetch_from_exchange,
    load_from_csv,
)
from src.core.logging import configure_logging

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONF_THRESHOLDS: list[float] = [25, 30, 35, 40, 45, 50, 55, 60]
DEFAULT_ATR_MULTS:       list[float] = [1.5, 2.0, 2.5, 3.0]

# Minutes per timeframe — used to compute HTF resample ratios
_TF_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240,
}


def _htf_ratios(timeframe: str) -> tuple[int, int]:
    """Return (bars_per_1h, bars_per_4h) for the given LTF timeframe."""
    mins = _TF_MINUTES.get(timeframe, 15)
    return max(1, 60 // mins), max(1, 240 // mins)


# ── Precompute raw LTF signals (min_conf=1 → capture everything) ──────────────

def precompute_ltf_raw(
    df: pd.DataFrame,
    window: int,
    warmup: int,
    model_path: str,
    timeframe: str = "15m",
) -> pd.DataFrame:
    """
    Run compute_ensemble() on all bars with min_confidence=1.
    Returns DataFrame: direction, confidence, atr, close — indexed like df.
    """
    from src.signals.ensemble import compute_ensemble
    from src.signals.technical.indicators import atr as compute_atr

    # Load ML model once
    clf = None
    model_file = Path(model_path)
    if model_file.exists():
        try:
            from src.signals.ml_models.classifier import DirectionalClassifier
            clf = DirectionalClassifier(model_path=model_file)
            print(f"  ML model: {model_file.name} ({len(clf.feature_names)} features)")
        except Exception as e:
            print(f"  ML model load failed ({e}) — running without ML")
    else:
        print(f"  No ML model at {model_file}")

    n = len(df)
    raw_dir  = np.zeros(n, dtype=int)
    raw_conf = np.zeros(n, dtype=float)
    raw_atr  = np.zeros(n, dtype=float)
    closes   = df["close"].values

    print(f"  Precomputing {timeframe} raw signals ({n} bars, window={window}) ...")
    for i in range(warmup, n):
        start = max(0, i - window + 1)
        chunk = df.iloc[start: i + 1].copy()
        chunk.reset_index(drop=True, inplace=True)

        ml_signal, ml_conf = 0, 0.0
        if clf is not None:
            try:
                ml_signal, ml_conf = clf.predict(chunk)
            except Exception:
                pass

        try:
            sig = compute_ensemble(
                df=chunk, symbol="OPT", timeframe=timeframe,
                ml_signal=ml_signal, ml_confidence=ml_conf,
                min_confidence=1,   # no threshold — capture all signals
            )
            raw_dir[i]  = sig.direction
            raw_conf[i] = sig.confidence
        except Exception:
            pass

        # ATR is needed for sl/tp recomputation; always compute regardless of signal
        atr_s = compute_atr(chunk, 14)
        raw_atr[i] = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0

        if i % 200 == 0:
            print(f"    {i}/{n}", end="\r")

    print(f"    {n}/{n}   ")
    return pd.DataFrame(
        {"direction": raw_dir, "confidence": raw_conf, "atr": raw_atr, "close": closes},
        index=df.index,
    )


# ── Build filtered signals for a single (conf, atr_mult) combo ────────────────

def build_signals_for_combo(
    raw15: pd.DataFrame,
    htf_1h: list[int],
    htf_4h: list[int],
    conf_thresh: float,
    atr_mult: float,
    rr_ratio: float,
    min_agreeing: int,
    ratio_1h: int = 4,
    ratio_4h: int = 16,
) -> pd.DataFrame:
    """
    Filter precomputed LTF raw signals by conf_thresh, apply MTF confluence,
    and compute sl/tp using atr_mult. Returns signals DataFrame compatible
    with simulate().
    ratio_1h / ratio_4h: number of LTF bars per 1h / 4h bar (depends on timeframe).
    """
    n = len(raw15)
    directions  = np.zeros(n, dtype=int)
    confidences = np.zeros(n, dtype=float)
    sls         = np.zeros(n, dtype=float)
    tps         = np.zeros(n, dtype=float)
    atrs        = raw15["atr"].values.copy()

    dirs  = raw15["direction"].values
    confs = raw15["confidence"].values
    closs = raw15["close"].values

    for i in range(n):
        d = dirs[i]
        c = confs[i]
        if d == 0 or c < conf_thresh:
            continue

        # MTF confluence gate (same logic as backtest.generate_signals)
        if htf_1h and htf_4h:
            j_1h = i // ratio_1h - 1
            j_4h = i // ratio_4h - 1
            d1h = htf_1h[j_1h] if 0 <= j_1h < len(htf_1h) else 0
            d4h = htf_4h[j_4h] if 0 <= j_4h < len(htf_4h) else 0
            agreeing = sum(1 for x in [d, d1h, d4h] if x == d)
            if agreeing < min_agreeing:
                continue

        close = closs[i]
        atr_v = atrs[i]
        if atr_v <= 0:
            continue

        directions[i]  = d
        confidences[i] = c

        if d == 1:
            sls[i] = close - atr_mult * atr_v
            tps[i] = close + rr_ratio * atr_mult * atr_v
        else:
            sls[i] = close + atr_mult * atr_v
            tps[i] = close - rr_ratio * atr_mult * atr_v

    return pd.DataFrame(
        {"direction": directions, "confidence": confidences,
         "sl": sls, "tp": tps, "atr": atrs},
        index=raw15.index,
    )


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid(
    df: pd.DataFrame,
    raw15: pd.DataFrame,
    htf_1h: list[int],
    htf_4h: list[int],
    conf_thresholds: list[float],
    atr_mults: list[float],
    rr_ratio: float,
    min_agreeing: int,
    initial_equity: float,
    fee_pct: float,
    slippage_pct: float,
    kelly_fraction: float,
    oos_split: float,
    ratio_1h: int = 4,
    ratio_4h: int = 16,
) -> pd.DataFrame:
    """
    Simulate each (conf, atr_mult) combination, splitting data IS/OOS.
    Returns a DataFrame with per-combination metrics.
    """
    n = len(df)
    oos_start = int(n * (1 - oos_split))
    results   = []
    total     = len(conf_thresholds) * len(atr_mults)
    done      = 0

    for atr_mult in atr_mults:
        for conf in conf_thresholds:
            done += 1
            print(f"  [{done:2d}/{total}] conf={conf:.0f}%  atr={atr_mult:.1f}x ...", end=" ")

            signals = build_signals_for_combo(
                raw15=raw15, htf_1h=htf_1h, htf_4h=htf_4h,
                conf_thresh=conf, atr_mult=atr_mult, rr_ratio=rr_ratio,
                min_agreeing=min_agreeing,
                ratio_1h=ratio_1h, ratio_4h=ratio_4h,
            )

            n_sig = int((signals["direction"] != 0).sum())
            if n_sig == 0:
                print("0 signals — skip")
                results.append({
                    "conf_pct": conf, "atr_mult": atr_mult, "rr": rr_ratio,
                    "is_signals": 0, "oos_signals": 0,
                    "is_trades": 0, "oos_trades": 0,
                    "is_sharpe": 0.0, "oos_sharpe": 0.0,
                    "is_winrate": 0.0, "oos_winrate": 0.0,
                    "is_return_pct": 0.0, "oos_return_pct": 0.0,
                    "is_maxdd": 0.0, "oos_maxdd": 0.0,
                    "oos_expectancy": 0.0, "oos_profit_factor": 0.0,
                })
                continue

            # Simulate on full data, then split by exit index
            trades_full, curve_full = simulate(
                df=df, signals=signals,
                initial_equity=initial_equity,
                fee_pct=fee_pct, slippage_pct=slippage_pct,
                kelly_fraction=kelly_fraction,
                rr_ratio=rr_ratio, max_open=1,
                trailing_activation_atr=0,
            )

            is_trades  = [t for t in trades_full if t.exit_idx >= 0 and t.exit_idx < oos_start]
            oos_trades = [t for t in trades_full if t.exit_idx >= oos_start]

            is_curve  = curve_full.iloc[:oos_start]
            oos_curve = curve_full.iloc[oos_start:]

            # Rebase OOS curve to initial_equity for fair Sharpe comparison
            oos_base = float(is_curve.iloc[-1]) if not is_curve.empty else initial_equity
            oos_curve_r = oos_curve - oos_base + initial_equity

            m_is  = compute_metrics(is_trades,  is_curve,    initial_equity) if is_trades  else {}
            m_oos = compute_metrics(oos_trades, oos_curve_r, initial_equity) if oos_trades else {}

            row = {
                "conf_pct":    conf,
                "atr_mult":    atr_mult,
                "rr":          rr_ratio,
                "is_signals":  int((signals["direction"].iloc[:oos_start] != 0).sum()),
                "oos_signals": int((signals["direction"].iloc[oos_start:] != 0).sum()),
                "is_trades":   len(is_trades),
                "oos_trades":  len(oos_trades),
                "is_sharpe":   m_is.get("sharpe_ratio",    0.0),
                "oos_sharpe":  m_oos.get("sharpe_ratio",   0.0),
                "is_winrate":  m_is.get("win_rate",        0.0),
                "oos_winrate": m_oos.get("win_rate",       0.0),
                "is_return_pct":  m_is.get("total_return_pct",  0.0),
                "oos_return_pct": m_oos.get("total_return_pct", 0.0),
                "is_maxdd":    m_is.get("max_drawdown_pct",  0.0),
                "oos_maxdd":   m_oos.get("max_drawdown_pct", 0.0),
                "oos_expectancy":    m_oos.get("expectancy_usdc",  0.0),
                "oos_profit_factor": m_oos.get("profit_factor",   0.0),
            }
            print(
                f"IS Sharpe={row['is_sharpe']:+.2f}  "
                f"OOS Sharpe={row['oos_sharpe']:+.2f}  "
                f"OOS trades={row['oos_trades']}"
            )
            results.append(row)

    return pd.DataFrame(results)


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_results(df_results: pd.DataFrame, symbol: str, top_n: int = 10) -> None:
    sep = "=" * 95
    print(f"\n{sep}")
    print(f"  GRID SEARCH RESULTS  |  {symbol}  |  Sorted by OOS Sharpe  (top {top_n})")
    print(sep)

    sorted_df = df_results.sort_values("oos_sharpe", ascending=False).head(top_n)
    hdr = f"  {'Rank':<4}  {'Conf%':<7} {'ATR':<5} {'IS Sharpe':<11} {'OOS Sharpe':<12} {'OOS WR':<9} {'OOS Trades':<12} {'OOS Ret%':<10} {'OOS DD%'}"
    print(hdr)
    print(f"  {'-'*4}  {'-'*6} {'-'*4} {'-'*10} {'-'*11} {'-'*8} {'-'*11} {'-'*9} {'-'*8}")

    for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
        print(
            f"  {rank:<4}  {row['conf_pct']:<7.0f} {row['atr_mult']:<5.1f} "
            f"{row['is_sharpe']:<11.2f} {row['oos_sharpe']:<12.2f} "
            f"{row['oos_winrate']:<9.1%} {row['oos_trades']:<12.0f} "
            f"{row['oos_return_pct']:<10.1%} {row['oos_maxdd']:.1%}"
        )

    print(sep)

    if not sorted_df.empty:
        best = sorted_df.iloc[0]
        if not pd.isna(best["oos_sharpe"]) and best["oos_trades"] > 0:
            print(f"\n  **  BEST: conf={best['conf_pct']:.0f}%  |  ATR={best['atr_mult']:.1f}x  "
                  f"|  OOS Sharpe={best['oos_sharpe']:.2f}  |  OOS trades={best['oos_trades']:.0f}")
            print(f"\n  To update config/settings.yaml:")
            print(f"    min_confidence_threshold: {best['conf_pct']:.0f}")
            print(f"    atr_sl_multiplier: {best['atr_mult']:.1f}")
        else:
            print("\n  No profitable OOS combination found — market may be trending or data too short.")

    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    configure_logging()
    from src.core.config import get_settings
    settings = get_settings()

    symbol    = args.symbol or settings.symbols[0]
    timeframe = args.timeframe or "15m"
    ratio_1h, ratio_4h = _htf_ratios(timeframe)

    # Auto-detect per-symbol + per-timeframe model
    model_path = args.model_path
    if model_path == "models/lgbm_classifier.joblib":
        slug     = symbol.lower().replace("-", "_")
        specific = Path(f"models/lgbm_{slug}_{timeframe}.joblib")
        if not specific.exists():
            specific = Path(f"models/lgbm_{slug}_15m.joblib")  # fallback to 15m
        if specific.exists():
            model_path = str(specific)
            print(f"  Using per-symbol model: {model_path}")

    # Load data
    if args.from_csv:
        print(f"\n  Loading data from {args.from_csv}")
        df = load_from_csv(args.from_csv)
    else:
        df = await fetch_from_exchange(symbol, timeframe, args.candles)

    if len(df) < 300:
        print("  ERROR: Need at least 300 bars.")
        return

    n_oos = int(len(df) * args.oos_split)
    print(
        f"\n  Dataset: {len(df)} bars ({timeframe})  |  OOS: last {args.oos_split:.0%} "
        f"({n_oos} bars)  |  IS: {len(df)-n_oos} bars"
    )
    print(f"  HTF ratios: 1h={ratio_1h}× bars, 4h={ratio_4h}× bars")

    # ── Step 1: Precompute HTF signals (once, reused for all combos) ──────────
    print("\n  [1/3] Precomputing 1h signals ...")
    df_1h  = resample_ohlcv(df, ratio_1h)
    htf_1h = _precompute_tf_signals(
        df_1h[["open", "high", "low", "close", "volume"]],
        window=max(args.window // ratio_1h, 50),
        min_confidence=1,
        timeframe_label="1h",
        clf=None,
    )

    print("  [1/3] Precomputing 4h signals ...")
    df_4h  = resample_ohlcv(df, ratio_4h)
    htf_4h = _precompute_tf_signals(
        df_4h[["open", "high", "low", "close", "volume"]],
        window=max(args.window // ratio_4h, 10),
        min_confidence=1,
        timeframe_label="4h",
        clf=None,
    )

    # ── Step 2: Precompute LTF raw signals ────────────────────────────────────
    print(f"\n  [2/3] Precomputing {timeframe} raw signals (min_conf=1) ...")
    raw15 = precompute_ltf_raw(
        df=df, window=args.window, warmup=args.warmup,
        model_path=model_path, timeframe=timeframe,
    )

    n_raw = int((raw15["direction"] != 0).sum())
    print(f"  Raw {timeframe} signals captured: {n_raw} "
          f"(longs={int((raw15['direction']==1).sum())}, "
          f"shorts={int((raw15['direction']==-1).sum())})")

    # ── Step 3: Grid search ───────────────────────────────────────────────────
    conf_thresholds = list(args.conf)
    atr_mults       = list(args.atr)
    n_combos        = len(conf_thresholds) * len(atr_mults)
    print(f"\n  [3/3] Grid: {len(conf_thresholds)} conf × {len(atr_mults)} ATR = {n_combos} combos")

    results = run_grid(
        df=df, raw15=raw15, htf_1h=htf_1h, htf_4h=htf_4h,
        conf_thresholds=conf_thresholds, atr_mults=atr_mults,
        rr_ratio=args.rr, min_agreeing=args.min_agreeing,
        initial_equity=args.equity,
        fee_pct=args.fee / 100.0,
        slippage_pct=args.slippage / 100.0,
        kelly_fraction=args.kelly,
        oos_split=args.oos_split,
        ratio_1h=ratio_1h, ratio_4h=ratio_4h,
    )

    # Save first — before print so a display crash never loses results
    out_dir  = Path("backtests")
    out_dir.mkdir(exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug     = symbol.lower().replace("-", "_")
    out_path = out_dir / f"optimize_{slug}_{timeframe}_{ts}.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  Full results saved -> {out_path}")

    # Print results
    print_results(results, symbol, top_n=args.top)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CryptoAgent parameter grid search optimizer")

    p.add_argument("--symbol",     type=str, default=None,
                   help="Symbol to optimize (default: first from settings.yaml)")
    p.add_argument("--timeframe",  type=str, default=None,
                   help="Timeframe to optimize: 1m, 3m, 5m, 15m, etc. (default: 15m)")
    p.add_argument("--candles",    type=int, default=3000,
                   help="Candles to fetch (default: 3000)")
    p.add_argument("--from-csv",   type=str, default=None,
                   help="Load OHLCV from CSV instead of fetching")
    p.add_argument("--window",     type=int, default=300,
                   help="Rolling window for signal engine (default: 300)")
    p.add_argument("--warmup",     type=int, default=100,
                   help="Bars to skip at start (default: 100)")
    p.add_argument("--model-path", type=str, default="models/lgbm_classifier.joblib")
    p.add_argument("--conf",       nargs="+", type=float,
                   default=DEFAULT_CONF_THRESHOLDS,
                   help="Confidence thresholds to test (default: 25 30 35 40 45 50 55 60)")
    p.add_argument("--atr",        nargs="+", type=float,
                   default=DEFAULT_ATR_MULTS,
                   help="ATR SL multipliers to test (default: 1.5 2.0 2.5 3.0)")
    p.add_argument("--rr",         type=float, default=2.0,
                   help="Reward:risk ratio (default: 2.0)")
    p.add_argument("--min-agreeing", type=int, default=2,
                   help="Min TFs that must agree for confluence (default: 2)")
    p.add_argument("--equity",     type=float, default=10000.0)
    p.add_argument("--fee",        type=float, default=0.05)
    p.add_argument("--slippage",   type=float, default=0.02)
    p.add_argument("--kelly",      type=float, default=0.25)
    p.add_argument("--oos-split",  type=float, default=0.25,
                   help="Fraction of data held out for OOS testing (default: 0.25)")
    p.add_argument("--top",        type=int,   default=10,
                   help="Top N results to display (default: 10)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
