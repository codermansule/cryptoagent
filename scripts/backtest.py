"""
Vectorized Backtesting Harness
Replays historical OHLCV through the full signal engine and simulates trade execution.

Features:
  - Fetches history from BloFin (or loads a local CSV)
  - Runs compute_ensemble() on each bar in a rolling window
  - Simulates fills with realistic fees + slippage
  - ATR-based SL/TP tracking
  - Full performance report: Sharpe, Sortino, Calmar, win rate, expectancy, max DD
  - Exports equity curve and trade log to backtests/ directory

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --symbol BTC-USDC --timeframe 15m --candles 3000
    python scripts/backtest.py --from-csv data/BTC_15m.csv
    python scripts/backtest.py --symbol ETH-USDC --initial-equity 5000 --leverage 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd

from src.core.logging import configure_logging


# ── Trade record ──────────────────────────────────────────────────────────────

class Trade:
    def __init__(
        self, idx: int, side: str, entry: float, sl: float, tp: float,
        size_usdc: float, fee_pct: float,
    ):
        self.entry_idx   = idx
        self.side        = side          # "long" | "short"
        self.entry_price = entry
        self.sl          = sl
        self.tp          = tp
        self.size_usdc   = size_usdc
        self.fee_pct     = fee_pct
        self.exit_idx:   int   = -1
        self.exit_price: float = 0.0
        self.exit_reason: str = ""
        self.pnl:         float = 0.0
        self.pnl_pct:     float = 0.0

    def close(self, idx: int, price: float, reason: str) -> None:
        self.exit_idx   = idx
        self.exit_price = price
        self.exit_reason = reason
        if self.side == "long":
            raw_pnl = (price - self.entry_price) / self.entry_price
        else:
            raw_pnl = (self.entry_price - price) / self.entry_price
        fee = self.fee_pct * 2   # entry + exit
        self.pnl_pct = raw_pnl - fee
        self.pnl     = self.size_usdc * self.pnl_pct

    @property
    def is_open(self) -> bool:
        return self.exit_idx == -1


# ── Signal generation (bar-by-bar rolling window) ─────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    warmup: int,
    window: int,
    min_confidence: float,
    atr_sl_mult: float,
    rr_ratio: float,
    model_path: str = "models/lgbm_classifier.joblib",
) -> pd.DataFrame:
    """
    Run compute_ensemble() on each bar with a rolling `window` of history.
    Loads the trained LightGBM classifier and passes its predictions to the ensemble.
    Returns a DataFrame with columns: direction, confidence, sl, tp, atr.
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
            print(f"  ML model loaded: {model_file} ({len(clf.feature_names)} features)")
        except Exception as e:
            print(f"  ML model load failed ({e}) — running without ML")
    else:
        print(f"  No ML model found at {model_file} — running without ML")

    n = len(df)
    directions   = np.zeros(n, dtype=int)
    confidences  = np.zeros(n, dtype=float)
    sls          = np.zeros(n, dtype=float)
    tps          = np.zeros(n, dtype=float)
    atrs         = np.zeros(n, dtype=float)

    print(f"  Generating signals for {n} bars (window={window}, warmup={warmup}) ...")

    for i in range(warmup, n):
        start = max(0, i - window + 1)
        chunk = df.iloc[start: i + 1].copy()
        chunk.reset_index(drop=True, inplace=True)

        # ML prediction for this bar
        ml_signal, ml_conf = 0, 0.0
        if clf is not None:
            try:
                ml_signal, ml_conf = clf.predict(chunk)
            except Exception:
                pass

        try:
            sig = compute_ensemble(
                df=chunk,
                symbol="BT",
                timeframe="bt",
                ml_signal=ml_signal,
                ml_confidence=ml_conf,
                min_confidence=min_confidence,
            )
        except Exception:
            continue

        close = float(df["close"].iloc[i])
        atr_s = compute_atr(chunk, 14)
        atr_v = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0

        directions[i]  = sig.direction
        confidences[i] = sig.confidence
        atrs[i]        = atr_v

        if sig.direction == 1:
            sls[i] = close - atr_sl_mult * atr_v
            tps[i] = close + rr_ratio * atr_sl_mult * atr_v
        elif sig.direction == -1:
            sls[i] = close + atr_sl_mult * atr_v
            tps[i] = close - rr_ratio * atr_sl_mult * atr_v

        if i % 200 == 0:
            print(f"    Progress: {i}/{n}", end="\r")

    print(f"    Progress: {n}/{n}   ")

    signals = pd.DataFrame({
        "direction":  directions,
        "confidence": confidences,
        "sl":         sls,
        "tp":         tps,
        "atr":        atrs,
    }, index=df.index)

    return signals


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    initial_equity: float,
    fee_pct: float,
    slippage_pct: float,
    kelly_fraction: float,
    rr_ratio: float,
    max_open: int,
) -> tuple[list[Trade], pd.Series]:
    """
    Simulate trade execution bar-by-bar.
    Returns (trade_list, equity_curve).
    """
    equity   = initial_equity
    curve    = pd.Series(np.nan, index=df.index, dtype=float)
    trades: list[Trade] = []
    open_trades: list[Trade] = []

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    for i in range(n):
        h = highs[i]
        l = lows[i]
        c = closes[i]

        # ── Check open trades for SL/TP hits
        still_open = []
        for t in open_trades:
            hit_sl = hit_tp = False
            if t.side == "long":
                if l <= t.sl:
                    hit_sl = True
                    exit_px = min(t.sl, c) * (1 - slippage_pct)
                elif h >= t.tp:
                    hit_tp = True
                    exit_px = t.tp * (1 - slippage_pct)
            else:
                if h >= t.sl:
                    hit_sl = True
                    exit_px = max(t.sl, c) * (1 + slippage_pct)
                elif l <= t.tp:
                    hit_tp = True
                    exit_px = t.tp * (1 + slippage_pct)

            if hit_sl or hit_tp:
                t.close(i, exit_px, "sl" if hit_sl else "tp")
                equity += t.pnl
            else:
                still_open.append(t)

        open_trades = still_open

        # ── Enter new trade on signal
        sig_dir  = int(signals["direction"].iloc[i])
        sig_conf = float(signals["confidence"].iloc[i])
        sl_px    = float(signals["sl"].iloc[i])
        tp_px    = float(signals["tp"].iloc[i])

        if sig_dir != 0 and sl_px > 0 and len(open_trades) < max_open:
            entry_px = c * (1 + slippage_pct) if sig_dir == 1 else c * (1 - slippage_pct)

            # Kelly sizing
            win_rate = sig_conf / 100.0
            kelly    = win_rate - (1 - win_rate) / rr_ratio
            kelly    = max(0.0, min(kelly, 0.5)) * kelly_fraction
            size_usdc = equity * kelly
            size_usdc = max(size_usdc, 1.0)

            t = Trade(
                idx=i,
                side="long" if sig_dir == 1 else "short",
                entry=entry_px,
                sl=sl_px,
                tp=tp_px,
                size_usdc=size_usdc,
                fee_pct=fee_pct,
            )
            open_trades.append(t)
            trades.append(t)

        # ── Mark open trade unrealized P&L in equity curve
        unreal = 0.0
        for t in open_trades:
            if t.side == "long":
                unreal += t.size_usdc * (c - t.entry_price) / t.entry_price
            else:
                unreal += t.size_usdc * (t.entry_price - c) / t.entry_price

        curve.iloc[i] = equity + unreal

    # Close any remaining open trades at last close
    for t in open_trades:
        t.close(n - 1, closes[-1], "end_of_data")
        equity += t.pnl

    curve.iloc[-1] = equity
    curve.ffill(inplace=True)
    return trades, curve


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade], equity_curve: pd.Series, initial_equity: float) -> dict:
    closed  = [t for t in trades if not t.is_open]
    n       = len(closed)
    if n == 0:
        return {"error": "No completed trades"}

    pnls     = np.array([t.pnl for t in closed])
    pnl_pcts = np.array([t.pnl_pct for t in closed])
    wins     = pnls[pnls > 0]
    losses   = pnls[pnls < 0]

    win_rate     = len(wins) / n
    avg_win      = wins.mean()  if len(wins)   > 0 else 0.0
    avg_loss     = losses.mean() if len(losses) > 0 else 0.0
    expectancy   = pnls.mean()
    profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf

    # Drawdown
    peak = equity_curve.cummax()
    dd   = (equity_curve - peak) / peak
    max_dd = float(dd.min())

    # Daily-like returns from equity curve
    ret = equity_curve.pct_change().dropna()
    if len(ret) < 2 or ret.std() == 0:
        sharpe = sortino = calmar = 0.0
    else:
        trading_periods = 365 * 24 * 4   # ~15-min bars per year
        sharpe  = float(ret.mean() / ret.std() * np.sqrt(trading_periods))
        neg_ret = ret[ret < 0]
        down_std = neg_ret.std() if len(neg_ret) > 1 else ret.std()
        sortino  = float(ret.mean() / down_std * np.sqrt(trading_periods)) if down_std else 0.0
        total_ret = (equity_curve.iloc[-1] / initial_equity - 1)
        calmar   = float(total_ret / abs(max_dd)) if max_dd != 0 else 0.0

    total_pnl = equity_curve.iloc[-1] - initial_equity
    total_pct = total_pnl / initial_equity

    return {
        "total_trades":   n,
        "win_rate":       win_rate,
        "avg_win_usdc":   avg_win,
        "avg_loss_usdc":  avg_loss,
        "expectancy_usdc": expectancy,
        "profit_factor":  profit_factor,
        "total_pnl_usdc": total_pnl,
        "total_return_pct": total_pct,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio":   sharpe,
        "sortino_ratio":  sortino,
        "calmar_ratio":   calmar,
        "sl_hits":        sum(1 for t in closed if t.exit_reason == "sl"),
        "tp_hits":        sum(1 for t in closed if t.exit_reason == "tp"),
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(metrics: dict, symbol: str, timeframe: str) -> None:
    if "error" in metrics:
        print(f"\n  No trades generated: {metrics['error']}")
        return

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS  |  {symbol} {timeframe}")
    print(sep)
    print(f"  Total trades:       {metrics['total_trades']}")
    print(f"  Win rate:           {metrics['win_rate']:.1%}")
    print(f"  TP hits / SL hits:  {metrics['tp_hits']} / {metrics['sl_hits']}")
    print(f"  Avg win:            ${metrics['avg_win_usdc']:,.2f}")
    print(f"  Avg loss:           ${metrics['avg_loss_usdc']:,.2f}")
    print(f"  Expectancy:         ${metrics['expectancy_usdc']:,.2f}")
    print(f"  Profit factor:      {metrics['profit_factor']:.2f}")
    print(f"  Total P&L:          ${metrics['total_pnl_usdc']:,.2f}  ({metrics['total_return_pct']:.1%})")
    print(f"  Max drawdown:       {metrics['max_drawdown_pct']:.1%}")
    print(f"  Sharpe ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino ratio:      {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar ratio:       {metrics['calmar_ratio']:.2f}")
    print(sep)


# ── Data loading ──────────────────────────────────────────────────────────────

async def fetch_from_exchange(symbol: str, timeframe: str, n_candles: int) -> pd.DataFrame:
    from src.exchanges.blofin import BloFinAdapter
    exchange = BloFinAdapter()
    await exchange.connect()

    all_candles = []
    since: int | None = None

    print(f"  Fetching {n_candles} candles for {symbol}/{timeframe} ...")

    while len(all_candles) < n_candles:
        batch = await exchange.get_candles(symbol, timeframe, limit=300, since=since)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[0].timestamp - 1
        print(f"    {len(all_candles)}/{n_candles}", end="\r")
        await asyncio.sleep(0.15)

    await exchange.disconnect()

    all_candles.sort(key=lambda c: c.timestamp)
    print(f"\n  Loaded {len(all_candles)} candles.")

    return pd.DataFrame([{
        "open": c.open, "high": c.high, "low": c.low,
        "close": c.close, "volume": c.volume,
    } for c in all_candles])


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[["open", "high", "low", "close", "volume"]].astype(float).reset_index(drop=True)


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(
    trades: list[Trade],
    equity_curve: pd.Series,
    metrics: dict,
    symbol: str,
    timeframe: str,
) -> None:
    out_dir = Path("backtests")
    out_dir.mkdir(exist_ok=True)

    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = f"{symbol.replace('-','_')}_{timeframe}_{ts}"

    # Trade log
    trade_rows = []
    for t in trades:
        trade_rows.append({
            "entry_idx":   t.entry_idx,
            "exit_idx":    t.exit_idx,
            "side":        t.side,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "sl":          t.sl,
            "tp":          t.tp,
            "size_usdc":   t.size_usdc,
            "pnl":         t.pnl,
            "pnl_pct":     t.pnl_pct,
            "exit_reason": t.exit_reason,
        })
    pd.DataFrame(trade_rows).to_csv(out_dir / f"{slug}_trades.csv", index=False)

    # Equity curve
    equity_curve.dropna().to_csv(out_dir / f"{slug}_equity.csv", header=["equity"])

    # Metrics JSON
    with open(out_dir / f"{slug}_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in metrics.items()}, f, indent=2)

    print(f"\n  Results saved → {out_dir / slug}_*.csv/json")


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    configure_logging()
    from src.core.config import get_settings
    settings = get_settings()

    symbol    = args.symbol or settings.symbols[0]
    timeframe = args.timeframe or settings.primary_timeframe

    # Load data
    if args.from_csv:
        print(f"\n  Loading data from {args.from_csv}")
        df = load_from_csv(args.from_csv)
    else:
        df = await fetch_from_exchange(symbol, timeframe, args.candles)

    if len(df) < 200:
        print("  ERROR: Not enough data for backtesting (need at least 200 bars).")
        return

    print(f"  Dataset: {len(df)} bars")

    # Generate signals
    signals = generate_signals(
        df=df,
        warmup=args.warmup,
        window=args.window,
        min_confidence=args.min_confidence,
        atr_sl_mult=args.atr_sl,
        rr_ratio=args.rr,
        model_path=args.model_path,
    )

    n_signals = int((signals["direction"] != 0).sum())
    print(f"  Signals generated: {n_signals}  "
          f"(longs={int((signals['direction']==1).sum())}, "
          f"shorts={int((signals['direction']==-1).sum())})")

    if n_signals == 0:
        print("  No signals generated — try lowering --min-confidence.")
        return

    # Simulate
    print("\n  Running simulation ...")
    trades, equity_curve = simulate(
        df=df,
        signals=signals,
        initial_equity=args.equity,
        fee_pct=args.fee / 100.0,
        slippage_pct=args.slippage / 100.0,
        kelly_fraction=args.kelly,
        rr_ratio=args.rr,
        max_open=args.max_open,
    )

    # Metrics
    metrics = compute_metrics(trades, equity_curve, args.equity)
    print_report(metrics, symbol, timeframe)

    # Save
    save_results(trades, equity_curve, metrics, symbol, timeframe)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CryptoAgent backtester")

    # Data source
    p.add_argument("--symbol",       type=str,   default=None)
    p.add_argument("--timeframe",    type=str,   default=None)
    p.add_argument("--candles",      type=int,   default=3000,
                   help="Candles to fetch from exchange (default: 3000)")
    p.add_argument("--from-csv",     type=str,   default=None,
                   help="Load OHLCV from local CSV instead of fetching")

    # Signal params
    p.add_argument("--warmup",       type=int,   default=100,
                   help="Bars to skip at start (indicator warm-up, default: 100)")
    p.add_argument("--window",       type=int,   default=300,
                   help="Rolling window fed into signal engine (default: 300)")
    p.add_argument("--min-confidence", type=float, default=55.0,
                   help="Min ensemble confidence to generate a signal (default: 55)")
    p.add_argument("--atr-sl",       type=float, default=1.5,
                   help="ATR multiplier for stop-loss (default: 1.5)")
    p.add_argument("--rr",           type=float, default=2.0,
                   help="Reward:risk ratio for take-profit (default: 2.0)")

    # Simulation params
    p.add_argument("--equity",       type=float, default=10000.0,
                   help="Starting equity in USDC (default: 10000)")
    p.add_argument("--fee",          type=float, default=0.05,
                   help="Fee percentage per side (default: 0.05 = 0.05%%)")
    p.add_argument("--slippage",     type=float, default=0.02,
                   help="Slippage percentage per fill (default: 0.02 = 0.02%%)")
    p.add_argument("--kelly",        type=float, default=0.25,
                   help="Fractional Kelly multiplier (default: 0.25)")
    p.add_argument("--max-open",     type=int,   default=1,
                   help="Max simultaneous open positions (default: 1)")
    p.add_argument("--model-path",   type=str,   default="models/lgbm_classifier.joblib",
                   help="Path to trained LightGBM model (default: models/lgbm_classifier.joblib)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
