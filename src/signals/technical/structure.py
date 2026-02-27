"""
Market Structure Analyzer
Detects institutional order flow concepts:
  - Swing Highs / Swing Lows
  - Break of Structure (BOS) — trend continuation
  - Change of Character (CHOCH) — potential reversal
  - Fair Value Gaps (FVG / imbalances)
  - Liquidity Sweeps (stop hunts above/below swing points)
  - Order Blocks (last bearish candle before bullish impulse, and vice versa)
  - Premium / Discount zones relative to daily range

All functions accept a pandas DataFrame with columns: open, high, low, close, volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class StructureType(str, Enum):
    BOS_BULL  = "bos_bull"    # bullish break of structure (higher high)
    BOS_BEAR  = "bos_bear"    # bearish break of structure (lower low)
    CHOCH_BULL = "choch_bull" # bullish change of character (reversal up)
    CHOCH_BEAR = "choch_bear" # bearish change of character (reversal down)


class FVGType(str, Enum):
    BULL = "bull"   # bullish imbalance (gap up — demand zone)
    BEAR = "bear"   # bearish imbalance (gap down — supply zone)


class OrderBlockType(str, Enum):
    BULL = "bull"   # last bearish candle before strong up move
    BEAR = "bear"   # last bullish candle before strong down move


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str           # "high" or "low"
    is_major: bool = False


@dataclass
class StructureBreak:
    index: int          # bar where break occurred
    kind: StructureType
    broken_price: float # the swing level that was broken
    close_price: float  # close that confirmed the break


@dataclass
class FVG:
    start_idx: int      # bar 1 of the 3-bar sequence
    end_idx: int        # bar 3 of the 3-bar sequence
    top: float
    bottom: float
    kind: FVGType
    filled: bool = False
    fill_idx: Optional[int] = None


@dataclass
class OrderBlock:
    index: int          # candle index of the order block
    open: float
    high: float
    low: float
    close: float
    kind: OrderBlockType
    mitigated: bool = False
    mitigation_idx: Optional[int] = None


@dataclass
class LiquiditySweep:
    index: int              # bar where sweep occurred
    swept_price: float      # the swing level that was swept
    direction: str          # "high" (sweep above) or "low" (sweep below)
    reclaimed: bool = False # did price quickly reclaim back inside?


@dataclass
class MarketStructure:
    """Complete market structure snapshot for a single symbol/timeframe."""
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows:  list[SwingPoint] = field(default_factory=list)
    structure_breaks: list[StructureBreak] = field(default_factory=list)
    fvgs: list[FVG] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    liquidity_sweeps: list[LiquiditySweep] = field(default_factory=list)
    trend: str = "unknown"          # "bullish", "bearish", "ranging"
    last_bos: Optional[StructureBreak] = None
    last_choch: Optional[StructureBreak] = None


# ── Swing Point Detection ─────────────────────────────────────────────────────

def find_swing_highs(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[SwingPoint]:
    """
    Pivot-based swing high detection.
    A bar is a swing high if its high is the highest within [i-left, i+right].
    """
    highs = df["high"].values
    n = len(highs)
    points = []
    for i in range(left, n - right):
        window = highs[i - left: i + right + 1]
        if highs[i] == window.max() and np.sum(window == highs[i]) == 1:
            points.append(SwingPoint(index=i, price=float(highs[i]), kind="high"))
    return points


def find_swing_lows(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[SwingPoint]:
    """
    Pivot-based swing low detection.
    A bar is a swing low if its low is the lowest within [i-left, i+right].
    """
    lows = df["low"].values
    n = len(lows)
    points = []
    for i in range(left, n - right):
        window = lows[i - left: i + right + 1]
        if lows[i] == window.min() and np.sum(window == lows[i]) == 1:
            points.append(SwingPoint(index=i, price=float(lows[i]), kind="low"))
    return points


def classify_swing_significance(
    swing_highs: list[SwingPoint],
    swing_lows:  list[SwingPoint],
    major_threshold: float = 0.02,  # 2% move to be "major"
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Mark swings as major (significant structural pivots) vs minor.
    A major swing high/low has a larger price deviation from neighbours.
    """
    def _mark(points: list[SwingPoint]) -> list[SwingPoint]:
        if len(points) < 3:
            return points
        for i in range(1, len(points) - 1):
            prev_p = points[i - 1].price
            curr_p = points[i].price
            move = abs(curr_p - prev_p) / prev_p if prev_p else 0
            points[i].is_major = move >= major_threshold
        return points

    return _mark(swing_highs), _mark(swing_lows)


# ── Break of Structure & Change of Character ──────────────────────────────────

def detect_structure_breaks(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows:  list[SwingPoint],
) -> list[StructureBreak]:
    """
    Detect BOS (trend continuation) and CHOCH (reversal signal).

    Rules:
      Bullish BOS  — in an established uptrend, close > last swing high  →  BOS_BULL
      Bearish BOS  — in an established downtrend, close < last swing low →  BOS_BEAR
      Bullish CHOCH — in a downtrend, close > last swing high (first HH) →  CHOCH_BULL
      Bearish CHOCH — in an uptrend, close < last swing low  (first LL)  →  CHOCH_BEAR
    """
    if not swing_highs or not swing_lows:
        return []

    close = df["close"].values
    breaks = []

    # Build combined sorted pivot list to determine trend context
    all_pivots = sorted(
        [(p.index, p.price, "high") for p in swing_highs] +
        [(p.index, p.price, "low")  for p in swing_lows],
        key=lambda x: x[0],
    )

    # Simple trend state machine
    trend = "unknown"   # "up" | "down" | "unknown"
    last_hh = None      # last confirmed higher high price
    last_ll = None      # last confirmed lower low price
    prev_sh_price = None
    prev_sl_price = None

    # Index swing highs/lows by bar index for quick lookup
    sh_by_idx = {p.index: p for p in swing_highs}
    sl_by_idx = {p.index: p for p in swing_lows}

    # Sliding window: for each bar, check if close breaks nearest swing level
    sh_queue: list[SwingPoint] = []  # swing highs seen so far
    sl_queue: list[SwingPoint] = []  # swing lows seen so far

    sh_iter = iter(sorted(swing_highs, key=lambda x: x.index))
    sl_iter = iter(sorted(swing_lows,  key=lambda x: x.index))
    next_sh = next(sh_iter, None)
    next_sl = next(sl_iter, None)

    for i in range(len(close)):
        # Advance iterators
        while next_sh and next_sh.index <= i:
            sh_queue.append(next_sh)
            next_sh = next(sh_iter, None)
        while next_sl and next_sl.index <= i:
            sl_queue.append(next_sl)
            next_sl = next(sl_iter, None)

        if len(sh_queue) < 2 or len(sl_queue) < 2:
            continue

        ref_sh = sh_queue[-2]   # second-to-last swing high
        ref_sl = sl_queue[-2]   # second-to-last swing low

        c = close[i]

        if trend == "unknown":
            # Determine initial trend from last two pivots
            if sh_queue[-1].price > sh_queue[-2].price and sl_queue[-1].price > sl_queue[-2].price:
                trend = "up"
            elif sh_queue[-1].price < sh_queue[-2].price and sl_queue[-1].price < sl_queue[-2].price:
                trend = "down"

        elif trend == "up":
            # In uptrend: close below last swing low = CHOCH (potential reversal)
            if c < ref_sl.price:
                breaks.append(StructureBreak(
                    index=i,
                    kind=StructureType.CHOCH_BEAR,
                    broken_price=ref_sl.price,
                    close_price=float(c),
                ))
                trend = "down"
            # In uptrend: close above last swing high = BOS (continuation)
            elif c > ref_sh.price and i > ref_sh.index:
                breaks.append(StructureBreak(
                    index=i,
                    kind=StructureType.BOS_BULL,
                    broken_price=ref_sh.price,
                    close_price=float(c),
                ))

        elif trend == "down":
            # In downtrend: close above last swing high = CHOCH (potential reversal)
            if c > ref_sh.price:
                breaks.append(StructureBreak(
                    index=i,
                    kind=StructureType.CHOCH_BULL,
                    broken_price=ref_sh.price,
                    close_price=float(c),
                ))
                trend = "up"
            # In downtrend: close below last swing low = BOS (continuation)
            elif c < ref_sl.price and i > ref_sl.index:
                breaks.append(StructureBreak(
                    index=i,
                    kind=StructureType.BOS_BEAR,
                    broken_price=ref_sl.price,
                    close_price=float(c),
                ))

    return breaks


# ── Fair Value Gaps ────────────────────────────────────────────────────────────

def find_fvgs(df: pd.DataFrame, min_gap_pct: float = 0.001) -> list[FVG]:
    """
    Detect Fair Value Gaps (3-candle imbalances).

    Bullish FVG: low[i+2] > high[i]  — gap between candle 1 top and candle 3 bottom
    Bearish FVG: high[i+2] < low[i]  — gap between candle 1 bottom and candle 3 top

    Also marks if a subsequent candle has filled the gap.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(df)
    fvgs: list[FVG] = []

    for i in range(n - 2):
        gap_top    = lows[i + 2]
        gap_bottom = highs[i]

        # Bullish FVG
        if gap_top > gap_bottom:
            gap_size = (gap_top - gap_bottom) / gap_bottom
            if gap_size >= min_gap_pct:
                fvg = FVG(
                    start_idx=i,
                    end_idx=i + 2,
                    top=float(gap_top),
                    bottom=float(gap_bottom),
                    kind=FVGType.BULL,
                )
                # Check if later filled
                for j in range(i + 3, n):
                    if lows[j] <= fvg.bottom:
                        fvg.filled = True
                        fvg.fill_idx = j
                        break
                fvgs.append(fvg)

        # Bearish FVG
        gap_top2    = lows[i]
        gap_bottom2 = highs[i + 2]
        if gap_bottom2 < gap_top2:
            gap_size = (gap_top2 - gap_bottom2) / gap_top2
            if gap_size >= min_gap_pct:
                fvg = FVG(
                    start_idx=i,
                    end_idx=i + 2,
                    top=float(gap_top2),
                    bottom=float(gap_bottom2),
                    kind=FVGType.BEAR,
                )
                # Check if later filled
                for j in range(i + 3, n):
                    if highs[j] >= fvg.top:
                        fvg.filled = True
                        fvg.fill_idx = j
                        break
                fvgs.append(fvg)

    return fvgs


def get_active_fvgs(fvgs: list[FVG], current_idx: int) -> list[FVG]:
    """Return FVGs that have formed before current_idx and are not yet filled."""
    return [f for f in fvgs if f.end_idx < current_idx and not f.filled]


# ── Order Blocks ──────────────────────────────────────────────────────────────

def find_order_blocks(
    df: pd.DataFrame,
    impulse_threshold: float = 0.005,  # 0.5% minimum impulse to qualify
    lookback: int = 5,
) -> list[OrderBlock]:
    """
    Detect Order Blocks — the last opposing candle before a strong impulsive move.

    Bullish OB: last bearish candle before a strong up-impulse
    Bearish OB: last bullish candle before a strong down-impulse

    An impulse is defined as a move that breaks structure (creates new swing level)
    within `lookback` bars.
    """
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(df)
    obs: list[OrderBlock] = []

    for i in range(1, n - lookback):
        # Look for an impulsive move in the next `lookback` bars
        future_high = highs[i + 1: i + lookback + 1].max()
        future_low  = lows[i + 1:  i + lookback + 1].min()

        up_impulse   = (future_high - closes[i]) / closes[i] if closes[i] else 0
        down_impulse = (closes[i] - future_low) / closes[i] if closes[i] else 0

        # Bullish OB: current bar is bearish, followed by strong up move
        if closes[i] < opens[i] and up_impulse >= impulse_threshold:
            ob = OrderBlock(
                index=i,
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                kind=OrderBlockType.BULL,
            )
            # Check mitigation (price returns into OB zone)
            for j in range(i + lookback + 1, n):
                if lows[j] <= ob.high and closes[j] >= ob.low:
                    ob.mitigated = True
                    ob.mitigation_idx = j
                    break
            obs.append(ob)

        # Bearish OB: current bar is bullish, followed by strong down move
        elif closes[i] > opens[i] and down_impulse >= impulse_threshold:
            ob = OrderBlock(
                index=i,
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                kind=OrderBlockType.BEAR,
            )
            # Check mitigation
            for j in range(i + lookback + 1, n):
                if highs[j] >= ob.low and closes[j] <= ob.high:
                    ob.mitigated = True
                    ob.mitigation_idx = j
                    break
            obs.append(ob)

    return obs


def get_active_order_blocks(
    obs: list[OrderBlock],
    current_idx: int,
    price: float,
    proximity_pct: float = 0.01,
) -> list[OrderBlock]:
    """
    Return unmitigated order blocks near current price (within proximity_pct).
    """
    active = []
    for ob in obs:
        if ob.index >= current_idx or ob.mitigated:
            continue
        mid = (ob.high + ob.low) / 2
        dist = abs(price - mid) / mid if mid else 1
        if dist <= proximity_pct:
            active.append(ob)
    return active


# ── Liquidity Sweeps ──────────────────────────────────────────────────────────

def find_liquidity_sweeps(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows:  list[SwingPoint],
    reclaim_bars: int = 3,
) -> list[LiquiditySweep]:
    """
    Detect liquidity sweeps — wicks beyond swing points that quickly reverse.

    A sweep of a high: the bar's high exceeds a prior swing high but closes below it.
    A sweep of a low:  the bar's low  drops below a prior swing low  but closes above it.
    A reclaim is confirmed if within `reclaim_bars` bars price moves back beyond the swept level.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(df)
    sweeps: list[LiquiditySweep] = []

    sh_prices = {p.index: p.price for p in swing_highs}
    sl_prices = {p.index: p.price for p in swing_lows}

    for i in range(1, n):
        c = closes[i]
        h = highs[i]
        l = lows[i]

        # Check sweep of each swing high that formed before this bar
        for idx, price in sh_prices.items():
            if idx >= i:
                continue
            if h > price and c < price:
                sweep = LiquiditySweep(
                    index=i,
                    swept_price=price,
                    direction="high",
                )
                # Check for reclaim (bullish reversal — price closes above swept level)
                for j in range(i + 1, min(i + reclaim_bars + 1, n)):
                    if closes[j] > price:
                        sweep.reclaimed = True
                        break
                sweeps.append(sweep)

        # Check sweep of each swing low that formed before this bar
        for idx, price in sl_prices.items():
            if idx >= i:
                continue
            if l < price and c > price:
                sweep = LiquiditySweep(
                    index=i,
                    swept_price=price,
                    direction="low",
                )
                # Check for reclaim (bearish reversal — price closes below swept level)
                for j in range(i + 1, min(i + reclaim_bars + 1, n)):
                    if closes[j] < price:
                        sweep.reclaimed = True
                        break
                sweeps.append(sweep)

    return sweeps


# ── Premium / Discount Zones ──────────────────────────────────────────────────

def premium_discount(
    df: pd.DataFrame,
    lookback: int = 50,
) -> pd.Series:
    """
    Classify each bar as premium (above equilibrium) or discount (below).
    Returns a Series: +1 = premium zone, -1 = discount zone, 0 = equilibrium.

    Equilibrium = 50% of the [lookback]-bar range.
    Premium zone  = above 75% of range (potential short area).
    Discount zone = below 25% of range (potential long area).
    """
    highs = df["high"].rolling(lookback).max()
    lows  = df["low"].rolling(lookback).min()
    rng   = highs - lows
    eq    = lows + rng * 0.5
    upper = lows + rng * 0.75
    lower = lows + rng * 0.25

    zone = pd.Series(0, index=df.index, dtype=int)
    zone[df["close"] > upper] = 1    # premium
    zone[df["close"] < lower] = -1   # discount
    return zone


# ── Trend Determination ───────────────────────────────────────────────────────

def determine_trend(
    swing_highs: list[SwingPoint],
    swing_lows:  list[SwingPoint],
    min_pivots: int = 3,
) -> str:
    """
    Determine overall trend from swing structure.
    Bullish:  HH + HL pattern in recent pivots
    Bearish:  LH + LL pattern in recent pivots
    Ranging:  mixed / insufficient pivots
    """
    if len(swing_highs) < min_pivots or len(swing_lows) < min_pivots:
        return "ranging"

    recent_highs = sorted(swing_highs, key=lambda x: x.index)[-min_pivots:]
    recent_lows  = sorted(swing_lows,  key=lambda x: x.index)[-min_pivots:]

    hh = all(recent_highs[i].price > recent_highs[i-1].price for i in range(1, len(recent_highs)))
    hl = all(recent_lows[i].price  > recent_lows[i-1].price  for i in range(1, len(recent_lows)))
    lh = all(recent_highs[i].price < recent_highs[i-1].price for i in range(1, len(recent_highs)))
    ll = all(recent_lows[i].price  < recent_lows[i-1].price  for i in range(1, len(recent_lows)))

    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze(
    df: pd.DataFrame,
    swing_left: int = 3,
    swing_right: int = 3,
    fvg_min_gap_pct: float = 0.001,
    ob_impulse_threshold: float = 0.005,
    sweep_reclaim_bars: int = 3,
) -> MarketStructure:
    """
    Full market structure analysis on an OHLCV DataFrame.
    Returns a MarketStructure object with all detected structural elements.

    Typical usage:
        ms = analyze(df_4h)
        if ms.trend == "bullish" and ms.last_choch:
            ...
    """
    if len(df) < (swing_left + swing_right + 10):
        return MarketStructure()

    swing_highs = find_swing_highs(df, swing_left, swing_right)
    swing_lows  = find_swing_lows(df,  swing_left, swing_right)
    swing_highs, swing_lows = classify_swing_significance(swing_highs, swing_lows)

    structure_breaks = detect_structure_breaks(df, swing_highs, swing_lows)
    fvgs   = find_fvgs(df, fvg_min_gap_pct)
    obs    = find_order_blocks(df, ob_impulse_threshold)
    sweeps = find_liquidity_sweeps(df, swing_highs, swing_lows, sweep_reclaim_bars)
    trend  = determine_trend(swing_highs, swing_lows)

    last_bos   = next((b for b in reversed(structure_breaks)
                       if b.kind in (StructureType.BOS_BULL, StructureType.BOS_BEAR)), None)
    last_choch = next((b for b in reversed(structure_breaks)
                       if b.kind in (StructureType.CHOCH_BULL, StructureType.CHOCH_BEAR)), None)

    return MarketStructure(
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        structure_breaks=structure_breaks,
        fvgs=fvgs,
        order_blocks=obs,
        liquidity_sweeps=sweeps,
        trend=trend,
        last_bos=last_bos,
        last_choch=last_choch,
    )


def to_signal_series(ms: MarketStructure, length: int) -> pd.Series:
    """
    Encode the latest structure bias as a numeric signal for the ensemble.
    Returns a scalar: +1 (bullish structure), -1 (bearish), 0 (neutral/ranging).
    """
    if ms.trend == "bullish":
        base = 1
    elif ms.trend == "bearish":
        base = -1
    else:
        base = 0

    # Strengthen signal if last event confirms trend
    if ms.last_choch:
        if ms.last_choch.kind == StructureType.CHOCH_BULL:
            base = max(base, 1)
        elif ms.last_choch.kind == StructureType.CHOCH_BEAR:
            base = min(base, -1)

    return base
