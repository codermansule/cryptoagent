"""
Market Regime Detector
Classifies the current market condition to inform strategy selection and risk sizing.

Regimes:
  TRENDING_UP    — strong uptrend (ADX > threshold, price above EMAs)
  TRENDING_DOWN  — strong downtrend
  RANGING        — low ADX, price oscillating inside BB bands
  VOLATILE       — high ATR, rapid directionless moves
  BREAKOUT       — BB squeeze release with directional move
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from src.signals.technical.indicators import (
    adx, atr, bollinger_bands, bb_squeeze, ema
)


class Regime(str, Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING       = "ranging"
    VOLATILE      = "volatile"
    BREAKOUT      = "breakout"
    UNKNOWN       = "unknown"


def detect_regime(
    df: pd.DataFrame,
    adx_trend_threshold: float = 25.0,
    adx_strong_threshold: float = 40.0,
    vol_percentile: float = 80.0,
    lookback: int = 20,
) -> Regime:
    """
    Classify the current market regime using the last bar of df.

    Rules (evaluated in priority order):
      1. BREAKOUT   — BB squeeze just released (was squeezed, now expanding)
      2. VOLATILE   — ATR in top vol_percentile% of recent range
      3. TRENDING_UP   — ADX > threshold AND price > EMA50
      4. TRENDING_DOWN — ADX > threshold AND price < EMA50
      5. RANGING    — ADX < threshold, price inside middle BB band
      6. UNKNOWN    — insufficient data
    """
    if len(df) < 50:
        return Regime.UNKNOWN

    adx_df    = adx(df, 14)
    atr_val   = atr(df, 14)
    close_series = df["close"]
    bb        = bollinger_bands(close_series, 20)
    squeeze   = bb_squeeze(df)
    ema50     = ema(close_series, 50)

    adx_now   = float(adx_df["adx"].iloc[-1])
    atr_now   = float(atr_val.iloc[-1])
    price     = float(df["close"].iloc[-1])
    sq_now    = bool(squeeze.iloc[-1])
    sq_prev   = bool(squeeze.iloc[-2]) if len(squeeze) > 1 else False
    ema50_now = float(ema50.iloc[-1])
    bb_bw_now = float(bb["bandwidth"].iloc[-1])

    # ATR percentile over lookback window
    atr_window  = atr_val.iloc[-lookback:].dropna()
    atr_pct_now = float((atr_window <= atr_now).mean()) * 100 if len(atr_window) else 50.0

    # 1. Breakout: squeeze was active, now released with expansion
    if sq_prev and not sq_now:
        return Regime.BREAKOUT

    # 2. Volatile: ATR very high relative to recent history
    if atr_pct_now >= vol_percentile:
        return Regime.VOLATILE

    # 3-4. Trending (ADX based)
    if adx_now >= adx_trend_threshold:
        plus_di  = float(adx_df["plus_di"].iloc[-1])
        minus_di = float(adx_df["minus_di"].iloc[-1])
        if plus_di > minus_di and price > ema50_now:
            return Regime.TRENDING_UP
        elif minus_di > plus_di and price < ema50_now:
            return Regime.TRENDING_DOWN

    # 5. Ranging
    return Regime.RANGING


def regime_multipliers(regime: Regime) -> dict:
    """
    Return strategy multipliers for a given regime.

    position_size_mult: scale down position size in unfavourable regimes
    confidence_boost:   bonus added to signal confidence in this regime
    allowed_sides:      which trade directions are allowed
    """
    cfg = {
        Regime.TRENDING_UP: {
            "position_size_mult": 1.0,
            "confidence_boost":   5,
            "allowed_sides":      ["long"],
        },
        Regime.TRENDING_DOWN: {
            "position_size_mult": 1.0,
            "confidence_boost":   5,
            "allowed_sides":      ["short"],
        },
        Regime.RANGING: {
            "position_size_mult": 0.7,
            "confidence_boost":   0,
            "allowed_sides":      ["long", "short"],   # mean-reversion
        },
        Regime.VOLATILE: {
            "position_size_mult": 0.5,
            "confidence_boost":   -5,
            "allowed_sides":      ["long", "short"],
        },
        Regime.BREAKOUT: {
            "position_size_mult": 1.2,
            "confidence_boost":   10,
            "allowed_sides":      ["long", "short"],
        },
        Regime.UNKNOWN: {
            "position_size_mult": 0.5,
            "confidence_boost":   -10,
            "allowed_sides":      [],
        },
    }
    return cfg.get(regime, cfg[Regime.UNKNOWN])
