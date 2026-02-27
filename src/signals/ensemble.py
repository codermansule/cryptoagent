"""
Signal Ensemble & Confidence Scorer

Aggregates signals from multiple sources into a single directional decision
with a confidence score [0–100].

Signal sources (weighted):
  1. Technical indicators  — EMA cross, Supertrend, RSI, MACD, StochRSI
  2. Market structure      — BOS/CHOCH, FVG proximity, order block alignment
  3. ML classifier         — LightGBM directional probability
  4. Volume analysis       — CVD trend, OBV direction, VWAP position
  5. Sentiment analysis    — Fear & Greed, news, Twitter, Reddit
  6. On-chain data         — Funding rates, OI, liquidations, BTC dominance

Final output:
  signal:     +1 (long) | -1 (short) | 0 (no trade)
  confidence: 0–100
  breakdown:  dict of per-source contributions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.technical.indicators import compute_all
from src.signals.technical.structure  import analyze as analyze_structure, MarketStructure, StructureType
from src.decision.regime import Regime, detect_regime, regime_multipliers

logger = logging.getLogger(__name__)


# ── Signal result container ───────────────────────────────────────────────────

@dataclass
class EnsembleSignal:
    symbol:      str
    timeframe:   str
    direction:   int            # +1 long | -1 short | 0 flat
    confidence:  float          # 0–100
    regime:      str = "unknown"
    breakdown:   dict = field(default_factory=dict)
    raw_scores:  dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """True if confidence exceeds the minimum threshold."""
        return abs(self.direction) == 1 and self.confidence >= 55.0


# ── Per-source signal computers ───────────────────────────────────────────────

def _technical_score(ind: pd.DataFrame) -> tuple[float, dict]:
    """
    Compute a directional score from technical indicator values on the last bar.
    Returns (net_score, breakdown) where net_score is in [-1, +1].
    """
    row = ind.iloc[-1]
    scores = {}

    # EMA cross: +1 bull cross, -1 bear cross, else trend bias
    ema_score = float(row.get("ema_cross", 0))
    if ema_score == 0:
        # Bias from EMA alignment
        ema9, ema21, ema50, ema200 = (
            row.get("ema9", np.nan), row.get("ema21", np.nan),
            row.get("ema50", np.nan), row.get("ema200", np.nan),
        )
        close = float(ind["close"].iloc[-1]) if "close" in ind else np.nan
        if not np.isnan(close) and not np.isnan(ema200):
            if close > ema50 > ema200:
                ema_score = 0.4
            elif close < ema50 < ema200:
                ema_score = -0.4
    scores["ema"] = ema_score

    # Supertrend direction
    st_dir = float(row.get("supertrend_dir", 0))
    scores["supertrend"] = np.clip(st_dir, -1, 1)

    # RSI — overbought/oversold + mid-zone bias
    rsi = float(row.get("rsi14", 50))
    if rsi > 70:
        rsi_score = -0.5   # overbought → lean short
    elif rsi < 30:
        rsi_score = 0.5    # oversold → lean long
    elif rsi > 55:
        rsi_score = 0.3
    elif rsi < 45:
        rsi_score = -0.3
    else:
        rsi_score = 0.0
    scores["rsi"] = rsi_score

    # MACD histogram direction
    hist = float(row.get("macd_hist", 0))
    hist_norm = float(row.get("macd_hist_norm", 0)) if not np.isnan(row.get("macd_hist_norm", np.nan)) else 0
    scores["macd"] = np.clip(np.sign(hist) * min(abs(hist_norm) * 2, 1.0), -1, 1)

    # StochRSI — oversold/overbought
    k = float(row.get("srsi_k", 50))
    d = float(row.get("srsi_d", 50))
    if k < 20 and d < 20:
        scores["stochrsi"] = 0.5
    elif k > 80 and d > 80:
        scores["stochrsi"] = -0.5
    elif k > d:
        scores["stochrsi"] = 0.2
    elif k < d:
        scores["stochrsi"] = -0.2
    else:
        scores["stochrsi"] = 0.0

    # Bollinger Band position
    pct_b = float(row.get("bb_pct_b", 0.5))
    if not np.isnan(pct_b):
        if pct_b > 0.9:
            scores["bb"] = -0.4
        elif pct_b < 0.1:
            scores["bb"] = 0.4
        else:
            scores["bb"] = 0.0
    else:
        scores["bb"] = 0.0

    # ADX — amplify signal in trending markets
    adx_val = float(row.get("adx", 20))
    adx_mult = min(adx_val / 40.0, 1.0)   # 0→0.0, 40+→1.0

    weights = {"ema": 0.25, "supertrend": 0.25, "rsi": 0.15,
               "macd": 0.20, "stochrsi": 0.10, "bb": 0.05}
    net = sum(scores[k] * weights[k] for k in scores)
    net *= (0.7 + 0.3 * adx_mult)    # scale by trend strength

    return float(np.clip(net, -1, 1)), scores


def _structure_score(ms: MarketStructure) -> tuple[float, dict]:
    """
    Derive a score from market structure analysis.
    Returns (net_score, breakdown) in [-1, +1].
    """
    scores = {}

    # Trend direction
    trend_score = {"bullish": 1.0, "bearish": -1.0, "ranging": 0.0}.get(ms.trend, 0.0)
    scores["trend"] = trend_score

    # Last CHoCH — strong reversal signal
    if ms.last_choch:
        choch_score = 1.0 if ms.last_choch.kind == StructureType.CHOCH_BULL else -1.0
        scores["choch"] = choch_score
    else:
        scores["choch"] = 0.0

    # Last BOS — continuation signal
    if ms.last_bos:
        bos_score = 0.5 if ms.last_bos.kind == StructureType.BOS_BULL else -0.5
        scores["bos"] = bos_score
    else:
        scores["bos"] = 0.0

    # Recent liquidity sweep with reclaim — high-probability reversal
    recent_sweeps = [s for s in ms.liquidity_sweeps if s.reclaimed]
    if recent_sweeps:
        last_sweep = recent_sweeps[-1]
        sweep_score = 0.6 if last_sweep.direction == "low" else -0.6
        scores["liq_sweep"] = sweep_score
    else:
        scores["liq_sweep"] = 0.0

    # Unfilled FVGs nearby (gap-fill magnets)
    unfilled = [f for f in ms.fvgs if not f.filled]
    fvg_score = 0.0
    if unfilled:
        last_fvg = unfilled[-1]
        fvg_score = 0.3 if last_fvg.kind.value == "bull" else -0.3
    scores["fvg"] = fvg_score

    weights = {"trend": 0.30, "choch": 0.30, "bos": 0.20,
               "liq_sweep": 0.15, "fvg": 0.05}
    net = sum(scores[k] * weights[k] for k in scores)

    return float(np.clip(net, -1, 1)), scores


def _volume_score(ind: pd.DataFrame) -> tuple[float, dict]:
    """
    Compute directional score from volume-based indicators.
    """
    row = ind.iloc[-1]
    scores = {}

    # VWAP position
    vwap_dist = float(row.get("vwap_dist", 0)) if not np.isnan(row.get("vwap_dist", np.nan)) else 0
    scores["vwap"] = np.clip(np.sign(vwap_dist) * min(abs(vwap_dist) * 20, 1.0), -1, 1)

    # CVD direction (recent delta)
    cvd_norm = float(row.get("cvd_norm", 0)) if not np.isnan(row.get("cvd_norm", np.nan)) else 0
    scores["cvd"] = np.clip(np.sign(cvd_norm) * min(abs(cvd_norm) * 2, 1.0), -1, 1)

    # OBV direction
    obv_diff = float(row.get("obv_diff", 0)) if not np.isnan(row.get("obv_diff", np.nan)) else 0
    scores["obv"] = np.clip(np.sign(obv_diff), -1, 1)

    # Volume ratio (high volume confirms signals)
    vol_ratio = float(row.get("vol_ratio", 1.0)) if not np.isnan(row.get("vol_ratio", np.nan)) else 1.0
    vol_mult  = min(vol_ratio / 1.5, 1.5)

    weights = {"vwap": 0.35, "cvd": 0.40, "obv": 0.25}
    net = sum(scores[k] * weights[k] for k in scores) * vol_mult

    return float(np.clip(net, -1, 1)), scores


def _sentiment_score(sentiment_data: dict) -> tuple[float, dict]:
    """
    Compute directional score from sentiment data.
    Returns (net_score, breakdown) in [-1, +1].
    """
    scores = {}
    
    fear_greed = sentiment_data.get("fear_greed", 50)
    if fear_greed:
        fg_norm = fear_greed / 100.0
        scores["fear_greed"] = (fg_norm - 0.5) * 2
    
    news_sent = sentiment_data.get("news_sentiment", 0)
    scores["news"] = news_sent
    
    twitter_sent = sentiment_data.get("twitter_sentiment", 0)
    scores["twitter"] = twitter_sent
    
    reddit_sent = sentiment_data.get("reddit_sentiment", 0)
    scores["reddit"] = reddit_sent
    
    weights = {"fear_greed": 0.35, "news": 0.25, "twitter": 0.20, "reddit": 0.20}
    net = sum(scores.get(k, 0) * weights[k] for k in weights)
    
    return float(np.clip(net, -1, 1)), scores


def _onchain_score(onchain_data: dict) -> tuple[float, dict]:
    """
    Compute directional score from on-chain/market data.
    Returns (net_score, breakdown) in [-1, +1].
    """
    scores = {}
    
    funding_diff = onchain_data.get("funding_diff", 0)
    scores["funding"] = np.clip(funding_diff * 100, -1, 1)
    
    oi_change = onchain_data.get("oi_change", 0)
    scores["oi"] = np.clip(oi_change / 100, -1, 1)
    
    liq_dominance = onchain_data.get("liq_dominance", 0)
    scores["liquidations"] = np.clip(-liq_dominance, -1, 1)
    
    btc_dom = onchain_data.get("btc_dominance", 50)
    scores["btc_dominance"] = np.clip((btc_dom - 55) / 50, -1, 1)
    
    alt_season = onchain_data.get("altcoin_season", 50)
    scores["altcoin_season"] = np.clip((alt_season - 50) / 50, -1, 1)
    
    weights = {"funding": 0.25, "oi": 0.20, "liquidations": 0.25, 
               "btc_dominance": 0.15, "altcoin_season": 0.15}
    net = sum(scores.get(k, 0) * weights[k] for k in weights)
    
    return float(np.clip(net, -1, 1)), scores


# ── Main ensemble ─────────────────────────────────────────────────────────────

def compute_ensemble(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    ml_signal: int = 0,
    ml_confidence: float = 0.0,
    min_confidence: float = 55.0,
    weights: Optional[dict] = None,
    sentiment_data: Optional[dict] = None,
    onchain_data: Optional[dict] = None,
) -> EnsembleSignal:
    """
    Compute the ensemble signal for a symbol/timeframe.

    Args:
        df:             OHLCV DataFrame (recent history, ~200+ bars)
        symbol:         Trading pair (e.g. "BTC-USDC")
        timeframe:      Timeframe string (e.g. "15m")
        ml_signal:      Pre-computed ML classifier signal {-1, 0, +1}
        ml_confidence:  ML classifier probability [0.0, 1.0]
        min_confidence: Minimum ensemble score (0–100) to generate a signal
        weights:        Override default source weights
        sentiment_data:  Optional dict with sentiment features (fear_greed, news_sentiment, etc.)
        onchain_data:   Optional dict with on-chain data (funding_diff, oi_change, etc.)

    Returns:
        EnsembleSignal with direction, confidence, and breakdown
    """
    if len(df) < 50:
        return EnsembleSignal(symbol, timeframe, 0, 0.0)

    # Default source weights (expanded to include sentiment & on-chain)
    w = weights or {
        "technical": 0.25,
        "structure":  0.20,
        "ml":         0.25,
        "volume":     0.10,
        "sentiment":  0.10,
        "onchain":    0.10,
    }

    ind = compute_all(df)

    # ── Technical score
    tech_net, tech_breakdown = _technical_score(ind)

    # ── Market structure score
    ms = analyze_structure(df)
    struct_net, struct_breakdown = _structure_score(ms)

    # ── Volume score
    vol_net, vol_breakdown = _volume_score(ind)

    # ── ML score (normalise from probability to [-1,+1])
    ml_net = ml_signal * ml_confidence

    # ── Sentiment score
    if sentiment_data:
        sent_net, sent_breakdown = _sentiment_score(sentiment_data)
    else:
        sent_net, sent_breakdown = 0.0, {}

    # ── On-chain score
    if onchain_data:
        onchain_net, onchain_breakdown = _onchain_score(onchain_data)
    else:
        onchain_net, onchain_breakdown = 0.0, {}

    # ── Weighted combination
    raw_score = (
        tech_net     * w["technical"] +
        struct_net   * w["structure"] +
        ml_net       * w["ml"] +
        vol_net      * w["volume"] +
        sent_net     * w["sentiment"] +
        onchain_net  * w["onchain"]
    )

    # ── Regime adjustment
    regime = detect_regime(df)
    reg_mult = regime_multipliers(regime)
    conf_boost = reg_mult["confidence_boost"]

    # Add sentiment/on-chain confidence boost
    if sentiment_data and sentiment_data.get("fear_greed"):
        fg = sentiment_data["fear_greed"]
        if fg < 25 or fg > 75:
            conf_boost += 5
    
    if onchain_data and onchain_data.get("liq_extreme"):
        conf_boost += 5

    # Convert to 0–100 confidence scale
    confidence = abs(raw_score) * 100.0 + conf_boost
    confidence = float(np.clip(confidence, 0.0, 100.0))

    # Direction: sign of raw_score — must exceed ±0.05 to avoid noise signals
    if raw_score > 0.05:
        direction = 1
    elif raw_score < -0.05:
        direction = -1
    else:
        direction = 0

    # Enforce minimum confidence
    if confidence < min_confidence:
        direction = 0

    # Enforce regime-allowed sides
    allowed = reg_mult["allowed_sides"]
    if direction == 1  and "long"  not in allowed:
        direction = 0
    if direction == -1 and "short" not in allowed:
        direction = 0

    return EnsembleSignal(
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        confidence=confidence,
        regime=regime.value,
        breakdown={
            "technical": tech_breakdown,
            "structure": struct_breakdown,
            "volume":    vol_breakdown,
            "ml":        {"signal": ml_signal, "confidence": ml_confidence},
            "sentiment":  sent_breakdown,
            "onchain":   onchain_breakdown,
        },
        raw_scores={
            "technical": round(tech_net, 4),
            "structure": round(struct_net, 4),
            "volume":    round(vol_net, 4),
            "ml":        round(float(ml_net), 4),
            "sentiment": round(float(sent_net), 4),
            "onchain":   round(float(onchain_net), 4),
            "raw_combined": round(raw_score, 4),
        },
    )
