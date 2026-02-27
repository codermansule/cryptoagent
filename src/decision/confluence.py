"""
Multi-Timeframe Confluence Engine

Combines ensemble signals across multiple timeframes and a pair of symbols
to surface high-conviction trade setups where timeframes align.

HTF (higher timeframe) defines the trend bias.
LTF (lower timeframe) provides the entry trigger.
A trade is only considered when HTF and LTF signals agree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.signals.ensemble import EnsembleSignal

logger = logging.getLogger(__name__)

# Timeframe ordering for HTF/LTF classification
TF_ORDER = ["1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "1W"]


def _tf_rank(tf: str) -> int:
    try:
        return TF_ORDER.index(tf)
    except ValueError:
        return -1


@dataclass
class ConfluenceResult:
    symbol:        str
    direction:     int          # +1 | -1 | 0
    confidence:    float        # 0–100 (averaged + bonus)
    htf_aligned:   bool
    ltf_trigger:   bool
    signal_count:  int          # number of agreeing signals
    total_signals: int          # total signals evaluated
    signals:       list[EnsembleSignal] = field(default_factory=list)
    notes:         list[str]    = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (self.direction != 0
                and self.htf_aligned
                and self.ltf_trigger
                and self.confidence >= 55.0)


def compute_confluence(
    signals: list[EnsembleSignal],
    htf_timeframes: Optional[list[str]] = None,
    ltf_timeframes: Optional[list[str]] = None,
    min_agreeing: int = 2,
    htf_bonus: float = 10.0,
) -> Optional[ConfluenceResult]:
    """
    Compute multi-timeframe confluence from a list of EnsembleSignals
    for the same symbol across different timeframes.

    Args:
        signals:        List of EnsembleSignal (same symbol, different TFs)
        htf_timeframes: Higher-timeframe labels (trend bias). Defaults to 4H+
        ltf_timeframes: Lower-timeframe labels (entry trigger). Defaults to ≤1H
        min_agreeing:   Minimum agreeing signals to produce a result
        htf_bonus:      Confidence bonus when HTF aligns

    Returns:
        ConfluenceResult or None if no valid setup found
    """
    if not signals:
        return None

    symbol = signals[0].symbol
    htf_tf = htf_timeframes or ["4H", "1D", "1W"]
    ltf_tf = ltf_timeframes or ["5m", "15m", "30m", "1H"]

    directional = [s for s in signals if s.direction != 0]
    if not directional:
        return None

    # Determine majority direction
    longs  = [s for s in directional if s.direction == 1]
    shorts = [s for s in directional if s.direction == -1]

    if len(longs) > len(shorts):
        majority = 1
        agreeing = longs
    elif len(shorts) > len(longs):
        majority = -1
        agreeing = shorts
    else:
        return None   # tied

    if len(agreeing) < min_agreeing:
        return None

    # HTF alignment: any HTF signal agrees
    htf_signals = [s for s in agreeing if s.timeframe in htf_tf]
    htf_aligned = len(htf_signals) > 0

    # LTF trigger: any LTF signal agrees
    ltf_signals = [s for s in agreeing if s.timeframe in ltf_tf]
    ltf_trigger = len(ltf_signals) > 0

    # Confidence: average of agreeing signal confidences + HTF bonus
    avg_conf = sum(s.confidence for s in agreeing) / len(agreeing)
    bonus = htf_bonus if htf_aligned else 0.0
    final_conf = min(avg_conf + bonus, 100.0)

    notes = []
    if htf_aligned:
        notes.append(f"HTF confirms ({', '.join(s.timeframe for s in htf_signals)})")
    if ltf_trigger:
        notes.append(f"LTF trigger ({', '.join(s.timeframe for s in ltf_signals)})")

    result = ConfluenceResult(
        symbol=symbol,
        direction=majority,
        confidence=final_conf,
        htf_aligned=htf_aligned,
        ltf_trigger=ltf_trigger,
        signal_count=len(agreeing),
        total_signals=len(signals),
        signals=agreeing,
        notes=notes,
    )

    logger.debug(
        "Confluence [%s] dir=%+d conf=%.1f htf=%s ltf=%s (%d/%d agree)",
        symbol, majority, final_conf, htf_aligned, ltf_trigger,
        len(agreeing), len(signals),
    )

    return result
