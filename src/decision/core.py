"""
Decision Core
The brain of the trading agent.

Responsibilities:
  1. Receive new candle data from MarketFeedManager
  2. Run the ensemble signal computation
  3. Run multi-timeframe confluence check
  4. Apply risk filters (max positions, drawdown circuit breaker)
  5. Size the position using fractional Kelly
  6. Emit TradeDecision objects to the execution layer

TradeDecision schema:
  symbol, side, size_usdc, entry_price, stop_loss, take_profit, confidence, reason
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Awaitable

import numpy as np
import pandas as pd

from src.signals.ensemble   import EnsembleSignal, compute_ensemble
from src.decision.confluence import ConfluenceResult, compute_confluence
from src.decision.regime     import Regime, detect_regime, regime_multipliers
from src.signals.technical.indicators import atr, compute_all

logger = logging.getLogger(__name__)


# ── Trade Decision ─────────────────────────────────────────────────────────────

@dataclass
class TradeDecision:
    symbol:       str
    side:         str           # "long" | "short"
    size_usdc:    float         # notional in USDC
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    confidence:   float         # 0–100
    reason:       str           # human-readable explanation
    timeframe:    str = ""
    regime:       str = "unknown"
    signal_breakdown: dict = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        if self.side == "long":
            risk   = self.entry_price - self.stop_loss
            reward = self.take_profit  - self.entry_price
        else:
            risk   = self.stop_loss  - self.entry_price
            reward = self.entry_price - self.take_profit
        return reward / risk if risk > 0 else 0.0

    @property
    def stop_distance_pct(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price


# ── Decision Engine ────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Stateful decision engine. One instance per symbol set.

    Wired into the agent via:
        engine = DecisionEngine(config=..., portfolio_value_fn=..., open_positions_fn=...)
        market_feed.on_candle = engine.on_candle
    """

    def __init__(
        self,
        # Risk parameters
        min_confidence:          float = 60.0,
        max_open_positions:      int   = 5,
        max_portfolio_risk_pct:  float = 2.0,
        max_single_risk_pct:     float = 0.5,
        kelly_fraction:          float = 0.25,
        atr_sl_multiplier:       float = 2.5,   # stop = entry ± N * ATR
        rr_ratio:                float = 2.0,   # reward:risk ratio for TP
        adx_min_threshold:       float = 25.0,  # minimum ADX to confirm trend strength
        max_daily_drawdown_pct:  float = 5.0,
        # Callbacks
        portfolio_value_fn:      Optional[Callable[[], Awaitable[float]]] = None,
        open_positions_fn:       Optional[Callable[[], Awaitable[list]]]  = None,
        on_decision_fn:          Optional[Callable[[TradeDecision], Awaitable[None]]] = None,
        # ML model path (optional — if set, will load/train LightGBM)
        model_path:              Optional[str] = None,
    ):
        self.min_confidence         = min_confidence
        self.max_open_positions     = max_open_positions
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_single_risk_pct    = max_single_risk_pct
        self.kelly_fraction         = kelly_fraction
        self.atr_sl_multiplier      = atr_sl_multiplier
        self.rr_ratio               = rr_ratio
        self.adx_min_threshold      = adx_min_threshold
        self.max_daily_drawdown_pct = max_daily_drawdown_pct

        self._portfolio_value_fn = portfolio_value_fn
        self._open_positions_fn  = open_positions_fn
        self._on_decision_fn     = on_decision_fn

        # Per-symbol buffers: symbol → {timeframe: DataFrame}
        self._buffers: dict[str, dict[str, pd.DataFrame]] = {}
        # Per-symbol signals cache: symbol → {timeframe: EnsembleSignal}
        self._signal_cache: dict[str, dict[str, EnsembleSignal]] = {}

        # ML classifier (loaded lazily)
        self._classifier = None  # type: ignore[assignment]
        self._model_path = model_path
        self._classifier_lock = asyncio.Lock()

        # Circuit breaker state
        self._daily_pnl_pct: float = 0.0
        self._halted: bool = False

    # ── Candle ingestion ───────────────────────────────────────────────────

    async def on_candle(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """
        Called by MarketFeedManager each time a candle closes.
        Stores the data and triggers signal computation.
        """
        if self._halted:
            return

        # Update buffer
        if symbol not in self._buffers:
            self._buffers[symbol] = {}
        self._buffers[symbol][timeframe] = df

        # Run signal computation asynchronously
        asyncio.create_task(self._process(symbol, timeframe))

    # ── Signal computation ─────────────────────────────────────────────────

    async def _process(self, symbol: str, timeframe: str) -> None:
        try:
            df = self._buffers[symbol].get(timeframe)
            if df is None or len(df) < 50:
                return

            # ── ADX trend filter — skip signal computation in ranging markets ──
            # Compute indicators once; reused implicitly by compute_ensemble below.
            # ADX < adx_min_threshold means the market has no directional momentum.
            ind = compute_all(df)
            adx_val = float(ind["adx"].iloc[-1]) if "adx" in ind.columns else 0.0
            if not np.isnan(adx_val) and adx_val < self.adx_min_threshold:
                logger.debug(
                    "ADX filter blocked [%s/%s]: ADX=%.1f < %.0f (ranging market)",
                    symbol, timeframe, adx_val, self.adx_min_threshold,
                )
                return

            # ML prediction (run in executor to avoid blocking event loop)
            ml_signal, ml_conf = await self._get_ml_prediction(symbol, df)

            # Ensemble signal for this timeframe
            signal = compute_ensemble(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                ml_signal=ml_signal,
                ml_confidence=ml_conf,
                min_confidence=self.min_confidence,
            )

            # Cache signal
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
            self._signal_cache[symbol][timeframe] = signal

            logger.debug(
                "Signal [%s/%s] dir=%+d conf=%.1f regime=%s",
                symbol, timeframe, signal.direction, signal.confidence, signal.regime,
            )

            # Try confluence across all cached timeframes for this symbol
            all_signals = list(self._signal_cache[symbol].values())
            if len(all_signals) >= 2:
                await self._evaluate_confluence(symbol, all_signals)

        except Exception as exc:
            logger.exception("Error processing candle for %s/%s: %s", symbol, timeframe, exc)

    async def _evaluate_confluence(
        self,
        symbol: str,
        signals: list[EnsembleSignal],
    ) -> None:
        """Check for multi-timeframe confluence and emit a trade decision if valid."""
        # Treat 1H as HTF trend bias (alongside the usual 4H/1D/1W).
        # LTF entry triggers are 15m and below. This enables 15m+1H confluence.
        result = compute_confluence(
            signals,
            htf_timeframes=["1h", "4h", "1d", "1w"],
            ltf_timeframes=["5m", "15m", "30m"],
            min_agreeing=3,   # require all 3 subscribed TFs (15m+1h+4h) to agree
        )
        if result is None or not result.is_valid:
            return

        # Risk filters
        if not await self._passes_risk_filters(symbol, result):
            return

        # Build trade decision
        decision = await self._build_decision(symbol, result)
        if decision is None:
            return

        logger.info(
            "TRADE DECISION [%s] %s | conf=%.1f | SL=%.4f TP=%.4f | RR=%.2f | %s",
            symbol, decision.side.upper(), decision.confidence,
            decision.stop_loss, decision.take_profit, decision.risk_reward,
            decision.reason,
        )

        if self._on_decision_fn:
            await self._on_decision_fn(decision)

    # ── ML prediction ──────────────────────────────────────────────────────

    async def _get_ml_prediction(
        self, symbol: str, df: pd.DataFrame
    ) -> tuple[int, float]:
        """Run the ML classifier prediction. Returns (signal, confidence)."""
        if self._model_path is None:
            return 0, 0.0
        try:
            async with self._classifier_lock:
                if self._classifier is None:
                    from src.signals.ml_models.classifier import load_or_train
                    loop = asyncio.get_event_loop()
                    model_path_str = self._model_path if self._model_path else "models/lgbm_classifier.joblib"
                    self._classifier = await loop.run_in_executor(
                        None,
                        lambda: load_or_train(df, model_path=model_path_str),
                    )
            if self._classifier is None:
                return 0, 0.0
            loop = asyncio.get_event_loop()
            classifier = self._classifier
            signal, conf = await loop.run_in_executor(
                None, lambda: classifier.predict(df)
            )
            return signal, conf
        except Exception as exc:
            logger.warning("ML prediction failed for %s: %s", symbol, exc)
            return 0, 0.0

    # ── Risk filters ───────────────────────────────────────────────────────

    async def _passes_risk_filters(
        self, symbol: str, result: ConfluenceResult
    ) -> bool:
        """Return True if all risk filters pass."""
        # Circuit breaker
        if self._halted:
            logger.warning("Circuit breaker active — no new trades.")
            return False

        # Daily drawdown limit
        if self._daily_pnl_pct <= -self.max_daily_drawdown_pct:
            self._halted = True
            logger.warning(
                "Daily drawdown limit reached (%.2f%%). Trading halted.",
                self._daily_pnl_pct,
            )
            return False

        # Max open positions
        if self._open_positions_fn:
            positions = await self._open_positions_fn()
            if len(positions) >= self.max_open_positions:
                logger.debug("Max open positions (%d) reached.", self.max_open_positions)
                return False
            # Already in this symbol? (works with Position dataclasses and dicts)
            open_symbols = []
            for p in positions:
                sym = (getattr(p, "symbol", None)
                       or (p.get("symbol") if hasattr(p, "get") else None))
                if sym:
                    open_symbols.append(sym)
            if symbol in open_symbols:
                return False

        return True

    # ── Position sizing and trade construction ─────────────────────────────

    async def _build_decision(
        self, symbol: str, result: ConfluenceResult
    ) -> Optional[TradeDecision]:
        """Build a TradeDecision from a ConfluenceResult."""
        # Get latest bar data for ATR-based SL
        tf_signals = {s.timeframe: s for s in result.signals}
        # Use primary timeframe data for price/ATR
        sorted_tfs = sorted(tf_signals.keys())
        primary_tf: str = sorted_tfs[0] if sorted_tfs else ""
        if not primary_tf:
            return None
        buffers = self._buffers.get(symbol, {})
        df = buffers.get(primary_tf)
        if df is None or len(df) < 20:
            return None

        entry_price = float(df["close"].iloc[-1])
        atr_val     = float(atr(df, 14).iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            return None

        side = "long" if result.direction == 1 else "short"

        if side == "long":
            stop_loss   = entry_price - self.atr_sl_multiplier * atr_val
            take_profit = entry_price + self.rr_ratio * self.atr_sl_multiplier * atr_val
        else:
            stop_loss   = entry_price + self.atr_sl_multiplier * atr_val
            take_profit = entry_price - self.rr_ratio * self.atr_sl_multiplier * atr_val

        # Position sizing: fractional Kelly
        portfolio_value = 10_000.0   # fallback
        if self._portfolio_value_fn:
            try:
                portfolio_value = await self._portfolio_value_fn()
            except Exception:
                pass

        risk_per_trade = portfolio_value * (self.max_single_risk_pct / 100.0)
        stop_distance  = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            return None

        # Kelly-scaled notional
        win_rate = result.confidence / 100.0
        kelly    = win_rate - (1 - win_rate) / self.rr_ratio
        kelly    = max(0.0, min(kelly, 0.5))   # cap at 50% and floor at 0
        kelly   *= self.kelly_fraction         # fractional Kelly (conservative)

        # Convert to notional size
        size_usdc = portfolio_value * kelly
        # Also cap by fixed risk per trade
        max_size  = (risk_per_trade / stop_distance) * entry_price
        size_usdc = min(size_usdc, max_size)
        size_usdc = max(size_usdc, 1.0)   # minimum $1

        reason = (
            f"MTF confluence ({result.signal_count}/{result.total_signals} TFs agree); "
            + "; ".join(result.notes)
        )

        return TradeDecision(
            symbol=symbol,
            side=side,
            size_usdc=round(size_usdc, 2),
            entry_price=entry_price,
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            confidence=result.confidence,
            reason=reason,
            timeframe=primary_tf or "",
            regime=result.signals[0].regime if result.signals else "unknown",
            signal_breakdown={s.timeframe: s.raw_scores for s in result.signals},
        )

    # ── Daily PnL update (called by execution layer) ───────────────────────

    def update_daily_pnl(self, pct_change: float) -> None:
        """Update the daily PnL tracker. Called by paper/live engine on fills."""
        self._daily_pnl_pct += pct_change

    def reset_daily_stats(self) -> None:
        """Reset daily counters (call at midnight UTC)."""
        self._daily_pnl_pct = 0.0
        self._halted = False
        logger.info("Daily stats reset.")
