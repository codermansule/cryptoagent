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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Awaitable

import numpy as np
import pandas as pd

from src.signals.ensemble   import EnsembleSignal, compute_ensemble
from src.decision.confluence import ConfluenceResult, compute_confluence
from src.decision.regime     import Regime, detect_regime, regime_multipliers
from src.signals.technical.indicators import atr, compute_all
from src.data.feeds.sentiment_feed import SentimentFeed

logger = logging.getLogger(__name__)

_RUNTIME_CFG = Path(__file__).resolve().parents[2] / "config" / "runtime.json"


def _read_runtime_threshold(default: float) -> float:
    """Read dashboard-controlled threshold override from config/runtime.json."""
    try:
        if _RUNTIME_CFG.exists():
            import json
            data = json.loads(_RUNTIME_CFG.read_text())
            val = float(data.get("min_confidence_threshold", default))
            return max(1.0, min(100.0, val))
    except Exception:
        pass
    return default


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
        max_correlated_positions: int  = 2,     # max positions in same direction
        # Callbacks
        portfolio_value_fn:      Optional[Callable[[], Awaitable[float]]] = None,
        open_positions_fn:       Optional[Callable[[], Awaitable[list]]]  = None,
        on_decision_fn:          Optional[Callable[[TradeDecision], Awaitable[None]]] = None,
        # ML model path (optional — if set, will load/train LightGBM)
        model_path:              Optional[str] = None,
    ):
        self.min_confidence          = min_confidence
        self.max_open_positions      = max_open_positions
        self.max_portfolio_risk_pct  = max_portfolio_risk_pct
        self.max_single_risk_pct     = max_single_risk_pct
        self.kelly_fraction          = kelly_fraction
        self.atr_sl_multiplier       = atr_sl_multiplier
        self.rr_ratio                = rr_ratio
        self.adx_min_threshold       = adx_min_threshold
        self.max_daily_drawdown_pct  = max_daily_drawdown_pct
        self.max_correlated_positions = max_correlated_positions

        # Per-symbol parameter overrides {symbol: {min_confidence_threshold, atr_sl_multiplier, ...}}
        try:
            from src.core.config import get_settings
            self._symbol_overrides: dict[str, dict] = get_settings().symbol_overrides
        except Exception:
            self._symbol_overrides = {}

        self._portfolio_value_fn = portfolio_value_fn
        self._open_positions_fn  = open_positions_fn
        self._on_decision_fn     = on_decision_fn

        # Per-symbol buffers: symbol → {timeframe: DataFrame}
        self._buffers: dict[str, dict[str, pd.DataFrame]] = {}
        # Per-symbol signals cache: symbol → {timeframe: EnsembleSignal}
        self._signal_cache: dict[str, dict[str, EnsembleSignal]] = {}
        # Signal freshness: symbol → {timeframe: unix_timestamp_of_last_update}
        self._signal_ts: dict[str, dict[str, float]] = {}
        # Max signal age per TF before it's considered stale (seconds)
        self._signal_max_age = {"1m": 120, "3m": 360, "5m": 600, "15m": 1800,
                                 "1h": 7200, "4h": 18000, "1d": 90000}

        # Per-symbol ML classifiers (loaded lazily on first prediction for each symbol)
        self._model_path = model_path
        self._classifiers: dict[str, object] = {}
        self._classifier_locks: dict[str, asyncio.Lock] = {}

        # Per-symbol LSTM classifiers (loaded lazily alongside LightGBM)
        self._lstm_classifiers: dict[str, object] = {}
        self._lstm_classifier_locks: dict[str, asyncio.Lock] = {}

        # Live sentiment + on-chain data (refreshed every 5 min)
        self._sentiment_feed = SentimentFeed()

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

            # ML predictions — only on LTF entry timeframes where models were trained.
            # Calling LGBM/LSTM on 4h/1h data produces out-of-distribution predictions
            # (4h ATR is 16× larger than 15m ATR) and causes counter-trend noise.
            # HTF (4h, 1h) signals are driven purely by TA + structure + volume.
            # LSTM is 15m-only; _get_lstm_prediction returns (0, 0.0) for other TFs.
            if timeframe in ("1m", "3m", "5m", "15m"):
                lgbm_sig, lgbm_conf = await self._get_ml_prediction(symbol, timeframe, df)
                lstm_sig, lstm_conf = await self._get_lstm_prediction(symbol, timeframe, df)
                ml_signal, ml_conf = self._blend_ml(lgbm_sig, lgbm_conf, lstm_sig, lstm_conf)
            else:
                lgbm_sig, lgbm_conf = 0, 0.0
                lstm_sig, lstm_conf = 0, 0.0
                ml_signal, ml_conf  = 0, 0.0

            # Sentiment + on-chain data (cached, refreshed every 5 min)
            await self._sentiment_feed.refresh_if_stale()

            # Check for dashboard-controlled threshold override (runtime.json)
            global_threshold = _read_runtime_threshold(self.min_confidence)
            # Per-symbol override takes precedence over runtime.json global
            effective_threshold = float(
                self._symbol_overrides.get(symbol, {}).get(
                    "min_confidence_threshold", global_threshold
                )
            )

            # For HTF timeframes (1h/4h) use a fixed 20% direction threshold instead of
            # the per-symbol entry threshold. The HTF signal only provides trend BIAS;
            # the full per-symbol threshold is enforced later at confluence.is_valid.
            # Without this, high per-symbol thresholds (e.g. ETH=40%) zero out all 4h
            # signals below 40%, making htf_aligned permanently False → no trades ever.
            _HTF_DIRECTION_THRESHOLD = 20.0
            signal_min_conf = (
                _HTF_DIRECTION_THRESHOLD
                if timeframe in ("1h", "4h")
                else effective_threshold
            )

            # Ensemble signal for this timeframe (pass pre-computed ind to avoid double compute_all)
            signal = compute_ensemble(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                ml_signal=ml_signal,
                ml_confidence=ml_conf,
                min_confidence=signal_min_conf,
                sentiment_data=self._sentiment_feed.sentiment_data,
                onchain_data=self._sentiment_feed.onchain_data,
                indicators=ind,
            )

            # Cache signal with freshness timestamp
            if symbol not in self._signal_cache:
                self._signal_cache[symbol] = {}
                self._signal_ts[symbol] = {}
            self._signal_cache[symbol][timeframe] = signal
            self._signal_ts[symbol][timeframe] = time.time()

            raw_combined = signal.raw_scores.get("raw_combined", 0.0)
            if signal.direction != 0 or signal.confidence >= 20.0 or abs(raw_combined) >= 0.02:
                logger.info(
                    "Signal [%s/%s] dir=%+d conf=%.1f raw=%.3f adx=%.1f regime=%s "
                    "ml=(%+d,%.0f%%) lgbm=(%+d,%.0f%%) lstm=(%+d,%.0f%%) "
                    "scores: tech=%.2f struct=%.2f vol=%.2f sent=%.2f onchain=%.2f",
                    symbol, timeframe, signal.direction, signal.confidence, raw_combined, adx_val,
                    signal.regime, ml_signal, ml_conf * 100,
                    lgbm_sig, lgbm_conf * 100, lstm_sig, lstm_conf * 100,
                    signal.raw_scores.get("technical", 0.0),
                    signal.raw_scores.get("structure", 0.0),
                    signal.raw_scores.get("volume", 0.0),
                    signal.raw_scores.get("sentiment", 0.0),
                    signal.raw_scores.get("onchain", 0.0),
                )
            else:
                logger.debug(
                    "Signal [%s/%s] dir=%+d conf=%.1f raw=%.3f adx=%.1f regime=%s",
                    symbol, timeframe, signal.direction, signal.confidence, raw_combined, adx_val, signal.regime,
                )

            # Try confluence across all cached timeframes for this symbol
            # Filter out stale signals before evaluating confluence
            now = time.time()
            all_signals = [
                sig for tf, sig in self._signal_cache[symbol].items()
                if now - self._signal_ts.get(symbol, {}).get(tf, 0)
                   <= self._signal_max_age.get(tf, 7200)
            ]
            if len(all_signals) >= 2:
                await self._evaluate_confluence(symbol, all_signals, effective_threshold)

        except Exception as exc:
            logger.exception("Error processing candle for %s/%s: %s", symbol, timeframe, exc)

    async def _evaluate_confluence(
        self,
        symbol: str,
        signals: list[EnsembleSignal],
        min_confidence: float = 25.0,
    ) -> None:
        """Check for multi-timeframe confluence and emit a trade decision if valid."""
        # TF hierarchy: 1h defines trend bias (HTF), 15m is the entry trigger (LTF).
        # 4h is also HTF when available; 1h alone is sufficient for htf_aligned.
        # Require at least 2 agreeing signals — verified working in Phase 3.6.
        result = compute_confluence(
            signals,
            htf_timeframes=["1h", "4h", "1d", "1w"],
            ltf_timeframes=["1m", "3m", "5m", "15m", "30m"],
            min_agreeing=2,
            min_confidence=min_confidence,
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
        self, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> tuple[int, float]:
        """Run the per-symbol LGBM classifier. Returns (signal, confidence)."""
        if self._model_path is None:
            return 0, 0.0
        try:
            slug = symbol.lower().replace("-", "_")
            cache_key = f"{symbol}_{timeframe}"
            specific = Path(f"models/lgbm_{slug}_{timeframe}.joblib")
            model_path_str = str(specific) if specific.exists() else (
                self._model_path or "models/lgbm_classifier.joblib"
            )
            if cache_key not in self._classifier_locks:
                self._classifier_locks[cache_key] = asyncio.Lock()
            async with self._classifier_locks[cache_key]:
                if cache_key not in self._classifiers:
                    from src.signals.ml_models.classifier import load_or_train
                    loop = asyncio.get_event_loop()
                    self._classifiers[cache_key] = await loop.run_in_executor(
                        None,
                        lambda: load_or_train(df, model_path=model_path_str),
                    )
                    logger.info("Loaded LGBM model for %s: %s", cache_key, model_path_str)
            clf = self._classifiers.get(cache_key)
            if clf is None:
                return 0, 0.0
            loop = asyncio.get_event_loop()
            signal, conf = await loop.run_in_executor(None, lambda: clf.predict(df))
            return signal, conf
        except Exception as exc:
            logger.warning("ML prediction failed for %s: %s", symbol, exc)
            return 0, 0.0

    async def _get_lstm_prediction(
        self, symbol: str, timeframe: str, df: pd.DataFrame
    ) -> tuple[int, float]:
        """Run the per-symbol LSTM classifier. Returns (signal, confidence)."""
        # LSTM trained on 15m only — skip other timeframes
        if timeframe != "15m":
            return 0, 0.0
        try:
            slug = symbol.lower().replace("-", "_")
            specific = Path(f"models/lstm_{slug}_15m.pt")
            model_path_str = str(specific) if specific.exists() else "models/lstm.pt"
            if symbol not in self._lstm_classifier_locks:
                self._lstm_classifier_locks[symbol] = asyncio.Lock()
            async with self._lstm_classifier_locks[symbol]:
                if symbol not in self._lstm_classifiers:
                    from src.signals.ml_models.lstm import load_or_train as lstm_load_or_train
                    loop = asyncio.get_event_loop()
                    self._lstm_classifiers[symbol] = await loop.run_in_executor(
                        None,
                        lambda: lstm_load_or_train(df, model_path=model_path_str),
                    )
                    logger.info("Loaded LSTM model for %s: %s", symbol, model_path_str)
            lstm_clf = self._lstm_classifiers.get(symbol)
            if lstm_clf is None:
                return 0, 0.0
            loop = asyncio.get_event_loop()
            signal, conf = await loop.run_in_executor(None, lambda: lstm_clf.predict(df))
            return signal, conf
        except Exception as exc:
            logger.debug("LSTM prediction skipped for %s: %s", symbol, exc)
            return 0, 0.0

    @staticmethod
    def _blend_ml(
        lgbm_sig: int, lgbm_conf: float,
        lstm_sig: int, lstm_conf: float,
    ) -> tuple[int, float]:
        """
        Blend LightGBM and LSTM predictions.
        - Both agree:    return that direction, average confidence
        - One is flat:   return the non-flat signal at 70% confidence weight
        - Disagree:      cancel out → (0, 0)
        """
        if lgbm_sig != 0 and lstm_sig != 0:
            if lgbm_sig == lstm_sig:
                return lgbm_sig, (lgbm_conf + lstm_conf) / 2.0
            else:
                return 0, 0.0   # models disagree — no ML contribution
        elif lgbm_sig != 0:
            return lgbm_sig, lgbm_conf * 0.7
        elif lstm_sig != 0:
            return lstm_sig, lstm_conf * 0.7
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

            # Correlation guard: cap same-direction positions to avoid concentration
            # e.g. BTC/ETH/SOL all SHORT simultaneously = overexposed to one move
            proposed_side = "long" if result.direction == 1 else "short"
            same_dir = 0
            for p in positions:
                p_side = (getattr(p, "side", None)
                          or (p.get("side") if hasattr(p, "get") else None))
                if p_side is None:
                    continue
                p_side_str = p_side.value if hasattr(p_side, "value") else str(p_side).lower()
                if p_side_str == proposed_side:
                    same_dir += 1
            if same_dir >= self.max_correlated_positions:
                logger.debug(
                    "Correlation guard blocked [%s] %s: %d/%d same-direction positions open.",
                    symbol, proposed_side.upper(), same_dir, self.max_correlated_positions,
                )
                return False

        # ── Funding rate pre-filter ────────────────────────────────────────
        # Skip entries when perpetual funding is extreme — signals crowded positioning
        # that is prone to a flush (longs) or short squeeze (shorts).
        #   funding_rate > +0.1%  (0.001):  longs heavily crowded → skip long entry
        #   funding_rate < -0.1%  (-0.001): shorts heavily crowded → skip short entry
        onchain = self._sentiment_feed.onchain_data
        funding_rate = float(onchain.get("funding_rate", 0.0))
        if funding_rate != 0.0:
            if result.direction == 1 and funding_rate > 0.001:
                logger.debug(
                    "Funding pre-filter: blocked LONG [%s] funding=%.4f%% (>0.1%%)",
                    symbol, funding_rate * 100,
                )
                return False
            if result.direction == -1 and funding_rate < -0.001:
                logger.debug(
                    "Funding pre-filter: blocked SHORT [%s] funding=%.4f%% (<-0.1%%)",
                    symbol, funding_rate * 100,
                )
                return False

        # ── OI trend filter ───────────────────────────────────────────────
        # Sharp OI collapse (>5% drop in 5 min) while taking a directional trade
        # signals forced liquidations — do not enter into a cascade.
        oi_change = float(onchain.get("oi_change", 0.0))
        if abs(oi_change) > 5.0:
            logger.debug(
                "OI collapse filter: blocked [%s] oi_change=%.2f%%",
                symbol, oi_change,
            )
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

        # Use per-symbol ATR multiplier if configured
        sym_ovr     = self._symbol_overrides.get(symbol, {})
        atr_mult    = float(sym_ovr.get("atr_sl_multiplier", self.atr_sl_multiplier))
        rr          = float(sym_ovr.get("rr_ratio", self.rr_ratio))

        if side == "long":
            stop_loss   = entry_price - atr_mult * atr_val
            take_profit = entry_price + rr * atr_mult * atr_val
        else:
            stop_loss   = entry_price + atr_mult * atr_val
            take_profit = entry_price - rr * atr_mult * atr_val

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

        # Kelly-scaled notional (uses per-symbol rr if configured)
        win_rate = result.confidence / 100.0
        kelly    = win_rate - (1 - win_rate) / rr
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
