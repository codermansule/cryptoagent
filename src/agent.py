"""
CryptoAgent — Main Orchestrator
Entry point: python -m src.agent
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.data.feeds.market_feed import MarketFeedManager
from src.data.storage.schema import Database
from src.exchanges.blofin import BloFinAdapter
from src.execution.paper import PaperEngine
from src.execution.live import LiveEngine
from src.decision.core import DecisionEngine, TradeDecision
from src.monitoring.alerts import get_alerter

logger = get_logger(__name__)


class CryptoAgent:
    """
    Top-level agent that wires together all subsystems:
      Exchange → MarketFeed → Signal Engine → Decision Core → Execution
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._running = False
        # Per-symbol SL cooldown: symbol -> unix timestamp of last SL hit
        # Prevents re-entry for SL_COOLDOWN_SECS after a stop-loss
        self._sl_cooldown: dict[str, float] = {}
        self._SL_COOLDOWN_SECS = 2700  # 45 minutes — prevents re-entry after SL in same direction
        self._sl_hit_count: dict[str, int] = {}         # consecutive SL hits per symbol
        self._adaptive_threshold: dict[str, float] = {} # symbol -> elevated conf threshold
        self._adaptive_threshold_until: dict[str, float] = {}  # symbol -> expiry unix ts

        # Core subsystems
        self._exchange = BloFinAdapter()
        self._db = Database()
        self._feed = MarketFeedManager(self._exchange, self._db)
        self._paper = PaperEngine() if self._settings.is_paper else None
        # Live engine (always instantiated; only executes when mode == 'live')
        self._live: LiveEngine | None = (
            LiveEngine(
                exchange=self._exchange,
                dry_run=True,                      # change to False to send real orders
                max_open_positions=self._settings.max_open_positions,
                max_daily_drawdown_pct=self._settings.max_daily_drawdown_pct,
                atr_sl_multiplier=self._settings.atr_sl_multiplier,
                rr_ratio=self._settings.rr_ratio,
                default_leverage=self._settings.default_leverage,
            )
            if not self._settings.is_paper
            else None
        )

        # Signal engine + decision core
        self._engine = DecisionEngine(
            min_confidence=float(self._settings.min_confidence_threshold),
            max_open_positions=self._settings.max_open_positions,
            max_portfolio_risk_pct=self._settings.max_portfolio_risk_pct,
            max_single_risk_pct=self._settings.max_single_trade_risk_pct,
            kelly_fraction=self._settings.kelly_fraction,
            atr_sl_multiplier=self._settings.atr_sl_multiplier,
            rr_ratio=self._settings.rr_ratio,
            adx_min_threshold=self._settings.adx_min_threshold,
            max_daily_drawdown_pct=self._settings.max_daily_drawdown_pct,
            max_correlated_positions=getattr(self._settings, "max_correlated_positions", 2),
            portfolio_value_fn=self._get_portfolio_value,
            open_positions_fn=self._get_open_positions,
            on_decision_fn=self._on_trade_decision,
            model_path="models/lgbm_classifier.joblib",
        )

        # Per-symbol OHLCV buffers fed by the market feed
        self._candle_buffers: dict[str, dict[str, list]] = {}

        # Dead-signal watchdog: timestamp of last closed candle processed
        self._last_candle_ts: float = time.monotonic()
        # How long (seconds) before alerting on silence (10 minutes)
        self._watchdog_threshold: float = 600.0

        # Telegram alerter
        self._alerter = get_alerter()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info(
            "CryptoAgent starting",
            mode=self._settings.mode,
            exchange=self._settings.primary_exchange,
            symbols=self._settings.symbols,
        )

        await self._exchange.connect()
        await self._db.connect()
        await self._db.initialize_schema()

        # Recover any positions that were open when the agent last stopped
        recovered_symbols: list = []
        if self._paper:
            recovered_symbols = await self._recover_paper_positions() or []

        # Pre-load historical candles so the signal engine is warm on first tick
        await self._preload_buffers()

        # Recalculate SL/TP for recovered positions using current ATR from warm buffers
        if self._paper and recovered_symbols:
            await self._recalculate_recovered_sl(recovered_symbols)

        # Subscribe private streams (orders & positions) — live mode only.
        # Paper mode: the paper engine handles fills internally; connecting to
        # BloFin's private WS is unnecessary and causes constant auth failures.
        if self._settings.is_live:
            await self._exchange.subscribe_orders(self._on_order_update)
            await self._exchange.subscribe_positions(self._on_position_update)

        # Wire candle callback for signal engine (Phase 2)
        self._feed.on_candle(self._on_candle)

        # Start market feed — retry indefinitely with exponential backoff.
        # 429 rate-limit: wait 300s. Other errors: standard 5-320s backoff.
        _attempt = 0
        while True:
            try:
                await self._feed.start()
                break
            except Exception as exc:
                is_429 = "429" in str(exc)
                if is_429:
                    wait = 300  # BloFin rate limit — give it 5 min to clear
                else:
                    wait = min(5 * (2 ** _attempt), 120)
                logger.warning(
                    "Market feed start failed, retrying",
                    attempt=_attempt, error=str(exc), retry_in=wait,
                )
                await asyncio.sleep(wait)
                _attempt += 1

        self._running = True
        logger.info("CryptoAgent running", mode=self._settings.mode)

        await self._alerter.agent_started(
            mode=self._settings.mode,
            symbols=self._settings.symbols,
        )

        # Launch watchdog alongside main loop
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._main_loop())
            tg.create_task(self._watchdog_loop())
            tg.create_task(self._command_loop())

    async def _recover_paper_positions(self) -> None:
        """
        On restart, reload any positions that were open when the agent last stopped.

        Strategy:
          1. Query the DB for filled orders that have no matching paper_trades row
             (those are positions still "open" from the previous run).
          2. Compute the current cash balance: last paper_trades.balance_after minus
             margin+fee for each recovered position (since the DB balance was recorded
             BEFORE those positions were opened).
          3. Inject PaperPosition objects directly into the paper engine's _positions
             dict and correct _balance_usdt accordingly.
        """
        from src.execution.paper import PaperPosition
        from src.exchanges.base import PositionSide
        import time

        open_rows = await self._db.get_open_paper_positions()
        if not open_rows:
            logger.info("No open paper positions to recover — starting fresh.")
            return

        # Cash baseline: last recorded balance_after + its timestamp
        # balance_after already has the margins of any positions opened BEFORE that
        # close already deducted. We only need to deduct margins for positions
        # opened AFTER the last close.
        last = await self._db.get_last_paper_balance()
        if last is None:
            last_balance = self._settings.paper_starting_balance
            last_closed_at = None
        else:
            last_balance, last_closed_at = last

        fee_pct = self._settings.paper_fee_pct / 100.0
        leverage = self._settings.default_leverage

        recovered_balance = last_balance
        recovered = []
        for row in open_rows:
            symbol      = row["symbol"]
            side_str    = row["side"]       # "buy" | "sell"
            entry_price = float(row["entry_price"])
            size        = float(row["size"])
            sl_price    = float(row["sl_price"] or 0.0)
            tp_price    = float(row["tp_price"] or 0.0)
            opened_at   = row["created_at"]   # tz-aware datetime
            opened_at_ms = int(opened_at.timestamp() * 1000)

            notional    = entry_price * size
            margin      = notional / leverage
            fee_on_open = notional * fee_pct

            # Only deduct margin+fee for positions opened AFTER the last recorded balance.
            # Positions opened before that close already had their margin deducted from
            # the running _balance_usdt that produced balance_after.
            if last_closed_at is None or opened_at > last_closed_at:
                recovered_balance -= (margin + fee_on_open)

            side = PositionSide.LONG if side_str == "buy" else PositionSide.SHORT
            pos  = PaperPosition(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                leverage=leverage,
                margin=margin,
                sl_price=sl_price,
                tp_price=tp_price,
                opened_at=opened_at_ms,
            )
            self._paper._positions[symbol] = pos
            recovered.append(symbol)

        # Patch the paper engine's cash balance to match recovered state
        self._paper._balance_usdt = max(recovered_balance, 0.0)

        logger.info(
            "Recovered paper positions from DB",
            positions=recovered,
            recovered_balance=round(recovered_balance, 4),
        )
        return recovered

    async def _recalculate_recovered_sl(self, recovered_symbols: list) -> None:
        """
        After buffers are warm, recalculate SL/TP for recovered positions
        using the current ATR so stale SLs from a previous run don't fire instantly.
        """
        import pandas as pd
        import numpy as np
        from src.signals.technical.indicators import atr as compute_atr

        atr_mult = self._settings.atr_sl_multiplier
        rr_ratio = self._settings.rr_ratio

        for symbol in recovered_symbols:
            pos = self._paper._positions.get(symbol)
            if pos is None:
                continue
            buf = self._candle_buffers.get(symbol, {}).get("15m")
            if not buf or len(buf) < 20:
                continue

            df = pd.DataFrame(buf)
            atr_series = compute_atr(df, 14)
            atr_val = float(atr_series.iloc[-1])
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            current_price = float(df["close"].iloc[-1])
            from src.exchanges.base import PositionSide
            if pos.side == PositionSide.LONG:
                new_sl = current_price - atr_mult * atr_val
                new_tp = current_price + rr_ratio * atr_mult * atr_val
            else:
                new_sl = current_price + atr_mult * atr_val
                new_tp = current_price - rr_ratio * atr_mult * atr_val

            pos.sl_price = round(new_sl, 6)
            pos.tp_price = round(new_tp, 6)
            logger.info(
                "Recalculated SL/TP for recovered position",
                symbol=symbol, side=pos.side.value,
                entry=pos.entry_price, current=current_price,
                new_sl=round(new_sl, 4), new_tp=round(new_tp, 4),
                atr=round(atr_val, 4),
            )

    async def _preload_buffers(self) -> None:
        """Fetch recent historical candles so signal engine is warm immediately."""
        import pandas as pd
        lookback = self._settings.lookback_candles  # 500
        # Pre-load all signal timeframes so MTF confluence can fire on the very first live candle.
        # 1m/3m/5m: fast entry triggers | 15m: primary ML entry | 1h: intermediate trend | 4h: bias
        signal_tfs = ["1m", "3m", "5m", "15m", "1h", "4h"]
        consecutive_429s = 0
        for symbol in self._settings.symbols:
            for tf in signal_tfs:
                if consecutive_429s >= 3:
                    # IP is rate-limited — bail out of preload immediately.
                    # Buffers will fill naturally as WS candles arrive.
                    logger.warning("IP rate-limited (429) — skipping remaining preload",
                                   completed=f"{symbol}/{tf}")
                    return
                try:
                    logger.info("Pre-loading candles", symbol=symbol, timeframe=tf, bars=lookback)
                    # BloFin max 300/request — fetch two pages for 500+ bars
                    page1 = await self._exchange.get_candles(symbol, tf, limit=300)
                    candles = list(page1)
                    consecutive_429s = 0  # reset on success
                    if len(candles) == 300 and lookback > 300:
                        oldest_ts = min(c.timestamp for c in candles)
                        page2 = await self._exchange.get_candles(
                            symbol, tf, limit=300, since=oldest_ts
                        )
                        candles = list(page2) + candles
                    if not candles:
                        logger.warning("No history returned", symbol=symbol, timeframe=tf)
                        continue
                    # Sort oldest-first; drop last candle if it is still forming
                    candles.sort(key=lambda c: c.timestamp)
                    candles = [c for c in candles if getattr(c, "closed", True)]
                    rows = [{"open": c.open, "high": c.high, "low": c.low,
                             "close": c.close, "volume": c.volume, "ts": c.timestamp} for c in candles]
                    if symbol not in self._candle_buffers:
                        self._candle_buffers[symbol] = {}
                    self._candle_buffers[symbol][tf] = rows[-lookback:]
                    logger.info("Buffer pre-loaded", symbol=symbol, timeframe=tf,
                                bars=len(self._candle_buffers[symbol][tf]))
                    # Seed signal engine so _signal_cache has both TFs ready
                    # before the first live candle arrives (avoids 60-min wait for 1H)
                    if len(self._candle_buffers[symbol][tf]) >= 50:
                        df = pd.DataFrame(self._candle_buffers[symbol][tf])
                        await self._engine.on_candle(symbol, tf, df)
                        logger.info("Signal cache seeded", symbol=symbol, timeframe=tf)
                    # Also seed Redis with latest price
                    if candles:
                        last = candles[-1]
                        import redis.asyncio as aioredis
                        from src.core.config import get_settings
                        r = aioredis.from_url(get_settings().database.redis_url, decode_responses=True)
                        await r.hset(f"price:{symbol}", mapping={
                            "close": last.close, "high": last.high,
                            "low": last.low, "volume": last.volume, "ts": last.timestamp,
                        })
                        await r.aclose()
                except Exception as exc:
                    logger.warning("Failed to pre-load history", symbol=symbol,
                                   timeframe=tf, error=str(exc))
                    if "429" in str(exc):
                        consecutive_429s += 1

    async def stop(self) -> None:
        logger.info("CryptoAgent shutting down...")
        self._running = False

        # Safety: cancel all open orders if live
        if self._settings.is_live:
            try:
                cancelled = await self._exchange.cancel_all_orders()
                logger.info("Cancelled open orders on shutdown", count=cancelled)
            except Exception as exc:
                logger.error("Failed to cancel orders on shutdown", error=str(exc))

        await self._feed.stop()
        await self._exchange.disconnect()
        await self._db.disconnect()
        await self._alerter.agent_stopped()
        await self._alerter.close()
        logger.info("CryptoAgent stopped cleanly")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def _watchdog_loop(self) -> None:
        """
        Alert via Telegram if no closed candles are received for >10 minutes.
        Runs every 60 seconds. Resets after each alert to avoid spam (re-alerts
        after another full threshold period of silence).
        """
        check_interval = 60.0   # check every minute
        alerted_at: float = 0.0

        while self._running:
            await asyncio.sleep(check_interval)
            silent_secs = time.monotonic() - self._last_candle_ts
            if silent_secs >= self._watchdog_threshold:
                # Only alert once per silence period (suppress repeat alerts)
                if time.monotonic() - alerted_at >= self._watchdog_threshold:
                    minutes = silent_secs / 60.0
                    logger.warning(
                        "Dead signal watchdog: no closed candles for %.1f min — sending alert",
                        minutes,
                    )
                    try:
                        await self._alerter.dead_signal_alert(
                            minutes_silent=minutes,
                            symbols=self._settings.symbols,
                        )
                    except Exception as exc:
                        logger.debug("Watchdog alert failed: %s", exc)
                    alerted_at = time.monotonic()

    async def _command_loop(self) -> None:
        """Listen for dashboard commands via Redis pub/sub."""
        import redis.asyncio as aioredis
        import json
        
        while self._running:
            try:
                r = aioredis.from_url(self._settings.database.redis_url, decode_responses=True)
                pubsub = r.pubsub()
                await pubsub.subscribe("agent_commands")
                async for message in pubsub.listen():
                    if not self._running:
                        break
                    if message["type"] == "message":
                        try:
                            cmd = json.loads(message["data"])
                            action = cmd.get("action")
                            symbol = cmd.get("symbol")
                            if not symbol:
                                continue
                            
                            if action == "close_position":
                                logger.info("Manual close command received", symbol=symbol)
                                if self._settings.is_paper and self._paper:
                                    price = self._paper._prices.get(symbol)
                                    if price:
                                        self._paper._close_position(symbol, price, reason="manual_close")
                                elif self._live:
                                    await self._live.close_position(symbol, reason="manual_close")
                                    
                            elif action == "take_profit":
                                pct = float(cmd.get("pct", 0.5))
                                logger.info("Manual take profit command received", symbol=symbol, pct=pct)
                                if self._settings.is_paper and self._paper:
                                    pos = self._paper._positions.get(symbol)
                                    price = self._paper._prices.get(symbol)
                                    if pos and price:
                                        close_size = pos.size * pct
                                        if close_size > 0:
                                            # Close a percentage of the position
                                            pnl = (price - pos.entry_price) * close_size if pos.side.value == "long" else (pos.entry_price - price) * close_size
                                            fee = price * close_size * self._paper._fee_pct
                                            net_pnl = pnl - fee
                                            
                                            pos.size -= close_size
                                            pos.margin -= pos.margin * pct
                                            pos.realized_pnl += net_pnl
                                            self._paper._balance_usdt += pos.margin * pct + net_pnl
                                            self._paper._total_fees_paid += fee
                                            
                                            self._paper._pending_closes.append({
                                                "symbol": symbol, "side": pos.side.value,
                                                "size": close_size, "entry_price": pos.entry_price,
                                                "exit_price": price, "sl_price": pos.sl_price, "tp_price": pos.tp_price,
                                                "pnl": round(net_pnl, 4), "pnl_pct": round(net_pnl / (pos.margin * pct), 6) if pos.margin else 0,
                                                "fee": round(fee, 4), "close_reason": "manual_take_profit",
                                                "opened_at": pos.opened_at, "closed_at": int(time.time() * 1000),
                                                "balance_after": round(self._paper._balance_usdt, 4),
                                            })
                                            if pos.size < 0.00001:
                                                self._paper._positions.pop(symbol, None)
                                elif self._live:
                                    # Partial close in live mode logic goes here if implemented
                                    # For now, default to full close.
                                    await self._live.close_position(symbol, reason="manual_take_profit")
                                    
                        except Exception as e:
                            logger.error("Error processing manual command", error=str(e))
                # If we break out of listen()
                await pubsub.unsubscribe("agent_commands")
                await r.aclose()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    logger.error("Command loop error", error=str(exc))
                    await asyncio.sleep(5.0)

    async def _main_loop(self) -> None:
        heartbeat = self._settings.heartbeat_interval
        while self._running:
            try:
                await self._heartbeat()
                await asyncio.sleep(heartbeat)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Main loop error", error=str(exc), exc_info=True)
                await asyncio.sleep(heartbeat)

    async def _heartbeat(self) -> None:
        """Periodic health check and portfolio snapshot."""
        if self._settings.is_paper and self._paper:
            stats = self._paper.get_stats()
            logger.info("Paper portfolio", **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})
            bal = self._paper.get_balance()
            await self._db.log_portfolio_snapshot(
                total_equity=bal.total,
                available=bal.available,
                unrealized_pnl=bal.unrealized_pnl,
                num_positions=len(self._paper.get_positions()),
                mode="paper",
            )
        else:
            try:
                balances = await self._exchange.get_balance()
                usdt = next((b for b in balances if b.currency == "USDT"), None)
                if usdt:
                    positions = await self._exchange.get_positions()
                    await self._db.log_portfolio_snapshot(
                        total_equity=usdt.total,
                        available=usdt.available,
                        unrealized_pnl=usdt.unrealized_pnl,
                        num_positions=len(positions),
                        mode="live",
                    )
            except Exception as exc:
                logger.warning("Heartbeat balance fetch failed", error=str(exc))

    # ── Portfolio info callbacks ───────────────────────────────────────────────

    async def _get_portfolio_value(self) -> float:
        if self._paper:
            return self._paper.get_balance().total
        try:
            balances = await self._exchange.get_balance()
            usdc = next((b for b in balances if b.currency in ("USDC", "USDT")), None)
            return usdc.total if usdc else 10_000.0
        except Exception:
            return 10_000.0

    async def _get_open_positions(self) -> list:
        if self._paper:
            return self._paper.get_positions()
        try:
            return await self._exchange.get_positions()
        except Exception:
            return []

    # ── Event Handlers ────────────────────────────────────────────────────────

    async def _on_candle(self, candle) -> None:
        """Called on every candle update from the market feed."""
        symbol    = candle.symbol
        timeframe = candle.timeframe if hasattr(candle, "timeframe") else self._settings.primary_timeframe

        # Always update paper engine mark price on every tick (including forming candles)
        # so SL/TP triggers fire promptly at the correct price.
        if self._settings.is_paper and self._paper:
            self._paper.update_price(symbol, candle.close)
            pending_closes = self._paper.drain_pending_closes()
            # Set SL cooldowns synchronously BEFORE any awaits so that concurrent
            # candle-processing tasks cannot bypass the guard (asyncio race condition fix).
            for evt in pending_closes:
                if evt.get("close_reason") in ("stop_loss", "breakeven_stop"):
                    sym = evt["symbol"]
                    self._sl_cooldown[sym] = time.time()
                    # Adaptive threshold: raise conf threshold after consecutive SL hits
                    self._sl_hit_count[sym] = self._sl_hit_count.get(sym, 0) + 1
                    hits = self._sl_hit_count[sym]
                    if hits >= 2:
                        base = self._settings.min_confidence_threshold
                        overrides = self._settings.symbol_overrides.get(sym, {})
                        base = overrides.get("min_confidence_threshold", base)
                        bump = min(hits * 10.0, 30.0)  # +10% per hit, cap at +30%
                        self._adaptive_threshold[sym] = base + bump
                        self._adaptive_threshold_until[sym] = time.time() + 4 * 3600  # 4 hours
                        logger.warning(
                            "Adaptive threshold raised after consecutive SL hits",
                            symbol=sym, consecutive_sl_hits=hits,
                            new_threshold=self._adaptive_threshold[sym],
                            expires_in_hours=4,
                        )
                    logger.info(
                        "SL cooldown set",
                        symbol=sym,
                        cooldown_secs=self._SL_COOLDOWN_SECS,
                        consecutive_hits=hits,
                    )
            for close_evt in pending_closes:
                try:
                    await self._db.log_paper_trade(close_evt)
                    await self._alerter.trade_closed(
                        symbol=close_evt["symbol"],
                        side=close_evt["side"],
                        pnl=close_evt["pnl"],
                        pnl_pct=close_evt["pnl_pct"],
                        reason=close_evt["close_reason"],
                        exit_price=close_evt["exit_price"],
                    )
                except Exception as exc:
                    logger.warning("Failed to record paper trade close", error=str(exc))

        # Only append CONFIRMED (closed) candles to the signal buffer.
        # Update watchdog timestamp whenever a confirmed candle arrives.
        # Forming candles flood the buffer (BTC sends 100s of updates/sec for the
        # live 4h candle), pushing out historical rows and corrupting indicator
        # computations (ADX→NaN, LGBM features NaN→(0,0%)).
        if not getattr(candle, "closed", True):
            return

        # Reset watchdog — we have a live confirmed candle
        self._last_candle_ts = time.monotonic()

        # Maintain rolling OHLCV buffer for signal engine
        if symbol not in self._candle_buffers:
            self._candle_buffers[symbol] = {}
        if timeframe not in self._candle_buffers[symbol]:
            self._candle_buffers[symbol][timeframe] = []

        buf = self._candle_buffers[symbol][timeframe]
        buf.append({
            "open":   candle.open,
            "high":   candle.high,
            "low":    candle.low,
            "close":  candle.close,
            "volume": candle.volume,
            "ts":     candle.timestamp,
        })

        # Keep last 500 candles in memory
        lookback = self._settings.lookback_candles
        if len(buf) > lookback:
            buf[:] = buf[-lookback:]

        # Feed signal engine once we have enough bars
        if len(buf) >= 50:
            import pandas as pd
            df = pd.DataFrame(buf)
            await self._engine.on_candle(symbol, timeframe, df)

    async def _on_trade_decision(self, decision: TradeDecision) -> None:
        """Handle a trade decision from the decision engine."""
        logger.info(
            "Trade decision received",
            symbol=decision.symbol,
            side=decision.side,
            size_usdc=decision.size_usdc,
            confidence=round(decision.confidence, 1),
            rr=round(decision.risk_reward, 2),
            reason=decision.reason,
        )

        if self._settings.is_paper and self._paper:
            # ── Guard 1: SL cooldown — block re-entry for 45 min after SL hit ──
            last_sl = self._sl_cooldown.get(decision.symbol, 0.0)
            remaining = self._SL_COOLDOWN_SECS - (time.time() - last_sl)
            if remaining > 0:
                logger.warning(
                    "Trade blocked by SL cooldown",
                    symbol=decision.symbol,
                    cooldown_remaining_s=round(remaining),
                )
                return

            # ── Guard 1b: Adaptive threshold — elevated after consecutive SL hits ──
            adaptive_until = self._adaptive_threshold_until.get(decision.symbol, 0.0)
            if time.time() < adaptive_until:
                effective_threshold = self._adaptive_threshold.get(decision.symbol, 0.0)
                if decision.confidence < effective_threshold:
                    logger.warning(
                        "Trade blocked by adaptive threshold",
                        symbol=decision.symbol,
                        confidence=round(decision.confidence, 1),
                        adaptive_threshold=round(effective_threshold, 1),
                        expires_in_min=round((adaptive_until - time.time()) / 60),
                    )
                    return
            else:
                # Adaptive threshold expired — reset consecutive hit counter
                if decision.symbol in self._adaptive_threshold:
                    del self._adaptive_threshold[decision.symbol]
                    del self._adaptive_threshold_until[decision.symbol]
                    self._sl_hit_count[decision.symbol] = 0

            # ── Guard 2: Adjust SL/TP to live price; reject if price past SL ──
            from src.exchanges.base import OrderSide, OrderType
            live_price = self._paper._prices.get(decision.symbol, decision.entry_price)
            sl_dist = abs(decision.stop_loss - decision.entry_price)
            tp_dist = abs(decision.take_profit - decision.entry_price)

            if decision.side == "short":
                sl_price = live_price + sl_dist
                tp_price = live_price - tp_dist
                if live_price >= sl_price:
                    logger.warning(
                        "Trade rejected: live price already at/past SL",
                        symbol=decision.symbol, side="short",
                        live=live_price, sl=sl_price,
                    )
                    return
            else:
                sl_price = live_price - sl_dist
                tp_price = live_price + tp_dist
                if live_price <= sl_price:
                    logger.warning(
                        "Trade rejected: live price already at/past SL",
                        symbol=decision.symbol, side="long",
                        live=live_price, sl=sl_price,
                    )
                    return

            if abs(live_price - decision.entry_price) / max(decision.entry_price, 1) > 0.001:
                logger.info(
                    "Entry price adjusted to live price",
                    symbol=decision.symbol,
                    old_entry=decision.entry_price,
                    live_price=live_price,
                    new_sl=round(sl_price, 6),
                    new_tp=round(tp_price, 6),
                )

        # Fire Telegram alert for every trade decision
        await self._alerter.trade_opened(decision)

        if self._settings.is_paper and self._paper:
            # Compute trailing stop distances in price units from ATR approximation
            settings = self._settings
            atr_approx = sl_dist / settings.atr_sl_multiplier if settings.atr_sl_multiplier > 0 else sl_dist
            if settings.trailing_stop_activation_atr > 0:
                trailing_activation = settings.trailing_stop_activation_atr * atr_approx
                trailing_distance = settings.trailing_stop_distance_atr * atr_approx
            else:
                trailing_activation = 0.0
                trailing_distance = 0.0
            breakeven_activation = settings.breakeven_activation_atr * atr_approx if settings.breakeven_activation_atr > 0 else 0.0

            # Partial TP at 1:1 R (entry ± 1×risk_distance) — takes 50% profit early,
            # moves SL to entry, lets remaining 50% run to full TP at 2:1 R.
            if decision.side == "long":
                partial_tp_price = round(live_price + sl_dist, 6)
            else:
                partial_tp_price = round(live_price - sl_dist, 6)

            # Time-based exit: close after max_hold_hours (read from settings)
            max_hold_ms = int(settings.max_hold_hours * 3600 * 1000)

            side = OrderSide.BUY if decision.side == "long" else OrderSide.SELL
            order = self._paper.place_order(
                symbol=decision.symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=decision.size_usdc / live_price,
                price=live_price,
                sl_price=round(sl_price, 6),
                tp_price=round(tp_price, 6),
                trailing_activation=trailing_activation,
                trailing_distance=trailing_distance,
                breakeven_activation=breakeven_activation,
                partial_tp_price=partial_tp_price,
                max_hold_ms=max_hold_ms,
            )
            if order:
                logger.info(
                    "Paper order placed",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    size=order.size,
                )
                # Persist order so dashboard can display it immediately
                try:
                    await self._db.upsert_order(order)
                except Exception as exc:
                    logger.warning("Failed to persist paper order to DB", error=str(exc))
                await self._db.log_signal(
                    symbol=decision.symbol,
                    timeframe=decision.timeframe,
                    direction=decision.side,
                    confidence=decision.confidence,
                    data=decision.signal_breakdown,
                )

        elif not self._settings.is_paper and self._live:
            order = await self._live.execute(decision)
            if order:
                logger.info(
                    "Live order submitted",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    size=order.size,
                    dry_run=self._live.is_dry_run,
                )
                try:
                    await self._db.upsert_order(order)
                except Exception as exc:
                    logger.warning("Failed to persist live order to DB", error=str(exc))
                await self._db.log_signal(
                    symbol=decision.symbol,
                    timeframe=decision.timeframe,
                    direction=decision.side,
                    confidence=decision.confidence,
                    data=decision.signal_breakdown,
                )

    async def _on_order_update(self, order) -> None:
        await self._db.upsert_order(order)
        logger.info(
            "Order update",
            id=order.order_id, symbol=order.symbol,
            status=order.status.value, filled=order.filled_size,
        )

    async def _on_position_update(self, position) -> None:
        await self._db.insert_position_snapshot(position)
        logger.info(
            "Position update",
            symbol=position.symbol, side=position.side.value,
            size=position.size, pnl=position.unrealized_pnl,
        )


# ── Entry Point ────────────────────────────────────────────────────────────────

async def main() -> None:
    configure_logging()

    # Kill any previously running agent instance, then write our PID.
    # This prevents zombie agent accumulation when the process is restarted
    # without a clean shutdown (e.g. terminal killed, system restart).
    import os
    import subprocess
    from pathlib import Path as _Path
    _pid_file = _Path(__file__).parent.parent / "agent.pid"
    if _pid_file.exists():
        try:
            _old_pid = int(_pid_file.read_text().strip())
            if _old_pid != os.getpid():
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(_old_pid)],
                    capture_output=True, timeout=5,
                )
                await asyncio.sleep(1)
        except Exception:
            pass
    _pid_file.write_text(str(os.getpid()))

    agent = CryptoAgent()

    loop = asyncio.get_event_loop()

    def _shutdown(sig):
        logger.info("Shutdown signal received", signal=sig.name)
        loop.create_task(agent.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for all signals
            pass

    try:
        await agent.start()
    except KeyboardInterrupt:
        await agent.stop()
    finally:
        try:
            _pid_file.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
