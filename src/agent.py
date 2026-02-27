"""
CryptoAgent — Main Orchestrator
Entry point: python -m src.agent
"""

from __future__ import annotations

import asyncio
import signal
import sys

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.data.feeds.market_feed import MarketFeedManager
from src.data.storage.schema import Database
from src.exchanges.blofin import BloFinAdapter
from src.execution.paper import PaperEngine
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

        # Core subsystems
        self._exchange = BloFinAdapter()
        self._db = Database()
        self._feed = MarketFeedManager(self._exchange, self._db)
        self._paper = PaperEngine() if self._settings.is_paper else None

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
            portfolio_value_fn=self._get_portfolio_value,
            open_positions_fn=self._get_open_positions,
            on_decision_fn=self._on_trade_decision,
            model_path="models/lgbm_classifier.joblib",
        )

        # Per-symbol OHLCV buffers fed by the market feed
        self._candle_buffers: dict[str, dict[str, list]] = {}

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
        if self._paper:
            await self._recover_paper_positions()

        # Pre-load historical candles so the signal engine is warm on first tick
        await self._preload_buffers()

        # Subscribe private streams (orders & positions)
        await self._exchange.subscribe_orders(self._on_order_update)
        await self._exchange.subscribe_positions(self._on_position_update)

        # Wire candle callback for signal engine (Phase 2)
        self._feed.on_candle(self._on_candle)

        await self._feed.start()

        self._running = True
        logger.info("CryptoAgent running", mode=self._settings.mode)

        await self._alerter.agent_started(
            mode=self._settings.mode,
            symbols=self._settings.symbols,
        )

        await self._main_loop()

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

    async def _preload_buffers(self) -> None:
        """Fetch recent historical candles so signal engine is warm immediately."""
        import pandas as pd
        lookback = self._settings.lookback_candles  # 500
        # Pre-load both the LTF trigger (15m) and the HTF trend bias (1H)
        # so that MTF confluence can fire on the very first live candle,
        # instead of waiting up to 60 minutes for the first live 1H candle.
        signal_tfs = ["15m", "1h", "4h"]
        for symbol in self._settings.symbols:
            for tf in signal_tfs:
                try:
                    logger.info("Pre-loading candles", symbol=symbol, timeframe=tf, bars=lookback)
                    # BloFin max 300/request — fetch two pages for 500+ bars
                    page1 = await self._exchange.get_candles(symbol, tf, limit=300)
                    candles = list(page1)
                    if len(candles) == 300 and lookback > 300:
                        oldest_ts = min(c.timestamp for c in candles)
                        page2 = await self._exchange.get_candles(
                            symbol, tf, limit=300, since=oldest_ts
                        )
                        candles = list(page2) + candles
                    if not candles:
                        logger.warning("No history returned", symbol=symbol, timeframe=tf)
                        continue
                    # Sort oldest-first
                    candles.sort(key=lambda c: c.timestamp)
                    rows = [{"open": c.open, "high": c.high, "low": c.low,
                             "close": c.close, "volume": c.volume} for c in candles]
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
        """Called on every new closed candle from the market feed."""
        symbol    = candle.symbol
        timeframe = candle.timeframe if hasattr(candle, "timeframe") else self._settings.primary_timeframe

        # Update paper engine mark price, then drain any SL/TP-triggered closes
        if self._settings.is_paper and self._paper:
            self._paper.update_price(symbol, candle.close)
            for close_evt in self._paper.drain_pending_closes():
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

        # Fire Telegram alert for every trade decision
        await self._alerter.trade_opened(decision)

        if self._settings.is_paper and self._paper:
            from src.exchanges.base import OrderSide, OrderType
            side = OrderSide.BUY if decision.side == "long" else OrderSide.SELL
            order = self._paper.place_order(
                symbol=decision.symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=decision.size_usdc / decision.entry_price,
                price=decision.entry_price,
                sl_price=decision.stop_loss,
                tp_price=decision.take_profit,
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


if __name__ == "__main__":
    asyncio.run(main())
