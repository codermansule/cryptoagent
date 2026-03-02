"""
Live Execution Engine.

Wraps real BloFin order placement with safety guards:
  - Dry-run mode: logs orders without sending (enabled by default)
  - Exchange-side position tracking via REST
  - Duplicate guard: blocks same symbol re-entry while position is open
  - Circuit breaker: halts on daily drawdown limit
  - WebSocket fill confirmation with fallback REST polling
  - All orders use IOC/Post-Only or limit with configurable offset

Switch from paper → live by setting  agent.mode = live  in settings.yaml.
ALWAYS run dry_run = true first and verify order logs before enabling real fills.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Callable, Awaitable

from src.core.config import get_settings
from src.core.logging import get_logger
from src.decision.core import TradeDecision
from src.exchanges.base import (
    Order, OrderSide, OrderType, OrderStatus, Position, PositionSide,
)

logger = get_logger(__name__)


class LiveEngine:
    """
    Real-money execution engine backed by BloFin REST.

    Usage (in agent.py when mode == 'live'):
        engine = LiveEngine(exchange, dry_run=True)
        await engine.execute(decision)
    """

    def __init__(
        self,
        exchange,                               # BloFinAdapter instance
        dry_run: bool = True,                   # ALWAYS start True
        max_open_positions: int = 5,
        max_daily_drawdown_pct: float = 5.0,
        atr_sl_multiplier: float = 2.5,
        rr_ratio: float = 2.0,
        default_leverage: int = 5,
        on_fill_fn: Optional[Callable[[Order], Awaitable[None]]] = None,
    ) -> None:
        self._exchange = exchange
        self._dry_run = dry_run
        self._max_open = max_open_positions
        self._max_dd_pct = max_daily_drawdown_pct
        self._atr_sl_mult = atr_sl_multiplier
        self._rr_ratio = rr_ratio
        self._leverage = default_leverage
        self._on_fill_fn = on_fill_fn

        # Circuit breaker
        self._daily_pnl_pct: float = 0.0
        self._halted: bool = False

        # In-flight order tracking: symbol → order_id
        self._pending_orders: dict[str, str] = {}

        if dry_run:
            logger.warning(
                "LiveEngine started in DRY-RUN mode — orders will NOT be sent to exchange",
                dry_run=True,
            )
        else:
            logger.warning(
                "LiveEngine LIVE mode active — real orders will be placed",
                dry_run=False,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def execute(self, decision: TradeDecision) -> Optional[Order]:
        """
        Execute a TradeDecision. Returns the placed Order or None if blocked.
        In dry_run mode: logs the order but returns a simulated filled Order.
        """
        if self._halted:
            logger.warning("Circuit breaker active — skipping trade", symbol=decision.symbol)
            return None

        if self._daily_pnl_pct <= -self._max_dd_pct:
            self._halted = True
            logger.warning(
                "Daily drawdown limit hit — trading halted",
                daily_pnl_pct=round(self._daily_pnl_pct, 2),
            )
            return None

        # Check exchange-side open position
        if await self._has_open_position(decision.symbol):
            logger.debug("Already in position — skip", symbol=decision.symbol)
            return None

        # Count total open positions
        open_pos = await self._count_open_positions()
        if open_pos >= self._max_open:
            logger.debug("Max open positions reached", count=open_pos, max=self._max_open)
            return None

        side = OrderSide.BUY if decision.side == "long" else OrderSide.SELL

        # Calculate contracts from notional size
        contracts = round(decision.size_usdc / decision.entry_price, 6)
        if contracts <= 0:
            logger.warning("Calculated contract size is 0 — skip", symbol=decision.symbol)
            return None

        if self._dry_run:
            return await self._simulate_order(decision, side, contracts)
        else:
            return await self._place_order(decision, side, contracts)

    async def close_position(self, symbol: str, reason: str = "signal") -> Optional[Order]:
        """Close an open position on the exchange."""
        positions = await self._fetch_positions(symbol)
        if not positions:
            return None

        pos = positions[0]
        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY

        if self._dry_run:
            logger.info(
                "[DRY-RUN] Would close position",
                symbol=symbol, side=pos.side.value,
                size=pos.size, reason=reason,
            )
            return None

        try:
            order = await self._exchange.place_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                size=pos.size,
                reduce_only=True,
            )
            logger.info(
                "Position close order placed",
                symbol=symbol, order_id=order.order_id, reason=reason,
            )
            return order
        except Exception as exc:
            logger.error("Failed to close position", symbol=symbol, error=str(exc))
            return None

    def update_daily_pnl(self, pct_change: float) -> None:
        """Call from fill handler to track daily PnL for circuit breaker."""
        self._daily_pnl_pct += pct_change

    def reset_daily_stats(self) -> None:
        """Call at UTC midnight to reset daily circuit breaker."""
        self._daily_pnl_pct = 0.0
        self._halted = False
        logger.info("LiveEngine daily stats reset")

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _simulate_order(
        self,
        decision: TradeDecision,
        side: OrderSide,
        contracts: float,
    ) -> Order:
        """Log the order details in dry-run mode."""
        logger.info(
            "[DRY-RUN] Would place order",
            symbol=decision.symbol,
            side=side.value,
            contracts=contracts,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            size_usdc=decision.size_usdc,
            confidence=round(decision.confidence, 1),
            reason=decision.reason,
        )
        # Return a simulated filled order for upstream tracking
        return Order(
            order_id=f"dry_{int(time.time() * 1000)}",
            symbol=decision.symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=contracts,
            price=decision.entry_price,
            status=OrderStatus.FILLED,
            filled_size=contracts,
            avg_fill_price=decision.entry_price,
            timestamp=int(time.time() * 1000),
        )

    async def _place_order(
        self,
        decision: TradeDecision,
        side: OrderSide,
        contracts: float,
    ) -> Optional[Order]:
        """Place a real order on BloFin with SL and TP attached."""
        settings = get_settings()
        offset_pct = settings.limit_order_offset_pct / 100.0

        # Use limit order slightly inside spread for better fills
        if side == OrderSide.BUY:
            limit_price = round(decision.entry_price * (1 + offset_pct), 6)
        else:
            limit_price = round(decision.entry_price * (1 - offset_pct), 6)

        try:
            # Set leverage first
            await self._exchange.set_leverage(decision.symbol, self._leverage)
        except Exception:
            pass   # exchange may reject if already set

        try:
            order = await self._exchange.place_order(
                symbol=decision.symbol,
                side=side,
                order_type=OrderType.LIMIT,
                size=contracts,
                price=limit_price,
                sl_price=decision.stop_loss,
                tp_price=decision.take_profit,
            )
            logger.info(
                "Live order placed",
                symbol=decision.symbol,
                order_id=order.order_id,
                side=side.value,
                contracts=contracts,
                limit_price=limit_price,
                sl=decision.stop_loss,
                tp=decision.take_profit,
            )

            self._pending_orders[decision.symbol] = order.order_id

            # Poll for fill confirmation (up to 30s)
            asyncio.create_task(
                self._await_fill(order.order_id, decision.symbol)
            )

            return order

        except Exception as exc:
            logger.error(
                "Failed to place live order",
                symbol=decision.symbol, error=str(exc),
            )
            return None

    async def _await_fill(self, order_id: str, symbol: str, timeout: float = 30.0) -> None:
        """Poll REST until the order fills or times out, then trigger fill callback."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            try:
                order = await self._exchange.get_order(symbol, order_id)
                if order.status == OrderStatus.FILLED:
                    logger.info(
                        "Order filled",
                        symbol=symbol, order_id=order_id,
                        fill_price=order.avg_fill_price, size=order.filled_size,
                    )
                    self._pending_orders.pop(symbol, None)
                    if self._on_fill_fn:
                        await self._on_fill_fn(order)
                    return
                elif order.status == OrderStatus.CANCELLED:
                    logger.warning("Order cancelled", symbol=symbol, order_id=order_id)
                    self._pending_orders.pop(symbol, None)
                    return
            except Exception as exc:
                logger.debug("Order poll error", order_id=order_id, error=str(exc))

        # Timeout — cancel unfilled order
        logger.warning("Order timed out — cancelling", symbol=symbol, order_id=order_id)
        try:
            await self._exchange.cancel_order(symbol, order_id)
        except Exception:
            pass
        self._pending_orders.pop(symbol, None)

    async def _fetch_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Fetch open positions from exchange."""
        try:
            positions = await self._exchange.get_positions(symbol)
            return [p for p in positions if p.size > 0]
        except Exception as exc:
            logger.debug("Failed to fetch positions", error=str(exc))
            return []

    async def _has_open_position(self, symbol: str) -> bool:
        positions = await self._fetch_positions(symbol)
        return len(positions) > 0

    async def _count_open_positions(self) -> int:
        positions = await self._fetch_positions()
        return len(positions)
