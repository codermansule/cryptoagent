"""
Paper Trading Engine.
Simulates order execution using real market data with configurable
fee and slippage models. Maintains a virtual portfolio in memory.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.core.config import get_settings
from src.core.logging import get_logger
from src.exchanges.base import (
    Balance, MarginMode, Order, OrderSide, OrderStatus, OrderType,
    Position, PositionSide,
)

logger = get_logger(__name__)


@dataclass
class PaperPosition:
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    leverage: int
    margin: float                     # USDT locked as margin
    sl_price: float = 0.0
    tp_price: float = 0.0
    realized_pnl: float = 0.0
    opened_at: int = 0                # Unix ms when position was opened
    # Trailing stop
    trailing_activation: float = 0.0  # price distance from entry to activate trail (0 = disabled)
    trailing_distance: float = 0.0    # trail this many price units behind best_price
    best_price: float = 0.0           # most favorable price seen since open
    trailing_active: bool = False     # True once profit >= trailing_activation
    # Breakeven stop
    breakeven_activation: float = 0.0  # price distance from entry to trigger breakeven (0 = disabled)
    breakeven_triggered: bool = False   # True once SL has been moved to entry
    # Partial take-profit (50% at 1:1 R)
    partial_tp_price: float = 0.0      # price to take 50% profit (0 = disabled)
    partial_tp_taken: bool = False     # True once 50% has been closed
    # Time-based exit
    max_hold_ms: int = 0               # max hold time in ms (0 = disabled); close at market if exceeded

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side == PositionSide.LONG:
            return (mark_price - self.entry_price) * self.size
        return (self.entry_price - mark_price) * self.size

    def to_position(self, mark_price: float) -> Position:
        liq_distance = self.entry_price / self.leverage
        liq_price = (
            self.entry_price - liq_distance
            if self.side == PositionSide.LONG
            else self.entry_price + liq_distance
        )
        return Position(
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            entry_price=self.entry_price,
            mark_price=mark_price,
            liquidation_price=liq_price,
            unrealized_pnl=self.unrealized_pnl(mark_price),
            realized_pnl=self.realized_pnl,
            leverage=self.leverage,
            margin_mode=MarginMode.ISOLATED,
            margin=self.margin,
        )


class PaperEngine:
    """
    Simulated trading engine backed by real market prices.

    Lifecycle:
        engine = PaperEngine()
        engine.update_price("BTC-USDT", 65000.0)
        order = await engine.place_order(...)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._balance_usdt: float = self._settings.paper_starting_balance
        self._fee_pct: float = self._settings.paper_fee_pct / 100
        self._slippage_pct: float = self._settings.paper_slippage_pct / 100

        self._positions: dict[str, PaperPosition] = {}   # symbol -> position
        self._open_orders: dict[str, Order] = {}          # order_id -> order
        self._order_history: list[Order] = []
        self._prices: dict[str, float] = {}               # symbol -> last price
        self._peak_equity: float = self._settings.paper_starting_balance
        self._total_fees_paid: float = 0.0
        self._pending_closes: list[dict] = []             # closed trade events for async processing
        self._order_trailing: dict[str, tuple[float, float, float, float, int]] = {}  # order_id -> (activation, distance, breakeven, partial_tp, max_hold_ms)

        logger.info(
            "Paper engine initialized",
            balance=self._balance_usdt,
            fee_pct=self._settings.paper_fee_pct,
        )

    # ── Price Updates ─────────────────────────────────────────────────────────

    def update_price(self, symbol: str, price: float) -> None:
        """Called by market feed on every new price tick."""
        self._prices[symbol] = price
        self._check_sl_tp(symbol, price)
        self._check_limit_orders(symbol, price)

    def _check_sl_tp(self, symbol: str, price: float) -> None:
        pos = self._positions.get(symbol)
        if not pos:
            return

        # Update trailing stop before evaluating SL/TP
        if pos.trailing_activation > 0:
            if pos.best_price == 0.0:
                pos.best_price = pos.entry_price
            if pos.side == PositionSide.LONG:
                if price > pos.best_price:
                    pos.best_price = price
                if not pos.trailing_active and (pos.best_price - pos.entry_price) >= pos.trailing_activation:
                    pos.trailing_active = True
                    logger.info("Trailing stop activated", symbol=symbol,
                                best=round(pos.best_price, 4), activation=round(pos.trailing_activation, 4))
                if pos.trailing_active:
                    new_sl = pos.best_price - pos.trailing_distance
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
            else:  # SHORT
                if price < pos.best_price:
                    pos.best_price = price
                if not pos.trailing_active and (pos.entry_price - pos.best_price) >= pos.trailing_activation:
                    pos.trailing_active = True
                    logger.info("Trailing stop activated", symbol=symbol,
                                best=round(pos.best_price, 4), activation=round(pos.trailing_activation, 4))
                if pos.trailing_active:
                    new_sl = pos.best_price + pos.trailing_distance
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl

        # Breakeven stop: once price moves breakeven_activation in our favour,
        # slide the SL to entry price (no more losing trades that "went positive").
        if pos.breakeven_activation > 0 and not pos.breakeven_triggered:
            if pos.side == PositionSide.LONG:
                if price - pos.entry_price >= pos.breakeven_activation:
                    new_sl = pos.entry_price
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
                        pos.breakeven_triggered = True
                        logger.info("Breakeven stop triggered", symbol=symbol,
                                    entry=round(pos.entry_price, 4), sl_moved_to=round(new_sl, 4))
            else:  # SHORT
                if pos.entry_price - price >= pos.breakeven_activation:
                    new_sl = pos.entry_price
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl
                        pos.breakeven_triggered = True
                        logger.info("Breakeven stop triggered", symbol=symbol,
                                    entry=round(pos.entry_price, 4), sl_moved_to=round(new_sl, 4))

        # ── Partial TP: close 50% at 1:1 R, move SL to breakeven ──────────────
        if pos.partial_tp_price > 0 and not pos.partial_tp_taken:
            hit_partial = (
                (pos.side == PositionSide.LONG  and price >= pos.partial_tp_price) or
                (pos.side == PositionSide.SHORT and price <= pos.partial_tp_price)
            )
            if hit_partial:
                # Close half position at current price
                half_size = pos.size / 2
                half_margin = pos.margin / 2
                pnl = (price - pos.entry_price) * half_size if pos.side == PositionSide.LONG \
                      else (pos.entry_price - price) * half_size
                fee = price * half_size * self._fee_pct
                net_pnl = pnl - fee
                # Update position in-place (keep other half running)
                pos.size   -= half_size
                pos.margin -= half_margin
                pos.realized_pnl += net_pnl
                self._balance_usdt += half_margin + net_pnl
                self._total_fees_paid += fee
                pos.partial_tp_taken = True
                # Move SL to entry (free ride on remaining half)
                if pos.side == PositionSide.LONG:
                    pos.sl_price = max(pos.entry_price, pos.sl_price)
                else:
                    pos.sl_price = min(pos.entry_price, pos.sl_price)
                pos.breakeven_triggered = True
                logger.info("Partial TP taken (50%)", symbol=symbol,
                            price=round(price, 4), partial_pnl=round(net_pnl, 4),
                            sl_moved_to=round(pos.sl_price, 4))
                self._pending_closes.append({
                    "symbol": symbol, "side": pos.side.value,
                    "size": half_size, "entry_price": pos.entry_price,
                    "exit_price": price, "sl_price": pos.sl_price, "tp_price": pos.tp_price,
                    "pnl": round(net_pnl, 4), "pnl_pct": round(net_pnl / half_margin, 6) if half_margin else 0,
                    "fee": round(fee, 4), "close_reason": "partial_take_profit",
                    "opened_at": pos.opened_at, "closed_at": int(time.time() * 1000),
                    "balance_after": round(self._balance_usdt, 4),
                })

        # ── Time-based exit: close if max_hold_ms exceeded ──────────────────
        if pos.max_hold_ms > 0:
            held_ms = int(time.time() * 1000) - pos.opened_at
            if held_ms >= pos.max_hold_ms:
                logger.info("Time-based exit triggered", symbol=symbol,
                            held_hours=round(held_ms / 3_600_000, 1))
                self._close_position(symbol, price, reason="time_exit")
                return

        if pos.tp_price and (
            (pos.side == PositionSide.LONG and price >= pos.tp_price)
            or (pos.side == PositionSide.SHORT and price <= pos.tp_price)
        ):
            logger.info("TP triggered", symbol=symbol, price=price, tp=pos.tp_price)
            self._close_position(symbol, price, reason="take_profit")
        elif pos.sl_price and (
            (pos.side == PositionSide.LONG and price <= pos.sl_price)
            or (pos.side == PositionSide.SHORT and price >= pos.sl_price)
        ):
            if pos.trailing_active:
                reason = "trailing_stop"
            elif pos.breakeven_triggered:
                reason = "breakeven_stop"
            else:
                reason = "stop_loss"
            logger.info("SL triggered", symbol=symbol, price=price, sl=pos.sl_price, reason=reason)
            self._close_position(symbol, price, reason=reason)

    def _check_limit_orders(self, symbol: str, price: float) -> None:
        for oid, order in list(self._open_orders.items()):
            if order.symbol != symbol or order.order_type != OrderType.LIMIT:
                continue
            filled = False
            if order.side == OrderSide.BUY and price <= order.price:
                filled = True
            elif order.side == OrderSide.SELL and price >= order.price:
                filled = True
            if filled:
                self._fill_order(order, order.price)

    # ── Order Management ──────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: float,
        price: float | None = None,
        sl_price: float | None = None,
        tp_price: float | None = None,
        reduce_only: bool = False,
        trailing_activation: float = 0.0,
        trailing_distance: float = 0.0,
        breakeven_activation: float = 0.0,
        partial_tp_price: float = 0.0,
        max_hold_ms: int = 0,
    ) -> Order:
        mark = self._prices.get(symbol) or price
        if mark is None:
            raise ValueError(f"No price available for {symbol} — feed may not be running")
        # Seed the price cache so SL/TP checks work immediately after order placement
        if symbol not in self._prices:
            self._prices[symbol] = mark

        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        exec_price = price or mark

        # Store all exit params keyed by order_id for use when the order fills
        self._order_trailing[order_id] = (trailing_activation, trailing_distance, breakeven_activation, partial_tp_price, max_hold_ms)

        order = Order(
            order_id=order_id,
            client_order_id=f"ca_{order_id}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.OPEN,
            price=exec_price,
            size=size,
            sl_price=sl_price or 0.0,
            tp_price=tp_price or 0.0,
            reduce_only=reduce_only,
            timestamp=int(time.time() * 1000),
        )

        if order_type == OrderType.MARKET:
            # Apply slippage to market orders
            slip = exec_price * self._slippage_pct
            fill_price = exec_price + slip if side == OrderSide.BUY else exec_price - slip
            self._fill_order(order, fill_price)
        else:
            self._open_orders[order_id] = order

        logger.info(
            "Paper order placed",
            id=order_id, symbol=symbol, side=side.value,
            type=order_type.value, size=size, price=exec_price,
        )
        return order

    def _fill_order(self, order: Order, fill_price: float) -> None:
        notional = fill_price * order.size
        fee = notional * self._fee_pct
        settings = get_settings()
        trailing_activation, trailing_distance, breakeven_activation, partial_tp_price, max_hold_ms = \
            self._order_trailing.pop(order.order_id, (0.0, 0.0, 0.0, 0.0, 0))

        if not order.reduce_only:
            # If a position already exists for this symbol, close it first
            # to avoid silent overwrites that skip the pending-closes queue.
            if order.symbol in self._positions:
                self._close_position(order.symbol, fill_price, reason="replaced_by_new")

            # Open new position
            margin_required = notional / settings.default_leverage
            total_cost = margin_required + fee

            if total_cost > self._balance_usdt:
                order.status = OrderStatus.REJECTED
                logger.warning(
                    "Paper order rejected — insufficient balance",
                    required=total_cost, available=self._balance_usdt,
                )
                return

            self._balance_usdt -= total_cost
            self._total_fees_paid += fee

            side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
            pos = PaperPosition(
                symbol=order.symbol,
                side=side,
                size=order.size,
                entry_price=fill_price,
                leverage=settings.default_leverage,
                margin=margin_required,
                sl_price=order.sl_price,
                tp_price=order.tp_price,
                opened_at=int(time.time() * 1000),
                trailing_activation=trailing_activation,
                trailing_distance=trailing_distance,
                best_price=fill_price,
                breakeven_activation=breakeven_activation,
                partial_tp_price=partial_tp_price,
                max_hold_ms=max_hold_ms,
            )
            self._positions[order.symbol] = pos
        else:
            # Close / reduce position
            self._close_position(order.symbol, fill_price)

        order.status = OrderStatus.FILLED
        order.filled_size = order.size
        order.avg_fill_price = fill_price
        order.fee = fee
        order.update_timestamp = int(time.time() * 1000)

        self._open_orders.pop(order.order_id, None)
        self._order_history.append(order)

        logger.info(
            "Paper order filled",
            id=order.order_id, symbol=order.symbol,
            fill_price=fill_price, fee=fee,
        )

    def _close_position(
        self, symbol: str, close_price: float, reason: str = "manual"
    ) -> float:
        pos = self._positions.pop(symbol, None)
        if not pos:
            return 0.0

        pnl = pos.unrealized_pnl(close_price)
        fee = close_price * pos.size * self._fee_pct
        net_pnl = pnl - fee

        self._balance_usdt += pos.margin + net_pnl
        self._total_fees_paid += fee

        pnl_pct = net_pnl / pos.margin if pos.margin else 0.0

        # Queue for async processing (DB write + Telegram alert)
        self._pending_closes.append({
            "symbol": symbol,
            "side": pos.side.value,
            "size": pos.size,
            "entry_price": pos.entry_price,
            "exit_price": close_price,
            "sl_price": pos.sl_price,
            "tp_price": pos.tp_price,
            "pnl": round(net_pnl, 4),
            "pnl_pct": round(pnl_pct, 6),
            "fee": round(fee, 4),
            "close_reason": reason,
            "opened_at": pos.opened_at or int(time.time() * 1000),
            "closed_at": int(time.time() * 1000),
            "balance_after": round(self._balance_usdt, 4),
        })

        logger.info(
            "Paper position closed",
            symbol=symbol, reason=reason,
            pnl=round(net_pnl, 4), close_price=close_price,
        )
        return net_pnl

    def drain_pending_closes(self) -> list[dict]:
        """Return and clear all closed trade events for async DB/alert processing."""
        events, self._pending_closes = self._pending_closes, []
        return events

    def cancel_order(self, order_id: str) -> bool:
        order = self._open_orders.pop(order_id, None)
        if order:
            order.status = OrderStatus.CANCELLED
            self._order_history.append(order)
            self._order_trailing.pop(order_id, None)
            return True
        return False

    # ── Portfolio State ───────────────────────────────────────────────────────

    def get_balance(self) -> Balance:
        unrealized = sum(
            p.unrealized_pnl(self._prices.get(sym, p.entry_price))
            for sym, p in self._positions.items()
        )
        total = self._balance_usdt + unrealized
        self._peak_equity = max(self._peak_equity, total)
        return Balance(
            currency="USDT",
            total=total,
            available=self._balance_usdt,
            frozen=sum(p.margin for p in self._positions.values()),
            unrealized_pnl=unrealized,
        )

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        items = (
            [(symbol, self._positions[symbol])]
            if symbol and symbol in self._positions
            else self._positions.items()
        )
        return [
            p.to_position(self._prices.get(sym, p.entry_price))
            for sym, p in items
        ]

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        orders = list(self._open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_stats(self) -> dict:
        bal = self.get_balance()
        starting = self._settings.paper_starting_balance
        return {
            "starting_balance": starting,
            "current_equity": bal.total,
            "pnl": bal.total - starting,
            "pnl_pct": (bal.total - starting) / starting * 100,
            "peak_equity": self._peak_equity,
            "max_drawdown_pct": (self._peak_equity - bal.total) / self._peak_equity * 100,
            "total_fees_paid": self._total_fees_paid,
            "open_positions": len(self._positions),
            "total_trades": len(self._order_history),
        }
