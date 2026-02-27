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
            logger.info("SL triggered", symbol=symbol, price=price, sl=pos.sl_price)
            self._close_position(symbol, price, reason="stop_loss")

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
    ) -> Order:
        mark = self._prices.get(symbol)
        if mark is None:
            raise ValueError(f"No price available for {symbol} — feed may not be running")

        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        exec_price = price or mark

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
