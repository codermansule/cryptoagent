"""
Abstract base exchange interface.
All exchange adapters must implement this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncIterator, Callable, Union


# Type for callbacks that can be sync or async
Callback = Callable[..., Union[None, Any]]


# ── Enums ─────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NET = "net"


class MarginMode(str, Enum):
    ISOLATED = "isolated"
    CROSS = "cross"


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class Candle:
    symbol: str
    timeframe: str
    timestamp: int           # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


@dataclass
class OrderBook:
    symbol: str
    timestamp: int
    bids: list[tuple[float, float]]   # [(price, size), ...]
    asks: list[tuple[float, float]]


@dataclass
class Trade:
    symbol: str
    timestamp: int
    price: float
    size: float
    side: OrderSide


@dataclass
class Ticker:
    symbol: str
    timestamp: int
    bid: float
    ask: float
    last: float
    mark_price: float
    index_price: float
    funding_rate: float
    open_interest: float
    volume_24h: float
    change_24h_pct: float


@dataclass
class Order:
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    price: float
    size: float
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    fee: float = 0.0
    timestamp: int = 0
    update_timestamp: int = 0
    reduce_only: bool = False
    sl_price: float = 0.0
    tp_price: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: int
    margin_mode: MarginMode
    margin: float
    timestamp: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.size * self.mark_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return (self.mark_price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - self.mark_price) / self.entry_price * 100


@dataclass
class Balance:
    currency: str
    total: float
    available: float
    frozen: float
    unrealized_pnl: float = 0.0
    margin_ratio: float = 0.0


# ── Abstract Exchange ─────────────────────────────────────────────────────────

class BaseExchange(ABC):
    """All exchange adapters implement this interface."""

    name: str = "base"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connections (REST session + WebSocket)."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly close all connections."""

    # ── Market Data (REST) ─────────────────────────────────────────────────────

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        since: int | None = None,
    ) -> list[Candle]:
        """Fetch historical OHLCV candles."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker for a symbol."""

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get current order book snapshot."""

    @abstractmethod
    async def get_instruments(self) -> list[dict]:
        """Get list of tradable instruments with contract specs."""

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current perpetual funding rate."""

    @abstractmethod
    async def get_open_interest(self, symbol: str) -> float:
        """Get current open interest."""

    # ── Account (REST) ─────────────────────────────────────────────────────────

    @abstractmethod
    async def get_balance(self) -> list[Balance]:
        """Get account balances."""

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions."""

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""

    @abstractmethod
    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        """Set margin mode (isolated / cross)."""

    # ── Trading (REST) ─────────────────────────────────────────────────────────

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: float,
        price: float | None = None,
        sl_price: float | None = None,
        tp_price: float | None = None,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit a new order."""

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders. Returns count cancelled."""

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Fetch single order state."""

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Fetch all open orders."""

    @abstractmethod
    async def get_order_history(
        self, symbol: str | None = None, limit: int = 100
    ) -> list[Order]:
        """Fetch order history."""

    # ── WebSocket Subscriptions ────────────────────────────────────────────────

    @abstractmethod
    async def subscribe_candles(
        self, symbol: str, timeframe: str, callback: Callback
    ) -> None:
        """Subscribe to live candle updates."""

    @abstractmethod
    async def subscribe_orderbook(
        self, symbol: str, callback: Callback
    ) -> None:
        """Subscribe to order book updates."""

    @abstractmethod
    async def subscribe_trades(
        self, symbol: str, callback: Callback
    ) -> None:
        """Subscribe to live trade stream."""

    @abstractmethod
    async def subscribe_orders(
        self, callback: Callback
    ) -> None:
        """Subscribe to private order updates."""

    @abstractmethod
    async def subscribe_positions(
        self, callback: Callback
    ) -> None:
        """Subscribe to private position updates."""
