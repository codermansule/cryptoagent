"""
Market data feed manager.
Subscribes to live streams and stores data to TimescaleDB + Redis cache.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Union

import redis.asyncio as aioredis

from src.core.config import get_settings
from src.core.logging import get_logger
from src.data.storage.schema import Database
from src.exchanges.base import BaseExchange, Candle, OrderBook, Trade

logger = get_logger(__name__)

# Type for callbacks that can be sync or async
CandleCallback = Callable[[Candle], Union[None, Any]]


class MarketFeedManager:
    """
    Manages live market data subscriptions for all configured symbols.
    Writes candles to TimescaleDB and hot-caches latest prices in Redis.
    """

    def __init__(self, exchange: BaseExchange, db: Database) -> None:
        self._exchange = exchange
        self._db = db
        self._settings = get_settings()
        self._redis: aioredis.Redis | None = None
        self._external_candle_callbacks: list[CandleCallback] = []

    async def start(self) -> None:
        self._redis = await aioredis.from_url(
            self._settings.database.redis_url, decode_responses=True
        )
        await self._subscribe_all()
        logger.info("Market feed manager started")

    async def stop(self) -> None:
        if self._redis:
            await self._redis.close()

    def on_candle(self, callback: CandleCallback) -> None:
        """Register an external callback to be called on every new candle."""
        self._external_candle_callbacks.append(callback)

    async def _subscribe_all(self) -> None:
        symbols = self._settings.symbols
        primary_tf = self._settings.primary_timeframe  # 15m — LTF entry trigger
        # Three-timeframe stack: 15m entry trigger + 1h intermediate + 4h trend bias
        signal_tfs = [primary_tf, "1h", "4h"]

        for symbol in symbols:
            for tf in signal_tfs:
                await self._exchange.subscribe_candles(symbol, tf, self._on_candle)
            await self._exchange.subscribe_orderbook(symbol, self._on_orderbook)
            await self._exchange.subscribe_trades(symbol, self._on_trade)
            logger.info("Subscribed market feeds", symbol=symbol, timeframes=signal_tfs)

    # ── Handlers ──────────────────────────────────────────────────────────────

    async def _on_candle(self, candle: Candle) -> None:
        # Cache latest close in Redis
        if self._redis:
            await self._redis.hset(
                f"price:{candle.symbol}",
                mapping={
                    "close": candle.close,
                    "high": candle.high,
                    "low": candle.low,
                    "volume": candle.volume,
                    "ts": candle.timestamp,
                },
            )

        # Persist closed candles to DB
        if candle.closed:
            await self._db.insert_candles([candle])

        # Notify signal engine
        for cb in self._external_candle_callbacks:
            try:
                result = cb(candle)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("Candle callback error", error=str(exc))

    async def _on_orderbook(self, ob: OrderBook) -> None:
        # Cache best bid/ask
        if self._redis and ob.bids and ob.asks:
            await self._redis.hset(
                f"orderbook:{ob.symbol}",
                mapping={
                    "best_bid": ob.bids[0][0],
                    "best_ask": ob.asks[0][0],
                    "spread": ob.asks[0][0] - ob.bids[0][0],
                    "ts": ob.timestamp,
                },
            )

    async def _on_trade(self, trade: Trade) -> None:
        # Lightweight — just cache last trade
        if self._redis:
            await self._redis.hset(
                f"trade:{trade.symbol}",
                mapping={"price": trade.price, "size": trade.size, "side": trade.side.value},
            )

    # ── Data Access ───────────────────────────────────────────────────────────

    async def get_latest_price(self, symbol: str) -> float | None:
        if self._redis is None:
            return None
        val = await self._redis.hget(f"price:{symbol}", "close")
        return float(val) if val else None

    async def get_spread(self, symbol: str) -> float | None:
        if self._redis is None:
            return None
        val = await self._redis.hget(f"orderbook:{symbol}", "spread")
        return float(val) if val else None

    async def fetch_historical_candles(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> list[Candle]:
        """Pull candles from DB for ML model input."""
        rows = await self._db.get_candles(symbol, timeframe, limit)
        return [
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=int(r["time"].timestamp() * 1000),
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
                closed=r["closed"],
            )
            for r in rows
        ]
