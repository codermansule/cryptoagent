"""
Binance Exchange Adapter — REST + WebSocket
Supports USDT-margined futures (USDT-M)
Docs: https://binance-docs.github.io/apidocs/futures/
Authentication: HMAC-SHA256 (API-Key, Signature, Timestamp)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Callable
from urllib.parse import urlencode

import aiohttp

from src.core.config import get_settings
from src.core.logging import get_logger
from src.exchanges.base import (
    Balance, BaseExchange, Candle, MarginMode, Order, OrderBook,
    OrderSide, OrderStatus, OrderType, Position, PositionSide, Ticker, Trade,
)

logger = get_logger(__name__)

# Timeframe mapping
_TF_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
    "1d": "1d", "1w": "1w",
}

_ORDER_STATUS_MAP = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
}


class BinanceAdapter(BaseExchange):
    """
    Binance USDT-margined futures adapter.
    
    Usage:
        async with BinanceAdapter() as exchange:
            balance = await exchange.get_balance()
    """

    name = "binance"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cfg = self._settings.exchange_config.get("binance", {})
        
        # Use config values or defaults
        self._api_key = self._settings.binance.api_key if hasattr(self._settings, 'binance') else ""
        self._api_secret = self._settings.binance.api_secret if hasattr(self._settings, 'binance') else ""
        
        # REST
        self._base_url: str = self._cfg.get("rest_base_url", "https://fapi.binance.com")
        self._session: aiohttp.ClientSession | None = None

        # WebSocket
        self._ws_url: str = self._cfg.get("ws_base_url", "wss://fstream.binance.com")
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_conn: aiohttp.ClientSession | None = None

        # Callback registries
        self._candle_callbacks: dict[str, Callable] = {}
        self._orderbook_callbacks: dict[str, Callable] = {}
        self._trade_callbacks: dict[str, Callable] = {}
        self._order_callbacks: list[Callable] = []
        self._position_callbacks: list[Callable] = []

        # Stored subscription args
        self._ws_subscriptions: list[dict] = []

        # Background tasks
        self._ws_task: asyncio.Task | None = None

    # ── Context Manager ────────────────────────────────────────────────────────

    async def __aenter__(self) -> "BinanceAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Content-Type": "application/json"}
        )
        self._ws_conn = aiohttp.ClientSession()
        logger.info("Binance adapter connected")

    async def disconnect(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
            self._ws_task = None
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
        if self._ws_conn:
            await self._ws_conn.close()
        logger.info("Binance adapter disconnected")

    # ── Authentication ─────────────────────────────────────────────────────

    def _timestamp(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> str:
        """Generate HMAC-SHA256 signature."""
        query = urlencode(sorted(params.items()))
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, params: dict) -> dict[str, str]:
        params["timestamp"] = self._timestamp()
        params["signature"] = self._sign(params)
        return {"X-MBX-APIKEY": self._api_key}

    # ── REST Core ─────────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None, auth: bool = False) -> Any:
        url = f"{self._base_url}{path}"
        query = f"?{urlencode(params)}" if params else ""
        headers = self._auth_headers(params) if auth else {}

        for attempt in range(self._settings.retry_attempts):
            try:
                async with self._session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.warning("Binance GET error", path=path, status=resp.status, body=data)
                        resp.raise_for_status()
                    return data
            except aiohttp.ClientError as exc:
                if attempt == self._settings.retry_attempts - 1:
                    raise
                wait = self._settings.retry_backoff_seconds * (2 ** attempt)
                logger.warning("Binance GET retry", path=path, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)

    async def _post(self, path: str, body: dict | None = None, auth: bool = False) -> Any:
        url = f"{self._base_url}{path}"
        payload = json.dumps(body) if body else ""
        headers = self._auth_headers(body or {}) if auth else {}
        headers["Content-Type"] = "application/json"

        for attempt in range(self._settings.retry_attempts):
            try:
                async with self._session.post(url, data=payload, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.warning("Binance POST error", path=path, status=resp.status, body=data)
                        resp.raise_for_status()
                    return data
            except aiohttp.ClientError as exc:
                if attempt == self._settings.retry_attempts - 1:
                    raise
                wait = self._settings.retry_backoff_seconds * (2 ** attempt)
                logger.warning("Binance POST retry", path=path, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)

    # ── Market Data ───────────────────────────────────────────────────────

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        since: int | None = None,
    ) -> list[Candle]:
        tf = _TF_MAP.get(timeframe, timeframe)
        params: dict = {"symbol": symbol.replace("-", ""), "interval": tf, "limit": min(limit, 1500)}
        if since:
            params["startTime"] = since

        path = "/fapi/v1/klines"
        raw = await self._get(path, params=params)
        candles = []
        for row in raw:
            candles.append(Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                closed=True,
            ))
        return sorted(candles, key=lambda c: c.timestamp)

    async def get_ticker(self, symbol: str) -> Ticker:
        bin_symbol = symbol.replace("-", "")
        path = "/fapi/v1/ticker/24hr"
        raw = await self._get(path, params={"symbol": bin_symbol})
        last = float(raw.get("lastPrice", 0))
        open_price = float(raw.get("openPrice", last))
        change_pct = float(raw.get("priceChangePercent", 0))
        return Ticker(
            symbol=symbol,
            timestamp=int(raw.get("closeTime", time.time() * 1000)),
            bid=float(raw.get("bidPrice", 0)),
            ask=float(raw.get("askPrice", 0)),
            last=last,
            mark_price=float(raw.get("markPrice", last)),
            index_price=float(raw.get("indexPrice", last)),
            funding_rate=float(raw.get("fundingRate", 0)),
            open_interest=float(raw.get("openInterest", 0)),
            volume_24h=float(raw.get("volume", 0)),
            change_24h_pct=change_pct,
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        bin_symbol = symbol.replace("-", "")
        path = "/fapi/v1/depth"
        raw = await self._get(path, params={"symbol": bin_symbol, "limit": depth})
        return OrderBook(
            symbol=symbol,
            timestamp=int(raw.get("E", time.time() * 1000)),
            bids=[[float(p), float(s)] for p, s, *_ in raw.get("bids", [])],
            asks=[[float(p), float(s)] for p, s, *_ in raw.get("asks", [])],
        )

    async def get_instruments(self) -> list[dict]:
        path = "/fapi/v1/exchangeInfo"
        raw = await self._get(path)
        return [s for s in raw.get("symbols", []) if s.get("contractType") == "PERPETUAL"]

    async def get_funding_rate(self, symbol: str) -> float:
        bin_symbol = symbol.replace("-", "")
        path = "/fapi/v1/premiumIndex"
        raw = await self._get(path, params={"symbol": bin_symbol})
        return float(raw.get("lastFundingRate", 0))

    async def get_open_interest(self, symbol: str) -> float:
        bin_symbol = symbol.replace("-", "")
        path = "/fapi/v1/openInterest"
        raw = await self._get(path, params={"symbol": bin_symbol})
        return float(raw.get("openInterest", 0))

    # ── Account ─────────────────────────────────────────────────────────────

    async def get_balance(self) -> list[Balance]:
        path = "/fapi/v2/account"
        raw = await self._get(path, auth=True)
        balances = []
        for asset in raw.get("assets", []):
            wallet_balance = float(asset.get("walletBalance", 0))
            if wallet_balance > 0:
                balances.append(Balance(
                    currency=asset.get("asset", "USDT"),
                    total=wallet_balance,
                    available=float(asset.get("availableBalance", wallet_balance)),
                    frozen=float(asset.get("locked", 0)),
                    unrealized_pnl=float(asset.get("unrealizedProfit", 0)),
                    margin_ratio=float(raw.get("totalMaintMargin", 1)) / max(float(raw.get("totalMarginBalance", 1)), 1),
                ))
        return balances

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        path = "/fapi/v2/positionRisk"
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("-", "")
        raw = await self._get(path, params=params, auth=True)
        positions = []
        for d in raw:
            size = float(d.get("positionAmt", 0))
            if size == 0:
                continue
            side_str = d.get("positionSide", "BOTH").lower()
            if side_str == "long":
                side = PositionSide.LONG
            elif side_str == "short":
                side = PositionSide.SHORT
            else:
                side = PositionSide.NET
            positions.append(Position(
                symbol=d.get("symbol", symbol or ""),
                side=side,
                size=abs(size),
                entry_price=float(d.get("entryPrice", 0)),
                mark_price=float(d.get("markPrice", 0)),
                liquidation_price=float(d.get("liquidationPrice", 0)),
                unrealized_pnl=float(d.get("unrealizedProfit", 0)),
                realized_pnl=0.0,
                leverage=int(float(d.get("leverage", 1))),
                margin_mode=MarginMode.ISOLATED if d.get("isolated", False) else MarginMode.CROSS,
                margin=float(d.get("margin", 0)),
                timestamp=int(time.time() * 1000),
                raw=d,
            ))
        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        path = "/fapi/v1/leverage"
        await self._post(path, {
            "symbol": symbol.replace("-", ""),
            "leverage": leverage,
        }, auth=True)
        logger.info("Leverage set", symbol=symbol, leverage=leverage)

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        path = "/fapi/v1/marginType"
        await self._post(path, {
            "symbol": symbol.replace("-", ""),
            "marginType": "isolated" if mode == MarginMode.ISOLATED else "cross",
        }, auth=True)
        logger.info("Margin mode set", symbol=symbol, mode=mode.value)

    # ── Trading ───────────────────────────────────────────────────────────

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
        path = "/fapi/v1/order"
        bin_symbol = symbol.replace("-", "")
        cloid = client_order_id or f"ca_{uuid.uuid4().hex[:16]}"

        _type_map = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_MARKET: "STOP_MARKET",
            OrderType.STOP_LIMIT: "STOP_LIMIT",
        }

        body: dict = {
            "symbol": bin_symbol,
            "side": side.value.upper(),
            "type": _type_map[order_type],
            "quantity": size,
            "newClientOrderId": cloid,
            "reduceOnly": str(reduce_only).lower(),
        }

        if price and order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            body["price"] = str(price)
            body["timeInForce"] = "GTC"

        if sl_price:
            body["stopPrice"] = str(sl_price)
            if order_type == OrderType.MARKET:
                body["type"] = "STOP_MARKET"

        if tp_price:
            body["workingType"] = "MARK_PRICE"

        logger.info(
            "Placing order", symbol=symbol, side=side.value,
            type=order_type.value, size=size, price=price,
        )

        raw = await self._post(path, body, auth=True)

        return Order(
            order_id=raw.get("orderId", ""),
            client_order_id=raw.get("clientOrderId", cloid),
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=_ORDER_STATUS_MAP.get(raw.get("status", "NEW"), OrderStatus.PENDING),
            price=float(raw.get("price", 0)),
            size=float(raw.get("origQty", 0)),
            timestamp=int(raw.get("updateTime", time.time() * 1000)),
            sl_price=float(raw.get("stopPrice", 0)),
            tp_price=float(raw.get("stopPrice", 0)) if tp_price else 0.0,
            raw=raw,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        path = "/fapi/v1/order"
        raw = await self._delete(path, {
            "symbol": symbol.replace("-", ""),
            "orderId": order_id,
        }, auth=True)
        success = raw.get("status") == "CANCELED"
        logger.info("Cancel order", symbol=symbol, order_id=order_id, success=success)
        return success

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        open_orders = await self.get_open_orders(symbol)
        if not open_orders:
            return 0
        for order in open_orders:
            await self.cancel_order(order.symbol, order.order_id)
        logger.info("Cancelled all orders", count=len(open_orders), symbol=symbol)
        return len(open_orders)

    async def _delete(self, path: str, params: dict | None = None, auth: bool = False) -> Any:
        url = f"{self._base_url}{path}"
        headers = self._auth_headers(params) if auth else {}

        for attempt in range(self._settings.retry_attempts):
            try:
                async with self._session.delete(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.warning("Binance DELETE error", path=path, status=resp.status, body=data)
                        resp.raise_for_status()
                    return data
            except aiohttp.ClientError as exc:
                if attempt == self._settings.retry_attempts - 1:
                    raise
                wait = self._settings.retry_backoff_seconds * (2 ** attempt)
                logger.warning("Binance DELETE retry", path=path, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)

    async def get_order(self, symbol: str, order_id: str) -> Order:
        path = "/fapi/v1/order"
        raw = await self._get(path, params={
            "symbol": symbol.replace("-", ""),
            "orderId": order_id,
        }, auth=True)
        return self._parse_order(raw)

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        path = "/fapi/v1/openOrders"
        params = {}
        if symbol:
            params["symbol"] = symbol.replace("-", "")
        raw = await self._get(path, params=params, auth=True)
        return [self._parse_order(d) for d in raw]

    async def get_order_history(
        self, symbol: str | None = None, limit: int = 100
    ) -> list[Order]:
        path = "/fapi/v1/allOrders"
        params = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol.replace("-", "")
        raw = await self._get(path, params=params, auth=True)
        return [self._parse_order(d) for d in raw]

    def _parse_order(self, d: dict) -> Order:
        side_str = d.get("side", "buy").lower()
        type_str = d.get("type", "limit").lower()
        status_str = d.get("status", "new").lower()

        _type_rev = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP_MARKET,
            "stop_market": OrderType.STOP_MARKET,
            "stop_limit": OrderType.STOP_LIMIT,
        }

        return Order(
            order_id=str(d.get("orderId", "")),
            client_order_id=d.get("clientOrderId", ""),
            symbol=d.get("symbol", "").replace("USDT", "-USDT"),
            side=OrderSide(side_str),
            order_type=_type_rev.get(type_str, OrderType.LIMIT),
            status=_ORDER_STATUS_MAP.get(status_str, OrderStatus.OPEN),
            price=float(d.get("price", 0)),
            size=float(d.get("origQty", 0)),
            filled_size=float(d.get("executedQty", 0)),
            avg_fill_price=float(d.get("avgPrice", 0)),
            fee=0.0,
            timestamp=int(d.get("time", 0)),
            update_timestamp=int(d.get("updateTime", 0)),
            reduce_only=d.get("reduceOnly", False),
            sl_price=float(d.get("stopPrice", 0)),
            tp_price=float(d.get("stopPrice", 0)),
            raw=d,
        )

    # ── WebSocket — Market Data ─────────────────────────────────────────────

    async def _ensure_ws(self) -> aiohttp.ClientWebSocketResponse:
        if self._ws is None or self._ws.closed:
            self._ws = await self._ws_conn.ws_connect(f"{self._ws_url}/ws")
            # Re-subscribe to all channels
            if self._ws_subscriptions:
                await self._ws.send_str(json.dumps({
                    "method": "SUBSCRIBE",
                    "params": self._ws_subscriptions,
                    "id": 1,
                }))
            self._ws_task = asyncio.create_task(self._ws_listener())
        return self._ws

    async def _ws_listener(self) -> None:
        while True:
            try:
                ws = self._ws
                if ws is None:
                    break
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        self._dispatch_ws(data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        logger.warning("Binance WS closed, reconnecting")
                        self._ws = None
                        break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Binance WS error", error=str(exc))
                await asyncio.sleep(2)

    def _dispatch_ws(self, data: dict) -> None:
        if "e" not in data:
            return
        
        event_type = data.get("e", "")
        
        if event_type == "kline":
            symbol = data.get("s", "").replace("USDT", "-USDT")
            kline = data.get("k", {})
            tf = _TF_MAP.get(kline.get("i", "1m"), "1m")
            key = f"{symbol}:{tf}"
            cb = self._candle_callbacks.get(key)
            if cb:
                candle = Candle(
                    symbol=symbol,
                    timeframe=tf,
                    timestamp=int(kline.get("t", 0)),
                    open=float(kline.get("o", 0)),
                    high=float(kline.get("h", 0)),
                    low=float(kline.get("l", 0)),
                    close=float(kline.get("c", 0)),
                    volume=float(kline.get("v", 0)),
                    closed=kline.get("x", False),
                )
                asyncio.create_task(self._maybe_await(cb, candle))

        elif event_type == "depthUpdate":
            symbol = data.get("s", "").replace("USDT", "-USDT")
            cb = self._orderbook_callbacks.get(symbol)
            if cb:
                ob = OrderBook(
                    symbol=symbol,
                    timestamp=int(data.get("E", time.time() * 1000)),
                    bids=[[float(p), float(s)] for p, s, *_ in data.get("b", [])],
                    asks=[[float(p), float(s)] for p, s, *_ in data.get("a", [])],
                )
                asyncio.create_task(self._maybe_await(cb, ob))

        elif event_type == "trade":
            symbol = data.get("s", "").replace("USDT", "-USDT")
            cb = self._trade_callbacks.get(symbol)
            if cb:
                trade = Trade(
                    symbol=symbol,
                    timestamp=int(data.get("T", 0)),
                    price=float(data.get("p", 0)),
                    size=float(data.get("q", 0)),
                    side=OrderSide(data.get("m", False) and "sell" or "buy"),
                )
                asyncio.create_task(self._maybe_await(cb, trade))

    async def subscribe_candles(
        self, symbol: str, timeframe: str, callback: Callable
    ) -> None:
        tf = _TF_MAP.get(timeframe, timeframe)
        bin_symbol = symbol.replace("-", "")
        key = f"{symbol}:{timeframe}"
        self._candle_callbacks[key] = callback

        stream = f"{bin_symbol.lower()}@kline_{tf}"
        if stream not in self._ws_subscriptions:
            self._ws_subscriptions.append(stream)

        ws = await self._ensure_ws()
        await ws.send_str(json.dumps({
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": 1,
        }))
        logger.info("Subscribed to candles", symbol=symbol, timeframe=timeframe)

    async def subscribe_orderbook(
        self, symbol: str, callback: Callable
    ) -> None:
        bin_symbol = symbol.replace("-", "")
        self._orderbook_callbacks[symbol] = callback

        stream = f"{bin_symbol.lower()}@depth@100ms"
        if stream not in self._ws_subscriptions:
            self._ws_subscriptions.append(stream)

        ws = await self._ensure_ws()
        await ws.send_str(json.dumps({
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": 1,
        }))
        logger.info("Subscribed to order book", symbol=symbol)

    async def subscribe_trades(
        self, symbol: str, callback: Callable
    ) -> None:
        bin_symbol = symbol.replace("-", "")
        self._trade_callbacks[symbol] = callback

        stream = f"{bin_symbol.lower()}@trade"
        if stream not in self._ws_subscriptions:
            self._ws_subscriptions.append(stream)

        ws = await self._ensure_ws()
        await ws.send_str(json.dumps({
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": 1,
        }))
        logger.info("Subscribed to trades", symbol=symbol)

    async def subscribe_orders(self, callback: Callable) -> None:
        self._order_callbacks.append(callback)
        logger.info("Subscribed to private order updates (not implemented for Binance)")

    async def subscribe_positions(self, callback: Callable) -> None:
        self._position_callbacks.append(callback)
        logger.info("Subscribed to private position updates (not implemented for Binance)")

    @staticmethod
    async def _maybe_await(fn: Callable, *args) -> None:
        import inspect
        result = fn(*args)
        if inspect.iscoroutine(result):
            await result
