"""
BloFin Exchange Adapter — REST + WebSocket
Docs: https://blofin.com/docs
Authentication: HMAC-SHA256 (API-Key, Timestamp, Nonce, Passphrase, Signature)
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

# ── Timeframe mapping ─────────────────────────────────────────────────────────
_TF_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
    "1d": "1D", "1w": "1W",
}

_ORDER_STATUS_MAP = {
    "live": OrderStatus.OPEN,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
}


class BloFinAdapter(BaseExchange):
    """
    Full BloFin exchange adapter.

    Usage:
        async with BloFinAdapter() as exchange:
            balance = await exchange.get_balance()
    """

    name = "blofin"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._cfg = self._settings.exchange_config["blofin"]

        # Auth credentials
        self._api_key = self._settings.blofin.api_key
        self._api_secret = self._settings.blofin.api_secret
        self._passphrase = self._settings.blofin.passphrase
        self._testnet = self._settings.blofin.testnet

        # REST
        self._base_url: str = self._cfg["rest_base_url"]
        self._session: aiohttp.ClientSession | None = None

        # WebSocket
        self._ws_public_url: str = self._cfg["ws_public_url"]
        self._ws_private_url: str = self._cfg["ws_private_url"]
        self._ws_public: aiohttp.ClientWebSocketResponse | None = None
        self._ws_private: aiohttp.ClientWebSocketResponse | None = None
        self._ws_conn: aiohttp.ClientSession | None = None

        # Callback registries
        self._candle_callbacks: dict[str, Callable] = {}
        self._orderbook_callbacks: dict[str, Callable] = {}
        self._trade_callbacks: dict[str, Callable] = {}
        self._order_callbacks: list[Callable] = []
        self._position_callbacks: list[Callable] = []

        # Stored subscription args — re-sent on public WS reconnect
        self._public_sub_args: list[dict] = []

        # Background tasks
        self._ws_tasks: list[asyncio.Task] = []

    # ── Context Manager ────────────────────────────────────────────────────────

    async def __aenter__(self) -> "BloFinAdapter":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Content-Type": "application/json"}
        )
        self._ws_conn = aiohttp.ClientSession()
        logger.info("BloFin adapter connected", testnet=self._testnet)

    async def disconnect(self) -> None:
        for task in self._ws_tasks:
            task.cancel()
        if self._ws_public and not self._ws_public.closed:
            await self._ws_public.close()
        if self._ws_private and not self._ws_private.closed:
            await self._ws_private.close()
        if self._session:
            await self._session.close()
        if self._ws_conn:
            await self._ws_conn.close()
        logger.info("BloFin adapter disconnected")

    # ── Authentication ─────────────────────────────────────────────────────────

    def _timestamp(self) -> str:
        return str(int(time.time() * 1000))

    def _nonce(self) -> str:
        return str(uuid.uuid4()).replace("-", "")[:32]

    def _sign(self, timestamp: str, nonce: str, method: str, path: str, body: str = "") -> str:
        """
        BloFin signature:
        sign = HMAC-SHA256( timestamp + nonce + method + path + body, secret )
        encoded as base64
        """
        msg = f"{timestamp}{nonce}{method.upper()}{path}{body}"
        sig = hmac.new(
            self._api_secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(sig).decode("utf-8")

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = self._timestamp()
        nonce = self._nonce()
        sign = self._sign(ts, nonce, method, path, body)
        return {
            "ACCESS-KEY": self._api_key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-NONCE": nonce,
            "ACCESS-PASSPHRASE": self._passphrase,
        }

    # ── REST Core ─────────────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None, auth: bool = False) -> Any:
        url = f"{self._base_url}{path}"
        query = f"?{urlencode(params)}" if params else ""
        headers = self._auth_headers("GET", f"{path}{query}") if auth else {}
        
        session = self._session
        assert session is not None, "Session not initialized. Call connect() first."

        for attempt in range(self._settings.retry_attempts):
            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.warning("BloFin GET error", path=path, status=resp.status, body=data)
                        resp.raise_for_status()
                    return data
            except aiohttp.ClientError as exc:
                if attempt == self._settings.retry_attempts - 1:
                    raise
                wait = self._settings.retry_backoff_seconds * (2 ** attempt)
                logger.warning("BloFin GET retry", path=path, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)

    async def _post(self, path: str, body: dict | None = None) -> Any:
        url = f"{self._base_url}{path}"
        payload = json.dumps(body or {})
        headers = self._auth_headers("POST", path, payload)
        
        session = self._session
        assert session is not None, "Session not initialized. Call connect() first."

        for attempt in range(self._settings.retry_attempts):
            try:
                async with session.post(url, data=payload, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        logger.warning("BloFin POST error", path=path, status=resp.status, body=data)
                        resp.raise_for_status()
                    return data
            except aiohttp.ClientError as exc:
                if attempt == self._settings.retry_attempts - 1:
                    raise
                wait = self._settings.retry_backoff_seconds * (2 ** attempt)
                logger.warning("BloFin POST retry", path=path, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)

    # ── Market Data ───────────────────────────────────────────────────────────

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        since: int | None = None,
    ) -> list[Candle]:
        tf = _TF_MAP.get(timeframe, timeframe)
        params: dict = {"instId": symbol, "bar": tf, "limit": str(min(limit, 300))}
        if since:
            params["after"] = str(since)

        path = self._cfg["endpoints"]["candles"]
        raw = await self._get(path, params=params)
        candles = []
        for row in raw.get("data", []):
            # BloFin format: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
            candles.append(Candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                closed=row[8] == "1" if len(row) > 8 else True,
            ))
        return sorted(candles, key=lambda c: c.timestamp)

    async def get_ticker(self, symbol: str) -> Ticker:
        # BloFin uses the tickers endpoint with instId param for single ticker
        path = self._cfg["endpoints"]["tickers"]
        raw = await self._get(path, params={"instId": symbol})
        data = raw.get("data", raw)
        d = data[0] if isinstance(data, list) else data
        last = float(d.get("last", 0))
        open24h = float(d.get("open24h", last))
        change_pct = ((last - open24h) / open24h * 100) if open24h else 0.0
        return Ticker(
            symbol=symbol,
            timestamp=int(d.get("ts", 0)),
            bid=float(d.get("bidPrice", d.get("bidPx", 0))),
            ask=float(d.get("askPrice", d.get("askPx", 0))),
            last=last,
            mark_price=float(d.get("markPrice", d.get("markPx", last))),
            index_price=float(d.get("indexPrice", d.get("idxPx", last))),
            funding_rate=float(d.get("fundingRate", 0)),
            open_interest=float(d.get("openInterest", d.get("oi", 0))),
            volume_24h=float(d.get("vol24h", 0)),
            change_24h_pct=change_pct,
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        path = self._cfg["endpoints"]["orderbook"]
        raw = await self._get(path, params={"instId": symbol, "sz": str(depth)})
        d = raw["data"][0]
        return OrderBook(
            symbol=symbol,
            timestamp=int(d.get("ts", int(time.time() * 1000))),
            bids=[[float(p), float(s)] for p, s, *_ in d.get("bids", [])],
            asks=[[float(p), float(s)] for p, s, *_ in d.get("asks", [])],
        )

    async def get_instruments(self) -> list[dict]:
        path = self._cfg["endpoints"]["instruments"]
        raw = await self._get(path, params={"instType": "SWAP"})
        return raw.get("data", [])

    async def get_funding_rate(self, symbol: str) -> float:
        path = self._cfg["endpoints"]["funding_rate"]
        raw = await self._get(path, params={"instId": symbol})
        return float(raw["data"][0].get("fundingRate", 0))

    async def get_open_interest(self, symbol: str) -> float:
        path = self._cfg["endpoints"]["open_interest"]
        raw = await self._get(path, params={"instId": symbol})
        return float(raw["data"][0].get("oi", 0))

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_balance(self) -> list[Balance]:
        path = self._cfg["endpoints"]["account_balance"]
        raw = await self._get(path, auth=True)
        balances = []
        for item in raw.get("data", []):
            for detail in item.get("details", [item]):
                balances.append(Balance(
                    currency=detail.get("ccy", "USDT"),
                    total=float(detail.get("eq", detail.get("bal", 0))),
                    available=float(detail.get("availEq", detail.get("availBal", 0))),
                    frozen=float(detail.get("frozenBal", 0)),
                    unrealized_pnl=float(detail.get("upl", 0)),
                    margin_ratio=float(detail.get("mgnRatio", 0)),
                ))
        return balances

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        path = self._cfg["endpoints"]["positions"]
        params: dict = {"instType": "SWAP"}
        if symbol:
            params["instId"] = symbol
        raw = await self._get(path, params=params, auth=True)
        positions = []
        for d in raw.get("data", []):
            size = float(d.get("pos", 0))
            if size == 0:
                continue
            side_str = d.get("posSide", "net").lower()
            side = PositionSide(side_str) if side_str in ("long", "short") else PositionSide.NET
            positions.append(Position(
                symbol=d.get("instId", symbol or ""),
                side=side,
                size=abs(size),
                entry_price=float(d.get("avgPx", 0)),
                mark_price=float(d.get("markPx", 0)),
                liquidation_price=float(d.get("liqPx", 0)),
                unrealized_pnl=float(d.get("upl", 0)),
                realized_pnl=float(d.get("realizedPnl", 0)),
                leverage=int(float(d.get("lever", 1))),
                margin_mode=MarginMode(d.get("mgnMode", "isolated")),
                margin=float(d.get("margin", d.get("imr", 0))),
                timestamp=int(d.get("uTime", 0)),
                raw=d,
            ))
        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        path = self._cfg["endpoints"]["set_leverage"]
        await self._post(path, {"instId": symbol, "lever": str(leverage), "mgnMode": "isolated"})
        logger.info("Leverage set", symbol=symbol, leverage=leverage)

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> None:
        path = self._cfg["endpoints"]["set_margin_mode"]
        await self._post(path, {"instId": symbol, "mgnMode": mode.value})
        logger.info("Margin mode set", symbol=symbol, mode=mode.value)

    # ── Trading ───────────────────────────────────────────────────────────────

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
        path = self._cfg["endpoints"]["place_order"]
        cloid = client_order_id or f"ca_{uuid.uuid4().hex[:16]}"

        _type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_MARKET: "market",
            OrderType.STOP_LIMIT: "limit",
        }

        body: dict = {
            "instId": symbol,
            "tdMode": "isolated",
            "side": side.value,
            "ordType": _type_map[order_type],
            "sz": str(size),
            "clOrdId": cloid,
            "reduceOnly": str(reduce_only).lower(),
        }

        if price and order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            body["px"] = str(price)

        if sl_price:
            body["slTriggerPx"] = str(sl_price)
            body["slOrdPx"] = "-1"           # market SL
            body["slTriggerPxType"] = "mark"

        if tp_price:
            body["tpTriggerPx"] = str(tp_price)
            body["tpOrdPx"] = "-1"           # market TP
            body["tpTriggerPxType"] = "mark"

        logger.info(
            "Placing order", symbol=symbol, side=side.value,
            type=order_type.value, size=size, price=price,
        )

        raw = await self._post(path, body)
        d = raw.get("data", [{}])[0]

        return Order(
            order_id=d.get("ordId", ""),
            client_order_id=d.get("clOrdId", cloid),
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.PENDING,
            price=price or 0.0,
            size=size,
            timestamp=int(time.time() * 1000),
            sl_price=sl_price or 0.0,
            tp_price=tp_price or 0.0,
            raw=d,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        path = self._cfg["endpoints"]["cancel_order"]
        raw = await self._post(path, {"instId": symbol, "ordId": order_id})
        success = raw.get("code") == "0"
        logger.info("Cancel order", symbol=symbol, order_id=order_id, success=success)
        return success

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        open_orders = await self.get_open_orders(symbol)
        if not open_orders:
            return 0
        path = self._cfg["endpoints"]["cancel_all_orders"]
        payload = [{"instId": o.symbol, "ordId": o.order_id} for o in open_orders]
        await self._post(path, payload)
        logger.info("Cancelled all orders", count=len(open_orders), symbol=symbol)
        return len(open_orders)

    async def get_order(self, symbol: str, order_id: str) -> Order:
        path = self._cfg["endpoints"]["get_order"]
        raw = await self._get(path, params={"instId": symbol, "ordId": order_id}, auth=True)
        return self._parse_order(raw["data"][0])

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        path = self._cfg["endpoints"]["get_orders"]
        params: dict = {"instType": "SWAP"}
        if symbol:
            params["instId"] = symbol
        raw = await self._get(path, params=params, auth=True)
        return [self._parse_order(d) for d in raw.get("data", [])]

    async def get_order_history(
        self, symbol: str | None = None, limit: int = 100
    ) -> list[Order]:
        path = self._cfg["endpoints"]["get_history"]
        params: dict = {"instType": "SWAP", "limit": str(min(limit, 100))}
        if symbol:
            params["instId"] = symbol
        raw = await self._get(path, params=params, auth=True)
        return [self._parse_order(d) for d in raw.get("data", [])]

    def _parse_order(self, d: dict) -> Order:
        side_str = d.get("side", "buy").lower()
        type_str = d.get("ordType", "limit").lower()
        status_str = d.get("state", "live").lower()

        _type_rev = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP_MARKET,
            "conditional": OrderType.STOP_LIMIT,
        }

        return Order(
            order_id=d.get("ordId", ""),
            client_order_id=d.get("clOrdId", ""),
            symbol=d.get("instId", ""),
            side=OrderSide(side_str),
            order_type=_type_rev.get(type_str, OrderType.LIMIT),
            status=_ORDER_STATUS_MAP.get(status_str, OrderStatus.OPEN),
            price=float(d.get("px", 0)),
            size=float(d.get("sz", 0)),
            filled_size=float(d.get("fillSz", 0)),
            avg_fill_price=float(d.get("avgPx", 0)),
            fee=float(d.get("fee", 0)),
            timestamp=int(d.get("cTime", 0)),
            update_timestamp=int(d.get("uTime", 0)),
            reduce_only=d.get("reduceOnly", "false") == "true",
            sl_price=float(d.get("slTriggerPx", 0)),
            tp_price=float(d.get("tpTriggerPx", 0)),
            raw=d,
        )

    # ── WebSocket — Public ─────────────────────────────────────────────────────

    async def _ensure_public_ws(self) -> tuple[aiohttp.ClientWebSocketResponse, bool]:
        """Returns (ws, is_new). is_new=True means the connection was just created."""
        if self._ws_public is None or self._ws_public.closed:
            self._ws_public = await self._ws_conn.ws_connect(self._ws_public_url)
            return self._ws_public, True
        return self._ws_public, False

    async def _ensure_private_ws(self) -> aiohttp.ClientWebSocketResponse:
        if self._ws_private is None or self._ws_private.closed:
            self._ws_private = await self._ws_conn.ws_connect(self._ws_private_url)
            await self._ws_auth(self._ws_private)
        return self._ws_private

    async def _ws_auth(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # BloFin WS signature format (DIFFERENT from REST):
        #   sign_string = path + method + timestamp_ms + nonce
        #   sign = base64( hexdigest(HMAC-SHA256(sign_string)) )
        ts    = self._timestamp()   # milliseconds
        nonce = self._nonce()
        sign_msg = f"/users/self/verifyGET{ts}{nonce}"
        hex_sig  = hmac.new(
            self._api_secret.encode("utf-8"),
            sign_msg.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest().encode("utf-8")
        sign = base64.b64encode(hex_sig).decode("utf-8")

        msg = {
            "op": "login",
            "args": [{
                "apiKey": self._api_key,
                "passphrase": self._passphrase,
                "timestamp": ts,
                "nonce": nonce,
                "sign": sign,
            }],
        }
        await ws.send_str(json.dumps(msg))
        resp = await ws.receive_str(timeout=5)
        data = json.loads(resp)
        if data.get("event") != "login":
            raise RuntimeError(f"BloFin WS auth failed: {data}")
        logger.info("BloFin private WebSocket authenticated")

    async def _ws_send(self, ws: aiohttp.ClientWebSocketResponse, msg: dict) -> None:
        await ws.send_str(json.dumps(msg))

    def _ensure_public_listener(self) -> None:
        """Start a single shared public WebSocket listener (idempotent)."""
        if not any(
            not t.done() and getattr(t, "_public_listener", False)
            for t in self._ws_tasks
        ):
            task = asyncio.create_task(self._public_listener_loop())
            task._public_listener = True   # type: ignore[attr-defined]
            self._ws_tasks.append(task)

    def _ensure_private_listener(self) -> None:
        """Start a single shared private WebSocket listener (idempotent)."""
        if not any(
            not t.done() and getattr(t, "_private_listener", False)
            for t in self._ws_tasks
        ):
            task = asyncio.create_task(self._private_listener())
            task._private_listener = True   # type: ignore[attr-defined]
            self._ws_tasks.append(task)

    async def subscribe_candles(
        self, symbol: str, timeframe: str, callback: Callable
    ) -> None:
        tf = _TF_MAP.get(timeframe, timeframe)
        channel = f"candle{tf}"
        key = f"{symbol}:{timeframe}"
        self._candle_callbacks[key] = callback

        arg = {"channel": channel, "instId": symbol}
        if arg not in self._public_sub_args:
            self._public_sub_args.append(arg)

        ws, _ = await self._ensure_public_ws()
        await self._ws_send(ws, {"op": "subscribe", "args": [arg]})
        self._ensure_public_listener()
        logger.info("Subscribed to candles", symbol=symbol, timeframe=timeframe)

    async def subscribe_orderbook(
        self, symbol: str, callback: Callable
    ) -> None:
        self._orderbook_callbacks[symbol] = callback

        arg = {"channel": "books5", "instId": symbol}
        if arg not in self._public_sub_args:
            self._public_sub_args.append(arg)

        ws, _ = await self._ensure_public_ws()
        await self._ws_send(ws, {"op": "subscribe", "args": [arg]})
        self._ensure_public_listener()
        logger.info("Subscribed to order book", symbol=symbol)

    async def subscribe_trades(
        self, symbol: str, callback: Callable
    ) -> None:
        self._trade_callbacks[symbol] = callback

        arg = {"channel": "trades", "instId": symbol}
        if arg not in self._public_sub_args:
            self._public_sub_args.append(arg)

        ws, _ = await self._ensure_public_ws()
        await self._ws_send(ws, {"op": "subscribe", "args": [arg]})
        self._ensure_public_listener()
        logger.info("Subscribed to trades", symbol=symbol)

    async def subscribe_orders(self, callback: Callable) -> None:
        self._order_callbacks.append(callback)
        ws = await self._ensure_private_ws()
        await self._ws_send(ws, {"op": "subscribe", "args": [{"channel": "orders", "instType": "SWAP"}]})
        self._ensure_private_listener()
        logger.info("Subscribed to private order updates")

    async def subscribe_positions(self, callback: Callable) -> None:
        self._position_callbacks.append(callback)
        ws = await self._ensure_private_ws()
        await self._ws_send(ws, {"op": "subscribe", "args": [{"channel": "positions", "instType": "SWAP"}]})
        logger.info("Subscribed to private position updates")

    # ── WebSocket Listeners ────────────────────────────────────────────────────

    async def _public_listener_loop(self) -> None:
        """Single shared listener for ALL public WebSocket channels."""
        while True:
            try:
                ws, is_new = await self._ensure_public_ws()
                # Only batch re-subscribe when the connection was freshly created
                # (i.e. a reconnect). On first start the individual subscribe_*
                # calls have already sent their own subscribe messages.
                if is_new and self._public_sub_args:
                    await self._ws_send(ws, {"op": "subscribe", "args": self._public_sub_args})
                    logger.info("Re-subscribed public channels", count=len(self._public_sub_args))
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        # Application-level ping/pong required by BloFin
                        if data.get("op") == "ping" or data.get("event") == "ping":
                            await self._ws_send(ws, {"op": "pong"})
                            continue
                        await self._dispatch_public(data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        logger.warning("Public WS closed, reconnecting")
                        self._ws_public = None
                        break
                    elif msg.type == aiohttp.WSMsgType.PING:
                        await ws.pong(msg.data)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "Public WS error, reconnecting",
                    error=str(exc),
                    exc_type=type(exc).__name__,
                    ws_closed=self._ws_public.closed if self._ws_public else True,
                    ws_close_code=self._ws_public.close_code if self._ws_public else None,
                )
                self._ws_public = None
                await asyncio.sleep(3)

    async def _private_listener(self) -> None:
        """Persistent listener for private WebSocket messages."""
        while True:
            try:
                ws = await self._ensure_private_ws()
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(msg.data)
                    await self._dispatch_private(data)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Private WS error, reconnecting", error=str(exc))
                await asyncio.sleep(2)

    async def _dispatch_public(self, data: dict) -> None:
        # Log any error events from BloFin for diagnostics
        if data.get("event") == "error":
            logger.warning("BloFin WS error event", code=data.get("code"), msg=data.get("msg"))
            return

        channel = data.get("arg", {}).get("channel", "")
        symbol = data.get("arg", {}).get("instId", "")
        payload = data.get("data", [])
        if not payload:
            return

        if channel.startswith("candle"):
            tf_label = channel.replace("candle", "")
            # Reverse map BloFin tf label back to standard
            _rev = {v: k for k, v in _TF_MAP.items()}
            timeframe = _rev.get(tf_label, tf_label.lower())
            key = f"{symbol}:{timeframe}"
            cb = self._candle_callbacks.get(key)
            if cb:
                for row in payload:
                    candle = Candle(
                        symbol=symbol, timeframe=timeframe,
                        timestamp=int(row[0]),
                        open=float(row[1]), high=float(row[2]),
                        low=float(row[3]), close=float(row[4]),
                        volume=float(row[5]),
                        closed=row[8] == "1" if len(row) > 8 else False,
                    )
                    await self._maybe_await(cb, candle)

        elif channel.startswith("books"):
            cb = self._orderbook_callbacks.get(symbol)
            if cb:
                # books5 returns data as a dict; other book channels may return a list
                d = payload if isinstance(payload, dict) else payload[0]
                ob = OrderBook(
                    symbol=symbol,
                    timestamp=int(d.get("ts", time.time() * 1000)),
                    bids=[[float(p), float(s)] for p, s, *_ in d.get("bids", [])],
                    asks=[[float(p), float(s)] for p, s, *_ in d.get("asks", [])],
                )
                await self._maybe_await(cb, ob)

        elif channel == "trades":
            cb = self._trade_callbacks.get(symbol)
            if cb:
                for t in payload:
                    trade = Trade(
                        symbol=symbol,
                        timestamp=int(t.get("ts", 0)),
                        price=float(t.get("px", 0)),
                        size=float(t.get("sz", 0)),
                        side=OrderSide(t.get("side", "buy")),
                    )
                    await self._maybe_await(cb, trade)

    async def _dispatch_private(self, data: dict) -> None:
        channel = data.get("arg", {}).get("channel", "")
        payload = data.get("data", [])
        if not payload:
            return

        if channel == "orders":
            for d in payload:
                order = self._parse_order(d)
                for cb in self._order_callbacks:
                    await self._maybe_await(cb, order)

        elif channel == "positions":
            for d in payload:
                size = float(d.get("pos", 0))
                side_str = d.get("posSide", "net").lower()
                side = PositionSide(side_str) if side_str in ("long", "short") else PositionSide.NET
                pos = Position(
                    symbol=d.get("instId", ""),
                    side=side,
                    size=abs(size),
                    entry_price=float(d.get("avgPx", 0)),
                    mark_price=float(d.get("markPx", 0)),
                    liquidation_price=float(d.get("liqPx", 0)),
                    unrealized_pnl=float(d.get("upl", 0)),
                    realized_pnl=float(d.get("realizedPnl", 0)),
                    leverage=int(float(d.get("lever", 1))),
                    margin_mode=MarginMode(d.get("mgnMode", "isolated")),
                    margin=float(d.get("imr", 0)),
                    timestamp=int(d.get("uTime", 0)),
                    raw=d,
                )
                for cb in self._position_callbacks:
                    await self._maybe_await(cb, pos)

    @staticmethod
    async def _maybe_await(fn: Callable, *args) -> None:
        """Call fn(*args), awaiting if it's a coroutine."""
        import inspect
        result = fn(*args)
        if inspect.isawaitable(result):
            await result
