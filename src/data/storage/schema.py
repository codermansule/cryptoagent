"""
TimescaleDB schema definitions and async database client.
Run scripts/init_db.sql first to create the hypertables.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

from src.core.config import get_settings
from src.core.logging import get_logger
from src.exchanges.base import Candle, Order, Position, Trade

logger = get_logger(__name__)


# ── SQL ───────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
-- Candlestick hypertable
CREATE TABLE IF NOT EXISTS candles (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    closed      BOOLEAN DEFAULT TRUE
);
SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS candles_sym_tf_time
    ON candles (symbol, timeframe, time DESC);

-- Trades hypertable
CREATE TABLE IF NOT EXISTS trades (
    time    TIMESTAMPTZ NOT NULL,
    symbol  TEXT        NOT NULL,
    price   DOUBLE PRECISION,
    size    DOUBLE PRECISION,
    side    TEXT
);
SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

-- Orders table (flat, not time-series)
CREATE TABLE IF NOT EXISTS orders (
    order_id            TEXT PRIMARY KEY,
    client_order_id     TEXT,
    symbol              TEXT,
    side                TEXT,
    order_type          TEXT,
    status              TEXT,
    price               DOUBLE PRECISION,
    size                DOUBLE PRECISION,
    filled_size         DOUBLE PRECISION DEFAULT 0,
    avg_fill_price      DOUBLE PRECISION DEFAULT 0,
    fee                 DOUBLE PRECISION DEFAULT 0,
    sl_price            DOUBLE PRECISION DEFAULT 0,
    tp_price            DOUBLE PRECISION DEFAULT 0,
    reduce_only         BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ
);

-- Positions snapshot hypertable
CREATE TABLE IF NOT EXISTS positions (
    time                TIMESTAMPTZ NOT NULL,
    symbol              TEXT        NOT NULL,
    side                TEXT,
    size                DOUBLE PRECISION,
    entry_price         DOUBLE PRECISION,
    mark_price          DOUBLE PRECISION,
    liquidation_price   DOUBLE PRECISION,
    unrealized_pnl      DOUBLE PRECISION,
    realized_pnl        DOUBLE PRECISION,
    leverage            INT,
    margin              DOUBLE PRECISION
);
SELECT create_hypertable('positions', 'time', if_not_exists => TRUE);

-- Signal log hypertable
CREATE TABLE IF NOT EXISTS signals (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    timeframe   TEXT,
    direction   TEXT,
    confidence  DOUBLE PRECISION,
    signal_data JSONB
);
SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    total_equity    DOUBLE PRECISION,
    available       DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    num_positions   INT,
    mode            TEXT
);
SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

-- Paper trades: complete round-trip records with P&L
CREATE TABLE IF NOT EXISTS paper_trades (
    id              SERIAL PRIMARY KEY,
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ NOT NULL,
    symbol          TEXT        NOT NULL,
    side            TEXT,
    size            DOUBLE PRECISION,
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    sl_price        DOUBLE PRECISION DEFAULT 0,
    tp_price        DOUBLE PRECISION DEFAULT 0,
    pnl             DOUBLE PRECISION,
    pnl_pct         DOUBLE PRECISION,
    fee             DOUBLE PRECISION DEFAULT 0,
    close_reason    TEXT,
    balance_after   DOUBLE PRECISION
);
"""


# ── Database Client ───────────────────────────────────────────────────────────

class Database:
    """Async PostgreSQL/TimescaleDB client."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._settings.database.timescale_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Database pool created")

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")

    async def initialize_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLES_SQL)
        logger.info("Database schema initialized")

    # ── Candles ───────────────────────────────────────────────────────────────

    async def insert_candles(self, candles: list[Candle]) -> None:
        if not candles:
            return
        rows = [
            (
                _ms_to_ts(c.timestamp), c.symbol, c.timeframe,
                c.open, c.high, c.low, c.close, c.volume, c.closed,
            )
            for c in candles
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO candles (time, symbol, timeframe, open, high, low, close, volume, closed)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (symbol, timeframe, time) DO UPDATE
                SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                    close=EXCLUDED.close, volume=EXCLUDED.volume, closed=EXCLUDED.closed
                """,
                rows,
            )

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, open, high, low, close, volume, closed
                FROM candles
                WHERE symbol=$1 AND timeframe=$2 AND closed=TRUE
                ORDER BY time DESC LIMIT $3
                """,
                symbol, timeframe, limit,
            )
        return [dict(r) for r in reversed(rows)]

    # ── Orders ────────────────────────────────────────────────────────────────

    async def upsert_order(self, order: Order) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orders (
                    order_id, client_order_id, symbol, side, order_type, status,
                    price, size, filled_size, avg_fill_price, fee,
                    sl_price, tp_price, reduce_only, created_at, updated_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,NOW(),NOW())
                ON CONFLICT (order_id) DO UPDATE
                SET status=EXCLUDED.status, filled_size=EXCLUDED.filled_size,
                    avg_fill_price=EXCLUDED.avg_fill_price, fee=EXCLUDED.fee,
                    updated_at=NOW()
                """,
                order.order_id, order.client_order_id, order.symbol,
                order.side.value, order.order_type.value, order.status.value,
                order.price, order.size, order.filled_size, order.avg_fill_price,
                order.fee, order.sl_price, order.tp_price, order.reduce_only,
            )

    # ── Positions ─────────────────────────────────────────────────────────────

    async def insert_position_snapshot(self, pos: Position) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO positions (time, symbol, side, size, entry_price,
                    mark_price, liquidation_price, unrealized_pnl, realized_pnl,
                    leverage, margin)
                VALUES (NOW(),$1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                pos.symbol, pos.side.value, pos.size, pos.entry_price,
                pos.mark_price, pos.liquidation_price, pos.unrealized_pnl,
                pos.realized_pnl, pos.leverage, pos.margin,
            )

    # ── Signals ───────────────────────────────────────────────────────────────

    async def log_signal(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        confidence: float,
        data: dict,
    ) -> None:
        import json as _json
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO signals (time, symbol, timeframe, direction, confidence, signal_data)
                VALUES (NOW(),$1,$2,$3,$4,$5)
                """,
                symbol, timeframe, direction, confidence, _json.dumps(data),
            )

    # ── Portfolio ─────────────────────────────────────────────────────────────

    async def log_portfolio_snapshot(
        self,
        total_equity: float,
        available: float,
        unrealized_pnl: float,
        num_positions: int,
        mode: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO portfolio_snapshots
                    (time, total_equity, available, unrealized_pnl, num_positions, mode)
                VALUES (NOW(),$1,$2,$3,$4,$5)
                """,
                total_equity, available, unrealized_pnl, num_positions, mode,
            )


    async def log_paper_trade(self, trade: dict) -> None:
        """Persist a closed paper trade round-trip record."""
        from datetime import datetime, timezone

        def _ms(ms: int):
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO paper_trades (
                    opened_at, closed_at, symbol, side, size,
                    entry_price, exit_price, sl_price, tp_price,
                    pnl, pnl_pct, fee, close_reason, balance_after
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                """,
                _ms(trade["opened_at"]), _ms(trade["closed_at"]),
                trade["symbol"], trade["side"], trade["size"],
                trade["entry_price"], trade["exit_price"],
                trade.get("sl_price", 0.0), trade.get("tp_price", 0.0),
                trade["pnl"], trade.get("pnl_pct", 0.0), trade.get("fee", 0.0),
                trade["close_reason"], trade.get("balance_after", 0.0),
            )


    async def get_open_paper_positions(self) -> list[dict]:
        """
        Return filled orders that have no matching paper_trades close event.
        These are positions that were open when the agent last stopped —
        used to recover in-memory state on restart.
        """
        rows = await self._pool.fetch(
            """
            SELECT o.symbol, o.side, o.avg_fill_price AS entry_price,
                   o.filled_size AS size, o.sl_price, o.tp_price,
                   o.created_at
            FROM orders o
            WHERE o.status = 'filled'
              AND NOT EXISTS (
                SELECT 1 FROM paper_trades pt
                WHERE pt.symbol = o.symbol
                  AND pt.opened_at >= o.created_at - interval '30 seconds'
                  AND pt.opened_at <= o.created_at + interval '30 seconds'
              )
            ORDER BY o.created_at ASC
            """
        )
        return [dict(r) for r in rows]

    async def get_last_paper_balance(self) -> tuple[float, object] | None:
        """Return (balance_after, closed_at) from the most recent paper trade, or None."""
        row = await self._pool.fetchrow(
            "SELECT balance_after, closed_at FROM paper_trades ORDER BY closed_at DESC LIMIT 1"
        )
        return (float(row["balance_after"]), row["closed_at"]) if row else None


def _ms_to_ts(ms: int):
    """Convert Unix milliseconds to a Python datetime-compatible string for asyncpg."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
