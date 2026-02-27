-- Run automatically by docker-compose via /docker-entrypoint-initdb.d/
-- Creates TimescaleDB extension and all hypertables.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

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
CREATE UNIQUE INDEX IF NOT EXISTS candles_sym_tf_time ON candles (symbol, timeframe, time DESC);

CREATE TABLE IF NOT EXISTS trades (
    time    TIMESTAMPTZ NOT NULL,
    symbol  TEXT        NOT NULL,
    price   DOUBLE PRECISION,
    size    DOUBLE PRECISION,
    side    TEXT
);
SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

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

CREATE TABLE IF NOT EXISTS signals (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    timeframe   TEXT,
    direction   TEXT,
    confidence  DOUBLE PRECISION,
    signal_data JSONB
);
SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);

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
