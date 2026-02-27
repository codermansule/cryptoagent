# CryptoAgent — Autonomous Crypto Trading Agent

## Vision

Build a high-accuracy, AI-driven autonomous trading agent capable of live trading in **crypto spot**, **crypto perpetual futures**, and **traditional futures** markets. The agent connects natively to exchanges such as **BloFin** and is designed around maximizing prediction accuracy while managing risk with surgical precision.

---

## Core Objectives

- **Maximum Prediction Accuracy** — Ensemble ML/AI models informed by multi-source market data
- **Live Execution** — Real-time order placement, modification, and cancellation via exchange APIs
- **Risk-First Architecture** — No trade executes without passing a multi-layer risk gate
- **Exchange Agnostic Core** — BloFin as primary, designed to plug in other exchanges (Binance, Bybit, OKX, etc.)
- **Full Observability** — Every signal, decision, and execution is logged and auditable

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CryptoAgent                          │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Data Layer  │──▶│ Signal Engine│──▶│ Decision Core  │  │
│  └──────────────┘   └──────────────┘   └────────────────┘  │
│          │                                      │           │
│  ┌──────────────┐                    ┌────────────────────┐ │
│  │  Market Feed │                    │   Risk Manager     │ │
│  └──────────────┘                    └────────────────────┘ │
│                                               │             │
│                                    ┌────────────────────┐   │
│                                    │  Execution Engine  │   │
│                                    └────────────────────┘   │
│                                               │             │
│                                    ┌────────────────────┐   │
│                                    │  Exchange Adapters │   │
│                                    │  (BloFin, Binance) │   │
│                                    └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Implemented Modules

### 1. Data Layer ✅
**Responsibility:** Ingest, normalize, and store all market data in real time.

- **WebSocket feeds** — Live OHLCV, order book (L2), trades via BloFin WebSocket
- **REST polling** — Historical candles, tickers, funding rates, open interest
- **Storage** — TimescaleDB (PostgreSQL) for time-series data, Redis for hot cache
- **Location:** `src/data/feeds/`, `src/data/storage/`, `src/data/preprocessing/`

### 2. Signal Engine ✅
**Responsibility:** Generate trading signals using a multi-strategy ensemble.

#### Technical Analysis Signals
- Trend: EMA crossovers (9/21/50/200), Supertrend, ADX
- Momentum: RSI (14), MACD histogram, Stochastic RSI
- Volatility: Bollinger Bands, ATR, BB Squeeze detection
- Volume: VWAP, CVD (Cumulative Volume Delta), OBV

#### Market Structure Analysis
- Trend direction detection
- Break of Structure (BOS) / Change of Character (CHoCH) identification
- Liquidity sweeps detection
- Fair Value Gaps (FVG) analysis

#### Machine Learning Models
| Model | Status | Purpose |
|---|---|---|
| LightGBM | ✅ Implemented | Directional classifier (-1/0/+1) |
| Walk-forward training | ✅ Implemented | Time-series cross-validation |
| Feature engineering | ✅ Implemented | 50+ technical features |

**Location:** `src/signals/technical/`, `src/signals/ml_models/`, `src/signals/ensemble.py`

### 3. Decision Core ✅
**Responsibility:** Evaluate signals and decide whether to open, hold, scale, or close a position.

- Signal aggregation with confidence scoring (0–100%)
- Multi-timeframe confluence (1m, 5m, 15m, 30m, 1h, 4h, 1D)
- Regime detection (trending_up / trending_down / ranging / volatile / breakout) — adapts strategy per regime
- Entry trigger: Signal score > threshold AND risk gate passes
- Exit logic: ATR-based stop loss, take profit ladder (2:1 RR ratio)

**Location:** `src/decision/core.py`, `src/decision/confluence.py`, `src/decision/regime.py`

### 4. Risk Manager ✅
**Responsibility:** Protect capital. This module can veto any trade.

- **Position Sizing** — Fractional Kelly Criterion
- **Max drawdown circuit breaker** — Halts trading if daily DD exceeds limit (default 5%)
- **Max open positions** — Configurable limit (default 5)
- **Portfolio risk cap** — Total open risk capped as % of portfolio (default 2%)
- **Single trade risk** — Max risk per trade (default 0.5%)

**Location:** Integrated in `src/decision/core.py`

### 5. Execution Engine ✅
**Responsibility:** Submit and manage orders with optimal fill quality.

- **Order types supported:** Market, Limit, Stop-Loss, Take-Profit
- **Order state machine:** Tracks pending → open → partially filled → filled → cancelled
- **Paper trading mode:** Simulated execution with configurable fees and slippage

**Location:** `src/execution/paper.py`

### 6. Exchange Adapters ✅

#### BloFin (Primary) ✅
- REST API: Account info, order management, position management, market data
- WebSocket: Real-time price feed, order updates, position updates
- Endpoints: USDT-margined perpetuals
- Authentication: HMAC-SHA256 API key/secret signing

#### Binance ✅
- REST API for USDT-margined futures
- WebSocket market data streams
- Order and position management

**Location:** `src/exchanges/blofin.py`, `src/exchanges/binance.py`, `src/exchanges/base.py`

### 7. Data Enrichment ✅

#### Sentiment Feed ✅
- Fear & Greed Index integration
- Crypto news aggregation
- Sentiment scoring with confidence modifiers

#### On-Chain Data ✅
- Exchange flow data (placeholder for premium APIs)
- Whale transaction tracking
- On-chain metrics aggregation

**Location:** `src/data/feeds/sentiment.py`, `src/data/feeds/onchain.py`

### 8. Risk & Position Management ✅

#### RL Position Sizer ✅
- PPO-based dynamic position sizing
- Rule-based fallback
- Adapts to regime, volatility, and portfolio risk

**Location:** `src/risk/rl_sizer.py`

### 8. Monitoring & Observability ✅

#### Dashboard Features
- **7 Tabs:** Overview, Charts, Notifications, Positions, Signals, Trades, Logs
- **Real-time prices** from BloFin WebSocket
- **Price charts** with candlestick visualization (Altair)
- **Trade notifications** with color-coded alerts (🟢 buy, 🔴 sell, 🟡 pending)
- **Log viewer** with filtering by level (INFO/WARNING/ERROR)
- **Manual refresh** button (no auto-refresh to avoid page reload)
- **Demo data loader** for testing

#### Alerting & Analytics
- **Telegram alerts** — Trade notifications and system events
- **Structured JSON logging** — Every signal, decision, order, fill logged
- **Performance metrics** — Sharpe, Sortino, win rate, expectancy, max DD
- **Backtesting** — Vectorized engine with realistic simulation
- **Strategy optimizer** — Grid/random search for parameter tuning

**Location:** `src/monitoring/dashboard.py`, `src/monitoring/alerts.py`, `src/monitoring/analytics.py`

### 9. Portfolio Management ✅

- Position tracking across multiple symbols
- P&L attribution (realized/unrealized)
- Equity curve history
- Risk metrics (concentration, correlation)
- Dynamic rebalancing

**Location:** `src/risk/portfolio.py`

### 10. Strategy Optimization ✅

- Grid search optimization
- Random search optimization
- Multi-objective (Sharpe, returns, drawdown)
- Result export to JSON

**Location:** `scripts/optimizer.py`

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| Async Runtime | asyncio + aiohttp |
| ML Framework | PyTorch, scikit-learn, LightGBM |
| RL Framework | Stable-Baselines3 |
| Data Storage | TimescaleDB (PostgreSQL), Redis |
| Dashboard | Streamlit |
| Exchange Connectivity | Custom BloFin + Binance adapters |
| Backtesting | Custom vectorized engine |
| Config Management | Pydantic Settings + YAML |
| Secrets Management | python-dotenv |
| Containerization | Docker + docker-compose |
| Testing | pytest, pytest-asyncio |

---

## Project Phases

### Phase 1 — Foundation ✅ COMPLETE
- [x] Project scaffolding, config system, logging setup
- [x] BloFin exchange adapter (REST + WebSocket)
- [x] Market data pipeline (live OHLCV, order book, trades)
- [x] TimescaleDB schema and data storage
- [x] Paper trading mode (simulated execution, real data)

### Phase 2 — Signal Engine ✅ COMPLETE
- [x] Technical indicator library (custom pandas implementation)
- [x] Feature engineering pipeline (50+ features)
- [x] LightGBM directional classifier
- [x] Ensemble signal scorer (technical + structure + ML + volume)
- [x] Market structure analysis (BOS/CHoCH, FVGs, liquidity sweeps)

### Phase 3 — Risk & Execution ✅ COMPLETE
- [x] Risk manager module with all guards
- [x] Execution engine with order state machine
- [x] TP/SL automation
- [x] Portfolio state tracker
- [x] Backtesting harness with historical BloFin data

### Phase 4 — Intelligence Layer 🚧 IN PROGRESS
- [x] Market regime classifier (trending/ranging/volatile/breakout)
- [x] Multi-timeframe confluence engine
- [x] Reinforcement learning position sizing agent
- [x] Sentiment feed integration
- [x] On-chain data integration

### Phase 5 — Live Deployment 🚧 IN PROGRESS
- [x] Paper trading mode on real exchange (BloFin testnet / minimal capital)
- [x] Performance monitoring dashboard (7-tab Streamlit: Terminal, Charts, Paper Trades, Inventory, Intelligence, Ledger, System)
- [x] Telegram alerting bot
- [x] Production signal thresholds restored (min_confidence=55, direction_threshold=0.05, confluence_confidence≥55)
- [x] Stop-loss widened to 2.5× ATR (was 1.5×) to survive crypto volatility noise
- [x] atr_sl_multiplier + rr_ratio configurable from settings.yaml + wired into agent via config
- [x] ADX trend filter (require ADX > 25 before entry — blocks ranging market trades)
- [x] 4h candle subscription + 3-TF confluence requirement (15m entry + 1h + 4h trend bias must all agree)
- [x] Start/Stop Agent button in dashboard UI (sidebar: spawns python -m src.agent, tracks PID, live status badge)
- [x] Paper position recovery on restart — agent reloads open positions from DB on startup, corrects balance using last paper_trades.balance_after timestamp logic
- [ ] Stress testing and edge case hardening
- [ ] Gradual capital scaling protocol

---

## File Structure

```
cryptoagent/
├── project.md                  # This file
├── .env                        # API keys and secrets (never committed)
├── config/
│   ├── settings.yaml           # Strategy and risk parameters
│   └── exchanges.yaml          # Exchange endpoint config
├── src/
│   ├── agent.py                # Main agent orchestrator
│   ├── core/
│   │   ├── config.py           # Pydantic settings loader
│   │   └── logging.py          # Structured logging setup
│   ├── data/
│   │   ├── feeds/              # WebSocket + REST data collectors
│   │   │   ├── market_feed.py      # Live market data manager
│   │   │   ├── historical.py       # Historical data fetcher
│   │   │   ├── sentiment.py        # Sentiment data feed
│   │   │   └── onchain.py          # On-chain data feed
│   │   ├── storage/            # DB write/read models
│   │   │   └── schema.py           # TimescaleDB schema
│   │   └── preprocessing/      # Feature engineering
│   │       └── features.py          # ML feature generation
│   ├── signals/
│   │   ├── ensemble.py         # Signal aggregator
│   │   ├── technical/         # TA indicators
│   │   │   ├── indicators.py      # 20+ technical indicators
│   │   │   └── structure.py       # Market structure analysis
│   │   └── ml_models/         # Trained model wrappers
│   │       └── classifier.py      # LightGBM classifier
│   ├── decision/
│   │   ├── regime.py          # Market regime detector
│   │   ├── confluence.py      # Multi-timeframe logic
│   │   └── core.py           # Entry/exit decision logic
│   ├── execution/
│   │   └── paper.py          # Paper trading engine
│   ├── risk/
│   │   └── rl_sizer.py       # RL position sizing
│   ├── exchanges/
│   │   ├── base.py           # Abstract exchange interface
│   │   ├── blofin.py         # BloFin adapter
│   │   └── binance.py        # Binance adapter
│   └── monitoring/
│       ├── dashboard.py      # Streamlit dashboard
│       ├── alerts.py         # Telegram alerting
│       └── analytics.py     # Performance metrics
├── models/                    # Saved ML model weights
├── backtests/                 # Backtest results and reports
├── tests/                     # Unit and integration tests
│   ├── unit/
│   │   ├── test_regime.py
│   │   └── test_indicators.py
│   └── integration/
│       └── test_agent.py
├── scripts/
│   ├── train_models.py       # Model training pipeline
│   ├── download_history.py   # Historical data downloader
│   └── backtest.py          # Vectorized backtesting harness
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
cryptoagent/
├── project.md                  # This file
├── .env                        # API keys and secrets (never committed)
├── config/
│   ├── settings.yaml           # Strategy and risk parameters
│   └── exchanges.yaml          # Exchange endpoint config
├── src/
│   ├── agent.py                # Main agent orchestrator
│   ├── core/
│   │   ├── config.py           # Pydantic settings loader
│   │   └── logging.py          # Structured logging setup
│   ├── data/
│   │   ├── feeds/              # WebSocket + REST data collectors
│   │   │   ├── market_feed.py      # Live market data manager
│   │   │   └── historical.py       # Historical data fetcher
│   │   ├── storage/            # DB write/read models
│   │   │   └── schema.py           # TimescaleDB schema
│   │   └── preprocessing/      # Feature engineering
│   │       └── features.py          # ML feature generation
│   ├── signals/
│   │   ├── ensemble.py         # Signal aggregator
│   │   ├── technical/         # TA indicators
│   │   │   ├── indicators.py      # 20+ technical indicators
│   │   │   └── structure.py       # Market structure analysis
│   │   └── ml_models/         # Trained model wrappers
│   │       └── classifier.py      # LightGBM classifier
│   ├── decision/
│   │   ├── regime.py          # Market regime detector
│   │   ├── confluence.py      # Multi-timeframe logic
│   │   └── core.py           # Entry/exit decision logic
│   ├── execution/
│   │   └── paper.py          # Paper trading engine
│   ├── risk/
│   │   ├── rl_sizer.py       # RL position sizing
│   │   └── portfolio.py       # Portfolio manager
│   ├── exchanges/
│   │   ├── base.py           # Abstract exchange interface
│   │   ├── blofin.py        # BloFin adapter
│   │   └── binance.py       # Binance adapter
│   └── monitoring/
│       ├── dashboard.py      # Streamlit dashboard (7 tabs)
│       ├── alerts.py         # Telegram alerting
│       └── analytics.py     # Performance metrics
├── models/                    # Saved ML model weights
├── backtests/                 # Backtest results and reports
├── tests/                     # Unit and integration tests
│   ├── unit/
│   │   ├── test_regime.py
│   │   └── test_indicators.py
│   └── integration/
│       └── test_agent.py
├── scripts/
│   ├── train_models.py       # Model training pipeline
│   ├── download_history.py   # Historical data downloader
│   ├── backtest.py          # Vectorized backtesting harness
│   └── optimizer.py          # Strategy parameter optimizer
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Configuration

### Trading Settings (`config/settings.yaml`)
```yaml
agent:
  mode: paper                         # paper | live
  primary_exchange: blofin
  heartbeat_interval: 5

trading:
  symbols:
    - BTC-USDC
    - ETH-USDC
    - SOL-USDC
  default_leverage: 5
  timeframes: [1m, 5m, 15m, 1h, 4h, 1d]
  primary_timeframe: 15m

signals:
  min_confidence_threshold: 65
  confluence_required: 3
  lookback_candles: 500

risk:
  max_portfolio_risk_pct: 2.0
  max_single_trade_risk_pct: 0.5
  max_daily_drawdown_pct: 5.0
  kelly_fraction: 0.25
  max_open_positions: 5
```

### Exchange Configuration (`config/exchanges.yaml`)
- BloFin REST/WSS endpoints
- Binance planned (disabled)

---

## Running the Agent

### Start Paper Trading
```bash
python -m src
# or
python -m src.agent
```

### Start Live Trading
Set `.env` with `BLOFIN_API_KEY`, `BLOFIN_API_SECRET`, `BLOFIN_PASSPHRASE`, then:
```bash
export CRYPTOAGENT_MODE=live
python -m src
```

### Run Dashboard
```bash
streamlit run src/monitoring/dashboard.py --server.port 8501
```

### Run Backtest
```bash
python scripts/backtest.py --symbol BTC-USDC --timeframe 15m --candles 500
```

### Run Tests
```bash
python -m pytest tests/ -v
```

# Run specific test file
python -m pytest tests/unit/test_indicators.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Key Design Principles

1. **Safety First** — Paper trade and backtest every strategy before live capital
2. **No Black Boxes** — Every decision must be explainable and logged
3. **Fail Safe** — Any unhandled exception closes open positions and halts trading
4. **Modularity** — Strategies and models are plug-and-play without touching core logic
5. **Reproducibility** — All experiments are seeded and versioned
6. **Capital Preservation** — The risk manager is the most important component

---

## Risk Disclaimer

This agent will manage real capital in live markets. Crypto and futures markets are highly volatile. All strategies must be:
- Backtested on at minimum 2 years of data
- Forward-tested on paper for minimum 2 weeks
- Deployed with minimal capital initially
- Monitored continuously during live trading
- Accompanied by hard stop-loss limits at the account level

---

## Performance Metrics

The system tracks:
- **Win Rate** — Percentage of profitable trades
- **Sharpe Ratio** — Risk-adjusted returns
- **Sortino Ratio** — Downside risk-adjusted returns
- **Max Drawdown** — Peak-to-trough decline
- **Expectancy** — Average trade P&L
- **Profit Factor** — Gross profits / gross losses

---

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `aiohttp` — Async HTTP/WebSocket
- `pandas` — Data manipulation
- `lightgbm` — ML classifier
- `sqlalchemy` + `asyncpg` — Database
- `redis` — Cache
- `streamlit` — Dashboard
- `pydantic` — Config validation
- `structlog` — Structured logging
