# CryptoAgent — Autonomous Crypto Trading Agent

## Vision

Build a high-accuracy, AI-driven autonomous trading agent capable of live trading in **crypto spot**, **crypto perpetual futures**, and **traditional futures** markets. The agent connects natively to **BloFin** and is designed around maximizing prediction accuracy while managing risk with surgical precision.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                            CryptoAgent                              │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │   Data Layer    │──▶│  Signal Engine   │──▶│ Decision Core   │  │
│  │  market_feed    │   │  technical +     │   │ confluence +    │  │
│  │  sentiment_feed │   │  LightGBM +      │   │ regime +        │  │
│  │  TimescaleDB    │   │  LSTM (blended)  │   │ risk filters    │  │
│  │  Redis cache    │   └──────────────────┘   └─────────────────┘  │
│  └─────────────────┘                                  │            │
│                                            ┌──────────────────────┐ │
│                                            │   Risk Manager       │ │
│                                            │  Kelly sizing        │ │
│                                            │  ADX trend filter    │ │
│                                            │  correlation guard   │ │
│                                            │  drawdown breaker    │ │
│                                            └──────────────────────┘ │
│                                                       │            │
│                                            ┌──────────────────────┐ │
│                                            │  Execution Engine    │ │
│                                            │  Paper (default)     │ │
│                                            │  Live (dry_run=True) │ │
│                                            └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current Status — Mar 03 2026

| Component | Status |
|---|---|
| Agent (paper mode) | ✅ Running — PID 44640 — 10 symbols × 6 timeframes = 60 live feeds |
| Signal engine | ✅ All 10 symbols × 6 TFs (1m/3m/5m/15m/1h/4h) |
| ML models | ✅ 40 LGBM (75 features, n=1000) + 10 LSTM (all 10 symbols, 15m) |
| Per-symbol thresholds | ✅ Grid-search optimized (BTC=25, ETH=40, SOL=30, XRP=55, DOGE=25 ...) |
| Dashboard | ✅ http://localhost:8501 — TradingView charts, live P&L, 7 tabs (Chat added) |
| Watchdog | ✅ Auto-restarts agent + dashboard if either dies |
| Portfolio | $10,000 paper — ~25 closed trades (Clean Sharpe: 8.73 since guards) |

---

## Trading Universe — ALL 10 BloFin USDC Pairs

| Symbol | Min Conf | ATR SL | Notes |
|---|---|---|---|
| BTC-USDC | 25% | 2.0× | Core pair |
| ETH-USDC | 40% | 2.0× | Higher threshold — noisier |
| SOL-USDC | 30% | 2.0× | — |
| XRP-USDC | 55% | 2.0× | Very noisy — high filter |
| DOGE-USDC | 25% | 2.0× | — |
| BNB-USDC | 25% | 2.0× | — |
| SUI-USDC | 35% | 2.5× | — |
| ENA-USDC | 25% | 1.5× | — |
| LINK-USDC | 25% | 3.0× | — |
| ADA-USDC | 30% | 2.0× | — |
| PIPPIN-USDT | 25% | 3.0× | High volatility meme coin |

---

## Implemented Modules

### 1. Data Layer ✅
- **Six-timeframe WebSocket feeds** — 1m/3m/5m/15m/1h/4h per symbol (60 channels total)
- **REST polling** — Historical candles (paginated), tickers, funding rates, open interest
- **Storage** — TimescaleDB (PostgreSQL port 5433) for time-series; Redis (port 6379) for hot cache
- **Sentiment & On-Chain** — Fear & Greed Index, BTC dominance, Binance funding rate + OI (every 5 min)
- **Location:** `src/data/feeds/`, `src/data/storage/`, `src/data/preprocessing/`

### 2. Signal Engine ✅
**Ensemble signal scorer: 6 sources × weighted → 0–100 confidence**

| Source | Weight | Details |
|---|---|---|
| Technical | 30% | EMA cross, Supertrend, ADX, RSI, MACD, StochRSI, BB, ATR — trend-aware |
| Market Structure | 25% | BOS/CHoCH, FVG, liquidity sweeps, order blocks |
| ML (blended) | 25% | LightGBM + LSTM agree → avg conf; disagree → 0 |
| Volume | 10% | VWAP position, VWAP anchor cross, CVD, OBV |
| Sentiment | 5% | Fear & Greed index, news NLP (VADER + crypto keywords) |
| On-chain | 5% | BTC dominance, funding diff, OI change |

**ML Models (retrained Mar 02-03, balanced weights + drop-flat, 18k bars, 10-fold walk-forward):**
- **LightGBM** — 40 models (10 symbols × 1m/3m/5m/15m), **75 features** (+time-of-day: hour_sin/cos/dow_sin/cos/is_weekend)
  - n_estimators=1000, num_leaves=127, reg_alpha/lambda=0.1, early_stopping=50
  - Grid-search optimized per symbol; OOS Sharpe: BTC=6.50, ETH=9.16, SOL=9.22
  - SHAP: time-of-day features rank #2-8 by gain — strong intraday seasonality
- **LSTM** — Bidirectional 2-layer, 60-bar lookback, **GPU accelerated (RTX 5070)**
  - **10 models**: all 10 symbols at 15m (previously only 5)
- **ML gating:** LightGBM on 1m/3m/5m/15m; LSTM on 15m only; HTFs (1h/4h) = pure TA

**Location:** `src/signals/technical/`, `src/signals/ml_models/`, `src/signals/ensemble.py`

### 3. Decision Core ✅
- **Six-timeframe confluence** — LTF (1m/3m/5m/15m) entry + 1h intermediate + 4h bias
- **Adaptive:** 4h flat → 2-TF sufficient; 4h active → 3-TF required
- **Regime detection** — trending_up / trending_down / ranging / volatile / breakout
- **Correlation guard** — blocks new positions when ≥2 same-direction positions already open
- **Entry gate:** confidence ≥ per-symbol threshold AND MTF confluence AND all risk filters
- **ATR-based SL** (2.0× default), TP at 2:1 RR; trailing stop disabled by default

**Location:** `src/decision/core.py`, `src/decision/confluence.py`

### 4. Risk Manager ✅
- **Kelly position sizing** — 25% fraction, capped at 0.5% of portfolio per trade
- **ADX filter** — Blocks entries when ADX < 20
- **Correlation guard** — Max 2 correlated (same-direction) positions simultaneously
- **Dead signal watchdog** — Telegram alert if no closed candles for >10 min
- **Daily drawdown circuit breaker** — Halts at 5% daily DD

### 5. Execution Engine ✅
- **Paper engine** (`src/execution/paper.py`) — Simulated fills, SL/TP tick monitoring, position recovery on restart
- **Live engine** (`src/execution/live.py`) — `dry_run=True` by default

### 6. Exchange Adapter ✅ — BloFin
- REST + WebSocket (OHLCV, order book, trades, orders, positions)
- `ThreadedResolver` for all aiohttp sessions (Windows DNS fix)
- `heartbeat=20` on WS (protocol-level PING)
- Private WS only in live mode
- 429 rate limit: 300s backoff on startup, bail-out preload after 3 consecutive 429s

### 7. Monitoring ✅
- **Dashboard** (`src/monitoring/dashboard.py`) — Streamlit, 6 tabs:
  - **Terminal** — Live portfolio equity, open positions with real P&L, signals table
  - **Advanced Charts** — TradingView-quality candlesticks (streamlit-lightweight-charts), EMA 9/21/50, Bollinger Bands, volume overlay, RSI pane
  - **Paper Trades** — Full trade history with P&L cards
  - **Brain** — 6-TF signal badges per symbol, AI trade analyst
  - **Intelligence** — Fear & Greed, BTC dominance, funding rates, news
  - **System** — Agent control, logs, configuration
- **Stale price fallback** — Dashboard uses Redis prices up to 30 min old; shows banner when WS is reconnecting
- **Watchdog** (`scripts/watchdog.sh`) — Checks every 30s, auto-restarts agent + dashboard

---

## Six-Timeframe Stack

| TF | Role | ML | DB bars |
|---|---|---|---|
| 1m | Fast scalp trigger | LightGBM | ~240/day |
| 3m | Fast entry confirmation | LightGBM | ~480/day |
| 5m | Entry signal | LightGBM | ~288/day |
| 15m | Primary entry | LightGBM + LSTM | ~96/day |
| 1h | Intermediate trend | Pure TA | ~24/day |
| 4h | Bias / regime | Pure TA | ~6/day |

---

## Grid Search Results (Mar 01 — all symbols, 3000 bars, 25% OOS)

| Symbol | Best Conf | ATR | OOS Sharpe | OOS WR |
|---|---|---|---|---|
| BTC-USDC | 25% | 2.0× | 6.50 | 55% |
| ETH-USDC | 40% | 2.0× | 9.16 | 75% |
| SOL-USDC | 30% | 2.0× | 9.22 | 63% |
| DOGE-USDC | 25% | 2.0× | 6.84 | 58% |

Full results: `backtests/optimize_*_*.csv`

---

## Known Bugs Fixed

| # | Date | Bug | Fix |
|---|---|---|---|
| 1 | Early | `aiodns` fails on Windows DNS | `ThreadedResolver` in all `ClientSession` instances |
| 2 | Early | Dashboard spawned duplicate agents | Idempotency guard in `_start_agent()` |
| 3 | Early | Private WS reconnect in paper mode | Skip private WS when `is_paper=True` |
| 4 | Early | BloFin rejects `{"op":"ping"}` | `ws_connect(heartbeat=20)` |
| 5 | Feb 28 | Forming candle buffer flood (CRITICAL) | Skip forming candles in `_on_candle()` — `closed=True` only |
| 6 | Feb 28 | LGBM/LSTM on wrong timeframe (OOD) | ML gated to LTF timeframes only |
| 7 | Mar 01 | Volume score always 0 (CRITICAL) | Added `vwap_dist`, `cvd_norm`, etc. to `compute_all()` |
| 8 | Mar 01 | `runtime.json` threshold override | Dashboard slider was overriding settings.yaml to 9 |
| 9 | Mar 01 | Session VWAP not resetting | VWAP now groups by UTC date |
| 10 | Mar 01 | Trailing stop destroying TP hits | `trailing_stop_activation_atr: 0` (disabled by default) |
| 11 | Mar 01 | Correlation overexposure | `max_correlated_positions=2` guard in risk filter |
| 12 | Mar 02 | DB malformed 15m candles (CRITICAL) | WS stored forming candle receipt timestamps as 15m candles (3000 bad rows). Deleted + `fetch_price_history()` now filters on TF-aligned timestamps only |
| 13 | Mar 02 | DB stale no-dash symbols | `BTCUSDC`/`ETHUSDC`/`SOLUSDC` (17,928 rows) deleted |
| 14 | Mar 02 | Open positions showing P&L=$0 | `get_live_prices()` fell through to 429-limited REST returning `price=0.0`. Fixed: stale Redis (< 30 min) used before REST calls |
| 15 | Mar 02 | BloFin IP rate limit death loop | Preload bails on 3rd consecutive 429; startup retry waits 300s on 429 (was 5s→10s→crash); dashboard stops hammering REST |
| 16 | Mar 02 | Agent crashes after 10 startup retries | Startup retry now loops **indefinitely** with 300s wait on 429 — no more death loop |
| 17 | Mar 03 | SL cooldown asyncio race condition | Cooldown was set AFTER `await db.log_paper_trade()` inside for loop → concurrent tasks bypassed it. Fixed: set all cooldowns synchronously BEFORE any awaits |
| 18 | Mar 03 | Zombie agent accumulation | `main()` overwrote `agent.pid` without killing old process → multiple agents writing to log. Fixed: startup reads old pid, `taskkill /F /PID <old>` before writing new pid |
| 19 | Mar 04 | 429 Rate Limit Hammering | Dashboard made 10+ REST calls/poll. Fixed: Refactored to single batch ticker call. |
| 20 | Mar 04 | GPT-5.2 Chat 400 Errors | `temperature` and `max_tokens` unsupported. Fixed: Removed temp, use `max_completion_tokens`. |
| 21 | Mar 04 | Signal Breakdown Missing | Dashboard cards only showed total. Fixed: Added tech/struct/vol/sent/onchain breakdown logic. |

---

## Phase Roadmap

### Phases 1–8 ✅ COMPLETE

- **Phase 1** — BloFin exchange adapter, TimescaleDB, Redis, paper engine
- **Phase 2** — Full TA library, market structure, LightGBM, ensemble, MTF confluence
- **Phase 3** — Backtester, Streamlit dashboard, Telegram alerts
- **Phase 3.5** — Market Intelligence tab (Fear & Greed, funding, news)
- **Phase 3.6** — SentimentFeed into ensemble, adaptive 4h confluence, SL recovery
- **Phase 4** — LSTM model, OI/funding pre-filter, live engine scaffold, thresholds tuned
- **Phase 5** — VWAP anchor, session VWAP, 5-symbol universe, 70-feature LGBM
- **Phase 6** — Watchdog, correlation guard, SHAP analysis, OOS validation, grid search, weekly retrain
- **Phase 7** — Per-symbol optimization (grid search all 5 symbols, symbol_overrides in settings.yaml)
- **Phase 8** — 6-TF stack (1m/3m/5m added), 10-symbol universe (all BloFin USDC pairs), 40 LGBM models, GPU training (RTX 5070)

### Phase 9 — Robustness & Production Hardening (CURRENT)

- [x] DB cleanup — deleted 20,928 malformed/stale candle rows
- [x] Dashboard charts — TradingView-quality (streamlit-lightweight-charts), aligned timestamps only
- [x] Rate limit resilience — 300s backoff on 429, preload bail-out, stale Redis fallback
- [x] Watchdog — auto-restarts agent + dashboard
- [x] ML accuracy — 75 features (+ time-of-day), n_estimators=1000, 10 LSTM models
- [x] SL cooldown race condition fix — cooldowns set synchronously before any awaits
- [x] Zombie agent guard — startup kills previous PID via taskkill before writing new pid
- [ ] Cloud hosting — VPS deployment (see Hosting section)
- [ ] Paper validation — 50+ trades, Sharpe > 2, max DD < 10%

### Phase 10 — Live Deployment

- [ ] Paper validation complete (50+ trades)
- [ ] Switch `AGENT_MODE=live`, `dry_run=False`
- [ ] Start with 1% of capital, scale weekly
- [ ] Multi-exchange fallback (Bybit, OKX)

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 (`.venv_312`) |
| Async runtime | asyncio + aiohttp + ThreadedResolver (Windows DNS fix) |
| ML | PyTorch 2.10+cu128 (LSTM, RTX 5070 GPU), LightGBM 4.6, scikit-learn |
| NLP | vaderSentiment + custom crypto keyword dictionary |
| Storage | TimescaleDB port 5433, Redis port 6379 (Docker) |
| Dashboard | Streamlit + streamlit-lightweight-charts (TradingView) |
| Charts | TradingView lightweight-charts — candlestick + EMA + BB + Volume + RSI |

---

## Running Locally

```bash
# 1. Start infrastructure
docker-compose up -d timescaledb redis

# 2. Activate venv
source .venv_312/Scripts/activate

# 3. Start everything (agent + dashboard + watchdog)
PYTHONUTF8=1 PYTHONUNBUFFERED=1 python -u -m src.agent >> logs/agent.log 2>&1 &
echo $! > agent.pid
streamlit run src/monitoring/dashboard.py --server.port 8501 --server.headless true >> dashboard_startup.log 2>&1 &
echo $! > dashboard.pid
bash scripts/watchdog.sh >> logs/watchdog.log 2>&1 &
echo $! > watchdog.pid

# 4. Remote access via ngrok
ngrok http 8501

# 5. Kill everything
taskkill //F //PID $(cat agent.pid)
taskkill //F //PID $(cat dashboard.pid)
taskkill //F //PID $(cat watchdog.pid)
```

## Training & Backtesting

```bash
# Retrain LGBM (single symbol)
python scripts/train_models.py --symbol BTC-USDC --candles 18000 --splits 10 \
  --threshold 0.003 --drop-flat --force-retrain

# Retrain LSTM (GPU)
python scripts/train_models.py --symbol BTC-USDC --candles 18000 \
  --train-lstm --force-retrain

# Grid search optimal params
python scripts/optimize_params.py --symbol BTC-USDC --candles 3000 --timeframe 15m

# Backtest
python scripts/backtest.py --symbol BTC-USDC --candles 2000 --min-confidence 25 --oos-split 0.25

# Weekly retrain (all 10 symbols)
bash scripts/retrain_weekly.sh
```

---

## Key Risk Parameters (`config/settings.yaml`)

```yaml
signals:
  min_confidence_threshold: 25   # global default (per-symbol overrides in symbol_overrides:)
  confluence_required: 3         # adaptive: 2 when 4h is flat

risk:
  atr_sl_multiplier: 2.0         # SL = entry ± 2.0×ATR (grid-search optimal)
  rr_ratio: 2.0                  # TP = 2× SL distance
  max_open_positions: 10
  max_correlated_positions: 2    # max same-direction positions
  max_daily_drawdown_pct: 5.0    # circuit breaker
  kelly_fraction: 0.25
  trailing_stop_activation_atr: 0  # disabled (was destroying TP hits)

ensemble_weights:
  technical: 0.30
  structure: 0.25
  ml: 0.25
  volume: 0.10
  sentiment: 0.05
  onchain: 0.05
```

---

## Cloud Hosting Plan

See **Hosting** section below for full deployment guide.

---

## Backtest Results (Grid Search OOS — Mar 01)

| Symbol | Bars | Conf | ATR | OOS Trades | OOS WR | OOS Sharpe |
|---|---|---|---|---|---|---|
| BTC-USDC 15m | 3000 | 25% | 2.0× | 20 | 55% | 6.50 |
| ETH-USDC 15m | 3000 | 40% | 2.0× | 8 | 75% | 9.16 |
| SOL-USDC 15m | 3000 | 30% | 2.0× | 16 | 63% | 9.22 |
| DOGE-USDC 15m | 3000 | 25% | 2.0× | 19 | 58% | 6.84 |

---

## Hosting — Production Deployment Plan

### Overview

The system has 4 components that need to run 24/7:

| Component | What it does | Resource needs |
|---|---|---|
| **Agent** (Python) | Trading logic, WS feeds, ML inference | 2 CPU cores, 4 GB RAM |
| **Dashboard** (Streamlit) | Web UI | 1 CPU core, 1 GB RAM |
| **TimescaleDB** (PostgreSQL) | Candle/trade storage | 2 GB RAM, 20 GB SSD |
| **Redis** | Price hot-cache | 256 MB RAM |

### Recommended: Single VPS (Best cost/simplicity)

**Provider: Hetzner Cloud CX31** (~€10/month, ~$11/month)
- 2 vCPU (AMD), 8 GB RAM, 80 GB NVMe SSD, unlimited traffic
- Location: Nuremberg / Helsinki (low latency to BloFin servers in Asia — use Singapore if available)

**Alternative providers:**
| Provider | Plan | Cost | Notes |
|---|---|---|---|
| Hetzner CX31 | 2 vCPU / 8 GB | ~$11/mo | Best value in EU |
| DigitalOcean Droplet | 2 vCPU / 4 GB | ~$24/mo | Easy setup, good docs |
| Vultr | 2 vCPU / 4 GB | ~$24/mo | Singapore region available (lower BloFin latency) |
| Linode/Akamai | 2 vCPU / 4 GB | ~$24/mo | Reliable |
| AWS EC2 t3.medium | 2 vCPU / 4 GB | ~$30/mo | Overkill but familiar |

> **Note:** LSTM training needs GPU. Keep training on your Windows machine with the RTX 5070, then `scp` the `.joblib`/`.pt` model files to the VPS. The VPS only does **inference** (fast, CPU-only).

### Step-by-Step Migration

#### 1. Provision VPS
```bash
# After creating VPS (Ubuntu 24.04 LTS recommended):
ssh root@<vps-ip>
adduser cryptoagent
usermod -aG sudo cryptoagent
```

#### 2. Install dependencies
```bash
# Docker + Docker Compose
curl -fsSL https://get.docker.com | sh
apt install -y python3.12 python3.12-venv git nginx certbot

# Clone repo
git clone https://github.com/yourrepo/cryptoagent.git
cd cryptoagent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Start infrastructure
```bash
docker-compose up -d timescaledb redis
```

#### 4. Configure secrets
```bash
cp .env.example .env
nano .env
# Fill in: BLOFIN_API_KEY, BLOFIN_API_SECRET, BLOFIN_API_PASSPHRASE
# Set: BLOFIN_TESTNET=false, AGENT_MODE=paper (then live when ready)
```

#### 5. Run with systemd (auto-restart on crash/reboot)

Create `/etc/systemd/system/cryptoagent.service`:
```ini
[Unit]
Description=CryptoAgent Trading Bot
After=network.target docker.service
Requires=docker.service

[Service]
User=cryptoagent
WorkingDirectory=/home/cryptoagent/cryptoagent
Environment=PYTHONUTF8=1
EnvironmentFile=/home/cryptoagent/cryptoagent/.env
ExecStart=/home/cryptoagent/cryptoagent/.venv/bin/python -u -m src.agent
Restart=always
RestartSec=30
StandardOutput=append:/home/cryptoagent/cryptoagent/logs/agent.log
StandardError=append:/home/cryptoagent/cryptoagent/logs/agent.log

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/cryptodashboard.service`:
```ini
[Unit]
Description=CryptoAgent Dashboard
After=network.target

[Service]
User=cryptoagent
WorkingDirectory=/home/cryptoagent/cryptoagent
EnvironmentFile=/home/cryptoagent/cryptoagent/.env
ExecStart=/home/cryptoagent/cryptoagent/.venv/bin/streamlit run src/monitoring/dashboard.py --server.port 8501 --server.headless true --server.address 127.0.0.1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable cryptoagent cryptodashboard
systemctl start cryptoagent cryptodashboard
```

#### 6. HTTPS dashboard with Nginx + Let's Encrypt

```nginx
# /etc/nginx/sites-available/cryptoagent
server {
    listen 443 ssl;
    server_name dashboard.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/dashboard.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/dashboard.yourdomain.com/privkey.pem;

    # Password protect the dashboard
    auth_basic "CryptoAgent";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

```bash
certbot --nginx -d dashboard.yourdomain.com
htpasswd -c /etc/nginx/.htpasswd admin
systemctl restart nginx
```

#### 7. Deploy model updates (from Windows → VPS)
```bash
# After retraining on Windows RTX 5070, push models to VPS
scp models/lgbm_*.joblib cryptoagent@<vps-ip>:~/cryptoagent/models/
scp models/lstm_*.pt cryptoagent@<vps-ip>:~/cryptoagent/models/
ssh cryptoagent@<vps-ip> "sudo systemctl restart cryptoagent"
```

### Cost Summary

| Item | Monthly Cost |
|---|---|
| Hetzner CX31 VPS | ~$11 |
| Domain (optional) | ~$1 |
| **Total** | **~$12/month** |

### Security Checklist

- [ ] Firewall: only allow ports 22 (SSH), 80, 443 inbound
- [ ] SSH key auth only (disable password login)
- [ ] Dashboard behind Nginx + basic auth (or VPN-only access)
- [ ] `.env` never committed to git — use `scp` or secrets manager
- [ ] BloFin API key: enable only "Trade" permission, whitelist VPS IP
- [ ] Regular `apt upgrade` and Docker image updates
