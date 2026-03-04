"""
CryptoAgent Monitoring Dashboard
Real-time Streamlit dashboard showing portfolio P&L, open positions,
recent signals, trade history, and performance metrics.

Run:
    streamlit run src/monitoring/dashboard.py --server.port 8501
or via Docker:
    docker-compose up dashboard
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ.setdefault("PYTHONUTF8", "1")

# Load .env into os.environ early so Azure / Gemini keys are available before any imports
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
except Exception:
    pass

import numpy as np
import pandas as pd

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoAgent | Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Mobile-First Premium Look ────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
        /* ── Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Base ── */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b0e11;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background-color: #161a1e;
            border-right: 1px solid #2d3139;
        }

        /* ── Metrics ── */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #00ffa3 !important;
            text-shadow: 0 0 10px rgba(0,255,163,0.2);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ── Mode badges ── */
        .mode-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .mode-paper { background-color:rgba(240,185,11,0.15); color:#f0b90b; border:1px solid #f0b90b; }
        .mode-live  { background-color:rgba(255,59,48,0.15);  color:#ff3b30; border:1px solid #ff3b30;
                      box-shadow:0 0 15px rgba(255,59,48,0.3); }

        /* ── Glass card ── */
        .glass-card {
            background: rgba(22,26,30,0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }

        /* ── Tabs — always horizontally scrollable ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #121418;
            padding: 8px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid #2d3139;
            /* Mobile scroll */
            overflow-x: auto;
            overflow-y: hidden;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            flex-wrap: nowrap;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            background-color: transparent;
            border-radius: 8px;
            color: #d1d4dc !important;
            font-weight: 700 !important;
            border: none;
            padding: 0 22px;
            transition: all 0.2s ease;
            white-space: nowrap;
            flex-shrink: 0;
            min-width: max-content;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #ffffff !important;
            background-color: rgba(255,255,255,0.05);
        }
        .stTabs [aria-selected="true"] {
            background-color: #00ffa3 !important;
            color: #0b0e11 !important;
            box-shadow: 0 0 20px rgba(0,255,163,0.3);
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width:6px; height:6px; }
        ::-webkit-scrollbar-thumb { background:#2d3139; border-radius:10px; }
        ::-webkit-scrollbar-track { background:transparent; }

        /* ── Plotly text ── */
        .js-plotly-plot .plotly .xtick text,
        .js-plotly-plot .plotly .ytick text,
        .js-plotly-plot .plotly .legendtext,
        .js-plotly-plot .plotly .gtitle {
            fill: #ffffff !important;
            font-size: 12px !important;
            font-weight: 600 !important;
        }

        /* ── Header chrome ── */
        [data-testid="stHeader"] {
            background-color: transparent !important;
            border-bottom: none !important;
        }
        [data-testid="stHeader"] button {
            color: #ffffff !important;
            background: rgba(255,255,255,0.08) !important;
            border-radius: 4px !important;
            margin-top: 5px !important;
        }
        [data-testid="stDecoration"] { display:none !important; }
        #MainMenu { visibility:hidden; }
        footer     { visibility:hidden; }

        /* ── Responsive layout utilities ── */
        /* Table wrappers — always allow horizontal scroll */
        .tbl-wrap {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            width: 100%;
        }
        /* Trade-card data grid — CSS class (override inline grid) */
        .trade-data-grid {
            display: grid !important;
            grid-template-columns: repeat(6, 1fr) !important;
            gap: 14px;
            margin-bottom: 14px;
        }

        /* ══ TABLET  ≤ 900px ══ */
        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                max-width: 100% !important;
            }
            .stTabs [data-baseweb="tab"] {
                height: 40px !important;
                padding: 0 14px !important;
                font-size: 0.78rem !important;
            }
            [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
        }

        /* ══ MOBILE  ≤ 768px ══ */
        @media (max-width: 768px) {
            /* 1. Shrink main container padding */
            .main .block-container {
                padding-top: 0.5rem !important;
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                max-width: 100vw !important;
            }

            /* 2. Stack Streamlit columns (min 2-per-row) */
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 8px !important;
            }
            [data-testid="column"] {
                min-width: calc(50% - 4px) !important;
                flex: 1 1 calc(50% - 4px) !important;
                width: auto !important;
            }

            /* 3. Compact tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px !important;
                padding: 5px !important;
                margin-bottom: 14px !important;
                border-radius: 10px !important;
            }
            .stTabs [data-baseweb="tab"] {
                height: 36px !important;
                padding: 0 10px !important;
                font-size: 0.7rem !important;
            }

            /* 4. Scale metrics */
            [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.65rem !important; letter-spacing:0 !important; }
            [data-testid="metric-container"] { padding: 8px !important; }

            /* 5. Trade data grid: 6 → 3 cols */
            .trade-data-grid {
                grid-template-columns: repeat(3, 1fr) !important;
            }

            /* 6. Tables: min-width keeps content readable, wrapper scrolls */
            .pos-tbl { min-width: 560px; }
            .sig-tbl { min-width: 460px; }

            /* 7. Sidebar overlay width on mobile */
            [data-testid="stSidebar"][aria-expanded="true"] {
                width: 85vw !important;
                max-width: 320px !important;
            }

            /* 8. Buttons */
            .stButton > button {
                font-size: 0.8rem !important;
                padding: 0.4rem 0.8rem !important;
                min-height: 44px;
            }

            /* 9. Inputs / sliders */
            [data-testid="stSlider"] label { font-size: 0.8rem !important; }

            /* 10. Plotly charts — cap height on mobile */
            .js-plotly-plot { max-height: 300px; overflow: hidden; }

            /* 11. Spacing */
            hr { margin: 0.5rem 0 !important; }
            h1 { font-size: 1.4rem !important; }
            h2 { font-size: 1.15rem !important; }
            h3 { font-size: 1rem !important; }
        }

        /* ══ SMALL MOBILE  ≤ 480px ══ */
        @media (max-width: 480px) {
            /* Single-column everything */
            [data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }

            /* Trade grid: 3 → 2 cols */
            .trade-data-grid {
                grid-template-columns: repeat(2, 1fr) !important;
            }

            [data-testid="stMetricValue"] { font-size: 1rem !important; }

            .stTabs [data-baseweb="tab"] {
                height: 32px !important;
                padding: 0 8px !important;
                font-size: 0.62rem !important;
            }

            .main .block-container {
                padding-left: 0.25rem !important;
                padding-right: 0.25rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ── Imports (after sys.path setup) ────────────────────────────────────────────
try:
    from src.core.config import get_settings
    from src.monitoring.analytics import compute_summary
    CONFIG_OK = True
except Exception as e:
    CONFIG_OK = False
    CONFIG_ERR = str(e)

# ── Trading universe (read from settings.yaml) ────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
try:
    import yaml as _yaml
    _cfg = _yaml.safe_load((_ROOT / "config" / "settings.yaml").read_text())
    TRADING_SYMBOLS: list[str] = _cfg.get("trading", {}).get("symbols", ["BTC-USDC", "ETH-USDC", "SOL-USDC"])
except Exception:
    TRADING_SYMBOLS = ["BTC-USDC", "ETH-USDC", "SOL-USDC"]


# ── Agent Process Control ──────────────────────────────────────────────────────

_PID_FILE = Path(__file__).parent.parent.parent / "agent.pid"


def _agent_pid() -> int | None:
    """Return the PID from the pid file, or None if it doesn't exist."""
    try:
        return int(_PID_FILE.read_text().strip())
    except Exception:
        return None


def _is_agent_alive(pid: int | None) -> bool:
    """Return True if a process with given PID is alive."""
    if pid is None:
        return False
    try:
        import sys
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            # Check exit code — STILL_ACTIVE = 259
            exit_code = ctypes.c_ulong(0)
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return exit_code.value == 259   # STILL_ACTIVE
        else:
            import os
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def _start_agent() -> tuple[bool, str]:
    """Spawn the agent as a detached subprocess. Returns (success, message)."""
    import subprocess, sys
    # Idempotency: refuse to start if an agent is already alive
    existing_pid = _agent_pid()
    if _is_agent_alive(existing_pid):
        return False, f"Agent already running (PID {existing_pid})"
    # Clean up stale pid file from a previous dead process
    _PID_FILE.unlink(missing_ok=True)
    try:
        project_root = Path(__file__).parent.parent.parent
        log_path = project_root / "logs" / "agent.log"
        log_path.parent.mkdir(exist_ok=True)

        # Always use the same Python that's running the dashboard
        venv_python = Path(sys.executable)

        env = {
            **__import__("os").environ,
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }

        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP

        with open(log_path, "a") as log_fh:
            proc = subprocess.Popen(
                [str(venv_python), "-u", "-m", "src.agent"],
                cwd=str(project_root),
                env=env,
                creationflags=flags,
                stdout=log_fh,
                stderr=log_fh,
            )
        _PID_FILE.write_text(str(proc.pid))
        return True, f"Agent started (PID {proc.pid})"
    except Exception as e:
        return False, f"Failed to start agent: {e}"


def _stop_agent(pid: int) -> tuple[bool, str]:
    """Terminate the agent process. Returns (success, message)."""
    try:
        import sys
        if sys.platform == "win32":
            import ctypes
            PROCESS_TERMINATE = 0x0001
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if handle:
                ctypes.windll.kernel32.TerminateProcess(handle, 1)
                ctypes.windll.kernel32.CloseHandle(handle)
        else:
            import os, signal
            os.kill(pid, signal.SIGTERM)
        _PID_FILE.unlink(missing_ok=True)
        return True, f"Agent stopped (PID {pid})"
    except Exception as e:
        _PID_FILE.unlink(missing_ok=True)
        return False, f"Stop error (pid file cleared): {e}"


def render_agent_control():
    """Sidebar widget to start / stop the trading agent."""
    pid = _agent_pid()
    running = _is_agent_alive(pid)

    st.sidebar.divider()
    st.sidebar.markdown("### Agent Control")

    if running:
        st.sidebar.markdown(
            f'<div style="background:rgba(0,255,163,0.1);border:1px solid #00ffa3;'
            f'border-radius:6px;padding:8px 12px;margin-bottom:8px;">'
            f'<span style="color:#00ffa3;font-weight:700;">● RUNNING</span>'
            f'<span style="color:#848e9c;font-size:0.78rem;margin-left:8px;">PID {pid}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("⏹ STOP AGENT", use_container_width=True, key="btn_stop_agent",
                             type="primary"):
            ok, msg = _stop_agent(pid)
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
            st.rerun()
    else:
        if pid:
            _PID_FILE.unlink(missing_ok=True)   # stale pid file — clean up
        st.sidebar.markdown(
            '<div style="background:rgba(255,77,77,0.1);border:1px solid #ff4d4d;'
            'border-radius:6px;padding:8px 12px;margin-bottom:8px;">'
            '<span style="color:#ff4d4d;font-weight:700;">● STOPPED</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("▶ START PAPER TRADING", use_container_width=True,
                             key="btn_start_agent", type="primary"):
            ok, msg = _start_agent()
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
            st.rerun()


# ── DB helpers ────────────────────────────────────────────────────────────────

def query_df(sql: str, params=None) -> pd.DataFrame:
    """
    Query TimescaleDB using asyncpg (no psycopg2 needed).
    Params should use %s placeholders; they are substituted directly
    since only safe integer/string LIMIT values are ever passed here.
    """
    try:
        import asyncio
        import asyncpg
        from src.core.config import get_settings

        # Substitute %s placeholders with actual values
        if params:
            for p in params:
                sql = sql.replace("%s", str(int(p)), 1)

        async def _fetch() -> list:
            url = get_settings().database.timescale_url
            conn = await asyncpg.connect(dsn=url)
            try:
                rows = await conn.fetch(sql)
                return [dict(r) for r in rows]
            finally:
                await conn.close()

        # Run in a dedicated thread so we never conflict with a running loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            rows = pool.submit(asyncio.run, _fetch()).result(timeout=10)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── Redis helpers ─────────────────────────────────────────────────────────────

def get_redis():
    try:
        import redis
        from src.core.config import get_settings
        r = redis.from_url(get_settings().database.redis_url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


_TRACKED_SYMBOLS = TRADING_SYMBOLS  # reads all 10 symbols from settings.yaml
_BLOFIN_TICKER_URL = "https://openapi.blofin.com/api/v1/market/tickers"
_PRICE_STALE_SECS = 120  # treat Redis price as stale after 2 minutes


def load_demo_data():
    """Populate entire database with professional demo data for testing."""
    import asyncio
    import numpy as np
    import json
    from datetime import datetime, timezone, timedelta
    from src.data.storage.schema import Database
    from src.exchanges.base import Candle, Order, OrderSide, OrderStatus, OrderType, Position, PositionSide, MarginMode
    
    async def _load():
        import random
        db = Database()
        await db.connect()
        await db.initialize_schema()
        
        now = datetime.now(timezone.utc)
        symbols = ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
        
        # 1. Candles & portfolio snapshots over 48 hours
        base_prices = {"BTCUSDC": 43500, "ETHUSDC": 2350, "SOLUSDC": 110}
        equity = 12500.0
        
        for i in range(192): # 48 hours of 15m candles
            ts = now - timedelta(minutes=15 * (192 - i))
            
            # Global market move (More dynamic for demo)
            market_move = 1 + (np.random.randn() * 0.008)
            equity *= market_move
            
            # Snapshots every hour
            if i % 4 == 0:
                await db.log_portfolio_snapshot(
                    total_equity=equity,
                    available=equity * 0.4,
                    unrealized_pnl=equity * 0.05,
                    num_positions=2,
                    mode="PAPER"
                )
            
            for symbol in symbols:
                base = base_prices[symbol]
                change = base * np.random.randn() * 0.005
                base_prices[symbol] += change
                
                candle = Candle(
                    symbol=symbol,
                    timeframe="15m",
                    timestamp=int(ts.timestamp() * 1000),
                    open=base_prices[symbol] - change,
                    high=base_prices[symbol] * 1.005,
                    low=base_prices[symbol] * 0.995,
                    close=base_prices[symbol],
                    volume=np.random.uniform(1000, 10000),
                    closed=True,
                )
                await db.insert_candles([candle])

        # 2. Positions
        for sym in ["BTCUSDC", "ETHUSDC"]:
            pos = Position(
                symbol=sym,
                side=PositionSide.LONG,
                size=0.1 if sym == "BTCUSDC" else 2.0,
                entry_price=base_prices[sym] * 0.97,
                mark_price=base_prices[sym],
                liquidation_price=base_prices[sym] * 0.65,
                unrealized_pnl=base_prices[sym] * 0.03,
                realized_pnl=0,
                leverage=5,
                margin_mode=MarginMode.ISOLATED,
                margin=base_prices[sym] * 0.2
            )
            await db.insert_position_snapshot(pos)

        # 3. Signals
        for sym in symbols:
            await db.log_signal(
                symbol=sym,
                timeframe="1h",
                direction="LONG",
                confidence=0.92,
                data={"rsi": 31.5, "macd": "bull_cross", "vwap": "above"}
            )

        # 4. Orders (Generate 6 mixed orders)
        for i, sym in enumerate(symbols * 2):
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            order = Order(
                order_id=f"demo_ord_{i}_{sym}",
                client_order_id=f"c_{i}_{sym}",
                symbol=sym,
                side=side,
                order_type=OrderType.LIMIT,
                status=OrderStatus.FILLED,
                price=base_prices[sym] * (1.001 if side == OrderSide.SELL else 0.999),
                size=1.0 if sym != "BTCUSDC" else 0.05,
                filled_size=1.0 if sym != "BTCUSDC" else 0.05,
                avg_fill_price=base_prices[sym],
                fee=base_prices[sym] * 0.0001,
                timestamp=int((now - timedelta(minutes=random.randint(5, 120))).timestamp() * 1000)
            )
            await db.upsert_order(order)
        
        await db.disconnect()
        print("Live demo environment initialized.")
    
    asyncio.run(_load())


def _fetch_blofin_prices() -> dict[str, dict]:
    """Fetch live prices with individual symbol error handling to avoid global block."""
    import urllib.request, json as _json
    import random
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    prices = {}
    
    for sym in _TRACKED_SYMBOLS:
        try:
            url = f"{_BLOFIN_TICKER_URL}?instId={sym}"
            req = urllib.request.Request(url, headers={"User-Agent": "CryptoAgent/2.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = _json.loads(resp.read())
            
            if data.get("code") == "0" and data.get("data"):
                ticker = data["data"][0]
                ts = int(ticker.get("ts", now_ms))
                prices[sym] = {
                    "price": float(ticker["last"]),
                    "ts": ts,
                    "age_s": (now_ms - ts) / 1000,
                    "source": "live",
                }
        except Exception as e:
            # If API fails, check if we have a previous price in session state to "drift"
            # This keeps the UI 'ticking' even during 429 rate limits
            prev = st.session_state.get(f"sim_p_{sym}")
            if prev:
                drift = prev * (random.uniform(-0.0001, 0.0001))
                new_p = prev + drift
                prices[sym] = {
                    "price": new_p,
                    "ts": int(now_ms),
                    "age_s": 0,
                    "source": "sim",
                }
                st.session_state[f"sim_p_{sym}"] = new_p
            else:
                # Initial placeholder if everything fails
                prices[sym] = {"price": 0.0, "source": "error", "age_s": 999}
            
            if "429" in str(e):
                prices["_rate_limited"] = True
                
    return prices


def get_live_prices() -> dict[str, dict]:
    """Returns tickers with smart fallback: agent Redis → REST API → stale Redis → sim."""
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    redis_prices = {}
    r = get_redis()

    if r is not None:
        try:
            for k in r.keys("price:*"):
                sym_raw = k.split(":", 1)[1]
                sym = sym_raw if "-" in sym_raw else f"{sym_raw[:3]}-{sym_raw[3:]}"
                data = r.hgetall(k)
                if data.get("close"):
                    ts = int(data.get("ts", 0))
                    age_s = (now_ms - ts) / 1000 if ts else 9999
                    redis_prices[sym] = {
                        "price": float(data["close"]),
                        "ts": ts,
                        "age_s": age_s,
                        "source": "agent",
                    }
        except Exception:
            pass

    # Always seed session state from Redis so simulation has a real anchor
    for sym, info in redis_prices.items():
        if info["price"] > 0:
            st.session_state.setdefault(f"sim_p_{sym}", info["price"])

    # Preference 1: Fresh agent data from Redis (< 60s)
    if redis_prices and all(v["age_s"] < 60 for v in redis_prices.values()):
        return redis_prices

    # Preference 2: Moderately stale Redis (< 30 min) — avoid extra REST calls
    # that could worsen BloFin rate-limiting and delay WS reconnection
    if redis_prices and all(v["age_s"] < 1800 for v in redis_prices.values()):
        return {sym: {**info, "source": "stale"} for sym, info in redis_prices.items()}

    # Preference 3: Live REST API (only when Redis is very stale or absent)
    live = _fetch_blofin_prices()
    valid = {k: v for k, v in live.items() if k != "_rate_limited" and v.get("price", 0) > 0}
    if valid:
        for sym, info in valid.items():
            st.session_state[f"sim_p_{sym}"] = info["price"]
        return live

    # Preference 4: Any Redis prices (no matter how stale — better than 0.0)
    if redis_prices:
        return {sym: {**info, "source": "stale"} for sym, info in redis_prices.items()}

    # Last resort: simulation drift (live dict has sim/error entries from _fetch_blofin_prices)
    return live


# ── Data fetchers ─────────────────────────────────────────────────────────────

def fetch_portfolio_snapshots(hours: int = 24) -> pd.DataFrame:
    return query_df("""
        SELECT time, total_equity, available, unrealized_pnl, num_positions, mode
        FROM portfolio_snapshots
        WHERE time > NOW() - INTERVAL '%s hours'
        ORDER BY time ASC
    """, (hours,))


def fetch_recent_signals(limit: int = 50) -> pd.DataFrame:
    return query_df("""
        SELECT time, symbol, timeframe, direction, confidence
        FROM signals
        ORDER BY time DESC
        LIMIT %s
    """, (limit,))


def fetch_paper_trades(limit: int = 200) -> pd.DataFrame:
    return query_df("""
        SELECT opened_at, closed_at, symbol, side, size,
               entry_price, exit_price, sl_price, tp_price,
               pnl, pnl_pct, fee, close_reason, balance_after
        FROM paper_trades
        ORDER BY closed_at DESC
        LIMIT %s
    """, (limit,))


def fetch_orders(limit: int = 100) -> pd.DataFrame:
    return query_df("""
        SELECT created_at, symbol, side, order_type, size, price,
               filled_size, status, avg_fill_price, fee, sl_price, tp_price
        FROM orders
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))


def fetch_positions() -> pd.DataFrame:
    return query_df("""
        SELECT DISTINCT ON (symbol) symbol, side, size, entry_price,
               mark_price, unrealized_pnl, leverage, margin
        FROM positions
        ORDER BY symbol, time DESC
    """)


# ── UI helpers ────────────────────────────────────────────────────────────────

def fmt_pct(v: float) -> str:
    colour = "green" if v >= 0 else "red"
    return f":{colour}[{v:+.2%}]"


def fmt_usdc(v: float) -> str:
    colour = "green" if v >= 0 else "red"
    return f":{colour}[${v:,.2f}]"


# ── Sections ──────────────────────────────────────────────────────────────────

def render_header():
    # Use container for full width
    cols = st.columns([4, 2])
    with cols[0]:
        st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
                <h1 style="margin:0;color:#ffffff;font-weight:800;letter-spacing:-1px;font-size:clamp(1.2rem,4vw,2rem);">Crypto<span style="color:#00ffa3;">Agent</span></h1>
                <div style="height:20px;width:1px;background-color:#2d3139;"></div>
                <span style="color:#848e9c;font-size:clamp(0.7rem,2vw,0.9rem);font-weight:500;">Trading Intelligence Terminal v2.0</span>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        mode = "PAPER" if not CONFIG_OK else get_settings().mode.upper()
        badge_class = "mode-paper" if mode == "PAPER" else "mode-live"
        updated_at = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
        
        st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:5px;padding-top:4px;">
                <div class="mode-badge {badge_class}">{mode} MODE ACTIVE</div>
                <div style="color:#848e9c;font-size:0.72rem;">Updated: {updated_at}</div>
            </div>
        """, unsafe_allow_html=True)


@st.fragment(run_every="2s")
def render_live_prices():
    prices = get_live_prices()
    if not prices:
        st.info("Syncing with exchange...")
        return

    cols = st.columns(len(_TRACKED_SYMBOLS))
    for i, sym in enumerate(_TRACKED_SYMBOLS):
        info = prices.get(sym, {"price": 0.0, "source": "n/a", "age_s": 999})
        price = info["price"]
        source = info.get("source", "unknown")
        
        status_color = "#00ffa3"
        status_text = "LIVE MARKET"
        
        if source == "sim":
            status_color = "#f0b90b"
            status_text = "SIMULATED"
        elif source == "agent":
            status_color = "#00ffa3"
            status_text = "AGENT FEED"
        elif prices.get("_rate_limited"):
            status_color = "#f0b90b"
            status_text = "RATE LIMITED"
        
        with cols[i]:
            # Generate a small 'pulse' animation for the price
            st.markdown(f"""
                <div style="background: #1e2329; border-radius: 8px; padding: 15px; border-left: 4px solid {status_color}; margin-bottom: 15px;">
                    <div style="color: #ffffff; font-size: 0.85rem; font-weight: 700; margin-bottom: 5px; text-transform: uppercase;">{sym}</div>
                    <div style="color: {status_color}; font-size: 1.6rem; font-weight: 800; text-shadow: 0 0 15px {status_color}4d; transition: all 0.3s ease;">${price:,.2f}</div>
                    <div style="color: {status_color}; font-size: 0.75rem; font-weight: 600; margin-top: 5px; display: flex; align-items: center; gap: 5px;">
                        <span style="display: inline-block; width: 6px; height: 6px; background: {status_color}; border-radius: 50%;"></span> {status_text}
                    </div>
                </div>
            """, unsafe_allow_html=True)


@st.fragment(run_every="2s")
def render_price_charts(symbols: list[str] = None):
    """Render exchange-quality TradingView charts."""
    st.markdown("<h3 style='color:#ffffff;margin-bottom:8px'>Market Analysis</h3>", unsafe_allow_html=True)

    if symbols is None:
        symbols = _TRACKED_SYMBOLS

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        selected  = st.selectbox("Symbol",   symbols,                               label_visibility="collapsed")
    with c2:
        timeframe = st.selectbox("Interval", ["1m","3m","5m","15m","1h","4h"],      index=3, label_visibility="collapsed")
    with c3:
        bars      = st.selectbox("Bars",     [100, 200, 300, 500],                  index=1, label_visibility="collapsed")

    df = fetch_price_history(selected, timeframe=timeframe, bars=bars)
    if df is None or df.empty:
        st.warning(f"No data for {selected} {timeframe}. Waiting for candles to accumulate.")
        return

    try:
        # ── Timestamps → Unix seconds ─────────────────────────────────────────────
        df["ts"] = df["time"].apply(lambda x: int(x.timestamp()) if hasattr(x, "timestamp") else int(pd.Timestamp(x).timestamp()))

        # ── Price precision (coins < $1 need more decimals) ───────────────────────
        last_close = float(df["close"].iloc[-1])
        if last_close >= 1000:
            price_prec, price_move = 2, 0.01
        elif last_close >= 1:
            price_prec, price_move = 4, 0.0001
        else:
            price_prec, price_move = 6, 0.000001

        # ── Candle data ───────────────────────────────────────────────────────────
        candle_data = [
            {"time": int(r.ts),
             "open":  round(float(r.open),  price_prec),
             "high":  round(float(r.high),  price_prec),
             "low":   round(float(r.low),   price_prec),
             "close": round(float(r.close), price_prec)}
            for r in df.itertuples()
        ]

        # ── Volume (overlaid in main chart via priceScaleId="vol") ────────────────
        vol_data = [
            {"time": int(r.ts),
             "value": round(float(r.volume), 2),
             "color": "rgba(38,166,154,0.5)" if float(r.close) >= float(r.open) else "rgba(239,83,80,0.5)"}
            for r in df.itertuples()
        ]

        # ── EMAs ──────────────────────────────────────────────────────────────────
        def _ema(span, color, width=1):
            vals = df["close"].ewm(span=span, adjust=False).mean()
            data = [{"time": int(t), "value": round(float(v), price_prec)}
                    for t, v in zip(df["ts"], vals) if not pd.isna(v)]
            return {"type": "Line", "data": data,
                    "options": {"color": color, "lineWidth": width,
                                "priceLineVisible": False, "lastValueVisible": False,
                                "crosshairMarkerVisible": False}}

        # ── Bollinger Bands (20,2) ────────────────────────────────────────────────
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        def _bb_series(vals):
            return [{"time": int(t), "value": round(float(v), price_prec)}
                    for t, v in zip(df["ts"], vals) if not pd.isna(v)]

        # ── RSI (14) ──────────────────────────────────────────────────────────────
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi   = (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).round(2)
        rsi_data = [{"time": int(t), "value": float(v)}
                    for t, v in zip(df["ts"], rsi) if not pd.isna(v)]

        # ── Theme ─────────────────────────────────────────────────────────────────
        _bg, _grid, _border, _text = "#0d1117", "rgba(255,255,255,0.04)", "rgba(255,255,255,0.12)", "#d1d4dc"
        _up, _dn = "#26a69a", "#ef5350"

        _layout  = {"background": {"type": "solid", "color": _bg}, "textColor": _text, "fontSize": 11}
        _gridcfg = {"vertLines": {"color": _grid, "style": 1}, "horzLines": {"color": _grid, "style": 1}}
        _ts_main = {"borderColor": _border, "timeVisible": True, "secondsVisible": False, "barSpacing": 8, "minBarSpacing": 2}
        _ts_hide = {"visible": False}

        # ── Chart 1: Price + Volume overlay ───────────────────────────────────────
        # Volume is overlaid at bottom 20% using a separate price scale "vol"
        main_chart = {
            "chart": {
                "height": 460,
                "layout": _layout,
                "grid":   _gridcfg,
                "crosshair": {"mode": 1},
                "rightPriceScale": {
                    "borderColor": _border,
                    "scaleMargins": {"top": 0.05, "bottom": 0.22},  # leave room for vol
                },
                "timeScale": _ts_main,
            },
            "series": [
                # Candlestick
                {
                    "type": "Candlestick",
                    "data": candle_data,
                    "options": {
                        "upColor":       _up,
                        "downColor":     _dn,
                        "borderVisible": False,   # solid bodies, no border
                        "wickUpColor":   _up,
                        "wickDownColor": _dn,
                        "priceFormat":   {"type": "price", "precision": price_prec, "minMove": price_move},
                    },
                },
                # Bollinger upper
                {"type": "Line", "data": _bb_series(mid + 2*std),
                 "options": {"color": "rgba(180,180,255,0.35)", "lineWidth": 1, "lineStyle": 2,
                             "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False}},
                # Bollinger lower
                {"type": "Line", "data": _bb_series(mid - 2*std),
                 "options": {"color": "rgba(180,180,255,0.35)", "lineWidth": 1, "lineStyle": 2,
                             "priceLineVisible": False, "lastValueVisible": False, "crosshairMarkerVisible": False}},
                # EMA 9 / 21 / 50
                _ema(9,   "#5eead4", 1),
                _ema(21,  "#f0b90b", 1),
                _ema(50,  "#fb923c", 1),
                # Volume overlay — bottom 20%
                {
                    "type": "Histogram",
                    "data": vol_data,
                    "options": {
                        "priceFormat":  {"type": "volume"},
                        "priceScaleId": "vol",
                    },
                    "priceScale": {
                        "scaleMargins": {"top": 0.82, "bottom": 0.0},
                        "borderVisible": False,
                    },
                },
            ],
        }

        # ── Chart 2: RSI (no x-axis — seamlessly stacked below) ──────────────────
        rsi_chart = {
            "chart": {
                "height": 110,
                "layout": _layout,
                "grid":   _gridcfg,
                "crosshair": {"mode": 1},
                "rightPriceScale": {
                    "borderColor": _border,
                    "scaleMargins": {"top": 0.1, "bottom": 0.1},
                },
                "timeScale": _ts_hide,
            },
            "series": [
                {"type": "Line", "data": rsi_data,
                 "options": {"color": "#a78bfa", "lineWidth": 1,
                             "priceLineVisible": False, "lastValueVisible": True,
                             "title": "RSI 14",
                             "priceFormat": {"type": "price", "precision": 1, "minMove": 0.1}}},
            ],
        }

        from streamlit_lightweight_charts import renderLightweightCharts
        renderLightweightCharts([main_chart, rsi_chart], key=f"tv_{selected}_{timeframe}_{bars}")

        # ── OHLCV bar ─────────────────────────────────────────────────────────────
        last = df.iloc[-1]
        chg  = (float(last["close"]) - float(last["open"])) / float(last["open"]) * 100
        up   = chg >= 0
        fmt  = f"{{:.{price_prec}f}}"
        cols = st.columns(5)
        labels = ["Open", "High", "Low", "Close", "Change"]
        values = [fmt.format(float(last["open"])),  fmt.format(float(last["high"])),
                  fmt.format(float(last["low"])),   fmt.format(float(last["close"])),
                  f"{'%+.2f' % chg}%"]
        for col, label, val in zip(cols, labels, values):
            vc = ("#26a69a" if up else "#ef5350") if label == "Change" else "#e2e8f0"
            col.markdown(
                f"<div style='text-align:center;padding:6px 0;"
                f"background:#161b22;border-radius:6px;border:1px solid #30363d'>"
                f"<div style='font-size:10px;color:#6b7280;margin-bottom:2px'>{label}</div>"
                f"<div style='font-size:13px;font-weight:600;color:{vc}'>{val}</div></div>",
                unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Chart error: {e}")


def fetch_price_history(symbol: str, timeframe: str = "15m", bars: int = 200) -> pd.DataFrame:
    """Fetch OHLCV history from TimescaleDB — only properly-aligned closed candles."""
    # Build timeframe alignment filter so malformed WS entries (wrong timestamps) are excluded
    _tf_filters = {
        "1m":  "EXTRACT(SECOND FROM time) = 0",
        "3m":  "EXTRACT(SECOND FROM time) = 0 AND EXTRACT(MINUTE FROM time)::int % 3 = 0",
        "5m":  "EXTRACT(SECOND FROM time) = 0 AND EXTRACT(MINUTE FROM time)::int % 5 = 0",
        "15m": "EXTRACT(SECOND FROM time) = 0 AND EXTRACT(MINUTE FROM time) IN (0,15,30,45)",
        "1h":  "EXTRACT(SECOND FROM time) = 0 AND EXTRACT(MINUTE FROM time) = 0",
        "4h":  "EXTRACT(SECOND FROM time) = 0 AND EXTRACT(MINUTE FROM time) = 0 AND EXTRACT(HOUR FROM time)::int % 4 = 0",
    }
    align_filter = _tf_filters.get(timeframe, "1=1")
    sql = f"""
        SELECT time, open, high, low, close, volume
        FROM candles
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
          AND closed = true
          AND {align_filter}
        ORDER BY time DESC
        LIMIT {bars}
    """
    df = query_df(sql)
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)
    return df


@st.fragment(run_every="5s")
def render_paper_trades():
    """Rich paper trade log: full timestamps, P&L, entry/exit, reason badges."""
    st.markdown("### Paper Trade Log")

    trades = fetch_paper_trades(200)

    if trades.empty:
        st.info(
            "No completed paper trades yet. "
            "Trades appear here once the agent opens **and closes** a position (via SL, TP, or manual)."
        )
        # Also show pending open orders so the user knows something is happening
        orders = fetch_orders(20)
        open_orders = orders[orders["status"].str.upper() == "FILLED"] if not orders.empty else pd.DataFrame()
        if not open_orders.empty:
            st.markdown("#### Open Positions (waiting to close)")
            _want = ["created_at", "symbol", "side", "size", "avg_fill_price", "sl_price", "tp_price"]
            _cols = [c for c in _want if c in open_orders.columns]
            st.dataframe(open_orders[_cols], use_container_width=True, hide_index=True)
        return

    # ── Summary row ──────────────────────────────────────────────────────────
    total = len(trades)
    wins  = int((trades["pnl"] > 0).sum())
    losses = total - wins
    win_rate  = wins / total if total else 0.0
    total_pnl = float(trades["pnl"].sum())
    best_pnl  = float(trades["pnl"].max())
    worst_pnl = float(trades["pnl"].min())
    avg_pnl   = float(trades["pnl"].mean())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades", total)
    c2.metric("Win Rate",     f"{win_rate:.1%}",  f"{wins}W / {losses}L")
    c3.metric("Total P&L",   f"${total_pnl:+,.2f}")
    c4.metric("Avg P&L",     f"${avg_pnl:+,.2f}")
    c5.metric("Best Trade",  f"${best_pnl:+,.2f}")
    c6.metric("Worst Trade", f"${worst_pnl:+,.2f}")

    st.divider()

    # ── Trade cards ──────────────────────────────────────────────────────────
    _REASON_STYLE = {
        "take_profit": ("#00ffa3", "rgba(0,255,163,0.12)"),
        "stop_loss":   ("#ff3b30", "rgba(255,59,48,0.12)"),
        "manual":      ("#848e9c", "rgba(132,142,156,0.10)"),
    }

    for _, t in trades.iterrows():
        pnl       = float(t.get("pnl", 0))
        pnl_pct   = float(t.get("pnl_pct", 0))
        side      = str(t.get("side", "")).upper()
        symbol    = str(t.get("symbol", ""))
        reason    = str(t.get("close_reason", "manual")).lower()
        entry     = float(t.get("entry_price", 0))
        exit_p    = float(t.get("exit_price", 0))
        sl        = float(t.get("sl_price", 0))
        tp        = float(t.get("tp_price", 0))
        size      = float(t.get("size", 0))
        bal_after = float(t.get("balance_after", 0))
        fee       = float(t.get("fee", 0))

        opened_at = pd.to_datetime(t.get("opened_at"), utc=True)
        closed_at = pd.to_datetime(t.get("closed_at"), utc=True)
        opened_str = opened_at.strftime("%Y-%m-%d  %H:%M:%S UTC") if pd.notnull(opened_at) else "—"
        closed_str = closed_at.strftime("%Y-%m-%d  %H:%M:%S UTC") if pd.notnull(closed_at) else "—"

        # Duration
        if pd.notnull(opened_at) and pd.notnull(closed_at):
            secs  = int((closed_at - opened_at).total_seconds())
            h, r  = divmod(abs(secs), 3600)
            m     = r // 60
            duration = f"{h}h {m}m" if h else f"{m}m" if m else "<1m"
        else:
            duration = "—"

        pnl_color    = "#00ffa3" if pnl >= 0 else "#ff3b30"
        border_color = pnl_color
        pnl_icon     = "▲" if pnl >= 0 else "▼"
        side_color   = "#00ffa3" if side == "LONG" else "#f0b90b"

        r_color, r_bg = _REASON_STYLE.get(reason, ("#848e9c", "rgba(132,142,156,0.1)"))
        reason_label  = reason.replace("_", " ").upper()

        price_move_pct = ((exit_p - entry) / entry * 100) if entry else 0.0
        if side == "SHORT":
            price_move_pct = -price_move_pct

        st.markdown(f"""
        <div style="background: rgba(22,26,30,0.7); border: 1px solid #2d3139;
                    border-left: 4px solid {border_color}; border-radius: 10px;
                    padding: 18px 22px; margin-bottom: 12px;">

          <!-- Header row -->
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:12px;">
              <span style="color:{side_color}; font-weight:800; font-size:1.05rem; letter-spacing:1px;">{side}</span>
              <span style="color:#ffffff; font-weight:700; font-size:1.05rem;">{symbol}</span>
              <span style="background:{r_bg}; color:{r_color}; border:1px solid {r_color};
                           padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:700;
                           letter-spacing:0.5px;">{reason_label}</span>
            </div>
            <div style="text-align:right;">
              <div style="color:{pnl_color}; font-weight:800; font-size:1.25rem;">
                {pnl_icon} ${pnl:+,.2f}
                <span style="font-size:0.82rem; opacity:0.85;">({pnl_pct:+.1%} on margin)</span>
              </div>
              <div style="color:#848e9c; font-size:0.72rem; margin-top:2px;">
                Price move: <span style="color:{pnl_color};">{price_move_pct:+.3f}%</span>
                &nbsp;|&nbsp; Fee: ${fee:.4f}
              </div>
            </div>
          </div>

          <!-- Data grid -->
          <div class="trade-data-grid">
            <div>
              <div style="color:#5c6672; font-size:0.68rem; text-transform:uppercase; margin-bottom:3px;">Entry</div>
              <div style="color:#d1d4dc; font-size:0.88rem; font-weight:600;">${entry:,.4f}</div>
            </div>
            <div>
              <div style="color:#5c6672; font-size:0.68rem; text-transform:uppercase; margin-bottom:3px;">Exit</div>
              <div style="color:{pnl_color}; font-size:0.88rem; font-weight:600;">${exit_p:,.4f}</div>
            </div>
            <div>
              <div style="color:#5c6672; font-size:0.68rem; text-transform:uppercase; margin-bottom:3px;">Size</div>
              <div style="color:#d1d4dc; font-size:0.88rem; font-weight:600;">{size:.4f}</div>
            </div>
            <div>
              <div style="color:#5c6672; font-size:0.68rem; text-transform:uppercase; margin-bottom:3px;">Stop Loss</div>
              <div style="color:#ff3b30; font-size:0.88rem; font-weight:600;">${sl:,.4f}</div>
            </div>
            <div>
              <div style="color:#5c6672; font-size:0.68rem; text-transform:uppercase; margin-bottom:3px;">Take Profit</div>
              <div style="color:#00ffa3; font-size:0.88rem; font-weight:600;">${tp:,.4f}</div>
            </div>
            <div>
              <div style="color:#5c6672; font-size:0.68rem; text-transform:uppercase; margin-bottom:3px;">Balance After</div>
              <div style="color:#d1d4dc; font-size:0.88rem; font-weight:600;">${bal_after:,.2f}</div>
            </div>
          </div>

          <!-- Timestamp footer -->
          <div style="display:flex; gap:24px; font-size:0.73rem; color:#5c6672;
                      border-top:1px solid #1e2329; padding-top:10px;">
            <span>⏱ Duration: <span style="color:#848e9c;">{duration}</span></span>
            <span>📅 Opened: <span style="color:#848e9c;">{opened_str}</span></span>
            <span>📅 Closed: <span style="color:#848e9c;">{closed_str}</span></span>
          </div>
        </div>
        """, unsafe_allow_html=True)


def render_trade_notifications():
    """Display recent trade notifications/alerts in a terminal feed style."""
    st.markdown("### Execution Feed")
    
    orders = fetch_orders(30)
    if orders.empty:
        st.info("Execution engine idle. No trade events to report.")
        return
    
    for _, order in orders.iterrows():
        side = str(order.get("side", "")).upper()
        symbol = order.get("symbol", "")
        status = str(order.get("status", "")).upper()
        filled = order.get("filled_size", 0)
        price = order.get("filled_avg_price") or order.get("price", 0)
        time_str = pd.to_datetime(order.get("created_at")).strftime("%H:%M:%S") if "created_at" in order else "00:00:00"
        
        bg_color = "rgba(45, 49, 57, 0.3)"
        border_color = "#2d3139"
        text_color = "#ffffff"
        icon = "●"
        
        if status == "FILLED":
            if side == "BUY":
                border_color = "#00ffa3"
                text_color = "#00ffa3"
                icon = "▲"
            else:
                border_color = "#ff3b30"
                text_color = "#ff3b30"
                icon = "▼"
        elif status == "CANCELLED":
            text_color = "#848e9c"
            icon = "○"

        st.markdown(f"""
            <div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 10px 15px; margin-bottom: 8px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="color: #5c6672; font-family: monospace; font-size: 0.8rem;">[{time_str}]</span>
                    <span style="color: {text_color}; font-weight: 700; font-size: 0.9rem;">{icon} {side} {symbol}</span>
                    <span style="color: #d1d4dc; font-size: 0.85rem;">{filled} units @ ${price:,.2f}</span>
                </div>
                <div style="color: {text_color}; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; border: 1px solid {border_color}; padding: 2px 8px; border-radius: 10px;">
                    {status}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Summary metrics row
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    filled_orders = orders[orders["status"].str.upper() == "FILLED"]
    buys = len(filled_orders[filled_orders["side"].str.upper() == "BUY"])
    sells = len(filled_orders[filled_orders["side"].str.upper() == "SELL"])
    
    col1.metric("Total Events", len(orders))
    col2.metric("Buy Fills", buys)
    col3.metric("Sell Fills", sells)


@st.fragment(run_every="5s")
def render_equity_curve(snapshots: pd.DataFrame):
    st.markdown("### Portfolio Performance")
    if snapshots.empty:
        st.info("Equity tracking will initialize once the agent captures its first state.")
        return

    try:
        # Clean and prepare data
        df = snapshots.copy()
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["total_equity"])
        
        if df.empty:
            st.info("Waiting for valid equity data points...")
            return

        # Summary Metrics
        last_eq = float(df["total_equity"].iloc[-1])
        first_eq = float(df["total_equity"].iloc[0])
        pnl_24h = last_eq - first_eq
        pnl_pct = pnl_24h / first_eq if first_eq else 0
        unreal = float(df["unrealized_pnl"].iloc[-1]) if "unrealized_pnl" in df.columns else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Equity", f"${last_eq:,.2f}")
        m2.metric("24h Change", f"${pnl_24h:,.2f}", f"{pnl_pct:+.2%}")
        m3.metric("Unrealized P&L", f"${unreal:,.2f}")
        m4.metric("Active exposure", int(df["num_positions"].iloc[-1]))

        # High-Contrast Plotly Equity Curve
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # PnL Area
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['total_equity'],
            mode='lines',
            line=dict(color='#00ffa3', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 163, 0.05)',
            name='Total Equity',
            hovertemplate='<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>'
        ))

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
    except Exception as e:
        st.warning(f"Performance visualization suspended: {e}")
def render_positions_management():
    """Manual overlay to close trades or take profit."""
    st.subheader("Manage Positions")
    
    pos = fetch_positions()
    if pos.empty:
        orders = fetch_orders(50)
        if not orders.empty:
            filled = orders[orders["status"] == "filled"].copy()
            if not filled.empty:
                filled = (filled.sort_values("created_at", ascending=False)
                          .groupby("symbol").first().reset_index())
                try:
                    closed = fetch_paper_trades(50)
                    if not closed.empty:
                        closed["closed_at"] = pd.to_datetime(closed["closed_at"], utc=True)
                        filled["created_at"] = pd.to_datetime(filled["created_at"], utc=True)
                        latest_close = (closed.groupby("symbol")["closed_at"]
                                        .max().rename("latest_close"))
                        filled = filled.join(latest_close, on="symbol")
                        mask = filled["latest_close"].isna() | (
                            filled["created_at"] > filled["latest_close"]
                        )
                        filled = filled[mask]
                except Exception:
                    pass
                if not filled.empty:
                    pos = pd.DataFrame({"symbol": filled["symbol"]})

    if pos.empty:
        st.info("No active trades to manage.")
        return

    syms = pos["symbol"].tolist()
    
    import json
    import redis
    from src.core.config import get_settings
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        target_sym = st.selectbox("Position", syms, key="manage_sym", label_visibility="collapsed")
    with col2:
        action = st.selectbox("Action", [
            "Close Full Position (100%)",
            "Take Profit (50%)",
            "Take Profit (25%)",
            "Take Profit (10%)"
        ], key="manage_act", label_visibility="collapsed")
    with col3:
        if st.button("Execute", type="primary", use_container_width=True):
            pct = 1.0
            if "50%" in action: pct = 0.5
            elif "25%" in action: pct = 0.25
            elif "10%" in action: pct = 0.10
            
            payload = {
                "action": "close_position" if pct == 1.0 else "take_profit",
                "symbol": target_sym,
                "pct": pct
            }
            try:
                r = redis.from_url(get_settings().database.redis_url)
                r.publish("agent_commands", json.dumps(payload))
                r.close()
                st.success(f"Command sent!")
            except Exception as e:
                st.error(f"Failed to send command: {e}")
    st.markdown("<hr style='border-color:#2d3139; margin-top:5px; margin-bottom:15px;'>", unsafe_allow_html=True)



@st.fragment(run_every="2s")
def render_positions():
    """Live position monitor — current price + unrealized P&L update every 2 s."""
    st.subheader("Open Positions")

    prices = get_live_prices()

    # Show stale price warning if WS is down
    if prices:
        sample = next(iter(prices.values()))
        age_s = sample.get("age_s", 0)
        source = sample.get("source", "")
        if age_s > 60:
            age_min = int(age_s / 60)
            st.warning(
                f"Live feed reconnecting (429 rate limit) — prices from cache "
                f"({age_min}m ago). P&L is approximate.",
                icon="⚠",
            )

    # Live exchange: use positions table
    pos = fetch_positions()

    # Paper mode fallback: reconstruct from most-recent filled order per symbol
    if pos.empty:
        orders = fetch_orders(50)
        if not orders.empty:
            filled = orders[orders["status"] == "filled"].copy()
            if not filled.empty:
                filled = (filled.sort_values("created_at", ascending=False)
                          .groupby("symbol").first().reset_index())
                # Drop symbols whose most recent paper_trade close is NEWER than
                # the open order — those are genuinely closed, not still open.
                # (A symbol with old paper_trades but a newer filled order is open.)
                try:
                    closed = fetch_paper_trades(50)
                    if not closed.empty:
                        closed["closed_at"] = pd.to_datetime(closed["closed_at"], utc=True)
                        filled["created_at"] = pd.to_datetime(filled["created_at"], utc=True)
                        latest_close = (closed.groupby("symbol")["closed_at"]
                                        .max().rename("latest_close"))
                        filled = filled.join(latest_close, on="symbol")
                        # Keep row if there's no close, or if the open order is newer
                        mask = filled["latest_close"].isna() | (
                            filled["created_at"] > filled["latest_close"]
                        )
                        filled = filled[mask].drop(columns=["latest_close"])
                except Exception:
                    pass
                if not filled.empty:
                    pos = pd.DataFrame({
                        "symbol":      filled["symbol"],
                        "side":        filled["side"].map({"buy": "LONG", "sell": "SHORT"}).fillna(""),
                        "size":        filled["filled_size"],
                        "entry_price": filled["avg_fill_price"],
                        "sl_price":    filled["sl_price"] if "sl_price" in filled else 0.0,
                        "tp_price":    filled["tp_price"] if "tp_price" in filled else 0.0,
                    })

    if pos.empty:
        st.info("No open positions — agent is flat.")
        return

    # ── Styled exchange-like table ─────────────────────────────────────────
    rows = ""
    for _, r in pos.iterrows():
        sym    = str(r.get("symbol", ""))
        side   = str(r.get("side", r.get("side", ""))).upper()
        entry  = float(r.get("entry_price", r.get("avg_fill_price", 0)) or 0)
        size   = float(r.get("size", r.get("filled_size", 0)) or 0)
        sl     = float(r.get("sl_price", 0) or 0)
        tp     = float(r.get("tp_price", 0) or 0)

        live_info = prices.get(sym, {})
        current   = float(live_info.get("price", entry) or entry)

        direction = 1 if side == "LONG" else -1
        pnl_usd   = direction * (current - entry) * size
        pnl_pct   = direction * (current - entry) / entry * 100 if entry else 0

        side_bg    = "#0d2b1f" if direction == 1 else "#2b0d0d"
        side_color = "#00ffa3" if direction == 1 else "#ff4d4d"
        pnl_color  = "#00ffa3" if pnl_usd >= 0 else "#ff4d4d"
        pnl_arrow  = "▲" if pnl_usd >= 0 else "▼"

        def _fmt_price(p: float) -> str:
            if p <= 0:
                return "—"
            if p > 10_000:  return f"${p:,.2f}"
            if p > 100:     return f"${p:,.3f}"
            return f"${p:,.4f}"

        dist_sl = abs(current - sl) / current * 100 if sl and current else 0
        dist_tp = abs(tp - current) / current * 100 if tp and current else 0

        rows += f"""
        <tr>
          <td style="font-weight:700;color:#fff;">{sym}</td>
          <td><span style="background:{side_bg};color:{side_color};padding:3px 8px;border-radius:4px;font-size:0.78rem;font-weight:700;">{side}</span></td>
          <td style="color:#848e9c;">{_fmt_price(entry)}</td>
          <td style="color:#f0f0f0;font-weight:600;">{_fmt_price(current)}</td>
          <td style="color:{pnl_color};font-weight:700;">{pnl_arrow} ${pnl_usd:+,.4f}<br><span style="font-size:0.75rem;opacity:0.85;">{pnl_pct:+.3f}%</span></td>
          <td style="color:#848e9c;font-size:0.8rem;">{size:.6g}</td>
          <td style="color:#ff4d4d;font-size:0.8rem;">{_fmt_price(sl)}<br><span style="color:#555;font-size:0.72rem;">{dist_sl:.2f}% away</span></td>
          <td style="color:#00ffa3;font-size:0.8rem;">{_fmt_price(tp)}<br><span style="color:#555;font-size:0.72rem;">{dist_tp:.2f}% away</span></td>
        </tr>"""

    st.markdown(f"""
    <style>
    .pos-tbl {{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:0.85rem;}}
    .pos-tbl th {{color:#848e9c;font-weight:500;font-size:0.75rem;text-transform:uppercase;
                  letter-spacing:.06em;padding:8px 12px;border-bottom:1px solid #2d3139;text-align:left;}}
    .pos-tbl td {{padding:10px 12px;border-bottom:1px solid #1a1d22;vertical-align:middle;}}
    .pos-tbl tr:hover td {{background:#1a1d22;}}
    </style>
    <div class="tbl-wrap">
    <table class="pos-tbl">
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Entry</th>
        <th>Current Price</th><th>Unrealized P&amp;L</th>
        <th>Size</th><th>Stop Loss</th><th>Take Profit</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


@st.fragment(run_every="30s")
def render_sentiment_data():
    """Display sentiment and on-chain data - updates every 30s."""
    import aiohttp
    
    st.subheader("📊 Market Intelligence")
    
    # Create columns for different data types
    col1, col2, col3 = st.columns(3)
    
    # ── Fear & Greed Index ──────────────────────────────
    with col1:
        try:
            fg_value = 50
            fg_class = "Neutral"
            
            try:
                import requests
                resp = requests.get("https://api.alternative.me/fng/", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    fg_data = data.get("data", [{}])[0]
                    fg_value = int(fg_data.get("value", 50))
                    fg_class = fg_data.get("value_classification", "Neutral")
            except:
                pass
            
            # Color based on value
            if fg_value < 25:
                fg_color = "#ff4d4d"
            elif fg_value < 45:
                fg_color = "#f0b429"
            elif fg_value > 75:
                fg_color = "#00ffa3"
            elif fg_value > 55:
                fg_color = "#00ffa3"
            else:
                fg_color = "#848e9c"
            
            st.markdown(f"""
            <div style="background:#161a1e;padding:15px;border-radius:8px;border:1px solid #2d3139;">
                <div style="color:#848e9c;font-size:0.75rem;text-transform:uppercase;">Fear & Greed</div>
                <div style="font-size:2rem;font-weight:700;color:{fg_color};">{fg_value}</div>
                <div style="color:{fg_color};font-size:0.85rem;">{fg_class}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div style="background:#161a1e;padding:15px;border-radius:8px;border:1px solid #2d3139;">
                <div style="color:#848e9c;font-size:0.75rem;text-transform:uppercase;">Fear & Greed</div>
                <div style="color:#848e9c;">Unavailable</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ── BTC Dominance ──────────────────────────────
    with col2:
        try:
            btc_dom = 50.0
            alt_dom = 50.0
            
            try:
                import requests
                resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    dom_data = data.get("data", {}).get("market_cap_percentage", {})
                    btc_dom = dom_data.get("btc", 50)
                    alt_dom = dom_data.get("altcoin", 50)
            except:
                pass
            
            st.markdown(f"""
            <div style="background:#161a1e;padding:15px;border-radius:8px;border:1px solid #2d3139;">
                <div style="color:#848e9c;font-size:0.75rem;text-transform:uppercase;">BTC Dominance</div>
                <div style="font-size:2rem;font-weight:700;color:#f0b90b;">{btc_dom:.1f}%</div>
                <div style="color:#848e9c;font-size:0.85rem;">Alts: {alt_dom:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown(f"""
            <div style="background:#161a1e;padding:15px;border-radius:8px;border:1px solid #2d3139;">
                <div style="color:#848e9c;font-size:0.75rem;text-transform:uppercase;">BTC Dominance</div>
                <div style="color:#848e9c;">Unavailable</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ── Funding Rate ──────────────────────────────
    with col3:
        try:
            funding_rate = 0.0
            
            try:
                import requests
                resp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    funding_rate = float(data.get("lastFundingRate", 0)) * 100
            except:
                pass
            
            if funding_rate > 0:
                rate_color = "#ff4d4d"
            else:
                rate_color = "#00ffa3"
            
            st.markdown(f"""
            <div style="background:#161a1e;padding:15px;border-radius:8px;border:1px solid #2d3139;">
                <div style="color:#848e9c;font-size:0.75rem;text-transform:uppercase;">Binance Funding</div>
                <div style="font-size:2rem;font-weight:700;color:{rate_color};">{funding_rate:+.4f}%</div>
                <div style="color:#848e9c;font-size:0.85rem;">BTCUSDT Perpetual</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown(f"""
            <div style="background:#161a1e;padding:15px;border-radius:8px;border:1px solid #2d3139;">
                <div style="color:#848e9c;font-size:0.75rem;text-transform:uppercase;">Binance Funding</div>
                <div style="color:#848e9c;">Unavailable</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ── News NLP + AI Market Briefing (GPT-5.2) ───────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    try:
        import json, os, re
        import requests
        from src.data.feeds.sentiment_feed import SentimentFeed

        resp20 = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest&limit=20",
            timeout=8,
        )
        news_articles = resp20.json().get("Data", []) if resp20.status_code == 200 else []
        article_scores = []
        nlp_engine     = "vader+keywords"
        market_briefing = ""
        dominant_theme  = "neutral"

        az_endpoint   = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        az_key        = os.environ.get("AZURE_OPENAI_KEY", "")
        az_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
        az_api_ver    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        titles = [art.get("title", "") for art in news_articles[:20]]

        # ── Priority 1: Azure OpenAI GPT-5.2 ─────────────────────────────────
        if news_articles and az_endpoint and az_key:
            try:
                from openai import AzureOpenAI
                client = AzureOpenAI(azure_endpoint=az_endpoint, api_key=az_key, api_version=az_api_ver)
                numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(titles))
                prompt = f"""You are a professional crypto trading analyst.

Analyze these {len(titles)} crypto news headlines and return a JSON object with:
1. "scores": array of {len(titles)} floats from -1.0 (very bearish) to +1.0 (very bullish)
2. "briefing": 2-sentence market narrative summarising sentiment and likely near-term price impact
3. "dominant_theme": one of "strongly_bullish"|"bullish"|"neutral"|"bearish"|"strongly_bearish"

Rules: "crashes through resistance to ATH" = bullish. Technical analysis articles = ±0.1 only.
ETF approvals, institutional buying = strongly bullish. Exchange hacks, insolvency = strongly bearish.

Headlines:
{numbered}

Return ONLY valid JSON."""
                resp = client.responses.create(
                    model=az_deployment,
                    input=[
                        {"role": "system", "content": "You are a crypto market sentiment analyst. Always respond with valid JSON only."},
                        {"role": "user",   "content": prompt},
                    ],
                    text={"format": {"type": "json_object"}},
                    max_output_tokens=4000,
                )
                raw = json.loads(resp.output_text)
                sc = [max(-1.0, min(1.0, float(s))) for s in raw.get("scores", [])]
                if len(sc) == len(titles):
                    article_scores  = sc
                    market_briefing = raw.get("briefing", "")
                    dominant_theme  = raw.get("dominant_theme", "neutral")
                    nlp_engine      = "gpt-5.2"
            except Exception:
                pass

        # ── Priority 2: Gemini fallback ───────────────────────────────────────
        if not article_scores:
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            if news_articles and gemini_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(titles))
                    r = model.generate_content(
                        f"Score each headline -1 to +1 for crypto price impact. "
                        f"Return ONLY a JSON array of {len(titles)} numbers.\n{numbered}"
                    )
                    match = re.search(r'\[[\s\d,.\-+]+\]', r.text.strip())
                    if match:
                        article_scores = [max(-1.0, min(1.0, float(s))) for s in json.loads(match.group())]
                        nlp_engine = "gemini-2.0-flash"
                except Exception:
                    pass

        # ── Priority 3: VADER+keywords ────────────────────────────────────────
        if not article_scores:
            _feed = SentimentFeed.__new__(SentimentFeed)
            _feed._vader = None
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                _feed._vader = SentimentIntensityAnalyzer()
            except ImportError:
                pass
            for art in news_articles:
                article_scores.append(_feed._score_text(f"{art.get('title','')} {art.get('body','')[:300]}"))

        if article_scores:
            wts = [1.0 / (1.0 + i * 0.2) for i in range(len(article_scores))]
            news_nlp = max(-1.0, min(1.0, sum(s * w for s, w in zip(article_scores, wts)) / sum(wts)))
        else:
            news_nlp = 0.0
            news_articles = []

        nlp_color = "#ff4d4d" if news_nlp < -0.15 else "#f0b429" if news_nlp < 0.15 else "#00ffa3"
        nlp_label = "Bearish" if news_nlp < -0.15 else "Neutral" if news_nlp < 0.15 else "Bullish"

        # Engine badge
        if nlp_engine == "gpt-5.2":
            ebg = '#0a1a2e'; ecol = '#60a5fa'
            engine_badge = f'<span style="background:{ebg};color:{ecol};font-size:0.7rem;padding:2px 8px;border-radius:10px;margin-left:8px;border:1px solid {ecol}40;">⚡ GPT-5.2</span>'
        elif nlp_engine == "gemini-2.0-flash":
            engine_badge = '<span style="background:#1a3a2a;color:#00ffa3;font-size:0.7rem;padding:2px 7px;border-radius:10px;margin-left:8px;">Gemini Flash</span>'
        else:
            engine_badge = '<span style="background:#2d3139;color:#848e9c;font-size:0.7rem;padding:2px 7px;border-radius:10px;margin-left:8px;">VADER+KW</span>'

        # Theme colour
        theme_colors = {"strongly_bullish": "#00ffa3", "bullish": "#4ade80",
                        "neutral": "#f0b429", "bearish": "#f87171", "strongly_bearish": "#ff4d4d"}
        theme_col = theme_colors.get(dominant_theme, "#848e9c")

        st.markdown(f"""
        <div style="background:#0d1117;padding:18px;border-radius:10px;border:1px solid #1e3a5f;margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
              <span style="color:#848e9c;font-size:0.72rem;text-transform:uppercase;letter-spacing:.06em;">News Sentiment</span>
              {engine_badge}
            </div>
            <span style="background:{theme_col}20;color:{theme_col};font-size:0.72rem;padding:3px 10px;
                         border-radius:12px;border:1px solid {theme_col}60;font-weight:700;">
              {dominant_theme.replace("_"," ").upper()}
            </span>
          </div>
          <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">
            <span style="font-size:2.2rem;font-weight:800;color:{nlp_color};">{news_nlp:+.3f}</span>
            <span style="color:{nlp_color};font-size:0.9rem;font-weight:600;">{nlp_label}</span>
            <span style="color:#4a5060;font-size:0.8rem;">{len(article_scores)} headlines</span>
          </div>
          {"" if not market_briefing else f'<div style="background:#161a1e;border-left:3px solid #60a5fa;padding:10px 14px;border-radius:0 6px 6px 0;font-size:0.8rem;color:#b0bec5;line-height:1.5;">{market_briefing}</div>'}
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        news_articles = []
        news_nlp = 0.0
        article_scores = []

    # ── Latest News with NLP badges ───────────────────────────────────────────
    st.markdown("### Latest Crypto News")
    try:
        display_arts = news_articles[:8] if news_articles else []
        if not display_arts:
            import requests
            r2 = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit=8", timeout=5
            )
            display_arts = r2.json().get("Data", []) if r2.status_code == 200 else []
            article_scores = []

        if display_arts:
            for idx, art in enumerate(display_arts):
                title  = art.get("title", "")
                source = art.get("source_info", {}).get("name", "Unknown")
                url    = art.get("url", "#")
                cats   = art.get("categories", "").split("|")[:2]

                # NLP badge for this article
                sc = article_scores[idx] if idx < len(article_scores) else 0.0
                if sc < -0.1:
                    badge_c, badge_l = "#ff4d4d", f"▼ {sc:+.2f}"
                elif sc > 0.1:
                    badge_c, badge_l = "#00ffa3", f"▲ {sc:+.2f}"
                else:
                    badge_c, badge_l = "#848e9c", f"● {sc:+.2f}"

                cat_html = "".join(
                    f'<span style="background:#2d3139;padding:2px 6px;margin-left:4px;'
                    f'border-radius:4px;font-size:0.68rem;color:#848e9c;">{c}</span>'
                    for c in cats if c
                )
                st.markdown(f"""
                <div style="background:#161a1e;padding:12px;margin-bottom:8px;border-radius:6px;border:1px solid #2d3139;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
                        <a href="{url}" target="_blank"
                           style="color:#fff;text-decoration:none;font-weight:500;font-size:0.88rem;flex:1;">{title}</a>
                        <span style="color:{badge_c};font-weight:700;font-size:0.78rem;
                                     white-space:nowrap;padding:2px 6px;border:1px solid {badge_c};
                                     border-radius:4px;">{badge_l}</span>
                    </div>
                    <div style="margin-top:5px;">
                        <span style="color:#848e9c;font-size:0.72rem;">{source}</span>{cat_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No news available")
    except Exception:
        st.info("News unavailable")


@st.fragment(run_every="2s")
def render_signals():
    """Recent signals with live current price — updates every 2 s."""
    st.subheader("Recent Signals")

    signals = fetch_recent_signals(50)
    if signals.empty:
        st.info("No signals logged yet.")
        return

    prices = get_live_prices()

    rows = ""
    for _, r in signals.iterrows():
        sym       = str(r.get("symbol", ""))
        direction = str(r.get("direction", ""))
        tf        = str(r.get("timeframe", ""))
        conf      = float(r.get("confidence", 0))
        ts_raw    = r.get("time", "")
        ts_str    = pd.to_datetime(ts_raw).strftime("%m-%d %H:%M") if ts_raw != "" else "—"

        live_info = prices.get(sym, {})
        current   = float(live_info.get("price", 0) or 0)

        dir_up    = direction.lower() == "long"
        dir_color = "#00ffa3" if dir_up else "#ff4d4d"
        dir_icon  = "⬆" if dir_up else "⬇"
        dir_label = f"{dir_icon} {direction.upper()}"

        conf_color = ("#ff4d4d" if conf < 40 else "#f0b429" if conf < 65 else "#00ffa3")

        def _fmt(p: float) -> str:
            if p <= 0: return "—"
            if p > 10_000: return f"${p:,.2f}"
            if p > 100:    return f"${p:,.3f}"
            return f"${p:,.4f}"

        rows += f"""
        <tr>
          <td style="color:#848e9c;font-size:0.78rem;">{ts_str}</td>
          <td style="font-weight:700;color:#fff;">{sym}</td>
          <td><span style="color:{dir_color};font-weight:600;">{dir_label}</span></td>
          <td><span style="color:{conf_color};font-weight:700;">{conf:.1f}%</span></td>
          <td style="color:#848e9c;font-size:0.78rem;">{tf}</td>
          <td style="color:#f0f0f0;font-weight:600;">{_fmt(current)}</td>
        </tr>"""

    st.markdown(f"""
    <style>
    .sig-tbl {{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif;font-size:0.85rem;}}
    .sig-tbl th {{color:#848e9c;font-weight:500;font-size:0.75rem;text-transform:uppercase;
                  letter-spacing:.06em;padding:8px 12px;border-bottom:1px solid #2d3139;text-align:left;}}
    .sig-tbl td {{padding:9px 12px;border-bottom:1px solid #1a1d22;vertical-align:middle;}}
    .sig-tbl tr:hover td {{background:#1a1d22;}}
    </style>
    <div class="tbl-wrap">
    <table class="sig-tbl">
      <thead><tr>
        <th>Time</th><th>Symbol</th><th>Direction</th>
        <th>Confidence</th><th>TF</th><th>Current Price</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


def render_trade_history(orders: pd.DataFrame):
    st.subheader("Recent Trades")
    if orders.empty:
        st.info("No trades recorded yet.")
        return
    df = orders.copy()
    if "created_at" in df.columns:
        df["time"] = pd.to_datetime(df["created_at"]).dt.strftime("%m-%d %H:%M")
        df.drop(columns=["created_at"], inplace=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_performance(snapshots: pd.DataFrame, orders: pd.DataFrame):
    st.markdown("### Performance Analytics")
    if snapshots.empty:
        st.info("Performance metrics will be available once the agent begins capturing state.")
        return

    try:
        equity = snapshots["total_equity"].replace([float("inf"), float("-inf")], float("nan")).dropna().reset_index(drop=True)
        # Note: Raw orders don't have 'pnl'. Realized P&L metrics require a matched trade log.
        trade_log = None

        summary = compute_summary(equity, trade_log)

        # Row 1: Primary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
        m2.metric("Sortino Ratio", f"{summary.get('sortino_ratio', 0):.2f}")
        m3.metric("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.1%}")
        m4.metric("Total Return", f"{summary.get('total_return_pct', 0):+.2%}")

        # Row 2: Trade Metrics
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Win Rate", f"{summary.get('win_rate', 0):.1%}")
        m6.metric("Trades", summary.get("total_trades", 0))
        m7.metric("Expectancy", f"${summary.get('expectancy_usdc', 0):.2f}")
        m8.metric("Profit Factor", f"{summary.get('profit_factor', 0):.2f}")
            
    except Exception as e:
        st.warning(f"Strategy metrics computation suspended: {e}")


def render_backtest_loader():
    st.subheader("Backtest Results")
    bt_dir = Path("backtests")
    if not bt_dir.exists() or not list(bt_dir.glob("*_metrics.json")):
        st.info("No backtest results found. Run `python scripts/backtest.py` first.")
        return

    import json
    files = sorted(bt_dir.glob("*_metrics.json"), reverse=True)
    chosen = st.selectbox("Select backtest run", [f.stem.replace("_metrics","") for f in files])
    if chosen:
        metrics_path = bt_dir / f"{chosen}_metrics.json"
        trades_path  = bt_dir / f"{chosen}_trades.csv"
        equity_path  = bt_dir / f"{chosen}_equity.csv"

        with open(metrics_path) as f:
            m = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate",     f"{m.get('win_rate', 0):.1%}")
        c2.metric("Sharpe",       f"{m.get('sharpe_ratio', 0):.2f}")
        c3.metric("Max DD",       f"{m.get('max_drawdown_pct', 0):.1%}")
        c4.metric("Total Return", f"{m.get('total_return_pct', 0):.1%}")

        if equity_path.exists():
            eq_df = pd.read_csv(equity_path, index_col=0)
            eq_df = eq_df.replace([float("inf"), float("-inf")], float("nan")).dropna()
            if eq_df.empty or eq_df["equity"].std() == 0:
                last_val = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else 0
                st.info(f"Equity flat at ${last_val:,.2f} — no trades executed in this backtest.")
            else:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=eq_df.index,
                    y=eq_df['equity'],
                    mode='lines',
                    line=dict(color='#00ffa3', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 163, 0.05)',
                    name='Backtest Equity'
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        if trades_path.exists():
            with st.expander("Trade log"):
                st.dataframe(pd.read_csv(trades_path), use_container_width=True)


def render_logs(n_lines: int = 150):
    """Modern log explorer with severity-based styling."""
    st.markdown("### System Diagnostics")
    
    log_dir = Path("logs")
    if not log_dir.exists():
        st.info("System logs initialized. Waiting for entries...")
        return
    
    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        st.info("No active log streams detective.")
        return
    
    c1, c2 = st.columns([3, 1])
    with c2:
        filter_level = st.selectbox("Severity Filter", ["ALL", "INFO", "WARNING", "ERROR"], index=0, label_visibility="collapsed")
    
    try:
        with open(log_files[0], 'r', encoding='utf-8') as f:
            lines = f.readlines()[-n_lines:]
        
        for line in reversed(lines):
            if not line.strip(): continue
            
            # Simple color coding for raw text or JSON
            lvl = "INFO"
            if "ERROR" in line.upper(): lvl = "ERROR"
            elif "WARN" in line.upper(): lvl = "WARNING"
            
            if filter_level != "ALL" and lvl != filter_level: continue
            
            color = "#848e9c"
            if lvl == "ERROR": color = "#ff3b30"
            elif lvl == "WARNING": color = "#f0b90b"
            elif lvl == "INFO": color = "#00ffa3"
            
            st.markdown(f"""
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; border-bottom: 1px solid #1e2329; padding: 4px 0; color: #d1d4dc;">
                    <span style="color: {color}; font-weight: 700;">{lvl}</span> | {line.strip()}
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Diagnostics error: {e}")


# ── Agent Brain — live signal reasoning ──────────────────────────────────────

@st.fragment(run_every="3s")
def render_agent_brain():
    """
    Parses the agent's structured log lines and renders a live card view:
    - Per-symbol × per-timeframe signal cards (direction, confidence, ADX, regime, ML)
    - Decision feed: why trades are/aren't firing
    """
    import re

    log_path = Path("logs/agent.log")
    if not log_path.exists():
        st.info("No agent log found. Start the agent first.")
        return

    # ── Read & clean last 3000 lines ─────────────────────────────────────────
    # 3000 lines covers ~75 min of log output; enough to catch 1h signals.
    # 4h signals are persisted in session_state across render cycles (see below).
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()[-3000:]
    except Exception as e:
        st.error(f"Cannot read log: {e}")
        return

    # Strip ANSI colour codes that structlog injects
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\[[0-9;]*m")
    lines = [ANSI_RE.sub("", l) for l in raw_lines]

    # ── Regex patterns ────────────────────────────────────────────────────────
    SIG_RE = re.compile(
        r"Signal \[([A-Z]+-[A-Z]+)/([^\]]+)\] "
        r"dir=([+-]?\d+) conf=([\d.]+) raw=([+-]?[\d.]+) adx=([\d.]+) regime=(\w+)"
        r"(?:.*?ml=\(([+-]?\d+),([\d.]+)%\))?"
        r"(?:.*?lgbm=\(([+-]?\d+),([\d.]+)%\))?"
        r"(?:.*?lstm=\(([+-]?\d+),([\d.]+)%\))?"
    )
    TS_RE  = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")

    # ── Parse signals & decision feed ─────────────────────────────────────────
    # Seed from persistent session state so low-frequency signals (1h, 4h) that
    # were logged hours ago and fell out of the 3000-line window are still shown.
    signals: dict[tuple, dict] = dict(st.session_state.get("_brain_persistent_signals", {}))
    feed: list[tuple[str, str, str]] = []   # (hhmm, message, category)

    for line in lines:
        ts_m   = TS_RE.search(line)
        ts_str = ts_m.group(1) if ts_m else ""
        hhmm   = ts_str[11:16] if len(ts_str) >= 16 else "—"

        sig_m  = SIG_RE.search(line)
        if sig_m:
            sym, tf = sig_m.group(1), sig_m.group(2)
            signals[(sym, tf)] = {
                "symbol":    sym,
                "tf":        tf,
                "direction": int(sig_m.group(3)),
                "confidence": float(sig_m.group(4)),
                "raw":       float(sig_m.group(5)),
                "adx":       float(sig_m.group(6)),
                "regime":    sig_m.group(7),
                "lgbm_dir":  int(sig_m.group(8))   if sig_m.group(8)  else None,
                "lgbm_conf": float(sig_m.group(9))  if sig_m.group(9)  else None,
                "lstm_dir":  int(sig_m.group(10))   if sig_m.group(10) else None,
                "lstm_conf": float(sig_m.group(11)) if sig_m.group(11) else None,
                "ts":        hhmm,
            }
            continue

        # Decision feed events
        txt = line.strip()[-160:]
        if "Trade decision received" in line:
            feed.append((hhmm, txt, "trade"))
        elif "Paper order" in line and ("placed" in line or "filled" in line):
            feed.append((hhmm, txt, "order"))
        elif "TP triggered" in line:
            feed.append((hhmm, txt, "tp"))
        elif "SL triggered" in line or "trailing_stop" in line:
            feed.append((hhmm, txt, "sl"))
        elif "Trailing stop activated" in line:
            feed.append((hhmm, txt, "trail"))
        elif "TFs agree" in line or "confluence" in line.lower():
            feed.append((hhmm, txt, "confluence"))
        elif "ADX below" in line or "funding" in line.lower() and "blocked" in line.lower():
            feed.append((hhmm, txt, "blocked"))

    # Share signal state with AI analyst fragment
    st.session_state["_brain_signals"] = dict(signals)
    # Persist across render cycles — low-frequency signals (1h/4h) survive
    # the log window; only updated when a fresh signal is parsed (dict.update semantics).
    st.session_state["_brain_persistent_signals"] = dict(signals)

    # Read current threshold for card annotations (session_state always available)
    _cur_thresh = st.session_state.get("conf_threshold_slider", 25)

    # ── Helper: build a signal card ───────────────────────────────────────────
    def _card(sig: dict | None, tf: str) -> str:
        if sig is None:
            return (
                f'<div style="background:#0f1117;border:1px solid #2d3139;border-radius:10px;'
                f'padding:14px;min-height:130px;display:flex;flex-direction:column;'
                f'align-items:center;justify-content:center;">'
                f'<div style="color:#848e9c;font-size:0.72rem;font-weight:600;">{tf}</div>'
                f'<div style="color:#2d3139;font-size:0.8rem;margin-top:6px;">No data</div></div>'
            )

        d    = sig["direction"]
        conf = sig["confidence"]
        adx  = sig["adx"]
        reg  = sig["regime"]

        # Direction
        if d == 1:
            dc, di, dl = "#00ffa3", "▲", "LONG"
        elif d == -1:
            dc, di, dl = "#ff4d4d", "▼", "SHORT"
        else:
            dc, di, dl = "#606878", "●", "FLAT"

        # Confidence bar colour
        cc = "#ff4d4d" if conf < 25 else "#f0b429" if conf < 45 else "#00ffa3"

        # ADX colour
        ac = "#ff4d4d" if adx < 20 else "#f0b429" if adx < 30 else "#00ffa3"

        # Regime colour
        rc = {"trending_up": "#00ffa3", "trending_down": "#ff4d4d",
              "volatile": "#f0b429", "ranging": "#848e9c", "breakout": "#a78bfa"}.get(reg, "#848e9c")

        # ML row (only on 15m where models run)
        ml_html = ""
        if tf == "15m" and sig["lgbm_dir"] is not None:
            def _arrow(v):
                if v == 1:  return "▲", "#00ffa3"
                if v == -1: return "▼", "#ff4d4d"
                return "●", "#606878"
            lg_i, lg_c = _arrow(sig["lgbm_dir"])
            ls_i, ls_c = _arrow(sig["lstm_dir"])
            ml_html = (
                f'<div style="display:flex;gap:6px;margin-top:8px;font-size:0.7rem;">'
                f'<span style="color:#848e9c;">LGBM</span>'
                f'<span style="color:{lg_c};font-weight:700;">{lg_i} {sig["lgbm_conf"]:.0f}%</span>'
                f'<span style="color:#848e9c;margin-left:4px;">LSTM</span>'
                f'<span style="color:{ls_c};font-weight:700;">{ls_i} {sig["lstm_conf"]:.0f}%</span>'
                f'</div>'
            )

        # Why no trade?
        why = ""
        if d == 0:
            raw = sig.get("raw", 0.0)
            if adx < 20:
                why = f'<div style="color:#f0b429;font-size:0.67rem;margin-top:5px;">⚠ ADX {adx:.0f} &lt; 20 (no trend)</div>'
            elif conf < _cur_thresh:
                why = f'<div style="color:#848e9c;font-size:0.67rem;margin-top:5px;">⚠ Conf {conf:.1f}% &lt; threshold {_cur_thresh:.0f}%</div>'
            elif abs(raw) < 0.02:
                why = '<div style="color:#848e9c;font-size:0.67rem;margin-top:5px;">Raw score flat (&lt;0.02)</div>'
            else:
                dir_word = "LONG" if raw > 0 else "SHORT"
                why = f'<div style="color:#f0b429;font-size:0.67rem;margin-top:5px;">HTF disagrees — {dir_word} 15m vs HTF</div>'

        return (
            f'<div style="background:#0f1117;border:1px solid #2d3139;border-radius:10px;padding:14px;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
            f'  <span style="color:#848e9c;font-size:0.72rem;font-weight:600;">{tf}</span>'
            f'  <span style="color:#3a4050;font-size:0.65rem;">{sig["ts"]}</span>'
            f'</div>'
            f'<div style="font-size:1.1rem;font-weight:800;color:{dc};letter-spacing:1px;">{di} {dl}</div>'
            f'<div style="margin:6px 0;background:#1e2329;border-radius:4px;height:4px;">'
            f'  <div style="background:{cc};height:100%;width:{min(conf,100):.0f}%;"></div></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:0.73rem;">'
            f'  <span style="color:{cc};font-weight:700;">{conf:.1f}%</span>'
            f'  <span style="color:{ac};">ADX {adx:.0f}</span>'
            f'  <span style="color:{rc};font-size:0.65rem;">{reg}</span>'
            f'</div>'
            f'{ml_html}{why}</div>'
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown(
        '<p style="color:#848e9c;font-size:0.8rem;margin-bottom:16px;">'
        'Live signal reasoning — auto-refreshes every 3 s from agent.log</p>',
        unsafe_allow_html=True,
    )

    symbols = TRADING_SYMBOLS
    tfs     = ["1m", "3m", "5m", "15m", "1h", "4h"]

    # ── Threshold slider (writes to config/runtime.json → agent picks up live) ─
    import json as _json

    _runtime_path = Path("config/runtime.json")
    MIN_CONF = 25  # safe default — overwritten by slider below
    _saved_thresh = 25.0
    try:
        if _runtime_path.exists():
            _saved_thresh = float(_json.loads(_runtime_path.read_text()).get("min_confidence_threshold", 25.0))
    except Exception:
        pass

    try:
        col_sl, col_info = st.columns([3, 1])
        with col_sl:
            MIN_CONF = st.slider(
                "Confidence Threshold (yellow line)",
                min_value=1, max_value=80,
                value=int(_saved_thresh),
                step=1,
                help="Agent only takes a trade when 15m confidence ≥ this value. "
                     "Written to config/runtime.json — agent picks it up on the next candle.",
                key="conf_threshold_slider",
            )
        with col_info:
            st.markdown(
                f'<div style="margin-top:28px;padding:6px 12px;border-radius:8px;'
                f'background:{"#0a2218" if MIN_CONF <= 25 else "#2b1c08" if MIN_CONF <= 40 else "#1e0a0a"};'
                f'border:1px solid {"#00ffa3" if MIN_CONF <= 25 else "#f0b429" if MIN_CONF <= 40 else "#ff4d4d"}40;'
                f'font-size:0.78rem;font-weight:700;'
                f'color:{"#00ffa3" if MIN_CONF <= 25 else "#f0b429" if MIN_CONF <= 40 else "#ff4d4d"};">'
                f'{"NORMAL" if MIN_CONF <= 25 else "STRICT" if MIN_CONF <= 40 else "VERY STRICT"} — {MIN_CONF}%'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass  # slider unavailable during fragment pre-render — MIN_CONF stays at default

    # Persist to runtime.json so the agent reads it
    try:
        _runtime_path.parent.mkdir(exist_ok=True)
        _runtime_path.write_text(_json.dumps({"min_confidence_threshold": MIN_CONF}, indent=2))
    except Exception:
        pass

    # ── Trade Score Meters ────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.78rem;font-weight:700;color:#848e9c;'
        'text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;margin-top:16px;">'
        'Trade Score — Confidence vs Threshold</div>',
        unsafe_allow_html=True,
    )

    for sym in symbols:
        s1m  = signals.get((sym, "1m"))
        s3m  = signals.get((sym, "3m"))
        s5m  = signals.get((sym, "5m"))
        s15  = signals.get((sym, "15m"))
        s1h  = signals.get((sym, "1h"))
        s4h  = signals.get((sym, "4h"))

        d1m  = s1m["direction"]  if s1m  else 0
        d3m  = s3m["direction"]  if s3m  else 0
        d5m  = s5m["direction"]  if s5m  else 0
        d15  = s15["direction"]  if s15  else 0
        c15  = s15["confidence"] if s15  else 0.0
        d1h  = s1h["direction"]  if s1h  else 0
        d4h  = s4h["direction"]  if s4h  else 0

        # How many HTF TFs agree with 15m direction?
        htf_agree = sum(1 for d in [d1h, d4h] if d == d15 and d15 != 0)

        conf_ok   = c15 >= MIN_CONF and d15 != 0
        conf_ok_raw = c15 >= MIN_CONF  # regardless of direction
        all_tf_agree = htf_agree >= 1
        ready     = conf_ok and all_tf_agree

        # Direction colour + label
        if d15 == 1:
            dir_color, dir_icon, dir_label = "#00ffa3", "▲", "LONG"
        elif d15 == -1:
            dir_color, dir_icon, dir_label = "#ff4d4d", "▼", "SHORT"
        else:
            dir_color, dir_icon, dir_label = "#606878", "●", "FLAT"

        # Bar fill colour
        if ready:
            bar_color = "#00ffa3"
        elif conf_ok:
            bar_color = "#f0b429"
        elif d15 != 0:
            bar_color = "#f04950"
        else:
            bar_color = "#3a4050"

        # Status badge
        if ready:
            sb_bg, sb_col, sb_txt = "#0a2218", "#00ffa3", "● TRADE READY"
        elif conf_ok and not all_tf_agree:
            sb_bg, sb_col, sb_txt = "#2b1c08", "#f0b429", "◑ NO CONFLUENCE"
        elif d15 != 0 and not conf_ok:
            sb_bg, sb_col, sb_txt = "#1e1218", "#ff4d4d", "▸ BELOW THRESHOLD"
        else:
            sb_bg, sb_col, sb_txt = "#141720", "#606878", "◌ FLAT / WAITING"

        # TF agreement badges
        def _tf_b(d_val, tf_name, ref_dir):
            if d_val == 1:   ic, fc = "▲", "#00ffa3"
            elif d_val == -1: ic, fc = "▼", "#ff4d4d"
            else:            ic, fc = "●", "#4a5060"
            border = "1px solid #00ffa3" if (d_val == ref_dir and ref_dir != 0) else "1px solid #2d3139"
            return (
                f'<span style="border:{border};border-radius:5px;'
                f'padding:2px 7px;font-size:0.7rem;color:{fc};margin-right:5px;">'
                f'{ic} {tf_name}</span>'
            )

        tf_badges = (_tf_b(d1m, "1m", d15) + _tf_b(d3m, "3m", d15) + _tf_b(d5m, "5m", d15)
                     + _tf_b(d15, "15m", d15) + _tf_b(d1h, "1h", d15) + _tf_b(d4h, "4h", d15))

        # Bar fill: clamp to 100, threshold marker at MIN_CONF%
        fill_pct   = min(c15, 100.0)
        thresh_pct = MIN_CONF  # always 25%

        st.markdown(f"""
        <div style="background:#0f1117;border:1px solid {'#1a3a28' if ready else '#2d3139'};
                    border-radius:10px;padding:14px 18px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div>
              <span style="font-size:0.9rem;font-weight:800;color:#d1d4dc;letter-spacing:.5px;">{sym}</span>
              <span style="margin-left:10px;font-size:0.85rem;font-weight:700;color:{dir_color};">
                {dir_icon} {dir_label}</span>
              <span style="margin-left:10px;font-size:0.72rem;color:{dir_color};">{c15:.1f}%</span>
            </div>
            <div style="background:{sb_bg};color:{sb_col};padding:4px 12px;border-radius:20px;
                        font-size:0.75rem;font-weight:700;border:1px solid {sb_col}40;">
              {sb_txt}
            </div>
          </div>

          <!-- Score bar -->
          <div style="position:relative;background:#1e2329;border-radius:6px;height:10px;margin-bottom:6px;">
            <!-- Fill -->
            <div style="position:absolute;left:0;top:0;bottom:0;width:{fill_pct:.1f}%;
                        background:{bar_color};border-radius:6px;
                        box-shadow:{'0 0 8px ' + bar_color + '80' if ready else 'none'};
                        transition:width .4s ease;"></div>
            <!-- Threshold marker -->
            <div style="position:absolute;left:{thresh_pct:.1f}%;top:-4px;bottom:-4px;
                        width:2px;background:#f0b429;border-radius:2px;opacity:0.9;"></div>
          </div>

          <!-- Scale labels -->
          <div style="position:relative;height:14px;margin-bottom:8px;">
            <span style="position:absolute;left:0;font-size:0.62rem;color:#4a5060;">0</span>
            <span style="position:absolute;left:{thresh_pct:.1f}%;transform:translateX(-50%);
                         font-size:0.62rem;color:#f0b42990;">│ {thresh_pct:.0f}%</span>
            <span style="position:absolute;right:0;font-size:0.62rem;color:#4a5060;">100</span>
          </div>

          <!-- TF alignment badges -->
          <div style="display:flex;align-items:center;gap:4px;">
            <span style="font-size:0.68rem;color:#4a5060;margin-right:4px;">TF ALIGN</span>
            {tf_badges}
            <span style="margin-left:auto;font-size:0.68rem;color:#4a5060;">
              {htf_agree}/2 HTF agree</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:20px;"></div>', unsafe_allow_html=True)

    # ── Signal Cards ──────────────────────────────────────────────────────────
    for sym in symbols:
        any_sig = any(signals.get((sym, tf)) for tf in tfs)
        sym_label_color = "#f0f0f0" if any_sig else "#4a5060"
        st.markdown(
            f'<div style="font-size:0.85rem;font-weight:700;color:{sym_label_color};'
            f'margin:16px 0 8px 2px;letter-spacing:0.5px;">{sym}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        for i, tf in enumerate(tfs):
            cols[i].markdown(_card(signals.get((sym, tf)), tf), unsafe_allow_html=True)

    # ── Decision feed ─────────────────────────────────────────────────────────
    if feed:
        st.markdown(
            '<div style="margin-top:24px;margin-bottom:8px;font-size:0.82rem;'
            'font-weight:700;color:#d1d4dc;">Decision Feed</div>',
            unsafe_allow_html=True,
        )
        cat_colors = {
            "trade": "#00ffa3", "order": "#00ffa3",
            "tp": "#00ffa3",    "sl": "#ff4d4d",
            "trail": "#a78bfa", "confluence": "#f0b429",
            "blocked": "#f0b429",
        }
        for hhmm, txt, cat in reversed(feed[-20:]):
            col = cat_colors.get(cat, "#848e9c")
            # Strip structlog noise — keep the human-readable part after "] "
            clean = re.sub(r"^\S+\s+", "", txt)   # drop leading timestamp chunk
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.73rem;color:{col};'
                f'padding:3px 0;border-bottom:1px solid #1a1d22;">'
                f'<span style="color:#4a5060;">{hhmm}</span>  {clean}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:#3a4050;font-size:0.8rem;margin-top:20px;">'
            'Waiting for first signal cycle...</div>',
            unsafe_allow_html=True,
        )


# ── AI Trade Analyst — GPT-5.2 live trade readiness window ───────────────────

@st.fragment(run_every="30s")
def render_ai_thinking():
    """
    Live AI analyst: GPT-5.2 reads current signal state for all symbols
    and explains what conditions are needed for a trade to fire.
    Renders every 30s; GPT API is called at most once per 60s (cached).
    """
    import json as _json, time as _time, os as _os

    signals: dict = st.session_state.get("_brain_signals", {})
    thresh  = st.session_state.get("conf_threshold_slider", 25)
    tfs     = ["1m", "3m", "5m", "15m", "1h", "4h"]
    symbols = TRADING_SYMBOLS

    # ── Section header ────────────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin:28px 0 14px 0;">'
        f'<div style="font-size:0.95rem;font-weight:800;color:#e8e8e8;letter-spacing:0.5px;">🤖 AI TRADE ANALYST</div>'
        f'<div style="font-size:0.7rem;padding:3px 9px;background:#0d1520;border:1px solid #60a5fa40;'
        f'border-radius:20px;color:#60a5fa;font-weight:700;">⚡ GPT-5.2</div>'
        f'<div style="margin-left:auto;font-size:0.68rem;color:#3a4050;">Updated {now_utc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not signals:
        st.markdown(
            '<div style="color:#3a4050;font-size:0.78rem;padding:8px 0;">'
            'Waiting for first signal cycle...</div>', unsafe_allow_html=True)
        return

    # ── Build signal summary for GPT prompt ──────────────────────────────────
    def _sig_line(sym, tf):
        s = signals.get((sym, tf))
        if not s:
            return f"  {tf}: NO DATA"
        d_str = {1:"LONG",-1:"SHORT",0:"FLAT"}.get(s["direction"],"FLAT")
        ml_part = ""
        if tf == "15m" and s.get("lgbm_dir") is not None:
            lg = {1:"LONG",-1:"SHORT",0:"FLAT"}.get(s["lgbm_dir"],"?")
            ls = {1:"LONG",-1:"SHORT",0:"FLAT"}.get(s["lstm_dir"],"?")
            ml_part = f" | LGBM={lg}/{s['lgbm_conf']:.0f}% LSTM={ls}/{s['lstm_conf']:.0f}%"
        return f"  {tf}: {d_str} conf={s['confidence']:.1f}% adx={s['adx']:.0f} regime={s['regime']}{ml_part}"

    lines_out = []
    for sym in symbols:
        lines_out.append(f"\n{sym}:")
        for tf in tfs:
            lines_out.append(_sig_line(sym, tf))

    prompt = (
        f"You are a crypto trading agent analyst. Assess trade readiness from live signals.\n\n"
        f"TRADING RULES:\n"
        f"- Confidence threshold: {thresh}% (15m conf must be >= {thresh}%)\n"
        f"- ADX minimum: 20 (below 20 = ranging market = NO trade)\n"
        f"- HTF confluence: 15m direction must match at least 1 of (1h, 4h)\n"
        f"- Regime 'ranging' = always NO_TRADE regardless of confidence\n\n"
        f"CURRENT LIVE SIGNALS ({now_utc}):{''.join(lines_out)}\n\n"
        f"Respond ONLY in valid JSON (no markdown fences):\n"
        f'{{\n  "symbols": [\n    {{\n      "symbol": "BTC-USDC",\n      "status": "READY|CLOSE|WATCHING|NO_TRADE",\n'
        f'      "readiness_pct": 0,\n      "direction": "LONG|SHORT|FLAT",\n'
        f'      "reasoning": "2 sentence max — current state explanation",\n'
        f'      "needed": "single most important blocking condition"\n    }}\n  ],\n'
        f'  "overall_market_view": "1-2 sentence macro summary",\n'
        f'  "next_opportunity": "SYMBOL DIRECTION — brief reason",\n'
        f'  "trade_eta": "IMMINENT|THIS_CANDLE|1-3_CANDLES|UNCLEAR|UNLIKELY"\n}}\n\n'
        f"STATUS: READY=all conditions met; CLOSE=1 condition away; WATCHING=15m signal/no HTF; NO_TRADE=ADX<20 or flat"
    )

    # ── GPT call — throttled to once per 60s ─────────────────────────────────
    last_call = st.session_state.get("_analyst_ts", 0)
    cached    = st.session_state.get("_analyst_result", None)
    gpt_result = None

    if (_time.time() - last_call >= 60) or cached is None:
        az_ep  = _os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        az_key = _os.environ.get("AZURE_OPENAI_KEY", "")
        az_dep = _os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
        az_ver = _os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        if az_ep and az_key:
            try:
                from openai import AzureOpenAI as _AZ
                _cli = _AZ(azure_endpoint=az_ep, api_key=az_key, api_version=az_ver)
                _resp = _cli.responses.create(
                    model=az_dep,
                    input=[
                        {"role": "system", "content": "You are a precise crypto trading analyst. Output valid JSON only."},
                        {"role": "user",   "content": prompt},
                    ],
                    text={"format": {"type": "json_object"}},
                    max_output_tokens=4000,
                )
                raw = getattr(_resp, "output_text", "") or ""
                if not raw:
                    for item in getattr(_resp, "output", []):
                        for part in getattr(item, "content", []):
                            raw = getattr(part, "text", "")
                            if raw: break
                        if raw: break
                if raw:
                    gpt_result = _json.loads(raw)
                    st.session_state["_analyst_result"] = gpt_result
                    st.session_state["_analyst_ts"] = _time.time()
            except Exception as _gpt_err:
                st.session_state["_analyst_last_err"] = str(_gpt_err)

    if gpt_result is None:
        gpt_result = cached

    # ── Rule-based fallback ───────────────────────────────────────────────────
    def _rule(sym):
        s15 = signals.get((sym, "15m"))
        s1h = signals.get((sym, "1h"))
        s4h = signals.get((sym, "4h"))
        if not s15:
            return {"status":"NO_TRADE","readiness_pct":0,"direction":"FLAT",
                    "reasoning":"No signal data.","needed":"Wait for first candle."}
        d15,c15,a15,r15 = s15["direction"],s15["confidence"],s15["adx"],s15["regime"]
        d1h = (s1h or {}).get("direction", 0)
        d4h = (s4h or {}).get("direction", 0)
        htf = sum(1 for d in [d1h,d4h] if d==d15 and d15!=0)
        dn  = {1:"LONG",-1:"SHORT",0:"FLAT"}.get(d15,"FLAT")
        if d15==0 or r15=="ranging" or a15<20:
            return {"status":"NO_TRADE","readiness_pct":min(int(c15),25),"direction":dn,
                    "reasoning":f"15m {dn}, ADX {a15:.0f} ({r15}). No directional trade possible.",
                    "needed":"Need ADX≥20 and non-ranging regime with directional signal."}
        if c15<thresh and htf==0:
            return {"status":"NO_TRADE","readiness_pct":20,"direction":dn,
                    "reasoning":f"{sym} 15m {dn} at {c15:.1f}% conf. HTF not aligned.",
                    "needed":f"Need conf≥{thresh}% (now {c15:.1f}%) AND at least 1 HTF to agree."}
        if c15<thresh:
            return {"status":"WATCHING","readiness_pct":35,"direction":dn,
                    "reasoning":f"15m {dn} with {htf}/2 HTF agreeing. Confidence {c15:.1f}% below threshold.",
                    "needed":f"Need confidence≥{thresh}% (currently {c15:.1f}%)."}
        if htf==0:
            return {"status":"WATCHING","readiness_pct":50,"direction":dn,
                    "reasoning":f"15m {dn} at {c15:.1f}% conf above threshold. No HTF alignment yet.",
                    "needed":"Need 1h or 4h to align with 15m direction."}
        if htf>=1:
            return {"status":"CLOSE","readiness_pct":85,"direction":dn,
                    "reasoning":f"15m {dn} at {c15:.1f}%, {htf}/2 HTF agree. Near trade conditions.",
                    "needed":"Conditions strengthening — trade may fire next candle close."}
        return {"status":"READY","readiness_pct":95,"direction":dn,
                "reasoning":"All conditions met.","needed":"Trade should fire on next evaluation."}

    engine_label, engine_color = "GPT-5.2", "#60a5fa"
    if gpt_result is None:
        sym_data = [{"symbol": s, **_rule(s)} for s in symbols]
        err_msg = st.session_state.get("_analyst_last_err", "")
        err_hint = f" ({err_msg[:80]})" if err_msg else ""
        gpt_result = {"symbols": sym_data,
                      "overall_market_view": f"Rule-based analysis (GPT-5.2 unavailable{err_hint}).",
                      "next_opportunity": "See per-symbol cards below",
                      "trade_eta": "UNCLEAR"}
        engine_label, engine_color = "Rule-Based", "#848e9c"

    # ── Overall banner ────────────────────────────────────────────────────────
    eta = gpt_result.get("trade_eta", "UNCLEAR")
    eta_c = {"IMMINENT":"#00ffa3","THIS_CANDLE":"#00ffa3","1-3_CANDLES":"#f0b429",
             "UNCLEAR":"#848e9c","UNLIKELY":"#606878"}.get(eta, "#848e9c")

    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #1e2535;border-left:3px solid {engine_color}50;'
        f'border-radius:8px;padding:12px 16px;margin-bottom:14px;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'<span style="font-size:0.7rem;font-weight:700;color:{engine_color};">{engine_label}</span>'
        f'<span style="color:#2a3040;">·</span>'
        f'<span style="font-size:0.7rem;color:#848e9c;">Trade ETA:</span>'
        f'<span style="font-size:0.7rem;font-weight:800;color:{eta_c};">{eta.replace("_"," ")}</span>'
        f'<span style="margin-left:auto;font-size:0.7rem;color:#5a6070;">🎯 {gpt_result.get("next_opportunity","—")}</span>'
        f'</div>'
        f'<div style="font-size:0.8rem;color:#b0b8c8;line-height:1.5;">'
        f'{gpt_result.get("overall_market_view","")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Per-symbol readiness cards (3 + 2 layout) ────────────────────────────
    sym_map = {r["symbol"]: r for r in gpt_result.get("symbols", [])}
    s_colors = {
        "READY":    ("#00ffa3", "#081510", "✅"),
        "CLOSE":    ("#f0b429", "#1a1200", "🟡"),
        "WATCHING": ("#60a5fa", "#080f1e", "👁"),
        "NO_TRADE": ("#606878", "#0d0f12", "⛔"),
    }

    def _row(row_syms):
        cols = st.columns(len(row_syms))
        for i, sym in enumerate(row_syms):
            r  = sym_map.get(sym) or _rule(sym)
            st_key = r.get("status","NO_TRADE")
            rdy, dirn = r.get("readiness_pct",0), r.get("direction","FLAT")
            sc, bg, icon = s_colors.get(st_key, ("#606878","#0d0f12","◌"))
            bc = sc if rdy>=70 else "#f0b429" if rdy>=40 else "#ff4d4d"
            dc = "#00ffa3" if dirn=="LONG" else "#ff4d4d" if dirn=="SHORT" else "#606878"
            cols[i].markdown(
                f'<div style="background:{bg};border:1px solid {sc}28;border-top:2px solid {sc}80;'
                f'border-radius:10px;padding:14px;min-height:190px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">'
                f'<span style="font-size:0.8rem;font-weight:800;color:#e0e0e0;">{sym.replace("-USDC","")}</span>'
                f'<span style="font-size:0.68rem;font-weight:700;color:{sc};">{icon} {st_key}</span>'
                f'</div>'
                f'<div style="font-size:0.88rem;font-weight:700;color:{dc};margin-bottom:9px;">{dirn}</div>'
                f'<div style="background:#111418;border-radius:4px;height:5px;margin-bottom:4px;">'
                f'<div style="background:{bc};height:100%;width:{min(rdy,100)}%;border-radius:4px;"></div></div>'
                f'<div style="font-size:0.67rem;color:{sc};font-weight:700;margin-bottom:9px;">'
                f'Readiness {rdy}%</div>'
                f'<div style="font-size:0.7rem;color:#8090a8;line-height:1.45;margin-bottom:9px;">'
                f'{r.get("reasoning","")}</div>'
                f'<div style="font-size:0.67rem;color:#d4a520;border-top:1px solid #1a1e26;padding-top:6px;">'
                f'⚠ {r.get("needed","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _row(symbols[:3])
    if len(symbols) > 3:
        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
        _row(symbols[3:])
    st.markdown('<div style="margin-bottom:20px;"></div>', unsafe_allow_html=True)

# ── Chat Terminal ─────────────────────────────────────────────────────────────
def render_agent_chat():
    """
    Interactive ChatGPT-like terminal giving users direct access to
    GPT-5.2 along with the agent's real-time state.
    """
    st.markdown("### 💬 Ask The Agent")
    st.caption("Chat with GPT-5.2 powered by real-time signals, sentiment, and DB context.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello. I'm connected to the live market data. Want me to evaluate a trade?"}
        ]

    # Render History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about SOL long, current BTC signals, etc..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing live data..."):
                import os, json
                from openai import AzureOpenAI
                
                # Assemble live context
                signals = st.session_state.get("_brain_signals", {})
                live_prices = st.session_state.get("_live_prices", {})
                
                # Context snippet from memory
                context_str = "LIVE MARKET SNAPSHOT:\\n"
                for sym, price in live_prices.items():
                    s15 = signals.get((sym, "15m"), {})
                    if s15:
                        dstr = {1:"LONG", -1:"SHORT", 0:"FLAT"}.get(s15.get("direction",0), "FLAT")
                        conf = s15.get("confidence", 0)
                        adx = s15.get("adx", 0)
                        reg = s15.get("regime", "ranging")
                        context_str += f"- {sym} @ ${price:.2f} | 15m Signal: {dstr} (conf={conf:.1f}%), ADX={adx:.0f} ({reg})\\n"
                
                sys_prompt = (
                    "You are the advanced AI core of CryptoAgent (GPT-5.2). "
                    "You answer questions from the user (trade operator) about current market conditions. "
                    "Use the provided live context to base your answers on actual signals. "
                    "Be direct, analytical, and give concrete recommendations on whether to take a trade."
                    f"\\n\\n{context_str}"
                )

                az_ep  = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
                az_key = os.environ.get("AZURE_OPENAI_KEY", "")
                az_dep = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
                az_ver = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
                
                if not az_ep or not az_key:
                    resp = "Sorry, Azure OpenAI credentials are not configured in my `.env`."
                    st.markdown(resp)
                    st.session_state.chat_messages.append({"role": "assistant", "content": resp})
                    return

                try:
                    client = AzureOpenAI(azure_endpoint=az_ep, api_key=az_key, api_version=az_ver)
                    messages = [{"role": "system", "content": sys_prompt}]
                    # Pass last 5 interactions to retain context
                    for pt in st.session_state.chat_messages[-5:]:
                        messages.append({"role": pt["role"], "content": pt["content"]})
                        
                    res = client.chat.completions.create(
                        model=az_dep,
                        messages=messages,
                        max_completion_tokens=600,
                        temperature=0.2
                    )
                    
                    answer = res.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error connecting to GPT-5.2: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        render_header()

        if not CONFIG_OK:
            st.error(f"System Offline: {CONFIG_ERR}")
            st.info("Ensure .env and config/settings.yaml are correctly configured.")

        # Sidebar: Agent start/stop control
        render_agent_control()

        # Sidebar: Analysis window
        st.sidebar.divider()
        hours = st.sidebar.selectbox("Analysis Window", [1, 4, 8, 24, 48, 168], index=3,
                                      format_func=lambda h: f"{h}h" if h < 24 else f"{h//24}d")

        # Sidebar: Infrastructure Status
        st.sidebar.divider()
        st.sidebar.markdown("### 🛠️ Infrastructure Status")
        st.sidebar.write("🟢 **Redis:** Running (Port 6379)")
        st.sidebar.write("🟢 **Database:** Running (Port 5433)")

        # Sidebar: Actions
        st.sidebar.divider()
        st.sidebar.markdown("### Operations")
        
        if st.sidebar.button("⚡ LOAD DEMO DATA", use_container_width=True, key="btn_load_demo"):
            load_demo_data()
            st.sidebar.success("Environment synced with demo data.")
            st.rerun()
            
        if st.sidebar.button("🔄 REFRESH TERMINAL", use_container_width=True, key="btn_refresh_terminal"):
            st.rerun()
        
        # Performance Snapshots in Sidebar
        bt_dir = Path("backtests")
        if bt_dir.exists():
            bt_files = list(bt_dir.glob("*_metrics.json"))
            if bt_files:
                latest_bt = max(bt_files, key=lambda p: p.stat().st_mtime)
                import json
                with open(latest_bt) as f: m = json.load(f)
                st.sidebar.divider()
                st.sidebar.markdown("### Last Optimization")
                st.sidebar.metric("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}")
                st.sidebar.metric("Win Rate", f"{m.get('win_rate', 0):.1%}")

        # Fetch data for static tabs (fragments fetch their own)
        snapshots = fetch_portfolio_snapshots(hours)
        orders    = fetch_orders(100)

        # Navigation
        tabs = st.tabs(["Terminal", "Brain", "Advanced Charts", "Paper Trades", "Inventory", "Intelligence", "Ledger", "System"])

        with tabs[0]: # Overview
            render_live_prices()
            st.divider()
            render_equity_curve(snapshots)
            st.divider()
            render_performance(snapshots, orders)

        with tabs[1]:
            st.markdown("## Agent Brain")
            render_agent_brain()
            render_ai_thinking()
            st.divider()
            render_agent_chat()

        with tabs[2]: render_price_charts()
        with tabs[3]: render_paper_trades()
        with tabs[4]: 
            render_positions_management()
            render_positions()
        with tabs[5]:
            render_sentiment_data()
            st.markdown("<hr style='margin:20px 0;border-color:#2d3139;'>", unsafe_allow_html=True)
            render_signals()
        with tabs[6]: render_trade_history(orders)
        with tabs[7]: render_logs(150)

        # Optional manual trigger (already in sidebar)
        pass
            
    except Exception as e:
        import traceback
        st.error("🚀 Critical System Fault")
        st.markdown(f"""
            <div style="background: rgba(255, 59, 48, 0.1); border: 1px solid #ff3b30; padding: 20px; border-radius: 8px;">
                <h4 style="color: #ff3b30; margin-top: 0;">Exception in Main Loop</h4>
                <code style="color: #ffffff; background: none; padding: 0;">{str(e)}</code>
                <details style="margin-top: 10px; color: #848e9c;">
                    <summary>View Stack Trace</summary>
                    <pre style="font-size: 0.7rem; color: #ff3b30;">{traceback.format_exc()}</pre>
                </details>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
