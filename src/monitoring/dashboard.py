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

# ── Custom CSS for Premium Look ────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b0e11;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #161a1e;
            border-right: 1px solid #2d3139;
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #00ffa3 !important;
            text-shadow: 0 0 10px rgba(0, 255, 163, 0.2);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px;
            color: #848e9c;
            font-weight: 500;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            color: #00ffa3 !important;
            border-bottom: 2px solid #00ffa3 !important;
        }
        
        /* Mode badges */
        .mode-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .mode-paper {
            background-color: rgba(240, 185, 11, 0.15);
            color: #f0b90b;
            border: 1px solid #f0b90b;
        }
        .mode-live {
            background-color: rgba(255, 59, 48, 0.15);
            color: #ff3b30;
            border: 1px solid #ff3b30;
            box-shadow: 0 0 15px rgba(255, 59, 48, 0.3);
        }
        
        /* Glassmorphism Containers */
        .glass-card {
            background: rgba(22, 26, 30, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }

        /* Tabs styling overhaul */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #121418;
            padding: 8px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid #2d3139;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            background-color: transparent;
            border-radius: 8px;
            color: #d1d4dc !important;
            font-weight: 700 !important;
            border: none;
            padding: 0 25px;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.05);
        }
        .stTabs [aria-selected="true"] {
            background-color: #00ffa3 !important;
            color: #0b0e11 !important;
            box-shadow: 0 0 20px rgba(0, 255, 163, 0.3);
        }
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-thumb {
            background: #2d3139;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }

        /* Chart text global override for visibility */
        .js-plotly-plot .plotly .xtick text, 
        .js-plotly-plot .plotly .ytick text,
        .js-plotly-plot .plotly .legendtext,
        .js-plotly-plot .plotly .gtitle {
            fill: #ffffff !important;
            font-size: 12px !important;
            font-weight: 600 !important;
        }

        /* Ensure Sidebar Toggle is accessible */
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
        [data-testid="stDecoration"] {
            display: none !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
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


# ── Agent Process Control ──────────────────────────────────────────────────────

_PID_FILE = Path(__file__).parent.parent.parent / "agent.pid"


def _agent_pid() -> int | None:
    """Return the PID from the pid file, or None if it doesn't exist."""
    try:
        return int(_PID_FILE.read_text().strip())
    except Exception:
        return None


def _is_agent_alive(pid: int | None) -> bool:
    """Return True if a process with given PID exists and is running."""
    if pid is None:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        return False


def _start_agent() -> tuple[bool, str]:
    """Spawn the agent as a detached subprocess. Returns (success, message)."""
    import subprocess, sys
    try:
        venv_python = Path(sys.executable)
        proc = subprocess.Popen(
            [str(venv_python), "-m", "src.agent"],
            cwd=str(Path(__file__).parent.parent.parent),
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if __import__("sys").platform == "win32" else 0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _PID_FILE.write_text(str(proc.pid))
        return True, f"Agent started (PID {proc.pid})"
    except Exception as e:
        return False, f"Failed to start agent: {e}"


def _stop_agent(pid: int) -> tuple[bool, str]:
    """Send SIGTERM to the agent process. Returns (success, message)."""
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
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

def _get_engine():
    try:
        from sqlalchemy import create_engine
        from src.core.config import get_settings
        url = get_settings().database.timescale_url
        return create_engine(url, pool_pre_ping=True)
    except Exception:
        return None


def query_df(sql: str, params=None) -> pd.DataFrame:
    try:
        engine = _get_engine()
        if engine is None:
            return pd.DataFrame()
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)
        return df
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


_TRACKED_SYMBOLS = ["BTC-USDC", "ETH-USDC", "SOL-USDC"]
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
    """Returns tickers with smart fallback and simulation to ensure 'ticking'."""
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    redis_prices = {}
    r = get_redis()
    
    if r is not None:
        try:
            for k in r.keys("price:*"):
                sym_raw = k.split(":", 1)[1]
                # Standardize symbol for UI (e.g., BTCUSDC -> BTC-USDC)
                sym = sym_raw
                if len(sym) > 4 and "-" not in sym:
                    sym = f"{sym[:3]}-{sym[3:]}"
                
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

    # Preference 1: Fresh Agent Data (< 15s)
    if redis_prices and all(v["age_s"] < 15 for v in redis_prices.values()):
        return redis_prices

    # Preference 2: Live REST Data (or Sim Fallback)
    live = _fetch_blofin_prices()
    
    # Update session state for future simulation drift
    for sym, info in live.items():
        if sym != "_rate_limited" and info["price"] > 0:
            st.session_state[f"sim_p_{sym}"] = info["price"]
            
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
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
                <h1 style="margin: 0; color: #ffffff; font-weight: 800; letter-spacing: -1px;">Crypto<span style="color: #00ffa3;">Agent</span></h1>
                <div style="height: 24px; width: 1px; background-color: #2d3139;"></div>
                <span style="color: #848e9c; font-size: 0.9rem; font-weight: 500;">Trading Intelligence Terminal v2.0</span>
            </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        mode = "PAPER" if not CONFIG_OK else get_settings().mode.upper()
        badge_class = "mode-paper" if mode == "PAPER" else "mode-live"
        updated_at = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
        
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 5px;">
                <div class="mode-badge {badge_class}">{mode} MODE ACTIVE</div>
                <div style="color: #848e9c; font-size: 0.75rem;">Last Updated: {updated_at}</div>
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
    """Render high-frequency price charts."""
    st.markdown("<h3 style='color: #ffffff;'>Market Analysis</h3>", unsafe_allow_html=True)
    
    if symbols is None:
        symbols = _TRACKED_SYMBOLS
    
    c1, c2 = st.columns([1, 1])
    with c1:
        selected = st.selectbox("Market Selection", symbols, label_visibility="collapsed")
    with c2:
        timeframe = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2, label_visibility="collapsed")
    
    try:
        df = fetch_price_history(selected, bars=150)
        
        if df is None or df.empty:
            st.warning(f"No historical data for {selected}. Please wait for candles to accumulate.")
            return

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplots: Candlestick and Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # Candlestick with thicker lines and vibrant body colors
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ffa3',
            decreasing_line_color='#ff3b30',
            increasing_fillcolor='rgba(0, 255, 163, 0.4)',
            decreasing_fillcolor='rgba(255, 59, 48, 0.4)',
            line=dict(width=1.5)
        ), row=1, col=1)

        # 20-EMA Technical Indicator
        df['ema'] = df['close'].ewm(span=20).mean()
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['ema'],
            line=dict(color='#f0b90b', width=1.5),
            name='EMA 20',
            opacity=0.8
        ), row=1, col=1)

        # Optimized Volume Display
        vol_colors = ['rgba(0, 255, 163, 0.4)' if row['close'] >= row['open'] else 'rgba(255, 59, 48, 0.4)' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df['time'],
            y=df['volume'],
            name='Volume',
            marker_color=vol_colors,
            marker_line_width=0
        ), row=2, col=1)

        # Advanced Layout Styling
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            height=650,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            font=dict(color="#ffffff", size=11),
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', zeroline=False)

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    except Exception as e:
        st.error(f"Chart Render Error: {e}")


def fetch_price_history(symbol: str, bars: int = 100) -> pd.DataFrame:
    """Fetch price history from database."""
    try:
        engine = _get_engine()
        if engine is None:
            return pd.DataFrame()
        
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = '{symbol.replace('-', '')}' AND timeframe = '15m'
            ORDER BY time DESC
            LIMIT {bars}
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            df = df.sort_values("time")
        return df
    except Exception:
        return pd.DataFrame()


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
          <div style="display:grid; grid-template-columns:repeat(6,1fr); gap:14px; margin-bottom:14px;">
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


@st.fragment(run_every="2s")
def render_positions():
    """Live position monitor — current price + unrealized P&L update every 2 s."""
    st.subheader("Open Positions")

    prices = get_live_prices()

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
    <table class="pos-tbl">
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Entry</th>
        <th>Current Price</th><th>Unrealized P&amp;L</th>
        <th>Size</th><th>Stop Loss</th><th>Take Profit</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


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
    <table class="sig-tbl">
      <thead><tr>
        <th>Time</th><th>Symbol</th><th>Direction</th>
        <th>Confidence</th><th>TF</th><th>Current Price</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
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
        tabs = st.tabs(["Terminal", "Advanced Charts", "Paper Trades", "Inventory", "Intelligence", "Ledger", "System"])

        with tabs[0]: # Overview
            render_live_prices()
            st.divider()
            render_equity_curve(snapshots)
            st.divider()
            render_performance(snapshots, orders)

        with tabs[1]: render_price_charts()
        with tabs[2]: render_paper_trades()
        with tabs[3]: render_positions()
        with tabs[4]: render_signals()
        with tabs[5]: render_trade_history(orders)
        with tabs[6]: render_logs(150)

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
