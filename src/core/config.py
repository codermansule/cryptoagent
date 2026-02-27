"""
Central configuration — loads .env + YAML settings into Pydantic models.
All modules import from here; never read env vars directly elsewhere.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ── Pydantic Settings (reads from .env) ───────────────────────────────────────

class BloFinSecrets(BaseSettings):
    api_key: str = Field("", alias="BLOFIN_API_KEY")
    api_secret: str = Field("", alias="BLOFIN_API_SECRET")
    passphrase: str = Field("", alias="BLOFIN_API_PASSPHRASE")
    testnet: bool = Field(True, alias="BLOFIN_TESTNET")

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")


class BinanceSecrets(BaseSettings):
    api_key: str = Field("", alias="BINANCE_API_KEY")
    api_secret: str = Field("", alias="BINANCE_API_SECRET")

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")


class DatabaseSettings(BaseSettings):
    timescale_url: str = Field(
        "postgresql://agent:password@localhost:5432/cryptoagent",
        alias="TIMESCALE_URL",
    )
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")


class TelegramSettings(BaseSettings):
    bot_token: str = Field("", alias="TELEGRAM_BOT_TOKEN")
    chat_id: str = Field("", alias="TELEGRAM_CHAT_ID")

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")


class AgentEnvSettings(BaseSettings):
    mode: Literal["paper", "live"] = Field("paper", alias="AGENT_MODE")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")


# ── YAML Settings ─────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Settings:
    """Aggregated settings object. Use get_settings() to obtain a cached instance."""

    def __init__(self) -> None:
        raw = _load_yaml(CONFIG_DIR / "settings.yaml")
        exc = _load_yaml(CONFIG_DIR / "exchanges.yaml")

        # Environment-sourced secrets
        self.blofin = BloFinSecrets()
        self.binance = BinanceSecrets()
        self.database = DatabaseSettings()
        self.telegram = TelegramSettings()
        env = AgentEnvSettings()

        # Agent
        _agent = raw["agent"]
        self.mode: Literal["paper", "live"] = env.mode
        self.log_level: str = env.log_level
        self.primary_exchange: str = _agent["primary_exchange"]
        self.heartbeat_interval: int = _agent["heartbeat_interval"]

        # Trading
        _t = raw["trading"]
        self.symbols: list[str] = _t["symbols"]
        self.market_types: list[str] = _t["market_types"]
        self.default_leverage: int = _t["default_leverage"]
        self.max_leverage: int = _t["max_leverage"]
        self.timeframes: list[str] = _t["timeframes"]
        self.primary_timeframe: str = _t["primary_timeframe"]

        # Signals
        _s = raw["signals"]
        self.min_confidence_threshold: float = _s["min_confidence_threshold"]
        self.confluence_required: int = _s["confluence_required"]
        self.lookback_candles: int = _s["lookback_candles"]

        # Risk
        _r = raw["risk"]
        self.max_portfolio_risk_pct: float = _r["max_portfolio_risk_pct"]
        self.max_single_trade_risk_pct: float = _r["max_single_trade_risk_pct"]
        self.max_daily_drawdown_pct: float = _r["max_daily_drawdown_pct"]
        self.max_weekly_drawdown_pct: float = _r["max_weekly_drawdown_pct"]
        self.kelly_fraction: float = _r["kelly_fraction"]
        self.min_liquidation_distance_atr: float = _r["min_liquidation_distance_atr"]
        self.max_open_positions: int = _r["max_open_positions"]
        self.max_correlated_positions: int = _r["max_correlated_positions"]
        self.atr_sl_multiplier: float = _r.get("atr_sl_multiplier", 2.5)
        self.rr_ratio: float = _r.get("rr_ratio", 2.0)
        self.adx_min_threshold: float = _r.get("adx_min_threshold", 25.0)

        # Execution
        _e = raw["execution"]
        self.default_order_type: str = _e["default_order_type"]
        self.limit_order_offset_pct: float = _e["limit_order_offset_pct"]
        self.max_slippage_pct: float = _e["max_slippage_pct"]
        self.order_timeout_seconds: int = _e["order_timeout_seconds"]
        self.retry_attempts: int = _e["retry_attempts"]
        self.retry_backoff_seconds: int = _e["retry_backoff_seconds"]

        # Paper trading
        _p = raw["paper"]
        self.paper_starting_balance: float = _p["starting_balance"]
        self.paper_fee_pct: float = _p["simulated_fee_pct"]
        self.paper_slippage_pct: float = _p["simulated_slippage_pct"]

        # Exchange endpoints
        self.exchange_config: dict = exc

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"

    def blofin_endpoint(self, name: str) -> str:
        base = self.exchange_config["blofin"]["rest_base_url"]
        path = self.exchange_config["blofin"]["endpoints"][name]
        return f"{base}{path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
