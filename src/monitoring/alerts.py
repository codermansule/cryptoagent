"""
Telegram Alerts Module
Sends real-time notifications to a Telegram chat via the Bot API.

Notification types:
  - Trade opened (entry price, SL, TP, confidence, RR)
  - Trade closed (exit price, P&L, reason)
  - Daily P&L summary
  - Circuit breaker fired (trading halted)
  - High-confidence signal (above configurable threshold)
  - Agent startup / shutdown

Setup:
  1. Create a bot via @BotFather → get TELEGRAM_BOT_TOKEN
  2. Get your chat ID by messaging @userinfobot → set TELEGRAM_CHAT_ID
  3. Set both in .env file
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Telegram Bot API base URL
_TG_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlerter:
    """
    Async Telegram notifier. All methods are fire-and-forget safe —
    failures are logged but never raised (alerts must never crash the agent).

    Usage:
        alerter = TelegramAlerter()
        await alerter.send_text("Hello from CryptoAgent!")
        await alerter.trade_opened(decision)
        await alerter.trade_closed(symbol, side, pnl, pnl_pct, reason)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        min_confidence_to_alert: float = 70.0,
    ):
        self._token   = token   or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._min_conf = min_confidence_to_alert
        self._session: Optional[aiohttp.ClientSession] = None
        self._enabled = bool(self._token and self._chat_id)

        if not self._enabled:
            logger.info(
                "Telegram alerts disabled — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in .env to enable."
            )

    # ── Session management ────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Core sender ───────────────────────────────────────────────────────

    async def send_text(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a raw text message. Returns True on success."""
        if not self._enabled:
            return False
        url  = _TG_API.format(token=self._token, method="sendMessage")
        data = {"chat_id": self._chat_id, "text": text, "parse_mode": parse_mode}
        try:
            session = await self._get_session()
            async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram API error %d: %s", resp.status, body[:200])
                    return False
                return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    # ── Notification templates ────────────────────────────────────────────

    async def agent_started(self, mode: str, symbols: list[str]) -> None:
        syms = ", ".join(symbols)
        msg = (
            f"🤖 <b>CryptoAgent Started</b>\n"
            f"Mode: <code>{mode.upper()}</code>\n"
            f"Symbols: <code>{syms}</code>\n"
            f"Time: {_now()}"
        )
        await self.send_text(msg)

    async def agent_stopped(self) -> None:
        await self.send_text(f"🛑 <b>CryptoAgent Stopped</b>\n{_now()}")

    async def trade_opened(self, decision) -> None:
        """
        decision: TradeDecision from src.decision.core
        """
        side_emoji = "🟢" if decision.side == "long" else "🔴"
        msg = (
            f"{side_emoji} <b>Trade Opened</b>\n"
            f"Symbol:     <code>{decision.symbol}</code>\n"
            f"Side:       <code>{decision.side.upper()}</code>\n"
            f"Entry:      <code>${decision.entry_price:,.4f}</code>\n"
            f"Stop Loss:  <code>${decision.stop_loss:,.4f}</code>\n"
            f"Take Profit:<code>${decision.take_profit:,.4f}</code>\n"
            f"Size:       <code>${decision.size_usdc:,.2f}</code>\n"
            f"R:R:        <code>{decision.risk_reward:.1f}x</code>\n"
            f"Confidence: <code>{decision.confidence:.1f}%</code>\n"
            f"Regime:     <code>{decision.regime}</code>\n"
            f"{_now()}"
        )
        await self.send_text(msg)

    async def trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        reason: str,
        exit_price: float,
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} <b>Trade Closed</b>\n"
            f"Symbol:  <code>{symbol}</code>\n"
            f"Side:    <code>{side.upper()}</code>\n"
            f"Exit:    <code>${exit_price:,.4f}</code>\n"
            f"P&L:     <code>${pnl:+,.2f}  ({pnl_pct:+.2%})</code>\n"
            f"Reason:  <code>{reason.upper()}</code>\n"
            f"{_now()}"
        )
        await self.send_text(msg)

    async def daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        n_trades: int,
        win_rate: float,
    ) -> None:
        emoji = "📈" if daily_pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Daily Summary</b>\n"
            f"Equity:    <code>${equity:,.2f}</code>\n"
            f"Daily P&L: <code>${daily_pnl:+,.2f}  ({daily_pnl_pct:+.2%})</code>\n"
            f"Trades:    <code>{n_trades}</code>\n"
            f"Win rate:  <code>{win_rate:.1%}</code>\n"
            f"{_now()}"
        )
        await self.send_text(msg)

    async def circuit_breaker(self, reason: str, daily_pnl_pct: float) -> None:
        msg = (
            f"🚨 <b>CIRCUIT BREAKER FIRED</b>\n"
            f"Reason:     {reason}\n"
            f"Daily P&L:  <code>{daily_pnl_pct:+.2%}</code>\n"
            f"Trading halted until next session reset.\n"
            f"{_now()}"
        )
        await self.send_text(msg)

    async def high_confidence_signal(
        self, symbol: str, timeframe: str, direction: str, confidence: float, regime: str
    ) -> None:
        if confidence < self._min_conf:
            return
        side_emoji = "⬆" if direction == "long" else "⬇"
        msg = (
            f"{side_emoji} <b>High-Confidence Signal</b>\n"
            f"Symbol:     <code>{symbol}</code>\n"
            f"Timeframe:  <code>{timeframe}</code>\n"
            f"Direction:  <code>{direction.upper()}</code>\n"
            f"Confidence: <code>{confidence:.1f}%</code>\n"
            f"Regime:     <code>{regime}</code>\n"
            f"{_now()}"
        )
        await self.send_text(msg)

    async def error_alert(self, component: str, error: str) -> None:
        msg = (
            f"⚠️ <b>Error in {component}</b>\n"
            f"<code>{error[:300]}</code>\n"
            f"{_now()}"
        )
        await self.send_text(msg)


# ── Helper ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Singleton factory ─────────────────────────────────────────────────────────

_alerter: Optional[TelegramAlerter] = None


def get_alerter() -> TelegramAlerter:
    """Return the global TelegramAlerter singleton."""
    global _alerter
    if _alerter is None:
        _alerter = TelegramAlerter()
    return _alerter
