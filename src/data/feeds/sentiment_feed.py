"""
Sentiment & On-Chain Data Feed

Fetches market sentiment and derivatives data from free public APIs.
Results are cached and refreshed every REFRESH_INTERVAL seconds.

Data sources:
  - Fear & Greed Index  — api.alternative.me/fng/         (free)
  - BTC Dominance       — api.coingecko.com/api/v3/global  (free)
  - Funding Rate        — fapi.binance.com/fapi/v1/premiumIndex (free)
  - Open Interest       — fapi.binance.com/fapi/v1/openInterest (free)

Output dicts (keys match ensemble.py _sentiment_score / _onchain_score):
  sentiment_data: { fear_greed }
  onchain_data:   { funding_diff, funding_rate, btc_dominance, oi_change, open_interest }
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = 300  # seconds (5 minutes)


class SentimentFeed:
    """
    Lightweight async cache for sentiment + on-chain data.

    Usage:
        feed = SentimentFeed()
        await feed.refresh_if_stale()
        sentiment = feed.sentiment_data
        onchain   = feed.onchain_data
    """

    def __init__(self, refresh_interval: int = REFRESH_INTERVAL) -> None:
        self._refresh_interval = refresh_interval
        self._last_refresh: float = 0.0
        self._sentiment: dict = {}
        self._onchain: dict = {}
        self._prev_oi: Optional[float] = None
        self._lock = asyncio.Lock()

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def sentiment_data(self) -> dict:
        return dict(self._sentiment)

    @property
    def onchain_data(self) -> dict:
        return dict(self._onchain)

    async def refresh_if_stale(self) -> None:
        """Refresh all data sources if cache is older than refresh_interval."""
        if time.monotonic() - self._last_refresh < self._refresh_interval:
            return
        async with self._lock:
            # Double-check inside the lock (another task may have refreshed)
            if time.monotonic() - self._last_refresh < self._refresh_interval:
                return
            await self._fetch_all()
            self._last_refresh = time.monotonic()

    # ── Internal fetch ────────────────────────────────────────────────────────

    async def _fetch_all(self) -> None:
        """Fetch all sources concurrently. Failures are silently swallowed."""
        fg_res, dom_res, fund_res = await asyncio.gather(
            self._fetch_fear_greed(),
            self._fetch_btc_dominance(),
            self._fetch_funding_and_oi("BTCUSDT"),
            return_exceptions=True,
        )

        sentiment: dict = {}
        if isinstance(fg_res, dict):
            sentiment.update(fg_res)
        if isinstance(dom_res, dict):
            sentiment.update(dom_res)

        onchain: dict = {}
        if isinstance(fund_res, dict):
            onchain.update(fund_res)

        self._sentiment = sentiment
        self._onchain = onchain

        logger.debug(
            "SentimentFeed refreshed  fear_greed=%s  btc_dom=%.1f%%  funding=%s",
            sentiment.get("fear_greed", "?"),
            onchain.get("btc_dominance", 0.0),
            onchain.get("funding_rate", "?"),
        )

    async def _fetch_fear_greed(self) -> dict:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        fg = data.get("data", [{}])[0]
                        return {"fear_greed": int(fg.get("value", 50))}
        except Exception as exc:
            logger.debug("Fear & greed fetch failed: %s", exc)
        return {}

    async def _fetch_btc_dominance(self) -> dict:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.coingecko.com/api/v3/global",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        pct = data.get("data", {}).get("market_cap_percentage", {})
                        return {"btc_dominance": float(pct.get("btc", 50.0))}
        except Exception as exc:
            logger.debug("BTC dominance fetch failed: %s", exc)
        return {}

    async def _fetch_funding_and_oi(self, symbol: str) -> dict:
        import aiohttp
        result: dict = {}
        try:
            async with aiohttp.ClientSession() as session:
                # ── Funding rate ──────────────────────────────────────────
                try:
                    async with session.get(
                        f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            rate = float(data.get("lastFundingRate", 0.0))
                            result["funding_rate"] = rate
                            result["funding_diff"] = rate   # deviation from 0
                except Exception as exc:
                    logger.debug("Funding rate fetch failed: %s", exc)

                # ── Open interest ─────────────────────────────────────────
                try:
                    async with session.get(
                        f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            oi = float(data.get("openInterest", 0.0))
                            result["open_interest"] = oi
                            # Compute % change from previous reading
                            if self._prev_oi and self._prev_oi > 0:
                                result["oi_change"] = (oi - self._prev_oi) / self._prev_oi * 100
                            else:
                                result["oi_change"] = 0.0
                            self._prev_oi = oi
                except Exception as exc:
                    logger.debug("Open interest fetch failed: %s", exc)

        except Exception as exc:
            logger.debug("Funding/OI session failed: %s", exc)
        return result
