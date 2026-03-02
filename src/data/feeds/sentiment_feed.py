"""
Sentiment & On-Chain Data Feed

Fetches market sentiment and derivatives data from free public APIs.
Results are cached and refreshed every REFRESH_INTERVAL seconds.

Data sources:
  - Fear & Greed Index  — api.alternative.me/fng/                      (free)
  - BTC Dominance       — api.coingecko.com/api/v3/global               (free)
  - Funding Rate        — fapi.binance.com/fapi/v1/premiumIndex         (free)
  - Open Interest       — fapi.binance.com/fapi/v1/openInterest         (free)
  - Crypto News NLP     — min-api.cryptocompare.com/data/v2/news/       (free)
                          Scored with Gemini Flash LLM (falls back to VADER+keywords)

Output dicts (keys match ensemble.py _sentiment_score / _onchain_score):
  sentiment_data: { fear_greed, news_sentiment, news_headline, news_score_count, nlp_engine }
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

    # ── Crypto-specific sentiment keyword dictionary ───────────────────────────
    _BULLISH_KW: dict[str, float] = {
        "all-time high": 1.0, "ath": 0.9, "record high": 0.9,
        "etf approved": 1.0, "sec approved": 0.9, "approved": 0.5,
        "institutional": 0.6, "bull market": 0.9, "bull run": 0.9,
        "breakout": 0.8, "surge": 0.8, "surges": 0.8, "rally": 0.7,
        "pump": 0.6, "moon": 0.5, "soar": 0.8, "skyrocket": 0.9,
        "adoption": 0.6, "partnership": 0.5, "launch": 0.3,
        "upgrade": 0.4, "bullish": 0.8, "accumulate": 0.5,
        "inflow": 0.6, "buy": 0.3, "recovery": 0.5, "rebound": 0.6,
        "gains": 0.5, "growth": 0.4, "outperform": 0.6,
        "legalized": 0.7, "legal tender": 0.8, "etf": 0.5,
    }
    _BEARISH_KW: dict[str, float] = {
        "crash": -1.0, "crashes": -1.0, "collapse": -0.9, "collapses": -0.9,
        "hack": -0.9, "hacked": -0.9, "exploit": -0.8, "stolen": -0.9,
        "liquidation": -0.8, "liquidated": -0.8, "mass liquidation": -1.0,
        "bankrupt": -0.9, "bankruptcy": -0.9, "insolvent": -0.9,
        "fraud": -0.9, "scam": -0.9, "ponzi": -1.0, "rug pull": -1.0,
        "ban": -0.8, "banned": -0.8, "crackdown": -0.7, "restrict": -0.6,
        "lawsuit": -0.7, "sue": -0.6, "investigation": -0.6, "sec charges": -0.9,
        "bear market": -0.9, "bear run": -0.8, "bearish": -0.8,
        "dump": -0.8, "dumping": -0.8, "sell-off": -0.7, "selloff": -0.7,
        "plunge": -0.8, "plunges": -0.8, "drops": -0.5, "decline": -0.5,
        "warning": -0.4, "concern": -0.3, "risk": -0.2, "outflow": -0.5,
        "fear": -0.4, "panic": -0.7, "contagion": -0.8,
    }

    def __init__(self, refresh_interval: int = REFRESH_INTERVAL) -> None:
        self._refresh_interval = refresh_interval
        self._last_refresh: float = 0.0
        self._sentiment: dict = {}
        self._onchain: dict = {}
        self._prev_oi: Optional[float] = None
        self._lock = asyncio.Lock()

        import os
        # Azure OpenAI GPT-5.2 (primary — best quality)
        self._az_endpoint:   str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._az_key:        str = os.environ.get("AZURE_OPENAI_KEY", "")
        self._az_deployment: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
        self._az_api_ver:    str = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        # Gemini (fallback if Azure not set)
        self._gemini_key: str = os.environ.get("GEMINI_API_KEY", "")

        if self._az_endpoint and self._az_key:
            logger.info("Azure OpenAI NLP enabled  deployment=%s", self._az_deployment)
        elif self._gemini_key:
            logger.info("Gemini NLP scorer enabled (gemini-2.0-flash) — Azure not configured")
        else:
            logger.info("No LLM API configured — falling back to VADER+keyword NLP")

        # VADER NLP scorer (fallback when Gemini not available)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except ImportError:
            self._vader = None

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
        fg_res, dom_res, fund_res, news_res = await asyncio.gather(
            self._fetch_fear_greed(),
            self._fetch_btc_dominance(),
            self._fetch_funding_and_oi("BTCUSDT"),
            self._fetch_news_sentiment(),
            return_exceptions=True,
        )

        sentiment: dict = {}
        if isinstance(fg_res, dict):
            sentiment.update(fg_res)
        if isinstance(dom_res, dict):
            sentiment.update(dom_res)
        if isinstance(news_res, dict):
            sentiment.update(news_res)

        onchain: dict = {}
        if isinstance(fund_res, dict):
            onchain.update(fund_res)

        self._sentiment = sentiment
        self._onchain = onchain

        logger.info(
            "SentimentFeed refreshed  fear_greed=%s  news_sentiment=%.3f (%s)  btc_dom=%.1f%%  funding=%s",
            sentiment.get("fear_greed", "?"),
            sentiment.get("news_sentiment", 0.0),
            sentiment.get("news_headline", "n/a")[:50],
            sentiment.get("btc_dominance", 0.0),
            onchain.get("funding_rate", "?"),
        )

    @staticmethod
    def _make_session():
        """Create an aiohttp session with ThreadedResolver for Windows DNS compat."""
        import aiohttp
        connector = aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
        return aiohttp.ClientSession(connector=connector)

    async def _fetch_fear_greed(self) -> dict:
        try:
            import aiohttp
            async with self._make_session() as session:
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
            async with self._make_session() as session:
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
            async with self._make_session() as session:
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

    async def _fetch_news_sentiment(self) -> dict:
        """
        Fetch recent crypto headlines from CryptoCompare and score with GPT-5.2.
        Priority: Azure OpenAI GPT-5.2 → Gemini → VADER+keywords.
        Returns full sentiment dict including market_briefing.
        """
        try:
            import aiohttp
            async with self._make_session() as session:
                async with session.get(
                    "https://min-api.cryptocompare.com/data/v2/news/"
                    "?lang=EN&sortOrder=latest&limit=20",
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json(content_type=None)

            articles = data.get("Data", [])
            if not articles:
                return {}

            titles = [art.get("title", "") for art in articles[:20]]

            # ── Priority 1: Azure OpenAI GPT-5.2 ─────────────────────────────
            nlp_engine = "vader+keywords"
            scores = []
            market_briefing = ""
            dominant_theme = "neutral"

            az_result = await self._score_with_azure_openai(titles)
            if az_result and len(az_result.get("scores", [])) == len(titles):
                scores = az_result["scores"]
                market_briefing = az_result.get("briefing", "")
                dominant_theme = az_result.get("dominant_theme", "neutral")
                nlp_engine = "gpt-5.2"
            else:
                # ── Priority 2: Gemini ────────────────────────────────────────
                gemini_scores = await self._score_with_gemini(titles)
                if gemini_scores and len(gemini_scores) == len(titles):
                    scores = gemini_scores
                    nlp_engine = "gemini-2.0-flash"
                else:
                    # ── Priority 3: VADER + keywords ──────────────────────────
                    for art in articles[:20]:
                        title = art.get("title", "")
                        body  = art.get("body", "")[:300]
                        scores.append(self._score_text(f"{title} {body}"))

            if not scores:
                return {}

            # Decay weights — most recent articles matter more
            weights = [1.0 / (1.0 + i * 0.2) for i in range(len(scores))]
            weighted = sum(s * w for s, w in zip(scores, weights))
            total_w  = sum(weights)
            net = weighted / total_w if total_w > 0 else 0.0
            net = max(-1.0, min(1.0, net))

            top_headline = articles[0].get("title", "") if articles else ""
            logger.debug(
                "News NLP [%s]: score=%.3f  theme=%s  n=%d  lead=%s",
                nlp_engine, net, dominant_theme, len(scores), top_headline[:60],
            )
            return {
                "news_sentiment":   float(net),
                "news_headline":    top_headline,
                "news_score_count": len(scores),
                "nlp_engine":       nlp_engine,
                "market_briefing":  market_briefing,
                "dominant_theme":   dominant_theme,
                "article_scores":   scores,
                "article_titles":   titles,
            }

        except Exception as exc:
            logger.debug("News sentiment fetch failed: %s", exc)
        return {}

    async def _score_with_azure_openai(self, headlines: list[str]) -> dict:
        """
        Score headlines using Azure OpenAI GPT-5.2.
        Returns {"scores": [...], "briefing": "...", "dominant_theme": "..."}.
        Returns {} on failure.
        """
        if not self._az_endpoint or not self._az_key or not headlines:
            return {}
        try:
            import json
            from openai import AzureOpenAI

            client = AzureOpenAI(
                azure_endpoint=self._az_endpoint,
                api_key=self._az_key,
                api_version=self._az_api_ver,
            )

            numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
            prompt = f"""You are a professional crypto trading analyst with deep expertise in Bitcoin, Ethereum, and macro market dynamics.

Analyze these {len(headlines)} crypto news headlines and return a JSON object with:
1. "scores": array of {len(headlines)} floats from -1.0 (very bearish) to +1.0 (very bullish) — one per headline, in order
2. "briefing": 2-sentence market narrative summarising the overall sentiment and its likely near-term price impact
3. "dominant_theme": one of "strongly_bullish" | "bullish" | "neutral" | "bearish" | "strongly_bearish"

Critical context rules:
- "crashes through resistance to new ATH" → bullish (+0.9), NOT bearish
- Large liquidations of shorts = bullish; long liquidations = bearish
- Regulatory clarity (even if restrictive) is usually medium-term bullish
- ETF approvals, institutional buying, treasuries adding BTC → strongly bullish
- Exchange hacks, rug pulls, insolvency → strongly bearish
- Technical analysis articles are near-neutral (score ±0.1 based on bullish/bearish bias in title)

Headlines:
{numbered}

Return ONLY valid JSON. No markdown, no explanation."""

            # Run synchronous SDK call in thread pool (Responses API for GPT-5.2)
            response = await asyncio.to_thread(
                client.responses.create,
                model=self._az_deployment,
                input=[
                    {"role": "system", "content": "You are a crypto market sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user",   "content": prompt},
                ],
                text={"format": {"type": "json_object"}},
                max_output_tokens=800,
            )

            # Extract text — try output_text convenience property first,
            # then fall back to manual traversal of output[*].content[*].text
            output_text = getattr(response, "output_text", None)
            if not output_text:
                for item in getattr(response, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                        t = getattr(c, "text", None) or getattr(c, "transcript", None)
                        if t:
                            output_text = t
                            break
                    if output_text:
                        break
            if not output_text:
                logger.info("GPT-5.2 returned empty output_text — falling back")
                return {}

            raw = json.loads(output_text)
            scores = [max(-1.0, min(1.0, float(s))) for s in raw.get("scores", [])]
            if len(scores) != len(headlines):
                logger.info("GPT-5.2 returned %d scores for %d headlines — falling back", len(scores), len(headlines))
                return {}

            logger.info(
                "GPT-5.2 NLP: theme=%s  briefing=%s",
                raw.get("dominant_theme", "?"),
                raw.get("briefing", "")[:80],
            )
            return {
                "scores":         scores,
                "briefing":       raw.get("briefing", ""),
                "dominant_theme": raw.get("dominant_theme", "neutral"),
            }

        except Exception as exc:
            logger.info("Azure OpenAI scoring failed: %s", exc)
            return {}

    async def _score_with_gemini(self, headlines: list[str]) -> list[float]:
        """Fallback: Gemini Flash batch headline scorer."""
        if not self._gemini_key or not headlines:
            return []
        try:
            import json
            import re
            import google.generativeai as genai

            genai.configure(api_key=self._gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
            prompt = (
                "You are a crypto trading sentiment analyst. Score each headline's "
                "expected impact on cryptocurrency prices from -1.0 (very bearish) "
                "to +1.0 (very bullish). Consider crypto-specific context carefully.\n\n"
                f"Headlines:\n{numbered}\n\n"
                f"Return ONLY a JSON array of {len(headlines)} numbers. Example: [-0.8, 0.5, 0.1]"
            )

            response = await asyncio.to_thread(model.generate_content, prompt)
            match = re.search(r'\[[\s\d,.\-+]+\]', response.text.strip())
            if not match:
                return []
            raw = json.loads(match.group())
            return [max(-1.0, min(1.0, float(s))) for s in raw]

        except Exception as exc:
            logger.debug("Gemini scoring failed: %s", exc)
            return []

    def _score_text(self, text: str) -> float:
        """
        Fallback scorer: hybrid VADER + crypto keyword dictionary.
        Used when Gemini is not available.
        """
        text_lower = text.lower()

        kw_scores = []
        for phrase, weight in self._BULLISH_KW.items():
            if phrase in text_lower:
                kw_scores.append(weight)
        for phrase, weight in self._BEARISH_KW.items():
            if phrase in text_lower:
                kw_scores.append(weight)

        if kw_scores:
            kw_net = max(-1.0, min(1.0, sum(kw_scores) / len(kw_scores)))
        else:
            kw_net = 0.0

        vader_net = float(self._vader.polarity_scores(text)["compound"]) if self._vader else 0.0

        return (0.7 * kw_net + 0.3 * vader_net) if kw_scores else vader_net
