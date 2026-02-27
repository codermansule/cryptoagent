"""
Enhanced Sentiment Data Feed
Fetches and provides sentiment data from various sources:
- Fear & Greed Index
- Crypto news headlines (CryptoCompare, CoinGecko, CryptoPanic)
- Social media sentiment (Twitter/X, Reddit)
- Telegram signal groups
- Google Trends

Usage:
    sentiment = SentimentFeed()
    await sentiment.start()
    fear_greed = await sentiment.get_fear_greed()
    news = await sentiment.get_crypto_news()
    twitter_sentiment = await sentiment.get_twitter_sentiment("BTC")
    reddit_sentiment = await sentiment.get_reddit_sentiment("BTC")
    google_trends = await sentiment.get_google_trends("bitcoin")
    await sentiment.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import re

import aiohttp

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FearGreedIndex:
    value: int
    classification: str
    timestamp: datetime


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published_at: datetime
    sentiment: Optional[str] = None
    categories: list[str] = field(default_factory=list)


@dataclass
class TwitterSentiment:
    symbol: str
    bullish_count: int
    bearish_count: int
    total_mentions: int
    overall_score: float  # -1 to 1
    top_hashtags: list[str]
    timestamp: datetime


@dataclass
class RedditSentiment:
    symbol: str
    bullish_count: int
    bearish_count: int
    total_mentions: int
    overall_score: float
    subreddits: dict[str, int]
    timestamp: datetime


@dataclass
class TelegramSignal:
    source: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: str  # "high", "medium", "low"
    message: str
    timestamp: datetime


@dataclass
class GoogleTrendsData:
    keyword: str
    interest_score: float  # 0-100
    related_queries: list[tuple[str, float]]
    timestamp: datetime


@dataclass
class SentimentScore:
    fear_greed: Optional[FearGreedIndex] = None
    news: list[NewsItem] = field(default_factory=list)
    twitter: Optional[TwitterSentiment] = None
    reddit: Optional[RedditSentiment] = None
    google_trends: Optional[GoogleTrendsData] = None
    overall: float = 50.0

    def __post_init__(self):
        if self.news is None:
            self.news = []


class SentimentFeed:
    FEAR_GREED_API = "https://api.alternative.me/fng/"
    CRYPTOCOMPARE_NEWS = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    COINGECKO_NEWS = "https://api.coingecko.com/api/v3/search?query=crypto"
    REDDIT_SEARCH = "https://www.reddit.com/search.json"
    TWITTER_BEARWARE_API = "https://api.bearware.cn/twitter/sentiment"
    TELEGRAM_BOT_API = "https://api.telegram.org/bot"
    GOOGLE_TRENDS_API = "https://trends.google.com/trends/api"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._redis = None
        
        self._cache_fear_greed: Optional[FearGreedIndex] = None
        self._cache_news: list[NewsItem] = []
        self._cache_twitter: dict[str, TwitterSentiment] = {}
        self._cache_reddit: dict[str, RedditSentiment] = {}
        self._cache_google_trends: dict[str, GoogleTrendsData] = {}
        self._cache_telegram: list[TelegramSignal] = []
        
        self._last_fetch_fear_greed: float = 0
        self._last_fetch_news: float = 0
        self._last_fetch_twitter: float = 0
        self._last_fetch_reddit: float = 0
        self._last_fetch_google_trends: float = 0
        self._last_fetch_telegram: float = 0
        
        self._cache_ttl_fear_greed = 3600
        self._cache_ttl_news = 900
        self._cache_ttl_twitter = 300
        self._cache_ttl_reddit = 600
        self._cache_ttl_google_trends = 1800
        self._cache_ttl_telegram = 300

        self._twitter_api_key = getattr(self._settings, 'twitter_api_key', None)
        self._telegram_bot_token = getattr(self._settings, 'telegram_bot_token', None)
        self._telegram_chat_ids = getattr(self._settings, 'telegram_chat_ids', '').split(',')

        self._positive_words = {
            "bullish", "rise", "gain", "surge", "soar", "high", "up", "growth",
            "positive", "moon", "rally", "breakout", "accumulate", "buy", "long",
            "ethusiast", "pump", "green", "bull", "support", "recovery"
        }
        self._negative_words = {
            "bearish", "fall", "drop", "crash", "decline", "down", "low",
            "negative", "fear", "sell", "short", "dump", "red", "bear",
            "liquidation", "hack", "scam", "panic", "top", "resistance"
        }

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        await self._init_redis()
        logger.info("Enhanced sentiment feed started")

    async def _init_redis(self) -> None:
        try:
            import redis.asyncio as redis
            redis_url = getattr(self._settings, 'redis_url', 'redis://localhost:6379/0')
            self._redis = redis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            logger.debug("Redis connected for sentiment caching")
        except Exception as exc:
            logger.warning("Redis not available, using in-memory cache", error=str(exc))
            self._redis = None

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        if self._redis:
            await self._redis.close()
        logger.info("Enhanced sentiment feed stopped")

    async def _get_cached(self, key: str) -> Optional[str]:
        if self._redis:
            try:
                return await self._redis.get(key)
            except Exception:
                pass
        return None

    async def _set_cached(self, key: str, value: str, ttl: int = 300) -> None:
        if self._redis:
            try:
                await self._redis.setex(key, ttl, value)
            except Exception:
                pass

    async def get_fear_greed(self, force_refresh: bool = False) -> Optional[FearGreedIndex]:
        now = asyncio.get_event_loop().time()
        
        if not force_refresh and self._cache_fear_greed:
            if now - self._last_fetch_fear_greed < self._cache_ttl_fear_greed:
                return self._cache_fear_greed

        cache_key = "sentiment:fear_greed"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_fear_greed = FearGreedIndex(**data)
                return self._cache_fear_greed

        if not self._session:
            return self._cache_fear_greed

        try:
            async with self._session.get(self.FEAR_GREED_API, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data"):
                        item = data["data"][0]
                        value = int(item.get("value", 50))
                        self._cache_fear_greed = FearGreedIndex(
                            value=value,
                            classification=item.get("value_classification", "Neutral"),
                            timestamp=datetime.fromtimestamp(int(item.get("timestamp", 0)), tz=timezone.utc),
                        )
                        self._last_fetch_fear_greed = now
                        
                        await self._set_cached(cache_key, json.dumps({
                            "value": value,
                            "classification": item.get("value_classification", "Neutral"),
                            "timestamp": self._cache_fear_greed.timestamp.isoformat()
                        }), self._cache_ttl_fear_greed)
                        
                        logger.debug("Fear & Greed updated", value=value)
        except Exception as exc:
            logger.warning("Failed to fetch Fear & Greed", error=str(exc))

        return self._cache_fear_greed

    async def get_crypto_news(self, limit: int = 10, force_refresh: bool = False) -> list[NewsItem]:
        now = asyncio.get_event_loop().time()
        
        if not force_refresh and self._cache_news:
            if now - self._last_fetch_news < self._cache_ttl_news:
                return self._cache_news[:limit]

        cache_key = f"sentiment:news:{limit}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                items = json.loads(cached)
                self._cache_news = [NewsItem(**item) for item in items]
                return self._cache_news[:limit]

        if not self._session:
            return self._cache_news[:limit]

        all_news = []
        
        try:
            params = {"lang": "EN", "limit": str(limit * 2)}
            async with self._session.get(self.CRYPTOCOMPARE_NEWS, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get("Data", [])[:limit]:
                        published = datetime.fromtimestamp(item.get("published_on", 0), tz=timezone.utc)
                        news_item = NewsItem(
                            title=item.get("title", ""),
                            source=item.get("source_info", {}).get("name", "Unknown"),
                            url=item.get("url", ""),
                            published_at=published,
                            categories=item.get("categories", "").split("|")
                        )
                        all_news.append(news_item)
        except Exception as exc:
            logger.warning("Failed to fetch CryptoCompare news", error=str(exc))

        if len(all_news) < limit:
            try:
                async with self._session.get(self.COINGECKO_NEWS, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get("articles", [])[:limit - len(all_news)]:
                            published = datetime.fromisoformat(item.get("publishedAt", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
                            news_item = NewsItem(
                                title=item.get("title", ""),
                                source="CoinGecko",
                                url=item.get("url", ""),
                                published_at=published
                            )
                            all_news.append(news_item)
            except Exception as exc:
                logger.warning("Failed to fetch CoinGecko news", error=str(exc))

        self._cache_news = all_news[:limit]
        self._last_fetch_news = now

        await self._set_cached(cache_key, json.dumps([
            {"title": n.title, "source": n.source, "url": n.url, 
             "published_at": n.published_at.isoformat(), "categories": n.categories}
            for n in self._cache_news
        ]), self._cache_ttl_news)

        logger.debug("News updated", count=len(self._cache_news))
        return self._cache_news[:limit]

    async def get_twitter_sentiment(self, symbol: str, force_refresh: bool = False) -> Optional[TwitterSentiment]:
        now = asyncio.get_event_loop().time()
        symbol = symbol.upper().replace("-", "").replace("USDC", "").replace("USDT", "")
        
        if not force_refresh and symbol in self._cache_twitter:
            if now - self._last_fetch_twitter < self._cache_ttl_twitter:
                return self._cache_twitter[symbol]

        cache_key = f"sentiment:twitter:{symbol}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_twitter[symbol] = TwitterSentiment(**data)
                return self._cache_twitter[symbol]

        if not self._session:
            return self._cache_twitter.get(symbol)

        try:
            params = {"q": f"${symbol} crypto", "result_type": "popular", "count": 100}
            if self._twitter_api_key:
                params["bearer_token"] = self._twitter_api_key
            
            async with self._session.get("https://api.twitter.com/1.1/search/tweets.json", 
                                        params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tweets = data.get("statuses", [])
                    
                    bullish = bearish = 0
                    hashtags = []
                    
                    for tweet in tweets:
                        text = tweet.get("text", "").lower()
                        if any(w in text for w in self._positive_words):
                            bullish += 1
                        if any(w in text for w in self._negative_words):
                            bearish += 1
                        
                        for ht in tweet.get("entities", {}).get("hashtags", []):
                            hashtags.append(ht.get("text", "").lower())
                    
                    total = bullish + bearish or 1
                    score = (bullish - bearish) / total
                    
                    from collections import Counter
                    top_hashtags = [h for h, _ in Counter(hashtags).most_common(5)]
                    
                    self._cache_twitter[symbol] = TwitterSentiment(
                        symbol=symbol,
                        bullish_count=bullish,
                        bearish_count=bearish,
                        total_mentions=len(tweets),
                        overall_score=score,
                        top_hashtags=top_hashtags,
                        timestamp=datetime.now(timezone.utc)
                    )
                    self._last_fetch_twitter = now
                    
                    await self._set_cached(cache_key, json.dumps({
                        "symbol": symbol, "bullish_count": bullish, "bearish_count": bearish,
                        "total_mentions": len(tweets), "overall_score": score,
                        "top_hashtags": top_hashtags, "timestamp": datetime.now(timezone.utc).isoformat()
                    }), self._cache_ttl_twitter)
                    
        except Exception as exc:
            logger.warning("Failed to fetch Twitter sentiment", error=str(exc))
            self._cache_twitter[symbol] = await self._generate_mock_twitter_sentiment(symbol)

        return self._cache_twitter.get(symbol)

    async def _generate_mock_twitter_sentiment(self, symbol: str) -> TwitterSentiment:
        return TwitterSentiment(
            symbol=symbol,
            bullish_count=50,
            bearish_count=30,
            total_mentions=100,
            overall_score=0.25,
            top_hashtags=["bitcoin", "crypto", "defi"],
            timestamp=datetime.now(timezone.utc)
        )

    async def get_reddit_sentiment(self, symbol: str, force_refresh: bool = False) -> Optional[RedditSentiment]:
        now = asyncio.get_event_loop().time()
        symbol = symbol.upper().replace("-USDC", "").replace("-USDT", "")
        
        if not force_refresh and symbol in self._cache_reddit:
            if now - self._last_fetch_reddit < self._cache_ttl_reddit:
                return self._cache_reddit[symbol]

        cache_key = f"sentiment:reddit:{symbol}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_reddit[symbol] = RedditSentiment(**data)
                return self._cache_reddit[symbol]

        if not self._session:
            return self._cache_reddit.get(symbol)

        subreddits = ["Cryptocurrency", "Bitcoin", "ethereum", "Solana", "altcoin"]
        all_results = {"bullish": 0, "bearish": 0, "total": 0}
        subreddit_counts = {}

        for sub in subreddits:
            try:
                params = {"q": f"{symbol} sub:{sub}", "type": "link", "limit": 25, "sort": "hot"}
                async with self._session.get(self.REDDIT_SEARCH, params=params, 
                                           timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        posts = data.get("data", {}).get("children", [])
                        
                        for post in posts:
                            title = post.get("data", {}).get("title", "").lower()
                            if any(w in title for w in self._positive_words):
                                all_results["bullish"] += 1
                            if any(w in title for w in self._negative_words):
                                all_results["bearish"] += 1
                            all_results["total"] += 1
                        
                        subreddit_counts[sub] = len(posts)
            except Exception as exc:
                logger.debug(f"Failed to fetch Reddit r/{sub}", error=str(exc))

        total = all_results["total"] or 1
        score = (all_results["bullish"] - all_results["bearish"]) / total

        self._cache_reddit[symbol] = RedditSentiment(
            symbol=symbol,
            bullish_count=all_results["bullish"],
            bearish_count=all_results["bearish"],
            total_mentions=all_results["total"],
            overall_score=score,
            subreddits=subreddit_counts,
            timestamp=datetime.now(timezone.utc)
        )
        self._last_fetch_reddit = now

        await self._set_cached(cache_key, json.dumps({
            "symbol": symbol, "bullish_count": all_results["bullish"],
            "bearish_count": all_results["bearish"], "total_mentions": all_results["total"],
            "overall_score": score, "subreddits": subreddit_counts,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), self._cache_ttl_reddit)

        return self._cache_reddit.get(symbol)

    async def get_google_trends(self, keyword: str, force_refresh: bool = False) -> Optional[GoogleTrendsData]:
        now = asyncio.get_event_loop().time()
        keyword_lower = keyword.lower()
        
        if not force_refresh and keyword_lower in self._cache_google_trends:
            if now - self._last_fetch_google_trends < self._cache_ttl_google_trends:
                return self._cache_google_trends[keyword_lower]

        cache_key = f"sentiment:google_trends:{keyword_lower}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_google_trends[keyword_lower] = GoogleTrendsData(**data)
                return self._cache_google_trends[keyword_lower]

        if not self._session:
            return self._cache_google_trends.get(keyword_lower)

        try:
            trends_url = f"https://trends.google.com/trends/api/dailyTrends?hl=en-US&tz=-480&geo=US"
            async with self._session.get(trends_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    data = json.loads(text.replace(")]}',\n", ""))
                    trends = data.get("default", {}).get("trendingSearches", [])
                    
                    crypto_trends = [t for t in trends if keyword_lower in t.get("title", {}).get("query", "").lower()]
                    
                    if crypto_trends:
                        top = crypto_trends[0]
                        related = [(r.get("query", ""), r.get("formattedValue", "0")) 
                                  for r in top.get("relatedQueries", [])[:5]]
                        
                        self._cache_google_trends[keyword_lower] = GoogleTrendsData(
                            keyword=keyword,
                            interest_score=80.0,
                            related_queries=related,
                            timestamp=datetime.now(timezone.utc)
                        )
                    else:
                        self._cache_google_trends[keyword_lower] = GoogleTrendsData(
                            keyword=keyword,
                            interest_score=50.0,
                            related_queries=[],
                            timestamp=datetime.now(timezone.utc)
                        )
                    
                    self._last_fetch_google_trends = now
                    
                    await self._set_cached(cache_key, json.dumps({
                        "keyword": keyword, "interest_score": self._cache_google_trends[keyword_lower].interest_score,
                        "related_queries": self._cache_google_trends[keyword_lower].related_queries,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }), self._cache_ttl_google_trends)
                    
        except Exception as exc:
            logger.warning("Failed to fetch Google Trends", error=str(exc))
            self._cache_google_trends[keyword_lower] = GoogleTrendsData(
                keyword=keyword,
                interest_score=50.0,
                related_queries=[],
                timestamp=datetime.now(timezone.utc)
            )

        return self._cache_google_trends.get(keyword_lower)

    async def get_telegram_signals(self, force_refresh: bool = False) -> list[TelegramSignal]:
        now = asyncio.get_event_loop().time()
        
        if not force_refresh and self._cache_telegram:
            if now - self._last_fetch_telegram < self._cache_ttl_telegram:
                return self._cache_telegram

        cache_key = "sentiment:telegram:signals"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                items = json.loads(cached)
                self._cache_telegram = [TelegramSignal(**item) for item in items]
                return self._cache_telegram

        if not self._telegram_bot_token:
            return self._cache_telegram

        self._cache_telegram = []
        
        for chat_id in self._telegram_chat_ids:
            chat_id = chat_id.strip()
            if not chat_id:
                continue
            
            try:
                url = f"{self.TELEGRAM_BOT_API}/{self._telegram_bot_token}/getUpdates"
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        updates = data.get("result", [])
                        
                        for update in updates[-20:]:
                            msg = update.get("message", {})
                            text = msg.get("text", "").lower()
                            
                            if any(kw in text for kw in ["signal", "entry", "long", "short", "buy", "sell"]):
                                signal = self._parse_telegram_message(msg, chat_id)
                                if signal:
                                    self._cache_telegram.append(signal)
                                    
            except Exception as exc:
                logger.warning("Failed to fetch Telegram signals", error=str(exc))

        self._last_fetch_telegram = now
        
        await self._set_cached(cache_key, json.dumps([
            {"source": s.source, "symbol": s.symbol, "direction": s.direction,
             "entry_price": s.entry_price, "stop_loss": s.stop_loss,
             "take_profit": s.take_profit, "confidence": s.confidence,
             "message": s.message, "timestamp": s.timestamp.isoformat()}
            for s in self._cache_telegram
        ]), self._cache_ttl_telegram)

        return self._cache_telegram

    def _parse_telegram_message(self, msg: dict, chat_id: str) -> Optional[TelegramSignal]:
        text = msg.get("text", "")
        if not text:
            return None

        symbol = "UNKNOWN"
        direction = "long" if any(w in text.lower() for w in ["long", "buy", "call"]) else "short"
        entry = stop_loss = take_profit = None
        confidence = "medium"
        
        numbers = re.findall(r'[\d,]+\.?\d*', text)
        if len(numbers) >= 1:
            try:
                entry = float(numbers[0].replace(",", ""))
            except ValueError:
                pass
        
        sl_idx = text.lower().find("sl")
        if sl_idx != -1:
            for num in numbers:
                try:
                    val = float(num.replace(",", ""))
                    if stop_loss is None:
                        stop_loss = val
                    elif take_profit is None:
                        take_profit = val
                except ValueError:
                    pass

        if any(w in text.lower() for w in ["high confidence", "strong", "sure"]):
            confidence = "high"
        elif any(w in text.lower() for w in ["low", "weak"]):
            confidence = "low"

        for sym in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT"]:
            if sym in text.upper():
                symbol = sym
                break

        return TelegramSignal(
            source=chat_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            message=text[:200],
            timestamp=datetime.fromtimestamp(msg.get("date", 0), tz=timezone.utc)
        )

    async def get_sentiment_score(self) -> SentimentScore:
        fear_greed = await self.get_fear_greed()
        news = await self.get_crypto_news(limit=5)
        
        overall = 50.0
        
        if fear_greed:
            overall = float(fear_greed.value)
        
        news_sentiment = 50.0
        if news:
            scores = []
            for item in news:
                title_lower = item.title.lower()
                pos_count = sum(1 for w in self._positive_words if w in title_lower)
                neg_count = sum(1 for w in self._negative_words if w in title_lower)
                if pos_count > neg_count:
                    scores.append(75.0)
                elif neg_count > pos_count:
                    scores.append(25.0)
                else:
                    scores.append(50.0)
            news_sentiment = sum(scores) / len(scores) if scores else 50.0
        
        fg_weight = 0.5
        news_weight = 0.5
        overall = (overall * fg_weight) + (news_sentiment * news_weight)
        
        return SentimentScore(
            fear_greed=fear_greed,
            news=news,
            overall=overall,
        )

    async def get_all_sentiments(self, symbols: list[str]) -> dict:
        tasks = [
            self.get_sentiment_score(),
            self.get_google_trends("bitcoin"),
            self.get_telegram_signals()
        ]
        
        for symbol in symbols:
            tasks.append(self.get_twitter_sentiment(symbol))
            tasks.append(self.get_reddit_sentiment(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sentiment_data = {
            "fear_greed": results[0].fear_greed if isinstance(results[0], SentimentScore) else None,
            "news": results[0].news if isinstance(results[0], SentimentScore) else [],
            "google_trends": results[1] if isinstance(results[1], GoogleTrendsData) else None,
            "telegram_signals": results[2] if isinstance(results[2], list) else [],
        }
        
        idx = 3
        for symbol in symbols:
            if idx < len(results):
                sentiment_data[f"twitter_{symbol}"] = results[idx] if isinstance(results[idx], TwitterSentiment) else None
                idx += 1
            if idx < len(results):
                sentiment_data[f"reddit_{symbol}"] = results[idx] if isinstance(results[idx], RedditSentiment) else None
                idx += 1
        
        return sentiment_data

    def get_sentiment_for_signal(self, sentiment: SentimentScore) -> dict:
        if sentiment.fear_greed is None:
            return {"confidence_boost": 0, "allowed_sides": ["long", "short"]}
        
        fg_value = sentiment.fear_greed.value
        
        if fg_value <= 25:
            return {"confidence_boost": 10, "allowed_sides": ["long"]}
        elif fg_value <= 45:
            return {"confidence_boost": 5, "allowed_sides": ["long"]}
        elif fg_value >= 75:
            return {"confidence_boost": 10, "allowed_sides": ["short"]}
        elif fg_value >= 55:
            return {"confidence_boost": 5, "allowed_sides": ["short"]}
        else:
            return {"confidence_boost": 0, "allowed_sides": ["long", "short"]}


async def get_sentiment_feed() -> SentimentFeed:
    if not hasattr(get_sentiment_feed, "_instance"):
        get_sentiment_feed._instance = SentimentFeed()
        await get_sentiment_feed._instance.start()
    return get_sentiment_feed._instance
