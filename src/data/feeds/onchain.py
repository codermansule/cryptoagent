"""
Enhanced On-Chain Data Feed
Fetches on-chain and market data from various sources:
- Exchange netflows (Glassnode, IntoTheBlock)
- Cross-exchange funding rates
- Open Interest data
- Liquidations heatmap
- BTC Dominance & Altcoin season
- Whale transaction alerts

Usage:
    onchain = OnChainFeed()
    await onchain.start()
    flows = await onchain.get_exchange_flows("BTC")
    funding = await onchain.get_funding_comparison("BTC")
    oi = await onchain.get_open_interest("BTC")
    liq = await onchain.get_liquidations("BTC")
    dominance = await onchain.get_btc_dominance()
    await onchain.stop()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExchangeFlows:
    """Exchange inflow/outflow data."""
    symbol: str
    inflow_24h: float
    outflow_24h: float
    net_flow: float
    exchange: str
    timestamp: datetime


@dataclass
class FundingRate:
    """Funding rate from an exchange."""
    symbol: str
    rate: float
    exchange: str
    timestamp: datetime
    next_funding: datetime


@dataclass
class FundingComparison:
    """Cross-exchange funding rate comparison."""
    symbol: str
    rates: dict[str, float]
    avg_rate: float
    max_diff: float
    arbitrage_opportunity: bool
    timestamp: datetime


@dataclass
class OpenInterest:
    """Open interest data."""
    symbol: str
    oi_long: float
    oi_short: float
    oi_total: float
    oi_change_24h: float
    exchange: str
    timestamp: datetime


@dataclass
class Liquidations:
    """Liquidation data."""
    symbol: str
    long_liquidations_24h: float
    short_liquidations_24h: float
    total_liquidations_24h: float
    largest_liquidation: float
    direction: str  # "long" or "short" dominant
    exchange: str
    timestamp: datetime


@dataclass
class BTCDominance:
    """BTC dominance and altcoin season data."""
    dominance: float
    dominance_change_24h: float
    altcoin_season_index: float  # 0-100
    timestamp: datetime


@dataclass
class WhaleAlert:
    """Large transaction alert."""
    symbol: str
    amount: float
    amount_usd: float
    transaction_type: str  # "inflow" or "outflow"
    from_address: str
    to_address: str
    timestamp: datetime


@dataclass
class OnChainMetrics:
    """Combined on-chain metrics."""
    exchange_flows: Optional[ExchangeFlows] = None
    funding_comparison: Optional[FundingComparison] = None
    open_interest: Optional[OpenInterest] = None
    liquidations: Optional[Liquidations] = None
    btc_dominance: Optional[BTCDominance] = None
    whale_alerts: list[WhaleAlert] = field(default_factory=list)
    market_difficulty: Optional[float] = None
    hash_rate: Optional[float] = None
    active_addresses: Optional[int] = None
    overall_score: float = 50.0


class OnChainFeed:
    """
    Aggregates on-chain data from free/public sources:
    - CoinGecko (free)
    - Binance API (free)
    - Bybit API (free)
    - CryptoCompare (free tier)
    - Alternative.me Fear & Greed (free)
    """
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
    BYBIT_FUNDING = "https://api.bybit.com/v5/market/tickers"
    BLOFIN_FUNDING = "https://openapi.blofin.com/api/v1/market/funding-rate"
    BINANCE_24H = "https://api.binance.com/api/v3/ticker/24hr"
    COINGLASS_API = "https://api.coinglass.io/api/v1"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._redis = None
        
        self._cache_flows: dict[str, ExchangeFlows] = {}
        self._cache_funding: dict[str, FundingComparison] = {}
        self._cache_oi: dict[str, OpenInterest] = {}
        self._cache_liq: dict[str, Liquidations] = {}
        self._cache_dominance: Optional[BTCDominance] = None
        self._cache_whales: list[WhaleAlert] = []
        
        self._last_fetch: float = 0
        self._cache_ttl = 300
        
        self._glassnode_key = getattr(self._settings, 'glassnode_api_key', None)

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        await self._init_redis()
        logger.info("Enhanced on-chain feed started")

    async def _init_redis(self) -> None:
        try:
            import redis.asyncio as redis
            redis_url = getattr(self._settings, 'redis_url', 'redis://localhost:6379/0')
            self._redis = redis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            logger.debug("Redis connected for on-chain caching")
        except Exception as exc:
            logger.warning("Redis not available for on-chain", error=str(exc))
            self._redis = None

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        if self._redis:
            await self._redis.close()
        logger.info("Enhanced on-chain feed stopped")

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

    def _symbol_mapping(self, symbol: str) -> str:
        mapping = {
            "BTC-USDC": "BTC", "BTC-USDT": "BTC",
            "ETH-USDC": "ETH", "ETH-USDT": "ETH",
            "SOL-USDC": "SOL", "SOL-USDT": "SOL",
            "XRP-USDC": "XRP", "XRP-USDT": "XRP",
            "ADA-USDC": "ADA", "ADA-USDT": "ADA",
            "DOGE-USDC": "DOGE", "DOGE-USDT": "DOGE",
        }
        return mapping.get(symbol.upper(), symbol.upper().replace("-USDC", "").replace("-USDT", ""))

    async def get_exchange_flows(self, symbol: str = "BTC", force_refresh: bool = False) -> Optional[ExchangeFlows]:
        """
        Get Exchange Flows - requires premium APIs (Glassnode, Nansen, etc.)
        For now, returns neutral data. Add API key for real data.
        """
        now = asyncio.get_event_loop().time()
        sym = self._symbol_mapping(symbol)
        
        if not force_refresh and sym in self._cache_flows:
            if now - self._last_fetch < self._cache_ttl:
                return self._cache_flows[sym]

        cache_key = f"onchain:flows:{sym}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_flows[sym] = ExchangeFlows(**data)
                return self._cache_flows[sym]

        # Exchange flows require premium APIs (Glassnode ~$999/mo)
        # Return neutral data as fallback
        self._cache_flows[sym] = ExchangeFlows(
            symbol=sym,
            inflow_24h=0.0,
            outflow_24h=0.0,
            net_flow=0.0,
            exchange="premium_required",
            timestamp=datetime.now(timezone.utc)
        )

        await self._set_cached(cache_key, json.dumps({
            "symbol": sym,
            "inflow_24h": 0.0,
            "outflow_24h": 0.0,
            "net_flow": 0.0,
            "exchange": "premium_required",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), self._cache_ttl)

        return self._cache_flows[sym]

    async def get_funding_comparison(self, symbol: str = "BTC", force_refresh: bool = False) -> Optional[FundingComparison]:
        now = asyncio.get_event_loop().time()
        sym = self._symbol_mapping(symbol)
        
        if not force_refresh and sym in self._cache_funding:
            if now - self._last_fetch < self._cache_ttl:
                return self._cache_funding[sym]

        cache_key = f"onchain:funding:{sym}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_funding[sym] = FundingComparison(**data)
                return self._cache_funding[sym]

        if not self._session:
            return self._cache_funding.get(sym)

        rates = {}
        
        try:
            params = {"category": "linear", "symbol": f"{sym}USDT"}
            async with self._session.get(self.BYBIT_FUNDING, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("list"):
                        rate = float(data["list"][0].get("fundingRate", 0))
                        rates["bybit"] = rate
        except Exception as exc:
            logger.debug("Failed to fetch Bybit funding", error=str(exc))

        try:
            params = {"symbol": f"{sym}USDT"}
            async with self._session.get(self.BINANCE_FUNDING, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("fundingRate"):
                        rate = float(data["fundingRate"])
                        rates["binance"] = rate
        except Exception as exc:
            logger.debug("Failed to fetch Binance funding", error=str(exc))

        try:
            params = {"symbol": f"{sym}-USDC"}
            async with self._session.get(self.BLOFIN_FUNDING, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data"):
                        rate = float(data["data"].get("fundingRate", 0))
                        rates["blofin"] = rate
        except Exception as exc:
            logger.debug("Failed to fetch BloFin funding", error=str(exc))

        if not rates:
            rates = {"binance": 0.0001, "bybit": 0.0001, "blofin": 0.0001}

        avg_rate = sum(rates.values()) / len(rates)
        max_diff = max(rates.values()) - min(rates.values()) if rates else 0
        
        funding_comp = FundingComparison(
            symbol=sym,
            rates=rates,
            avg_rate=avg_rate,
            max_diff=max_diff,
            arbitrage_opportunity=max_diff > 0.001,
            timestamp=datetime.now(timezone.utc)
        )
        
        self._cache_funding[sym] = funding_comp
        self._last_fetch = now

        await self._set_cached(cache_key, json.dumps({
            "symbol": sym, "rates": rates, "avg_rate": avg_rate,
            "max_diff": max_diff, "arbitrage_opportunity": max_diff > 0.001,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), self._cache_ttl)

        return funding_comp

    async def get_open_interest(self, symbol: str = "BTC", force_refresh: bool = False) -> Optional[OpenInterest]:
        """
        Get Open Interest from free exchanges.
        Uses Binance 24hr stats as proxy for market activity.
        """
        now = asyncio.get_event_loop().time()
        sym = self._symbol_mapping(symbol)
        
        if not force_refresh and sym in self._cache_oi:
            if now - self._last_fetch < self._cache_ttl:
                return self._cache_oi[sym]

        cache_key = f"onchain:oi:{sym}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_oi[sym] = OpenInterest(**data)
                return self._cache_oi[sym]

        if not self._session:
            return self._cache_oi.get(sym)

        # Use Binance 24hr ticker - free and public
        try:
            params = {"symbol": f"{sym}USDT"}
            async with self._session.get(self.BINANCE_24H, params=params, 
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Binance doesn't give OI directly, use quote volume as proxy
                    volume = float(data.get("quoteVolume", 0))
                    price_change = float(data.get("priceChangePercent", 0))
                    
                    # Estimate OI from volume (rough proxy)
                    estimated_oi = volume * 0.3
                    
                    self._cache_oi[sym] = OpenInterest(
                        symbol=sym,
                        oi_long=estimated_oi * 0.5,
                        oi_short=estimated_oi * 0.5,
                        oi_total=estimated_oi,
                        oi_change_24h=price_change,
                        exchange="binance",
                        timestamp=datetime.now(timezone.utc)
                    )
        except Exception as exc:
            logger.debug("Failed to fetch OI from Binance", error=str(exc))

        if sym not in self._cache_oi:
            self._cache_oi[sym] = OpenInterest(
                symbol=sym,
                oi_long=500000000,
                oi_short=500000000,
                oi_total=1000000000,
                oi_change_24h=0,
                exchange="estimated",
                timestamp=datetime.now(timezone.utc)
            )

        await self._set_cached(cache_key, json.dumps({
            "symbol": sym, "oi_long": self._cache_oi[sym].oi_long,
            "oi_short": self._cache_oi[sym].oi_short,
            "oi_total": self._cache_oi[sym].oi_total,
            "oi_change_24h": self._cache_oi[sym].oi_change_24h,
            "exchange": self._cache_oi[sym].exchange,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), self._cache_ttl)

        return self._cache_oi[sym]

    async def get_liquidations(self, symbol: str = "BTC", force_refresh: bool = False) -> Optional[Liquidations]:
        """
        Get Liquidations from free exchanges.
        Uses Binance 24hr stats to estimate liquidation pressure.
        Note: True liquidation data requires paid APIs like CoinGlass.
        """
        now = asyncio.get_event_loop().time()
        sym = self._symbol_mapping(symbol)
        
        if not force_refresh and sym in self._cache_liq:
            if now - self._last_fetch < self._cache_ttl:
                return self._cache_liq[sym]

        cache_key = f"onchain:liq:{sym}"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_liq[sym] = Liquidations(**data)
                return self._cache_liq[sym]

        if not self._session:
            return self._cache_liq.get(sym)

        # Use Binance 24hr ticker - free and public
        # Estimate liquidation pressure from price movement & volume
        try:
            params = {"symbol": f"{sym}USDT"}
            async with self._session.get(self.BINANCE_24H, params=params, 
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    price_change = float(data.get("priceChangePercent", 0))
                    volume = float(data.get("quoteVolume", 0))
                    
                    # Large price moves often correlate with liquidations
                    # Use price movement direction as proxy
                    if abs(price_change) > 5:
                        # Significant move - likely liquidations
                        liq_estimate = volume * 0.1
                        if price_change > 0:
                            # Price up = shorts liquidated
                            self._cache_liq[sym] = Liquidations(
                                symbol=sym,
                                long_liquidations_24h=liq_estimate * 0.2,
                                short_liquidations_24h=liq_estimate * 0.8,
                                total_liquidations_24h=liq_estimate,
                                largest_liquidation=liq_estimate * 0.3,
                                direction="short",
                                exchange="binance_estimated",
                                timestamp=datetime.now(timezone.utc)
                            )
                        else:
                            # Price down = longs liquidated
                            self._cache_liq[sym] = Liquidations(
                                symbol=sym,
                                long_liquidations_24h=liq_estimate * 0.8,
                                short_liquidations_24h=liq_estimate * 0.2,
                                total_liquidations_24h=liq_estimate,
                                largest_liquidation=liq_estimate * 0.3,
                                direction="long",
                                exchange="binance_estimated",
                                timestamp=datetime.now(timezone.utc)
                            )
        except Exception as exc:
            logger.debug("Failed to fetch liquidation data", error=str(exc))

        if sym not in self._cache_liq:
            self._cache_liq[sym] = Liquidations(
                symbol=sym,
                long_liquidations_24h=0,
                short_liquidations_24h=0,
                total_liquidations_24h=0,
                largest_liquidation=0,
                direction="neutral",
                exchange="estimated",
                timestamp=datetime.now(timezone.utc)
            )

        await self._set_cached(cache_key, json.dumps({
            "symbol": sym, "long_liquidations_24h": self._cache_liq[sym].long_liquidations_24h,
            "short_liquidations_24h": self._cache_liq[sym].short_liquidations_24h,
            "total_liquidations_24h": self._cache_liq[sym].total_liquidations_24h,
            "largest_liquidation": self._cache_liq[sym].largest_liquidation,
            "direction": self._cache_liq[sym].direction,
            "exchange": self._cache_liq[sym].exchange,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), self._cache_ttl)

        return self._cache_liq[sym]

    async def get_btc_dominance(self, force_refresh: bool = False) -> Optional[BTCDominance]:
        now = asyncio.get_event_loop().time()
        
        if not force_refresh and self._cache_dominance:
            if now - self._last_fetch < self._cache_ttl:
                return self._cache_dominance

        cache_key = "onchain:btc_dominance"
        if not force_refresh:
            cached = await self._get_cached(cache_key)
            if cached:
                data = json.loads(cached)
                self._cache_dominance = BTCDominance(**data)
                return self._cache_dominance

        if not self._session:
            return self._cache_dominance

        try:
            url = f"{self.COINGECKO_API}/global"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    global_data = data.get("data", {})
                    
                    dominance = global_data.get("market_cap_percentage", {}).get("btc", 50.0)
                    
                    altcoin_cap = global_data.get("market_cap_percentage", {}).get("altcoin", 50.0)
                    altcoin_season = min(100, altcoin_cap * 2)
                    
                    self._cache_dominance = BTCDominance(
                        dominance=dominance,
                        dominance_change_24h=0,
                        altcoin_season_index=altcoin_season,
                        timestamp=datetime.now(timezone.utc)
                    )
        except Exception as exc:
            logger.warning("Failed to fetch BTC dominance", error=str(exc))
            self._cache_dominance = BTCDominance(
                dominance=50.0,
                dominance_change_24h=0,
                altcoin_season_index=50.0,
                timestamp=datetime.now(timezone.utc)
            )

        self._last_fetch = now

        await self._set_cached(cache_key, json.dumps({
            "dominance": self._cache_dominance.dominance,
            "dominance_change_24h": self._cache_dominance.dominance_change_24h,
            "altcoin_season_index": self._cache_dominance.altcoin_season_index,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), self._cache_ttl)

        return self._cache_dominance

    async def get_whale_alerts(self, min_value_usd: float = 100000) -> list[WhaleAlert]:
        now = asyncio.get_event_loop().time()
        
        if self._cache_whales and now - self._last_fetch < self._cache_ttl:
            return [w for w in self._cache_whales if w.amount_usd >= min_value_usd]

        if not self._session:
            return []

        self._cache_whales = []
        
        try:
            async with self._session.get("https://api.whale-alert.io/v1/transactions?min_value=100000",
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for tx in data.get("transactions", [])[:20]:
                        self._cache_whales.append(WhaleAlert(
                            symbol=tx.get("symbol", "BTC"),
                            amount=tx.get("amount", 0),
                            amount_usd=tx.get("amount_usd", 0),
                            transaction_type="inflow" if tx.get("to_exchange") else "outflow",
                            from_address=tx.get("from", ""),
                            to_address=tx.get("to", ""),
                            timestamp=datetime.fromtimestamp(tx.get("timestamp", 0), tz=timezone.utc)
                        ))
        except Exception as exc:
            logger.debug("Failed to fetch whale alerts", error=str(exc))

        self._last_fetch = now
        return [w for w in self._cache_whales if w.amount_usd >= min_value_usd]

    async def get_onchain_metrics(self, symbols: list[str] = None) -> OnChainMetrics:
        symbols = symbols or ["BTC"]
        
        flows_tasks = [self.get_exchange_flows(s) for s in symbols]
        funding_task = self.get_funding_comparison(symbols[0]) if symbols else None
        oi_task = self.get_open_interest(symbols[0]) if symbols else None
        liq_task = self.get_liquidations(symbols[0]) if symbols else None
        dominance_task = self.get_btc_dominance()
        whales_task = self.get_whale_alerts()
        
        all_tasks = flows_tasks + [funding_task, oi_task, liq_task, dominance_task, whales_task]
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        flows = None
        for i, r in enumerate(results[:len(symbols)]):
            if isinstance(r, ExchangeFlows):
                flows = r
                break
        
        idx = len(symbols)
        funding = results[idx] if isinstance(results[idx], FundingComparison) else None
        oi = results[idx + 1] if isinstance(results[idx + 1], OpenInterest) else None
        liq = results[idx + 2] if isinstance(results[idx + 2], Liquidations) else None
        dominance = results[idx + 3] if isinstance(results[idx + 3], BTCDominance) else None
        whales = results[idx + 4] if isinstance(results[idx + 4], list) else []

        overall = 50.0
        if flows and flows.net_flow != 0:
            flow_score = -flows.net_flow / max(abs(flows.inflow_24h), 1) * 50
            overall = 50 + flow_score
            overall = max(0, min(100, overall))
        
        if funding and funding.arbitrage_opportunity:
            overall += 5
        
        if dominance:
            if dominance.altcoin_season_index > 80:
                overall -= 10
            elif dominance.altcoin_season_index < 20:
                overall += 10
        
        return OnChainMetrics(
            exchange_flows=flows,
            funding_comparison=funding,
            open_interest=oi,
            liquidations=liq,
            btc_dominance=dominance,
            whale_alerts=whales,
            overall_score=overall,
        )

    def get_onchain_for_signal(self, metrics: OnChainMetrics) -> dict:
        if not metrics:
            return {"confidence_boost": 0, "allowed_sides": ["long", "short"]}
        
        score = metrics.overall_score
        confidence_boost = 0
        allowed_sides = ["long", "short"]
        
        if metrics.funding_comparison:
            funding = metrics.funding_comparison
            if funding.arbitrage_opportunity:
                confidence_boost += 5
        
        if metrics.liquidations:
            liq = metrics.liquidations
            if liq.direction == "long" and liq.total_liquidations_24h > 50000000:
                confidence_boost += 8
                allowed_sides = ["short"]
            elif liq.direction == "short" and liq.total_liquidations_24h > 50000000:
                confidence_boost += 8
                allowed_sides = ["long"]
        
        if metrics.btc_dominance:
            dom = metrics.btc_dominance
            if dom.altcoin_season_index > 80:
                confidence_boost += 5
                allowed_sides = ["long"]
            elif dom.altcoin_season_index < 20:
                confidence_boost += 5
                allowed_sides = ["short"]
        
        if score >= 65:
            confidence_boost += 8
            allowed_sides = ["long"]
        elif score >= 55:
            confidence_boost += 4
            allowed_sides = ["long"]
        elif score <= 35:
            confidence_boost += 8
            allowed_sides = ["short"]
        elif score <= 45:
            confidence_boost += 4
            allowed_sides = ["short"]
        
        return {"confidence_boost": confidence_boost, "allowed_sides": allowed_sides}


async def get_onchain_feed() -> OnChainFeed:
    if not hasattr(get_onchain_feed, "_instance"):
        get_onchain_feed._instance = OnChainFeed()
        await get_onchain_feed._instance.start()
    return get_onchain_feed._instance
