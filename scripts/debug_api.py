"""Debug script — prints raw BloFin API responses so we can fix the parser."""

import asyncio, sys, os, json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp
from src.core.config import get_settings

BASE = "https://openapi.blofin.com"

async def get(path, params=None):
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE}{path}", params=params) as r:
            return await r.json()

async def main():
    cfg = get_settings().exchange_config["blofin"]["endpoints"]

    print("\n--- INSTRUMENTS (first 2) ---")
    r = await get(cfg["instruments"], {"instType": "SWAP"})
    print(json.dumps(r["data"][:2] if "data" in r else r, indent=2))

    print("\n--- TICKER ---")
    r = await get(cfg["ticker"], {"instId": "BTC-USDT"})
    print(json.dumps(r, indent=2))

    print("\n--- TICKERS (all, first entry) ---")
    r = await get(cfg["tickers"])
    first = r["data"][0] if "data" in r else r
    print(json.dumps(first, indent=2))

    print("\n--- CANDLES (1 candle) ---")
    r = await get(cfg["candles"], {"instId": "BTC-USDT", "bar": "1H", "limit": "1"})
    print(json.dumps(r, indent=2))

asyncio.run(main())
