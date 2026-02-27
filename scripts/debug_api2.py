"""Debug round 2 — check orderbook, funding rate, open interest responses."""

import asyncio, sys, json
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

    print("\n--- TICKER (BTC-USDC via tickers endpoint) ---")
    r = await get(cfg["tickers"], {"instId": "BTC-USDC"})
    print(json.dumps(r, indent=2))

    print("\n--- ORDER BOOK ---")
    r = await get(cfg["orderbook"], {"instId": "BTC-USDC", "sz": "3"})
    print(json.dumps(r, indent=2))

    print("\n--- FUNDING RATE ---")
    r = await get(cfg["funding_rate"], {"instId": "BTC-USDC"})
    print(json.dumps(r, indent=2))

    print("\n--- OPEN INTEREST ---")
    r = await get(cfg["open_interest"], {"instId": "BTC-USDC"})
    print(json.dumps(r, indent=2))

asyncio.run(main())
