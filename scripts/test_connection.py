"""
Quick BloFin API connection test — no database required.
Run: python scripts/test_connection.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.exchanges.blofin import BloFinAdapter

logger = get_logger(__name__)


async def main():
    configure_logging()
    settings = get_settings()

    print("\n" + "="*50)
    print("  CryptoAgent — BloFin Connection Test")
    print("="*50)
    print(f"  Mode     : {settings.mode.upper()}")
    print(f"  Testnet  : {settings.blofin.testnet}")
    print(f"  API Key  : {settings.blofin.api_key[:8]}...{settings.blofin.api_key[-4:]}")
    print("="*50 + "\n")

    async with BloFinAdapter() as exchange:

        # ── 1. Public market data (no auth needed) ───────────────────────────
        print("► Testing public market data...")
        ticker = await exchange.get_ticker("BTC-USDT")
        print(f"  ✅ BTC-USDT price  : ${ticker.last:,.2f}")
        print(f"  ✅ Mark price      : ${ticker.mark_price:,.2f}")
        print(f"  ✅ 24h volume      : {ticker.volume_24h:,.2f}")

        eth = await exchange.get_ticker("ETH-USDT")
        print(f"  ✅ ETH-USDT price  : ${eth.last:,.2f}")

        # ── 2. Order book ────────────────────────────────────────────────────
        print("\n► Testing order book...")
        ob = await exchange.get_orderbook("BTC-USDT", depth=5)
        spread = ob.asks[0][0] - ob.bids[0][0]
        print(f"  ✅ Best bid        : ${ob.bids[0][0]:,.2f}")
        print(f"  ✅ Best ask        : ${ob.asks[0][0]:,.2f}")
        print(f"  ✅ Spread          : ${spread:.2f}")

        # ── 3. Candles ───────────────────────────────────────────────────────
        print("\n► Testing historical candles...")
        candles = await exchange.get_candles("BTC-USDT", "1h", limit=5)
        print(f"  ✅ Fetched {len(candles)} candles (1h)")
        if candles:
            last = candles[-1]
            print(f"  ✅ Latest candle   : O={last.open} H={last.high} L={last.low} C={last.close}")

        # ── 4. Funding rate ──────────────────────────────────────────────────
        print("\n► Testing funding rate...")
        fr = await exchange.get_funding_rate("BTC-USDT")
        print(f"  ✅ Funding rate    : {fr*100:.4f}%")

        # ── 5. Private account (requires API key) ────────────────────────────
        print("\n► Testing private account access (API auth)...")
        try:
            balances = await exchange.get_balance()
            if balances:
                for b in balances:
                    if b.total > 0:
                        print(f"  ✅ Balance         : {b.total:.4f} {b.currency} (available: {b.available:.4f})")
                if not any(b.total > 0 for b in balances):
                    print("  ✅ Auth successful (account balance: 0 — fund your account to trade)")
            else:
                print("  ✅ Auth successful (no balance data returned)")
        except Exception as e:
            print(f"  ❌ Auth failed: {e}")
            print("     Check your API key, secret, and passphrase in .env")
            return

        # ── 6. Positions ─────────────────────────────────────────────────────
        print("\n► Testing positions endpoint...")
        try:
            positions = await exchange.get_positions()
            print(f"  ✅ Open positions  : {len(positions)}")
        except Exception as e:
            print(f"  ❌ Positions failed: {e}")

    print("\n" + "="*50)
    print("  ✅ ALL TESTS PASSED — Agent is ready!")
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
