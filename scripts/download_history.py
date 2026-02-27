"""
CLI script: download historical OHLCV data for all configured symbols.

Usage:
    python scripts/download_history.py
    python scripts/download_history.py --symbols BTC-USDT ETH-USDT --candles 3000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.data.feeds.historical import download_history
from src.data.storage.schema import Database
from src.exchanges.blofin import BloFinAdapter

logger = get_logger(__name__)


async def main(symbols: list[str], timeframes: list[str], candles: int) -> None:
    configure_logging()
    settings = get_settings()

    async with BloFinAdapter() as exchange:
        db = Database()
        await db.connect()
        await db.initialize_schema()

        await download_history(
            exchange=exchange,
            db=db,
            symbols=symbols or None,
            timeframes=timeframes or None,
            total_candles=candles,
        )

        await db.disconnect()
    logger.info("History download complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("--symbols", nargs="*", default=[], help="Override symbols from config")
    parser.add_argument("--timeframes", nargs="*", default=[], help="Override timeframes")
    parser.add_argument("--candles", type=int, default=2000, help="Candles per symbol/tf pair")
    args = parser.parse_args()

    asyncio.run(main(args.symbols, args.timeframes, args.candles))
