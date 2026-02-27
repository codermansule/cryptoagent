"""
Historical data downloader.
Fetches full OHLCV history for all configured symbols and timeframes,
then bulk-inserts into TimescaleDB.
"""

from __future__ import annotations

import asyncio

from src.core.config import get_settings
from src.core.logging import get_logger
from src.data.storage.schema import Database
from src.exchanges.base import BaseExchange

logger = get_logger(__name__)

# Max candles per request for BloFin
BATCH_SIZE = 300


async def download_history(
    exchange: BaseExchange,
    db: Database,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    total_candles: int = 2000,
) -> None:
    """
    Download historical OHLCV data and store in TimescaleDB.
    Paginates backwards until total_candles are fetched per symbol/tf pair.
    """
    settings = get_settings()
    symbols = symbols or settings.symbols
    timeframes = timeframes or settings.timeframes

    for symbol in symbols:
        for tf in timeframes:
            logger.info("Downloading history", symbol=symbol, timeframe=tf)
            all_candles = []
            since: int | None = None
            fetched = 0

            while fetched < total_candles:
                batch = await exchange.get_candles(
                    symbol, tf, limit=BATCH_SIZE, since=since
                )
                if not batch:
                    break

                all_candles.extend(batch)
                fetched += len(batch)

                # Paginate: use earliest timestamp as cursor
                since = batch[0].timestamp - 1
                logger.debug(
                    "Fetched batch",
                    symbol=symbol, tf=tf,
                    count=len(batch), total=fetched,
                )

                await asyncio.sleep(0.1)  # respect rate limit

            if all_candles:
                await db.insert_candles(all_candles)
                logger.info(
                    "History stored",
                    symbol=symbol, timeframe=tf, candles=len(all_candles),
                )
