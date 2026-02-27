"""Quick asyncpg direct connection test."""
import asyncio, sys, os
from pathlib import Path
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncpg

async def main():
    print("Connecting to TimescaleDB...")
    # Try with password first
    try:
        conn = await asyncpg.connect(
            host="localhost", port=5432,
            user="agent", password="password",
            database="cryptoagent"
        )
        result = await conn.fetchval("SELECT version()")
        print(f"OK: {result[:60]}")
        await conn.close()
    except Exception as e:
        print(f"FAIL with password: {e}")
        # Try without password
        try:
            conn = await asyncpg.connect(
                host="localhost", port=5432,
                user="agent", password="",
                database="cryptoagent"
            )
            result = await conn.fetchval("SELECT version()")
            print(f"OK (no password): {result[:60]}")
            await conn.close()
        except Exception as e2:
            print(f"FAIL without password too: {e2}")

asyncio.run(main())
