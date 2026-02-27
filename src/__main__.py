"""
Entry point for running CryptoAgent.
Usage:
    python -m src
    python src/agent.py
"""

from src.agent import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
