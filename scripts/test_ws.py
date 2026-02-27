"""
Minimal WebSocket diagnostic — shows exactly what BloFin sends back
before closing the connection.
"""
import asyncio
import json
import aiohttp

WS_URL = "wss://openapi.blofin.com/ws/public"

TESTS = [
    ("candle only",      [{"channel": "candle15m", "instId": "BTC-USDC"}]),
    ("books5 only",      [{"channel": "books5",    "instId": "BTC-USDC"}]),
    ("trades only",      [{"channel": "trades",    "instId": "BTC-USDC"}]),
    ("all 3 at once",    [
        {"channel": "candle15m", "instId": "BTC-USDC"},
        {"channel": "books5",   "instId": "BTC-USDC"},
        {"channel": "trades",   "instId": "BTC-USDC"},
    ]),
]

async def test_subscription(label: str, args: list):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  Sub args: {args}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL, timeout=aiohttp.ClientWSTimeout(ws_close=5)) as ws:
                await ws.send_str(json.dumps({"op": "subscribe", "args": args}))
                print("  Sent subscribe. Waiting for messages (5s)...")
                count = 0
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            print(f"  MSG [{count}]: {json.dumps(data)[:300]}")
                            count += 1
                            if count >= 5:
                                break
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            print(f"  CLOSE frame received: code={ws.close_code}")
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            print(f"  ERROR frame received")
                            break
                        elif msg.type == aiohttp.WSMsgType.PING:
                            print(f"  PING received, sending PONG")
                            await ws.pong(msg.data)
                except asyncio.TimeoutError:
                    print(f"  Timeout waiting for messages (got {count} messages)")
                except Exception as e:
                    print(f"  Exception in loop: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  Connection failed: {type(e).__name__}: {e}")

async def main():
    for label, args in TESTS:
        await test_subscription(label, args)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
