
import urllib.request
import json
from datetime import datetime, timezone

def test_blofin():
    url = "https://openapi.blofin.com/api/v1/market/tickers?instId=BTC-USDC"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CryptoAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            print(f"Status Code: {data.get('code')}")
            if data.get('data'):
                print(f"Price: {data['data'][0].get('last')}")
                print(f"Timestamp: {data['data'][0].get('ts')}")
            else:
                print("No data returned")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_blofin()
