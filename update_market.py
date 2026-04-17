"""
Daily market update script — run by GitHub Actions every day.
Fetches live BTC price from CoinGecko and appends to market_log.md.
"""
import requests
import json
import os
from datetime import datetime, timezone

def fetch_btc():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_market_cap": "true",
        "include_24hr_vol": "true"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["bitcoin"]

def main():
    try:
        data  = fetch_btc()
        price = data["usd"]
        chg   = data["usd_24h_change"]
        mcap  = data["usd_market_cap"]
        vol   = data["usd_24h_vol"]
        now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        trend = "🟢" if chg >= 0 else "🔴"

        log_path = os.path.join(os.path.dirname(__file__), "market_log.md")

        # Create file with header if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("# 📈 Daily BTC Market Log\n\n")
                f.write("> Auto-updated every day by GitHub Actions 🤖\n\n")
                f.write("| Date (UTC) | Price (USD) | 24h Change | Market Cap | Volume |\n")
                f.write("|-----------|-------------|------------|------------|--------|\n")

        entry = (
            f"| {now} | ${price:,.2f} | {trend} {chg:+.2f}% | "
            f"${mcap/1e9:.1f}B | ${vol/1e9:.1f}B |\n"
        )

        with open(log_path, "a") as f:
            f.write(entry)

        print(f"✅ Logged: {now} | ${price:,.2f} | {chg:+.2f}%")

    except Exception as e:
        print(f"❌ Error fetching BTC data: {e}")
        raise

if __name__ == "__main__":
    main()
