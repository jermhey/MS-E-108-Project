"""Fetch MLB and election trade data for AS backtest notebooks."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kalshi_client import KalshiClient
import pandas as pd
import numpy as np

client = KalshiClient()

def fetch_series(tickers, cache_dir, series_name):
    os.makedirs(cache_dir, exist_ok=True)
    for ticker in tickers:
        path = os.path.join(cache_dir, f"{ticker}_trades.parquet")
        if os.path.exists(path):
            continue
        all_trades = []
        cursor = None
        for _ in range(100):
            trades, cursor = client.get_market_trades(ticker, limit=1000, cursor=cursor)
            all_trades.extend(trades)
            if not cursor or not trades:
                break
        if not all_trades:
            continue
        df = pd.DataFrame(all_trades)
        ts_col = "created_time" if "created_time" in df.columns else "ts"
        df["timestamp"] = pd.to_datetime(df[ts_col], format="mixed")
        df["mid_price"] = (df.get("yes_price", df.get("price", 50)) / 100.0)
        df["logit"] = np.log(np.clip(df["mid_price"], 0.01, 0.99) / (1 - np.clip(df["mid_price"], 0.01, 0.99)))
        df["volume"] = df.get("count", df.get("quantity", 1))
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(path)
        print(f"  {ticker}: {len(df):,}")
        time.sleep(0.02)
    print(f"{series_name}: {len([f for f in os.listdir(cache_dir) if f.endswith('.parquet')])} files")

if __name__ == "__main__":
    # MLB - 40 tickers from games with 50K+ volume
    evs = client.get_all_events(status="settled", series_ticker="KXMLBGAME")
    mlb_tickers = []
    for e in evs:
        if len(mlb_tickers) >= 40:
            break
        full = client.get_event(e["event_ticker"])
        mkts = full.get("markets", [])
        tot = sum(m.get("volume", 0) for m in mkts)
        if tot >= 50000:
            for m in mkts:
                mlb_tickers.append(m["ticker"])
                if len(mlb_tickers) >= 40:
                    break
    print("Fetching MLB...")
    fetch_series(mlb_tickers, "cache/kalshi_hf_mlb", "MLB")
