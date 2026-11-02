
import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# Initial NSE stock universe. We can expand this later.
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS",
    "TECHM.NS", "LT.NS", "BHARTIARTL.NS",
    "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS",
    "BEL.NS", "HAL.NS", "BHEL.NS",
    "TRENT.NS", "ETERNAL.NS", "TITAN.NS",
    "SUNPHARMA.NS", "CIPLA.NS", "ITC.NS",
    "ONGC.NS", "POWERGRID.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "DLF.NS", "PERSISTENT.NS",
    "POLYCAB.NS"
]


def analyse_stock(ticker):
    try:
        df = yf.Ticker(ticker).history(
            period="6mo",
            interval="1d",
            auto_adjust=True
        )

        if df.empty or len(df) < 50:
            return None

        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]

        price = float(close.iloc[-1])
        previous = float(close.iloc[-2])

        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        ema50 = float(close.ewm(span=50).mean().iloc[-1])

        avg_volume = float(volume.iloc[-21:-1].mean())
        volume_ratio = (
            float(volume.iloc[-1] / avg_volume)
            if avg_volume > 0 else 0
        )

        previous_high = float(
            high.iloc[-21:-1].max()
        )

        return_5d = (
            (price / float(close.iloc[-6]) - 1) * 100
        )

        return_20d = (
            (price / float(close.iloc[-21]) - 1) * 100
        )

        # RSI calculation
        delta = close.diff()
        gains = delta.clip(lower=0).rolling(14).mean()
        losses = -delta.clip(upper=0).rolling(14).mean()
        rs = gains.iloc[-1] / max(losses.iloc[-1], 0.000001)
        rsi = float(100 - (100 / (1 + rs)))

        # Transparent scoring model (0–100)
        trend_score = 25 if price > ema20 > ema50 else (
            15 if price > ema20 else 0
        )

        momentum_score = min(
            20, max(0, return_5d * 2)
        )

        volume_score = min(
            20, max(0, (volume_ratio - 1) * 12)
        )

        breakout_score = (
            20 if price > previous_high else
            10 if price >= previous_high * 0.98 else 0
        )

        strength_score = min(
            15, max(0, return_20d)
        )

        score = round(
            trend_score + momentum_score +
            volume_score + breakout_score +
            strength_score, 1
        )

        if score >= 70:
            signal = "Strong watch"
        elif score >= 50:
            signal = "Watch"
        else:
            signal = "Low priority"

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(price, 2),
            "daily_change_pct": round(
                (price / previous - 1) * 100, 2
            ),
            "return_5d_pct": round(return_5d, 2),
            "return_20d_pct": round(return_20d, 2),
            "volume_ratio": round(volume_ratio, 2),
            "rsi": round(rsi, 1),
            "score": score,
            "signal": signal
        }

    except Exception as error:
        print(f"Skipped {ticker}: {error}")
        return None


def main():
    results = []

    for ticker in TICKERS:
        print("Scanning", ticker)
        result = analyse_stock(ticker)

        if result:
            results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    output = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_note": (
            "Unofficial Yahoo Finance daily data. "
            "Not a live or pre-market feed."
        ),
        "stocks_scanned": len(TICKERS),
        "stocks_found": len(results),
        "results": results
    }

    with open("scanner.json", "w") as file:
        json.dump(output, file, indent=2)

    print("Scanner completed:", len(results), "stocks")


if __name__ == "__main__":
    main()
