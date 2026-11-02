
import json
import math
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

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


def finite(value):
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError):
        return False


def extract_ticker_data(data, ticker):
    """Handle both flat and MultiIndex Yahoo Finance columns."""

    if data is None or data.empty:
        return None

    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()

    for level in range(data.columns.nlevels):
        values = data.columns.get_level_values(level)

        if ticker in values:
            result = data.xs(ticker, axis=1, level=level)
            if not result.empty:
                return result.copy()

    return None


def calculate_rsi(close):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    if not finite(avg_gain.iloc[-1]):
        return None

    g = float(avg_gain.iloc[-1])
    l = float(avg_loss.iloc[-1])

    if l == 0:
        return 100.0 if g > 0 else 50.0

    rs = g / l
    return 100 - (100 / (1 + rs))


def analyse_stock(ticker, df):
    try:
        if df is None or df.empty:
            raise ValueError("No Yahoo Finance data")

        df.columns = [
            str(c).strip().title() for c in df.columns
        ]

        required = ["Close", "High", "Volume"]

        if not all(c in df.columns for c in required):
            raise ValueError("Missing OHLCV columns")

        df = df[required].copy()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([float("inf"), float("-inf")], float("nan"))
        df = df.dropna(subset=["Close", "High"])

        if len(df) < 50:
            raise ValueError("Insufficient historical data")

        close = df["Close"]
        high = df["High"]
        volume = df["Volume"].fillna(0)

        price = float(close.iloc[-1])
        previous = float(close.iloc[-2])

        if not finite(price) or price <= 0 or previous <= 0:
            raise ValueError("Invalid closing price")

        ema20 = float(
            close.ewm(span=20, adjust=False).mean().iloc[-1]
        )

        ema50 = float(
            close.ewm(span=50, adjust=False).mean().iloc[-1]
        )

        avg_volume = float(volume.iloc[-21:-1].mean())
        current_volume = float(volume.iloc[-1])

        volume_ratio = (
            current_volume / avg_volume
            if avg_volume > 0 else 0
        )

        previous_high = float(high.iloc[-21:-1].max())

        return_5d = (
            price / float(close.iloc[-6]) - 1
        ) * 100

        return_20d = (
            price / float(close.iloc[-21]) - 1
        ) * 100

        daily_change = (price / previous - 1) * 100

        rsi = calculate_rsi(close)

        if rsi is None or not all(map(
            finite, [
                ema20, ema50, volume_ratio,
                return_5d, return_20d,
                daily_change, previous_high, rsi
            ]
        )):
            raise ValueError("Invalid technical indicators")

        trend_score = (
            25 if price > ema20 > ema50
            else 15 if price > ema20
            else 0
        )

        momentum_score = min(20, max(0, return_5d * 2))

        volume_score = min(
            20, max(0, (volume_ratio - 1) * 12)
        )

        breakout_score = (
            20 if price > previous_high
            else 10 if price >= previous_high * 0.98
            else 0
        )

        strength_score = min(15, max(0, return_20d))

        score = round(
            trend_score + momentum_score +
            volume_score + breakout_score +
            strength_score, 1
        )

        signal = (
            "Strong watch" if score >= 70
            else "Watch" if score >= 50
            else "Low priority"
        )

        return {
            "symbol": ticker.replace(".NS", ""),
            "price": round(price, 2),
            "daily_change_pct": round(daily_change, 2),
            "return_5d_pct": round(return_5d, 2),
            "return_20d_pct": round(return_20d, 2),
            "volume_ratio": round(volume_ratio, 2),
            "rsi": round(rsi, 1),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "score": score,
            "signal": signal
        }

    except Exception as error:
        print(f"Skipped {ticker}: {error}")
        return None


def main():
    results = []
    failed = []

    print("Downloading NSE market data...")

    try:
        market_data = yf.download(
            tickers=TICKERS,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
            timeout=30
        )
    except Exception as error:
        print("Batch download failed:", error)
        market_data = pd.DataFrame()

    for ticker in TICKERS:
        print("Analysing:", ticker)

        try:
            stock_data = extract_ticker_data(
                market_data, ticker
            )

            if stock_data is None or stock_data.empty:
                # Fallback for stocks missing in batch response
                stock_data = yf.Ticker(ticker).history(
                    period="6mo",
                    interval="1d",
                    auto_adjust=True,
                    timeout=20
                )

            result = analyse_stock(ticker, stock_data)

            if result:
                results.append(result)
            else:
                failed.append({
                    "symbol": ticker.replace(".NS", ""),
                    "reason": "No valid analysis produced"
                })

        except Exception as error:
            failed.append({
                "symbol": ticker.replace(".NS", ""),
                "reason": str(error)
            })

        time.sleep(0.15)

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
            "Not live or pre-market data."
        ),

        "stocks_scanned": len(TICKERS),
        "stocks_found": len(results),
        "stocks_failed": len(failed),

        "results": results,
        "failed": failed
    }

    with open("scanner.json", "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            allow_nan=False
        )

    print("--------------------------------")
    print("SCANNER COMPLETED")
    print("Scanned:", len(TICKERS))
    print("Successful:", len(results))
    print("Failed:", len(failed))
    print("--------------------------------")


if __name__ == "__main__":
    main()
