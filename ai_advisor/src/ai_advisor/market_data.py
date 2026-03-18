"""
ai_advisor/market_data.py

Fetches real-time market data for the approved stock/ETF universe.
Uses yfinance (free, no API key needed) for price data and performance.
Uses OpenAI web search for recent market news.

Install: pip install yfinance
"""

import json
import re
import time
from datetime import datetime, timedelta
import streamlit as st

from openai import OpenAI
from ai_advisor.stocks import APPROVED_STOCKS, APPROVED_ETFS, get_all_tickers


client = OpenAI()

@st.cache_data(ttl=3600)
def fetch_stock_data(tickers: list[str] | None = None) -> dict[str, dict]:
    """
    Fetch current price, recent performance, and key metrics for each ticker.
    Returns a dict like:
    {
        "AAPL": {
            "price": 195.23,
            "change_1d": -0.8,
            "change_1w": 2.1,
            "change_1m": -3.5,
            "change_3m": 5.2,
            "change_ytd": 12.3,
            "52w_high": 210.00,
            "52w_low": 150.00,
            "pe_ratio": 28.5,
            "dividend_yield": 0.55,
            "market_cap": "3.0T",
            "sector": "Technology",
            "name": "Apple Inc."
        },
        ...
    }
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  Warning: yfinance not installed. Run: pip install yfinance")
        return {}

    if tickers is None:
        tickers = get_all_tickers()

    data = {}
    now = datetime.now()

    # Download all tickers at once for efficiency
    try:
        # Get 3 months of daily history
        hist = yf.download(
            tickers,
            period="3mo",
            progress=False,
            threads=True,
        )

        # Get ticker info objects
        ticker_objs = {t: yf.Ticker(t) for t in tickers}

        for ticker in tickers:
            try:
                # Extract price history for this ticker
                if len(tickers) == 1:
                    close = hist["Close"]
                else:
                    close = hist["Close"][ticker]

                close = close.dropna()
                if close.empty:
                    continue

                current_price = float(close.iloc[-1])

                # Calculate returns
                def pct_change(days_ago):
                    if len(close) > days_ago:
                        old_price = float(close.iloc[-days_ago - 1])
                        if old_price > 0:
                            return round((current_price - old_price) / old_price * 100, 2)
                    return None

                change_1d = pct_change(1)
                change_1w = pct_change(5)
                change_1m = pct_change(21)
                change_3m = pct_change(63) if len(close) >= 64 else None

                # Get info with retry on rate limit
                info = {}
                for attempt in range(3):
                    try:
                        info = ticker_objs[ticker].info
                        break
                    except Exception as e:
                        if "Too Many Requests" in str(e) or "Rate" in str(e):
                            time.sleep(2 ** attempt * 2)  # 2s, 4s, 8s backoff
                        else:
                            raise

                # Prevent yfinance rate limiting
                time.sleep(0.5)

                # Format market cap
                mc = info.get("marketCap", 0)
                if mc >= 1e12:
                    mc_str = f"{mc/1e12:.1f}T"
                elif mc >= 1e9:
                    mc_str = f"{mc/1e9:.1f}B"
                elif mc >= 1e6:
                    mc_str = f"{mc/1e6:.0f}M"
                else:
                    mc_str = str(mc)

                data[ticker] = {
                    "price": round(current_price, 2),
                    "change_1d": change_1d,
                    "change_1w": change_1w,
                    "change_1m": change_1m,
                    "change_3m": change_3m,
                    "52w_high": round(float(info.get("fiftyTwoWeekHigh", 0)), 2),
                    "52w_low": round(float(info.get("fiftyTwoWeekLow", 0)), 2),
                    "pe_ratio": round(float(info.get("trailingPE", 0)), 1) if info.get("trailingPE") else None,
                    "dividend_yield": round(float(info.get("dividendYield", 0)) * 100, 2) if info.get("dividendYield") else 0,
                    "market_cap": mc_str,
                    "sector": info.get("sector", "N/A"),
                    "name": info.get("shortName", ticker),
                }

            except Exception as e:
                print(f"  Warning: Could not fetch data for {ticker}: {e}")
                continue

    except Exception as e:
        print(f"  Warning: yfinance download failed: {e}")
        return {}

    return data


def fetch_market_news() -> str:
    """
    Use OpenAI with web search to get a summary of recent market conditions.
    Returns a concise text block.
    """
    try:
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=(
                "Give me a brief summary of current stock market conditions as of today. "
                "Cover: 1) Overall market trend (S&P 500, Nasdaq direction), "
                "2) Key themes or sectors performing well/poorly, "
                "3) Any major recent events affecting markets (Fed decisions, earnings, "
                "geopolitical events). Keep it under 200 words. Facts only, no opinions."
            ),
        )

        # Extract text from response
        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        return block.text

        return "Market news unavailable."

    except Exception as e:
        print(f"  Warning: Could not fetch market news: {e}")
        # Fallback: try chat completions without web search
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": (
                        "Based on your training data, what are the most recent market "
                        "conditions and trends you're aware of? Note any limitations "
                        "in data recency. Keep it under 150 words."
                    ),
                }],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Market news unavailable."


def format_market_data_for_ai(stock_data: dict[str, dict], news: str) -> str:
    """
    Format all market data into a concise text block for the AI prompt.
    """
    lines = []

    lines.append("=== CURRENT MARKET CONDITIONS ===")
    lines.append(news)
    lines.append("")

    lines.append("=== REAL-TIME STOCK/ETF DATA ===")
    lines.append(f"Data as of: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Group by type
    stock_tickers = {s["ticker"] for s in APPROVED_STOCKS}
    etf_tickers = {e["ticker"] for e in APPROVED_ETFS}

    # Stocks
    lines.append("STOCKS:")
    for ticker in sorted(stock_data.keys()):
        if ticker in stock_tickers:
            d = stock_data[ticker]
            perf_parts = []
            if d.get("change_1d") is not None:
                perf_parts.append(f"1d:{d['change_1d']:+.1f}%")
            if d.get("change_1w") is not None:
                perf_parts.append(f"1w:{d['change_1w']:+.1f}%")
            if d.get("change_1m") is not None:
                perf_parts.append(f"1m:{d['change_1m']:+.1f}%")
            if d.get("change_3m") is not None:
                perf_parts.append(f"3m:{d['change_3m']:+.1f}%")
            perf = " | ".join(perf_parts)

            extra = []
            if d.get("pe_ratio"):
                extra.append(f"P/E:{d['pe_ratio']}")
            if d.get("dividend_yield"):
                extra.append(f"Div:{d['dividend_yield']:.1f}%")
            if d.get("sector") and d["sector"] != "N/A":
                extra.append(d["sector"])
            extra_str = f" [{', '.join(extra)}]" if extra else ""

            lines.append(f"  {ticker} ({d['name']}): ${d['price']:.2f} | {perf}{extra_str}")

    lines.append("")

    # ETFs
    lines.append("ETFs:")
    for ticker in sorted(stock_data.keys()):
        if ticker in etf_tickers:
            d = stock_data[ticker]
            perf_parts = []
            if d.get("change_1d") is not None:
                perf_parts.append(f"1d:{d['change_1d']:+.1f}%")
            if d.get("change_1w") is not None:
                perf_parts.append(f"1w:{d['change_1w']:+.1f}%")
            if d.get("change_1m") is not None:
                perf_parts.append(f"1m:{d['change_1m']:+.1f}%")
            if d.get("change_3m") is not None:
                perf_parts.append(f"3m:{d['change_3m']:+.1f}%")
            perf = " | ".join(perf_parts)

            lines.append(f"  {ticker} ({d['name']}): ${d['price']:.2f} | {perf}")

    return "\n".join(lines)


def get_market_context() -> str:
    """
    Main entry point: fetch all market data and news, return formatted text.
    Call this once at the start of the pipeline.
    """
    print("  Fetching real-time market data...")
    stock_data = fetch_stock_data()

    print("  Fetching market news...")
    news = fetch_market_news()

    if not stock_data:
        return f"=== MARKET NEWS ===\n{news}\n\n(Real-time price data unavailable — yfinance may not be installed)"

    return format_market_data_for_ai(stock_data, news)
