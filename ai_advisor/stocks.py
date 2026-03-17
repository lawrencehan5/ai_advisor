"""
Approved securities universe for the AI Financial Advisor.
Edit this file to add/remove stocks and ETFs from the recommendation pool.
"""

APPROVED_STOCKS = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corp."},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "AMZN", "name": "Amazon.com Inc."},
    {"ticker": "JNJ", "name": "Johnson & Johnson"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co."},
    {"ticker": "PG", "name": "Procter & Gamble Co."},
    {"ticker": "KO", "name": "Coca-Cola Co."},
    {"ticker": "PFE", "name": "Pfizer Inc."},
    {"ticker": "XOM", "name": "Exxon Mobil Corp."},
    {"ticker": "BRK.B", "name": "Berkshire Hathaway Inc."},
    {"ticker": "UNH", "name": "UnitedHealth Group Inc."},
    {"ticker": "HD", "name": "Home Depot Inc."},
    {"ticker": "DIS", "name": "Walt Disney Co."},
    {"ticker": "NVDA", "name": "NVIDIA Corp."},
]

APPROVED_ETFS = [
    {"ticker": "VOO", "name": "Vanguard S&P 500 ETF"},
    {"ticker": "BND", "name": "Vanguard Total Bond Market ETF"},
    {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF"},
    {"ticker": "VXUS", "name": "Vanguard Total International Stock ETF"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust - Nasdaq 100"},
    {"ticker": "VNQ", "name": "Vanguard Real Estate ETF"},
    {"ticker": "GLD", "name": "SPDR Gold Shares"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF"},
    {"ticker": "AGG", "name": "iShares Core US Aggregate Bond ETF"},
    {"ticker": "SCHD", "name": "Schwab US Dividend Equity ETF"},
    {"ticker": "VIG", "name": "Vanguard Dividend Appreciation ETF"},
    {"ticker": "IWM", "name": "iShares Russell 2000 ETF"},
    {"ticker": "EFA", "name": "iShares MSCI EAFE ETF"},
    {"ticker": "LQD", "name": "iShares Investment Grade Corporate Bond ETF"},
    {"ticker": "HYG", "name": "iShares iBoxx High Yield Corporate Bond ETF"},
]


def get_approved_universe_text() -> str:
    """Format the approved list as text for injection into task prompts."""
    lines = ["APPROVED STOCKS:"]
    for s in APPROVED_STOCKS:
        lines.append(f"  - {s['ticker']} ({s['name']})")
    lines.append("")
    lines.append("APPROVED ETFs:")
    for e in APPROVED_ETFS:
        lines.append(f"  - {e['ticker']} ({e['name']})")
    return "\n".join(lines)


def get_all_tickers() -> list[str]:
    """Return a flat list of all approved tickers."""
    return [s["ticker"] for s in APPROVED_STOCKS] + [e["ticker"] for e in APPROVED_ETFS]
