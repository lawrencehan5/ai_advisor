"""
ai_advisor/optimizer.py

Bridge between the AI agents and your CPLEX portfolio optimizer.
The AI picks a strategy → this module calls your optimizer → returns optimized weights.

INTEGRATION INSTRUCTIONS:
    Replace the placeholder functions below with calls to your actual CPLEX optimizer.
    The interface is designed so you only need to edit the body of `run_optimization()`.
"""

from dataclasses import dataclass, field
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from ai_advisor.stocks import APPROVED_STOCKS, APPROVED_ETFS, get_all_tickers


# ── Strategy Definitions ────────────────────────────────────────────────────

# Maps strategy names to optimizer method + parameter hints.
# These correspond to your already-implemented CPLEX strategies.

STRATEGIES = {
    "buy_and_hold": {
        "display_name": "Buy and Hold",
        "description": "Long-term static allocation, minimal rebalancing.",
        "risk_level": "varies",
    },
    "equally_weighted": {
        "display_name": "Equally Weighted",
        "description": "Equal allocation across all selected assets.",
        "risk_level": "moderate",
    },
    "minimum_variance": {
        "display_name": "Minimum Variance",
        "description": "Minimize portfolio volatility. Best for conservative investors.",
        "risk_level": "conservative",
    },
    "max_expected_return": {
        "display_name": "Maximum Expected Return",
        "description": "Maximize expected return regardless of risk. For aggressive investors.",
        "risk_level": "aggressive",
    },
    "max_sharpe_ratio": {
        "display_name": "Maximum Sharpe Ratio",
        "description": "Maximize risk-adjusted return. Best balance of risk and reward.",
        "risk_level": "balanced",
    },
    "equal_risk_contribution": {
        "display_name": "Equal Risk Contributions",
        "description": "Each asset contributes equally to total portfolio risk.",
        "risk_level": "moderate",
    },
    "leveraged_max_sharpe": {
        "display_name": "Leveraged Max Sharpe Ratio",
        "description": "Sharpe-optimal portfolio with leverage. For aggressive/experienced investors.",
        "risk_level": "aggressive",
    },
    "robust_mean_variance": {
        "display_name": "Robust Mean-Variance Optimization",
        "description": "Mean-variance optimization with parameter uncertainty. More stable allocations.",
        "risk_level": "moderate",
    },
}

# Maps the AI's risk category → recommended optimizer strategies (ordered by preference)
RISK_TO_STRATEGIES = {
    "CONSERVATIVE": ["minimum_variance", "equally_weighted", "buy_and_hold"],
    "MODERATE":     ["equal_risk_contribution", "robust_mean_variance", "equally_weighted"],
    "BALANCED":     ["max_sharpe_ratio", "robust_mean_variance", "equal_risk_contribution"],
    "GROWTH":       ["max_sharpe_ratio", "max_expected_return", "robust_mean_variance"],
    "AGGRESSIVE":   ["max_expected_return", "leveraged_max_sharpe", "max_sharpe_ratio"],
}


# ── Optimization Result ─────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Structured output from the optimizer."""
    strategy_used: str = ""
    strategy_display_name: str = ""
    success: bool = False
    allocations: list[dict] = field(default_factory=list)  # [{"ticker": "VOO", "weight": 0.30}, ...]
    expected_return: float = 0.0       # annualized
    expected_volatility: float = 0.0   # annualized
    sharpe_ratio: float = 0.0
    metadata: dict = field(default_factory=dict)  # any extra info from optimizer
    error: str = ""


# ── Strategy Selection ──────────────────────────────────────────────────────

def select_strategy(risk_category: str, user_preference: str = "") -> str:
    """
    Given the AI's risk category (and optional user preference),
    return the optimizer strategy key to use.

    Args:
        risk_category: One of CONSERVATIVE, MODERATE, BALANCED, GROWTH, AGGRESSIVE
        user_preference: Optional - if user explicitly asked for a strategy

    Returns:
        Strategy key from STRATEGIES dict
    """
    # If user explicitly requested a strategy, try to match it
    if user_preference:
        pref_lower = user_preference.lower().replace(" ", "_").replace("-", "_")
        for key in STRATEGIES:
            if pref_lower in key or key in pref_lower:
                return key

    # Otherwise map from risk category
    category = risk_category.upper()
    candidates = RISK_TO_STRATEGIES.get(category, ["max_sharpe_ratio"])
    return candidates[0]  # Return top recommendation


def select_assets(risk_category: str) -> list[str]:
    """
    Select which tickers from the approved universe to include in optimization.
    Conservative strategies get more ETFs/bonds; aggressive gets more individual stocks.

    Returns:
        List of ticker strings to optimize over.
    """
    all_stock_tickers = [s["ticker"] for s in APPROVED_STOCKS]
    all_etf_tickers = [e["ticker"] for e in APPROVED_ETFS]

    bond_etfs = ["BND", "AGG", "TLT", "LQD", "HYG"]
    equity_etfs = ["VOO", "VTI", "QQQ", "VXUS", "SCHD", "VIG", "IWM", "EFA"]
    alternative_etfs = ["VNQ", "GLD"]
    defensive_stocks = ["JNJ", "PG", "KO", "PFE", "UNH", "BRK.B"]
    growth_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "HD", "JPM", "XOM", "DIS"]

    category = risk_category.upper()

    if category == "CONSERVATIVE":
        return bond_etfs + ["VOO", "SCHD", "VIG"] + defensive_stocks[:3]
    elif category == "MODERATE":
        return bond_etfs[:3] + equity_etfs[:4] + alternative_etfs[:1] + defensive_stocks[:4]
    elif category == "BALANCED":
        return bond_etfs[:2] + equity_etfs + alternative_etfs + defensive_stocks[:3] + growth_stocks[:3]
    elif category == "GROWTH":
        return equity_etfs + alternative_etfs + growth_stocks + defensive_stocks[:2]
    elif category == "AGGRESSIVE":
        return equity_etfs[:4] + growth_stocks + alternative_etfs
    else:
        # Fallback: use everything
        return get_all_tickers()


# ── Price Data Fetching ─────────────────────────────────────────────────────

def fetch_price_data(tickers: list[str]) -> pd.DataFrame:
    """
    Download historical adjusted close prices for the given tickers.
    Attempts 5 years first; falls back to 3 years if any ticker has
    less than 3 years of data (e.g. recent IPOs).

    Returns:
        DataFrame of daily adjusted close prices, columns = tickers.
        Tickers that fail entirely are dropped and a warning is printed.
    """
    end = date.today()
    start_5y = end - timedelta(days=5 * 365)
    start_3y = end - timedelta(days=3 * 365)

    # Try 5-year window first
    raw = yf.download(tickers, start=start_5y, end=end, auto_adjust=True, progress=False)["Close"]

    # yfinance returns a Series (not DataFrame) when only 1 ticker
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    # Check which tickers have at least 3 years of non-NaN rows (~756 trading days)
    min_rows = 3 * 252
    sufficient = [col for col in raw.columns if raw[col].notna().sum() >= min_rows]
    short_history = [col for col in raw.columns if col not in sufficient]

    if short_history:
        print(f"  [yfinance] {short_history} have < 3y data — retrying on 3y window")
        raw_3y = yf.download(short_history, start=start_3y, end=end,
                             auto_adjust=True, progress=False)["Close"]
        if isinstance(raw_3y, pd.Series):
            raw_3y = raw_3y.to_frame()
        raw = raw[sufficient].join(raw_3y, how="outer")

    # Drop tickers that are entirely NaN (delistings, bad symbols)
    raw = raw.dropna(axis=1, how="all")
    failed = [t for t in tickers if t not in raw.columns]
    if failed:
        print(f"  [yfinance] Could not fetch data for: {failed} — dropping from universe")

    return raw


# ── CPLEX Optimizer Interface ───────────────────────────────────────────────

def run_optimization(
    strategy: str,
    tickers: list[str],
    investment_amount: float = 10000.0,
) -> OptimizationResult:
    """
    Run the CPLEX portfolio optimization.

    ╔══════════════════════════════════════════════════════════════╗
    ║  PLUG IN YOUR CPLEX OPTIMIZER HERE                         ║
    ║                                                            ║
    ║  Replace the placeholder code below with calls to your     ║
    ║  actual optimizer. The strategy parameter tells you which   ║
    ║  optimization method to run.                               ║
    ╚══════════════════════════════════════════════════════════════╝

    Args:
        strategy: Key from STRATEGIES dict (e.g. "max_sharpe_ratio")
        tickers: List of ticker symbols to optimize over
        investment_amount: Total amount to invest

    Returns:
        OptimizationResult with weights, metrics, etc.
    """

    # ── Fetch real price data ──────────────────────────────────────
    prices = fetch_price_data(tickers)
    tickers = list(prices.columns)
    print(f"  [optimizer] Price data loaded: {len(tickers)} tickers, "
          f"{len(prices)} trading days")

    # ─── PLACEHOLDER: Replace this with your actual CPLEX calls ───
    #
    # Example of what your real implementation might look like:
    #
    #   from your_cplex_module import CPLEXOptimizer
    #
    #   optimizer = CPLEXOptimizer()
    #   optimizer.load_price_data(tickers)
    #
    #   if strategy == "minimum_variance":
    #       result = optimizer.minimize_variance(tickers)
    #   elif strategy == "max_sharpe_ratio":
    #       result = optimizer.maximize_sharpe(tickers)
    #   elif strategy == "max_expected_return":
    #       result = optimizer.maximize_return(tickers)
    #   elif strategy == "equal_risk_contribution":
    #       result = optimizer.equal_risk(tickers)
    #   elif strategy == "leveraged_max_sharpe":
    #       result = optimizer.leveraged_sharpe(tickers, max_leverage=1.5)
    #   elif strategy == "robust_mean_variance":
    #       result = optimizer.robust_mv(tickers)
    #   elif strategy == "equally_weighted":
    #       result = optimizer.equal_weight(tickers)
    #   elif strategy == "buy_and_hold":
    #       result = optimizer.buy_and_hold(tickers)
    #
    #   return OptimizationResult(
    #       strategy_used=strategy,
    #       strategy_display_name=STRATEGIES[strategy]["display_name"],
    #       success=True,
    #       allocations=[{"ticker": t, "weight": w} for t, w in zip(tickers, result.weights)],
    #       expected_return=result.annual_return,
    #       expected_volatility=result.annual_vol,
    #       sharpe_ratio=result.sharpe,
    #   )

    # ─── TEMPORARY PLACEHOLDER (remove when you plug in CPLEX) ───
    import random
    random.seed(42)
    n = len(tickers)

    if strategy == "equally_weighted":
        weights = [1.0 / n] * n
    else:
        # Fake weights that sum to 1.0
        raw = [random.random() for _ in range(n)]
        total = sum(raw)
        weights = [w / total for w in raw]

    allocations = []
    for ticker, weight in zip(tickers, weights):
        if weight >= 0.01:  # Only include >= 1% allocations
            allocations.append({
                "ticker": ticker,
                "weight": round(weight, 4),
            })

    # Re-normalize after filtering
    alloc_total = sum(a["weight"] for a in allocations)
    for a in allocations:
        a["weight"] = round(a["weight"] / alloc_total, 4)

    return OptimizationResult(
        strategy_used=strategy,
        strategy_display_name=STRATEGIES.get(strategy, {}).get("display_name", strategy),
        success=True,
        allocations=allocations,
        expected_return=0.08,       # placeholder
        expected_volatility=0.15,   # placeholder
        sharpe_ratio=0.53,          # placeholder
        metadata={"note": "PLACEHOLDER — replace with real CPLEX output"},
    )
    # ─── END PLACEHOLDER ─────────────────────────────────────────


# ── High-Level Interface ────────────────────────────────────────────────────

def optimize_portfolio(
    risk_category: str,
    investment_amount: float = 10000.0,
    user_strategy_preference: str = "",
) -> OptimizationResult:
    """
    Full pipeline: select strategy → select assets → run optimizer.

    This is the function the AI agent calls.

    Args:
        risk_category: From risk assessor (CONSERVATIVE/MODERATE/BALANCED/GROWTH/AGGRESSIVE)
        investment_amount: How much the user wants to invest
        user_strategy_preference: Optional explicit strategy request from user

    Returns:
        OptimizationResult
    """
    strategy = select_strategy(risk_category, user_strategy_preference)
    tickers = select_assets(risk_category)
    result = run_optimization(strategy, tickers, investment_amount)
    return result


def get_strategies_description() -> str:
    """Return a formatted string of all available strategies for the AI agent."""
    lines = ["AVAILABLE OPTIMIZATION STRATEGIES:"]
    for key, info in STRATEGIES.items():
        lines.append(f"  - {info['display_name']} ({key}): {info['description']} [Risk: {info['risk_level']}]")
    return "\n".join(lines)