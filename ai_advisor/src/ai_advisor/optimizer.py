"""
ai_advisor/optimizer.py

Portfolio optimization using cvxpy (QP/LP strategies) and scipy (ERC).
The AI agents pick a strategy → this module fetches price data, computes
inputs, runs the optimizer, and returns structured results.
Portfolio optimization using cvxpy (QP/LP strategies) and scipy (ERC).
The AI agents pick a strategy → this module fetches price data, computes
inputs, runs the optimizer, and returns structured results.
"""

from dataclasses import dataclass, field
import numpy as np
import scipy.optimize
import cvxpy as cp
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

from ai_advisor.stocks import get_all_tickers


# ── Strategy Definitions ────────────────────────────────────────────────────

STRATEGIES = {
    "market_tracking": {
        "display_name": "Market Tracking",
        "description": "Minimize tracking error against the S&P 500 (SPY). Passive index-like exposure.",
        "risk_level": "moderate",
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
        "description": "Maximize expected return with diversification cap. For aggressive investors.",
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
        "description": "Sharpe-optimal portfolio with 1.5x leverage. For aggressive/experienced investors.",
        "risk_level": "aggressive",
    },
    "robust_mean_variance": {
        "display_name": "Robust Mean-Variance Optimization",
        "description": "Minimize variance with a return floor accounting for estimation error.",
        "risk_level": "moderate",
    },
}

# Maps the AI's risk category → recommended optimizer strategies (ordered by preference)
RISK_TO_STRATEGIES = {
    "CONSERVATIVE": ["minimum_variance", "market_tracking", "equally_weighted"],
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
    metadata: dict = field(default_factory=dict)
    error: str = ""


# ── Strategy Selection ──────────────────────────────────────────────────────

def select_strategy(
    risk_category: str,
    agent_strategy: str = "",
    *,
    experience_level: str = "None",
    investment_horizon: str = "3-5yr",
    investment_style: str = "neutral",
    leverage_comfort: str = "no",
) -> str:
    """
    Select optimizer strategy using multi-dimensional scoring.

    1. Trust the agent's recommendation if it's a valid strategy key
       and passes hard disqualification rules.
    2. Otherwise score all RISK_TO_STRATEGIES candidates using
       experience, horizon, style, and leverage signals, then return
       the highest-scored candidate (ties broken by list order).
    """
    leverage_ok = leverage_comfort.lower() not in ("no",)
    exp_lower = experience_level.lower()
    style_lower = investment_style.lower()
    horizon_lower = investment_horizon.lower()

    # Step 1: trust agent if valid and not hard-disqualified
    if agent_strategy and agent_strategy in STRATEGIES:
        disqualified = (
            agent_strategy == "leveraged_max_sharpe"
            and (not leverage_ok or exp_lower in ("none", "beginner"))
        )
        if not disqualified:
            return agent_strategy

    # Step 2: score candidates from the risk category
    category = risk_category.upper()
    candidates = RISK_TO_STRATEGIES.get(category, ["max_sharpe_ratio"])
    scores: dict[str, float] = {c: 0.0 for c in candidates}

    def adj(s: str, d: float):
        if s in scores:
            scores[s] += d

    # Passive/active style signals
    if style_lower == "passive":
        adj("market_tracking", +3); adj("equally_weighted", +1)
        adj("max_expected_return", -2); adj("leveraged_max_sharpe", -3)
    elif style_lower == "active":
        adj("max_expected_return", +2); adj("max_sharpe_ratio", +2)
        adj("leveraged_max_sharpe", +1); adj("market_tracking", -2)

    # Experience gates
    if exp_lower in ("none", "beginner"):
        adj("leveraged_max_sharpe", -5); adj("max_expected_return", -2)
        adj("minimum_variance", +2); adj("equally_weighted", +2); adj("market_tracking", +1)

    # Horizon gates
    if horizon_lower in ("<1yr", "1-3yr"):
        adj("minimum_variance", +2); adj("leveraged_max_sharpe", -5); adj("max_expected_return", -1)
    elif horizon_lower == "20+yr":
        adj("max_expected_return", +1)
        if leverage_ok:
            adj("leveraged_max_sharpe", +1)

    # Leverage hard gate
    if leverage_comfort.lower() == "no":
        adj("leveraged_max_sharpe", -10)
    elif leverage_comfort.lower() == "yes":
        adj("leveraged_max_sharpe", +2)

    # Prefer earlier in list on ties (preserves existing priority for neutral profiles)
    return max(candidates, key=lambda c: (scores[c], -candidates.index(c)))
    scores: dict[str, float] = {c: 0.0 for c in candidates}

    def adj(s: str, d: float):
        if s in scores:
            scores[s] += d

    # Passive/active style signals
    if style_lower == "passive":
        adj("market_tracking", +3); adj("equally_weighted", +1)
        adj("max_expected_return", -2); adj("leveraged_max_sharpe", -3)
    elif style_lower == "active":
        adj("max_expected_return", +2); adj("max_sharpe_ratio", +2)
        adj("leveraged_max_sharpe", +1); adj("market_tracking", -2)

    # Experience gates
    if exp_lower in ("none", "beginner"):
        adj("leveraged_max_sharpe", -5); adj("max_expected_return", -2)
        adj("minimum_variance", +2); adj("equally_weighted", +2); adj("market_tracking", +1)

    # Horizon gates
    if horizon_lower in ("<1yr", "1-3yr"):
        adj("minimum_variance", +2); adj("leveraged_max_sharpe", -5); adj("max_expected_return", -1)
    elif horizon_lower == "20+yr":
        adj("max_expected_return", +1)
        if leverage_ok:
            adj("leveraged_max_sharpe", +1)

    # Leverage hard gate
    if leverage_comfort.lower() == "no":
        adj("leveraged_max_sharpe", -10)
    elif leverage_comfort.lower() == "yes":
        adj("leveraged_max_sharpe", +2)

    # Prefer earlier in list on ties (preserves existing priority for neutral profiles)
    return max(candidates, key=lambda c: (scores[c], -candidates.index(c)))


def select_assets(risk_category: str) -> list[str]:
    """
    Select which tickers from the approved universe to include in optimization.
    Conservative strategies get more ETFs/bonds; aggressive gets more individual stocks.
    """
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
        return get_all_tickers()


# ── Price Data Fetching ─────────────────────────────────────────────────────

def fetch_price_data(tickers: list[str]) -> pd.DataFrame:
    """
    Download historical adjusted close prices for the given tickers.
    Attempts 5 years first; falls back to 3 years if any ticker has
    less than 3 years of data (e.g. recent IPOs).

    Returns:
        DataFrame of daily adjusted close prices, columns = tickers.
        Tickers that fail entirely are dropped.
    """
    end = date.today()
    start_5y = end - timedelta(days=5 * 365)
    start_3y = end - timedelta(days=3 * 365)

    raw = yf.download(tickers, start=start_5y, end=end, auto_adjust=True, progress=False)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    min_rows = 3 * 252  # ~756 trading days
    sufficient = [col for col in raw.columns if raw[col].notna().sum() >= min_rows]
    short_history = [col for col in raw.columns if col not in sufficient]

    if short_history:
        print(f"  [yfinance] {short_history} have < 3y data — retrying on 3y window")
        raw_3y = yf.download(short_history, start=start_3y, end=end,
                             auto_adjust=True, progress=False)["Close"]
        if isinstance(raw_3y, pd.Series):
            raw_3y = raw_3y.to_frame()
        raw = raw[sufficient].join(raw_3y, how="outer")

    raw = raw.dropna(axis=1, how="all")
    failed = [t for t in tickers if t not in raw.columns]
    if failed:
        print(f"  [yfinance] Could not fetch data for: {failed} — dropping from universe")

    return raw


# ── Risk-Free Rate ──────────────────────────────────────────────────────────

_CACHED_RF: float | None = None

def _get_risk_free_rate() -> float:
    """
    Fetch the current annualized risk-free rate from the 13-week US T-bill (^IRX).
    yfinance returns the yield as an annualized percentage (e.g. 5.2 = 5.2%).
    Result is cached for the lifetime of the process so yfinance is only called once.
    Falls back to 4.5% if the fetch fails.
    """
    global _CACHED_RF
    if _CACHED_RF is not None:
        return _CACHED_RF
    try:
        tbill = yf.download("^IRX", period="5d", auto_adjust=True, progress=False)["Close"]
        series = tbill.dropna()
        rate = float(series.iloc[-1]) / 100.0
        if 0.0 < rate < 0.20:
            print(f"  [optimizer] Risk-free rate (^IRX): {rate:.2%}")
            _CACHED_RF = rate
            return _CACHED_RF
    except Exception:
        pass
    print("  [optimizer] Could not fetch ^IRX — using fallback rf = 4.50%")
    _CACHED_RF = 0.045
    return _CACHED_RF


# ── Portfolio Optimizer Class ───────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Computes portfolio weights for a given set of tickers using various
    optimization strategies. All strategies assume a fresh (empty) portfolio —
    the starting cash balance equals the full investment amount.

    Strategies:
        equally_weighted        — 1/n for each asset
        market_tracking         — minimize tracking error vs SPY benchmark (cvxpy QP)
        minimum_variance        — minimize portfolio variance (cvxpy QP)
        max_expected_return     — maximize return with 40% per-asset cap (cvxpy LP)
        max_sharpe_ratio        — maximize Sharpe ratio via parametric QP (cvxpy)
        leveraged_max_sharpe    — max Sharpe scaled to 1.5x leverage
        equal_risk_contribution — each asset contributes equally to risk (scipy SLSQP)
        robust_mean_variance    — minimize variance with return floor (cvxpy QP)
    """

    def __init__(self, prices: pd.DataFrame, investment_amount: float):
        self.tickers = list(prices.columns)
        self.n = len(self.tickers)
        self.investment_amount = investment_amount
        self._prices = prices  # kept for market_tracking benchmark alignment

        ret = prices.pct_change().dropna().values  # shape (T, n), plain numpy array
        self.mu = ret.mean(axis=0) * 252               # annualized expected returns
        self.Q  = np.cov(ret, rowvar=False) * 252      # annualized covariance matrix
        self.cur_prices = prices.iloc[-1].values       # latest close prices
        self.rf = _get_risk_free_rate()

    # ── Simple strategies ───────────────────────────────────────

    def equally_weighted(self) -> np.ndarray:
        return np.ones(self.n) / self.n

    def market_tracking(self, benchmark: str = "SPY") -> np.ndarray:
        """
        Minimize tracking error variance against a market benchmark (default: SPY).
        Solves: min  w'*Cov(r)*w  -  2 * cov(r, r_bench)' * w
                s.t. sum(w) = 1, w >= 0
        which is equivalent to minimising E[(w'r - r_bench)^2].
        """
        prices = self._prices
        bench_raw = yf.download(
            benchmark,
            start=prices.index[0],
            end=prices.index[-1],
            auto_adjust=True,
            progress=False,
        )["Close"]
        bench_ret = bench_raw.pct_change().dropna()

        asset_ret = prices.pct_change().dropna()
        if isinstance(bench_ret, pd.DataFrame):
            bench_ret = bench_ret.iloc[:, 0]
        bench_ret.name = "__bench__"
        aligned = asset_ret.join(bench_ret, how="inner").dropna()

        r = aligned.drop(columns="__bench__").values          # (T, n)
        r_bench = aligned["__bench__"].values                  # (T,)

        Q_te = np.cov(r, rowvar=False) * 252
        sigma_bench = np.array(
            [np.cov(r[:, i], r_bench)[0, 1] for i in range(self.n)]
        ) * 252

        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, Q_te) - 2 * sigma_bench @ w),
            [cp.sum(w) == 1, w >= 0],
        )
        prob.solve(solver=cp.CLARABEL)
        return np.array(w.value)

    # ── cvxpy strategies ────────────────────────────────────────

    def minimum_variance(self) -> np.ndarray:
        """Minimize portfolio variance: min w'Qw  s.t. sum(w)=1, w>=0"""
        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, self.Q)),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve(solver=cp.CLARABEL)
        return np.array(w.value)

    def max_expected_return(self, max_weight: float = 0.40) -> np.ndarray:
        """
        Maximize expected return with a per-asset cap to prevent 100% concentration.
        max mu'w  s.t. sum(w)=1, w>=0, w<=max_weight
        """
        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Maximize(self.mu @ w),
            [cp.sum(w) == 1, w >= 0, w <= max_weight]
        )
        prob.solve(solver=cp.CLARABEL)
        return np.array(w.value)

    def max_sharpe_ratio(self) -> np.ndarray:
        """
        Maximize Sharpe ratio via the Dinkelbach parametric transformation.
        Let y = w/kappa. Solve: min y'Qy  s.t. (mu-rf)'y = 1, y >= 0.
        Then w = y / sum(y).
        """
        excess = self.mu - self.rf
        y = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(y, self.Q)),
            [excess @ y == 1, y >= 0]
        )
        prob.solve(solver=cp.CLARABEL)
        y_val = np.array(y.value)
        return y_val / y_val.sum()

    def leveraged_max_sharpe(self, max_leverage: float = 1.5) -> np.ndarray:
        """
        Max Sharpe portfolio scaled by max_leverage. Weights sum to max_leverage.
        Per Capital Market Line theory, levering the tangency portfolio preserves
        the Sharpe ratio while scaling return and volatility proportionally.
        """
        return self.max_sharpe_ratio() * max_leverage

    def robust_mean_variance(self) -> np.ndarray:
        """
        Minimize variance subject to a return floor that accounts for
        estimation uncertainty (mirrors the reference strat_robust_optim logic).
        The floor = max_sharpe_return - avg_sharpe/2.
        """
        w_msr = self.max_sharpe_ratio()
        ret_msr = float(self.mu @ w_msr)

        vol = np.sqrt(np.diag(self.Q))
        avg_sr = float(np.mean((self.mu - self.rf) / (vol + 1e-8)))
        ep_est_err = abs(avg_sr / 2)

        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, self.Q)),
            [cp.sum(w) == 1, w >= 0, self.mu @ w >= ret_msr - ep_est_err]
        )
        prob.solve(solver=cp.CLARABEL)
        return np.array(w.value)

    # ── scipy strategy ──────────────────────────────────────────

    def equal_risk_contribution(self) -> np.ndarray:
        """
        Equal Risk Contribution (Risk Parity): each asset contributes equally
        to total portfolio variance. Uses scipy SLSQP with the same objective
        and analytical gradient as the reference IPOPT implementation.
        """
        Q = self.Q

        def objective(w):
            Qw = Q @ w
            RC = w * Qw
            diffs = RC[:, None] - RC[None, :]
            return 2.0 * float(np.sum(diffs ** 2))

        def gradient(w):
            Qw = Q @ w
            RC = w * Qw
            grad = np.zeros(self.n)
            for k in range(self.n):
                dRCk = Qw[k] + w[k] * Q[k]
                grad[k] = 4.0 * float(np.sum((RC[k] - RC) * dRCk))
            return grad

        w0 = np.ones(self.n) / self.n
        result = scipy.optimize.minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * self.n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return result.x

    # ── Helper ──────────────────────────────────────────────────

    def _to_allocations(self, weights: np.ndarray) -> list[dict]:
        """
        Convert a weights array to the allocations list format, filtering
        out positions below 1% and re-normalising the remainder.
        """
        raw = [
            {"ticker": t, "weight": float(w)}
            for t, w in zip(self.tickers, weights)
            if float(w) >= 0.01
        ]
        total = sum(a["weight"] for a in raw)
        for a in raw:
            a["weight"] = round(a["weight"] / total, 4)
        return raw


# ── Optimizer Entry Point ───────────────────────────────────────────────────

def run_optimization(
    strategy: str,
    tickers: list[str],
    investment_amount: float = 10000.0,
) -> OptimizationResult:
    """
    Fetch price data, build a PortfolioOptimizer, run the requested strategy,
    and return a fully populated OptimizationResult.
    """
    prices = fetch_price_data(tickers)
    tickers = list(prices.columns)
    print(f"  [optimizer] Price data loaded: {len(tickers)} tickers, {len(prices)} trading days")

    opt = PortfolioOptimizer(prices, investment_amount)

    dispatch = {
        "equally_weighted":        opt.equally_weighted,
        "market_tracking":         opt.market_tracking,
        "minimum_variance":        opt.minimum_variance,
        "max_expected_return":     opt.max_expected_return,
        "max_sharpe_ratio":        opt.max_sharpe_ratio,
        "leveraged_max_sharpe":    opt.leveraged_max_sharpe,
        "equal_risk_contribution": opt.equal_risk_contribution,
        "robust_mean_variance":    opt.robust_mean_variance,
    }

    if strategy not in dispatch:
        return OptimizationResult(error=f"Unknown strategy: {strategy}")

    try:
        weights = dispatch[strategy]()
    except Exception as e:
        return OptimizationResult(error=f"Solver failed: {e}")

    allocations = opt._to_allocations(weights)

    # Compute annualised portfolio metrics from the filtered/renormalised weights
    final_w = np.array([a["weight"] for a in allocations])
    final_tickers = [a["ticker"] for a in allocations]
    idx = [opt.tickers.index(t) for t in final_tickers]
    mu_sub = opt.mu[idx]
    Q_sub  = opt.Q[np.ix_(idx, idx)]

    exp_ret = float(mu_sub @ final_w)
    exp_vol = float(np.sqrt(final_w @ Q_sub @ final_w))
    sharpe  = (exp_ret - opt.rf) / exp_vol if exp_vol > 0 else 0.0

    return OptimizationResult(
        strategy_used=strategy,
        strategy_display_name=STRATEGIES[strategy]["display_name"],
        success=True,
        allocations=allocations,
        expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe_ratio=sharpe,
        metadata={"risk_free_rate": opt.rf, "n_assets": len(final_tickers)},
        expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe_ratio=sharpe,
        metadata={"risk_free_rate": opt.rf, "n_assets": len(final_tickers), "investment_amount": investment_amount},
    )


# ── High-Level Interface ────────────────────────────────────────────────────

def optimize_portfolio(
    risk_category: str,
    investment_amount: float = 10000.0,
    user_strategy_preference: str = "",
) -> OptimizationResult:
    """Full pipeline: select strategy → select assets → run optimizer."""
    strategy = select_strategy(risk_category, user_strategy_preference)
    tickers = select_assets(risk_category)
    return run_optimization(strategy, tickers, investment_amount)


def get_strategies_description() -> str:
    """Return a formatted string of all available strategies for the AI agent."""
    lines = ["AVAILABLE OPTIMIZATION STRATEGIES:"]
    for key, info in STRATEGIES.items():
        lines.append(f"  - {info['display_name']} ({key}): {info['description']} [Risk: {info['risk_level']}]")
    return "\n".join(lines)
