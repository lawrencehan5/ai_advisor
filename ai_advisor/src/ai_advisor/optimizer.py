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
    Return historical adjusted-close prices for the given tickers.

    Data is served from the local parquet cache (data/price_cache.parquet).
    The cache covers the full approved universe and is updated incrementally —
    only missing trading days are downloaded from yfinance on each run.

    Returns:
        DataFrame of daily adjusted close prices, columns = tickers.
        Tickers unavailable in the cache are silently dropped.
    """
    from ai_advisor.price_cache import load_prices
    return load_prices(tickers)


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

    def __init__(self, prices: pd.DataFrame, investment_amount: float, max_weight: float = 1.0):
        self.investment_amount = investment_amount
        self.max_weight = min(max_weight, 1.0)  # cap at 100%

        # ── Per-column daily returns (NaN before each ticker's IPO/launch) ──
        ret = prices.pct_change()

        # Drop tickers with fewer than 1 year of actual data — not enough to
        # estimate statistics, and would otherwise drag the entire return matrix
        # down to their short history when dropna() is applied below.
        min_obs = 252  # ~1 trading year
        valid_cols = [c for c in ret.columns if ret[c].notna().sum() >= min_obs]
        dropped = sorted(set(ret.columns) - set(valid_cols))
        if dropped:
            print(f"  [optimizer] Dropping {dropped} — fewer than 1 year of data")
            ret    = ret[valid_cols]
            prices = prices[valid_cols]

        self.tickers    = list(ret.columns)
        self.n          = len(self.tickers)
        self.cur_prices = prices.iloc[-1].values
        self._prices    = prices  # kept for market_tracking benchmark alignment

        # Expected returns: use each ticker's own full available history.
        # skipna=True means TSLA (since 2010) uses 15 years of its returns even
        # if another ticker only has 5 years — no data thrown away for mu.
        self.mu = ret.mean(skipna=True).values * 252

        # Covariance matrix: requires all tickers to have a value on the same
        # day, so we take the inner-join (dropna) of the filtered return matrix.
        # With very-short-history tickers already removed above, the overlap
        # period is now determined by the shortest *valid* ticker (~3–15 years)
        # rather than the shortest ticker overall (e.g. IBIT at 1 year).
        ret_aligned = ret.dropna()
        print(
            f"  [optimizer] Covariance window: {len(ret_aligned)} days "
            f"({ret_aligned.index[0].date()} → {ret_aligned.index[-1].date()}) "
            f"across {self.n} tickers"
        )
        self.Q  = np.cov(ret_aligned.values, rowvar=False) * 252
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
        SPY data is served from the local price cache (no extra yfinance call).
        """
        from ai_advisor.price_cache import load_prices
        prices = self._prices
        bench_full = load_prices([benchmark])
        # Slice to the same date window as the portfolio prices
        bench_raw = bench_full.loc[prices.index[0]:prices.index[-1], benchmark]
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
            [cp.sum(w) == 1, w >= 0, w <= self.max_weight],
        )
        prob.solve(solver=cp.CLARABEL)
        return np.array(w.value)

    # ── cvxpy strategies ────────────────────────────────────────

    def minimum_variance(self) -> np.ndarray:
        """Minimize portfolio variance: min w'Qw  s.t. sum(w)=1, w>=0, w<=max_weight"""
        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, self.Q)),
            [cp.sum(w) == 1, w >= 0, w <= self.max_weight]
        )
        prob.solve(solver=cp.CLARABEL)
        return np.array(w.value)

    def max_expected_return(self) -> np.ndarray:
        """
        Maximize expected return with a per-asset cap to prevent 100% concentration.
        max mu'w  s.t. sum(w)=1, w>=0, w<=max_weight
        The cap is driven by self.max_weight (set from user's preferred number of holdings).
        Default fallback cap is 40% if max_weight was not restricted by the user.
        """
        cap = min(self.max_weight, 0.40)
        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Maximize(self.mu @ w),
            [cp.sum(w) == 1, w >= 0, w <= cap]
        )
        prob.solve(solver=cp.CLARABEL)
        return np.array(w.value)

    def max_sharpe_ratio(self) -> np.ndarray:
        """
        Maximize Sharpe ratio via the Dinkelbach parametric transformation.
        Let y = w/kappa. Solve: min y'Qy  s.t. (mu-rf)'y = 1, y >= 0.
        The per-asset cap w_i <= max_weight becomes y_i <= max_weight * sum(y) (linear in y).
        Then w = y / sum(y).
        """
        excess = self.mu - self.rf
        y = cp.Variable(self.n)
        constraints = [excess @ y == 1, y >= 0]
        if self.max_weight < 1.0:
            constraints.append(y <= self.max_weight * cp.sum(y))
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, self.Q)), constraints)
        prob.solve(solver=cp.CLARABEL)
        y_val = np.array(y.value)
        return y_val / y_val.sum()

    def leveraged_max_sharpe(self, max_leverage: float = 1.5) -> np.ndarray:
        """
        Max Sharpe portfolio scaled by max_leverage. Weights sum to max_leverage.
        Per Capital Market Line theory, levering the tangency portfolio preserves
        the Sharpe ratio while scaling return and volatility proportionally.
        Note: post-scaling, per-asset weights can reach max_weight * max_leverage.
        This is intentional — leverage inherently increases concentration.
        """
        return self.max_sharpe_ratio() * max_leverage

    def robust_mean_variance(self) -> np.ndarray:
        """
        Minimize variance subject to a return floor that accounts for
        estimation uncertainty (mirrors the reference strat_robust_optim logic).
        The floor = unconstrained_max_sharpe_return - avg_sharpe/2.

        The return floor uses the UNCONSTRAINED tangency return so it remains
        a meaningful reference point regardless of max_weight tightness.
        The variance minimisation itself still respects max_weight.
        """
        # Return floor from unconstrained tangency portfolio
        excess = self.mu - self.rf
        y_ref = cp.Variable(self.n)
        cp.Problem(
            cp.Minimize(cp.quad_form(y_ref, self.Q)),
            [excess @ y_ref == 1, y_ref >= 0],
        ).solve(solver=cp.CLARABEL)
        y_val = np.array(y_ref.value)
        w_msr_ref = y_val / y_val.sum()
        ret_msr = float(self.mu @ w_msr_ref)

        vol = np.sqrt(np.diag(self.Q))
        avg_sr = float(np.mean((self.mu - self.rf) / (vol + 1e-8)))
        ep_est_err = abs(avg_sr / 2)

        w = cp.Variable(self.n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, self.Q)),
            [cp.sum(w) == 1, w >= 0, w <= self.max_weight, self.mu @ w >= ret_msr - ep_est_err]
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
            bounds=[(0.0, self.max_weight)] * self.n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        return result.x

    # ── Helper ──────────────────────────────────────────────────

    def _to_allocations(self, weights: np.ndarray) -> list[dict]:
        """
        Convert a weights array to the allocations list format, filtering
        out positions below 1% and re-normalising the remainder.
        After re-normalisation, clips to max_weight and renormalises once
        more to ensure the constraint is never violated by rounding.
        """
        raw = [
            {"ticker": t, "weight": float(w)}
            for t, w in zip(self.tickers, weights)
            if float(w) >= 0.01
        ]
        total = sum(a["weight"] for a in raw)
        for a in raw:
            a["weight"] = a["weight"] / total
        # Re-normalising after dropping small positions can push weights above max_weight.
        # Clip then renormalise to restore the invariant.
        if self.max_weight < 1.0:
            for a in raw:
                a["weight"] = min(a["weight"], self.max_weight)
            total2 = sum(a["weight"] for a in raw)
            for a in raw:
                a["weight"] = round(a["weight"] / total2, 4)
        else:
            for a in raw:
                a["weight"] = round(a["weight"], 4)
        return raw


# ── Optimizer Entry Point ───────────────────────────────────────────────────

def run_optimization(
    strategy: str,
    tickers: list[str],
    investment_amount: float = 10000.0,
    max_weight: float = 1.0,
) -> OptimizationResult:
    """
    Fetch price data, build a PortfolioOptimizer, run the requested strategy,
    and return a fully populated OptimizationResult.

    Args:
        max_weight: Per-asset weight cap (0–1). Derived from user's preferred
                    number of holdings, e.g. 0.15 for a 10-stock portfolio.
    """
    prices = fetch_price_data(tickers)
    tickers = list(prices.columns)
    print(f"  [optimizer] Price data loaded: {len(tickers)} tickers, {len(prices)} trading days")
    print(f"  [optimizer] Per-asset max weight: {max_weight:.0%}")

    opt = PortfolioOptimizer(prices, investment_amount, max_weight=max_weight)

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
        metadata={"risk_free_rate": opt.rf, "n_assets": len(final_tickers), "investment_amount": investment_amount, "max_weight": max_weight},
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
