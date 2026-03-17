"""
Example: calling the AI advisor from another Python file.
Run from project root:  python -m ai_advisor.example_usage
"""

from ai_advisor.run_advisor import run_advisor

# ── Step 1: Provide survey answers as a plain dict ──────────────────────────

survey_answers = {
    "age": "30",
    "employment_status": "Full-time employed",
    "dependents": "0",
    "annual_income": "$120,000",
    "monthly_expenses": "$3,500",
    "monthly_savings": "$2,000",
    "total_savings": "$45,000",
    "existing_investments": "$10,000 in 401k",
    "total_debt": "$15,000",
    "debt_details": "$15k student loan at 4.5%",
    "experience_level": "Beginner",
    "current_holdings": "401k with target-date fund",
    "investment_horizon": "20+ years",
    "investment_amount": "$20,000",
    "short_term_goals": "Build 6-month emergency fund",
    "long_term_goals": "Retire by 55, buy a house in 5 years",
    "risk_comfort": "I'd stay the course — market dips are normal",
    "return_vs_safety": "Mostly growth — I can handle short-term losses",
    "loss_tolerance": "20%",
    "special_considerations": "Interested in tech sector, no ESG constraints",
}

# ── Step 2: Run the advisor ─────────────────────────────────────────────────

result = run_advisor(survey_answers)

# ── Step 3: Access structured data ──────────────────────────────────────────

print("=== STRATEGY ===")
print(result.strategy)          # e.g. "GROWTH"

print("\n=== RISK SCORE ===")
print(result.risk_score)        # e.g. "7"

print("\n=== ALLOCATIONS ===")
for a in result.allocations:
    print(f"  {a['ticker']:6s}  {a['pct']:3d}%  {a['name']}")
    # a['rationale'] is also available

print(f"\n  Total: {sum(a['pct'] for a in result.allocations)}%")

print("\n=== FULL OUTPUTS (raw text) ===")
print("--- Financial Profile ---")
print(result.financial_profile[:200], "...\n")
print("--- Risk Assessment ---")
print(result.risk_assessment[:200], "...\n")
print("--- Portfolio Recommendation ---")
print(result.portfolio_recommendation[:200], "...\n")


# ── Step 4: Use in your own code / pass to CPLEX ────────────────────────────

# The allocations list is ready for your optimizer:
#
#   result.allocations = [
#       {"ticker": "VOO", "pct": 30, "name": "Vanguard S&P 500 ETF", "rationale": "..."},
#       {"ticker": "QQQ", "pct": 20, "name": "Invesco QQQ Trust", "rationale": "..."},
#       ...
#   ]
#
#   result.strategy = "GROWTH"   ← use this to select your CPLEX strategy
#
# Example: feed into your optimizer
#
#   from your_optimizer import optimize_portfolio
#   tickers = [a["ticker"] for a in result.allocations]
#   weights = [a["pct"] / 100 for a in result.allocations]
#   optimized = optimize_portfolio(tickers, weights, strategy=result.strategy)
