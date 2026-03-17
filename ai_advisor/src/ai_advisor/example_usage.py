"""
Example: calling the AI advisor programmatically from another Python file.
Shows both initial pipeline and follow-up questions.

Run: python -m ai_advisor.example_usage
"""

from ai_advisor.run_advisor import run_initial_pipeline, run_followup

# ── Step 1: Provide survey answers as a formatted string ────────────────────

survey = """
Age: 30
Employment Status: Full-time employed
Number of Dependents: 0
Annual Income (pre-tax): $120,000
Monthly Expenses: $3,500
Monthly Savings/Investment: $2,000
Total Liquid Savings: $45,000
Existing Investments Value: $10,000 in 401k
Total Outstanding Debt: $15,000
Debt Details: $15k student loan at 4.5%
Investment Experience: Beginner
Current Holdings: 401k with target-date fund
Investment Horizon: 20+ years
Initial Investment Amount: $20,000
Short-term Goals: Build 6-month emergency fund
Long-term Goals: Retire by 55
Reaction to 20% Portfolio Drop: Stay the course — market dips are normal
Growth vs Safety Priority: Mostly growth — I can handle short-term losses
Maximum Tolerable Annual Loss: 20%
Special Considerations: Interested in tech sector
"""

# ── Step 2: Run the full pipeline ───────────────────────────────────────────

result = run_initial_pipeline(survey)

# ── Step 3: Access structured data ──────────────────────────────────────────

print("=== STRATEGY ===")
print(f"Risk Category: {result.risk_category}")
print(f"Risk Score: {result.risk_score}")
print(f"Optimizer Strategy: {result.optimizer_strategy}")

print("\n=== OPTIMIZER RESULTS ===")
opt = result.optimization_result
print(f"Expected Return:     {opt.expected_return:.1%}")
print(f"Expected Volatility: {opt.expected_volatility:.1%}")
print(f"Sharpe Ratio:        {opt.sharpe_ratio:.2f}")

print("\n=== ALLOCATIONS (from CPLEX) ===")
for a in result.allocations:
    print(f"  {a['ticker']:6s}  {a['weight']*100:5.1f}%")

print("\n=== RECOMMENDATION (agent's explanation) ===")
print(result.portfolio_recommendation[:500], "...\n")

# ── Step 4: Ask follow-up questions ─────────────────────────────────────────

print("=== FOLLOW-UP QUESTIONS ===\n")

answer1 = run_followup("Why was this strategy chosen over minimum variance?", result)
print(f"Q: Why was this strategy chosen over minimum variance?")
print(f"A: {answer1}\n")

answer2 = run_followup("What would change if I were 10 years older?", result)
print(f"Q: What would change if I were 10 years older?")
print(f"A: {answer2}\n")

# ── Step 5: Feed into your CPLEX optimizer (example hook) ───────────────────
#
#   tickers = [a["ticker"] for a in result.allocations]
#   weights = [a["weight"] for a in result.allocations]
#   strategy = result.optimizer_strategy
#
#   from your_cplex_module import run_cplex
#   optimized = run_cplex(tickers, weights, strategy)
