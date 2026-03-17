"""
ai_advisor/run_advisor.py

Programmatic interface to the AI Financial Advisor.
Import and call from any other Python file — no interactive CLI needed.
"""

from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass, field
from ai_advisor.crew import AiAdvisor
from ai_advisor.stocks import get_approved_universe_text


# ── Structured Output ───────────────────────────────────────────────────────

@dataclass
class AdvisorResult:
    """Structured result from the AI advisor pipeline."""
    # Raw text outputs from each agent
    financial_profile: str = ""
    risk_assessment: str = ""
    portfolio_recommendation: str = ""

    # Parsed fields (extracted from agent outputs)
    strategy: str = ""                     # e.g. "GROWTH", "CONSERVATIVE"
    risk_score: str = ""                   # e.g. "7"
    allocations: list[dict] = field(default_factory=list)  # [{"ticker": "VOO", "pct": 30, "name": "...", "rationale": "..."}, ...]

    # The full raw CrewAI result object
    raw_result: object = None


# ── Survey Data Helper ──────────────────────────────────────────────────────

def build_survey_string(answers: dict[str, str]) -> str:
    """
    Convert a dict of survey answers into the formatted string the crew expects.

    Expected keys (all strings):
        age, employment_status, dependents,
        annual_income, monthly_expenses, monthly_savings,
        total_savings, existing_investments, total_debt, debt_details,
        experience_level, current_holdings,
        investment_horizon, investment_amount,
        short_term_goals, long_term_goals,
        risk_comfort, return_vs_safety, loss_tolerance,
        special_considerations
    """
    labels = {
        "age": "Age",
        "employment_status": "Employment Status",
        "dependents": "Number of Dependents",
        "annual_income": "Annual Income (pre-tax)",
        "monthly_expenses": "Monthly Expenses",
        "monthly_savings": "Monthly Savings/Investment",
        "total_savings": "Total Liquid Savings",
        "existing_investments": "Existing Investments Value",
        "total_debt": "Total Outstanding Debt",
        "debt_details": "Debt Details",
        "experience_level": "Investment Experience",
        "current_holdings": "Current Holdings",
        "investment_horizon": "Investment Horizon",
        "investment_amount": "Initial Investment Amount",
        "short_term_goals": "Short-term Goals (1-3 years)",
        "long_term_goals": "Long-term Goals",
        "risk_comfort": "Reaction to 20% Portfolio Drop",
        "return_vs_safety": "Growth vs Safety Priority",
        "loss_tolerance": "Maximum Tolerable Annual Loss",
        "special_considerations": "Special Considerations",
    }

    lines = []
    for key, answer in answers.items():
        label = labels.get(key, key)
        lines.append(f"{label}: {answer}")
    return "\n".join(lines)


# ── Parser ──────────────────────────────────────────────────────────────────

def _parse_strategy(text: str) -> str:
    """Extract strategy category from risk assessment or portfolio text."""
    for strategy in ["AGGRESSIVE", "GROWTH", "BALANCED", "MODERATE", "CONSERVATIVE"]:
        if strategy in text.upper():
            return strategy
    return "UNKNOWN"


def _parse_risk_score(text: str) -> str:
    """Extract risk tolerance score from risk assessment text."""
    import re
    match = re.search(r"Risk Tolerance Score:\s*(\d+)", text)
    return match.group(1) if match else ""


def _parse_allocations(text: str) -> list[dict]:
    """
    Extract ticker allocations from the portfolio recommendation text.
    Looks for patterns like: AAPL - Apple Inc. - 15% - some rationale
    """
    import re
    allocations = []
    # Match lines like: 1. VOO - Vanguard S&P 500 ETF - 30% - core holding
    pattern = r"\d+\.\s*([A-Z.]+)\s*[-–—]\s*(.+?)\s*[-–—]\s*(\d+)%\s*[-–—]\s*(.+)"
    for match in re.finditer(pattern, text):
        allocations.append({
            "ticker": match.group(1).strip(),
            "name": match.group(2).strip(),
            "pct": int(match.group(3)),
            "rationale": match.group(4).strip(),
        })
    return allocations


# ── Main Runner ─────────────────────────────────────────────────────────────

def run_advisor(answers: dict[str, str]) -> AdvisorResult:
    """
    Run the full advisor pipeline with the given survey answers.

    Args:
        answers: Dict of survey question keys -> answer strings.

    Returns:
        AdvisorResult with structured data.
    """
    survey_text = build_survey_string(answers)

    inputs = {
        "survey_responses": survey_text,
        "approved_securities": get_approved_universe_text(),
    }

    crew = AiAdvisor().crew()
    raw = crew.kickoff(inputs=inputs)

    # CrewAI stores each task's output in crew.tasks[i].output
    tasks = crew.tasks
    profile_text = str(tasks[0].output) if len(tasks) > 0 and tasks[0].output else ""
    risk_text = str(tasks[1].output) if len(tasks) > 1 and tasks[1].output else ""
    portfolio_text = str(tasks[2].output) if len(tasks) > 2 and tasks[2].output else ""

    # If task outputs aren't available, fall back to the final result
    if not portfolio_text:
        portfolio_text = str(raw)

    return AdvisorResult(
        financial_profile=profile_text,
        risk_assessment=risk_text,
        portfolio_recommendation=portfolio_text,
        strategy=_parse_strategy(risk_text + portfolio_text),
        risk_score=_parse_risk_score(risk_text),
        allocations=_parse_allocations(portfolio_text),
        raw_result=raw,
    )
