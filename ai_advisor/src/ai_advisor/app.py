#!/usr/bin/env python
"""
AI Financial Advisor — CLI Chatbot
1. Conducts a financial survey
2. Runs AI analysis + CPLEX optimization
3. Presents portfolio recommendation
4. Enters follow-up conversation loop
"""

from dotenv import load_dotenv
load_dotenv()

import warnings
import re
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from ai_advisor.run_advisor import run_initial_pipeline, run_followup, AdvisorResult


# ── Survey Definition ───────────────────────────────────────────────────────

SURVEY_SECTIONS = [
    {
        "title": "PERSONAL INFORMATION",
        "questions": [
            {"key": "age", "prompt": "How old are you?"},
            {
                "key": "employment_status",
                "prompt": "What is your employment status?",
                "choices": [
                    "1. Full-time employed",
                    "2. Part-time employed",
                    "3. Self-employed / Freelance",
                    "4. Retired",
                    "5. Student",
                    "6. Unemployed",
                ],
            },
            {"key": "dependents", "prompt": "How many financial dependents do you have?"},
        ],
    },
    {
        "title": "INCOME & EXPENSES",
        "questions": [
            {"key": "annual_income", "prompt": "What is your approximate annual income (before tax)?"},
            {"key": "monthly_expenses", "prompt": "What are your approximate total monthly expenses?"},
            {"key": "monthly_savings", "prompt": "How much do you save or invest per month?"},
        ],
    },
    {
        "title": "ASSETS & LIABILITIES",
        "questions": [
            {"key": "total_savings", "prompt": "Total value of your liquid savings?"},
            {"key": "existing_investments", "prompt": "Total value of current investments? (0 if none)"},
            {"key": "total_debt", "prompt": "Total outstanding debt? (0 if none)"},
            {"key": "debt_details", "prompt": "Describe your debts (e.g. '$20k student loan'). Type 'none' if no debt."},
        ],
    },
    {
        "title": "INVESTMENT EXPERIENCE",
        "questions": [
            {
                "key": "experience_level",
                "prompt": "How would you describe your investment experience?",
                "choices": [
                    "1. None — I've never invested",
                    "2. Beginner — I've done a little investing",
                    "3. Intermediate — I invest regularly",
                    "4. Advanced — I actively manage a diversified portfolio",
                ],
            },
            {"key": "current_holdings", "prompt": "What investments do you currently hold? Type 'none' if none."},
        ],
    },
    {
        "title": "INVESTMENT HORIZON & GOALS",
        "questions": [
            {
                "key": "investment_horizon",
                "prompt": "How long do you plan to keep this money invested?",
                "choices": [
                    "1. Less than 1 year",
                    "2. 1–3 years",
                    "3. 3–5 years",
                    "4. 5–10 years",
                    "5. 10–20 years",
                    "6. 20+ years",
                ],
            },
            {"key": "investment_amount", "prompt": "How much are you looking to invest initially?"},
            {"key": "short_term_goals", "prompt": "Short-term financial goals (next 1-3 years)?"},
            {"key": "long_term_goals", "prompt": "Long-term financial goals?"},
        ],
    },
    {
        "title": "RISK TOLERANCE",
        "questions": [
            {
                "key": "risk_comfort",
                "prompt": "How would you react if your portfolio dropped 20% in one month?",
                "choices": [
                    "1. Sell everything immediately",
                    "2. Very uncomfortable, consider selling some",
                    "3. Concerned but would hold and wait",
                    "4. Stay the course — dips are normal",
                    "5. See it as a buying opportunity",
                ],
            },
            {
                "key": "return_vs_safety",
                "prompt": "Which best describes your priority?",
                "choices": [
                    "1. Protect my capital above all",
                    "2. Mostly safe with a little growth",
                    "3. Balance between growth and safety",
                    "4. Mostly growth, can handle losses",
                    "5. Maximum growth, comfortable with high volatility",
                ],
            },
            {
                "key": "loss_tolerance",
                "prompt": "Maximum annual loss you could tolerate?",
                "choices": [
                    "1. 5% or less",
                    "2. 10%",
                    "3. 20%",
                    "4. 30%",
                    "5. 40% or more",
                ],
            },
        ],
    },
    {
        "title": "ADDITIONAL INFORMATION",
        "questions": [
            {"key": "special_considerations", "prompt": "Anything else? (ESG preferences, upcoming expenses, etc.) Type 'none' if nothing."},
        ],
    },
]


# ── CLI Helpers ─────────────────────────────────────────────────────────────

def print_header(text: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def ask_question(question: dict) -> str:
    print()
    print(f"  {question['prompt']}")
    if "choices" in question:
        for choice in question["choices"]:
            print(f"    {choice}")
        print()
        while True:
            answer = input("  Your choice (enter number): ").strip()
            valid = [str(i + 1) for i in range(len(question["choices"]))]
            if answer in valid:
                return question["choices"][int(answer) - 1]
            print("  Please enter a valid option number.")
    else:
        answer = input("  Your answer: ").strip()
        while not answer:
            answer = input("  Please enter a response: ").strip()
        return answer


def format_responses(responses: dict[str, str]) -> str:
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
    return "\n".join(f"{labels.get(k, k)}: {v}" for k, v in responses.items())


def _parse_amount_to_number(raw: str) -> float:
    """Parse user-entered money/number into a float.

    Handles inputs like "$15,000", "15000", "15k", "1.5m", and "none".
    If parsing fails, returns 0.0.
    """
    if raw is None:
        return 0.0

    s = str(raw).strip().lower()
    if s in {"", "none", "n/a", "na"}:
        return 0.0

    s = s.replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([kmb])?", s)
    if not match:
        return 0.0

    value = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    elif suffix == "b":
        value *= 1_000_000_000
    return value


def conduct_survey() -> str:
    print_header("AI FINANCIAL ADVISOR")
    print("  Welcome! I'll ask you a series of questions about your")
    print("  financial situation, goals, and risk tolerance.")
    print()
    print("  Type your answers when prompted. For multiple-choice")
    print("  questions, enter the number of your choice.")
    print("-" * 60)

    responses: dict[str, str] = {}
    for section in SURVEY_SECTIONS:
        print_header(section["title"])
        for q in section["questions"]:
            if q["key"] == "debt_details":
                # If total debt is $0, skip the debt description question.
                if _parse_amount_to_number(responses.get("total_debt", "")) <= 0:
                    responses["debt_details"] = "none"
                    continue

            responses[q["key"]] = ask_question(q)

    formatted = format_responses(responses)

    print_header("SURVEY COMPLETE")
    print("  Thank you! Processing your responses...")
    print("-" * 60)

    return formatted


# ── Follow-up Conversation Loop ────────────────────────────────────────────

def conversation_loop(result: AdvisorResult) -> None:
    """
    After the initial recommendation, let the user ask follow-up questions.
    Each question is answered by the followup_advisor with full context.
    """
    print()
    print("-" * 60)
    print("  You can now ask follow-up questions about your recommendation.")
    print("  Examples:")
    print("    • Why did you pick VOO over VTI?")
    print("    • What would change if I had a shorter time horizon?")
    print("    • Explain the Sharpe ratio in simple terms.")
    print("    • What if I want a more aggressive strategy?")
    print("    • Can you compare minimum variance vs max Sharpe?")
    print()
    print("  Type 'quit' or 'exit' to end the session.")
    print("-" * 60)

    while True:
        print()
        question = input("  You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "bye"):
            print()
            print("  Thank you for using the AI Financial Advisor!")
            print("  Your recommendation has been saved to: portfolio_recommendation.md")
            print("=" * 60)
            break

        print()
        print("  Thinking...\n")

        try:
            answer = run_followup(question, result)
            print(f"  Advisor: {answer}")
        except Exception as e:
            print(f"  Sorry, I encountered an error: {e}")
            print("  Please try rephrasing your question.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Survey
    survey_responses = conduct_survey()

    # Step 2: Run the full pipeline (analysis → optimizer → presentation)
    print("\n  Starting AI advisor pipeline...\n")
    try:
        result = run_initial_pipeline(survey_responses)
    except Exception as e:
        print(f"\n  Error running the advisor: {e}")
        return

    # Step 3: Show the recommendation
    print_header("YOUR PORTFOLIO RECOMMENDATION")
    print()
    print(result.portfolio_recommendation)

    # Step 4: Enter follow-up conversation
    conversation_loop(result)


if __name__ == "__main__":
    main()
