#!/usr/bin/env python
"""
AI Financial Advisor — CLI Chatbot
Conducts a financial survey, then passes responses to the CrewAI pipeline
for risk assessment and portfolio recommendation.
"""

from dotenv import load_dotenv
load_dotenv()
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from ai_advisor.crew import AiAdvisor
from ai_advisor.stocks import get_approved_universe_text

# ── Survey Definition ───────────────────────────────────────────────────────
# Each question has: key, prompt text, and optional choices for guided input.

SURVEY_SECTIONS = [
    {
        "title": "PERSONAL INFORMATION",
        "questions": [
            {
                "key": "age",
                "prompt": "How old are you?",
            },
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
            {
                "key": "dependents",
                "prompt": "How many financial dependents do you have (children, elderly parents, etc.)?",
            },
        ],
    },
    {
        "title": "INCOME & EXPENSES",
        "questions": [
            {
                "key": "annual_income",
                "prompt": "What is your approximate annual income (before tax)?",
            },
            {
                "key": "monthly_expenses",
                "prompt": "What are your approximate total monthly expenses?",
            },
            {
                "key": "monthly_savings",
                "prompt": "How much do you save or invest per month (approximately)?",
            },
        ],
    },
    {
        "title": "ASSETS & LIABILITIES",
        "questions": [
            {
                "key": "total_savings",
                "prompt": "What is the total value of your liquid savings (checking + savings accounts)?",
            },
            {
                "key": "existing_investments",
                "prompt": "What is the total value of your current investments (401k, IRA, brokerage, etc.)? Enter 0 if none.",
            },
            {
                "key": "total_debt",
                "prompt": "What is your total outstanding debt (student loans, credit cards, mortgage, etc.)? Enter 0 if none.",
            },
            {
                "key": "debt_details",
                "prompt": "Briefly describe your debts (e.g., '$20k student loan, $5k credit card'). Type 'none' if no debt.",
            },
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
                    "3. Intermediate — I invest regularly and understand the basics",
                    "4. Advanced — I actively manage a diversified portfolio",
                ],
            },
            {
                "key": "current_holdings",
                "prompt": "What types of investments do you currently hold (if any)? e.g., stocks, ETFs, bonds, crypto, real estate. Type 'none' if none.",
            },
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
            {
                "key": "investment_amount",
                "prompt": "How much are you looking to invest initially?",
            },
            {
                "key": "short_term_goals",
                "prompt": "What are your short-term financial goals (next 1-3 years)? e.g., emergency fund, vacation, down payment.",
            },
            {
                "key": "long_term_goals",
                "prompt": "What are your long-term financial goals? e.g., retirement, children's education, financial independence.",
            },
        ],
    },
    {
        "title": "RISK TOLERANCE",
        "questions": [
            {
                "key": "risk_comfort",
                "prompt": "How would you react if your portfolio dropped 20% in one month?",
                "choices": [
                    "1. I would sell everything immediately — I can't afford to lose money",
                    "2. I'd be very uncomfortable and would consider selling some",
                    "3. I'd be concerned but would hold and wait for recovery",
                    "4. I'd stay the course — market dips are normal",
                    "5. I'd see it as a buying opportunity and invest more",
                ],
            },
            {
                "key": "return_vs_safety",
                "prompt": "Which statement best describes your priority?",
                "choices": [
                    "1. Protect my capital — I'd rather earn less than risk losing money",
                    "2. Mostly safe with a little growth potential",
                    "3. A balance between growth and safety",
                    "4. Mostly growth — I can handle short-term losses",
                    "5. Maximum growth — I'm comfortable with high volatility",
                ],
            },
            {
                "key": "loss_tolerance",
                "prompt": "What is the maximum percentage loss you could tolerate in a single year before you'd want to change strategy?",
                "choices": [
                    "1. 5% or less",
                    "2. 10%",
                    "3. 20%",
                    "4. 30%",
                    "5. 40% or more — I'm in it for the long haul",
                ],
            },
        ],
    },
    {
        "title": "ADDITIONAL INFORMATION",
        "questions": [
            {
                "key": "special_considerations",
                "prompt": "Any other details you'd like the advisor to know? (e.g., upcoming large expenses, ethical/ESG preferences, tax considerations). Type 'none' if nothing.",
            },
        ],
    },
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def print_header(text: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def ask_question(question: dict) -> str:
    """Ask a single question and return the answer."""
    print()
    print(f"  {question['prompt']}")

    if "choices" in question:
        for choice in question["choices"]:
            print(f"    {choice}")
        print()
        while True:
            answer = input("  Your choice (enter number): ").strip()
            valid_numbers = [str(i + 1) for i in range(len(question["choices"]))]
            if answer in valid_numbers:
                return question["choices"][int(answer) - 1]
            print("  Please enter a valid option number.")
    else:
        answer = input("  Your answer: ").strip()
        while not answer:
            answer = input("  Please enter a response: ").strip()
        return answer


def conduct_survey() -> str:
    """Run the interactive survey and return formatted responses as a string."""
    print_header("AI FINANCIAL ADVISOR")
    print("  Welcome! I'll ask you a series of questions about your")
    print("  financial situation, goals, and risk tolerance.")
    print("  Your answers will be used to generate a personalized")
    print("  portfolio recommendation.")
    print()
    print("  Type your answers when prompted. For multiple-choice")
    print("  questions, enter the number of your choice.")
    print("-" * 60)

    responses: dict[str, str] = {}

    for section in SURVEY_SECTIONS:
        print_header(section["title"])
        for question in section["questions"]:
            answer = ask_question(question)
            responses[question["key"]] = answer

    formatted = format_responses(responses)

    print_header("SURVEY COMPLETE")
    print("  Thank you! Processing your responses...")
    print("  The AI advisor crew is now analyzing your profile.")
    print("-" * 60)

    return formatted


def format_responses(responses: dict[str, str]) -> str:
    """Format the survey responses into a clean text block for the crew."""
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
    for key, answer in responses.items():
        label = labels.get(key, key)
        lines.append(f"{label}: {answer}")

    return "\n".join(lines)


# ── Main Entry Point ────────────────────────────────────────────────────────

def main():
    # Step 1: Conduct the interactive survey
    survey_responses = conduct_survey()

    # Step 2: Feed survey responses into the CrewAI pipeline
    inputs = {
        "survey_responses": survey_responses,
        "approved_securities": get_approved_universe_text(),
    }

    print()
    print("  Starting AI advisor crew...\n")

    try:
        result = AiAdvisor().crew().kickoff(inputs=inputs)
    except Exception as e:
        print(f"\n  Error running the advisor crew: {e}")
        return

    # Step 3: Display the final recommendation
    print_header("YOUR PORTFOLIO RECOMMENDATION")
    print()
    print(result)
    print()
    print("-" * 60)
    print("  A copy has been saved to: portfolio_recommendation.md")
    print("  Thank you for using the AI Financial Advisor!")
    print("=" * 60)


if __name__ == "__main__":
    main()