#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from ai_advisor.run_advisor import run_initial_pipeline


def run(survey_responses: str | None = None):
    if survey_responses is None:
        survey_responses = (
            "Age: 30\n"
            "Employment Status: Full-time employed\n"
            "Number of Dependents: 0\n"
            "Annual Income (pre-tax): $120,000\n"
            "Monthly Expenses: $3,500\n"
            "Monthly Savings/Investment: $2,000\n"
            "Total Liquid Savings: $45,000\n"
            "Existing Investments Value: $10,000 in 401k\n"
            "Total Outstanding Debt: $15,000\n"
            "Debt Details: $15k student loan\n"
            "Investment Experience: Beginner\n"
            "Current Holdings: 401k target-date fund\n"
            "Investment Horizon: 20+ years\n"
            "Initial Investment Amount: $20,000\n"
            "Short-term Goals: Build emergency fund\n"
            "Long-term Goals: Retire by 55\n"
            "Reaction to 20% Portfolio Drop: Stay the course\n"
            "Growth vs Safety Priority: Mostly growth\n"
            "Maximum Tolerable Annual Loss: 20%\n"
            "Special Considerations: None\n"
        )

    result = run_initial_pipeline(survey_responses)
    print(result.portfolio_recommendation)
    return result


if __name__ == "__main__":
    run()
