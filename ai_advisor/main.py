#!/usr/bin/env python
import sys
import warnings

from ai_advisor.crew import AiAdvisor
from ai_advisor.stocks import get_approved_universe_text

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run(survey_responses: str | None = None):
    """
    Run the crew with survey responses.
    If survey_responses is None, a small demo payload is used.
    """
    if survey_responses is None:
        survey_responses = (
            "Age: 30\n"
            "Employment: Full-time software engineer, $120k/year\n"
            "Dependents: None\n"
            "Monthly expenses: ~$3,500\n"
            "Savings: $45,000 in savings account\n"
            "Existing investments: $10,000 in a 401k\n"
            "Debts: $15,000 student loan\n"
            "Investment experience: Beginner\n"
            "How long do you plan to invest: 20+ years\n"
            "Risk comfort: I'm okay with some ups and downs\n"
            "Short-term goals: Build emergency fund\n"
            "Long-term goals: Retire by 55\n"
            "Other notes: None"
        )

    inputs = {
        'survey_responses': survey_responses,
        'approved_securities': get_approved_universe_text(),
    }

    try:
        result = AiAdvisor().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    run()