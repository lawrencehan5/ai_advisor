"""
ai_advisor/run_advisor.py

Orchestrates the full pipeline:
  1. LLM extraction    — GPT-4.1 extracts structured data from raw survey
  2. CrewAI agents     — survey analysis + risk assessment
  3. LLM extraction    — GPT-4.1 extracts risk category/score/strategy from agent output
  4. CPLEX optimizer   — portfolio optimization
  5. CrewAI agent      — present results
  6. Follow-up Q&A     — CrewAI agent built manually
"""

from dotenv import load_dotenv
load_dotenv()

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI
from crewai import Agent, Task, Crew, Process

from ai_advisor.crew import AiAdvisor, DEFAULT_LLM
from ai_advisor.stocks import get_approved_universe_text, get_all_tickers
from ai_advisor.market_data import get_market_context
from ai_advisor.optimizer import (
    select_strategy,
    run_optimization,
    get_strategies_description,
    OptimizationResult,
    STRATEGIES,
)

# ── OpenAI client for structured extraction ─────────────────────────────────

client = OpenAI()
EXTRACTION_MODEL = "gpt-4.1"


# ── Structured Output ───────────────────────────────────────────────────────

@dataclass
class AdvisorResult:
    """Full result from the advisor pipeline."""
    survey_responses: str = ""
    financial_profile: str = ""
    risk_assessment: str = ""
    risk_category: str = ""
    risk_score: str = ""
    optimizer_strategy: str = ""
    optimization_result: OptimizationResult = field(default_factory=OptimizationResult)
    portfolio_recommendation: str = ""
    allocations: list[dict] = field(default_factory=list)
    excluded_tickers: list[str] = field(default_factory=list)
    included_tickers: list[str] = field(default_factory=list)
    market_context: str = ""
    experience_level: str = ""
    investment_horizon: str = ""
    investment_style: str = ""
    leverage_comfort: str = ""


# ── LLM Extraction (replaces all regex parsers) ────────────────────────────

def _extract_survey_data(survey_text: str) -> dict:
    """
    Use GPT-4.1 to extract structured data from raw survey responses.
    Handles any input format: "$100k", "a hundred thousand", "100,000", etc.
    Returns a dict with: investment_amount, excluded_tickers, included_tickers
    """
    all_tickers = get_all_tickers()

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured financial data from survey responses. "
                    "Return ONLY valid JSON with no markdown, no backticks, no explanation.\n\n"
                    f"Known tickers: {', '.join(all_tickers)}\n\n"
                    "Return this exact JSON structure:\n"
                    '{\n'
                    '  "investment_amount": <number in dollars, e.g. 100000 for "100k" or "a hundred thousand">,\n'
                    '  "excluded_tickers": [<list of ticker strings the user wants EXCLUDED>],\n'
                    '  "included_tickers": [<list of ticker strings the user wants INCLUDED or prefers>],\n'
                    '  "experience_level": "<one of: None, Beginner, Intermediate, Advanced>",\n'
                    '  "investment_horizon": "<one of: <1yr, 1-3yr, 3-5yr, 5-10yr, 10-20yr, 20+yr>",\n'
                    '  "investment_style": "<one of: passive, slightly_passive, neutral, slightly_active, active>",\n'
                    '  "leverage_comfort": "<one of: no, maybe, yes>"\n'
                    '}\n\n'
                    "Rules:\n"
                    '- Convert ALL money formats to a plain number: "100k"→100000, "$50,000"→50000, '
                    '"a hundred thousand"→100000, "1.5m"→1500000, "fifty thousand"→50000\n'
                    "- If user says exclude/remove/avoid/no/don't include a ticker, put it in excluded_tickers\n"
                    "- If user says include/prefer/want/must have a ticker, put it in included_tickers\n"
                    "- If no investment amount found, use 10000\n"
                    "- If no exclusions/inclusions, use empty lists\n"
                    '- experience_level: map first word of answer — "None"→None, "Beginner"→Beginner, '
                    '"Intermediate"→Intermediate, "Advanced"→Advanced. Default: None\n'
                    '- investment_horizon: map "Less than 1 year"→<1yr, "1–3 years"→1-3yr, '
                    '"3–5 years"→3-5yr, "5–10 years"→5-10yr, "10–20 years"→10-20yr, "20+ years"→20+yr. Default: 3-5yr\n'
                    '- investment_style: map first word of answer — "Passive"→passive, '
                    '"Slightly"→slightly_passive or slightly_active (check full text), '
                    '"No"→neutral, "Active"→active. Default: neutral\n'
                    '- leverage_comfort: "No —"→no, "Maybe —"→maybe, "Yes —"→yes. Default: no'
                ),
            },
            {
                "role": "user",
                "content": survey_text,
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Clean potential markdown wrapping
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return {
        "investment_amount": float(data.get("investment_amount", 10000)),
        "excluded_tickers": [t.upper() for t in data.get("excluded_tickers", [])],
        "included_tickers": [t.upper() for t in data.get("included_tickers", [])],
        "experience_level": data.get("experience_level", "None"),
        "investment_horizon": data.get("investment_horizon", "3-5yr"),
        "investment_style": data.get("investment_style", "neutral"),
        "leverage_comfort": data.get("leverage_comfort", "no"),
    }


def _extract_risk_data(risk_text: str, profile_text: str) -> dict:
    """
    Use GPT-4.1 to extract risk category, score, and strategy from agent output.
    """
    strategy_keys = list(STRATEGIES.keys())

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured risk assessment data from financial advisor agent output. "
                    "Return ONLY valid JSON with no markdown, no backticks, no explanation.\n\n"
                    f"Valid strategies: {', '.join(strategy_keys)}\n"
                    "Valid risk categories: CONSERVATIVE, MODERATE, BALANCED, GROWTH, AGGRESSIVE\n\n"
                    "Return this exact JSON structure:\n"
                    '{\n'
                    '  "risk_category": "<one of the valid categories>",\n'
                    '  "risk_score": "<1-10>",\n'
                    '  "optimizer_strategy": "<one of the valid strategy keys>",\n'
                    '  "investment_amount": <number if mentioned, else null>\n'
                    '}'
                ),
            },
            {
                "role": "user",
                "content": f"RISK ASSESSMENT:\n{risk_text}\n\nFINANCIAL PROFILE:\n{profile_text}",
            },
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return {
        "risk_category": data.get("risk_category", "BALANCED"),
        "risk_score": str(data.get("risk_score", "")),
        "optimizer_strategy": data.get("optimizer_strategy", ""),
        "investment_amount": data.get("investment_amount"),
    }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _select_tickers_with_ai(
    risk_category: str,
    excluded_tickers: list[str],
    included_tickers: list[str],
    profile_text: str,
    risk_text: str,
    market_context: str = "",
) -> list[str]:
    """
    Use GPT-4.1 to pick which tickers to include in the portfolio,
    based on risk category, user constraints, the full profile,
    AND real-time market data.
    """
    all_tickers = get_all_tickers()

    system_prompt = (
        "You are a portfolio construction expert. Select 8-15 tickers "
        "from the approved universe for the user's portfolio.\n\n"
        f"APPROVED TICKERS: {', '.join(all_tickers)}\n\n"
        "Return ONLY a JSON array of ticker strings. No explanation, "
        "no markdown, no backticks. Example: [\"VOO\",\"BND\",\"AAPL\"]\n\n"
        "Rules:\n"
        "- NEVER include tickers the user asked to exclude\n"
        "- Always include tickers the user asked to include\n"
        "- CONSERVATIVE: heavy bonds/fixed-income ETFs, defensive stocks\n"
        "- MODERATE: mix of bonds and broad equity ETFs\n"
        "- BALANCED: equal bonds and equities, some alternatives\n"
        "- GROWTH: equity-heavy, growth stocks, less bonds\n"
        "- AGGRESSIVE: mostly growth stocks and equity ETFs, minimal bonds\n"
        "- Diversify across sectors\n"
        "- USE the real-time market data below to inform your picks:\n"
        "  - Favor stocks/ETFs with positive recent momentum where appropriate\n"
        "  - Consider current valuations (P/E ratios)\n"
        "  - Factor in current market conditions and sector trends\n"
        "  - For conservative portfolios, favor stable/low-volatility picks\n"
        "  - For growth portfolios, favor strong recent performers\n"
    )

    user_prompt = (
        f"Risk Category: {risk_category}\n"
        f"Excluded tickers: {excluded_tickers if excluded_tickers else 'none'}\n"
        f"Must-include tickers: {included_tickers if included_tickers else 'none'}\n\n"
        f"User Profile:\n{profile_text[:500]}\n\n"
        f"Risk Assessment:\n{risk_text[:500]}\n\n"
    )

    if market_context:
        user_prompt += f"--- REAL-TIME MARKET DATA ---\n{market_context}\n--- END MARKET DATA ---\n"

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        tickers = json.loads(raw)
        # Validate: only approved tickers, no excluded ones
        valid = set(t.upper() for t in all_tickers)
        excluded_set = set(t.upper() for t in excluded_tickers)
        tickers = [t.upper() for t in tickers if t.upper() in valid and t.upper() not in excluded_set]

        # Make sure included tickers are in
        for t in included_tickers:
            if t.upper() in valid and t.upper() not in tickers:
                tickers.append(t.upper())

        return tickers if tickers else list(valid)[:10]
    except (json.JSONDecodeError, TypeError):
        return list(set(all_tickers) - set(excluded_tickers))[:10]


def _format_optimizer_allocations(opt_result: OptimizationResult) -> str:
    lines = []
    for a in opt_result.allocations:
        pct = a["weight"] * 100
        lines.append(f"  - {a['ticker']}: {pct:.1f}%")
    return "\n".join(lines)


# ── Pipeline Runner ─────────────────────────────────────────────────────────

def run_initial_pipeline(
    survey_responses: str,
    on_progress: callable = None,
) -> AdvisorResult:
    """
    Full pipeline with progress reporting.

    Args:
        survey_responses: Formatted survey text
        on_progress: Optional callback(stage_label: str) called at each stage
    """

    def progress(msg: str):
        print(f"  {msg}")
        if on_progress:
            on_progress(msg)

    # ── Phase 0: Extract structured data from raw survey ────────
    progress("Reading your survey responses")
    survey_data = _extract_survey_data(survey_responses)
    investment_amount = survey_data["investment_amount"]
    excluded_tickers = survey_data["excluded_tickers"]
    included_tickers = survey_data["included_tickers"]
    experience_level = survey_data["experience_level"]
    investment_horizon = survey_data["investment_horizon"]
    investment_style = survey_data["investment_style"]
    leverage_comfort = survey_data["leverage_comfort"]

    print(f"  Parsed: investment=${investment_amount:,.0f}, "
          f"exclude={excluded_tickers}, include={included_tickers}, "
          f"experience={experience_level}, horizon={investment_horizon}, "
          f"style={investment_style}, leverage={leverage_comfort}")

    # ── Phase 0b: Fetch real-time market data ───────────────────
    progress("Fetching real-time stock prices and market news")
    try:
        market_context = get_market_context()
    except Exception as e:
        print(f"  Warning: Could not fetch market data: {e}")
        market_context = ""

    # ── Phase 0b: Fetch real-time market data ───────────────────
    progress("Fetching real-time stock prices and market news")
    try:
        market_context = get_market_context()
    except Exception as e:
        print(f"  Warning: Could not fetch market data: {e}")
        market_context = ""

    # ── Phase 1: Survey analysis + risk assessment ──────────────
    progress("Building your financial profile")
    advisor = AiAdvisor()
    full_crew = advisor.crew()

    analysis_crew = Crew(
        agents=[full_crew.agents[0], full_crew.agents[1]],
        tasks=[full_crew.tasks[0], full_crew.tasks[1]],
        process=Process.sequential,
        verbose=True,
    )

    phase1_inputs = {
        "survey_responses": survey_responses,
        "approved_securities": get_approved_universe_text(),
        "available_strategies": get_strategies_description(),
        # Placeholders for task 3 (not used in phase 1)
        "optimizer_strategy": "",
        "expected_return": "",
        "expected_volatility": "",
        "sharpe_ratio": "",
        "optimizer_allocations": "",
        "risk_category": "",
        "investment_amount": f"${investment_amount:,.0f}",
        "user_constraints": "",
        "financial_profile_summary": "",
        "risk_assessment_summary": "",
    }

    analysis_result = analysis_crew.kickoff(inputs=phase1_inputs)

    profile_text = str(analysis_crew.tasks[0].output) if analysis_crew.tasks[0].output else ""
    risk_text = str(analysis_crew.tasks[1].output) if analysis_crew.tasks[1].output else str(analysis_result)

    # ── Phase 1b: Extract risk data from agent output ───────────
    progress("Assessing your risk tolerance")
    risk_data = _extract_risk_data(risk_text, profile_text)

    risk_category = risk_data["risk_category"]
    risk_score = risk_data["risk_score"]
    optimizer_strategy = risk_data["optimizer_strategy"]

    # If the agent found a different investment amount, prefer it
    if risk_data["investment_amount"] and risk_data["investment_amount"] >= 100:
        investment_amount = risk_data["investment_amount"]

    # ── Phase 2: AI selects tickers → run optimizer ───────────
    progress("Selecting stocks and ETFs using current market data")
    tickers = _select_tickers_with_ai(
        risk_category=risk_category,
        excluded_tickers=excluded_tickers,
        included_tickers=included_tickers,
        profile_text=profile_text,
        risk_text=risk_text,
        market_context=market_context,
    )

    strategy = select_strategy(
        risk_category,
        optimizer_strategy,
        experience_level=experience_level,
        investment_horizon=investment_horizon,
        investment_style=investment_style,
        leverage_comfort=leverage_comfort,
    )

    print(f"  Risk Category: {risk_category}")
    print(f"  Optimizer Strategy: {strategy}")
    print(f"  Investment Amount: ${investment_amount:,.0f}")
    print(f"  Selected Tickers ({len(tickers)}): {', '.join(tickers)}")
    if excluded_tickers:
        print(f"  Excluded: {', '.join(excluded_tickers)}")
    print()

    progress(f"Optimizing portfolio weights ({strategy.replace('_', ' ')})")
    opt_result = run_optimization(strategy, tickers, investment_amount)

    # ── Phase 3: Present results with FULL context ──────────────
    progress("Preparing your personalized recommendation")
    advisor2 = AiAdvisor()
    full_crew2 = advisor2.crew()

    presentation_crew = Crew(
        agents=[full_crew2.agents[2]],
        tasks=[full_crew2.tasks[2]],
        process=Process.sequential,
        verbose=True,
    )

    # Build constraint summary
    constraint_lines = []
    if excluded_tickers:
        constraint_lines.append(f"Excluded tickers (per user request): {', '.join(excluded_tickers)}")
    if included_tickers:
        constraint_lines.append(f"User-preferred tickers: {', '.join(included_tickers)}")
    constraints_text = "\n".join(constraint_lines) if constraint_lines else "None"

    alloc_text = _format_optimizer_allocations(opt_result)
    if constraint_lines:
        alloc_text += f"\n\n  User Constraints Applied:\n  " + "\n  ".join(constraint_lines)
    if market_context:
        alloc_text += f"\n\n{market_context}"

    presentation_inputs = {
        # Placeholders for tasks 1 & 2
        "survey_responses": "",
        "available_strategies": "",
        "approved_securities": "",
        # Actual values
        "optimizer_strategy": opt_result.strategy_display_name,
        "expected_return": f"{opt_result.expected_return:.1%}",
        "expected_volatility": f"{opt_result.expected_volatility:.1%}",
        "sharpe_ratio": f"{opt_result.sharpe_ratio:.2f}",
        "optimizer_allocations": alloc_text,
        "risk_category": risk_category,
        "investment_amount": f"${investment_amount:,.0f}",
        "user_constraints": constraints_text,
        "financial_profile_summary": profile_text,
        "risk_assessment_summary": risk_text,
    }

    presentation_result = presentation_crew.kickoff(inputs=presentation_inputs)
    portfolio_text = str(presentation_crew.tasks[0].output) if presentation_crew.tasks[0].output else str(presentation_result)

    return AdvisorResult(
        survey_responses=survey_responses,
        financial_profile=profile_text,
        risk_assessment=risk_text,
        risk_category=risk_category,
        risk_score=risk_score,
        optimizer_strategy=optimizer_strategy or opt_result.strategy_used,
        optimization_result=opt_result,
        portfolio_recommendation=portfolio_text,
        allocations=opt_result.allocations,
        excluded_tickers=excluded_tickers,
        included_tickers=included_tickers,
        market_context=market_context,
        experience_level=experience_level,
        investment_horizon=investment_horizon,
        investment_style=investment_style,
        leverage_comfort=leverage_comfort,
    )


# ── Follow-up ───────────────────────────────────────────────────────────────

def run_followup(question: str, advisor_result: AdvisorResult) -> str:
    """Answer a follow-up question with full context."""

    followup_agent = Agent(
        role="Financial Advisor Assistant",
        goal=(
            "Answer the user's follow-up questions about their portfolio "
            "recommendation. Explain decisions, compare strategies, and help "
            "them understand their options. Always reference the user's specific "
            "financial situation when answering."
        ),
        backstory=(
            "You are a knowledgeable financial advisor with full context of "
            "the user's financial profile, risk assessment, and portfolio "
            "recommendation. You can explain any aspect of the recommendation "
            "in detail, discuss alternative strategies, answer 'what if' "
            "questions, and help the user feel confident in their investment "
            "plan. You are patient, thorough, and always relate answers back "
            "to the user's specific situation."
        ),
        llm=DEFAULT_LLM,
        verbose=True,
    )

    context = (
        f"--- USER'S ORIGINAL SURVEY RESPONSES ---\n"
        f"{advisor_result.survey_responses}\n"
        f"--- END SURVEY ---\n\n"
        f"--- STRUCTURED FINANCIAL PROFILE ---\n"
        f"{advisor_result.financial_profile}\n"
        f"--- END PROFILE ---\n\n"
        f"--- RISK ASSESSMENT ---\n"
        f"Risk Category: {advisor_result.risk_category}\n"
        f"Risk Score: {advisor_result.risk_score}/10\n"
        f"Optimizer Strategy: {advisor_result.optimizer_strategy}\n"
        f"{advisor_result.risk_assessment}\n"
        f"--- END RISK ---\n\n"
        f"--- PORTFOLIO RECOMMENDATION ---\n"
        f"{advisor_result.portfolio_recommendation}\n"
        f"--- END RECOMMENDATION ---\n\n"
    )

    if advisor_result.excluded_tickers:
        context += f"Excluded tickers (user requested): {', '.join(advisor_result.excluded_tickers)}\n\n"

    if advisor_result.market_context:
        context += f"--- REAL-TIME MARKET DATA (from session start) ---\n{advisor_result.market_context}\n--- END MARKET DATA ---\n\n"

    context += (
        f"Available optimization strategies:\n"
        f"{get_strategies_description()}\n\n"
        f"--- USER'S QUESTION ---\n"
        f"{question}\n"
        f"--- END QUESTION ---\n\n"
        f"Answer the user's question thoroughly. Reference their specific "
        f"financial situation when explaining."
    )

    followup_task = Task(
        description=context,
        expected_output=(
            "A clear, helpful answer that references the user's specific "
            "financial profile, goals, and constraints."
        ),
        agent=followup_agent,
    )

    followup_crew = Crew(
        agents=[followup_agent],
        tasks=[followup_task],
        process=Process.sequential,
        verbose=True,
    )

    result = followup_crew.kickoff()
    return str(result)