from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


# ── LLM Configuration ──────────────────────────────────────────────────────
# CrewAI defaults to whatever OPENAI_MODEL_NAME is set in .env, which might
# be a tiny model like gpt-4.1-nano. We explicitly set a capable model here.
#
# Option A: OpenAI
#   llm = LLM(model="gpt-4o", temperature=0.2)
#
# Option B: Anthropic (set ANTHROPIC_API_KEY in .env)
#   llm = LLM(model="anthropic/claude-sonnet-4-20250514", temperature=0.2)
#
# Option C: Use environment variable CREWAI_LLM_MODEL
#   import os
#   llm = LLM(model=os.getenv("CREWAI_LLM_MODEL", "gpt-4o"), temperature=0.2)
#
# Change this to whichever model you prefer. gpt-4o is recommended as a
# minimum for financial reasoning tasks.

DEFAULT_LLM = LLM(model="gpt-4.1", temperature=0.2)


@CrewBase
class AiAdvisor():
    """
    Initial pipeline crew: survey analysis → risk assessment → portfolio presentation.
    The CPLEX optimizer runs between risk assessment and presentation (in Python code),
    so run_advisor.py splits the tasks into two separate crew runs.
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    # ── Agents ──────────────────────────────────────────────

    @agent
    def survey_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['survey_analyst'],
            llm=DEFAULT_LLM,
            verbose=True
        )

    @agent
    def risk_assessor(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_assessor'],
            llm=DEFAULT_LLM,
            verbose=True
        )

    @agent
    def portfolio_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_advisor'],
            llm=DEFAULT_LLM,
            verbose=True
        )

    # ── Tasks ───────────────────────────────────────────────

    @task
    def analyze_survey_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_survey_task'],
        )

    @task
    def assess_risk_task(self) -> Task:
        return Task(
            config=self.tasks_config['assess_risk_task'],
        )

    @task
    def present_portfolio_task(self) -> Task:
        return Task(
            config=self.tasks_config['present_portfolio_task'],
            output_file='portfolio_recommendation.md'
        )

    # ── Crew ────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
    