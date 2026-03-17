from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class AiAdvisor():
    """AiAdvisor crew — Financial Advisory Pipeline"""

    agents: list[BaseAgent]
    tasks: list[Task]

    # ── Agents ──────────────────────────────────────────────

    @agent
    def survey_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['survey_analyst'],  # type: ignore[index]
            verbose=True
        )

    @agent
    def risk_assessor(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_assessor'],  # type: ignore[index]
            verbose=True
        )

    @agent
    def portfolio_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['portfolio_advisor'],  # type: ignore[index]
            verbose=True
        )

    # ── Tasks (executed sequentially in this order) ─────────

    @task
    def analyze_survey_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_survey_task'],  # type: ignore[index]
        )

    @task
    def assess_risk_task(self) -> Task:
        return Task(
            config=self.tasks_config['assess_risk_task'],  # type: ignore[index]
        )

    @task
    def recommend_portfolio_task(self) -> Task:
        return Task(
            config=self.tasks_config['recommend_portfolio_task'],  # type: ignore[index]
            output_file='portfolio_recommendation.md'
        )

    # ── Crew ────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Creates the AiAdvisor crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )