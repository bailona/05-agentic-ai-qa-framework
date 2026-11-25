
from src.agents.requirements_analyzer import RequirementsAnalyzer
from src.agents.planner_agent import TestPlanner
from src.retrieval.knowledge_base import KnowledgeBase


class QAOrchestrator:
    """Coordinates AI agents to analyze requirements and generate test plans."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        requirements_analyzer=None,
        test_planner=None
    ):
        self.requirements_analyzer = (
            requirements_analyzer
            if requirements_analyzer is not None
            else RequirementsAnalyzer(
                knowledge_base=knowledge_base
            )
        )

        self.test_planner = (
            test_planner
            if test_planner is not None
            else TestPlanner()
        )

    def process_requirement(self, requirement: str) -> dict:
        """
        Executes the QA workflow:
        1. Analyze the software requirement.
        2. Generate a test plan based on the analysis.
        3. Return both results.
        """
        if not requirement or not requirement.strip():
            raise ValueError("Requirement cannot be empty.")

        requirements_analysis = self.requirements_analyzer.analyze(
            requirement
        )

        test_plan = self.test_planner.generate_test_plan(
            requirements_analysis
        )

        return {
            "requirements_analysis": requirements_analysis,
            "test_plan": test_plan
        }
