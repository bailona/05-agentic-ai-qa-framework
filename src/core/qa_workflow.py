
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.requirements_analyzer import RequirementsAnalyzer
from src.agents.planner_agent import TestPlanner
from src.retrieval.knowledge_base import KnowledgeBase


class QAWorkflowState(TypedDict, total=False):
    requirement: str
    requirements_analysis: dict
    test_plan: dict


class QAWorkflow:
    """LangGraph workflow for automated QA requirement analysis."""

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

        workflow = StateGraph(QAWorkflowState)

        workflow.add_node(
            "analyze_requirements",
            self.analyze_requirements
        )

        workflow.add_node(
            "generate_test_plan",
            self.generate_test_plan
        )

        workflow.add_edge(START, "analyze_requirements")
        workflow.add_edge(
            "analyze_requirements",
            "generate_test_plan"
        )
        workflow.add_edge("generate_test_plan", END)

        self.app = workflow.compile()

    def analyze_requirements(self, state: QAWorkflowState) -> dict:
        requirement = state.get("requirement", "")

        if not requirement or not requirement.strip():
            raise ValueError("Requirement cannot be empty.")

        analysis = self.requirements_analyzer.analyze(requirement)

        return {
            "requirements_analysis": analysis
        }

    def generate_test_plan(self, state: QAWorkflowState) -> dict:
        analysis = state.get("requirements_analysis")

        if not isinstance(analysis, dict) or not analysis:
            raise ValueError("Requirements analysis cannot be empty.")

        test_plan = self.test_planner.generate_test_plan(analysis)

        return {
            "test_plan": test_plan
        }

    def run(self, requirement: str) -> dict:
        if not requirement or not requirement.strip():
            raise ValueError("Requirement cannot be empty.")

        return self.app.invoke({
            "requirement": requirement
        })
