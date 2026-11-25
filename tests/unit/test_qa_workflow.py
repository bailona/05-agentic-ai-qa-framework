
from unittest.mock import MagicMock, patch

import pytest

from src.core.qa_workflow import QAWorkflow


def create_workflow_without_api():
    workflow = QAWorkflow.__new__(QAWorkflow)

    workflow.requirements_analyzer = MagicMock()
    workflow.test_planner = MagicMock()

    return workflow


def test_workflow_rejects_empty_requirement():
    workflow = create_workflow_without_api()

    with pytest.raises(
        ValueError,
        match="Requirement cannot be empty"
    ):
        workflow.run("")


def test_workflow_rejects_whitespace_requirement():
    workflow = create_workflow_without_api()

    with pytest.raises(
        ValueError,
        match="Requirement cannot be empty"
    ):
        workflow.run("   ")


def test_analyze_requirements_returns_analysis():
    workflow = create_workflow_without_api()

    expected_analysis = {
        "summary": "Login functionality",
        "suggested_test_scenarios": ["Valid login"]
    }

    workflow.requirements_analyzer.analyze.return_value = (
        expected_analysis
    )

    result = workflow.analyze_requirements({
        "requirement": "Users must be able to log in."
    })

    assert result == {
        "requirements_analysis": expected_analysis
    }

    workflow.requirements_analyzer.analyze.assert_called_once_with(
        "Users must be able to log in."
    )


def test_generate_test_plan_returns_plan():
    workflow = create_workflow_without_api()

    requirements_analysis = {
        "summary": "Login functionality"
    }

    expected_plan = {
        "test_plan_summary": "Login test plan",
        "total_scenarios": 1,
        "test_scenarios": []
    }

    workflow.test_planner.generate_test_plan.return_value = (
        expected_plan
    )

    result = workflow.generate_test_plan({
        "requirements_analysis": requirements_analysis
    })

    assert result == {
        "test_plan": expected_plan
    }

    workflow.test_planner.generate_test_plan.assert_called_once_with(
        requirements_analysis
    )


def test_generate_test_plan_rejects_invalid_analysis():
    workflow = create_workflow_without_api()

    with pytest.raises(
        ValueError,
        match="Requirements analysis cannot be empty"
    ):
        workflow.generate_test_plan({
            "requirements_analysis": {}
        })


def test_workflow_executes_nodes_in_sequence():
    workflow = create_workflow_without_api()

    requirements_analysis = {
        "summary": "Login functionality"
    }

    expected_plan = {
        "test_plan_summary": "Login test plan"
    }

    workflow.requirements_analyzer.analyze.return_value = (
        requirements_analysis
    )

    workflow.test_planner.generate_test_plan.return_value = (
        expected_plan
    )

    from langgraph.graph import END, START, StateGraph

    from src.core.qa_workflow import QAWorkflowState

    graph = StateGraph(QAWorkflowState)

    graph.add_node(
        "analyze_requirements",
        workflow.analyze_requirements
    )

    graph.add_node(
        "generate_test_plan",
        workflow.generate_test_plan
    )

    graph.add_edge(START, "analyze_requirements")
    graph.add_edge("analyze_requirements", "generate_test_plan")
    graph.add_edge("generate_test_plan", END)

    workflow.app = graph.compile()

    result = workflow.run(
        "Users must be able to log in."
    )

    assert result["requirements_analysis"] == requirements_analysis
    assert result["test_plan"] == expected_plan

    workflow.requirements_analyzer.analyze.assert_called_once_with(
        "Users must be able to log in."
    )

    workflow.test_planner.generate_test_plan.assert_called_once_with(
        requirements_analysis
    )


def test_workflow_passes_knowledge_base_to_requirements_analyzer():
    from src.core.qa_workflow import QAWorkflow

    knowledge_base = MagicMock()
    requirements_analyzer = MagicMock()
    test_planner = MagicMock()

    with patch(
        "src.core.qa_workflow.RequirementsAnalyzer",
        return_value=requirements_analyzer
    ) as analyzer_class:
        workflow = QAWorkflow(
            knowledge_base=knowledge_base,
            test_planner=test_planner
        )

    analyzer_class.assert_called_once_with(
        knowledge_base=knowledge_base
    )

    assert workflow.requirements_analyzer is requirements_analyzer
    assert workflow.test_planner is test_planner


def test_workflow_accepts_injected_requirements_analyzer():
    requirements_analyzer = MagicMock()
    test_planner = MagicMock()

    workflow = QAWorkflow(
        requirements_analyzer=requirements_analyzer,
        test_planner=test_planner
    )

    assert workflow.requirements_analyzer is requirements_analyzer
    assert workflow.test_planner is test_planner
