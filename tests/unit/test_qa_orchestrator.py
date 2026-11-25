
from unittest.mock import MagicMock, call, patch

import pytest

from src.core.qa_orchestrator import QAOrchestrator


def test_process_requirement_rejects_empty_requirement():
    orchestrator = QAOrchestrator.__new__(QAOrchestrator)

    with pytest.raises(
        ValueError,
        match="Requirement cannot be empty"
    ):
        orchestrator.process_requirement("")


def test_process_requirement_rejects_whitespace_requirement():
    orchestrator = QAOrchestrator.__new__(QAOrchestrator)

    with pytest.raises(
        ValueError,
        match="Requirement cannot be empty"
    ):
        orchestrator.process_requirement("   ")


def test_process_requirement_executes_both_agents():
    orchestrator = QAOrchestrator.__new__(QAOrchestrator)

    orchestrator.requirements_analyzer = MagicMock()
    orchestrator.test_planner = MagicMock()

    requirement = "Users must be able to log in."

    expected_analysis = {
        "summary": "Login functionality",
        "suggested_test_scenarios": ["Valid login"]
    }

    expected_test_plan = {
        "test_plan_summary": "Login test plan",
        "total_scenarios": 1,
        "test_scenarios": [
            {
                "id": "TC001",
                "title": "Successful login",
                "type": "positive",
                "priority": "high",
                "description": "Login with valid credentials",
                "expected_result": "User successfully logs in"
            }
        ]
    }

    orchestrator.requirements_analyzer.analyze.return_value = (
        expected_analysis
    )

    orchestrator.test_planner.generate_test_plan.return_value = (
        expected_test_plan
    )

    result = orchestrator.process_requirement(requirement)

    assert result == {
        "requirements_analysis": expected_analysis,
        "test_plan": expected_test_plan
    }

    orchestrator.requirements_analyzer.analyze.assert_called_once_with(
        requirement
    )

    orchestrator.test_planner.generate_test_plan.assert_called_once_with(
        expected_analysis
    )


def test_process_requirement_executes_agents_in_sequence():
    orchestrator = QAOrchestrator.__new__(QAOrchestrator)

    orchestrator.requirements_analyzer = MagicMock()
    orchestrator.test_planner = MagicMock()

    orchestrator.requirements_analyzer.analyze.return_value = {
        "summary": "Login functionality"
    }

    orchestrator.test_planner.generate_test_plan.return_value = {
        "test_plan_summary": "Login test plan"
    }

    orchestrator.process_requirement("Users must be able to log in.")

    assert orchestrator.requirements_analyzer.mock_calls == [
        call.analyze("Users must be able to log in.")
    ]

    orchestrator.test_planner.generate_test_plan.assert_called_once_with(
        {"summary": "Login functionality"}
    )


def test_orchestrator_passes_knowledge_base_to_requirements_analyzer():
    knowledge_base = MagicMock()

    with patch(
        "src.core.qa_orchestrator.RequirementsAnalyzer"
    ) as analyzer_class, patch(
        "src.core.qa_orchestrator.TestPlanner"
    ) as planner_class:

        orchestrator = QAOrchestrator(
            knowledge_base=knowledge_base
        )

    analyzer_class.assert_called_once_with(
        knowledge_base=knowledge_base
    )

    planner_class.assert_called_once_with()

    assert orchestrator.requirements_analyzer is analyzer_class.return_value
    assert orchestrator.test_planner is planner_class.return_value


def test_orchestrator_accepts_injected_agents():
    requirements_analyzer = MagicMock()
    test_planner = MagicMock()

    orchestrator = QAOrchestrator(
        requirements_analyzer=requirements_analyzer,
        test_planner=test_planner
    )

    assert orchestrator.requirements_analyzer is requirements_analyzer
    assert orchestrator.test_planner is test_planner
