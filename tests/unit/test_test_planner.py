
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agents.planner_agent import (
    TestPlanner as Planner,
    TestPlanOutput as PlanOutput,
    TestScenarioOutput as ScenarioOutput
)


@pytest.fixture
def expected_result():
    return PlanOutput(
        test_plan_summary="Login functionality test plan",
        total_scenarios=1,
        test_scenarios=[
            ScenarioOutput(
                id="TC001",
                title="Successful login",
                type="positive",
                priority="high",
                description="Test login with valid credentials",
                expected_result="User successfully logs in"
            )
        ]
    )


@pytest.fixture
def mock_client(expected_result):
    client = MagicMock()

    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=expected_result
    )

    return client


@pytest.fixture
def planner(mock_client):
    return Planner(client=mock_client)


def test_generate_test_plan_rejects_empty_analysis(planner):
    with pytest.raises(
        ValueError,
        match="Requirements analysis cannot be empty"
    ):
        planner.generate_test_plan({})


def test_generate_test_plan_rejects_invalid_input(planner):
    with pytest.raises(
        ValueError,
        match="Requirements analysis cannot be empty"
    ):
        planner.generate_test_plan("Invalid input")


def test_generate_test_plan_returns_structured_result(
    planner,
    mock_client
):
    requirements_analysis = {
        "summary": "Users must be able to log in.",
        "suggested_test_scenarios": ["Valid login"]
    }

    result = planner.generate_test_plan(requirements_analysis)

    assert isinstance(result, dict)
    assert result["test_plan_summary"] == "Login functionality test plan"
    assert result["total_scenarios"] == 1
    assert len(result["test_scenarios"]) == 1

    scenario = result["test_scenarios"][0]

    assert scenario["id"] == "TC001"
    assert scenario["type"] == "positive"
    assert scenario["priority"] == "high"

    mock_client.responses.parse.assert_called_once()

    call_arguments = mock_client.responses.parse.call_args.kwargs

    assert call_arguments["model"] == "gpt-5.6"
    assert call_arguments["text_format"] is PlanOutput
    assert "Users must be able to log in." in call_arguments["input"]


def test_generate_test_plan_rejects_missing_structured_output(
    mock_client
):
    mock_client.responses.parse.return_value = SimpleNamespace(
        output_parsed=None
    )

    planner = Planner(client=mock_client)

    with pytest.raises(
        ValueError,
        match="did not return a structured test plan"
    ):
        planner.generate_test_plan({"summary": "Login"})


def test_generate_test_plan_uses_injected_client(mock_client):
    planner = Planner(client=mock_client)

    assert planner.client is mock_client
