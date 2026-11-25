
import json
from unittest.mock import MagicMock

import pytest

from src.agents.planner_agent import TestPlanner as Planner


def test_generate_test_plan_rejects_empty_analysis():
    planner = Planner.__new__(Planner)

    with pytest.raises(
        ValueError,
        match="Requirements analysis cannot be empty"
    ):
        planner.generate_test_plan({})


def test_generate_test_plan_rejects_invalid_input():
    planner = Planner.__new__(Planner)

    with pytest.raises(
        ValueError,
        match="Requirements analysis cannot be empty"
    ):
        planner.generate_test_plan("Invalid input")


def test_generate_test_plan_returns_valid_json():
    planner = Planner.__new__(Planner)
    planner.client = MagicMock()
    planner.model = "gpt-5.6"

    expected_result = {
        "test_plan_summary": "Login functionality test plan",
        "total_scenarios": 3,
        "test_scenarios": [
            {
                "id": "TC001",
                "title": "Successful login",
                "type": "positive",
                "priority": "high",
                "description": "Test login with valid credentials",
                "expected_result": "User successfully logs in"
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.output_text = json.dumps(expected_result)

    planner.client.responses.create.return_value = mock_response

    requirements_analysis = {
        "summary": "Users must be able to log in.",
        "suggested_test_scenarios": ["Valid login"]
    }

    result = planner.generate_test_plan(requirements_analysis)

    assert result == expected_result
    planner.client.responses.create.assert_called_once()


def test_generate_test_plan_rejects_invalid_json():
    planner = Planner.__new__(Planner)
    planner.client = MagicMock()
    planner.model = "gpt-5.6"

    mock_response = MagicMock()
    mock_response.output_text = "Invalid JSON response"

    planner.client.responses.create.return_value = mock_response

    with pytest.raises(
        ValueError,
        match="invalid JSON response"
    ):
        planner.generate_test_plan({"summary": "Login"})


def test_generate_test_plan_rejects_non_object_json():
    planner = Planner.__new__(Planner)
    planner.client = MagicMock()
    planner.model = "gpt-5.6"

    mock_response = MagicMock()
    mock_response.output_text = json.dumps(["TC001", "TC002"])

    planner.client.responses.create.return_value = mock_response

    with pytest.raises(
        ValueError,
        match="must be a JSON object"
    ):
        planner.generate_test_plan({"summary": "Login"})
