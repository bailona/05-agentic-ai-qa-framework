
import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel


class TestScenarioOutput(BaseModel):
    """Structured output for an individual test scenario."""

    id: str
    title: str
    type: Literal["positive", "negative", "edge_case"]
    priority: Literal["high", "medium", "low"]
    description: str
    expected_result: str


class TestPlanOutput(BaseModel):
    """Structured output for a complete software test plan."""

    test_plan_summary: str
    total_scenarios: int
    test_scenarios: list[TestScenarioOutput]


class TestPlanner:
    """AI-powered test scenario planner."""

    def __init__(self, model="gpt-5.6", client=None):
        self.client = client if client is not None else OpenAI()
        self.model = model

    def generate_test_plan(self, requirements_analysis: dict) -> dict:
        if not isinstance(requirements_analysis, dict) or not requirements_analysis:
            raise ValueError("Requirements analysis cannot be empty.")

        prompt = f"""
        You are a Senior QA Automation Engineer.

        Based on the following requirements analysis, create a structured
        software test plan.

        Requirements analysis:
        {json.dumps(requirements_analysis, indent=2)}

        Include positive, negative, and edge-case scenarios.
        Assign realistic priorities.
        Do not invent functionality that is not supported by the requirements.
        Ensure total_scenarios matches the number of test scenarios.
        """

        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=TestPlanOutput
        )

        result = response.output_parsed

        if result is None:
            raise ValueError(
                "The AI did not return a structured test plan."
            )

        return result.model_dump()
