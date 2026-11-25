
import json
from openai import OpenAI


class TestPlanner:
    """AI-powered test scenario planner."""

    def __init__(self, model="gpt-5.6"):
        self.client = OpenAI()
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

        Return only a valid JSON object with this structure:

        {{
            "test_plan_summary": "Brief test plan summary",
            "total_scenarios": 0,
            "test_scenarios": [
                {{
                    "id": "TC001",
                    "title": "Test scenario title",
                    "type": "positive, negative, or edge_case",
                    "priority": "high, medium, or low",
                    "description": "What should be tested",
                    "expected_result": "Expected behavior"
                }}
            ]
        }}

        Include positive, negative, and edge-case scenarios.
        Assign realistic priorities.
        Do not invent functionality that is not supported by the requirements.
        Return only valid JSON.
        """

        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        try:
            result = json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "The AI returned an invalid JSON response."
            ) from error

        if not isinstance(result, dict):
            raise ValueError("The AI response must be a JSON object.")

        return result
