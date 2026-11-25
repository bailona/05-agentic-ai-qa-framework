
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.knowledge_base import KnowledgeBase


load_dotenv()


class RequirementsAnalyzer:
    """AI-powered software requirements analyzer with optional RAG."""

    def __init__(
        self,
        model="gpt-5.6",
        knowledge_base: KnowledgeBase | None = None,
        client=None
    ):
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment.")

            client = OpenAI()

        self.client = client
        self.model = model
        self.knowledge_base = knowledge_base

    def analyze(self, requirement: str) -> dict:
        if not requirement or not requirement.strip():
            raise ValueError("Requirement cannot be empty.")

        context = ""

        if self.knowledge_base is not None:
            results = self.knowledge_base.search_by_text(
                requirement,
                n_results=3
            )

            documents = results.get("documents") or [[]]
            relevant_documents = documents[0] if documents else []

            if relevant_documents:
                context = "\n".join(relevant_documents)

        prompt = f"""
        Analyze the following software requirement from a QA perspective.

        Relevant knowledge base context:
        {context if context else "No previous knowledge available."}

        Requirement:
        {requirement}

        Return a valid JSON object with these fields:
        {{
            "summary": "Brief requirement summary",
            "ambiguities": [],
            "missing_information": [],
            "quality_risks": [],
            "acceptance_criteria": [],
            "suggested_test_scenarios": []
        }}

        Identify positive, negative, and edge-case scenarios.
        Use the retrieved context as supporting information, not as
        unquestionable truth. Do not invent requirements.

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
