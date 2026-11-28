
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.retrieval.knowledge_base import KnowledgeBase


load_dotenv()


class RequirementsAnalysisOutput(BaseModel):
    """Structured output for software requirements analysis."""

    summary: str
    ambiguities: list[str]
    missing_information: list[str]
    quality_risks: list[str]
    acceptance_criteria: list[str]
    suggested_test_scenarios: list[str]


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

        Identify positive, negative, and edge-case scenarios.
        Use the retrieved context as supporting information, not as
        unquestionable truth. Do not invent requirements.
        """

        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=RequirementsAnalysisOutput
        )

        result = response.output_parsed

        if result is None:
            raise ValueError(
                "The AI did not return a structured requirements analysis."
            )

        return result.model_dump()
