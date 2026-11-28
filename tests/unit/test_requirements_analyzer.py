
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agents.requirements_analyzer import (
    RequirementsAnalyzer,
    RequirementsAnalysisOutput
)


@pytest.fixture
def mock_client():
    client = MagicMock()

    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=RequirementsAnalysisOutput(
            summary="User login requirement",
            ambiguities=[],
            missing_information=[],
            quality_risks=[],
            acceptance_criteria=[],
            suggested_test_scenarios=[]
        )
    )

    return client


@pytest.fixture
def analyzer(mock_client):
    return RequirementsAnalyzer(client=mock_client)


def test_analyze_rejects_empty_requirement(analyzer):
    with pytest.raises(
        ValueError,
        match="Requirement cannot be empty"
    ):
        analyzer.analyze("")


def test_analyzer_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY not found"
    ):
        RequirementsAnalyzer()


def test_analyze_rejects_whitespace_requirement(analyzer):
    with pytest.raises(
        ValueError,
        match="Requirement cannot be empty"
    ):
        analyzer.analyze("   ")


def test_analyze_returns_structured_result(analyzer, mock_client):
    result = analyzer.analyze(
        "Users must be able to log in."
    )

    assert isinstance(result, dict)
    assert result["summary"] == "User login requirement"
    assert "ambiguities" in result
    assert "missing_information" in result
    assert "quality_risks" in result
    assert "acceptance_criteria" in result
    assert "suggested_test_scenarios" in result

    mock_client.responses.parse.assert_called_once()

    call_arguments = mock_client.responses.parse.call_args.kwargs

    assert call_arguments["model"] == "gpt-5.6"
    assert "Users must be able to log in." in call_arguments["input"]
    assert call_arguments["text_format"] is RequirementsAnalysisOutput


def test_analyze_rejects_missing_structured_output(mock_client):
    mock_client.responses.parse.return_value = SimpleNamespace(
        output_parsed=None
    )

    analyzer = RequirementsAnalyzer(client=mock_client)

    with pytest.raises(
        ValueError,
        match="did not return a structured requirements analysis"
    ):
        analyzer.analyze("Users must be able to log in.")


def test_analyze_uses_retrieved_knowledge_base_context(mock_client):
    knowledge_base = MagicMock()

    knowledge_base.search_by_text.return_value = {
        "documents": [[
            "Previous requirement: login must validate email format.",
            "Previous test: invalid passwords must be rejected."
        ]]
    }

    analyzer = RequirementsAnalyzer(
        client=mock_client,
        knowledge_base=knowledge_base
    )

    analyzer.analyze("Users must be able to log in.")

    knowledge_base.search_by_text.assert_called_once_with(
        "Users must be able to log in.",
        n_results=3
    )

    prompt = mock_client.responses.parse.call_args.kwargs["input"]

    assert "login must validate email format" in prompt
    assert "invalid passwords must be rejected" in prompt


def test_analyze_works_without_knowledge_base(mock_client):
    analyzer = RequirementsAnalyzer(client=mock_client)

    result = analyzer.analyze(
        "Users must be able to log in."
    )

    assert isinstance(result, dict)

    prompt = mock_client.responses.parse.call_args.kwargs["input"]

    assert "No previous knowledge available." in prompt


def test_analyze_handles_empty_knowledge_base_results(mock_client):
    knowledge_base = MagicMock()

    knowledge_base.search_by_text.return_value = {
        "documents": [[]]
    }

    analyzer = RequirementsAnalyzer(
        client=mock_client,
        knowledge_base=knowledge_base
    )

    result = analyzer.analyze(
        "Users must be able to log in."
    )

    assert isinstance(result, dict)

    prompt = mock_client.responses.parse.call_args.kwargs["input"]

    assert "No previous knowledge available." in prompt
