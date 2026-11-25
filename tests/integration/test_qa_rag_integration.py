import json
from unittest.mock import MagicMock, patch

from src.core.qa_orchestrator import QAOrchestrator
from src.retrieval.knowledge_base import KnowledgeBase


REQUIREMENT = (
    "Users must be able to log in using their credentials."
)

EXPECTED_ANALYSIS = {
    "summary": "User login functionality",
    "ambiguities": [],
    "missing_information": [],
    "quality_risks": ["Account lockout behavior"],
    "acceptance_criteria": [
        "Valid credentials allow login"
    ],
    "suggested_test_scenarios": [
        "Successful login",
        "Invalid password",
        "Account lockout"
    ]
}

EXPECTED_TEST_PLAN = {
    "test_plan_summary": "Login functionality test plan",
    "total_scenarios": 3,
    "test_scenarios": [
        {
            "id": "TC001",
            "title": "Successful login",
            "type": "positive",
            "priority": "high",
            "description": "Login using valid credentials",
            "expected_result": "User successfully logs in"
        },
        {
            "id": "TC002",
            "title": "Invalid password",
            "type": "negative",
            "priority": "high",
            "description": "Login using an invalid password",
            "expected_result": "Access is denied"
        },
        {
            "id": "TC003",
            "title": "Account lockout",
            "type": "edge_case",
            "priority": "medium",
            "description": (
                "Attempt login repeatedly with invalid passwords"
            ),
            "expected_result": (
                "Account follows the lockout policy"
            )
        }
    ]
}


def create_mock_openai_clients(
    analysis=EXPECTED_ANALYSIS,
    test_plan=EXPECTED_TEST_PLAN
):
    """Create mocked OpenAI clients for integration tests."""

    analyzer_client = MagicMock()
    analyzer_client.responses.create.return_value.output_text = (
        json.dumps(analysis)
    )

    planner_client = MagicMock()
    planner_client.responses.create.return_value.output_text = (
        json.dumps(test_plan)
    )

    return analyzer_client, planner_client


def test_qa_orchestrator_executes_complete_rag_workflow(tmp_path):
    # Arrange: create a real ChromaDB knowledge base.
    embedding_service = MagicMock()
    embedding_service.generate_embedding.return_value = [
        1.0, 0.0, 0.0
    ]

    knowledge_base = KnowledgeBase(
        collection_name="qa_integration_test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_service=embedding_service
    )

    knowledge_base.add_documents(
        ids=["qa-001", "qa-002"],
        documents=[
            (
                "Login testing must cover valid credentials, "
                "invalid passwords, and account lockout."
            ),
            (
                "Shopping cart testing must cover adding and "
                "removing products."
            )
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
    )

    assert knowledge_base.count() == 2

    analyzer_client, planner_client = create_mock_openai_clients()

    # Act: execute the real agents and orchestrator.
    with patch(
        "src.agents.requirements_analyzer.OpenAI",
        return_value=analyzer_client
    ), patch(
        "src.agents.planner_agent.OpenAI",
        return_value=planner_client
    ):
        orchestrator = QAOrchestrator(
            knowledge_base=knowledge_base
        )

        result = orchestrator.process_requirement(REQUIREMENT)

    # Assert: verify the complete workflow output.
    assert result == {
        "requirements_analysis": EXPECTED_ANALYSIS,
        "test_plan": EXPECTED_TEST_PLAN
    }

    embedding_service.generate_embedding.assert_called_once_with(
        REQUIREMENT
    )

    analyzer_client.responses.create.assert_called_once()

    analyzer_prompt = (
        analyzer_client.responses.create.call_args.kwargs["input"]
    )

    assert (
        "Login testing must cover valid credentials"
        in analyzer_prompt
    )
    assert "Shopping cart testing" in analyzer_prompt

    planner_client.responses.create.assert_called_once()

    planner_prompt = (
        planner_client.responses.create.call_args.kwargs["input"]
    )

    assert "User login functionality" in planner_prompt
    assert "Successful login" in planner_prompt


def test_qa_orchestrator_works_without_knowledge_base():
    # Arrange: create mocked OpenAI clients.
    analyzer_client, planner_client = create_mock_openai_clients()

    # Act: execute the orchestrator without a knowledge base.
    with patch(
        "src.agents.requirements_analyzer.OpenAI",
        return_value=analyzer_client
    ), patch(
        "src.agents.planner_agent.OpenAI",
        return_value=planner_client
    ):
        orchestrator = QAOrchestrator()

        result = orchestrator.process_requirement(REQUIREMENT)

    # Assert: the workflow completes without RAG.
    assert result == {
        "requirements_analysis": EXPECTED_ANALYSIS,
        "test_plan": EXPECTED_TEST_PLAN
    }

    analyzer_client.responses.create.assert_called_once()

    analyzer_prompt = (
        analyzer_client.responses.create.call_args.kwargs["input"]
    )

    assert "No previous knowledge available." in analyzer_prompt

    planner_client.responses.create.assert_called_once()


def test_qa_orchestrator_works_when_rag_finds_no_documents():
    # Arrange: create an empty search result.
    knowledge_base = MagicMock()
    knowledge_base.search_by_text.return_value = {
        "documents": [[]]
    }

    analyzer_client, planner_client = create_mock_openai_clients()

    # Act: execute the orchestrator with no retrieved documents.
    with patch(
        "src.agents.requirements_analyzer.OpenAI",
        return_value=analyzer_client
    ), patch(
        "src.agents.planner_agent.OpenAI",
        return_value=planner_client
    ):
        orchestrator = QAOrchestrator(
            knowledge_base=knowledge_base
        )

        result = orchestrator.process_requirement(REQUIREMENT)

    # Assert: the workflow falls back to the normal prompt.
    assert result == {
        "requirements_analysis": EXPECTED_ANALYSIS,
        "test_plan": EXPECTED_TEST_PLAN
    }

    knowledge_base.search_by_text.assert_called_once_with(
        REQUIREMENT,
        n_results=3
    )

    analyzer_prompt = (
        analyzer_client.responses.create.call_args.kwargs["input"]
    )

    assert "No previous knowledge available." in analyzer_prompt

    planner_client.responses.create.assert_called_once()


def test_qa_orchestrator_rejects_invalid_ai_analysis():
    # Arrange: simulate an AI response containing a JSON array
    # instead of the expected JSON object.
    analyzer_client, planner_client = create_mock_openai_clients(
        analysis=[]
    )

    # Act and Assert: invalid analysis must stop the workflow.
    with patch(
        "src.agents.requirements_analyzer.OpenAI",
        return_value=analyzer_client
    ), patch(
        "src.agents.planner_agent.OpenAI",
        return_value=planner_client
    ):
        orchestrator = QAOrchestrator()

        try:
            orchestrator.process_requirement(REQUIREMENT)
            assert False, "Expected ValueError was not raised."
        except ValueError as error:
            assert str(error) == (
                "The AI response must be a JSON object."
            )

    analyzer_client.responses.create.assert_called_once()

    # The planner must not execute after an invalid analysis.
    planner_client.responses.create.assert_not_called()