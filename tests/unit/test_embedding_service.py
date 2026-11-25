
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval.embedding_service import EmbeddingService


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def embedding_service(mock_client):
    return EmbeddingService(client=mock_client)


def test_generate_embedding_returns_vector(
    embedding_service,
    mock_client
):
    expected_embedding = [0.1, 0.2, 0.3]

    mock_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=expected_embedding)
        ]
    )

    result = embedding_service.generate_embedding(
        "Software quality assurance"
    )

    assert result == expected_embedding

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="Software quality assurance"
    )


def test_generate_embeddings_returns_multiple_vectors(
    embedding_service,
    mock_client
):
    expected_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    mock_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=embedding)
            for embedding in expected_embeddings
        ]
    )

    result = embedding_service.generate_embeddings(
        ["Test automation", "Software testing"]
    )

    assert result == expected_embeddings

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["Test automation", "Software testing"]
    )


def test_generate_embedding_rejects_empty_text(embedding_service):
    with pytest.raises(ValueError, match="Text cannot be empty"):
        embedding_service.generate_embedding("")


def test_generate_embedding_rejects_whitespace(embedding_service):
    with pytest.raises(ValueError, match="Text cannot be empty"):
        embedding_service.generate_embedding("   ")


def test_generate_embeddings_rejects_empty_list(embedding_service):
    with pytest.raises(ValueError, match="Texts cannot be empty"):
        embedding_service.generate_embeddings([])


def test_generate_embeddings_rejects_invalid_text(embedding_service):
    with pytest.raises(
        ValueError,
        match="All texts must be non-empty strings"
    ):
        embedding_service.generate_embeddings(
            ["Valid text", ""]
        )


def test_generate_embeddings_rejects_non_string(embedding_service):
    with pytest.raises(
        ValueError,
        match="All texts must be non-empty strings"
    ):
        embedding_service.generate_embeddings(
            ["Valid text", 123]
        )


def test_generate_embedding_rejects_empty_api_response(
    embedding_service,
    mock_client
):
    mock_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[])
        ]
    )

    with pytest.raises(
        ValueError,
        match="API returned an empty embedding"
    ):
        embedding_service.generate_embedding("Test automation")


def test_generate_embeddings_rejects_unexpected_response_length(
    embedding_service,
    mock_client
):
    mock_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2, 0.3])
        ]
    )

    with pytest.raises(
        ValueError,
        match="unexpected number of embeddings"
    ):
        embedding_service.generate_embeddings(
            ["Text one", "Text two"]
        )


def test_embedding_service_accepts_custom_model(mock_client):
    service = EmbeddingService(
        model="custom-embedding-model",
        client=mock_client
    )

    assert service.model == "custom-embedding-model"


def test_embedding_service_rejects_empty_model(mock_client):
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        EmbeddingService(model="", client=mock_client)
