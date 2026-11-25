
import pytest
from unittest.mock import MagicMock

from src.retrieval.knowledge_base import KnowledgeBase


@pytest.fixture
def knowledge_base(tmp_path):
    return KnowledgeBase(
        collection_name="test_knowledge",
        persist_directory=str(tmp_path)
    )


def test_knowledge_base_creates_collection(knowledge_base):
    assert knowledge_base.collection is not None
    assert knowledge_base.count() == 0


def test_add_documents_stores_documents(knowledge_base):
    knowledge_base.add_documents(
        ids=["doc1", "doc2"],
        documents=["Software testing", "Test automation"],
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    )

    assert knowledge_base.count() == 2


def test_search_returns_relevant_documents(knowledge_base):
    knowledge_base.add_documents(
        ids=["doc1", "doc2"],
        documents=["Software testing", "Test automation"],
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    )

    result = knowledge_base.search(
        query_embedding=[0.1, 0.2, 0.3],
        n_results=1
    )

    assert len(result["ids"][0]) == 1
    assert result["ids"][0][0] == "doc1"


def test_add_documents_rejects_empty_inputs(knowledge_base):
    with pytest.raises(ValueError, match="IDs and documents are required"):
        knowledge_base.add_documents(
            ids=[],
            documents=[],
            embeddings=[]
        )


def test_add_documents_rejects_different_lengths(knowledge_base):
    with pytest.raises(
        ValueError,
        match="IDs and documents must have the same length"
    ):
        knowledge_base.add_documents(
            ids=["doc1", "doc2"],
            documents=["Software testing"],
            embeddings=[[0.1, 0.2, 0.3]]
        )


def test_add_documents_rejects_empty_document(knowledge_base):
    with pytest.raises(ValueError, match="Documents cannot be empty"):
        knowledge_base.add_documents(
            ids=["doc1"],
            documents=[""],
            embeddings=[[0.1, 0.2, 0.3]]
        )


def test_add_documents_rejects_invalid_metadata(knowledge_base):
    with pytest.raises(
        ValueError,
        match="Metadata and document lists must have the same length"
    ):
        knowledge_base.add_documents(
            ids=["doc1"],
            documents=["Software testing"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{}, {}]
        )


def test_search_rejects_empty_embedding(knowledge_base):
    with pytest.raises(ValueError, match="Query embedding cannot be empty"):
        knowledge_base.search(query_embedding=[])


def test_search_rejects_invalid_result_count(knowledge_base):
    with pytest.raises(
        ValueError,
        match="Number of results must be greater than zero"
    ):
        knowledge_base.search(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=0
        )


def test_add_documents_rejects_empty_id(knowledge_base):
    with pytest.raises(ValueError, match="Document IDs cannot be empty"):
        knowledge_base.add_documents(
            ids=[""],
            documents=["Software testing"],
            embeddings=[[0.1, 0.2, 0.3]]
        )


def test_add_documents_generates_embeddings_automatically(tmp_path):
    embedding_service = MagicMock()
    embedding_service.generate_embeddings.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    knowledge_base = KnowledgeBase(
        collection_name="automatic_embeddings",
        persist_directory=str(tmp_path),
        embedding_service=embedding_service
    )

    documents = [
        "Software testing",
        "Test automation"
    ]

    knowledge_base.add_documents(
        ids=["doc1", "doc2"],
        documents=documents
    )

    embedding_service.generate_embeddings.assert_called_once_with(
        documents
    )

    assert knowledge_base.count() == 2


def test_search_by_text_generates_query_embedding(tmp_path):
    embedding_service = MagicMock()
    embedding_service.generate_embedding.return_value = [
        0.1, 0.2, 0.3
    ]

    knowledge_base = KnowledgeBase(
        collection_name="text_search",
        persist_directory=str(tmp_path),
        embedding_service=embedding_service
    )

    knowledge_base.add_documents(
        ids=["doc1"],
        documents=["Software testing"],
        embeddings=[[0.1, 0.2, 0.3]]
    )

    result = knowledge_base.search_by_text(
        "How to test software?",
        n_results=1
    )

    embedding_service.generate_embedding.assert_called_once_with(
        "How to test software?"
    )

    assert result["ids"][0] == ["doc1"]


def test_add_documents_requires_embedding_service_when_embeddings_missing(
    tmp_path
):
    knowledge_base = KnowledgeBase(
        collection_name="missing_embedding_service",
        persist_directory=str(tmp_path)
    )

    with pytest.raises(
        ValueError,
        match="Embedding service is required"
    ):
        knowledge_base.add_documents(
            ids=["doc1"],
            documents=["Software testing"]
        )


def test_search_by_text_requires_embedding_service(tmp_path):
    knowledge_base = KnowledgeBase(
        collection_name="text_search_without_service",
        persist_directory=str(tmp_path)
    )

    with pytest.raises(
        ValueError,
        match="Embedding service is required"
    ):
        knowledge_base.search_by_text("Software testing")
