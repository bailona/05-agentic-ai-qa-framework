
import chromadb

from src.retrieval.embedding_service import EmbeddingService


class KnowledgeBase:
    """Vector database for storing and retrieving QA knowledge."""

    def __init__(
        self,
        collection_name="qa_knowledge",
        persist_directory="data/chroma",
        embedding_service=None
    ):
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )

        self.embedding_service = embedding_service

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None
    ) -> None:
        if not ids or not documents:
            raise ValueError("IDs and documents are required.")

        if len(ids) != len(documents):
            raise ValueError(
                "IDs and documents must have the same length."
            )

        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise ValueError("Document IDs cannot be empty.")

        if any(
            not isinstance(item, str) or not item.strip()
            for item in documents
        ):
            raise ValueError("Documents cannot be empty.")

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(
                "Metadata and document lists must have the same length."
            )

        if embeddings is None:
            if self.embedding_service is None:
                raise ValueError(
                    "Embedding service is required when embeddings are not provided."
                )

            embeddings = self.embedding_service.generate_embeddings(
                documents
            )

        if len(embeddings) != len(ids):
            raise ValueError(
                "IDs, documents, and embeddings must have the same length."
            )

        if any(not embedding for embedding in embeddings):
            raise ValueError("Embeddings cannot be empty.")

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5
    ) -> dict:
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if n_results < 1:
            raise ValueError("Number of results must be greater than zero.")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

    def search_by_text(
        self,
        query: str,
        n_results: int = 5
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if self.embedding_service is None:
            raise ValueError(
                "Embedding service is required for text-based search."
            )

        query_embedding = self.embedding_service.generate_embedding(
            query
        )

        return self.search(
            query_embedding=query_embedding,
            n_results=n_results
        )

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
