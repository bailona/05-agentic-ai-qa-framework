
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class EmbeddingService:
    """Generates text embeddings using the OpenAI API."""

    def __init__(self, model="text-embedding-3-small", client=None):
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment.")

            client = OpenAI()

        if not model or not model.strip():
            raise ValueError("Model name cannot be empty.")

        self.client = client
        self.model = model

    def generate_embedding(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        embedding = response.data[0].embedding

        if not embedding:
            raise ValueError("The API returned an empty embedding.")

        return embedding

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not texts:
            raise ValueError("Texts cannot be empty.")

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError("All texts must be non-empty strings.")

        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        embeddings = [
            item.embedding
            for item in response.data
        ]

        if len(embeddings) != len(texts):
            raise ValueError(
                "The API returned an unexpected number of embeddings."
            )

        if any(not embedding for embedding in embeddings):
            raise ValueError("The API returned an empty embedding.")

        return embeddings
