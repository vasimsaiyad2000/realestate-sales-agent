from collections.abc import Sequence
from typing import Protocol

import httpx
from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self.client = client or AsyncOpenAI(
            api_key=settings.open_router_api_key,
            base_url=settings.open_router_base_url
        )
    
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(
            model=settings.openai_embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


class OllamaEmbeddingProvider:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_embedding_model
        self.client = client or httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": list(texts)},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Run Ollama and pull model '{self.model}'."
            ) from exc

        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an unexpected embedding response")
        return embeddings


def create_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider.lower() == "ollama":
        return OllamaEmbeddingProvider()
    if settings.embedding_provider.lower() == "openai":
        return OpenAIEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
