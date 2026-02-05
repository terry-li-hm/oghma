import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import APIError, OpenAI


@dataclass
class EmbedConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    rate_limit_delay: float = 0.1
    max_retries: int = 3

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> "EmbedConfig":
        values = {
            "provider": cls.provider,
            "model": cls.model,
            "dimensions": cls.dimensions,
            "batch_size": cls.batch_size,
            "rate_limit_delay": cls.rate_limit_delay,
            "max_retries": cls.max_retries,
        }
        if data:
            for key in values:
                if key in data:
                    values[key] = data[key]
        for key, value in overrides.items():
            if key in values and value is not None:
                values[key] = value
        return cls(**values)


class Embedder(ABC):
    def __init__(self, config: EmbedConfig):
        self.config = config

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings."""


class OpenAIEmbedder(Embedder):
    BASE_RETRY_DELAY = 1.0

    def __init__(self, config: EmbedConfig):
        super().__init__(config)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            chunk = texts[start : start + self.config.batch_size]
            vectors.extend(self._embed_chunk(chunk))
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
        return vectors

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                    dimensions=self.config.dimensions,
                )
                return [list(item.embedding) for item in response.data]
            except APIError as exc:  # pragma: no cover - covered by retry test
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    delay = self.BASE_RETRY_DELAY * (2**attempt)
                    time.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("Embedding request failed")


def create_embedder(config: EmbedConfig) -> Embedder:
    if config.provider == "openai":
        return OpenAIEmbedder(config)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
