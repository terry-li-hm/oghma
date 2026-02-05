from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIError

from oghma.embedder import EmbedConfig, OpenAIEmbedder, create_embedder


@patch.dict("os.environ", {}, clear=True)
def test_openai_embedder_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(EmbedConfig())


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("oghma.embedder.OpenAI")
def test_openai_embedder_embed_batch(mock_openai_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    mock_client.embeddings.create.return_value = mock_response

    embedder = OpenAIEmbedder(EmbedConfig(batch_size=2, rate_limit_delay=0.0))
    vectors = embedder.embed_batch(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    mock_client.embeddings.create.assert_called_once()


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("oghma.embedder.time.sleep")
@patch("oghma.embedder.OpenAI")
def test_openai_embedder_retries(
    mock_openai_class: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    success_response = MagicMock()
    success_response.data = [MagicMock(embedding=[0.1, 0.2])]
    retry_error = APIError(
        "boom",
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        body=None,
    )
    mock_client.embeddings.create.side_effect = [retry_error, success_response]

    embedder = OpenAIEmbedder(EmbedConfig(max_retries=2, rate_limit_delay=0.0))
    vectors = embedder.embed_batch(["hello"])

    assert vectors == [[0.1, 0.2]]
    assert mock_client.embeddings.create.call_count == 2
    assert mock_sleep.called


def test_embed_config_from_dict_defaults_and_overrides() -> None:
    config = EmbedConfig.from_dict({"model": "text-embedding-3-large"}, batch_size=20)
    assert config.provider == "openai"
    assert config.model == "text-embedding-3-large"
    assert config.dimensions == 1536
    assert config.batch_size == 20


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("oghma.embedder.OpenAI")
def test_create_embedder_returns_openai(mock_openai_class: MagicMock) -> None:
    mock_openai_class.return_value = MagicMock()
    embedder = create_embedder(EmbedConfig(provider="openai"))
    assert isinstance(embedder, OpenAIEmbedder)


def test_create_embedder_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        create_embedder(EmbedConfig(provider="unknown"))
