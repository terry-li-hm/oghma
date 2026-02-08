import json
from unittest.mock import MagicMock, patch

import pytest

from oghma.config import Config
from oghma.extractor import Extractor
from oghma.parsers import Message


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config: Config = {
        "storage": {
            "db_path": ":memory:",
            "backup_enabled": False,
            "backup_dir": "",
            "backup_retention_days": 30,
        },
        "daemon": {
            "poll_interval": 300,
            "log_level": "INFO",
            "log_file": "",
            "pid_file": "",
            "min_messages": 6,
        },
        "extraction": {
            "model": "gpt-4o-mini",
            "max_content_chars": 4000,
            "categories": ["learning", "preference", "project_context", "gotcha", "workflow"],
            "confidence_threshold": 0.5,
        },
        "export": {"output_dir": "", "format": "markdown"},
        "tools": {
            "claude_code": {"enabled": True, "paths": []},
            "codex": {"enabled": True, "paths": []},
            "openclaw": {"enabled": True, "paths": []},
            "opencode": {"enabled": True, "paths": []},
            "cursor": {"enabled": False, "paths": []},
        },
    }
    return config


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    return [
        {
            "content": "User prefers using spaces over tabs for indentation",
            "category": "preference",
            "confidence": 0.9,
        },
        {
            "content": "Use pytest for testing Python projects",
            "category": "workflow",
            "confidence": 0.85,
        },
    ]


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role="user", content="How should I format my Python code?"),
        Message(
            role="assistant",
            content="You should use 4 spaces for indentation, not tabs. This is PEP 8 compliant.",
        ),
        Message(role="user", content="What about testing?"),
        Message(
            role="assistant", content="Use pytest for testing Python projects. It's the standard."
        ),
        Message(role="user", content="Any gotchas to watch out for?"),
        Message(
            role="assistant", content="Watch out for mutable default arguments in Python functions."
        ),
    ]


class TestExtractor:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_extractor_initialization(self, mock_config):
        """Test that extractor initializes correctly."""
        extractor = Extractor(config=mock_config)
        assert extractor.config is not None
        assert extractor.model == "gpt-4o-mini"
        assert extractor.max_chars == 4000
        assert extractor.categories == [
            "learning",
            "preference",
            "project_context",
            "gotcha",
            "workflow",
        ]
        assert extractor.confidence_threshold == 0.5

    @patch.dict("os.environ", {}, clear=True)
    def test_extractor_no_api_key(self, mock_config):
        """Test that extractor raises error without API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
            Extractor(config=mock_config)

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_extractor_openrouter_initialization(self, mock_config):
        """Test that extractor initializes with OpenRouter for supported models."""
        mock_config["extraction"]["model"] = "google/gemini-3-flash-preview"
        extractor = Extractor(config=mock_config)
        assert extractor.use_openrouter is True
        assert extractor.model == "google/gemini-3-flash-preview"

    @patch.dict("os.environ", {}, clear=True)
    def test_extractor_openrouter_no_api_key(self, mock_config):
        """Test that extractor raises error for OpenRouter model without API key."""
        mock_config["extraction"]["model"] = "google/gemini-3-flash-preview"
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable not set"):
            Extractor(config=mock_config)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    def test_extract_success(
        self, mock_openai_class, mock_config, sample_messages, mock_openai_response
    ):
        """Test successful extraction."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_openai_response)
        mock_client.chat.completions.create.return_value = mock_response

        extractor = Extractor(config=mock_config)
        memories = extractor.extract(sample_messages, "claude_code")

        assert len(memories) == 2
        assert memories[0].content == "User prefers using spaces over tabs for indentation"
        assert memories[0].category == "preference"
        assert memories[0].confidence == 0.9

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    def test_extract_empty_messages(self, mock_openai_class, mock_config):
        """Test extraction with empty messages."""
        extractor = Extractor(config=mock_config)
        memories = extractor.extract([], "claude_code")

        assert len(memories) == 0

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    def test_extract_no_valid_memories(self, mock_openai_class, mock_config, sample_messages):
        """Test extraction when no valid memories are returned."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"
        mock_client.chat.completions.create.return_value = mock_response

        extractor = Extractor(config=mock_config)
        memories = extractor.extract(sample_messages, "claude_code")

        assert len(memories) == 0

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    def test_extract_filters_low_confidence(self, mock_openai_class, mock_config, sample_messages):
        """Test that low confidence memories are filtered out."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response_data = [
            {"content": "sqlite-vec requires enable_load_extension before load", "category": "learning", "confidence": 0.8},
            {"content": "Low confidence memory about something vague", "category": "preference", "confidence": 0.3},
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        mock_client.chat.completions.create.return_value = mock_response

        extractor = Extractor(config=mock_config)
        memories = extractor.extract(sample_messages, "claude_code")

        assert len(memories) == 1
        assert "sqlite-vec" in memories[0].content

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    def test_extract_uses_custom_confidence_threshold(
        self, mock_openai_class, mock_config, sample_messages
    ):
        """Test that configured confidence threshold is respected."""
        mock_config["extraction"]["confidence_threshold"] = 0.9
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response_data = [
            {"content": "OpenAI embedding API requires dimensions param for ada-003", "category": "learning", "confidence": 0.95},
            {"content": "FTS5 query escaping prevents syntax errors in search", "category": "preference", "confidence": 0.85},
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        mock_client.chat.completions.create.return_value = mock_response

        extractor = Extractor(config=mock_config)
        memories = extractor.extract(sample_messages, "claude_code")

        assert len(memories) == 1
        assert "OpenAI" in memories[0].content

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    def test_extract_filters_invalid_categories(
        self, mock_openai_class, mock_config, sample_messages
    ):
        """Test that invalid categories are filtered out."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response_data = [
            {"content": "ruff check --fix auto-corrects import ordering issues", "category": "learning", "confidence": 0.8},
            {"content": "Invalid category memory about something", "category": "random", "confidence": 0.9},
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)
        mock_client.chat.completions.create.return_value = mock_response

        extractor = Extractor(config=mock_config)
        memories = extractor.extract(sample_messages, "claude_code")

        assert len(memories) == 1
        assert "ruff" in memories[0].content

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    @patch("time.sleep")
    def test_extract_retry_on_failure(
        self, mock_sleep, mock_openai_class, mock_config, sample_messages
    ):
        """Test that extractor retries on API failure."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response_data = [{"content": "pytest --tb=short gives concise tracebacks for faster debugging", "category": "learning", "confidence": 0.8}]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(mock_response_data)

        mock_client.chat.completions.create.side_effect = [Exception("API error"), mock_response]

        extractor = Extractor(config=mock_config)
        memories = extractor.extract(sample_messages, "claude_code")

        assert len(memories) == 1
        assert mock_client.chat.completions.create.call_count == 2

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("oghma.extractor.OpenAI")
    @patch("time.sleep")
    def test_extract_max_retries_exceeded(
        self, mock_sleep, mock_openai_class, mock_config, sample_messages
    ):
        """Test that extractor gives up after max retries."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.side_effect = [Exception("API error")] * 3

        extractor = Extractor(config=mock_config)
        memories = extractor.extract(sample_messages, "claude_code")

        assert len(memories) == 0
        assert mock_client.chat.completions.create.call_count == 3

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_parse_response_valid_json(self, mock_config):
        """Test parsing valid JSON response."""
        extractor = Extractor(config=mock_config)

        response_text = json.dumps(
            [{"content": "Test memory", "category": "learning", "confidence": 0.9}]
        )

        memories = extractor._parse_response(response_text)
        assert len(memories) == 1
        assert memories[0].content == "Test memory"
        assert memories[0].category == "learning"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_parse_response_with_markdown(self, mock_config):
        """Test parsing response with markdown code block."""
        extractor = Extractor(config=mock_config)

        response_text = """```json
[{"content": "Test memory", "category": "learning", "confidence": 0.9}]
```"""

        memories = extractor._parse_response(response_text)
        assert len(memories) == 1
        assert memories[0].content == "Test memory"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_parse_response_invalid_json(self, mock_config):
        """Test parsing invalid JSON response."""
        extractor = Extractor(config=mock_config)

        response_text = "not valid json"

        memories = extractor._parse_response(response_text)
        assert len(memories) == 0

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_parse_response_normalizes_confidence(self, mock_config):
        """Test that confidence values are normalized between 0 and 1."""
        extractor = Extractor(config=mock_config)

        response_text = json.dumps(
            [
                {"content": "High", "category": "learning", "confidence": 1.5},
                {"content": "Low", "category": "preference", "confidence": -0.5},
                {"content": "Normal", "category": "workflow", "confidence": 0.7},
            ]
        )

        memories = extractor._parse_response(response_text)
        assert len(memories) == 3
        assert memories[0].confidence == 1.0
        assert memories[1].confidence == 0.0
        assert memories[2].confidence == 0.7

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_build_prompt(self, mock_config):
        """Test prompt building."""
        extractor = Extractor(config=mock_config)

        messages = [
            Message(role="user", content="How do I test?"),
            Message(role="assistant", content="Use pytest"),
        ]

        prompt = extractor._build_prompt(messages)

        assert "memory extraction system" in prompt.lower()
        assert "User: How do I test?" in prompt
        assert "Assistant: Use pytest" in prompt
        assert "learning" in prompt
        assert "preference" in prompt
        assert "JSON" in prompt

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_build_prompt_uses_config_categories(self, mock_config):
        """Test prompt uses configured categories."""
        mock_config["extraction"]["categories"] = ["learning", "custom"]
        extractor = Extractor(config=mock_config)

        messages = [
            Message(role="user", content="How do I test?"),
            Message(role="assistant", content="Use pytest"),
        ]

        prompt = extractor._build_prompt(messages)

        assert "- learning" in prompt
        assert "- custom" in prompt
        assert "preference" not in prompt

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_is_noise_filters_short_content(self, mock_config):
        extractor = Extractor(config=mock_config)
        from oghma.extractor import Memory

        assert extractor._is_noise(Memory(content="Too short", category="learning"))
        assert not extractor._is_noise(
            Memory(content="sqlite-vec requires enable_load_extension before load", category="gotcha")
        )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_is_noise_filters_meta_references(self, mock_config):
        extractor = Extractor(config=mock_config)
        from oghma.extractor import Memory

        assert extractor._is_noise(
            Memory(content="CLAUDE.md is version controlled in ~/agent-config", category="workflow")
        )
        assert extractor._is_noise(
            Memory(content="The user's MEMORY.md contains sqlite-vec gotchas", category="gotcha")
        )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_is_noise_filters_shallow_user_observations(self, mock_config):
        extractor = Extractor(config=mock_config)
        from oghma.extractor import Memory

        # Short "The user is..." should be filtered
        assert extractor._is_noise(
            Memory(content="The user is located in Hong Kong", category="preference")
        )
        # Long "The user is..." with genuine detail should survive
        assert not extractor._is_noise(
            Memory(
                content="The user is experiencing timeouts when OpenCode processes small edits "
                "under 25 lines — the workaround is to use direct file edits instead",
                category="gotcha",
            )
        )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_build_prompt_strips_system_reminders(self, mock_config):
        extractor = Extractor(config=mock_config)
        messages = [
            Message(
                role="user",
                content="How do I test? <system-reminder>You must do X</system-reminder>",
            ),
            Message(role="assistant", content="Use pytest"),
        ]

        prompt = extractor._build_prompt(messages)
        assert "system-reminder" not in prompt
        assert "How do I test?" in prompt
