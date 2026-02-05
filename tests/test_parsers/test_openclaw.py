from pathlib import Path

import pytest

from oghma.parsers.openclaw import OpenClawParser


@pytest.fixture
def parser():
    return OpenClawParser()


@pytest.fixture
def fixture_dir(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    return fixtures


def test_can_parse_openclaw_file(fixture_dir):
    parser = OpenClawParser()
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    assert parser.can_parse(file_path) is True


def test_cannot_parse_non_openclaw_file(fixture_dir):
    parser = OpenClawParser()
    file_path = fixture_dir / "other" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    assert parser.can_parse(file_path) is False


def test_parse_user_message(parser, fixture_dir):
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "message", "message": {"role": "user", "content": "User input"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "User input"


def test_parse_assistant_message(parser, fixture_dir):
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "message", "message": {"role": "assistant", "content": "Assistant response"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "Assistant response"


def test_parse_multiple_messages(parser, fixture_dir):
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "message", "message": {"role": "user", "content": "First"}}\n'
        '{"type": "message", "message": {"role": "assistant", "content": "Second"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_skips_non_message_types(parser, fixture_dir):
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "system", "message": {"role": "system", "content": "System info"}}\n'
        '{"type": "message", "message": {"role": "user", "content": "Valid"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1


def test_handles_corrupt_json_gracefully(parser, fixture_dir):
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "message", "message": {"role": "user", "content": "Valid 1"}}\n'
        "invalid json\n"
        '{"type": "message", "message": {"role": "assistant", "content": "Valid 2"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 2


def test_returns_empty_list_for_nonexistent_file(parser):
    messages = parser.parse(Path("/nonexistent/file.jsonl"))

    assert messages == []


def test_handles_content_as_list(parser, fixture_dir):
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "message", "message": {"role": "user",'
        ' "content": [{"text": "Hello"}, {"text": " world"}]}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].content == "Hello\n world"
