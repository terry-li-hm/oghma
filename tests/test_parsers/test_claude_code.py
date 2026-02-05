from pathlib import Path

import pytest

from oghma.parsers.claude_code import ClaudeCodeParser


@pytest.fixture
def parser():
    return ClaudeCodeParser()


@pytest.fixture
def fixture_dir(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    return fixtures


def test_can_parse_claude_code_file(fixture_dir):
    parser = ClaudeCodeParser()
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    assert parser.can_parse(file_path) is True


def test_cannot_parse_non_claude_code_file(fixture_dir):
    parser = ClaudeCodeParser()
    file_path = fixture_dir / "other" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    assert parser.can_parse(file_path) is False


def test_parse_user_message(parser, fixture_dir):
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text('{"type": "user", "message": {"content": "Hello world"}}\n')

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello world"


def test_parse_assistant_message_with_text_blocks(parser, fixture_dir):
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"},'
        ' {"type": "text", "text": " world"}]}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "Hello\n world"


def test_parse_multiple_messages(parser, fixture_dir):
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "user", "message": {"content": "User message"}}\n'
        '{"type": "assistant", "message": {"content": "Assistant message"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_skips_empty_lines(parser, fixture_dir):
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "user", "message": {"content": "User message"}}\n\n'
        '{"type": "assistant", "message": {"content": "Assistant message"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 2


def test_handles_corrupt_json_gracefully(parser, fixture_dir):
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "user", "message": {"content": "Valid message"}}\n'
        "invalid json\n"
        '{"type": "assistant", "message": {"content": "Another valid"}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 2


def test_returns_empty_list_for_nonexistent_file(parser):
    messages = parser.parse(Path("/nonexistent/file.jsonl"))

    assert messages == []


def test_truncates_long_content(parser, fixture_dir):
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    long_content = "x" * 4000
    file_path.write_text(f'{{"type": "user", "message": {{"content": "{long_content}"}}}}\n')

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert len(messages[0].content) == 3000
