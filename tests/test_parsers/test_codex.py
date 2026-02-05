from pathlib import Path

import pytest

from oghma.parsers.codex import CodexParser


@pytest.fixture
def parser():
    return CodexParser()


@pytest.fixture
def fixture_dir(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    return fixtures


def test_can_parse_codex_file(fixture_dir):
    parser = CodexParser()
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    assert parser.can_parse(file_path) is True


def test_cannot_parse_non_codex_file(fixture_dir):
    parser = CodexParser()
    file_path = fixture_dir / "other" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    assert parser.can_parse(file_path) is False


def test_parse_user_message(parser, fixture_dir):
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "item", "payload": {"item": {"role": "user", "content": "User input"}}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "User input"


def test_parse_assistant_message(parser, fixture_dir):
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "item", "payload": {"item": {"role": "assistant",'
        ' "content": "Assistant output"}}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "Assistant output"


def test_parse_message_with_content_blocks(parser, fixture_dir):
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "item", "payload": {"item": {"role": "user",'
        ' "content": [{"type": "input_text", "text": "Hello"}]}}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].content == "Hello"


def test_parse_output_text_block(parser, fixture_dir):
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "item", "payload": {"item": {"role": "assistant",'
        ' "content": [{"type": "output_text", "text": "Response"}]}}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].content == "Response"


def test_skips_non_item_types(parser, fixture_dir):
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "metadata", "payload": {"key": "value"}}\n'
        '{"type": "item", "payload": {"item": {"role": "user", "content": "Valid"}}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 1
    assert messages[0].role == "user"


def test_handles_corrupt_json_gracefully(parser, fixture_dir):
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        '{"type": "item", "payload": {"item": {"role": "user", "content": "Valid"}}}\n'
        "invalid json\n"
        '{"type": "item", "payload": {"item": {"role": "assistant", "content": "Valid 2"}}}\n'
    )

    messages = parser.parse(file_path)

    assert len(messages) == 2


def test_returns_empty_list_for_nonexistent_file(parser):
    messages = parser.parse(Path("/nonexistent/file.jsonl"))

    assert messages == []
