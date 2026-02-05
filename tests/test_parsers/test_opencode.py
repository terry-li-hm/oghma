import json
from pathlib import Path

import pytest

from oghma.parsers.opencode import OpenCodeParser


@pytest.fixture
def parser():
    return OpenCodeParser()


@pytest.fixture
def fixture_dir(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    return fixtures


def test_can_parse_opencode_directory(fixture_dir):
    parser = OpenCodeParser()
    dir_path = fixture_dir / ".local" / "share" / "opencode" / "storage" / "message" / "ses_test"
    dir_path.mkdir(parents=True)

    assert parser.can_parse(dir_path) is True


def test_cannot_parse_non_opencode_directory(fixture_dir):
    parser = OpenCodeParser()
    dir_path = fixture_dir / "other" / "session"
    dir_path.mkdir(parents=True)

    assert parser.can_parse(dir_path) is False


def test_cannot_parse_file(fixture_dir):
    parser = OpenCodeParser()
    file_path = fixture_dir / "test.json"
    file_path.write_text("{}")

    assert parser.can_parse(file_path) is False


def test_parse_simple_messages(parser, fixture_dir):
    session_dir = fixture_dir / ".local" / "share" / "opencode" / "storage" / "message" / "ses_test"
    session_dir.mkdir(parents=True)

    msg1 = {"id": "msg1", "role": "user", "content": "User message"}
    msg2 = {"id": "msg2", "role": "assistant", "content": "Assistant message"}

    (session_dir / "msg_0001.json").write_text(json.dumps(msg1))
    (session_dir / "msg_0002.json").write_text(json.dumps(msg2))

    messages = parser.parse(session_dir)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "User message"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Assistant message"


def test_parse_messages_with_parts(parser, fixture_dir):
    session_dir = fixture_dir / ".local" / "share" / "opencode" / "storage" / "message" / "ses_test"
    session_dir.mkdir(parents=True)

    msg1 = {"id": "msg1", "role": "user"}
    msg2 = {"id": "msg2", "role": "assistant"}

    part1 = {"message_id": "msg1", "text": "First part"}
    part2 = {"message_id": "msg2", "text": "Response part 1"}
    part3 = {"message_id": "msg2", "text": "Response part 2"}

    (session_dir / "msg_0001.json").write_text(json.dumps(msg1))
    (session_dir / "msg_0002.json").write_text(json.dumps(msg2))

    part_dir = session_dir / "part" / "msg_msg1"
    part_dir.mkdir(parents=True)
    (part_dir / "prt_0001.json").write_text(json.dumps(part1))

    part_dir2 = session_dir / "part" / "msg_msg2"
    part_dir2.mkdir(parents=True)
    (part_dir2 / "prt_0001.json").write_text(json.dumps(part2))
    (part_dir2 / "prt_0002.json").write_text(json.dumps(part3))

    messages = parser.parse(session_dir)

    assert len(messages) == 2
    assert messages[0].content == "First part"
    assert messages[1].content == "Response part 1\nResponse part 2"


def test_skips_messages_without_role(parser, fixture_dir):
    session_dir = fixture_dir / ".local" / "share" / "opencode" / "storage" / "message" / "ses_test"
    session_dir.mkdir(parents=True)

    msg1 = {"id": "msg1", "role": "user", "content": "Valid"}
    msg2 = {"id": "msg2", "role": "unknown", "content": "Invalid role"}

    (session_dir / "msg_0001.json").write_text(json.dumps(msg1))
    (session_dir / "msg_0002.json").write_text(json.dumps(msg2))

    messages = parser.parse(session_dir)

    assert len(messages) == 1
    assert messages[0].role == "user"


def test_handles_corrupt_json_gracefully(parser, fixture_dir):
    session_dir = fixture_dir / ".local" / "share" / "opencode" / "storage" / "message" / "ses_test"
    session_dir.mkdir(parents=True)

    msg1 = {"id": "msg1", "role": "user", "content": "Valid"}
    (session_dir / "msg_0001.json").write_text(json.dumps(msg1))
    (session_dir / "msg_0002.json").write_text("invalid json")

    messages = parser.parse(session_dir)

    assert len(messages) == 1


def test_returns_empty_list_for_nonexistent_directory(parser):
    messages = parser.parse(Path("/nonexistent/directory"))

    assert messages == []


def test_truncates_long_content(parser, fixture_dir):
    session_dir = fixture_dir / ".local" / "share" / "opencode" / "storage" / "message" / "ses_test"
    session_dir.mkdir(parents=True)

    long_content = "x" * 4000
    msg1 = {"id": "msg1", "role": "user", "content": long_content}

    (session_dir / "msg_0001.json").write_text(json.dumps(msg1))

    messages = parser.parse(session_dir)

    assert len(messages) == 1
    assert len(messages[0].content) == 3000
