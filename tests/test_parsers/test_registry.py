import pytest

from oghma.parsers import get_parser_for_file


@pytest.fixture
def fixture_dir(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    return fixtures


def test_get_parser_for_claude_code_file(fixture_dir):
    file_path = fixture_dir / ".claude" / "projects" / "-Users-terry" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    parser = get_parser_for_file(file_path)

    assert parser is not None
    assert parser.__class__.__name__ == "ClaudeCodeParser"


def test_get_parser_for_codex_file(fixture_dir):
    file_path = fixture_dir / ".codex" / "sessions" / "2026" / "02" / "05" / "rollout-test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    parser = get_parser_for_file(file_path)

    assert parser is not None
    assert parser.__class__.__name__ == "CodexParser"


def test_get_parser_for_openclaw_file(fixture_dir):
    file_path = fixture_dir / ".openclaw" / "agents" / "main" / "sessions" / "test.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    parser = get_parser_for_file(file_path)

    assert parser is not None
    assert parser.__class__.__name__ == "OpenClawParser"


def test_get_parser_for_opencode_directory(fixture_dir):
    dir_path = fixture_dir / ".local" / "share" / "opencode" / "storage" / "message" / "ses_test"
    dir_path.mkdir(parents=True)

    parser = get_parser_for_file(dir_path)

    assert parser is not None
    assert parser.__class__.__name__ == "OpenCodeParser"


def test_get_parser_returns_none_for_unknown_file(fixture_dir):
    file_path = fixture_dir / "other" / "unknown.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch()

    parser = get_parser_for_file(file_path)

    assert parser is None
