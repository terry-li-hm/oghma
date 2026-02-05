import time

import pytest

from oghma.config import Config
from oghma.storage import Storage
from oghma.watcher import Watcher


@pytest.fixture
def temp_config():
    """Create a temporary config for testing."""
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
            "claude_code": {"enabled": True, "paths": ["~/.claude/projects/-Users-*/*.jsonl"]},
            "codex": {"enabled": True, "paths": ["~/.codex/sessions/**/rollout-*.jsonl"]},
            "openclaw": {"enabled": True, "paths": ["~/.openclaw/agents/*/sessions/*.jsonl"]},
            "opencode": {
                "enabled": True,
                "paths": ["~/.local/share/opencode/storage/message/ses_*"],
            },
            "cursor": {"enabled": False, "paths": []},
        },
    }
    return config


@pytest.fixture
def temp_storage(temp_config, tmp_path):
    """Create a temporary storage for testing."""
    db_path = tmp_path / "test.db"
    return Storage(db_path=str(db_path), config=temp_config)


@pytest.fixture
def watcher(temp_config, temp_storage):
    """Create a watcher instance for testing."""
    return Watcher(config=temp_config, storage=temp_storage)


class TestWatcher:
    def test_watcher_initialization(self, watcher):
        """Test that watcher initializes correctly."""
        assert watcher.config is not None
        assert watcher.storage is not None

    def test_discover_files_with_temp_directory(self, watcher, tmp_path):
        """Test file discovery with a temporary directory."""
        config = watcher.config
        test_dir = tmp_path / "test_tool"
        test_dir.mkdir()

        test_file = test_dir / "test.jsonl"
        test_file.write_text('{"type": "user", "message": {"content": "test"}}')

        config["tools"] = {"test_tool": {"enabled": True, "paths": [str(test_dir / "*.jsonl")]}}

        files = watcher.discover_files()
        assert len(files) >= 1
        assert test_file in files

    def test_get_changed_files_no_state(self, watcher, tmp_path):
        """Test changed files detection when no state exists."""
        test_dir = tmp_path / "test_tool"
        test_dir.mkdir()

        test_file = test_dir / "test.jsonl"
        test_file.write_text('{"type": "user", "message": {"content": "test"}}')

        watcher.config["tools"] = {
            "test_tool": {"enabled": True, "paths": [str(test_dir / "*.jsonl")]}
        }

        changed = watcher.get_changed_files()
        assert len(changed) >= 1

    def test_get_changed_files_with_existing_state(self, watcher, tmp_path):
        """Test changed files detection with existing state."""
        test_dir = tmp_path / "test_tool"
        test_dir.mkdir()

        test_file = test_dir / "test.jsonl"
        test_file.write_text('{"type": "user", "message": {"content": "test"}}')

        watcher.config["tools"] = {
            "test_tool": {"enabled": True, "paths": [str(test_dir / "*.jsonl")]}
        }

        actual_mtime = test_file.stat().st_mtime
        actual_size = test_file.stat().st_size
        watcher.storage.update_extraction_state(str(test_file), actual_mtime, actual_size, 5)

        changed = watcher.get_changed_files()
        assert test_file not in changed

        time.sleep(0.1)
        test_file.write_text('{"type": "user", "message": {"content": "updated"}}')

        changed = watcher.get_changed_files()
        assert test_file in changed

    def test_parse_file(self, watcher, tmp_path):
        """Test parsing a transcript file."""
        test_dir = tmp_path / ".claude" / "projects" / "-Users-test"
        test_dir.mkdir(parents=True)

        test_file = test_dir / "test.jsonl"
        test_file.write_text('{"type": "user", "message": {"content": "test message"}}\n')

        messages = watcher.parse_file(test_file)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "test message" in messages[0].content

    def test_should_process_enough_messages(self, watcher):
        """Test should_process with enough messages."""
        from oghma.parsers import Message

        messages = [Message(role="user", content=f"message {i}") for i in range(10)]
        assert watcher.should_process(messages) is True

    def test_should_process_not_enough_messages(self, watcher):
        """Test should_process with not enough messages."""
        from oghma.parsers import Message

        messages = [Message(role="user", content=f"message {i}") for i in range(3)]
        assert watcher.should_process(messages) is False

    def test_blocked_dirs_filtered(self, watcher, tmp_path):
        """Test that blocked directories are filtered out."""
        test_dir = tmp_path / "test_tool"
        test_dir.mkdir()

        venv_dir = test_dir / ".venv"
        venv_dir.mkdir()
        venv_file = venv_dir / "test.jsonl"
        venv_file.write_text('{"type": "user", "message": {"content": "test"}}')

        normal_dir = test_dir / "normal"
        normal_dir.mkdir()
        normal_file = normal_dir / "test.jsonl"
        normal_file.write_text('{"type": "user", "message": {"content": "test"}}')

        watcher.config["tools"] = {
            "test_tool": {"enabled": True, "paths": [str(test_dir / "*" / "*.jsonl")]}
        }

        files = watcher.discover_files()
        assert venv_file not in files
        assert normal_file in files

    def test_disabled_tools_not_discovered(self, watcher, tmp_path):
        """Test that disabled tools are not discovered."""
        test_dir = tmp_path / "test_tool"
        test_dir.mkdir()

        test_file = test_dir / "test.jsonl"
        test_file.write_text('{"type": "user", "message": {"content": "test"}}')

        watcher.config["tools"] = {
            "test_tool": {"enabled": False, "paths": [str(test_dir / "*.jsonl")]}
        }

        files = watcher.discover_files()
        assert test_file not in files
