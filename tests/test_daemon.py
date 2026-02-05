import os
import signal
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from oghma.config import Config
from oghma.daemon import Daemon, get_daemon_pid


@pytest.fixture
def temp_config() -> Generator[Config, None, None]:
    """Create a temporary config for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            {
                "storage": {
                    "db_path": str(Path(tmpdir) / "oghma.db"),
                    "backup_enabled": False,
                    "backup_dir": str(Path(tmpdir) / "backups"),
                    "backup_retention_days": 30,
                },
                "daemon": {
                    "poll_interval": 1,
                    "log_level": "DEBUG",
                    "log_file": str(Path(tmpdir) / "oghma.log"),
                    "pid_file": str(Path(tmpdir) / "oghma.pid"),
                    "min_messages": 1,
                },
                "extraction": {
                    "model": "gpt-4o-mini",
                    "max_content_chars": 1000,
                    "categories": ["learning", "preference"],
                    "confidence_threshold": 0.5,
                },
                "export": {
                    "output_dir": str(Path(tmpdir) / "export"),
                    "format": "markdown",
                },
                "tools": {},
            }
        )
        yield config


@pytest.fixture
def daemon(temp_config: Config) -> Generator[Daemon, None, None]:
    """Create a daemon instance for testing."""
    with patch("oghma.daemon.Extractor"):
        yield Daemon(temp_config)


def test_daemon_initialization(temp_config: Config) -> None:
    """Test that daemon initializes correctly."""
    with patch("oghma.daemon.Extractor"):
        daemon = Daemon(temp_config)
        assert daemon.config == temp_config
        assert daemon._running is False


def test_get_daemon_pid_no_file() -> None:
    """Test get_daemon_pid when PID file doesn't exist."""
    result = get_daemon_pid("/nonexistent/pidfile.pid")
    assert result is None


def test_get_daemon_pid_with_running_process(temp_config: Config) -> None:
    """Test get_daemon_pid with a running process."""
    pid_file = temp_config["daemon"]["pid_file"]

    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    result = get_daemon_pid(pid_file)
    assert result == os.getpid()

    Path(pid_file).unlink()


def test_get_daemon_pid_with_dead_process(temp_config: Config) -> None:
    """Test get_daemon_pid with a non-existent process."""
    pid_file = temp_config["daemon"]["pid_file"]

    with open(pid_file, "w") as f:
        f.write("999999")

    result = get_daemon_pid(pid_file)
    assert result is None

    assert not Path(pid_file).exists()


def test_get_daemon_pid_invalid_content(temp_config: Config) -> None:
    """Test get_daemon_pid with invalid PID content."""
    pid_file = temp_config["daemon"]["pid_file"]

    with open(pid_file, "w") as f:
        f.write("not-a-pid")

    result = get_daemon_pid(pid_file)
    assert result is None

    assert not Path(pid_file).exists()


def test_daemon_write_pid_file(daemon: Daemon) -> None:
    """Test that daemon writes PID file correctly."""
    daemon._write_pid_file(daemon.config["daemon"]["pid_file"])
    pid_file = Path(daemon.config["daemon"]["pid_file"])

    assert pid_file.exists()

    with open(pid_file) as f:
        content = f.read()
        assert content == str(os.getpid())

    pid_file.unlink()


def test_daemon_cleanup(daemon: Daemon) -> None:
    """Test that daemon cleanup removes PID file."""
    pid_file = Path(daemon.config["daemon"]["pid_file"])
    pid_file.touch()

    daemon._cleanup(daemon.config["daemon"]["pid_file"])

    assert not pid_file.exists()


def test_daemon_get_tool_name(daemon: Daemon) -> None:
    """Test tool name extraction from file paths."""
    assert (
        daemon._get_tool_name(Path("/home/user/.claude/projects/test/file.jsonl")) == "claude_code"
    )
    assert daemon._get_tool_name(Path("/home/user/.codex/sessions/file.jsonl")) == "codex"
    assert daemon._get_tool_name(Path("/home/user/.openclaw/agents/test/file.jsonl")) == "openclaw"
    assert (
        daemon._get_tool_name(Path("/home/user/.local/share/opencode/storage/file")) == "opencode"
    )
    assert daemon._get_tool_name(Path("/some/other/path/file.txt")) == "unknown"


def test_daemon_get_session_id(daemon: Daemon) -> None:
    """Test session ID extraction from file paths."""
    assert daemon._get_session_id(Path("/path/ses_12345/file.jsonl")) == "ses_12345"
    assert daemon._get_session_id(Path("/path/rollout-abc123/file.jsonl")) == "rollout-abc123"
    assert daemon._get_session_id(Path("/some/other/path/file.jsonl")) is None


def test_daemon_setup_logging(daemon: Daemon) -> None:
    """Test that logging is set up correctly."""
    daemon._setup_logging()

    log_file = Path(daemon.config["daemon"]["log_file"])
    assert log_file.parent.exists()


def test_signal_handler_sets_running_false(daemon: Daemon) -> None:
    """Test that signal handler sets _running to False."""
    daemon._running = True
    daemon._signal_handler(signal.SIGTERM, None)
    assert daemon._running is False
