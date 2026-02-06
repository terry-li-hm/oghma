import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from oghma.cli import cli
from oghma.config import Config


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


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
                    "poll_interval": 300,
                    "log_level": "INFO",
                    "log_file": str(Path(tmpdir) / "oghma.log"),
                    "pid_file": str(Path(tmpdir) / "oghma.pid"),
                    "min_messages": 6,
                },
                "extraction": {
                    "model": "gpt-4o-mini",
                    "max_content_chars": 4000,
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
def temp_db(temp_config: Config) -> Generator[Path, None, None]:
    """Create a temporary database with test data."""
    from oghma.storage import Storage

    storage = Storage(temp_config["storage"]["db_path"], temp_config)

    storage.add_memory(
        content="Test learning content",
        category="learning",
        source_tool="claude_code",
        source_file="/test/file.jsonl",
        source_session="ses_123",
        confidence=0.9,
    )

    storage.add_memory(
        content="Test preference content",
        category="preference",
        source_tool="opencode",
        source_file="/test/file2.jsonl",
        confidence=0.8,
    )

    yield Path(temp_config["storage"]["db_path"])


def test_cli_help(runner: CliRunner) -> None:
    """Test CLI help output."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "status" in result.output
    assert "start" in result.output
    assert "stop" in result.output
    assert "search" in result.output


def test_init_creates_config(runner: CliRunner, temp_config: Config) -> None:
    """Test that init command reports success on fresh run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Must patch where imported (oghma.cli), not where defined (oghma.config)
        with patch("oghma.cli.get_config_path", return_value=config_path):
            with patch("oghma.cli.create_default_config") as mock_create:
                mock_create.return_value = temp_config
                result = runner.invoke(cli, ["init"])

                assert result.exit_code == 0
                assert "Creating" in result.output
                mock_create.assert_called_once()


def test_init_overwrites_existing(runner: CliRunner) -> None:
    """Test that init can overwrite existing config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("old: config")

        with patch("oghma.cli.get_config_path", return_value=config_path):
            result = runner.invoke(cli, ["init"], input="y")

            assert result.exit_code == 0
            assert config_path.exists()


def test_init_cancelled(runner: CliRunner) -> None:
    """Test that init can be cancelled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("old: config")

        with patch("oghma.cli.get_config_path", return_value=config_path):
            result = runner.invoke(cli, ["init"], input="n")

            assert result.exit_code == 0
            assert "cancelled" in result.output.lower()


def test_status_shows_info(runner: CliRunner, temp_config: Config, temp_db: Path) -> None:
    """Test that status command shows database info."""
    with patch("oghma.cli.load_config", return_value=temp_config):
        with patch("oghma.cli.get_config_path", return_value=Path("/fake/config.yaml")):
            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Memory Count" in result.output
            assert "Database Status" in result.output


def test_status_with_empty_db(runner: CliRunner, temp_config: Config) -> None:
    """Test status when database doesn't exist yet."""
    db_path = Path(temp_config["storage"]["db_path"])
    if db_path.exists():
        db_path.unlink()

    with patch("oghma.cli.load_config", return_value=temp_config):
        with patch("oghma.cli.get_config_path", return_value=Path("/fake/config.yaml")):
            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Not created yet" in result.output


def test_search_no_results(runner: CliRunner, temp_config: Config) -> None:
    """Test search with no matching results."""
    with patch("oghma.cli.load_config", return_value=temp_config):
        result = runner.invoke(cli, ["search", "nonexistent"])

        assert result.exit_code == 0
        assert "No memories found" in result.output


def test_search_with_results(runner: CliRunner, temp_config: Config, temp_db: Path) -> None:
    """Test search with matching results."""
    with patch("oghma.cli.load_config", return_value=temp_config):
        result = runner.invoke(cli, ["search", "learning"])

        assert result.exit_code == 0
        assert "learning" in result.output.lower()


def test_search_with_category_filter(runner: CliRunner, temp_config: Config, temp_db: Path) -> None:
    """Test search with category filter."""
    with patch("oghma.cli.load_config", return_value=temp_config):
        result = runner.invoke(cli, ["search", "content", "-c", "preference"])

        assert result.exit_code == 0
        assert "preference" in result.output.lower()


def test_search_with_status_filter(
    runner: CliRunner, temp_config: Config, temp_db: Path
) -> None:
    """Test search with status filter."""
    from oghma.storage import Storage

    storage = Storage(temp_config["storage"]["db_path"], temp_config)
    archived = storage.add_memory(
        content="Archived memory content",
        category="learning",
        source_tool="claude_code",
        source_file="/test/file3.jsonl",
    )
    assert archived is not None
    storage.update_memory_status(archived, "archived")

    with patch("oghma.cli.load_config", return_value=temp_config):
        result = runner.invoke(cli, ["search", "Archived", "--status", "archived"])

        assert result.exit_code == 0
        assert "archived" in result.output.lower()


def test_search_with_limit(runner: CliRunner, temp_config: Config, temp_db: Path) -> None:
    """Test search with limit option."""
    with patch("oghma.cli.load_config", return_value=temp_config):
        result = runner.invoke(cli, ["search", "content", "-n", "1"])

        assert result.exit_code == 0


def test_search_no_config(runner: CliRunner) -> None:
    """Test search when config doesn't exist."""
    with patch("oghma.cli.load_config", side_effect=FileNotFoundError):
        result = runner.invoke(cli, ["search", "test"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


def test_start_daemon_already_running(runner: CliRunner, temp_config: Config) -> None:
    """Test start when daemon is already running."""
    import os

    pid_file = Path(temp_config["daemon"]["pid_file"])
    pid_file.write_text(str(os.getpid()))

    with patch("oghma.cli.load_config", return_value=temp_config):
        result = runner.invoke(cli, ["start"])

        assert result.exit_code == 1
        assert "already running" in result.output.lower()

    pid_file.unlink(missing_ok=True)


def test_start_no_config(runner: CliRunner) -> None:
    """Test start when config doesn't exist."""
    with patch("oghma.cli.load_config", side_effect=FileNotFoundError):
        result = runner.invoke(cli, ["start"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


def test_stop_no_daemon(runner: CliRunner, temp_config: Config) -> None:
    """Test stop when daemon is not running."""
    with patch("oghma.cli.load_config", return_value=temp_config):
        result = runner.invoke(cli, ["stop"])

        assert result.exit_code == 0
        assert "not running" in result.output.lower()


def test_stop_no_config(runner: CliRunner) -> None:
    """Test stop when config doesn't exist."""
    with patch("oghma.cli.load_config", side_effect=FileNotFoundError):
        result = runner.invoke(cli, ["stop"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


def test_status_shows_daemon_status(runner: CliRunner, temp_config: Config) -> None:
    """Test status shows daemon running/stopped status."""
    with patch("oghma.cli.load_config", return_value=temp_config):
        with patch("oghma.cli.get_config_path", return_value=Path("/fake/config.yaml")):
            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Daemon Status" in result.output
            assert "Stopped" in result.output or "Running" in result.output
