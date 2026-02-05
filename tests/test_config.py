import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from oghma.config import (
    DEFAULT_CONFIG,
    create_default_config,
    expand_path,
    get_config_path,
    get_db_path,
    load_config,
    validate_config,
)


def test_expand_path():
    assert expand_path("~") != "~"
    assert expand_path("~").startswith("/")
    assert expand_path("/tmp") == "/tmp"
    assert expand_path("/tmp/test") == "/tmp/test"


def test_get_config_path():
    config_path = get_config_path()
    assert config_path.name == "config.yaml"
    assert ".oghma" in str(config_path)


@patch("oghma.config.get_config_path")
def test_create_default_config(mock_get_config_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config_path = Path(tmpdir) / "config.yaml"
        mock_get_config_path.return_value = mock_config_path

        config = create_default_config()

        assert config is not None
        assert "storage" in config
        assert "daemon" in config
        assert "extraction" in config
        assert "export" in config
        assert "tools" in config
        assert mock_config_path.exists()


@patch("oghma.config.get_config_path")
def test_load_config_creates_default_if_not_exists(mock_get_config_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config_path = Path(tmpdir) / "config.yaml"
        mock_get_config_path.return_value = mock_config_path

        config = load_config()

        assert config is not None
        assert "storage" in config
        assert mock_config_path.exists()


def test_validate_config_valid():
    errors = validate_config(DEFAULT_CONFIG)
    assert len(errors) == 0


def test_validate_config_missing_section():
    invalid_config = {"storage": {"db_path": "/tmp/test.db"}}
    errors = validate_config(invalid_config)
    assert len(errors) > 0
    assert any("daemon" in error for error in errors)


def test_validate_config_invalid_log_level():
    invalid_config = {
        "storage": {"db_path": "/tmp/test.db"},
        "daemon": {"poll_interval": 300, "log_level": "INVALID"},
        "extraction": {"model": "gpt-4o-mini", "categories": ["learning"]},
        "export": {"output_dir": "/tmp/export", "format": "markdown"},
    }
    errors = validate_config(invalid_config)
    assert any("log_level" in error for error in errors)


def test_validate_config_missing_db_path():
    invalid_config = {
        "storage": {},
        "daemon": {"poll_interval": 300, "log_level": "INFO"},
        "extraction": {"model": "gpt-4o-mini", "categories": ["learning"]},
        "export": {"output_dir": "/tmp/export", "format": "markdown"},
    }
    errors = validate_config(invalid_config)
    assert any("db_path" in error for error in errors)


def test_get_db_path():
    db_path = get_db_path()
    assert db_path is not None
    assert isinstance(db_path, str)
    assert "oghma.db" in db_path


@patch.dict(os.environ, {"OGHMA_DB_PATH": "/custom/oghma.db"})
def test_env_override_db_path():
    db_path = get_db_path()
    assert db_path == "/custom/oghma.db"


@patch("oghma.config.get_config_path")
@patch.dict(os.environ, {"OGHMA_POLL_INTERVAL": "600"})
def test_env_override_poll_interval(mock_get_config_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config_path = Path(tmpdir) / "config.yaml"
        mock_get_config_path.return_value = mock_config_path

        config = create_default_config()
        assert config["daemon"]["poll_interval"] == 600


@patch("oghma.config.get_config_path")
@patch.dict(os.environ, {"OGHMA_LOG_LEVEL": "DEBUG"})
def test_env_override_log_level(mock_get_config_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_config_path = Path(tmpdir) / "config.yaml"
        mock_get_config_path.return_value = mock_config_path

        config = create_default_config()
        assert config["daemon"]["log_level"] == "DEBUG"
