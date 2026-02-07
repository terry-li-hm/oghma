import os
from copy import deepcopy
from pathlib import Path
from typing_extensions import TypedDict

import yaml


class StorageConfig(TypedDict):
    db_path: str
    backup_enabled: bool
    backup_dir: str
    backup_retention_days: int


class DaemonConfig(TypedDict):
    poll_interval: int
    log_level: str
    log_file: str
    pid_file: str
    min_messages: int


class ExtractionConfig(TypedDict):
    model: str
    max_content_chars: int
    categories: list[str]
    confidence_threshold: float


class ExportConfig(TypedDict):
    output_dir: str
    format: str


class EmbeddingConfig(TypedDict):
    provider: str
    model: str
    dimensions: int
    batch_size: int
    rate_limit_delay: float
    max_retries: int


class ToolConfig(TypedDict, total=False):
    enabled: bool
    paths: list[str]


class ToolsConfig(TypedDict, total=False):
    claude_code: ToolConfig
    codex: ToolConfig
    openclaw: ToolConfig
    opencode: ToolConfig
    cursor: ToolConfig


class Config(TypedDict):
    storage: StorageConfig
    daemon: DaemonConfig
    extraction: ExtractionConfig
    embedding: EmbeddingConfig
    export: ExportConfig
    tools: ToolsConfig


DEFAULT_CONFIG: Config = {
    "storage": {
        "db_path": "~/.oghma/oghma.db",
        "backup_enabled": True,
        "backup_dir": "~/.oghma/backups",
        "backup_retention_days": 30,
    },
    "daemon": {
        "poll_interval": 300,
        "log_level": "INFO",
        "log_file": "~/.oghma/oghma.log",
        "pid_file": "~/.oghma/oghma.pid",
        "min_messages": 6,
    },
    "extraction": {
        "model": "google/gemini-3-flash-preview",
        "max_content_chars": 4000,
        "categories": ["learning", "preference", "project_context", "gotcha", "workflow", "promoted"],
        "confidence_threshold": 0.5,
        "skip_content_patterns": ["MEMORY.md", "write_memory", "edit_memory"],
    },
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "batch_size": 100,
        "rate_limit_delay": 0.1,
        "max_retries": 3,
    },
    "export": {
        "output_dir": "~/.oghma/export",
        "format": "markdown",
    },
    "tools": {
        "claude_code": {"enabled": True, "paths": ["~/.claude/projects/-Users-*/*.jsonl"]},
        "codex": {"enabled": True, "paths": ["~/.codex/sessions/**/rollout-*.jsonl"]},
        "openclaw": {"enabled": True, "paths": ["~/.openclaw/agents/*/sessions/*.jsonl"]},
        "opencode": {"enabled": True, "paths": ["~/.local/share/opencode/storage/message/ses_*"]},
        "cursor": {"enabled": False, "paths": []},
    },
}


def expand_path(path: str) -> str:
    return str(Path(path).expanduser())


def get_config_path() -> Path:
    return Path.home() / ".oghma" / "config.yaml"


def load_config() -> Config:
    config_path = get_config_path()

    if not config_path.exists():
        return create_default_config()

    with open(config_path) as f:
        loaded = yaml.safe_load(f)

    if not loaded:
        return create_default_config()

    from typing import cast

    merged = _merge_defaults(deepcopy(cast(dict, DEFAULT_CONFIG)), loaded)
    merged = _apply_env_overrides(merged)
    _expand_paths_inplace(merged)

    return merged


def create_default_config() -> Config:
    from typing import cast

    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = _apply_env_overrides(cast(Config, deepcopy(DEFAULT_CONFIG)))
    _expand_paths_inplace(config)

    with open(config_path, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)

    return config


def _merge_defaults(defaults: dict, loaded: dict) -> Config:
    from typing import cast

    merged = defaults.copy()
    for key, value in loaded.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return cast(Config, merged)


def _apply_env_overrides(config: Config) -> Config:
    overrides = {
        "OGHMA_DB_PATH": ("storage", "db_path"),
        "OGHMA_BACKUP_ENABLED": ("storage", "backup_enabled"),
        "OGHMA_BACKUP_DIR": ("storage", "backup_dir"),
        "OGHMA_BACKUP_RETENTION_DAYS": ("storage", "backup_retention_days"),
        "OGHMA_POLL_INTERVAL": ("daemon", "poll_interval"),
        "OGHMA_LOG_LEVEL": ("daemon", "log_level"),
        "OGHMA_LOG_FILE": ("daemon", "log_file"),
        "OGHMA_PID_FILE": ("daemon", "pid_file"),
        "OGHMA_DAEMON_MIN_MESSAGES": ("daemon", "min_messages"),
        "OGHMA_EXPORT_DIR": ("export", "output_dir"),
        "OGHMA_EXPORT_FORMAT": ("export", "format"),
        "OGHMA_EXTRACTION_MODEL": ("extraction", "model"),
        "OGHMA_EXTRACTION_MAX_CONTENT_CHARS": ("extraction", "max_content_chars"),
        "OGHMA_EXTRACTION_CATEGORIES": ("extraction", "categories"),
        "OGHMA_EXTRACTION_CONFIDENCE_THRESHOLD": ("extraction", "confidence_threshold"),
        "OGHMA_EMBEDDING_PROVIDER": ("embedding", "provider"),
        "OGHMA_EMBEDDING_MODEL": ("embedding", "model"),
        "OGHMA_EMBEDDING_DIMENSIONS": ("embedding", "dimensions"),
        "OGHMA_EMBEDDING_BATCH_SIZE": ("embedding", "batch_size"),
        "OGHMA_EMBEDDING_RATE_LIMIT_DELAY": ("embedding", "rate_limit_delay"),
        "OGHMA_EMBEDDING_MAX_RETRIES": ("embedding", "max_retries"),
    }

    for env_var, (section, key) in overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            if key in [
                "poll_interval",
                "backup_retention_days",
                "dimensions",
                "batch_size",
                "max_retries",
                "max_content_chars",
                "min_messages",
            ]:
                config[section][key] = int(value)
            elif key in ["backup_enabled"]:
                config[section][key] = value.lower() in ("true", "1", "yes")
            elif key in ["categories"]:
                categories = [item.strip() for item in value.split(",") if item.strip()]
                config[section][key] = categories
            elif key in ["confidence_threshold", "rate_limit_delay"]:
                config[section][key] = float(value)
            else:
                config[section][key] = value

    return config


def _expand_paths_inplace(config: Config) -> None:
    path_keys = [
        ("storage", "db_path"),
        ("storage", "backup_dir"),
        ("daemon", "log_file"),
        ("daemon", "pid_file"),
        ("export", "output_dir"),
    ]

    for section, key in path_keys:
        if section in config and key in config[section]:
            config[section][key] = expand_path(config[section][key])


def validate_config(config: Config) -> list[str]:
    errors: list[str] = []

    required_sections = ["storage", "daemon", "extraction", "embedding", "export"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    if "storage" in config:
        storage = config["storage"]
        if "db_path" not in storage or not storage["db_path"]:
            errors.append("storage.db_path is required")

    if "daemon" in config:
        daemon = config["daemon"]
        if "poll_interval" not in daemon or daemon["poll_interval"] <= 0:
            errors.append("daemon.poll_interval must be positive")
        if "log_level" not in daemon or daemon["log_level"] not in [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ]:
            errors.append("daemon.log_level must be DEBUG, INFO, WARNING, or ERROR")

    if "extraction" in config:
        extraction = config["extraction"]
        if "model" not in extraction or not extraction["model"]:
            errors.append("extraction.model is required")
        if "categories" not in extraction or not extraction["categories"]:
            errors.append("extraction.categories must not be empty")

    if "embedding" in config:
        embedding = config["embedding"]
        if "provider" not in embedding or not embedding["provider"]:
            errors.append("embedding.provider is required")
        if "model" not in embedding or not embedding["model"]:
            errors.append("embedding.model is required")
        if "dimensions" not in embedding or embedding["dimensions"] <= 0:
            errors.append("embedding.dimensions must be positive")

    return errors


def get_db_path(config: Config | None = None) -> str:
    if config is None:
        config = load_config()
    return config["storage"]["db_path"]
