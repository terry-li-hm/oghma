import glob
import logging
from pathlib import Path
from typing import Any, cast

from oghma.config import Config
from oghma.parsers import Message, get_parser_for_file
from oghma.storage import Storage

logger = logging.getLogger(__name__)


class Watcher:
    """Discovers and tracks transcript files."""

    BLOCKED_DIRS = {".venv", "node_modules", ".git", "__pycache__", "dist", "build"}

    def __init__(self, config: Config, storage: Storage):
        self.config = config
        self.storage = storage

    def discover_files(self) -> list[Path]:
        """Find all transcript files from configured tool paths."""
        all_files: list[Path] = []
        tools = cast(dict[str, Any], self.config.get("tools") or {})

        for tool_name, tool_config in tools.items():
            if not tool_config.get("enabled", False):
                continue

            paths = tool_config.get("paths") or []
            for pattern in paths:
                files = self._expand_glob_pattern(Path(pattern).expanduser(), tool_name)
                all_files.extend(files)

        return sorted(set(all_files))

    def _expand_glob_pattern(self, pattern: Path, tool_name: str) -> list[Path]:
        """Expand a glob pattern and filter blocked directories."""
        files: list[Path] = []

        try:
            pattern_str = str(pattern)
            for path_str in glob.glob(pattern_str, recursive=True):
                path = Path(path_str)
                files.append(path)

            files = [f for f in files if (f.is_file() or f.is_dir()) and self._is_allowed(f)]
        except (OSError, PermissionError):
            logger.warning(f"Failed to expand pattern: {pattern}")

        return files

    def _is_allowed(self, path: Path) -> bool:
        """Check if path is allowed (not in blocked directories)."""
        parts = path.parts
        return not any(part in self.BLOCKED_DIRS for part in parts)

    def get_changed_files(self) -> list[Path]:
        """Return files that changed since last check (based on mtime)."""
        all_files = self.discover_files()
        changed_files: list[Path] = []

        for file_path in all_files:
            if not file_path.exists():
                continue

            current_mtime = file_path.stat().st_mtime
            current_size = file_path.stat().st_size

            state = self.storage.get_extraction_state(str(file_path))

            if state is None:
                changed_files.append(file_path)
            elif current_mtime > state["last_mtime"] or current_size != state["last_size"]:
                changed_files.append(file_path)

        return changed_files

    def parse_file(self, file_path: Path) -> list[Message]:
        """Parse a transcript file using appropriate parser."""
        parser = get_parser_for_file(file_path)
        if parser:
            return parser.parse(file_path)
        return []

    def should_process(self, messages: list[Message]) -> bool:
        """Check if file has enough messages to process."""
        daemon_config = cast(dict[str, Any], self.config.get("daemon") or {})
        min_messages = daemon_config.get("min_messages", 6)
        return len(messages) >= min_messages
