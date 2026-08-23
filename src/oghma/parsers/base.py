import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Message:
    role: str
    content: str
    timestamp: str | None = None


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> list[Message]:
        pass

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        pass


class JsonlParser(BaseParser):
    """Base for line-oriented UTF-8 JSONL session transcripts.

    Implements the shared parse loop: blank lines are skipped, malformed
    lines are ignored, and unreadable or non-UTF-8 files yield an empty
    result. Subclasses provide ``can_parse`` plus the per-line
    ``_extract_role``/``_extract_content`` hooks.
    """

    @abstractmethod
    def _extract_role(self, data: dict) -> str | None:
        pass

    @abstractmethod
    def _extract_content(self, data: dict) -> str:
        pass

    def parse(self, file_path: Path) -> list[Message]:
        messages: list[Message] = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        role = self._extract_role(data)
                        content = self._extract_content(data)

                        if role and content:
                            messages.append(Message(role=role, content=content[:3000]))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
        except (OSError, UnicodeDecodeError):
            return []

        return messages
