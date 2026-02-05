from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
