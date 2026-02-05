import json
from pathlib import Path

from oghma.parsers.base import BaseParser, Message


class ClaudeCodeParser(BaseParser):
    def can_parse(self, file_path: Path) -> bool:
        if not file_path.name.endswith(".jsonl"):
            return False
        path_str = str(file_path)
        return ".claude/projects/-Users-" in path_str

    def parse(self, file_path: Path) -> list[Message]:
        messages: list[Message] = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for _line_num, line in enumerate(f, 1):
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

    def _extract_role(self, data: dict) -> str | None:
        msg_type = data.get("type")
        if msg_type == "user":
            return "user"
        elif msg_type == "assistant":
            return "assistant"
        return None

    def _extract_content(self, data: dict) -> str:
        message = data.get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
            return "\n".join(parts)

        return str(content)
