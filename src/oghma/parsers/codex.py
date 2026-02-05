import json
from pathlib import Path

from oghma.parsers.base import BaseParser, Message


class CodexParser(BaseParser):
    def can_parse(self, file_path: Path) -> bool:
        if not file_path.name.endswith(".jsonl"):
            return False
        path_str = str(file_path)
        return ".codex/sessions/" in path_str and "rollout-" in file_path.name

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

    def _extract_role(self, data: dict) -> str | None:
        if data.get("type") != "item":
            return None

        payload = data.get("payload", {})
        item = payload.get("item", {})
        return item.get("role")

    def _extract_content(self, data: dict) -> str:
        payload = data.get("payload", {})
        item = payload.get("item", {})
        content = item.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    text = block.get("text", "")
                    if text and block_type in ("input_text", "output_text"):
                        parts.append(text)
            return "\n".join(parts)

        return str(content)
