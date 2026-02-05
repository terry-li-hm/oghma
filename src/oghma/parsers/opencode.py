import json
from pathlib import Path

from oghma.parsers.base import BaseParser, Message


class OpenCodeParser(BaseParser):
    def can_parse(self, file_path: Path) -> bool:
        if not file_path.is_dir():
            return False
        path_str = str(file_path)
        return ".local/share/opencode/storage/message/ses_" in path_str

    def parse(self, file_path: Path) -> list[Message]:
        messages: list[Message] = []

        message_files = sorted(file_path.glob("msg_*.json"))
        part_files = list(file_path.glob("part/msg_*/prt_*.json"))

        parts_map = self._build_parts_map(part_files)

        for msg_file in message_files:
            try:
                with open(msg_file, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            role = self._extract_role(data)
            if not role:
                continue

            content_parts = self._get_message_content(data, parts_map)
            if content_parts:
                content = "\n".join(content_parts)
                messages.append(Message(role=role, content=content[:3000]))

        return messages

    def _build_parts_map(self, part_files: list[Path]) -> dict[str, list[str]]:
        parts_map: dict[str, list[str]] = {}

        for part_file in part_files:
            try:
                with open(part_file, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            msg_id = data.get("message_id")
            if not msg_id:
                continue

            if msg_id not in parts_map:
                parts_map[msg_id] = []

            text = data.get("text", "")
            if text:
                parts_map[msg_id].append(text)

        return parts_map

    def _extract_role(self, data: dict) -> str | None:
        role = data.get("role")
        if role in ("user", "assistant"):
            return role
        return None

    def _get_message_content(self, data: dict, parts_map: dict[str, list[str]]) -> list[str]:
        msg_id = data.get("id")
        if msg_id in parts_map:
            return parts_map[msg_id]

        content = data.get("content", "")
        if isinstance(content, str):
            return [content]

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text = item.get("text", "")
                    if text:
                        parts.append(text)
            return parts

        if content:
            return [str(content)]

        return []
