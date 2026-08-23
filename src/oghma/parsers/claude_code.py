from pathlib import Path

from oghma.parsers.base import JsonlParser


class ClaudeCodeParser(JsonlParser):
    def can_parse(self, file_path: Path) -> bool:
        if not file_path.name.endswith(".jsonl"):
            return False
        path_str = str(file_path)
        return ".claude/projects/-Users-" in path_str

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

        # Null or otherwise empty content must not become the literal string
        # "None"; return "" so the caller skips the message.
        return str(content) if content else ""
