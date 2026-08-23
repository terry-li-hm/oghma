from pathlib import Path

from oghma.parsers.base import JsonlParser


class CodexParser(JsonlParser):
    def can_parse(self, file_path: Path) -> bool:
        if not file_path.name.endswith(".jsonl"):
            return False
        path_str = str(file_path)
        return ".codex/sessions/" in path_str and "rollout-" in file_path.name

    def _extract_role(self, data: dict) -> str | None:
        msg_type = data.get("type")
        # Support both old format (item) and new format (response_item, event_msg)
        if msg_type not in ("item", "response_item", "event_msg"):
            return None

        payload = data.get("payload", {})

        # New format: role directly in payload.
        # Old format: nested in payload.item.
        if "role" in payload:
            role = payload.get("role")
        else:
            item = payload.get("item", {})
            role = item.get("role")

        # Map developer/assistant to assistant, user to user; skip everything
        # else (system, tool, ...) in both formats.
        if role in ("developer", "assistant"):
            return "assistant"
        elif role == "user":
            return "user"
        return None

    def _extract_content(self, data: dict) -> str:
        payload = data.get("payload", {})

        # New format: content directly in payload
        if "content" in payload:
            content = payload.get("content", "")
        else:
            # Old format: nested in payload.item
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
                    if text and block_type in ("input_text", "output_text", "text"):
                        parts.append(text)
            return "\n".join(parts)

        # Null or otherwise empty content must not become the literal string
        # "None"; return "" so the caller skips the message.
        return str(content) if content else ""
