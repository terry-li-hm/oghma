import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from oghma.storage import MemoryRecord, Storage


@dataclass
class ExportOptions:
    output_dir: Path
    format: str = "markdown"
    group_by: str = "category"
    include_metadata: bool = True
    source_tool: str | None = None


class Exporter:
    def __init__(self, storage: Storage, options: ExportOptions):
        self.storage = storage
        self.options = options

    def export(self) -> list[Path]:
        """Export memories to files, returning list of created file paths."""
        memories = self.storage.get_all_memories(status="active")

        if self.options.source_tool:
            memories = [m for m in memories if m["source_tool"] == self.options.source_tool]

        if not memories:
            return []

        self.options.output_dir.mkdir(parents=True, exist_ok=True)

        if self.options.group_by == "category":
            return self._export_by_category(memories)
        elif self.options.group_by == "date":
            return self._export_by_date(memories)
        elif self.options.group_by == "source":
            return self._export_by_source(memories)
        else:
            raise ValueError(f"Unsupported group_by: {self.options.group_by}")

    def export_category(self, category: str) -> Path:
        """Export memories for a single category."""
        memories = self.storage.get_all_memories(status="active", category=category)

        if self.options.source_tool:
            memories = [m for m in memories if m["source_tool"] == self.options.source_tool]

        if not memories:
            raise ValueError(f"No memories found for category: {category}")

        self.options.output_dir.mkdir(parents=True, exist_ok=True)

        if self.options.format == "markdown":
            content = self._format_markdown(memories, category)
            ext = ".md"
        elif self.options.format == "json":
            content = self._format_json(memories)
            ext = ".json"
        else:
            raise ValueError(f"Unsupported format: {self.options.format}")

        safe_category = category.replace("/", "_").replace("\\", "_")
        filename = f"{safe_category}{ext}"
        file_path = self.options.output_dir / filename

        file_path.write_text(content, encoding="utf-8")
        return file_path

    def _export_by_category(self, memories: list[MemoryRecord]) -> list[Path]:
        categories = {m["category"] for m in memories}
        files = []

        for category in sorted(categories):
            category_memories = [m for m in memories if m["category"] == category]

            if self.options.format == "markdown":
                content = self._format_markdown(category_memories, category)
                ext = ".md"
            elif self.options.format == "json":
                content = self._format_json(category_memories)
                ext = ".json"
            else:
                raise ValueError(f"Unsupported format: {self.options.format}")

            safe_category = category.replace("/", "_").replace("\\", "_")
            filename = f"{safe_category}{ext}"
            file_path = self.options.output_dir / filename

            file_path.write_text(content, encoding="utf-8")
            files.append(file_path)

        return files

    def _export_by_date(self, memories: list[MemoryRecord]) -> list[Path]:
        dates = {m["created_at"][:10] for m in memories}
        files = []

        for date_str in sorted(dates):
            date_memories = [m for m in memories if m["created_at"].startswith(date_str)]

            if self.options.format == "markdown":
                content = self._format_markdown(date_memories, date_str)
                ext = ".md"
            elif self.options.format == "json":
                content = self._format_json(date_memories)
                ext = ".json"
            else:
                raise ValueError(f"Unsupported format: {self.options.format}")

            filename = f"{date_str}{ext}"
            file_path = self.options.output_dir / filename

            file_path.write_text(content, encoding="utf-8")
            files.append(file_path)

        return files

    def _export_by_source(self, memories: list[MemoryRecord]) -> list[Path]:
        sources = {m["source_tool"] for m in memories}
        files = []

        for source in sorted(sources):
            source_memories = [m for m in memories if m["source_tool"] == source]

            if self.options.format == "markdown":
                content = self._format_markdown(source_memories, source)
                ext = ".md"
            elif self.options.format == "json":
                content = self._format_json(source_memories)
                ext = ".json"
            else:
                raise ValueError(f"Unsupported format: {self.options.format}")

            safe_source = source.replace("/", "_").replace("\\", "_")
            filename = f"{safe_source}{ext}"
            file_path = self.options.output_dir / filename

            file_path.write_text(content, encoding="utf-8")
            files.append(file_path)

        return files

    def _format_markdown(self, memories: list[MemoryRecord], title: str) -> str:
        """Format memories as markdown with YAML frontmatter."""
        lines = [
            "---",
            f"category: {title}",
            f"exported_at: {datetime.now().isoformat()}",
            f"count: {len(memories)}",
            "---",
            "",
            f"# {title.title()}",
            "",
        ]

        for memory in memories:
            content_preview = (
                memory["content"][:80] + "..." if len(memory["content"]) > 80 else memory["content"]
            )
            lines.append(f"## {content_preview}")
            source_info = (
                f"*Source: {memory['source_tool']} | {memory['created_at'][:10]} | "
                f"Confidence: {memory['confidence']:.0%}*"
            )
            lines.append(source_info)
            lines.append("")
            lines.append(memory["content"])
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _format_json(self, memories: list[MemoryRecord]) -> str:
        """Format memories as JSON."""
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "count": len(memories),
            "memories": memories,
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
