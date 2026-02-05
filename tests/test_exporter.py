import json
import tempfile
from pathlib import Path

import pytest

from oghma.exporter import Exporter, ExportOptions
from oghma.storage import Storage


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage(temp_db_path):
    return Storage(db_path=temp_db_path)


@pytest.fixture
def populated_storage(storage):
    storage.add_memory(
        content="Python type hints improve code clarity",
        category="learning",
        source_tool="claude_code",
        source_file="/path/file1.jsonl",
        confidence=0.92,
    )
    storage.add_memory(
        content="SQLite FTS5 supports full-text search",
        category="learning",
        source_tool="opencode",
        source_file="/path/file2.jsonl",
        confidence=0.85,
    )
    storage.add_memory(
        content="Prefer short variable names in loops",
        category="preference",
        source_tool="codex",
        source_file="/path/file3.jsonl",
        confidence=0.90,
    )
    storage.add_memory(
        content="Use click for CLI tools",
        category="workflow",
        source_tool="openclaw",
        source_file="/path/file4.jsonl",
        confidence=0.88,
    )
    return storage


def test_exporter_initialization(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir)
    exporter = Exporter(populated_storage, options)
    assert exporter.storage is populated_storage
    assert exporter.options.output_dir == temp_output_dir
    assert exporter.options.format == "markdown"
    assert exporter.options.group_by == "category"


def test_export_markdown_single_category(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="markdown")
    exporter = Exporter(populated_storage, options)
    file_path = exporter.export_category("learning")

    assert file_path.exists()
    assert file_path.suffix == ".md"
    content = file_path.read_text()
    assert "learning" in content.lower()
    assert "Python type hints" in content
    assert "SQLite FTS5" in content


def test_export_markdown_all_categories(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="markdown", group_by="category")
    exporter = Exporter(populated_storage, options)
    files = exporter.export()

    assert len(files) == 3
    file_names = {f.name for f in files}
    assert "learning.md" in file_names
    assert "preference.md" in file_names
    assert "workflow.md" in file_names

    for file_path in files:
        assert file_path.exists()
        content = file_path.read_text()
        assert "---" in content


def test_export_json(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="json", group_by="category")
    exporter = Exporter(populated_storage, options)
    files = exporter.export()

    assert len(files) == 3

    for file_path in files:
        assert file_path.exists()
        assert file_path.suffix == ".json"
        content = file_path.read_text()
        data = json.loads(content)
        assert "exported_at" in data
        assert "count" in data
        assert "memories" in data


def test_export_group_by_date(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="markdown", group_by="date")
    exporter = Exporter(populated_storage, options)
    files = exporter.export()

    assert len(files) > 0
    for file_path in files:
        assert file_path.exists()
        assert file_path.suffix == ".md"


def test_export_group_by_source(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="markdown", group_by="source")
    exporter = Exporter(populated_storage, options)
    files = exporter.export()

    assert len(files) == 4
    file_names = {f.stem for f in files}
    assert "claude_code" in file_names
    assert "opencode" in file_names
    assert "codex" in file_names
    assert "openclaw" in file_names


def test_export_creates_output_dir(populated_storage):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "nested" / "export" / "dir"
        assert not output_dir.exists()

        options = ExportOptions(output_dir=output_dir, format="markdown", group_by="category")
        exporter = Exporter(populated_storage, options)
        files = exporter.export()

        assert output_dir.exists()
        assert len(files) > 0


def test_export_empty_database(storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="markdown", group_by="category")
    exporter = Exporter(storage, options)
    files = exporter.export()

    assert files == []


def test_export_category_not_found(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="markdown")
    exporter = Exporter(populated_storage, options)

    with pytest.raises(ValueError, match="No memories found for category"):
        exporter.export_category("nonexistent")


def test_export_format_markdown_includes_metadata(populated_storage, temp_output_dir):
    storage = populated_storage
    storage.add_memory(
        content="Memory with metadata",
        category="learning",
        source_tool="claude_code",
        source_file="/path/file5.jsonl",
        confidence=0.95,
        metadata={"tags": ["python", "typing"], "session_id": "123"},
    )

    options = ExportOptions(output_dir=temp_output_dir, format="markdown", group_by="category")
    exporter = Exporter(storage, options)
    files = exporter.export()

    learning_file = [f for f in files if "learning" in f.name][0]
    content = learning_file.read_text()
    assert "Memory with metadata" in content
    assert "category:" in content
    assert "exported_at:" in content
    assert "count:" in content


def test_export_json_serializes_all_fields(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="json", group_by="category")
    exporter = Exporter(populated_storage, options)
    files = exporter.export()

    for file_path in files:
        content = file_path.read_text()
        data = json.loads(content)
        for memory in data["memories"]:
            assert "id" in memory
            assert "content" in memory
            assert "category" in memory
            assert "source_tool" in memory
            assert "source_file" in memory
            assert "confidence" in memory
            assert "created_at" in memory
            assert "updated_at" in memory
            assert "status" in memory
            assert "metadata" in memory


def test_export_invalid_format(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="invalid", group_by="category")
    exporter = Exporter(populated_storage, options)

    with pytest.raises(ValueError, match="Unsupported format"):
        exporter.export()


def test_export_invalid_group_by(populated_storage, temp_output_dir):
    options = ExportOptions(output_dir=temp_output_dir, format="markdown", group_by="invalid")
    exporter = Exporter(populated_storage, options)

    with pytest.raises(ValueError, match="Unsupported group_by"):
        exporter.export()
