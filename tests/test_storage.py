import tempfile
from pathlib import Path

import pytest

from oghma.storage import Storage


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def storage(temp_db_path):
    return Storage(db_path=temp_db_path)


def test_storage_init(storage):
    assert storage is not None
    assert Path(storage.db_path).exists()


def test_add_memory(storage):
    memory_id = storage.add_memory(
        content="Test memory content",
        category="learning",
        source_tool="claude_code",
        source_file="/path/to/file.jsonl",
    )
    assert memory_id > 0

    memory = storage.get_memory_by_id(memory_id)
    assert memory is not None
    assert memory["content"] == "Test memory content"
    assert memory["category"] == "learning"
    assert memory["source_tool"] == "claude_code"


def test_add_memory_with_optional_fields(storage):
    memory_id = storage.add_memory(
        content="Test memory with metadata",
        category="preference",
        source_tool="codex",
        source_file="/path/to/file.jsonl",
        source_session="session-123",
        confidence=0.9,
        metadata={"key": "value"},
    )
    assert memory_id > 0

    memory = storage.get_memory_by_id(memory_id)
    assert memory["source_session"] == "session-123"
    assert memory["confidence"] == 0.9
    assert memory["metadata"] == {"key": "value"}


def test_search_memories(storage):
    storage.add_memory(
        content="Python is a great language for data science",
        category="learning",
        source_tool="claude_code",
        source_file="/path/file1.jsonl",
    )
    storage.add_memory(
        content="JavaScript is used for frontend development",
        category="learning",
        source_tool="codex",
        source_file="/path/file2.jsonl",
    )

    results = storage.search_memories("Python")
    assert len(results) > 0
    assert any("Python" in result["content"] for result in results)


def test_search_memories_with_filters(storage):
    storage.add_memory(
        content="Python learning",
        category="learning",
        source_tool="claude_code",
        source_file="/path/file1.jsonl",
    )
    storage.add_memory(
        content="Python preference",
        category="preference",
        source_tool="codex",
        source_file="/path/file2.jsonl",
    )

    results = storage.search_memories("Python", category="learning")
    assert len(results) == 1
    assert results[0]["category"] == "learning"

    results = storage.search_memories("Python", source_tool="codex")
    assert len(results) == 1
    assert results[0]["source_tool"] == "codex"


def test_search_memories_no_results(storage):
    results = storage.search_memories("nonexistent")
    assert len(results) == 0


def test_search_memories_with_limit(storage):
    for i in range(5):
        storage.add_memory(
            content=f"Memory {i}",
            category="learning",
            source_tool="claude_code",
            source_file=f"/path/file{i}.jsonl",
        )

    results = storage.search_memories("Memory", limit=2)
    assert len(results) == 2


def test_get_memory_by_id_not_found(storage):
    memory = storage.get_memory_by_id(999)
    assert memory is None


def test_update_memory_status(storage):
    memory_id = storage.add_memory(
        content="Test memory",
        category="learning",
        source_tool="claude_code",
        source_file="/path/file.jsonl",
    )

    result = storage.update_memory_status(memory_id, "archived")
    assert result is True

    memory = storage.get_memory_by_id(memory_id)
    assert memory["status"] == "archived"

    result = storage.update_memory_status(999, "deleted")
    assert result is False


def test_get_extraction_state_not_found(storage):
    state = storage.get_extraction_state("/nonexistent/path")
    assert state is None


def test_update_extraction_state(storage):
    storage.update_extraction_state(
        source_path="/path/to/file.jsonl",
        last_mtime=123456.789,
        last_size=1024,
        message_count=10,
    )

    state = storage.get_extraction_state("/path/to/file.jsonl")
    assert state is not None
    assert state["source_path"] == "/path/to/file.jsonl"
    assert state["last_mtime"] == 123456.789
    assert state["last_size"] == 1024
    assert state["message_count"] == 10


def test_update_extraction_state_upsert(storage):
    storage.update_extraction_state(
        source_path="/path/to/file.jsonl",
        last_mtime=111.111,
        last_size=512,
        message_count=5,
    )

    state = storage.get_extraction_state("/path/to/file.jsonl")
    assert state["last_mtime"] == 111.111

    storage.update_extraction_state(
        source_path="/path/to/file.jsonl",
        last_mtime=222.222,
        last_size=1024,
        message_count=10,
    )

    state = storage.get_extraction_state("/path/to/file.jsonl")
    assert state["last_mtime"] == 222.222
    assert state["last_size"] == 1024
    assert state["message_count"] == 10


def test_log_extraction(storage):
    log_id = storage.log_extraction(
        source_path="/path/to/file.jsonl",
        memories_extracted=5,
        tokens_used=1000,
        duration_ms=1500,
    )
    assert log_id > 0

    logs = storage.get_recent_extraction_logs(limit=1)
    assert len(logs) == 1
    assert logs[0]["memories_extracted"] == 5
    assert logs[0]["tokens_used"] == 1000
    assert logs[0]["duration_ms"] == 1500


def test_log_extraction_with_error(storage):
    log_id = storage.log_extraction(
        source_path="/path/to/file.jsonl",
        memories_extracted=0,
        tokens_used=0,
        duration_ms=0,
        error="API timeout",
    )
    assert log_id > 0

    logs = storage.get_recent_extraction_logs(limit=1)
    assert logs[0]["error"] == "API timeout"


def test_get_memory_count(storage):
    assert storage.get_memory_count() == 0

    storage.add_memory(
        content="Memory 1",
        category="learning",
        source_tool="claude_code",
        source_file="/path/file1.jsonl",
    )
    storage.add_memory(
        content="Memory 2",
        category="preference",
        source_tool="codex",
        source_file="/path/file2.jsonl",
    )

    assert storage.get_memory_count() == 2


def test_get_memory_count_by_status(storage):
    storage.add_memory(
        content="Active memory",
        category="learning",
        source_tool="claude_code",
        source_file="/path/file1.jsonl",
    )
    memory_id = storage.add_memory(
        content="Memory to archive",
        category="learning",
        source_tool="codex",
        source_file="/path/file2.jsonl",
    )
    storage.update_memory_status(memory_id, "archived")

    assert storage.get_memory_count(status="active") == 1
    assert storage.get_memory_count(status="archived") == 1


def test_get_all_extraction_states(storage):
    storage.update_extraction_state("/path1.jsonl", 111.0, 512, 5)
    storage.update_extraction_state("/path2.jsonl", 222.0, 1024, 10)

    states = storage.get_all_extraction_states()
    assert len(states) == 2


def test_get_recent_extraction_logs(storage):
    for i in range(3):
        storage.log_extraction(
            source_path=f"/path{i}.jsonl",
            memories_extracted=i,
            tokens_used=i * 100,
            duration_ms=i * 50,
        )

    logs = storage.get_recent_extraction_logs(limit=2)
    assert len(logs) == 2

    logs_all = storage.get_recent_extraction_logs(limit=10)
    assert len(logs_all) == 3
