import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.server.fastmcp")

from oghma import mcp_server
from oghma.config import Config
from oghma.storage import Storage


@pytest.fixture
def temp_config() -> Config:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Config(
            {
                "storage": {
                    "db_path": str(Path(tmpdir) / "oghma.db"),
                    "backup_enabled": False,
                    "backup_dir": str(Path(tmpdir) / "backups"),
                    "backup_retention_days": 30,
                },
                "daemon": {
                    "poll_interval": 300,
                    "log_level": "INFO",
                    "log_file": str(Path(tmpdir) / "oghma.log"),
                    "pid_file": str(Path(tmpdir) / "oghma.pid"),
                    "min_messages": 6,
                },
                "extraction": {
                    "model": "gpt-4o-mini",
                    "max_content_chars": 4000,
                    "categories": [
                        "learning",
                        "preference",
                        "project_context",
                        "gotcha",
                        "workflow",
                    ],
                    "confidence_threshold": 0.5,
                },
                "export": {"output_dir": str(Path(tmpdir) / "export"), "format": "markdown"},
                "tools": {},
            }
        )


@pytest.fixture
def seeded_storage(temp_config: Config) -> Storage:
    storage = Storage(config=temp_config)
    storage.add_memory(
        content="Python gotcha with async context",
        category="gotcha",
        source_tool="claude_code",
        source_file="/tmp/one.jsonl",
    )
    storage.add_memory(
        content="Use structured plans for big refactors",
        category="workflow",
        source_tool="codex",
        source_file="/tmp/two.jsonl",
    )
    storage.log_extraction(source_path="/tmp/one.jsonl", memories_extracted=2)
    return storage


def test_lifespan_initializes_writable_storage(
    monkeypatch: pytest.MonkeyPatch, temp_config: Config
) -> None:
    Storage(config=temp_config)
    monkeypatch.setattr(mcp_server, "load_config", lambda: temp_config)

    async def _run() -> None:
        async with mcp_server.lifespan(mcp_server.mcp) as context:
            storage = context["storage"]
            assert storage.read_only is False
            assert storage.get_memory_count() == 0

    asyncio.run(_run())


def test_oghma_search_uses_storage_context(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage = MagicMock()
    fake_storage.search_memories_hybrid.return_value = [{"id": 1, "content": "hit"}]
    monkeypatch.setattr(
        mcp_server,
        "_get_lifespan_context",
        lambda: {"storage": fake_storage},
    )
    monkeypatch.setattr(
        mcp_server,
        "create_embedder",
        lambda _: (_ for _ in ()).throw(ValueError("no embedder")),
    )

    results = mcp_server.oghma_search(
        query="hit",
        category="learning",
        source_tool="claude_code",
        limit=5,
    )

    assert results == [{"id": 1, "content": "hit"}]
    fake_storage.search_memories_hybrid.assert_called_once_with(
        query="hit",
        query_embedding=None,
        category="learning",
        source_tool="claude_code",
        limit=5,
        search_mode="hybrid",
    )


def test_oghma_search_rejects_invalid_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage = MagicMock()
    monkeypatch.setattr(mcp_server, "_get_lifespan_context", lambda: {"storage": fake_storage})

    with pytest.raises(ValueError, match="limit must be >= 1"):
        mcp_server.oghma_search(query="hit", limit=0)


def test_oghma_search_rejects_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage = MagicMock()
    monkeypatch.setattr(mcp_server, "_get_lifespan_context", lambda: {"storage": fake_storage})

    with pytest.raises(ValueError, match="search_mode must be one of"):
        mcp_server.oghma_search(query="hit", search_mode="invalid")


def test_oghma_get_uses_storage_context(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage = MagicMock()
    fake_storage.get_memory_by_id.return_value = {"id": 42, "content": "memory"}
    monkeypatch.setattr(mcp_server, "_get_lifespan_context", lambda: {"storage": fake_storage})

    result = mcp_server.oghma_get(memory_id=42)

    assert result == {"id": 42, "content": "memory"}
    fake_storage.get_memory_by_id.assert_called_once_with(42)


def test_oghma_stats_returns_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage = MagicMock()
    fake_storage.get_memory_count.return_value = 3
    fake_storage.get_all_memories.return_value = [
        {"category": "learning", "source_tool": "claude_code"},
        {"category": "learning", "source_tool": "codex"},
        {"category": "gotcha", "source_tool": "codex"},
    ]
    fake_storage.get_recent_extraction_logs.return_value = [{"created_at": "2026-02-05 10:00:00"}]
    monkeypatch.setattr(mcp_server, "_get_lifespan_context", lambda: {"storage": fake_storage})

    result = mcp_server.oghma_stats()

    assert result["total_memories"] == 3
    assert result["memories_by_category"] == {"learning": 2, "gotcha": 1}
    assert result["memories_by_source"] == {"claude_code": 1, "codex": 2}
    assert result["last_extraction_time"] == "2026-02-05 10:00:00"


def test_oghma_categories_returns_sorted_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage = MagicMock()
    fake_storage.get_all_memories.return_value = [
        {"category": "workflow"},
        {"category": "learning"},
        {"category": "workflow"},
    ]
    monkeypatch.setattr(mcp_server, "_get_lifespan_context", lambda: {"storage": fake_storage})

    result = mcp_server.oghma_categories()

    assert result == [
        {"category": "workflow", "count": 2},
        {"category": "learning", "count": 1},
    ]


def test_oghma_tools_work_with_real_storage_context(
    monkeypatch: pytest.MonkeyPatch, seeded_storage: Storage
) -> None:
    read_only_storage = Storage(db_path=seeded_storage.db_path, read_only=True)
    monkeypatch.setattr(mcp_server, "_get_lifespan_context", lambda: {"storage": read_only_storage})

    search_results = mcp_server.oghma_search(query="Python", limit=10)
    assert len(search_results) == 1
    assert search_results[0]["category"] == "gotcha"

    memory = mcp_server.oghma_get(memory_id=search_results[0]["id"])
    assert memory is not None
    assert memory["source_tool"] == "claude_code"

    stats = mcp_server.oghma_stats()
    assert stats["total_memories"] == 2
    assert stats["memories_by_category"]["gotcha"] == 1
    assert stats["last_extraction_time"] is not None
