from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from oghma.config import load_config
from oghma.storage import MemoryRecord, Storage


@asynccontextmanager
async def lifespan(_: FastMCP):
    config = load_config()
    storage = Storage(config=config, read_only=True)
    yield {"storage": storage}


mcp = FastMCP("Oghma Memory", lifespan=lifespan)


def _get_storage() -> Storage:
    return mcp.get_context()["storage"]


@mcp.tool()
def oghma_search(
    query: str,
    category: str | None = None,
    source_tool: str | None = None,
    limit: int = 10,
) -> list[MemoryRecord]:
    """Search memories by keyword."""
    if limit < 1:
        raise ValueError("limit must be >= 1")

    storage = _get_storage()
    return storage.search_memories(
        query=query,
        category=category,
        source_tool=source_tool,
        limit=limit,
    )


@mcp.tool()
def oghma_get(memory_id: int) -> MemoryRecord | None:
    """Get a memory by ID."""
    storage = _get_storage()
    return storage.get_memory_by_id(memory_id)


@mcp.tool()
def oghma_stats() -> dict[str, Any]:
    """Get memory database statistics."""
    storage = _get_storage()
    memories = storage.get_all_memories(status="active")
    extraction_logs = storage.get_recent_extraction_logs(limit=1)

    return {
        "total_memories": storage.get_memory_count(),
        "memories_by_category": dict(Counter(memory["category"] for memory in memories)),
        "memories_by_source": dict(Counter(memory["source_tool"] for memory in memories)),
        "last_extraction_time": extraction_logs[0]["created_at"] if extraction_logs else None,
    }


@mcp.tool()
def oghma_categories() -> list[dict[str, Any]]:
    """List categories with memory counts."""
    storage = _get_storage()
    memories = storage.get_all_memories(status="active")
    category_counts = Counter(memory["category"] for memory in memories)

    return [
        {"category": category, "count": count}
        for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
