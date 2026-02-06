from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from oghma.config import load_config
from oghma.extractor import Extractor
from oghma.embedder import EmbedConfig, create_embedder
from oghma.storage import MemoryRecord, Storage


@asynccontextmanager
async def lifespan(_: FastMCP):
    config = load_config()
    storage = Storage(config=config, read_only=False)
    try:
        embed_config_data = config.get("embedding", {})
        embedder = create_embedder(EmbedConfig.from_dict(embed_config_data))
    except (ValueError, Exception):
        embedder = None
    yield {"storage": storage, "config": config, "embedder": embedder}


mcp = FastMCP("Oghma Memory", lifespan=lifespan)


def _get_lifespan_context() -> dict[str, Any]:
    ctx = mcp.get_context()
    return ctx.request_context.lifespan_context


def _get_storage() -> Storage:
    return _get_lifespan_context()["storage"]


def _get_config() -> dict[str, Any]:
    return _get_lifespan_context().get("config", {})


def _get_embedder():
    return _get_lifespan_context().get("embedder")


@mcp.tool()
def oghma_search(
    query: str,
    category: str | None = None,
    source_tool: str | None = None,
    limit: int = 10,
    search_mode: str = "keyword",
) -> list[MemoryRecord]:
    """Search memories by keyword, vector, or hybrid mode."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if search_mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError("search_mode must be one of: keyword, vector, hybrid")

    storage = _get_storage()
    if search_mode == "keyword":
        return storage.search_memories(
            query=query,
            category=category,
            source_tool=source_tool,
            limit=limit,
        )

    query_embedding: list[float] | None = None
    try:
        embed_config = _get_config().get("embedding", {})
        embedder = create_embedder(EmbedConfig.from_dict(embed_config))
        query_embedding = embedder.embed(query)
    except Exception:
        if search_mode == "vector":
            return []

    return storage.search_memories_hybrid(
        query=query,
        query_embedding=query_embedding,
        category=category,
        source_tool=source_tool,
        limit=limit,
        search_mode=search_mode,
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
def oghma_add(
    content: str,
    category: str,
    source_tool: str = "manual",
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Add a memory directly. Categories follow extraction.categories in config."""
    valid_categories = (
        _get_config().get("extraction", {}).get("categories", Extractor.CATEGORIES)
        or Extractor.CATEGORIES
    )
    if category not in valid_categories:
        raise ValueError(f"category must be one of: {valid_categories}")

    storage = _get_storage()
    memory_id = storage.add_memory(
        content=content,
        category=category,
        source_tool=source_tool,
        source_file="mcp_direct",
        confidence=confidence,
    )

    embedder = _get_embedder()
    if embedder and memory_id:
        try:
            vector = embedder.embed(content)
            storage.upsert_memory_embedding(memory_id, vector)
        except Exception:
            pass

    return {"id": memory_id, "status": "created", "content": content, "category": category}


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
