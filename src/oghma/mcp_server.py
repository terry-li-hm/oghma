from collections import Counter
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from oghma.config import load_config
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


def _format_memories(memories: list[MemoryRecord]) -> str:
    """Format memories as readable text instead of raw JSON."""
    if not memories:
        return "No memories found."
    lines = []
    for m in memories:
        category = m.get("category", "unknown")
        mem_id = m.get("id", "?")
        content = m.get("content", "")
        lines.append(f"- [{category}] (#{mem_id}) {content}")
    return "\n".join(lines)


@mcp.tool()
def oghma_search(
    query: str,
    category: str | None = None,
    source_tool: str | None = None,
    limit: int = 10,
    search_mode: str = "hybrid",
) -> str:
    """Search memories by keyword, vector, or hybrid mode."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if search_mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError("search_mode must be one of: keyword, vector, hybrid")

    storage = _get_storage()
    if search_mode == "keyword":
        results = storage.search_memories(
            query=query,
            category=category,
            source_tool=source_tool,
            limit=limit,
        )
        return _format_memories(results)

    query_embedding: list[float] | None = None
    try:
        embed_config = _get_config().get("embedding", {})
        embedder = create_embedder(EmbedConfig.from_dict(embed_config))
        query_embedding = embedder.embed(query)
    except Exception:
        if search_mode == "vector":
            return "No memories found (embedding unavailable)."

    results = storage.search_memories_hybrid(
        query=query,
        query_embedding=query_embedding,
        category=category,
        source_tool=source_tool,
        limit=limit,
        search_mode=search_mode,
    )
    return _format_memories(results)


@mcp.tool()
def oghma_get(memory_id: int) -> str:
    """Get a memory by ID."""
    storage = _get_storage()
    m = storage.get_memory_by_id(memory_id)
    if not m:
        return f"Memory #{memory_id} not found."
    confidence = m.get("confidence", 0)
    return (
        f"#{m.get('id', '?')} [{m.get('category', 'unknown')}]\n"
        f"{m.get('content', '')}\n"
        f"Source: {m.get('source_tool', '?')} | Confidence: {confidence:.0%} | {m.get('created_at', '?')}"
    )


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
