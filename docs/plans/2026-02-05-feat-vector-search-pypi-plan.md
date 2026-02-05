---
title: "feat: Oghma v0.3.0 — Semantic Vector Search + PyPI Release"
type: feat
date: 2026-02-05
---

# Oghma v0.3.0 — Semantic Vector Search + PyPI Release

## Overview

Add semantic vector search alongside FTS5 keyword search, then ship real PyPI package.

**Why:** "how to fix permissions" should find "Screen Recording fails if tmux started via SSH" — meaning matters, not just keywords.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Oghma v0.3.0                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐                                                        │
│  │  Extractor   │  (existing - memory extraction)                        │
│  └──────┬───────┘                                                        │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐     ┌──────────────┐                                   │
│  │   Embedder   │◄────│ EmbedConfig  │  (new - embedding generation)     │
│  │  (pluggable) │     │ OpenAI/Local │                                   │
│  └──────┬───────┘     └──────────────┘                                   │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────┐                │
│  │                    SQLite Database                    │                │
│  │  ┌────────────┐  ┌─────────────┐  ┌────────────────┐ │                │
│  │  │  memories  │  │ memories_fts│  │  memories_vec  │ │                │
│  │  │  (content) │  │   (FTS5)    │  │   (sqlite-vec) │ │                │
│  │  └─────┬──────┘  └──────┬──────┘  └───────┬────────┘ │                │
│  │        │                │                  │          │                │
│  │        └────────────────┴──────────────────┘          │                │
│  │                         │                             │                │
│  └─────────────────────────┼─────────────────────────────┘                │
│                            │                                              │
│                   ┌────────▼────────┐                                     │
│                   │  Hybrid Search  │                                     │
│                   │  (RRF Fusion)   │                                     │
│                   └────────┬────────┘                                     │
│                            │                                              │
│         ┌──────────────────┼──────────────────┐                           │
│         │                  │                  │                           │
│    ┌────▼────┐       ┌─────▼─────┐      ┌─────▼─────┐                    │
│    │   CLI   │       │    MCP    │      │  Python   │                    │
│    │ search  │       │  Server   │      │    API    │                    │
│    └─────────┘       └───────────┘      └───────────┘                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Embedder Infrastructure

**New file:** `src/oghma/embedder.py`

- `EmbedConfig` dataclass (provider, model, dimensions)
- `Embedder` ABC with `embed()` and `embed_batch()`
- `OpenAIEmbedder` using existing OpenAI client pattern
- `LocalEmbedder` using sentence-transformers (optional)

**Recommendation:** OpenAI `text-embedding-3-small`
- Cost: ~$0.02/1M tokens (4,498 memories = ~$0.09 one-time)
- Quality: Top-tier on MTEB benchmarks
- Simplicity: Same API key as extraction

### Phase 2: Vector Storage with sqlite-vec

**Schema additions:**
```sql
-- Vector table for embeddings
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    memory_id INTEGER PRIMARY KEY,
    embedding float[1536]
);

-- Track embedding state
ALTER TABLE memories ADD COLUMN has_embedding INTEGER DEFAULT 0;
```

**Storage changes:**
- `_init_db()` — Load sqlite-vec extension, create vec0 table
- `add_memory()` — Accept optional `embedding`, insert into `memories_vec`

### Phase 3: Hybrid Search (RRF Fusion)

**New method:** `search_memories_hybrid()`

```python
def search_memories_hybrid(
    self,
    query: str,
    query_embedding: list[float] | None = None,
    category: str | None = None,
    source_tool: str | None = None,
    limit: int = 10,
    search_mode: str = "hybrid",  # "keyword" | "vector" | "hybrid"
    rrf_k: int = 60,
) -> list[MemoryRecord]:
    """
    Hybrid search using Reciprocal Rank Fusion.
    RRF: score = (1/(k + fts_rank)) * 0.5 + (1/(k + vec_rank)) * 0.5
    """
```

**Fallback behavior:**
- No embedding available → FTS5-only
- Query too short → keyword search
- Configurable via `search_mode`

### Phase 4: Migration for Existing Memories

**New CLI command:**
```bash
oghma migrate-embeddings [--batch-size 100] [--dry-run]
```

**Cost estimate:** ~$0.018 for 4,498 memories (negligible)

**Strategy:**
1. Query memories without embeddings
2. Batch process with rate limiting
3. Insert into `memories_vec`
4. Update `has_embedding = 1`

### Phase 5: Update MCP Server and CLI

**MCP:** Add `search_mode` parameter to `oghma_search`
**CLI:** Add `--mode` flag to search command

### Phase 6: PyPI Release

**Checklist:**
1. Update version to `0.3.0`
2. Run tests: `pytest --cov=oghma`
3. Build: `uv build`
4. Test on TestPyPI: `uv publish --index testpypi`
5. Verify: `uvx --index testpypi oghma --version`
6. Publish: `uv publish`
7. Tag: `git tag v0.3.0 && git push --tags`

## Dependencies to Add

```toml
# Required
"sqlite-vec>=0.1.0"

# Optional (for local embeddings)
[project.optional-dependencies]
local = ["sentence-transformers>=2.0"]
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/oghma/embedder.py` | **New** — Pluggable embedding |
| `src/oghma/migration.py` | **New** — Embedding migration |
| `src/oghma/storage.py` | Modify — Add vec table, hybrid search |
| `src/oghma/mcp_server.py` | Modify — Add search_mode |
| `src/oghma/cli.py` | Modify — Add --mode, migrate-embeddings |
| `pyproject.toml` | Modify — Add sqlite-vec, bump version |
| `.github/workflows/publish.yml` | **New** — PyPI publishing |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| sqlite-vec not available | Check at startup, fall back to FTS5-only |
| OpenAI API failures | Retry with backoff, skip embedding if fails |
| Large migration | Batch processing with checkpoints |
| Breaking change | Keep `search_memories()` unchanged, add new method |

## Success Criteria

- [ ] `oghma search "how to fix permissions"` finds tmux/Screen Recording memory
- [ ] `pip install oghma` works from PyPI
- [ ] `uvx oghma --version` shows 0.3.0
- [ ] Migration completes for 4,498 memories
- [ ] MCP server supports hybrid search

## References

- [Hybrid search with SQLite](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html) — RRF implementation
- [sqlite-vec GitHub](https://github.com/asg017/sqlite-vec) — Vector storage
- [Publishing with uv](https://docs.astral.sh/uv/guides/package/) — PyPI workflow
