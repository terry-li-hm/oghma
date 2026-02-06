# Oghma

Unified AI memory layer for coding assistants. Watches transcripts from Claude Code, Codex, OpenClaw, and OpenCode. Extracts memories via LLM. Searches with FTS5, vector similarity, and hybrid (RRF) retrieval.

## Features

- **Multi-tool support** — Parses transcripts from Claude Code, Codex, OpenClaw, and OpenCode
- **LLM extraction** — Uses GPT-4o-mini, Gemini Flash, or any OpenRouter model to extract structured memories
- **Hybrid search** — Keyword (FTS5), vector (sqlite-vec), and hybrid (RRF fusion) with recency boost
- **Inline embedding** — Memories are embedded immediately on extraction, no separate migration step
- **Cross-source dedup** — Same insight from different tools is stored once
- **Smart filtering** — Skips trivial sessions (cron heartbeats, tool-only runs) to save API costs
- **MCP server** — Native integration with Claude Code and other MCP-compatible tools
- **Write API** — Add memories directly via MCP or CLI, not just from transcripts

## Install

```bash
pip install oghma
```

For local embeddings (optional):
```bash
pip install oghma[local]
```

## Quick Start

```bash
export OPENAI_API_KEY=sk-...          # for embeddings
export OPENROUTER_API_KEY=sk-or-...   # for extraction (if using OpenRouter models)

oghma init
oghma start
oghma search "python typing"
oghma export -o ./memories
```

## Configuration

Config at `~/.oghma/config.yaml`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| daemon.poll_interval | 300 | Seconds between checks for new transcripts |
| extraction.model | gpt-4o-mini | LLM for memory extraction |
| embedding.model | text-embedding-3-small | Embedding model for vector search |
| embedding.provider | openai | Embedding provider (openai or local) |

### Model Selection

Oghma supports both OpenAI and OpenRouter models for extraction:

| Model | Provider | Quality | Cost | Notes |
|-------|----------|---------|------|-------|
| gpt-4o-mini | OpenAI | Good | ~$0.30/M | Default, factual |
| google/gemini-3-flash-preview | OpenRouter | Excellent | ~$1.50/M | Best quality/cost |
| google/gemini-2.0-flash-001 | OpenRouter | Good | ~$0.25/M | Budget option |
| deepseek/deepseek-chat-v3-0324 | OpenRouter | Good | ~$0.14/M | Cheapest |

Set via config:
```yaml
extraction:
  model: google/gemini-3-flash-preview
```

Or environment variable: `OGHMA_EXTRACTION_MODEL=google/gemini-3-flash-preview`

## Commands

| Command | Description |
|---------|-------------|
| oghma init | Create default config |
| oghma status | Show daemon and database status |
| oghma stats | Show memory counts by category/source |
| oghma start | Start background daemon |
| oghma stop | Stop daemon |
| oghma search "query" | Search memories (--mode keyword/vector/hybrid) |
| oghma export | Export memories to files |
| oghma migrate-embeddings | Backfill embeddings for existing memories |

## Search Modes

| Mode | How it works | Best for |
|------|-------------|----------|
| keyword (default) | FTS5 full-text search, ordered by recency | Exact term matching |
| vector | Cosine similarity via sqlite-vec embeddings | Semantic/conceptual search |
| hybrid | RRF fusion of keyword + vector with recency boost | Best overall relevance |

```bash
oghma search "async patterns" --mode hybrid --limit 20
```

## MCP Server

Native integration with Claude Code, Codex, and other MCP-compatible tools.

Add to `~/.claude.json`:
```json
{
  "mcpServers": {
    "oghma": {
      "command": "uvx",
      "args": ["--from", "oghma", "oghma-mcp"]
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| oghma_search | Search memories (keyword, vector, or hybrid mode) |
| oghma_get | Get a memory by ID |
| oghma_stats | Database statistics (counts by category and source) |
| oghma_add | Write a memory directly with auto-embedding |
| oghma_categories | List categories with counts |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Yes (embeddings) | OpenAI API key for text-embedding-3-small |
| OPENROUTER_API_KEY | If using OpenRouter | API key for Gemini, DeepSeek, etc. |
| OGHMA_DB_PATH | No | Override database path |
| OGHMA_POLL_INTERVAL | No | Override poll interval |
| OGHMA_LOG_LEVEL | No | Set log level (DEBUG/INFO/WARNING/ERROR) |
| OGHMA_EXTRACTION_MODEL | No | Override extraction model |

## Adding Custom Parsers

Implement `BaseParser` with `can_parse()` and `parse()` methods, then register in `src/oghma/parsers/__init__.py`.

## License

MIT
