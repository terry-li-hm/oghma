# Oghma

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Persistent memory for AI coding assistants.

Every time you close a session with Claude Code, Codex, or OpenCode, the discoveries you made disappear. That workaround you found for sqlite-vec? Gone. The API quirk that cost you an hour? Forgotten. Next week, you'll hit the same wall again.

Oghma watches your transcript files, extracts the non-obvious learnings via LLM, deduplicates against what it already knows, and makes everything searchable — keyword, semantic, or hybrid.

## How it works

```
┌─────────────┐     ┌───────────┐     ┌──────────┐     ┌────────────┐
│ Transcripts │────▶│  Extract  │────▶│  Dedup   │────▶│   Store    │
│ (JSONL)     │     │  (LLM)    │     │ (cosine) │     │ (SQLite)   │
└─────────────┘     └───────────┘     └──────────┘     └────────────┘
  Claude Code            │                                    │
  Codex                  │                                    ▼
  OpenCode          Categories:                    ┌──────────────────┐
                    - gotcha                       │  Search (MCP)    │
                    - learning                     │  keyword / vec / │
                    - workflow                     │  hybrid (RRF)    │
                    - preference                   └──────────────────┘
                    - project_context
```

A background daemon polls for new/changed transcripts, sends chunks to an LLM for extraction, embeds the results, checks for semantic duplicates, and stores what's genuinely new. Your AI assistant queries this via MCP — so it remembers what you've learned across every session and every tool.

## Features

- **Multi-tool extraction** — Parses transcripts from Claude Code, Codex, OpenCode (and OpenClaw)
- **LLM-powered filtering** — Configurable model with a tuned prompt that extracts gotchas and workarounds while filtering noise like "the user prefers Python"
- **Hybrid search** — SQLite FTS5 + [sqlite-vec](https://github.com/asg017/sqlite-vec) fused via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) with recency boost
- **Inline dedup** — New memories are checked against existing embeddings before insertion. Duplicates never enter the DB.
- **[MCP](https://modelcontextprotocol.io/) server** — Plug into Claude Code, Cursor, or any MCP-compatible client
- **Maintenance CLI** — Semantic dedup, noise purge, staleness pruning, memory promotion
- **Export** — Markdown or JSON, grouped by category, date, or source

## Quick start

```bash
# Install
pip install oghma

# Or from source
git clone https://github.com/terry-li-hm/oghma.git
cd oghma
pip install -e ".[dedup]"

# Set API keys
export OPENAI_API_KEY=sk-...          # for embeddings
export OPENROUTER_API_KEY=sk-or-...   # if using OpenRouter models for extraction

# Initialize and start
oghma init          # creates ~/.oghma/config.yaml
oghma start         # background daemon
```

Edit `~/.oghma/config.yaml` to configure your extraction model, tool paths, and embedding settings.

## MCP server

Add to your Claude Code config (`~/.claude.json`):

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

This exposes five tools to your AI assistant:

| Tool | Description |
|------|-------------|
| `oghma_search` | Search memories (keyword, vector, or hybrid) |
| `oghma_get` | Fetch a specific memory by ID |
| `oghma_add` | Add a memory directly (with auto-dedup) |
| `oghma_stats` | Memory counts by category and source |
| `oghma_categories` | List categories with counts |

## CLI reference

```
oghma init                  Create default config
oghma start [--foreground]  Start the extraction daemon
oghma stop                  Stop the daemon
oghma status [--json]       Daemon status and DB stats
oghma stats                 Memory counts by category/source

oghma search <query>        Search memories
  --mode keyword|vector|hybrid
  --category, --tool, --status, --limit

oghma dedup                 Find and remove semantic duplicates
oghma purge-noise           Remove memories matching noise patterns
oghma prune-stale           Delete memories older than N days
  --max-age-days 90
  --source-tool <name>

oghma promote <id>          Promote a memory to 'promoted' category
oghma export                Export to markdown or JSON
  --format, --group-by, --category

oghma validate-config       Check config for errors
oghma migrate-embeddings    Backfill embeddings for existing memories
```

All destructive commands default to `--dry-run`. Pass `--execute` to apply.

## Configuration

`~/.oghma/config.yaml`:

```yaml
daemon:
  poll_interval: 300          # seconds between checks
  min_messages: 6             # skip trivial sessions

extraction:
  model: google/gemini-3-flash-preview   # or gpt-4o-mini, deepseek/deepseek-chat, etc.
  confidence_threshold: 0.7
  dedup_threshold: 0.92       # cosine similarity — higher = stricter
  categories:
    - learning
    - preference
    - project_context
    - gotcha
    - workflow
    - promoted

embedding:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536

tools:
  claude_code:
    enabled: true
    paths:
      - ~/.claude/projects/-Users-*/*.jsonl
  codex:
    enabled: true
    paths:
      - ~/.codex/sessions/**/rollout-*.jsonl
  opencode:
    enabled: true
    paths:
      - ~/.local/share/opencode/storage/message/ses_*
```

### Extraction models

Oghma supports any OpenAI or OpenRouter model:

| Model | Provider | Quality | Cost |
|-------|----------|---------|------|
| google/gemini-3-flash-preview | OpenRouter | Excellent | ~$1.50/M tokens |
| gpt-4o-mini | OpenAI | Good | ~$0.30/M tokens |
| deepseek/deepseek-chat-v3-0324 | OpenRouter | Good | ~$0.14/M tokens |

## Search modes

| Mode | Engine | Best for |
|------|--------|----------|
| **keyword** | SQLite FTS5 | Exact term matching, fast |
| **vector** | sqlite-vec (cosine similarity) | Conceptual/semantic search |
| **hybrid** | RRF fusion of both + recency boost | Best overall relevance |

```bash
oghma search "async patterns" --mode hybrid --limit 20
```

## How extraction works

The daemon sends conversation chunks to an LLM with a prompt engineered to extract only actionable insights:

**Extracted:** Tool gotchas, bug workarounds, API quirks, architecture decisions, error solutions, workflow patterns.

**Filtered:** Setup facts ("uses Python 3.12"), config restatements, assistant narration ("The AI suggested..."), trivially obvious observations.

Each memory gets a confidence score and a category. Post-extraction, regex noise patterns catch stragglers. Pre-insertion, embedding similarity catches duplicates. The result: your database grows with genuine insights, not noise.

## Maintenance

```bash
# Recommended: run weekly via cron
oghma dedup --threshold 0.92 --execute
oghma purge-noise --execute

# Prune old memories from a retired tool
oghma prune-stale --max-age-days 90 --source-tool openclaw --execute

# Promote a frequently-useful memory
oghma promote 739
```

## Adding a custom parser

Implement a parser with `can_parse()` and `parse()` methods:

```python
from oghma.parsers import Message

class MyToolParser:
    def can_parse(self, file_path: Path) -> bool:
        return ".mytool" in str(file_path)

    def parse(self, file_path: Path) -> list[Message]:
        # Return list of Message(role="user"|"assistant", content="...")
        ...
```

Register in `src/oghma/parsers/__init__.py` and add glob patterns to your config.

## Requirements

- Python 3.10+
- SQLite with FTS5 (included in most distributions)
- [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search (optional, recommended)
- OpenAI API key for embeddings
- LLM API key for extraction (OpenAI or OpenRouter)

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | For embeddings (text-embedding-3-small) |
| `OPENROUTER_API_KEY` | If using OpenRouter | For Gemini, DeepSeek, etc. |
| `OGHMA_DB_PATH` | No | Override database path |
| `OGHMA_EXTRACTION_MODEL` | No | Override extraction model |
| `OGHMA_LOG_LEVEL` | No | DEBUG / INFO / WARNING / ERROR |

## License

MIT
