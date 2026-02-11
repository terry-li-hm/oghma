---
name: oghma
description: Search memories extracted from AI coding sessions. "search memories", "what did we learn about"
user_invocable: true
---

# Oghma Memory Search

Search memories extracted from your AI coding transcripts (Claude Code, Codex, OpenCode) via CLI. Zero MCP token overhead.

## Usage

```bash
# Hybrid search (best quality — combines keyword + semantic + recency)
oghma search "sqlite-vec gotcha" --mode hybrid --limit 5

# Filter by category
oghma search "async patterns" --category gotcha --mode hybrid --limit 5

# Filter by source tool
oghma search "deployment" --tool claude_code --mode hybrid --limit 5

# Quick keyword search (fastest)
oghma search "rate limit" --limit 10
```

## Search Modes

| Mode | Engine | Best for |
|------|--------|----------|
| `keyword` (default) | SQLite FTS5 | Exact term matching, fast |
| `vector` | sqlite-vec cosine similarity | Conceptual/semantic search |
| `hybrid` | RRF fusion of both + recency boost | Best overall relevance |

## Categories

| Category | What it contains |
|----------|------------------|
| `learning` | Technical insights, how things work |
| `gotcha` | Pitfalls, bugs, things that don't work as expected |
| `workflow` | Processes, commands, how to do things |
| `preference` | User preferences, style choices |
| `project_context` | Project-specific facts, people, dates |

## When to Search

- Before starting work on a topic — check for past learnings
- When the user asks "what did we learn about X?" or "didn't we hit this before?"
- When debugging something that feels familiar
- Prefer `--mode hybrid` for best results

## Setup

Requires `oghma` to be installed and the daemon running:

```bash
pip install oghma
oghma init
oghma start
```

Copy this file to your Claude Code skills directory:
```bash
mkdir -p ~/.claude/skills/oghma
cp integrations/claude-code/SKILL.md ~/.claude/skills/oghma/SKILL.md
```
