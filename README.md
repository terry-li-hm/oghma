# Oghma

Unified AI memory layer. Watches transcripts from Claude Code, Codex, OpenClaw, and OpenCode. Extracts memories via LLM. Search with FTS5.

## Install

```bash
pip install oghma
```

## Quick Start

```bash
oghma init
oghma start
oghma search "python typing"
oghma export -o ./memories
```

## Configuration

Config at ~/.oghma/config.yaml. Key settings:
- tools: Enable/disable tool watching
- daemon.poll_interval: How often to check for changes (default 300s)
- extraction.model: LLM for memory extraction (default gpt-4o-mini)

## Commands

| Command | Description |
|---------|-------------|
| oghma init | Create default config |
| oghma status | Show daemon and database status |
| oghma start | Start background daemon |
| oghma stop | Stop daemon |
| oghma search "query" | Search memories |
| oghma export | Export memories to files |

## Environment Variables

- OGHMA_DB_PATH: Override database path
- OGHMA_POLL_INTERVAL: Override poll interval
- OGHMA_LOG_LEVEL: Set log level (DEBUG/INFO/WARNING/ERROR)
- OPENAI_API_KEY: Required for memory extraction

## License

MIT
