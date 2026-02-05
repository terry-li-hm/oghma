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

### Model Selection

Oghma supports both OpenAI and OpenRouter models:

| Model | Provider | Quality | Cost | Notes |
|-------|----------|---------|------|-------|
| gpt-4o-mini | OpenAI | Good | ~$0.30/M | Default, factual |
| google/gemini-3-flash-preview | OpenRouter | Excellent | ~$1.50/M | Best quality/cost |
| google/gemini-2.0-flash-001 | OpenRouter | Good | ~$0.25/M | Budget option |
| deepseek/deepseek-chat-v3-0324 | OpenRouter | Good | ~$0.14/M | Cheapest |

To use OpenRouter models, set in config.yaml:
```yaml
extraction:
  model: google/gemini-3-flash-preview
```

Or via environment: `OGHMA_EXTRACTION_MODEL=google/gemini-3-flash-preview`

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
- OGHMA_EXTRACTION_MODEL: Override extraction model
- OPENAI_API_KEY: Required for OpenAI models (gpt-4o-mini)
- OPENROUTER_API_KEY: Required for OpenRouter models (Gemini, DeepSeek, etc.)

## License

MIT
