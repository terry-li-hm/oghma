# Oghma Architecture Brainstorm

*Date: 2026-02-05*

## What We're Building

**Oghma** — a unified, user-owned AI memory layer that aggregates context from multiple AI coding tools.

**Philosophy:** AI memory should be user-owned and tool-agnostic, not locked into provider silos.

**Problem:** Developers using multiple AI coding tools (Claude Code, Codex, OpenClaw, OpenCode, Cursor) lose context when switching between them. Each tool has its own memory silo. Estimated 5+ hours/week wasted re-explaining context.

## Why This Approach

### Primary Goal

All three, with priority order when tradeoffs arise:
1. **Ease of setup** — Zero deps wins adoption
2. **Open source adoption** — Easy for other multi-tool users
3. **Portfolio piece** — Demonstrates engineering skill through adoption, not complexity

### Key Differentiator

**Memory extraction, not just search.** Plain transcript search is what `cass` does (433 stars). Oghma's value is extracting structured memories via LLM — compressed, deduplicated, categorized knowledge that persists across sessions and tools.

### Competitive Landscape

| Project | Stars | Limitation |
|---------|-------|------------|
| claude-mem | 22k | Claude Code + Cursor only, has crypto token |
| cass | 433 | Search only, no extraction |
| Mem0 | — | Requires explicit writes |

**Gap:** No tool does passive multi-tool watching + LLM extraction.

## Key Decisions

### Storage: SQLite
- Single file, zero deps, embedded
- Use `sqlite-vec` for vector search OR FTS5 for keyword search
- Avoids PostgreSQL setup friction

### Extraction: LLM-based (GPT-4o-mini)
- This IS the differentiator — don't strip it out
- ~$1/month cost is acceptable
- User provides their own API key
- Optional: local LLM support later (Ollama)

### API Surface (build order)
1. **CLI first** — `oghma start`, `oghma search "..."`, `oghma status`
2. **MCP server second** — Claude Code/Cursor native integration (highest value)
3. **HTTP API third** — nice-to-have for extensibility

### Export
- **Markdown files** — universal, zero dependency
- Works with Obsidian vaults
- Skip claude-mem integration (crypto token risk)

### Tool Coverage
1. Claude Code — `~/.claude/projects/-Users-*/*.jsonl`
2. Codex — `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`
3. OpenClaw — `~/.openclaw/agents/main/sessions/*.jsonl`
4. OpenCode — `~/.local/share/opencode/storage/message/ses_*/`
5. Cursor — TBD (similar to Claude Code)

## Open Questions

1. **Cursor transcript location** — need to verify format
2. **Deduplication strategy** — same memory across sessions
3. **Category taxonomy** — what categories of memories?
4. **MCP protocol details** — what queries should it support?

## MVP Scope

1. Watch transcripts from 5 tools
2. Extract memories via LLM (GPT-4o-mini)
3. Store in SQLite
4. Search via CLI
5. Export to markdown

**Out of scope for MVP:**
- MCP server
- HTTP API
- Web UI
- Local LLM support

## Namespace Secured

| Platform | Name |
|----------|------|
| GitHub | `oghma` |
| PyPI | `oghma` |
| npm | `oghma-ai` |
| crates.io | `oghma` |
| Homebrew | `homebrew-oghma` |
| Scoop | `scoop-oghma` |

## Next Steps

Run `/workflows:plan` to create implementation plan.
