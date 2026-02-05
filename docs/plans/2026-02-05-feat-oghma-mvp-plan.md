---
title: "feat: Oghma MVP — Unified AI Memory Layer"
type: feat
date: 2026-02-05
brainstorm: ../brainstorms/2026-02-05-oghma-architecture-brainstorm.md
---

# Oghma MVP — Unified AI Memory Layer

## Overview

Build the MVP of Oghma: a daemon that passively watches AI coding tool transcripts, extracts memories via LLM, stores them in SQLite, and provides CLI search + markdown export.

**Philosophy:** AI memory should be user-owned and tool-agnostic, not locked into provider silos.

**Differentiator:** LLM-based memory extraction (not just transcript search like `cass`).

## Problem Statement

Developers using multiple AI coding tools lose context when switching between them. Each tool has its own memory silo. Oghma aggregates memories across tools so any tool can access the user's full context.

## Proposed Solution

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Oghma                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Claude   │   │  Codex   │   │ OpenClaw │   │ OpenCode │ │
│  │ Parser   │   │  Parser  │   │  Parser  │   │  Parser  │ │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘ │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │  Watcher  │  (polls every 5min)        │
│                    └─────┬─────┘                            │
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │ Extractor │  (GPT-4o-mini)             │
│                    └─────┬─────┘                            │
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │  SQLite   │  (~/.oghma/oghma.db)       │
│                    └─────┬─────┘                            │
│                          │                                   │
│           ┌──────────────┼──────────────┐                   │
│           │              │              │                   │
│     ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐           │
│     │    CLI    │  │  Export   │  │    MCP    │ (v2)      │
│     │  search   │  │ markdown  │  │  server   │           │
│     └───────────┘  └───────────┘  └───────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Technical Approach

**Storage: SQLite**
- Single file at `~/.oghma/oghma.db`
- Zero deps, embedded, portable
- FTS5 for keyword search (vector search deferred to v2)

**Extraction: GPT-4o-mini via OpenAI API**
- User provides `OPENAI_API_KEY` in environment
- ~$1/month cost for typical usage
- Extracts: learnings, preferences, project context, gotchas

**File Watching: Polling (not fsnotify)**
- Poll every 5 minutes (configurable)
- Track file mtime to detect changes
- Avoids fsnotify FD exhaustion issues (see learnings)

**CLI: Click-based**
- `oghma start` — start daemon (foreground by default)
- `oghma stop` — stop daemon
- `oghma status` — show daemon state, memory count, recent extractions
- `oghma search "query"` — search memories
- `oghma export` — export to markdown

## Technical Considerations

### Institutional Learnings Applied

1. **File Descriptor Exhaustion** — Use polling, not fsnotify. Validate watched paths. Block `.venv`, `node_modules`, etc.

2. **SQLite Schema** — Include source metadata, timestamps, status fields. Use JSON column for flexible extraction metadata.

3. **CLI Safety** — Require explicit paths (no root searches). Add `--dry-run` for destructive operations.

4. **Config Robustness** — YAML config file. Environment overrides. Secrets in env vars only.

5. **Change Detection** — Track file mtime before extracting. Skip unchanged files.

### Transcript Locations

| Tool | Location | Format |
|------|----------|--------|
| Claude Code | `~/.claude/projects/-Users-*/*.jsonl` | `type: "user"/"assistant"` |
| Codex | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` | `type: "item"` |
| OpenClaw | `~/.openclaw/agents/main/sessions/*.jsonl` | `type: "message"` |
| OpenCode | `~/.local/share/opencode/storage/message/ses_*/` | Split message + parts |
| Cursor | `~/.cursor/workspaceStorage/*/state.vscdb` | SQLite, needs extraction |

**Note:** Cursor uses SQLite internally — requires different parsing strategy.

## Acceptance Criteria

### Setup & Configuration

- [ ] `pip install oghma` installs cleanly
- [ ] First run creates `~/.oghma/` directory with default config
- [ ] Config file at `~/.oghma/config.yaml` with documented options
- [ ] Validates `OPENAI_API_KEY` present before extraction
- [ ] `oghma init` creates config with prompts for customization

### Daemon Lifecycle

- [ ] `oghma start` runs daemon in foreground (Ctrl+C to stop)
- [ ] `oghma start --daemon` runs detached (writes PID to `~/.oghma/oghma.pid`)
- [ ] `oghma stop` stops running daemon gracefully
- [ ] `oghma status` shows: running/stopped, memory count, last extraction time, any errors
- [ ] Daemon logs to `~/.oghma/oghma.log` (configurable level)

### Transcript Watching

- [ ] Discovers transcripts from all 5 tools automatically
- [ ] Polls every 5 minutes (configurable via `poll_interval`)
- [ ] Skips files unchanged since last check (mtime-based)
- [ ] Skips sessions with <6 messages (configurable threshold)
- [ ] Handles corrupt/partial JSONL gracefully (skip bad lines, log warning)
- [ ] Validates watched paths, blocks `.venv`, `node_modules`, `.git`

### Memory Extraction

- [ ] Extracts memories via GPT-4o-mini API call
- [ ] Categories: `learning`, `preference`, `project_context`, `gotcha`, `workflow`
- [ ] Stores: content, category, source_tool, source_file, timestamp, confidence
- [ ] Handles API errors gracefully (retry with backoff, then skip)
- [ ] Truncates large transcripts to fit token limits (4000 chars default)

### Search

- [ ] `oghma search "query"` returns matching memories
- [ ] FTS5-based full-text search
- [ ] Output shows: content snippet, category, source tool, timestamp
- [ ] `--limit N` controls result count (default 10)
- [ ] `--tool claude` filters by source tool
- [ ] `--category learning` filters by category

### Export

- [ ] `oghma export` writes memories to markdown
- [ ] Default location: `~/.oghma/export/`
- [ ] One file per category: `learnings.md`, `preferences.md`, etc.
- [ ] `--output-dir` overrides default location
- [ ] `--format` supports `markdown` (default) and `json`
- [ ] Incremental: only exports new memories since last export

### Quality Gates

- [ ] 80% test coverage on core modules (parsers, extractor, storage)
- [ ] All CLI commands have help text and examples
- [ ] README with quick start, configuration reference, troubleshooting
- [ ] Type hints on all public functions
- [ ] Passes `ruff check` and `ruff format`

## Database Schema

```sql
-- ~/.oghma/oghma.db

CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT NOT NULL,  -- learning, preference, project_context, gotcha, workflow
    source_tool TEXT NOT NULL,  -- claude_code, codex, openclaw, opencode, cursor
    source_file TEXT NOT NULL,
    source_session TEXT,  -- session ID if available
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',  -- active, archived, deleted
    metadata JSON  -- flexible field for extraction details
);

CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_source_tool ON memories(source_tool);
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX idx_memories_status ON memories(status);

-- Full-text search
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    category,
    source_tool,
    content=memories,
    content_rowid=id
);

-- Track extraction state per file
CREATE TABLE extraction_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT UNIQUE NOT NULL,
    last_mtime REAL NOT NULL,
    last_size INTEGER NOT NULL,
    last_extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0
);

CREATE INDEX idx_extraction_state_path ON extraction_state(source_path);

-- Audit log
CREATE TABLE extraction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    memories_extracted INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration Schema

```yaml
# ~/.oghma/config.yaml

# Storage
storage:
  db_path: ~/.oghma/oghma.db  # SQLite database location
  backup_enabled: true
  backup_dir: ~/.oghma/backups
  backup_retention_days: 30

# Daemon
daemon:
  poll_interval: 300  # seconds (5 minutes)
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR
  log_file: ~/.oghma/oghma.log
  pid_file: ~/.oghma/oghma.pid
  min_messages: 6  # minimum messages before processing

# Extraction
extraction:
  model: gpt-4o-mini
  max_content_chars: 4000  # truncate large transcripts
  categories:
    - learning
    - preference
    - project_context
    - gotcha
    - workflow
  confidence_threshold: 0.5  # minimum confidence to store

# Export
export:
  output_dir: ~/.oghma/export
  format: markdown  # markdown or json

# Tool-specific overrides (optional)
tools:
  claude_code:
    enabled: true
    paths:
      - ~/.claude/projects/-Users-*/*.jsonl
  codex:
    enabled: true
    paths:
      - ~/.codex/sessions/**/rollout-*.jsonl
  openclaw:
    enabled: true
    paths:
      - ~/.openclaw/agents/*/sessions/*.jsonl
  opencode:
    enabled: true
    paths:
      - ~/.local/share/opencode/storage/message/ses_*
  cursor:
    enabled: false  # TBD - needs research
    paths: []
```

## Project Structure

```
oghma/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/oghma/
│   ├── __init__.py
│   ├── cli.py              # Click CLI commands
│   ├── daemon.py           # Main daemon loop
│   ├── config.py           # Config loading/validation
│   ├── storage.py          # SQLite operations
│   ├── extractor.py        # LLM extraction logic
│   ├── watcher.py          # File discovery + change detection
│   ├── exporter.py         # Markdown/JSON export
│   └── parsers/
│       ├── __init__.py
│       ├── base.py         # Base parser interface
│       ├── claude_code.py
│       ├── codex.py
│       ├── openclaw.py
│       ├── opencode.py
│       └── cursor.py       # TBD
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_daemon.py
│   ├── test_storage.py
│   ├── test_extractor.py
│   ├── test_watcher.py
│   └── test_parsers/
│       ├── test_claude_code.py
│       ├── test_codex.py
│       └── ...
└── docs/
    ├── brainstorms/
    └── plans/
```

## Implementation Phases

### Phase 1: Foundation (Core Infrastructure)

**Goal:** Project skeleton, config, storage, basic CLI

- [ ] Initialize project with `pyproject.toml` (hatch build system)
- [ ] Create `src/oghma/__init__.py` with version
- [ ] Implement `config.py` — load/validate YAML config, env overrides
- [ ] Implement `storage.py` — SQLite init, CRUD operations, FTS5 setup
- [ ] Implement basic CLI skeleton with Click (`cli.py`)
- [ ] Add `oghma init` command — creates ~/.oghma/ and default config
- [ ] Add `oghma status` command — shows config location, db size
- [ ] Write tests for config and storage modules
- [ ] Setup CI with ruff + pytest

**Files:**
- `pyproject.toml`
- `src/oghma/__init__.py`
- `src/oghma/config.py`
- `src/oghma/storage.py`
- `src/oghma/cli.py`
- `tests/test_config.py`
- `tests/test_storage.py`

### Phase 2: Parsers (Transcript Parsing)

**Goal:** Parse transcripts from all 4 initial tools

- [ ] Define base parser interface (`parsers/base.py`)
- [ ] Implement Claude Code parser — JSONL with `type: "user"/"assistant"`
- [ ] Implement Codex parser — JSONL with `type: "item"`
- [ ] Implement OpenClaw parser — JSONL with `type: "message"`
- [ ] Implement OpenCode parser — split message/part JSON files
- [ ] Write comprehensive parser tests with fixture files
- [ ] Handle edge cases: corrupt lines, encoding issues, empty files

**Files:**
- `src/oghma/parsers/base.py`
- `src/oghma/parsers/claude_code.py`
- `src/oghma/parsers/codex.py`
- `src/oghma/parsers/openclaw.py`
- `src/oghma/parsers/opencode.py`
- `tests/test_parsers/*.py`
- `tests/fixtures/` — sample transcript files

### Phase 3: Watcher & Extractor

**Goal:** File watching + LLM extraction

- [ ] Implement `watcher.py` — discover files, track mtime, detect changes
- [ ] Implement `extractor.py` — call GPT-4o-mini, parse response, handle errors
- [ ] Define extraction prompt (few-shot with category examples)
- [ ] Add retry logic with exponential backoff
- [ ] Add token counting and truncation
- [ ] Write tests with mocked OpenAI API
- [ ] Integration test: file → parse → extract → store

**Files:**
- `src/oghma/watcher.py`
- `src/oghma/extractor.py`
- `tests/test_watcher.py`
- `tests/test_extractor.py`

### Phase 4: Daemon & Search

**Goal:** Running daemon + search CLI

- [ ] Implement `daemon.py` — main loop, graceful shutdown, signal handling
- [ ] Add `oghma start` command (foreground mode)
- [ ] Add `oghma start --daemon` (detached mode with PID file)
- [ ] Add `oghma stop` command
- [ ] Enhance `oghma status` — show memory count, last extraction, errors
- [ ] Add `oghma search "query"` — FTS5 search with formatting
- [ ] Add search filters: `--tool`, `--category`, `--limit`, `--since`
- [ ] Write daemon tests (process lifecycle)

**Files:**
- `src/oghma/daemon.py`
- `src/oghma/cli.py` (extend)
- `tests/test_daemon.py`
- `tests/test_cli.py` (extend)

### Phase 5: Export & Polish

**Goal:** Markdown export, documentation, release

- [ ] Implement `exporter.py` — markdown and JSON export
- [ ] Add `oghma export` command with options
- [ ] Write comprehensive README (quick start, config reference, troubleshooting)
- [ ] Add `--help` text to all CLI commands
- [ ] Final test pass — aim for 80% coverage
- [ ] Tag v0.1.0 release
- [ ] Update PyPI package from placeholder

**Files:**
- `src/oghma/exporter.py`
- `README.md`
- `tests/test_exporter.py`

## Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "click>=8.0",        # CLI framework
    "pyyaml>=6.0",       # Config parsing
    "openai>=1.0",       # LLM API
    "rich>=13.0",        # Pretty CLI output
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "respx>=0.20",       # Mock HTTP for OpenAI
]
```

## Open Questions (Deferred)

1. **Cursor support** — Need to research SQLite schema in `state.vscdb`
2. **Vector search** — Add `sqlite-vec` in v2 for semantic search
3. **MCP server** — Add in v2 for Claude Code/Cursor native integration
4. **Deduplication** — Same memory across sessions — hash-based or semantic?
5. **Local LLM** — Ollama support for offline extraction

## Success Metrics

- [ ] `pip install oghma && oghma init && oghma start` works in <2 minutes
- [ ] Extracts memories from real transcripts without errors
- [ ] Search returns relevant results for test queries
- [ ] Documentation allows new user to get started without support

## References

### Internal References

- Brainstorm: `docs/brainstorms/2026-02-05-oghma-architecture-brainstorm.md`
- Research note: `~/notes/AI Memory Aggregator Research.md`
- Existing memU daemon: `~/agent-config/scripts/memu/memu_proactive_daemon.py`

### Institutional Learnings Applied

- File watcher FD exhaustion: `~/docs/solutions/ai-tooling/openclaw-emfile-skill-watcher-crash.md`
- Multi-tool persistence: `~/docs/solutions/integration-issues/multi-tool-history-support-HistorySkill-20260126.md`
- CLI safety patterns: `~/docs/solutions/performance-issues/slow-root-search-CLITools-20260126.md`
- Config robustness: `~/docs/solutions/integration-issues/lfg-namespace-and-sync-robustness.md`

### External References

- Click documentation: https://click.palletsprojects.com/
- OpenAI API: https://platform.openai.com/docs/
- SQLite FTS5: https://www.sqlite.org/fts5.html
