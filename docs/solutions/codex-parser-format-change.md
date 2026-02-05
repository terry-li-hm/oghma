---
title: Codex CLI Transcript Format Change
category: integration-issues
module: parsers
tags: [codex, parser, jsonl, format-change]
symptoms:
  - Codex parser returns 0 messages
  - Empty extraction from Codex transcripts
date_created: 2026-02-05
date_updated: 2026-02-05
severity: medium
---

# Codex CLI Transcript Format Change

## Problem

Oghma's Codex parser returned 0 messages from valid transcript files. Other parsers (Claude Code, OpenClaw, OpenCode) worked correctly.

## Symptoms

```python
parser = CodexParser()
messages = parser.parse(codex_transcript_path)
print(len(messages))  # 0 - expected >0
```

## Investigation

1. Checked file exists and has content ✓
2. Checked parser `can_parse()` returns True ✓
3. Inspected actual JSONL content:

```bash
cat ~/.codex/sessions/2026/01/27/rollout-*.jsonl | python -c "
import json, sys
from collections import Counter
types = Counter()
for line in sys.stdin:
    d = json.loads(line)
    types[d.get('type', 'unknown')] += 1
print(types)
"
# Output: Counter({'response_item': 87, 'event_msg': 65, 'turn_context': 20, 'session_meta': 1})
```

**Discovery:** Parser expected `type: "item"` but Codex CLI now uses `type: "response_item"` and `type: "event_msg"`.

## Root Cause

Codex CLI (v0.91.0+) changed its transcript format:

| Field | Old Format | New Format |
|-------|------------|------------|
| Message type | `type: "item"` | `type: "response_item"` or `type: "event_msg"` |
| Role location | `payload.item.role` | `payload.role` |
| Content location | `payload.item.content` | `payload.content` |

## Solution

Updated `_extract_role()` and `_extract_content()` to handle both formats:

```python
def _extract_role(self, data: dict) -> str | None:
    msg_type = data.get("type")
    # Support both old format (item) and new format (response_item, event_msg)
    if msg_type not in ("item", "response_item", "event_msg"):
        return None

    payload = data.get("payload", {})

    # New format: role directly in payload
    if "role" in payload:
        role = payload.get("role")
        if role in ("developer", "assistant"):
            return "assistant"
        elif role == "user":
            return "user"
        return None

    # Old format: nested in payload.item
    item = payload.get("item", {})
    return item.get("role")
```

## Prevention

1. **Test parsers against real files** before deployment
2. **Check message types** when parser returns empty:
   ```bash
   cat transcript.jsonl | python -c "import json,sys; print(set(json.loads(l).get('type') for l in sys.stdin))"
   ```
3. **Support both formats** when external tools change — backward compatibility

## Related

- Oghma MVP plan: `docs/plans/2026-02-05-feat-oghma-mvp-plan.md`
- Codex CLI: https://github.com/openai/codex
