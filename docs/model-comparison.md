# Model Comparison for Memory Extraction

Tested Feb 2026 on Claude Code transcripts. Task: extract structured memories (JSON) from conversation text.

## Results

| Model | Provider | Memories | Quality | Cost (blended) | JSON Reliability |
|-------|----------|----------|---------|----------------|------------------|
| gpt-4o-mini | OpenAI | 10 | Good | ~$0.30/M | Excellent (native JSON mode) |
| google/gemini-2.0-flash-001 | OpenRouter | 6 | Good | ~$0.25/M | Good |
| deepseek/deepseek-chat-v3-0324 | OpenRouter | 5 | Good | ~$0.14/M | Good |
| **google/gemini-3-flash-preview** | OpenRouter | 6 | **Excellent** | ~$1.50/M | Good |
| moonshotai/kimi-k2 | OpenRouter | 5 | Excellent | ~$1.50/M | Good |
| anthropic/claude-3.5-haiku | OpenRouter | — | — | ~$1.00/M | Failed (JSON errors) |
| meta-llama/llama-3.3-70b-instruct | OpenRouter | — | — | ~$0.40/M | Failed (JSON errors) |

## Key Findings

### Quality vs Quantity Tradeoff
- GPT-4o-mini extracts more memories (10) but includes more noise (factual but verbose)
- Gemini 3 Flash and Kimi K2.5 extract fewer (5-6) but more insightful, selective memories
- For memory systems, **selectivity > volume** — fewer high-quality memories are more useful

### Thinking/Reasoning Modes Not Needed
Memory extraction is pattern recognition, not reasoning. Tested hypothesis: would thinking models improve quality? Conclusion: No — extraction is recognition (spot patterns) not reasoning (solve problems). Non-thinking Gemini 3 Flash already produces excellent results. Save thinking budgets for actual hard problems.

### JSON Reliability
- OpenAI models: Native `response_format=json_object` support, always clean
- OpenRouter models: Must parse from text, handle markdown wrappers
- Claude 3.5 Haiku and Llama 3.3 70B: Inconsistent JSON, not recommended for structured extraction

### Cost Analysis (for 966 sessions, ~15K tokens avg)
- GPT-4o-mini: ~$0.90
- Gemini 2.0 Flash: ~$0.75
- DeepSeek V3: ~$0.40
- Gemini 3 Flash: ~$1.50
- Kimi K2.5: ~$1.50

Absolute cost difference is trivial. Quality matters more than cost at this scale.

## Recommendation

**Primary: google/gemini-3-flash-preview**
- Best quality/cost balance
- Selective, insightful extractions
- Google infrastructure (reliable for background daemon)

**Fallback: moonshotai/kimi-k2**
- Equivalent quality
- Different provider for resilience

**Budget option: google/gemini-2.0-flash-001**
- Good quality at 1/6th the cost
- Acceptable if running at high volume

## Configuration

```yaml
# ~/.oghma/config.yaml
extraction:
  model: google/gemini-3-flash-preview  # or gpt-4o-mini for OpenAI
```

Environment variable: `OGHMA_EXTRACTION_MODEL`

Requires `OPENROUTER_API_KEY` for non-OpenAI models.
