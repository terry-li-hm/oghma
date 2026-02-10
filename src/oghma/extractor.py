import json
import logging
import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from oghma.config import Config
from oghma.parsers import Message

logger = logging.getLogger(__name__)

# Regex patterns for post-extraction noise filtering
_NOISE_PATTERNS = [
    # Meta-references to config/memory files
    re.compile(r"CLAUDE\.md|MEMORY\.md|memory file|auto.?memory", re.I),
    # Shallow "The user is/has/uses" observations under 80 chars
    re.compile(
        r"^The user (is |has |wants to |prefers |uses |works |likes |located |transitioning )"
    ),
    # Re-extracted system instructions
    re.compile(r"must be auto.?commit|after editing.*(commit|push)|skill.?sync", re.I),
    # Meta-observations about the conversation itself
    re.compile(r"^The (assistant|AI|system|conversation) ", re.I),
    # Narrating what a config file contains
    re.compile(r"^The .*(config|settings|configuration) (file |contains |specifies )", re.I),
]

# Strip these from message content before sending to LLM
_STRIP_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


@dataclass
class Memory:
    content: str
    category: str
    confidence: float = 1.0


class Extractor:
    """Extracts memories from conversations using LLM."""

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0

    CATEGORIES = ["learning", "preference", "project_context", "gotcha", "workflow", "promoted"]

    # Models that require OpenRouter
    OPENROUTER_PREFIXES = ("google/", "anthropic/", "meta-llama/", "deepseek/", "moonshotai/")

    def __init__(self, config: Config):
        self.config = config
        self.model = config.get("extraction", {}).get("model", "gpt-4o-mini")
        self.max_chars = config.get("extraction", {}).get("max_content_chars", 4000)
        self.categories = config.get("extraction", {}).get("categories") or self.CATEGORIES
        self.confidence_threshold = config.get("extraction", {}).get("confidence_threshold", 0.5)

        # Determine which API to use based on model name
        if self.model.startswith(self.OPENROUTER_PREFIXES):
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            self.use_openrouter = True
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            self.use_openrouter = False

    def extract(self, messages: list[Message], source_tool: str) -> list[Memory]:
        """Extract memories from a list of messages."""
        if not messages:
            return []

        prompt = self._build_prompt(messages)

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._call_openai(prompt)
                memories = self._parse_response(response)

                valid_memories = [
                    m
                    for m in memories
                    if m.category in self.categories
                    and m.confidence >= self.confidence_threshold
                    and not self._is_noise(m)
                ]

                logger.info(
                    f"Extracted {len(valid_memories)} memories from {source_tool} "
                    f"(attempt {attempt + 1})"
                )
                return valid_memories

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2**attempt)
                    logger.warning(
                        f"Extraction attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Extraction failed after {self.MAX_RETRIES} attempts: {e}")

        return []

    def _call_openai(self, prompt: str) -> str:
        """Call LLM API and return the response text."""
        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a memory extraction system. "
                    "Always respond with valid JSON only, no markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
        }

        # OpenAI models support structured output, OpenRouter models don't
        if not self.use_openrouter:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM API")

        return content

    def _build_prompt(self, messages: list[Message]) -> str:
        """Build the extraction prompt."""
        messages_text = ""
        for msg in messages[:100]:
            role_label = "User" if msg.role == "user" else "Assistant"
            content = _STRIP_RE.sub("", msg.content).strip()
            if content:
                messages_text += f"{role_label}: {content}\n\n"

        messages_text = messages_text[: self.max_chars]

        categories_desc = "\n".join(f"- {cat}" for cat in self.categories)

        prompt = (
            "You are a memory extraction system for an AI coding assistant. "
            "Extract ONLY genuinely useful memories that would help in future sessions.\n\n"
            f"Categories:\n{categories_desc}\n\n"
            "EXTRACT (high value):\n"
            "- Technical discoveries: bugs found, workarounds, API quirks, library gotchas\n"
            "- Tool-specific learnings: commands that worked, configurations that solved problems\n"
            "- Project decisions: architecture choices, why X was chosen over Y\n"
            "- Workflow patterns: what sequence of steps solved a problem\n"
            "- Error solutions: what error occurred and how it was fixed\n\n"
            "DO NOT EXTRACT (noise — be strict about this):\n"
            "- What the user's setup/environment is (timezone, OS, tools installed)\n"
            "- What config files contain or what instructions say\n"
            "- Observations about the user: 'The user prefers X', 'The user is working on Y', "
            "'The user wants to...', 'The user has...'\n"
            "- Observations about the assistant: 'The assistant suggested...', 'The AI helped...', "
            "'The system provided...', 'The conversation covered...'\n"
            "- Information from system prompts, CLAUDE.md, MEMORY.md, or README files\n"
            "- What the assistant said or did (focus on discoveries, not narration)\n"
            "- Trivially obvious facts ('The project uses Python', 'The app uses React')\n"
            "- Restatements of config: 'auto-memory captures learnings', 'skills must be synced'\n"
            "- Session logistics: what files were read, what tools were used, "
            "what was discussed\n\n"
            "Good examples:\n"
            '  {"content": "sqlite-vec requires enable_load_extension(True) BEFORE '
            'sqlite_vec.load(conn)", "category": "gotcha", "confidence": 0.95}\n'
            '  {"content": "Gmail MCP can send but cannot reply to threads — no '
            'thread ID support", "category": "gotcha", "confidence": 0.9}\n'
            '  {"content": "OpenCode times out on small edits (<25 lines); reserve '
            'for bulk work", "category": "learning", "confidence": 0.9}\n\n'
            "Bad examples (DO NOT extract these):\n"
            '  {"content": "The user is located in Hong Kong"} — setup info, not a learning\n'
            '  {"content": "The user prefers pnpm"} — config fact, not actionable\n'
            '  {"content": "The user wants to improve extraction quality"}'
            " — narrating the session\n"
            '  {"content": "The project uses SQLite for storage"} — trivially obvious\n'
            '  {"content": "CLAUDE.md must be auto-committed"} — system instruction\n'
            '  {"content": "The assistant helped debug the API issue"}'
            " — narrating assistant actions\n"
            '  {"content": "Non-obvious learnings should be captured to skills"}'
            " — restating config\n"
            '  {"content": "The conversation covered Oghma maintenance"} — session logistics\n\n'
            f"Conversation:\n{messages_text}\n\n"
            "Extract memories as JSON. Return [] if nothing worth remembering.\n"
            '[  {"content": "...", "category": "...", "confidence": 0.0-1.0}  ]\n'
            "Respond with valid JSON only, no markdown."
        )

        return prompt

    def _is_noise(self, memory: Memory) -> bool:
        """Check if a memory matches known noise patterns."""
        content = memory.content
        # Too short to be useful
        if len(content) < 30:
            return True
        # Check regex noise patterns
        for pattern in _NOISE_PATTERNS:
            if pattern.search(content):
                # "The user is/has/uses..." is noise UNLESS the content is long
                # enough to contain a genuine insight (>100 chars)
                if pattern.pattern.startswith("^The user") and len(content) > 100:
                    continue
                logger.debug(f"Filtered noise: {content[:80]}")
                return True
        return False

    def _parse_response(self, response_text: str) -> list[Memory]:
        """Parse LLM response into Memory objects."""
        response_text = response_text.strip()

        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {response_text[:200]}...")
            return []

        if not isinstance(data, list):
            return []

        memories: list[Memory] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            category = item.get("category")
            confidence = item.get("confidence", 1.0)

            if not content or not category:
                continue

            if not isinstance(confidence, (int, float)):
                confidence = 1.0

            confidence = max(0.0, min(1.0, float(confidence)))

            memories.append(Memory(content=content, category=category, confidence=confidence))

        return memories
