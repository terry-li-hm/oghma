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

    CATEGORIES = ["learning", "gotcha"]

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
            "Extract ONLY things that would save time if encountered again.\n\n"
            "TWO categories only:\n"
            "- learning: something discovered that wasn't obvious beforehand "
            "(API quirk, architecture decision rationale, command that solved a problem, "
            "performance finding, tool capability/limitation)\n"
            "- gotcha: something that broke, surprised, or wasted time "
            "(bug, silent failure, misleading docs, version incompatibility, "
            "environment-specific trap)\n\n"
            "STRICT RULES — return [] rather than extract noise:\n"
            "- Each memory must be a SPECIFIC, ACTIONABLE fact — not a summary or observation\n"
            "- Must contain enough detail to be useful without the original conversation\n"
            "- NO user profile observations ('The user prefers/is/has/wants...')\n"
            "- NO assistant narration ('The assistant suggested/helped...')\n"
            "- NO session logistics ('We discussed/read/searched...')\n"
            "- NO config restatements ('CLAUDE.md says...', 'skills must be synced')\n"
            "- NO trivially obvious facts ('Python uses pip', 'React has components')\n"
            "- NO meta-observations about the conversation itself\n"
            "- Aim for 0-3 memories per conversation. Most conversations have ZERO worth extracting.\n\n"
            "Good:\n"
            '  {"content": "sqlite-vec requires enable_load_extension(True) BEFORE '
            'sqlite_vec.load(conn)", "category": "gotcha", "confidence": 0.95}\n'
            '  {"content": "Oghma DB timestamps are UTC, not local time — '
            'compare with datetime.now(UTC)", "category": "gotcha", "confidence": 0.9}\n'
            '  {"content": "consilium --quick 6/6 unanimity usually means biased framing, '
            'not a clear answer", "category": "learning", "confidence": 0.9}\n\n'
            "Bad (DO NOT extract):\n"
            '  "The user is located in Hong Kong" — profile, not actionable\n'
            '  "Simon is the user\'s manager" — context, not a discovery\n'
            '  "The project uses SQLite for storage" — obvious\n'
            '  "The conversation covered Oghma maintenance" — logistics\n\n'
            f"Conversation:\n{messages_text}\n\n"
            "Return JSON array. Prefer [] over low-value extractions.\n"
            '[  {"content": "...", "category": "learning|gotcha", "confidence": 0.0-1.0}  ]\n'
            "Valid JSON only, no markdown."
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
