import json
import logging
import os
import time
from dataclasses import dataclass

from openai import OpenAI

from oghma.config import Config
from oghma.parsers import Message

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    content: str
    category: str
    confidence: float = 1.0


class Extractor:
    """Extracts memories from conversations using LLM."""

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0

    CATEGORIES = ["learning", "preference", "project_context", "gotcha", "workflow"]

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
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
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
            messages_text += f"{role_label}: {msg.content}\n\n"

        messages_text = messages_text[: self.max_chars]

        categories_desc = "\n".join(f"- {cat}" for cat in self.categories)

        prompt = (
            "You are a memory extraction system. "
            "Analyze this conversation and extract key memories.\n\n"
            f"Categories:\n{categories_desc}\n\n"
            f"Conversation:\n{messages_text}\n\n"
            "Extract memories in this JSON format:\n"
            '[  {"content": "...", "category": "...", "confidence": 0.0-1.0},\n'
            "  ...\n"
            "]\n\n"
            "Only extract clear, specific memories. Skip vague or trivial content.\n"
            "Return empty array [] if no significant memories found.\n"
            "Remember: respond with valid JSON only, no markdown formatting."
        )

        return prompt

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
