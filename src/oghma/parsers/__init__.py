from pathlib import Path

from oghma.parsers.base import BaseParser
from oghma.parsers.claude_code import ClaudeCodeParser
from oghma.parsers.codex import CodexParser
from oghma.parsers.openclaw import OpenClawParser
from oghma.parsers.opencode import OpenCodeParser

PARSERS: list[BaseParser] = [
    ClaudeCodeParser(),
    CodexParser(),
    OpenClawParser(),
    OpenCodeParser(),
]


def get_parser_for_file(file_path: Path) -> BaseParser | None:
    for parser in PARSERS:
        if parser.can_parse(file_path):
            return parser
    return None
