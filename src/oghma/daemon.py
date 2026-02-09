import logging
import os
import signal
import sys
import time
from pathlib import Path

from oghma.config import Config
from oghma.embedder import EmbedConfig, create_embedder
from oghma.extractor import Extractor
from oghma.parsers import get_parser_for_file
from oghma.storage import Storage
from oghma.watcher import Watcher

logger = logging.getLogger(__name__)


class Daemon:
    """Main daemon for Oghma memory extraction."""

    def __init__(self, config: Config):
        self.config = config
        self.storage = Storage(config=config)
        self.watcher = Watcher(config, self.storage)
        self.extractor = Extractor(config)
        try:
            embed_config_data = config.get("embedding", {})
            self.embedder = create_embedder(EmbedConfig.from_dict(embed_config_data))
        except (ValueError, Exception):
            self.embedder = None
        self._setup_logging()
        self._running = False

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.config["daemon"]["log_file"]
        log_level = self.config["daemon"]["log_level"]
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def start(self) -> None:
        """Start the daemon main loop."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        pid_file = self.config["daemon"]["pid_file"]
        self._write_pid_file(pid_file)

        try:
            self._running = True
            poll_interval = self.config["daemon"]["poll_interval"]

            logger.info("Oghma daemon started")
            logger.info(f"Poll interval: {poll_interval} seconds")

            while self._running:
                try:
                    self._run_cycle()
                except Exception as e:
                    logger.error(f"Error in extraction cycle: {e}", exc_info=True)

                for _ in range(poll_interval):
                    if not self._running:
                        break
                    time.sleep(1)

        finally:
            self._cleanup(pid_file)

    def _run_cycle(self) -> None:
        """Run one extraction cycle."""
        logger.debug("Starting extraction cycle")
        changed_files = self.watcher.get_changed_files()

        if not changed_files:
            logger.debug("No changed files found")
            return

        logger.info(f"Processing {len(changed_files)} changed files")

        for file_path in changed_files:
            self._process_file(file_path)

    def _process_file(self, file_path: Path) -> None:
        """Process a single file: parse, extract, and save memories."""
        logger.info(f"Processing file: {file_path}")

        try:
            parser = get_parser_for_file(file_path)
            if not parser:
                logger.warning(f"No parser found for {file_path}")
                return

            messages = parser.parse(file_path)

            # Filter out messages that are just MEMORY.md writes to avoid
            # redundant extraction (those insights are already persisted)
            skip_patterns = self.config.get("extraction", {}).get("skip_content_patterns", [])
            if skip_patterns:
                messages = [
                    m for m in messages
                    if not any(pat in m.content for pat in skip_patterns)
                ]

            if not self.watcher.should_process(messages):
                logger.debug(f"Skipping {file_path}: not enough messages")
                return

            # Skip sessions with no/trivial human content
            user_messages = [m for m in messages if m.role == "user"]
            if not user_messages:
                logger.info(f"Skipping {file_path}: no user messages")
                mtime = file_path.stat().st_mtime
                size = file_path.stat().st_size
                self.storage.update_extraction_state(str(file_path), mtime, size, len(messages))
                return

            total_user_chars = sum(len(m.content) for m in user_messages)
            if total_user_chars < 200:
                logger.info(f"Skipping {file_path}: only {total_user_chars} chars of user content")
                mtime = file_path.stat().st_mtime
                size = file_path.stat().st_size
                self.storage.update_extraction_state(str(file_path), mtime, size, len(messages))
                return

            source_tool = self._get_tool_name(file_path)
            memories = self.extractor.extract(messages, source_tool)

            mtime = file_path.stat().st_mtime
            size = file_path.stat().st_size
            source_session = self._get_session_id(file_path)

            # Embed first, then check for duplicates before inserting
            if self.embedder:
                try:
                    contents = [m.content for m in memories]
                    vectors = self.embedder.embed_batch(contents)
                except Exception as e:
                    logger.warning(f"Failed to embed memories from {file_path}: {e}")
                    vectors = [None] * len(memories)
            else:
                vectors = [None] * len(memories)

            dedup_threshold = self.config.get("extraction", {}).get(
                "dedup_threshold", 0.92
            )

            new_memory_ids = []
            skipped_dupes = 0
            for memory, vector in zip(memories, vectors):
                if vector is not None:
                    existing = self.storage.find_similar_memory(
                        vector, threshold=dedup_threshold
                    )
                    if existing:
                        existing_id, similarity = existing
                        logger.debug(
                            f"Skipping duplicate (sim={similarity:.3f} "
                            f"with id={existing_id}): {memory.content[:80]}"
                        )
                        skipped_dupes += 1
                        continue

                memory_id = self.storage.add_memory(
                    content=memory.content,
                    category=memory.category,
                    source_tool=source_tool,
                    source_file=str(file_path),
                    source_session=source_session,
                    confidence=memory.confidence,
                    embedding=vector,
                )
                if memory_id is not None:
                    new_memory_ids.append(memory_id)

            if skipped_dupes:
                logger.info(f"Skipped {skipped_dupes} duplicate memories from {file_path}")

            self.storage.update_extraction_state(str(file_path), mtime, size, len(messages))

            self.storage.log_extraction(
                source_path=str(file_path),
                memories_extracted=len(memories),
            )

            logger.info(f"Extracted {len(memories)} memories from {file_path}")

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
            self.storage.log_extraction(source_path=str(file_path), error=str(e))

    def _get_tool_name(self, file_path: Path) -> str:
        """Extract tool name from file path."""
        path_str = str(file_path)

        if ".claude" in path_str:
            return "claude_code"
        elif ".codex" in path_str:
            return "codex"
        elif ".openclaw" in path_str:
            return "openclaw"
        elif ".local/share/opencode" in path_str or "opencode" in path_str:
            return "opencode"
        else:
            return "unknown"

    def _get_session_id(self, file_path: Path) -> str | None:
        """Extract session ID from file path if possible."""
        parts = file_path.parts
        for part in reversed(parts):
            if part.startswith("ses_") or part.startswith("rollout-"):
                return part
        return None

    def _write_pid_file(self, pid_file: str) -> None:
        """Write current PID to lock file."""
        Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _cleanup(self, pid_file: str) -> None:
        """Cleanup on shutdown."""
        logger.info("Oghma daemon stopped")
        pid_path = Path(pid_file)
        if pid_path.exists():
            pid_path.unlink()


def get_daemon_pid(pid_file: str) -> int | None:
    """Read PID from lock file, check if process is running."""
    pid_path = Path(pid_file)

    if not pid_path.exists():
        return None

    try:
        with open(pid_path) as f:
            pid_str = f.read().strip()
            pid = int(pid_str)

        if pid > 0:
            try:
                os.kill(pid, 0)
                return pid
            except OSError:
                pid_path.unlink()
                return None

    except (OSError, ValueError):
        if pid_path.exists():
            pid_path.unlink()

    return None
