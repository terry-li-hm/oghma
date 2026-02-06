from dataclasses import dataclass

from openai import APIError

from oghma.embedder import Embedder
from oghma.storage import Storage


@dataclass
class MigrationResult:
    processed: int
    migrated: int
    skipped: int
    failed: int


class EmbeddingMigration:
    def __init__(
        self,
        storage: Storage,
        embedder: Embedder,
        batch_size: int = 100,
    ):
        self.storage = storage
        self.embedder = embedder
        self.batch_size = batch_size

    def run(self, dry_run: bool = False) -> MigrationResult:
        processed = 0
        migrated = 0
        failed = 0

        if dry_run:
            batch = self.storage.get_memories_without_embeddings(limit=10_000)
            processed = len(batch)
            return MigrationResult(
                processed=processed, migrated=0, skipped=processed, failed=0
            )

        while True:
            batch = self.storage.get_memories_without_embeddings(limit=self.batch_size)
            if not batch:
                break

            contents = [memory["content"] for memory in batch]
            processed += len(batch)

            try:
                vectors = self.embedder.embed_batch(contents)
            except (APIError, RuntimeError, ValueError):
                failed += len(batch)
                continue

            for memory, vector in zip(batch, vectors, strict=False):
                success = self.storage.upsert_memory_embedding(memory["id"], vector)
                if success:
                    migrated += 1
                else:
                    failed += 1

        skipped = max(processed - migrated - failed, 0)
        return MigrationResult(
            processed=processed,
            migrated=migrated,
            skipped=skipped,
            failed=failed,
        )
