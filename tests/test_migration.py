import tempfile
from pathlib import Path

import pytest

from oghma.embedder import EmbedConfig, Embedder
from oghma.migration import EmbeddingMigration
from oghma.storage import Storage


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__(EmbedConfig())

    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


@pytest.fixture
def storage() -> Storage:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = Storage(db_path=db_path)
    storage.add_memory(
        content="First memory",
        category="learning",
        source_tool="claude_code",
        source_file="/tmp/one.jsonl",
    )
    storage.add_memory(
        content="Second memory",
        category="workflow",
        source_tool="codex",
        source_file="/tmp/two.jsonl",
    )

    yield storage
    Path(db_path).unlink(missing_ok=True)


def test_migration_dry_run(storage: Storage) -> None:
    if not storage._vector_search_enabled:
        pytest.skip("sqlite-vec extension not available")

    migration = EmbeddingMigration(storage=storage, embedder=FakeEmbedder(), batch_size=1)
    result = migration.run(dry_run=True)

    assert result.processed == 2
    assert result.migrated == 0

    done, total = storage.get_embedding_progress()
    assert done == 0
    assert total == 2


def test_migration_updates_embeddings(storage: Storage) -> None:
    if not storage._vector_search_enabled:
        pytest.skip("sqlite-vec extension not available")

    migration = EmbeddingMigration(storage=storage, embedder=FakeEmbedder(), batch_size=2)
    result = migration.run(dry_run=False)

    assert result.processed == 2
    assert result.migrated == 2
    assert result.failed == 0

    done, total = storage.get_embedding_progress()
    assert done == 2
    assert total == 2


def test_dedup_migration_removes_duplicates() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    storage = Storage(db_path=db_path)

    with storage._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO memories
            (content, category, source_tool, source_file, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Duplicate memory", "learning", "claude_code", "/tmp/file.jsonl", None),
        )
        cursor.execute(
            """
            INSERT INTO memories
            (content, category, source_tool, source_file, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Duplicate memory", "learning", "claude_code", "/tmp/file.jsonl", None),
        )
        cursor.execute(
            """
            INSERT INTO memories
            (content, category, source_tool, source_file, content_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("Unique memory", "workflow", "codex", "/tmp/other.jsonl", None),
        )

    assert storage.get_memory_count() == 3

    storage2 = Storage(db_path=db_path)

    assert storage2.get_memory_count() == 2

    Path(db_path).unlink(missing_ok=True)
