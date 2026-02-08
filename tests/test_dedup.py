import tempfile
from pathlib import Path

import pytest

from oghma.storage import Storage


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def storage(temp_db_path):
    return Storage(db_path=temp_db_path)


def _make_embedding(base: list[float], dimensions: int = 1536) -> list[float]:
    """Create a full embedding by repeating base values."""
    emb = (base * ((dimensions // len(base)) + 1))[:dimensions]
    return emb


def _similar_embedding(
    base: list[float], noise: float = 0.01, dimensions: int = 1536
) -> list[float]:
    """Create a slightly different embedding (high cosine similarity to base)."""
    import random

    random.seed(42)
    emb = _make_embedding(base, dimensions)
    return [v + random.uniform(-noise, noise) for v in emb]


class TestUnionFind:
    def test_basic_union(self):
        from oghma.dedup import UnionFind

        uf = UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(3)

    def test_clusters(self):
        from oghma.dedup import UnionFind

        uf = UnionFind()
        uf.union(1, 2)
        uf.union(3, 4)
        clusters = uf.clusters()
        assert len(clusters) == 2

    def test_no_clusters_without_unions(self):
        from oghma.dedup import UnionFind

        uf = UnionFind()
        uf.find(1)
        uf.find(2)
        clusters = uf.clusters()
        assert len(clusters) == 0


class TestFindDuplicates:
    def test_finds_near_duplicates(self, storage):
        pytest.importorskip("numpy")
        from oghma.dedup import find_duplicates

        base = [0.1, 0.2, 0.3, 0.4, 0.5]
        emb1 = _make_embedding(base)
        emb2 = _similar_embedding(base, noise=0.001)
        emb3_full = _make_embedding([0.9, -0.5, 0.1, -0.3, 0.7])

        storage.add_memory(
            content="The delegation pattern uses Claude Code",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
            embedding=emb1,
        )
        storage.add_memory(
            content="Delegation pattern involves Claude Code orchestrating",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
            embedding=emb2,
        )
        storage.add_memory(
            content="SQLite FTS5 requires query escaping",
            category="gotcha",
            source_tool="test",
            source_file="test3.jsonl",
            embedding=emb3_full,
        )

        result = find_duplicates(storage, threshold=0.90)
        assert result.clusters_found == 1
        assert result.duplicates_removed == 1
        assert len(result.removed_ids) == 1

    def test_no_duplicates_when_different(self, storage):
        pytest.importorskip("numpy")
        from oghma.dedup import find_duplicates

        emb1 = _make_embedding([0.1, 0.2, 0.3])
        emb2 = _make_embedding([0.9, -0.5, 0.1])

        storage.add_memory(
            content="Memory A",
            category="learning",
            source_tool="test",
            source_file="test1.jsonl",
            embedding=emb1,
        )
        storage.add_memory(
            content="Memory B",
            category="learning",
            source_tool="test",
            source_file="test2.jsonl",
            embedding=emb2,
        )

        result = find_duplicates(storage, threshold=0.92)
        assert result.clusters_found == 0
        assert result.duplicates_removed == 0

    def test_category_filter(self, storage):
        pytest.importorskip("numpy")
        from oghma.dedup import find_duplicates

        base = [0.1, 0.2, 0.3, 0.4, 0.5]
        emb1 = _make_embedding(base)
        emb2 = _similar_embedding(base, noise=0.001)

        storage.add_memory(
            content="Workflow memory 1",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
            embedding=emb1,
        )
        storage.add_memory(
            content="Workflow memory 2",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
            embedding=emb2,
        )

        result = find_duplicates(storage, threshold=0.90, category="gotcha")
        assert result.clusters_found == 0


class TestRunDedup:
    def test_dry_run_does_not_delete(self, storage):
        pytest.importorskip("numpy")
        from oghma.dedup import run_dedup

        base = [0.1, 0.2, 0.3]
        emb1 = _make_embedding(base)
        emb2 = _similar_embedding(base, noise=0.001)

        storage.add_memory(
            content="Memory A short",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
            embedding=emb1,
        )
        storage.add_memory(
            content="Memory B with longer content here",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
            embedding=emb2,
        )

        before = storage.get_memory_count()
        result = run_dedup(storage, threshold=0.90, dry_run=True)
        after = storage.get_memory_count()
        assert before == after
        assert result.duplicates_removed > 0

    def test_execute_deletes(self, storage):
        pytest.importorskip("numpy")
        from oghma.dedup import run_dedup

        base = [0.1, 0.2, 0.3]
        emb1 = _make_embedding(base)
        emb2 = _similar_embedding(base, noise=0.001)

        storage.add_memory(
            content="Memory A short",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
            embedding=emb1,
        )
        storage.add_memory(
            content="Memory B with longer content here",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
            embedding=emb2,
        )

        before = storage.get_memory_count()
        result = run_dedup(storage, threshold=0.90, dry_run=False)
        after = storage.get_memory_count()
        assert after == before - result.duplicates_removed
        assert result.duplicates_removed == 1


class TestPickBest:
    def test_picks_highest_confidence(self, storage):
        from oghma.dedup import _pick_best

        id1 = storage.add_memory(
            content="Low conf memory",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
            confidence=0.5,
        )
        id2 = storage.add_memory(
            content="High conf memory",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
            confidence=0.9,
        )

        best = _pick_best(storage, [id1, id2])
        assert best == id2

    def test_picks_longest_on_tie(self, storage):
        from oghma.dedup import _pick_best

        id1 = storage.add_memory(
            content="Short",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
            confidence=0.9,
        )
        id2 = storage.add_memory(
            content="Much longer content here for testing",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
            confidence=0.9,
        )

        best = _pick_best(storage, [id1, id2])
        assert best == id2


class TestDeleteMemoriesBatch:
    def test_delete_batch(self, storage):
        id1 = storage.add_memory(
            content="Memory 1 to delete",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
        )
        id2 = storage.add_memory(
            content="Memory 2 to delete",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
        )
        id3 = storage.add_memory(
            content="Memory 3 to keep",
            category="workflow",
            source_tool="test",
            source_file="test3.jsonl",
        )

        deleted = storage.delete_memories_batch([id1, id2])
        assert deleted == 2
        assert storage.get_memory_count() == 1
        assert storage.get_memory_by_id(id3) is not None

    def test_delete_empty_list(self, storage):
        deleted = storage.delete_memories_batch([])
        assert deleted == 0


class TestGetAllEmbeddings:
    def test_returns_embeddings(self, storage):
        emb = _make_embedding([0.1, 0.2, 0.3])
        storage.add_memory(
            content="With embedding",
            category="workflow",
            source_tool="test",
            source_file="test1.jsonl",
            embedding=emb,
        )
        storage.add_memory(
            content="Without embedding",
            category="workflow",
            source_tool="test",
            source_file="test2.jsonl",
        )

        embeddings = storage.get_all_embeddings()
        assert len(embeddings) == 1
