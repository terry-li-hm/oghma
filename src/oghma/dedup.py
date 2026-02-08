"""Semantic deduplication and noise purging for memories."""

import logging
import struct
from dataclasses import dataclass, field

from oghma.storage import Storage

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    total_memories: int = 0
    embedded_memories: int = 0
    clusters_found: int = 0
    duplicates_removed: int = 0
    kept_ids: list[int] = field(default_factory=list)
    removed_ids: list[int] = field(default_factory=list)


class UnionFind:
    """Simple union-find for clustering."""

    def __init__(self):
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return {r: members for r, members in groups.items() if len(members) > 1}


def _bytes_to_floats(raw: bytes, dimensions: int) -> list[float]:
    """Deserialize sqlite-vec float32 blob to list of floats."""
    return list(struct.unpack(f"{dimensions}f", raw))


def find_duplicates(
    storage: Storage,
    threshold: float = 0.92,
    category: str | None = None,
    batch_size: int = 500,
) -> DedupResult:
    """Find semantic duplicate clusters.

    Uses numpy for efficient batch cosine similarity computation.
    Processes in blocks to manage memory on large datasets.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required for dedup. Install with: pip install oghma[dedup]"
        ) from e

    result = DedupResult()
    result.total_memories = storage.get_memory_count()

    all_embeddings = storage.get_all_embeddings()

    if category:
        category_memories = storage.get_all_memories(category=category)
        category_ids = {m["id"] for m in category_memories}
        all_embeddings = {k: v for k, v in all_embeddings.items() if k in category_ids}

    if not all_embeddings:
        logger.info("No embedded memories found")
        return result

    memory_ids = sorted(all_embeddings.keys())
    result.embedded_memories = len(memory_ids)
    dimensions = storage.embedding_dimensions

    matrix = np.array(
        [_bytes_to_floats(all_embeddings[mid], dimensions) for mid in memory_ids],
        dtype=np.float32,
    )

    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    uf = UnionFind()
    n = len(memory_ids)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = matrix[start:end]
        sim = batch @ matrix.T

        for i_local in range(end - start):
            i_global = start + i_local
            for j_global in range(i_global + 1, n):
                if sim[i_local, j_global] >= threshold:
                    uf.union(memory_ids[i_global], memory_ids[j_global])

    clusters = uf.clusters()
    result.clusters_found = len(clusters)

    if not clusters:
        return result

    for _root, members in clusters.items():
        best = _pick_best(storage, members)
        to_remove = [mid for mid in members if mid != best]
        result.kept_ids.append(best)
        result.removed_ids.extend(to_remove)

    result.duplicates_removed = len(result.removed_ids)
    return result


def _pick_best(storage: Storage, memory_ids: list[int]) -> int:
    """Pick the best memory from a cluster to keep.

    Priority: highest confidence > longest content > lowest ID (oldest).
    """
    best_id = memory_ids[0]
    best_conf = 0.0
    best_len = 0

    for mid in memory_ids:
        memory = storage.get_memory_by_id(mid)
        if memory is None:
            continue
        conf = memory["confidence"]
        content_len = len(memory["content"])
        if (
            (conf > best_conf)
            or (conf == best_conf and content_len > best_len)
            or (conf == best_conf and content_len == best_len and mid < best_id)
        ):
            best_id = mid
            best_conf = conf
            best_len = content_len

    return best_id


def run_dedup(
    storage: Storage,
    threshold: float = 0.92,
    category: str | None = None,
    dry_run: bool = True,
    batch_size: int = 500,
) -> DedupResult:
    """Find and optionally remove semantic duplicates."""
    result = find_duplicates(storage, threshold=threshold, category=category, batch_size=batch_size)

    if not dry_run and result.removed_ids:
        deleted = storage.delete_memories_batch(result.removed_ids)
        logger.info(f"Deleted {deleted} duplicate memories")

    return result


@dataclass
class PurgeResult:
    total_memories: int = 0
    noise_found: int = 0
    removed_ids: list[int] = field(default_factory=list)
    by_reason: dict[str, int] = field(default_factory=dict)


def find_noise(storage: Storage) -> PurgeResult:
    """Scan all active memories against noise patterns from extractor."""
    from oghma.extractor import _NOISE_PATTERNS

    result = PurgeResult()
    memories = storage.get_all_memories()
    result.total_memories = len(memories)

    for m in memories:
        content = m["content"]
        reason = _check_noise(content, _NOISE_PATTERNS)
        if reason:
            result.removed_ids.append(m["id"])
            result.by_reason[reason] = result.by_reason.get(reason, 0) + 1

    result.noise_found = len(result.removed_ids)
    return result


def _check_noise(content: str, patterns: list) -> str | None:
    """Check if content matches a noise pattern. Returns reason or None."""
    if len(content) < 30:
        return "too_short"
    for pattern in patterns:
        if pattern.search(content):
            if pattern.pattern.startswith("^The user") and len(content) > 100:
                continue
            return pattern.pattern[:40]
    return None


def run_purge(storage: Storage, dry_run: bool = True) -> PurgeResult:
    """Find and optionally remove noisy memories."""
    result = find_noise(storage)

    if not dry_run and result.removed_ids:
        deleted = storage.delete_memories_batch(result.removed_ids)
        logger.info(f"Purged {deleted} noisy memories")

    return result
