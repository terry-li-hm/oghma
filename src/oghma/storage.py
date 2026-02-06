import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing_extensions import TypedDict

from oghma.config import Config, get_db_path

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - optional runtime dependency in tests
    sqlite_vec = None

logger = logging.getLogger(__name__)


class MemoryRecord(TypedDict):
    id: int
    content: str
    category: str
    source_tool: str
    source_file: str
    source_session: str | None
    confidence: float
    created_at: str
    updated_at: str
    status: str
    has_embedding: int
    metadata: dict[str, Any]


class ExtractionStateRecord(TypedDict):
    id: int
    source_path: str
    last_mtime: float
    last_size: int
    last_extracted_at: str
    message_count: int


class ExtractionLogRecord(TypedDict):
    id: int
    source_path: str
    memories_extracted: int
    tokens_used: int
    duration_ms: int
    error: str | None
    created_at: str


class Storage:
    # Hybrid search tuning constants.
    MIN_HYBRID_QUERY_LENGTH = 3
    VECTOR_K_MULTIPLIER = 4
    VECTOR_K_MIN = 25
    RRF_K_DEFAULT = 60
    _vec_load_warned = False

    def __init__(
        self,
        db_path: str | None = None,
        config: Config | None = None,
        read_only: bool = False,
    ):
        self.db_path = db_path or get_db_path(config)
        self.read_only = read_only
        self._config = config
        self.embedding_dimensions = (
            config.get("embedding", {}).get("dimensions", 1536) if config else 1536
        )
        self._vec_available = sqlite_vec is not None
        self._vector_search_enabled = self._vec_available

        if self.read_only:
            db_file = Path(self.db_path)
            if not db_file.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")
            self._connection_target = f"file:{db_file.resolve()}?mode=ro"
            self._use_uri = True
        else:
            self._connection_target = self.db_path
            self._use_uri = False
            self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self._connection_target, uri=self._use_uri)
        conn.row_factory = sqlite3.Row
        self._configure_connection(conn)
        try:
            yield conn
            if not self.read_only:
                conn.commit()
        except Exception:
            if not self.read_only:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        if not self._vec_available:
            return
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            self._vector_search_enabled = False
            if not self._vec_load_warned:
                logger.debug("sqlite-vec extension failed to load; vector search disabled")
                Storage._vec_load_warned = True

    def _fallback_keyword_search(
        self,
        *,
        query: str,
        category: str | None,
        source_tool: str | None,
        status: str,
        limit: int,
        offset: int,
        reason: str,
        exc_info: bool = False,
    ) -> list[MemoryRecord]:
        log_fn = logger.warning if exc_info else logger.info
        log_fn("Hybrid/vector search fell back to keyword search: %s", reason, exc_info=exc_info)
        return self.search_memories(
            query=query,
            category=category,
            source_tool=source_tool,
            status=status,
            limit=limit,
            offset=offset,
        )

    def _ensure_column(
        self,
        cursor: sqlite3.Cursor,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _compute_content_hash(self, content: str, category: str, source_file: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def _migrate_dedup(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("DROP INDEX IF EXISTS idx_memories_dedup")

        cursor.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "content_hash" not in columns:
            cursor.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")

        cursor.execute("SELECT id, content, category, source_file, content_hash FROM memories")
        rows = cursor.fetchall()

        needs_backfill = any(row[4] is None for row in rows)
        if needs_backfill:
            for row in rows:
                memory_id, content, category, source_file, content_hash = row
                if content_hash is None:
                    computed_hash = self._compute_content_hash(content, category, source_file)
                    cursor.execute(
                        "UPDATE memories SET content_hash = ? WHERE id = ?",
                        (computed_hash, memory_id),
                    )

        cursor.execute("""
            DELETE FROM memories
            WHERE id NOT IN (
                SELECT MIN(id) FROM memories
                WHERE content_hash IS NOT NULL
                GROUP BY content_hash
            )
        """)

        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_dedup ON memories(content_hash)
        """)

    def _serialize_embedding(self, embedding: list[float]) -> Any:
        if sqlite_vec and hasattr(sqlite_vec, "serialize_float32"):
            return sqlite_vec.serialize_float32(embedding)
        return json.dumps(embedding)

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source_tool TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_session TEXT,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    metadata JSON
                )
            """)
            self._ensure_column(cursor, "memories", "has_embedding", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "memories", "content_hash", "TEXT")

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_source_tool ON memories(source_tool)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)
            """)

            self._migrate_dedup(cursor)

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    category,
                    source_tool,
                    content=memories,
                    content_rowid=id
                )
            """)

            if self._vector_search_enabled:
                cursor.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                        memory_id INTEGER PRIMARY KEY,
                        embedding float[{self.embedding_dimensions}]
                    )
                    """
                )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extraction_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT UNIQUE NOT NULL,
                    last_mtime REAL NOT NULL,
                    last_size INTEGER NOT NULL,
                    last_extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0
                )
            """)

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_extraction_state_path
                ON extraction_state(source_path)
            """
            )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extraction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    memories_extracted INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_extraction_log_path ON extraction_log(source_path)
            """)

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_extraction_log_created_at
                ON extraction_log(created_at DESC)
            """
            )

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, category, source_tool)
                    VALUES (NEW.id, NEW.content, NEW.category, NEW.source_tool);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = OLD.id;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = OLD.id;
                    INSERT INTO memories_fts(rowid, content, category, source_tool)
                    VALUES (NEW.id, NEW.content, NEW.category, NEW.source_tool);
                END
            """)

    def add_memory(
        self,
        content: str,
        category: str,
        source_tool: str,
        source_file: str,
        source_session: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> int | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            content_hash = self._compute_content_hash(content, category, source_file)
            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute(
                """
                INSERT OR IGNORE INTO memories
                (content, category, source_tool, source_file,
                 source_session, confidence, metadata, has_embedding, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content,
                    category,
                    source_tool,
                    source_file,
                    source_session,
                    confidence,
                    metadata_json,
                    1 if embedding is not None and self._vector_search_enabled else 0,
                    content_hash,
                ),
            )
            memory_id = cursor.lastrowid or 0

            if memory_id == 0:
                return None

            if embedding is not None and self._vector_search_enabled:
                cursor.execute(
                    "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, self._serialize_embedding(embedding)),
                )

            return memory_id

    def search_memories(
        self,
        query: str,
        category: str | None = None,
        source_tool: str | None = None,
        status: str = "active",
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        fts_query = self._escape_fts_query(query)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            sql = """
                SELECT m.*
                FROM memories m
                WHERE m.id IN (
                    SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?
                )
                AND m.status = ?
            """
            params: list[str | int] = [fts_query, status]

            if category:
                sql += " AND m.category = ?"
                params.append(category)

            if source_tool:
                sql += " AND m.source_tool = ?"
                params.append(source_tool)

            sql += " ORDER BY m.created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            return [self._row_to_memory_record(row) for row in rows]

    def _escape_fts_query(self, query: str) -> str:
        if not query.strip():
            return '""'
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    def _row_to_memory_record(self, row: sqlite3.Row) -> MemoryRecord:
        return {
            "id": row["id"],
            "content": row["content"],
            "category": row["category"],
            "source_tool": row["source_tool"],
            "source_file": row["source_file"],
            "source_session": row["source_session"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "has_embedding": row["has_embedding"] if "has_embedding" in row.keys() else 0,
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }

    def upsert_memory_embedding(self, memory_id: int, embedding: list[float]) -> bool:
        if not self._vector_search_enabled:
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM memories WHERE id = ?", (memory_id,))
            if cursor.fetchone() is None:
                return False

            cursor.execute(
                "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                (memory_id, self._serialize_embedding(embedding)),
            )
            cursor.execute(
                (
                    "UPDATE memories SET has_embedding = 1, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?"
                ),
                (memory_id,),
            )
            return True

    def get_memories_without_embeddings(self, limit: int = 100) -> list[MemoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM memories
                WHERE status = 'active' AND has_embedding = 0
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [self._row_to_memory_record(row) for row in rows]

    def get_embedding_progress(self) -> tuple[int, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'")
            total_row = cursor.fetchone()
            cursor.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'active' AND has_embedding = 1"
            )
            done_row = cursor.fetchone()
            total = int(total_row[0]) if total_row else 0
            done = int(done_row[0]) if done_row else 0
            return done, total

    def search_memories_hybrid(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        category: str | None = None,
        source_tool: str | None = None,
        status: str = "active",
        limit: int = 10,
        offset: int = 0,
        search_mode: str = "hybrid",
        rrf_k: int = RRF_K_DEFAULT,
    ) -> list[MemoryRecord]:
        if search_mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError("search_mode must be one of: keyword, vector, hybrid")

        if search_mode == "keyword":
            return self.search_memories(
                query=query,
                category=category,
                source_tool=source_tool,
                status=status,
                limit=limit,
                offset=offset,
            )

        if not self._vector_search_enabled:
            return self._fallback_keyword_search(
                query=query,
                category=category,
                source_tool=source_tool,
                status=status,
                limit=limit,
                offset=offset,
                reason="sqlite-vec unavailable",
            )

        if len(query.strip()) < self.MIN_HYBRID_QUERY_LENGTH:
            return self._fallback_keyword_search(
                query=query,
                category=category,
                source_tool=source_tool,
                status=status,
                limit=limit,
                offset=offset,
                reason=f"query shorter than {self.MIN_HYBRID_QUERY_LENGTH} chars",
            )

        if not query_embedding:
            return self._fallback_keyword_search(
                query=query,
                category=category,
                source_tool=source_tool,
                status=status,
                limit=limit,
                offset=offset,
                reason="query embedding missing",
            )

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM memories WHERE status = ? AND has_embedding = 1",
                (status,),
            )
            row = cursor.fetchone()
            if not row or row[0] == 0:
                return self._fallback_keyword_search(
                    query=query,
                    category=category,
                    source_tool=source_tool,
                    status=status,
                    limit=limit,
                    offset=offset,
                    reason="no embedded memories available",
                )

            filters = ""
            filter_params: list[str] = [status]
            if category:
                filters += " AND m.category = ?"
                filter_params.append(category)
            if source_tool:
                filters += " AND m.source_tool = ?"
                filter_params.append(source_tool)

            vector_k = max(limit * self.VECTOR_K_MULTIPLIER, self.VECTOR_K_MIN)
            vec_query = self._serialize_embedding(query_embedding)

            try:
                if search_mode == "vector":
                    sql = f"""
                        WITH vec AS (
                            SELECT m.id AS memory_id
                            FROM memories_vec v
                            JOIN memories m ON m.id = v.memory_id
                            WHERE v.embedding MATCH ? AND k = ?
                              AND m.status = ?
                              {filters}
                            ORDER BY v.distance
                            LIMIT ?
                        )
                        SELECT m.*
                        FROM vec
                        JOIN memories m ON m.id = vec.memory_id
                        ORDER BY (1.0 + 0.3 / (1.0 + (julianday('now') - julianday(m.created_at)))) DESC
                        LIMIT ? OFFSET ?
                    """
                    params: list[Any] = [
                        vec_query,
                        vector_k,
                        *filter_params,
                        vector_k,
                        limit,
                        offset,
                    ]
                else:
                    sql = f"""
                        WITH
                        fts AS (
                            SELECT
                                m.id AS memory_id,
                                ROW_NUMBER() OVER (ORDER BY bm25(memories_fts)) AS fts_rank
                            FROM memories_fts
                            JOIN memories m ON m.id = memories_fts.rowid
                            WHERE memories_fts MATCH ?
                              AND m.status = ?
                              {filters}
                            LIMIT ?
                        ),
                        vec AS (
                            SELECT
                                m.id AS memory_id,
                                ROW_NUMBER() OVER (ORDER BY v.distance) AS vec_rank
                            FROM memories_vec v
                            JOIN memories m ON m.id = v.memory_id
                            WHERE v.embedding MATCH ? AND k = ?
                              AND m.status = ?
                              {filters}
                            LIMIT ?
                        ),
                        rrf AS (
                            SELECT memory_id, (1.0 / (? + fts_rank)) * 0.5 AS score FROM fts
                            UNION ALL
                            SELECT memory_id, (1.0 / (? + vec_rank)) * 0.5 AS score FROM vec
                        ),
                        ranked AS (
                            SELECT rrf.memory_id,
                                   SUM(rrf.score) * (1.0 + 0.5 / (1.0 + (julianday('now') - julianday(m.created_at)))) AS rrf_score
                            FROM rrf
                            JOIN memories m ON m.id = rrf.memory_id
                            GROUP BY rrf.memory_id
                        )
                        SELECT m.*
                        FROM ranked
                        JOIN memories m ON m.id = ranked.memory_id
                        ORDER BY ranked.rrf_score DESC, m.created_at DESC
                        LIMIT ? OFFSET ?
                    """
                    params = [
                        query,
                        *filter_params,
                        vector_k,
                        vec_query,
                        vector_k,
                        *filter_params,
                        vector_k,
                        rrf_k,
                        rrf_k,
                        limit,
                        offset,
                    ]

                cursor.execute(sql, params)
                rows = cursor.fetchall()
                return [self._row_to_memory_record(row) for row in rows]
            except sqlite3.Error:
                return self._fallback_keyword_search(
                    query=query,
                    category=category,
                    source_tool=source_tool,
                    status=status,
                    limit=limit,
                    offset=offset,
                    reason="sqlite query error",
                    exc_info=True,
                )

    def get_memory_by_id(self, memory_id: int) -> MemoryRecord | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_memory_record(row)

    def update_memory_status(self, memory_id: int, status: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memories SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, memory_id),
            )
            return cursor.rowcount > 0

    def get_extraction_state(self, source_path: str) -> ExtractionStateRecord | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM extraction_state WHERE source_path = ?", (source_path,))
            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "id": row["id"],
                "source_path": row["source_path"],
                "last_mtime": row["last_mtime"],
                "last_size": row["last_size"],
                "last_extracted_at": row["last_extracted_at"],
                "message_count": row["message_count"],
            }

    def update_extraction_state(
        self,
        source_path: str,
        last_mtime: float,
        last_size: int,
        message_count: int = 0,
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO extraction_state
                (source_path, last_mtime, last_size, message_count, last_extracted_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_path) DO UPDATE SET
                    last_mtime = excluded.last_mtime,
                    last_size = excluded.last_size,
                    message_count = excluded.message_count,
                    last_extracted_at = CURRENT_TIMESTAMP
                """,
                (source_path, last_mtime, last_size, message_count),
            )

    def log_extraction(
        self,
        source_path: str,
        memories_extracted: int = 0,
        tokens_used: int = 0,
        duration_ms: int = 0,
        error: str | None = None,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO extraction_log
                (source_path, memories_extracted, tokens_used, duration_ms, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_path, memories_extracted, tokens_used, duration_ms, error),
            )
            return cursor.lastrowid or 0

    def get_memory_count(self, status: str = "active") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories WHERE status = ?", (status,))
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_all_memories(
        self, status: str = "active", category: str | None = None
    ) -> list[MemoryRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            sql = "SELECT * FROM memories WHERE status = ?"
            params: list[str] = [status]

            if category:
                sql += " AND category = ?"
                params.append(category)

            sql += " ORDER BY created_at DESC"
            cursor.execute(sql, params)
            rows = cursor.fetchall()

            return [self._row_to_memory_record(row) for row in rows]

    def get_all_extraction_states(self) -> list[ExtractionStateRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM extraction_state")
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "source_path": row["source_path"],
                    "last_mtime": row["last_mtime"],
                    "last_size": row["last_size"],
                    "last_extracted_at": row["last_extracted_at"],
                    "message_count": row["message_count"],
                }
                for row in rows
            ]

    def get_recent_extraction_logs(self, limit: int = 10) -> list[ExtractionLogRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM extraction_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "source_path": row["source_path"],
                    "memories_extracted": row["memories_extracted"],
                    "tokens_used": row["tokens_used"],
                    "duration_ms": row["duration_ms"],
                    "error": row["error"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
