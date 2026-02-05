import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

from oghma.config import Config, get_db_path


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
    def __init__(self, db_path: str | None = None, config: Config | None = None):
        self.db_path = db_path or get_db_path(config)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

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

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    category,
                    source_tool,
                    content=memories,
                    content_rowid=id
                )
            """)

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
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute(
                """
                INSERT INTO memories
                (content, category, source_tool, source_file,
                 source_session, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content,
                    category,
                    source_tool,
                    source_file,
                    source_session,
                    confidence,
                    metadata_json,
                ),
            )
            return cursor.lastrowid or 0

    def search_memories(
        self,
        query: str,
        category: str | None = None,
        source_tool: str | None = None,
        status: str = "active",
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryRecord]:
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
            params: list[str | int] = [query, status]

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

            results = []
            for row in rows:
                results.append(
                    {
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
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    }
                )
            return results

    def get_memory_by_id(self, memory_id: int) -> MemoryRecord | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cursor.fetchone()

            if row is None:
                return None

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
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }

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

            return [
                {
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
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

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
