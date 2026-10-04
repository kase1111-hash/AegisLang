"""
SQLite-backed persistent storage for AegisLang API.

Drop-in replacement for the in-memory Storage class. Enable with:
    AEGISLANG_STORAGE_BACKEND=sqlite

Data is stored in AEGISLANG_SQLITE_PATH (default: aegislang_data.db).
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class _SqliteDict:
    """Dict-like interface backed by a SQLite table.

    Supports __getitem__, __setitem__, __delitem__, __contains__,
    __len__, items(), keys(), values(), get(), and pop().
    """

    def __init__(self, conn: sqlite3.Connection, table: str, lock: threading.Lock):
        self._conn = conn
        self._table = table
        self._lock = lock

    def __getitem__(self, key: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {self._table} WHERE key = ?", (key,)  # noqa: S608  # nosec B608
            ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __setitem__(self, key: str, value: Any) -> None:
        blob = json.dumps(value, default=str)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self._table} (key, value) VALUES (?, ?)",  # noqa: S608  # nosec B608
                (key, blob),
            )
            self._conn.commit()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM {self._table} WHERE key = ?", (key,)  # noqa: S608  # nosec B608
            )
            self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {self._table} WHERE key = ?", (str(key),)  # noqa: S608  # nosec B608
            ).fetchone()
        return row is not None

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {self._table}"  # noqa: S608  # nosec B608
            ).fetchone()
        return row[0]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str, *args: Any) -> Any:
        try:
            val = self[key]
            del self[key]
            return val
        except KeyError:
            if args:
                return args[0]
            raise

    def items(self) -> Iterator[tuple[str, dict[str, Any]]]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key, value FROM {self._table}"  # noqa: S608  # nosec B608
            ).fetchall()
        for key, blob in rows:
            yield key, json.loads(blob)

    def keys(self) -> Iterator[str]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key FROM {self._table}"  # noqa: S608  # nosec B608
            ).fetchall()
        for (key,) in rows:
            yield key

    def values(self) -> Iterator[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT value FROM {self._table}"  # noqa: S608  # nosec B608
            ).fetchall()
        for (blob,) in rows:
            yield json.loads(blob)


class _SqliteListDict:
    """Dict-like interface for tables that store lists of values (clauses, artifacts).

    Keys are tracked in a companion ``{table}_keys`` table so that a key
    stored with an empty list is still reported as present.
    """

    def __init__(self, conn: sqlite3.Connection, table: str, lock: threading.Lock):
        self._conn = conn
        self._table = table
        self._keys_table = f"{table}_keys"
        self._lock = lock

    def __getitem__(self, key: str) -> list[dict[str, Any]]:
        if key not in self:
            raise KeyError(key)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT value FROM {self._table} WHERE key = ? ORDER BY rowid",  # noqa: S608  # nosec B608
                (key,),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def __setitem__(self, key: str, values: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute(
                f"DELETE FROM {self._table} WHERE key = ?", (key,)  # noqa: S608  # nosec B608
            )
            self._conn.execute(
                f"INSERT OR IGNORE INTO {self._keys_table} (key) VALUES (?)",  # noqa: S608  # nosec B608
                (key,),
            )
            for val in values:
                blob = json.dumps(val, default=str)
                self._conn.execute(
                    f"INSERT INTO {self._table} (key, value) VALUES (?, ?)",  # noqa: S608  # nosec B608
                    (key, blob),
                )
            self._conn.commit()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {self._keys_table} WHERE key = ?", (str(key),)  # noqa: S608  # nosec B608
            ).fetchone()
        return row is not None

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {self._keys_table}"  # noqa: S608  # nosec B608
            ).fetchone()
        return row[0]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> Iterator[tuple[str, list[dict[str, Any]]]]:
        with self._lock:
            keys = [
                r[0]
                for r in self._conn.execute(
                    f"SELECT key FROM {self._keys_table}"  # noqa: S608  # nosec B608
                ).fetchall()
            ]
        for key in keys:
            yield key, self[key]


class SqliteStorage:
    """SQLite-backed persistent storage with the same interface as in-memory Storage."""

    DEFAULT_JOB_TTL = 24 * 60 * 60
    CLEANUP_INTERVAL = 60 * 60

    def __init__(
        self,
        db_path: str | None = None,
        job_ttl_seconds: int | None = None,
    ):
        self._db_path = db_path or os.environ.get(
            "AEGISLANG_SQLITE_PATH", "aegislang_data.db"
        )
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._lock = threading.Lock()

        self._init_tables()

        self.job_ttl = job_ttl_seconds or int(
            os.environ.get("AEGISLANG_JOB_TTL_SECONDS", str(self.DEFAULT_JOB_TTL))
        )
        self._last_cleanup = time.time()

        # Expose dict-like interfaces matching in-memory Storage
        self.jobs = _SqliteDict(self._conn, "jobs", self._lock)
        self.documents = _SqliteDict(self._conn, "documents", self._lock)
        self.schemas = _SqliteDict(self._conn, "schemas", self._lock)
        self.traces = _SqliteDict(self._conn, "traces", self._lock)
        self.clauses = _SqliteListDict(self._conn, "clauses", self._lock)
        self.artifacts = _SqliteListDict(self._conn, "artifacts", self._lock)

        logger.info(
            "sqlite_storage_initialized",
            db_path=self._db_path,
        )

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS schemas (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS traces (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS clauses (
                    key TEXT NOT NULL,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_clauses_key ON clauses(key);
                CREATE TABLE IF NOT EXISTS clauses_keys (
                    key TEXT PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    key TEXT NOT NULL,
                    value TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_key ON artifacts(key);
                CREATE TABLE IF NOT EXISTS artifacts_keys (
                    key TEXT PRIMARY KEY
                );
                INSERT OR IGNORE INTO clauses_keys (key) SELECT DISTINCT key FROM clauses;
                INSERT OR IGNORE INTO artifacts_keys (key) SELECT DISTINCT key FROM artifacts;
            """)

    def create_job(self, job_type: str) -> str:
        """Create a new job and return its ID."""
        if time.time() - self._last_cleanup > self.CLEANUP_INTERVAL:
            self._cleanup_expired_jobs()
            self._last_cleanup = time.time()

        job_id = f"{job_type}_{uuid.uuid4().hex[:8]}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
        }
        return job_id

    def update_job(
        self,
        job_id: str,
        status: Any,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update job status."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job["status"] = status.value if hasattr(status, "value") else str(status)
            job["result"] = result
            job["error"] = error
            if str(status) in ("completed", "failed") or (
                hasattr(status, "value") and status.value in ("completed", "failed")
            ):
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.jobs[job_id] = job

    def store_document(self, doc_id: str, doc_data: dict[str, Any]) -> None:
        """Persist a document."""
        self.documents[doc_id] = doc_data

    def store_clauses(self, doc_id: str, clauses: list[dict[str, Any]]) -> None:
        """Persist the clauses parsed from a document."""
        self.clauses[doc_id] = clauses

    def store_artifacts(self, doc_id: str, artifacts: list[dict[str, Any]]) -> None:
        """Persist the artifacts compiled from a document."""
        self.artifacts[doc_id] = artifacts

    def store_trace(self, doc_id: str, trace: dict[str, Any]) -> None:
        """Persist validation results and the provenance graph for a document."""
        self.traces[doc_id] = trace

    def _cleanup_expired_jobs(self) -> None:
        """Remove jobs that have exceeded their TTL."""
        now = datetime.now(timezone.utc)
        expired = []
        for job_id, job in self.jobs.items():
            status = job.get("status", "")
            status_str = status.value if hasattr(status, "value") else str(status)
            if status_str not in ("completed", "failed"):
                continue
            completed_at = job.get("completed_at")
            if not completed_at:
                continue
            try:
                completed_time = datetime.fromisoformat(
                    completed_at.replace("Z", "+00:00")
                )
                if (now - completed_time).total_seconds() > self.job_ttl:
                    expired.append(job_id)
            except (ValueError, TypeError):
                continue

        for job_id in expired:
            del self.jobs[job_id]

        if expired:
            logger.info("expired_jobs_cleaned", count=len(expired))

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
