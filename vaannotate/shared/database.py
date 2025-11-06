"""Lightweight SQLite helpers and simple ORM primitives for the VAAnnotate apps."""
from __future__ import annotations

import contextlib
import sqlite3
import uuid
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Type, TypeVar

Row = sqlite3.Row
T = TypeVar("T", bound="Record")


class Database:
    """A thin wrapper around sqlite3 with sane defaults for network shares.

    The helper applies WAL journaling and NORMAL synchronous writes which are
    recommended settings for SQLite files that live on SMB shares.  The class
    also supports loading the database into an in-memory cache so that callers
    can operate on slow network files with minimal disk I/O; call
    :meth:`enable_memory_cache` to opt in and :meth:`flush_to_disk` to persist
    pending changes.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_uri: Optional[str] = None
        self._memory_keeper: Optional[sqlite3.Connection] = None
        self._cache_enabled = False
        self._dirty = False

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

    def enable_memory_cache(self) -> None:
        """Load the SQLite file into a shared in-memory database."""

        if self._cache_enabled:
            return
        cache_id = uuid.uuid4().hex
        uri = f"file:vaannotate-cache-{cache_id}?mode=memory&cache=shared"
        keeper = sqlite3.connect(uri, uri=True)
        self._apply_pragmas(keeper)
        if self.path.exists():
            disk_conn = sqlite3.connect(self.path)
            disk_conn.row_factory = sqlite3.Row
            disk_conn.backup(keeper)
            disk_conn.close()
        self._memory_uri = uri
        self._memory_keeper = keeper
        self._cache_enabled = True
        self._dirty = False

    def connect(self) -> sqlite3.Connection:
        if self._cache_enabled and self._memory_uri:
            conn = sqlite3.connect(self._memory_uri, uri=True)
            self._apply_pragmas(conn)
            return conn
        conn = sqlite3.connect(self.path)
        self._apply_pragmas(conn)
        return conn

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
            if self._cache_enabled and conn.total_changes:
                self._dirty = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def flush_to_disk(self) -> None:
        """Persist pending changes from the in-memory cache to disk."""

        if not self._cache_enabled or not self._memory_uri:
            return
        if not self._dirty and self.path.exists():
            return
        mem_conn = sqlite3.connect(self._memory_uri, uri=True)
        self._apply_pragmas(mem_conn)
        disk_conn = sqlite3.connect(self.path)
        self._apply_pragmas(disk_conn)
        mem_conn.backup(disk_conn)
        disk_conn.close()
        mem_conn.close()
        self._dirty = False

    def close(self) -> None:
        if self._memory_keeper is not None:
            self._memory_keeper.close()
            self._memory_keeper = None
        self._memory_uri = None
        self._cache_enabled = False
        self._dirty = False

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def refresh_from_disk(self) -> None:
        """Reload the cached database contents from disk."""

        if not self._cache_enabled:
            return

        if self._memory_keeper is not None:
            self._memory_keeper.close()
            self._memory_keeper = None

        self._memory_uri = None
        self._cache_enabled = False
        self.enable_memory_cache()


class Record:
    """Base class for ORM style records."""

    __tablename__: str = ""
    __schema__: str = ""

    @classmethod
    def create_table(cls, conn: sqlite3.Connection) -> None:
        if not cls.__schema__:
            raise ValueError(f"{cls.__name__} does not define __schema__")
        conn.execute(cls.__schema__)

    @classmethod
    def from_row(cls: Type[T], row: Row) -> T:
        data = {field.name: row[field.name] for field in fields(cls) if field.name in row.keys()}
        return cls(**data)  # type: ignore[arg-type]

    def to_row(self) -> Dict[str, Any]:
        values = asdict(self)
        return values

    @classmethod
    def insert_many(cls, conn: sqlite3.Connection, records: Iterable[T]) -> None:
        placeholders = ", ".join([":" + f.name for f in fields(cls)])
        sql = f"INSERT OR REPLACE INTO {cls.__tablename__} VALUES ({placeholders})"
        conn.executemany(sql, [r.to_row() for r in records])

    def save(self, conn: sqlite3.Connection) -> None:
        placeholders = ", ".join([":" + f.name for f in fields(self)])
        sql = f"INSERT OR REPLACE INTO {self.__tablename__} VALUES ({placeholders})"
        conn.execute(sql, self.to_row())


def ensure_schema(conn: sqlite3.Connection, models: Sequence[Type[Record]]) -> None:
    for model in models:
        model.create_table(conn)


def fetch_all(conn: sqlite3.Connection, sql: str, params: Sequence[Any] | None = None) -> List[sqlite3.Row]:
    cur = conn.execute(sql, params or [])
    return cur.fetchall()


def fetch_one(conn: sqlite3.Connection, sql: str, params: Sequence[Any] | None = None) -> Optional[sqlite3.Row]:
    cur = conn.execute(sql, params or [])
    return cur.fetchone()
