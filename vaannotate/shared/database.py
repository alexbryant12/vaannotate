"""Lightweight SQLite helpers and simple ORM primitives for the VAAnnotate apps."""
from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Type, TypeVar

Row = sqlite3.Row
T = TypeVar("T", bound="Record")


class Database:
    """A thin wrapper around sqlite3 with sane defaults for network shares.

    The helper applies WAL journaling and NORMAL synchronous writes which are the
    recommended settings for SQLite files that live on SMB shares.  Connections
    are short lived; call :meth:`transaction` or :meth:`connect` to work with
    them.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


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
