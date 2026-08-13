"""SQLite lock-contention helpers shared by the LCM engine.

Isolated from ``engine.py`` (WS5 seam): lock-contention detection, bounded
``busy_timeout`` changes, and transaction-preserving savepoints are pure SQLite
concerns with no engine state. Callers keep their own policy constants (for
example the session-end timeout budget).
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, List


_PROCESS_SQLITE_WRITE_LOCK_TIMEOUT_SECONDS = 30.0
_PROCESS_SQLITE_WRITE_LOCKS_GUARD = threading.Lock()
_LOCK_OWNER_LABEL_LIMIT = 96
_LOCK_TIMEOUT_DETAIL_LIMIT = 384


def _bounded_diagnostic_text(value: Any, *, limit: int) -> str:
    """Return one bounded, single-line diagnostic field."""
    try:
        text = str(value)[:limit]
    except Exception:
        return "unknown"
    sanitized = "".join(character if character.isprintable() else "?" for character in text)
    return sanitized or "unknown"


def _bounded_diagnostic_label(value: Any) -> str:
    """Return a bounded token that cannot inject another key/value field."""
    text = _bounded_diagnostic_text(value, limit=_LOCK_OWNER_LABEL_LIMIT)
    label = "".join(
        character
        if character.isascii()
        and (character.isalnum() or character in "-_.:/@()")
        else "_"
        for character in text
    )
    return label or "unknown"


def _lock_operation_label() -> str:
    """Infer the lock caller without allowing frame telemetry to break acquisition."""
    try:
        operation_frame = sys._getframe(2)
        if operation_frame.f_code.co_name == "__enter__":
            operation_frame = operation_frame.f_back or operation_frame
        operation = operation_frame.f_code.co_name
    except Exception:
        operation = "unknown"
    return _bounded_diagnostic_label(operation)


def _optional_timeout_detail(write_lock: Any) -> str:
    """Return optional bounded telemetry without masking the primary timeout."""
    try:
        timeout_detail = getattr(write_lock, "timeout_detail", None)
        if not callable(timeout_detail):
            return ""
        detail = _bounded_diagnostic_text(
            timeout_detail(), limit=_LOCK_TIMEOUT_DETAIL_LIMIT
        )
    except Exception:
        return ""
    return "; " + detail


class ProcessSQLiteWriteLock:
    """Reentrant same-process writer gate with a SQLite-aligned wait bound."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._metadata_lock = threading.Lock()
        self._owner_thread_id: int | None = None
        self._owner_thread_name = ""
        self._owner_operation = ""
        self._owner_acquired_at = 0.0
        self._owner_depth = 0

    def acquire(
        self,
        blocking: bool = True,
        timeout: float = -1,
        *,
        operation: str | None = None,
    ) -> bool:
        thread_id = threading.get_ident()
        try:
            thread_name = threading.current_thread().name
        except Exception:
            thread_name = "unknown"
        thread_name = _bounded_diagnostic_label(thread_name)
        operation = _bounded_diagnostic_label(operation or _lock_operation_label())
        acquired = self._lock.acquire(blocking, timeout)
        if not acquired:
            return False
        try:
            now = time.monotonic()
        except Exception:
            now = 0.0
        with self._metadata_lock:
            if self._owner_thread_id == thread_id:
                self._owner_depth += 1
            else:
                self._owner_thread_id = thread_id
                self._owner_thread_name = thread_name
                self._owner_operation = operation
                self._owner_acquired_at = now
                self._owner_depth = 1
        return True

    def release(self) -> None:
        with self._metadata_lock:
            if self._owner_thread_id != threading.get_ident() or self._owner_depth <= 0:
                raise RuntimeError("cannot release un-acquired lock")
            self._owner_depth -= 1
            if self._owner_depth == 0:
                self._owner_thread_id = None
                self._owner_thread_name = ""
                self._owner_operation = ""
                self._owner_acquired_at = 0.0
            self._lock.release()

    def owner_snapshot(self) -> dict[str, Any]:
        """Return bounded lock-owner diagnostics without database or SQL content."""
        with self._metadata_lock:
            if self._owner_thread_id is None:
                return {}
            try:
                age_seconds = max(0.0, time.monotonic() - self._owner_acquired_at)
            except Exception:
                age_seconds = 0.0
            return {
                "thread_id": self._owner_thread_id,
                "thread_name": self._owner_thread_name,
                "operation": self._owner_operation,
                "depth": self._owner_depth,
                "age_seconds": age_seconds,
            }

    def timeout_detail(self) -> str:
        owner = self.owner_snapshot()
        if not owner:
            return "owner_thread_id=unknown owner_thread_name=unknown owner_operation=unknown owner_age_s=unknown"
        return (
            f"owner_thread_id={owner['thread_id']} "
            f"owner_thread_name={owner['thread_name']} "
            f"owner_operation={owner['operation']} "
            f"owner_age_s={owner['age_seconds']:.3f}"
        )

    def _is_owned(self) -> bool:
        return self._lock._is_owned()

    def __enter__(self) -> "ProcessSQLiteWriteLock":
        if not self.acquire(timeout=_PROCESS_SQLITE_WRITE_LOCK_TIMEOUT_SECONDS):
            raise sqlite3.OperationalError(
                "database is locked: timed out waiting for process-wide SQLite writer"
                + _optional_timeout_detail(self)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()

    @contextmanager
    def attributed(self, operation: str) -> Iterator["ProcessSQLiteWriteLock"]:
        """Hold the coordinator with an explicit bounded operation label."""
        if not self.acquire(
            timeout=_PROCESS_SQLITE_WRITE_LOCK_TIMEOUT_SECONDS,
            operation=operation,
        ):
            raise sqlite3.OperationalError(
                "database is locked: timed out waiting for process-wide SQLite writer"
                + _optional_timeout_detail(self)
            )
        try:
            yield self
        finally:
            self.release()


_PROCESS_SQLITE_WRITE_LOCKS: dict[str, ProcessSQLiteWriteLock] = {}


def process_sqlite_write_lock(db_path: str | Path) -> ProcessSQLiteWriteLock:
    """Return the process-wide reentrant writer lock for one SQLite file."""
    key = str(Path(db_path).expanduser().resolve())
    with _PROCESS_SQLITE_WRITE_LOCKS_GUARD:
        lock = _PROCESS_SQLITE_WRITE_LOCKS.get(key)
        if lock is None:
            lock = ProcessSQLiteWriteLock()
            _PROCESS_SQLITE_WRITE_LOCKS[key] = lock
        return lock


def _is_sqlite_locked_error(exc: BaseException) -> bool:
    """Return True when an exception chain represents SQLite lock contention."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        code = getattr(current, "sqlite_errorcode", None)
        primary_code = code & 0xFF if isinstance(code, int) else None
        if isinstance(current, sqlite3.Error) and (
            primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
            or "locked" in message
            or "busy" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _sqlite_busy_timeout_ms(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


@contextmanager
def _sqlite_savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    """Isolate helper writes without taking ownership of a caller transaction."""
    # UUID hex contains only identifier-safe characters and keeps every nested
    # helper's SAVEPOINT name unique with a fixed upper bound on name length.
    name = f"lcm_{uuid.uuid4().hex}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        try:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
        finally:
            conn.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {name}")


@contextmanager
def _temporary_sqlite_busy_timeout(
    connections: List[sqlite3.Connection | None],
    timeout_ms: int,
    *,
    write_lock: Any = None,
    write_lock_timeout_ms: int | None = None,
) -> Iterator[None]:
    """Temporarily bound lock waits while excluding same-process writers."""
    bounded_timeout = max(0, int(timeout_ms))
    bounded_write_lock_timeout = (
        bounded_timeout
        if write_lock_timeout_ms is None
        else max(0, int(write_lock_timeout_ms))
    )
    lock_context = nullcontext()
    if write_lock is not None:
        lock_context = _acquire_process_write_lock(
            write_lock, timeout_seconds=bounded_write_lock_timeout / 1000.0
        )
    with lock_context:
        originals: list[tuple[sqlite3.Connection, int]] = []
        for conn in connections:
            if conn is None:
                continue
            original = _sqlite_busy_timeout_ms(conn)
            conn.execute(f"PRAGMA busy_timeout={bounded_timeout}")
            originals.append((conn, original))
        try:
            yield
        finally:
            for conn, original in reversed(originals):
                conn.execute(f"PRAGMA busy_timeout={original}")


@contextmanager
def _acquire_process_write_lock(
    write_lock: Any, *, timeout_seconds: float
) -> Iterator[None]:
    """Acquire a coordinator using the caller's existing bounded wait budget."""
    if not write_lock.acquire(timeout=max(0.0, float(timeout_seconds))):
        raise sqlite3.OperationalError(
            "database is locked: timed out waiting for process-wide SQLite writer"
            + _optional_timeout_detail(write_lock)
        )
    try:
        yield
    finally:
        write_lock.release()
