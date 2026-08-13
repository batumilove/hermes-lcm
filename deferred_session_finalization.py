"""Bounded, owner-scoped deferred session-end finalization."""

from __future__ import annotations

import copy
import logging
import sqlite3
import threading
import time
import weakref
from typing import Any

from .db_bootstrap import configure_connection
from .sqlite_util import _is_sqlite_locked_error, _temporary_sqlite_busy_timeout
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)


class DeferredSessionFinalizationCoordinator:
    """One serialized retry worker for a shared storage-owner family."""

    def __init__(self, owner: Any, *, writer_timeout_ms: int, busy_timeout_ms: int) -> None:
        self._owner_ref = weakref.ref(owner)
        self._source_store = owner.store
        self._source_lifecycle = owner.lifecycle
        self._store: Any | None = None
        self._lifecycle: Any | None = None
        self._writer_timeout_ms = int(writer_timeout_ms)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._condition = threading.Condition()
        self._pending: dict[str, dict[str, Any]] = {}
        self._accepting = True
        self._stopping = False
        self._worker: threading.Thread | None = None
        self.exhaustion_budget_seconds = 30.0
        self.retry_interval_seconds = 0.05
        # Keep the lifecycle callback's bounded return independent from the
        # worker's first SQLite attempt.  In particular, do not let a freshly
        # started worker contend with the callback while an external writer is
        # holding SQLite's busy wait.
        self.initial_handoff_seconds = 0.25

    def enqueue(
        self,
        *,
        session_id: str,
        conversation_id: str,
        source: str,
        protected_messages: list[dict],
        frontier_store_id: int,
        ingest_pending: bool,
    ) -> bool:
        payload = {
            "session_id": str(session_id),
            "conversation_id": str(conversation_id or ""),
            "source": str(source or ""),
            "messages": copy.deepcopy(protected_messages),
            "frontier_store_id": int(frontier_store_id or 0),
            "ingest_pending": bool(ingest_pending and protected_messages),
            "first_enqueued": time.monotonic(),
        }
        with self._condition:
            if not self._accepting:
                return False
            previous = self._pending.get(session_id)
            if previous is not None:
                payload["first_enqueued"] = previous["first_enqueued"]
            self._pending[session_id] = payload
            self._start_worker_locked()
            self._condition.notify_all()
        self._record("deferred")
        return True

    def _record(self, outcome: str) -> None:
        from .lifecycle_metrics import record_deferred_session_finalization

        record_deferred_session_finalization(outcome)

    def _start_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run,
            name="lcm-deferred-session-finalizer",
            daemon=True,
        )
        self._worker.start()

    def has_pending(self, session_id: str) -> bool:
        with self._condition:
            return session_id in self._pending

    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending)

    def _run(self) -> None:
        try:
            time.sleep(self.initial_handoff_seconds)
            while True:
                with self._condition:
                    while not self._pending and not self._stopping:
                        if self._owner_ref() is None:
                            return
                        self._condition.wait(timeout=0.25)
                    if self._stopping:
                        return
                    session_id = next(iter(self._pending))
                self._attempt(session_id)
        except BaseException:
            logger.exception(
                "LCM deferred session-end worker failed; a later enqueue may restart it"
            )
        finally:
            with self._condition:
                if self._worker is threading.current_thread():
                    self._worker = None
                self._condition.notify_all()

    def _attempt(self, session_id: str) -> None:
        phase = "raw_ingest"
        payload: dict[str, Any] | None = None
        try:
            self._ensure_helpers()
            assert self._store is not None
            assert self._lifecycle is not None
            with _temporary_sqlite_busy_timeout(
                [self._store._conn, self._lifecycle._conn],
                self._busy_timeout_ms,
                write_lock=self._store._write_lock,
                write_lock_timeout_ms=self._writer_timeout_ms,
                write_lock_operation="deferred_session_end_ingest_finalize",
            ):
                with self._condition:
                    payload = self._pending.get(session_id)
                if payload is None:
                    return
                if payload["ingest_pending"]:
                    stored = self._store.get_range(
                        payload["session_id"],
                        limit=len(payload["messages"]),
                        conversation_id=payload["conversation_id"] or None,
                    )
                    prefix_count = 0
                    for stored_message, pending_message in zip(stored, payload["messages"]):
                        if not self._same_message(stored_message, pending_message):
                            break
                        prefix_count += 1
                    remaining_messages = payload["messages"][prefix_count:]
                    if remaining_messages:
                        self._store._append_protected_batch(
                            payload["session_id"],
                            remaining_messages,
                            [count_message_tokens(msg) for msg in remaining_messages],
                            source=payload["source"],
                            conversation_id=payload["conversation_id"],
                        )
                    payload["ingest_pending"] = False
                phase = "lifecycle_finalization"
                self._lifecycle.finalize_session(
                    payload["conversation_id"],
                    payload["session_id"],
                    frontier_store_id=payload["frontier_store_id"],
                )
        except Exception as exc:
            with self._condition:
                current = self._pending.get(session_id)
            payload = current or payload
            if payload is None:
                return
            elapsed = time.monotonic() - payload["first_enqueued"]
            if elapsed >= self.exhaustion_budget_seconds:
                with self._condition:
                    if self._pending.get(session_id) is payload:
                        self._pending.pop(session_id, None)
                    self._condition.notify_all()
                self._record("exhausted")
                logger.warning(
                    "LCM deferred session-end finalization exhausted: session=%s "
                    "phase=%s error_kind=%s elapsed_s=%.3f: %s",
                    session_id,
                    phase,
                    "sqlite_lock" if _is_sqlite_locked_error(exc) else "callback_error",
                    elapsed,
                    exc,
                )
            else:
                with self._condition:
                    self._condition.wait(timeout=self.retry_interval_seconds)
            return

        with self._condition:
            if self._pending.get(session_id) is payload:
                self._pending.pop(session_id, None)
            self._condition.notify_all()
        self._record("completed")

    def _ensure_helpers(self) -> None:
        if self._store is not None:
            return
        source_store = self._source_store
        source_lifecycle = self._source_lifecycle
        required_store = ("db_path", "_ingest_protection_config", "_hermes_home", "_write_lock")
        if not all(hasattr(source_store, name) for name in required_store) or not all(
            hasattr(source_lifecycle, name) for name in ("db_path", "_lock")
        ):
            raise RuntimeError("deferred finalization requires SQLite-backed storage helpers")

        store = object.__new__(type(source_store))
        store.db_path = source_store.db_path
        store._ingest_protection_config = source_store._ingest_protection_config
        store._hermes_home = source_store._hermes_home
        store._write_lock = source_store._write_lock
        store._conn = self._open_connection(store.db_path)

        lifecycle = object.__new__(type(source_lifecycle))
        lifecycle.db_path = source_lifecycle.db_path
        lifecycle._lock = source_lifecycle._lock
        try:
            lifecycle._conn = self._open_connection(lifecycle.db_path)
        except Exception:
            store._conn.close()
            store._conn = None
            raise
        self._store = store
        self._lifecycle = lifecycle

    @staticmethod
    def _open_connection(db_path: Any) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(db_path), timeout=30.0, check_same_thread=False, isolation_level=None
        )
        configure_connection(connection)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _same_message(stored: dict[str, Any], pending: dict[str, Any]) -> bool:
        return all(
            stored.get(key) == pending.get(key)
            for key in ("role", "content", "tool_call_id", "tool_name")
        )

    def shutdown(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            if not self._stopping:
                self._accepting = False
                while self._pending and time.monotonic() < deadline:
                    self._condition.notify_all()
                    self._condition.wait(
                        timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                    )
                retained = tuple(self._pending)
                self._stopping = True
                self._condition.notify_all()
                for session_id in retained:
                    logger.warning(
                        "LCM deferred session-end finalization retained after bounded "
                        "shutdown: session=%s",
                        session_id,
                    )
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, deadline - time.monotonic()) + 0.5)
        stopped = worker is None or worker is threading.current_thread() or not worker.is_alive()
        if not stopped:
            logger.warning("LCM deferred session-end worker did not stop within bounded shutdown")
            return False
        first_error: Exception | None = None
        for helper in (self._store, self._lifecycle):
            if helper is None or helper._conn is None:
                continue
            try:
                helper._conn.close()
                helper._conn = None
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return stopped
