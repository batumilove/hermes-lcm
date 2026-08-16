"""Durable lifecycle/checkpoint state for hermes-lcm.

This is the smallest viable substrate for cross-turn/session lifecycle state:
- which logical conversation a session belongs to
- which session is currently bound
- which session was last finalized
- the active session frontier/checkpoint marker
- the last finalized frontier marker
"""

from __future__ import annotations

import functools
import sqlite3
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Optional

from .db_bootstrap import configure_connection, refuse_schema_version_too_new, run_versioned_migrations
from .sqlite_util import process_sqlite_write_lock


_MAX_EMPTY_LIFECYCLE_GC_BATCH_SIZE = 100


def _synchronized(method):
    """Serialize a read-modify-write method on the store's reentrant lock.

    The lifecycle connection is shared across threads (check_same_thread=False,
    autocommit). Without serialization, two callers reading state and then
    writing can interleave and clobber each other's update.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock.attributed(method.__name__):
            return method(self, *args, **kwargs)
    return wrapper


@dataclass
class LifecycleState:
    conversation_id: str
    current_session_id: str | None
    last_finalized_session_id: str | None
    current_frontier_store_id: int
    last_finalized_frontier_store_id: int
    debt_kind: str | None
    debt_size_estimate: int
    current_bound_at: float | None
    last_finalized_at: float | None
    debt_updated_at: float | None
    last_maintenance_attempt_at: float | None
    last_rollover_at: float | None
    last_reset_at: float | None
    updated_at: float


class LifecycleStateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        # Serialize this store's read-modify-write flows and coordinate them
        # with every other in-process helper writing the same database.
        self._lock = process_sqlite_write_lock(self.db_path)
        with self._lock:
            self._init_db()
        # Lifecycle metrics — registered *after* __init__ succeeds.
        from .lifecycle_metrics import register_lifecycle_state_store_created
        register_lifecycle_state_store_created(self)

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        self._conn.commit()

    def close(self) -> None:
        from .lifecycle_metrics import (
            begin_lifecycle_state_store_close,
            complete_lifecycle_state_store_close,
            fail_lifecycle_state_store_close,
        )

        begin_lifecycle_state_store_close(self)
        conn = getattr(self, "_conn", None)
        try:
            with getattr(self, "_lock", nullcontext()):
                if conn is not None:
                    try:
                        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except sqlite3.Error:
                        pass
                    conn.close()
                    self._conn = None
        except Exception:
            fail_lifecycle_state_store_close(self)
            raise
        else:
            complete_lifecycle_state_store_close(self)

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live SQLite connection, or ``None`` once :meth:`close` has run.

        Exposed for read-oriented diagnostics -- for example the doctor's
        maintenance-debt scan -- that need ad-hoc queries the store does not wrap
        in a purpose-built method. Callers must treat it as read-only; writes go
        through the store's own methods.
        """
        return getattr(self, "_conn", None)

    def row_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM lcm_lifecycle_state").fetchone()
        return int(row["count"] if row else 0)

    def _row_to_state(self, row: sqlite3.Row | None) -> LifecycleState | None:
        if row is None:
            return None
        return LifecycleState(
            conversation_id=row["conversation_id"],
            current_session_id=row["current_session_id"],
            last_finalized_session_id=row["last_finalized_session_id"],
            current_frontier_store_id=int(row["current_frontier_store_id"] or 0),
            last_finalized_frontier_store_id=int(row["last_finalized_frontier_store_id"] or 0),
            debt_kind=row["debt_kind"],
            debt_size_estimate=int(row["debt_size_estimate"] or 0),
            current_bound_at=row["current_bound_at"],
            last_finalized_at=row["last_finalized_at"],
            debt_updated_at=row["debt_updated_at"],
            last_maintenance_attempt_at=row["last_maintenance_attempt_at"],
            last_rollover_at=row["last_rollover_at"],
            last_reset_at=row["last_reset_at"],
            updated_at=float(row["updated_at"] or 0.0),
        )

    def get_by_conversation(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        row = self._conn.execute(
            "SELECT * FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return self._row_to_state(row)

    def get_by_session(self, session_id: str | None) -> LifecycleState | None:
        if not session_id:
            return None
        row = self._conn.execute(
            """
            SELECT *
            FROM lcm_lifecycle_state
            WHERE current_session_id = ? OR last_finalized_session_id = ?
            ORDER BY CASE WHEN current_session_id = ? THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1
            """,
            (session_id, session_id, session_id),
        ).fetchone()
        return self._row_to_state(row)

    @_synchronized
    def bind_session(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> LifecycleState:
        existing = self.get_by_conversation(conversation_id) if conversation_id else self.get_by_session(session_id)
        conversation_id = conversation_id or (existing.conversation_id if existing else session_id)
        now = time.time()
        current_frontier = 0
        current_bound_at = now
        last_finalized_session_id = None
        last_finalized_frontier = 0
        debt_kind = None
        debt_size_estimate = 0
        last_finalized_at = None
        debt_updated_at = None
        last_maintenance_attempt_at = None
        last_rollover_at = None
        last_reset_at = None

        if existing is not None:
            if existing.current_session_id == session_id:
                return existing
            current_frontier = (
                existing.current_frontier_store_id if existing.current_session_id == session_id else 0
            )
            current_bound_at = (
                existing.current_bound_at if existing.current_session_id == session_id else now
            )
            last_finalized_session_id = existing.last_finalized_session_id
            last_finalized_frontier = existing.last_finalized_frontier_store_id
            debt_kind = existing.debt_kind
            debt_size_estimate = existing.debt_size_estimate
            last_finalized_at = existing.last_finalized_at
            debt_updated_at = existing.debt_updated_at
            last_maintenance_attempt_at = existing.last_maintenance_attempt_at
            last_rollover_at = (
                now
                if (
                    (existing.current_session_id and existing.current_session_id != session_id)
                    or (
                        existing.current_session_id is None
                        and existing.last_finalized_session_id
                        and existing.last_finalized_session_id != session_id
                    )
                )
                else existing.last_rollover_at
            )
            last_reset_at = existing.last_reset_at

        self._conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id,
                current_session_id,
                last_finalized_session_id,
                current_frontier_store_id,
                last_finalized_frontier_store_id,
                debt_kind,
                debt_size_estimate,
                current_bound_at,
                last_finalized_at,
                debt_updated_at,
                last_maintenance_attempt_at,
                last_rollover_at,
                last_reset_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = excluded.current_frontier_store_id,
                last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id,
                debt_kind = excluded.debt_kind,
                debt_size_estimate = excluded.debt_size_estimate,
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                debt_updated_at = excluded.debt_updated_at,
                last_maintenance_attempt_at = excluded.last_maintenance_attempt_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                session_id,
                last_finalized_session_id,
                current_frontier,
                last_finalized_frontier,
                debt_kind,
                debt_size_estimate,
                current_bound_at,
                last_finalized_at,
                debt_updated_at,
                last_maintenance_attempt_at,
                last_rollover_at,
                last_reset_at,
                now,
            ),
        )
        self._conn.commit()
        state = self.get_by_conversation(conversation_id)
        assert state is not None
        return state

    @_synchronized
    def finalize_session(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int = 0,
    ) -> LifecycleState | None:
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        current_session_id = state.current_session_id
        current_frontier = state.current_frontier_store_id
        if current_session_id == session_id:
            current_session_id = None
            current_frontier = 0
        finalized_frontier = max(
            int(frontier_store_id or 0),
            state.last_finalized_frontier_store_id,
        )
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_session_id = ?,
                last_finalized_session_id = ?,
                current_frontier_store_id = ?,
                last_finalized_frontier_store_id = ?,
                debt_kind = debt_kind,
                debt_size_estimate = debt_size_estimate,
                last_finalized_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (
                current_session_id,
                session_id,
                current_frontier,
                finalized_frontier,
                now,
                now,
                state.conversation_id,
            ),
        )
        self._conn.commit()
        return self.get_by_conversation(state.conversation_id)

    @_synchronized
    def record_rollover(
        self,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        finalized_frontier_store_id: int = 0,
    ) -> LifecycleState:
        state = self.get_by_conversation(conversation_id)
        if (
            state is not None
            and state.current_session_id == new_session_id
            and state.last_finalized_session_id == old_session_id
        ):
            return state

        now = time.time()
        last_finalized_frontier = max(
            int(finalized_frontier_store_id or 0),
            state.last_finalized_frontier_store_id if state else 0,
        )
        self._conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id,
                current_session_id,
                last_finalized_session_id,
                current_frontier_store_id,
                last_finalized_frontier_store_id,
                current_bound_at,
                last_finalized_at,
                last_rollover_at,
                last_reset_at,
                updated_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = 0,
                last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id,
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                new_session_id,
                old_session_id,
                last_finalized_frontier,
                now,
                now,
                now,
                now,
                now,
            ),
        )
        self._conn.commit()
        updated = self.get_by_conversation(conversation_id)
        assert updated is not None
        return updated

    def get_fragmentation_stats(
        self,
        state_db_path: str | Path | None = None,
        *,
        empty_lifecycle_max_age_hours: float | None = None,
    ) -> dict[str, Any]:
        """Return read-only lifecycle/session fragmentation diagnostics.

        This intentionally reports mismatches only. It does not infer that every
        mismatch is corrupt, and it never rewrites lifecycle, message, DAG, or
        Hermes host state. Repair/cleanup flows must stay explicit and separate.
        """
        conn = self._conn
        assert conn is not None

        def _count(query: str, params: tuple[Any, ...] = ()) -> int:
            row = conn.execute(query, params).fetchone()
            return int(row[0] if row else 0)

        def _session_ids(query: str) -> set[str]:
            return {
                str(row[0])
                for row in conn.execute(query).fetchall()
                if row[0]
            }

        message_sessions = _session_ids("SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL")
        node_sessions = _session_ids("SELECT DISTINCT session_id FROM summary_nodes WHERE session_id IS NOT NULL")
        lcm_any_sessions = message_sessions | node_sessions
        state_sessions: set[str] = set()
        state_db_read_success = False
        lifecycle_current_sessions = _session_ids(
            "SELECT DISTINCT current_session_id FROM lcm_lifecycle_state WHERE current_session_id IS NOT NULL"
        )
        lifecycle_last_finalized_sessions = _session_ids(
            "SELECT DISTINCT last_finalized_session_id FROM lcm_lifecycle_state WHERE last_finalized_session_id IS NOT NULL"
        )
        lifecycle_referenced_sessions = lifecycle_current_sessions | lifecycle_last_finalized_sessions

        empty_lifecycle_rows = 0
        actionable_empty_lifecycle_rows = 0
        now = time.time()
        max_age_seconds = (
            float(empty_lifecycle_max_age_hours) * 3600.0
            if empty_lifecycle_max_age_hours is not None
            else None
        )
        for row in conn.execute(
            """
            SELECT current_session_id, last_finalized_session_id,
                   current_bound_at, last_finalized_at, updated_at
            FROM lcm_lifecycle_state
            """
        ).fetchall():
            refs = {
                str(value)
                for value in (row["current_session_id"], row["last_finalized_session_id"])
                if value
            }
            if not refs or refs.isdisjoint(lcm_any_sessions):
                empty_lifecycle_rows += 1
                row_age_anchor = (
                    row["current_bound_at"]
                    or row["last_finalized_at"]
                    or row["updated_at"]
                )
                if (
                    max_age_seconds is None
                    or row_age_anchor is None
                    or (now - float(row_age_anchor)) >= max_age_seconds
                ):
                    actionable_empty_lifecycle_rows += 1

        stats: dict[str, Any] = {
            "read_only": True,
            "lifecycle_rows": _count("SELECT COUNT(*) FROM lcm_lifecycle_state"),
            "empty_lifecycle_rows": empty_lifecycle_rows,
            "actionable_empty_lifecycle_rows": actionable_empty_lifecycle_rows,
            "recent_gc_protected_empty_lifecycle_rows": (
                empty_lifecycle_rows - actionable_empty_lifecycle_rows
            ),
            "messages_total": _count("SELECT COUNT(*) FROM messages"),
            "summary_nodes_total": _count("SELECT COUNT(*) FROM summary_nodes"),
            "distinct_message_sessions": len(message_sessions),
            "distinct_node_sessions": len(node_sessions),
            "distinct_lcm_any_sessions": len(lcm_any_sessions),
            "lifecycle_current_sessions": len(lifecycle_current_sessions),
            "lifecycle_last_finalized_sessions": len(lifecycle_last_finalized_sessions),
            "lifecycle_current_missing_in_messages": len(lifecycle_current_sessions - message_sessions),
            "lifecycle_current_missing_in_nodes": len(lifecycle_current_sessions - node_sessions),
            "lifecycle_current_missing_in_lcm_any": len(lifecycle_current_sessions - lcm_any_sessions),
            "lifecycle_last_finalized_missing_in_messages": len(lifecycle_last_finalized_sessions - message_sessions),
            "lifecycle_last_finalized_missing_in_nodes": len(lifecycle_last_finalized_sessions - node_sessions),
            "lifecycle_last_finalized_missing_in_lcm_any": len(lifecycle_last_finalized_sessions - lcm_any_sessions),
            "message_sessions_without_lifecycle_current": len(message_sessions - lifecycle_current_sessions),
            "message_sessions_without_lifecycle_reference": len(message_sessions - lifecycle_referenced_sessions),
            "node_sessions_without_lifecycle_reference": len(node_sessions - lifecycle_referenced_sessions),
            "state_db_checked": False,
            "state_db_error": "",
            "state_sessions_total": 0,
            "lifecycle_current_missing_in_state": 0,
            "lifecycle_last_finalized_missing_in_state": 0,
            "lcm_message_sessions_missing_in_state": 0,
            "lcm_node_sessions_missing_in_state": 0,
            "state_sessions_missing_in_lcm_messages": 0,
            "state_sessions_missing_in_lcm_any": 0,
        }

        if state_db_path:
            path = Path(state_db_path).expanduser()
            if path.exists():
                stats["state_db_checked"] = True
                try:
                    state_uri = path.resolve().as_uri() + "?mode=ro"
                    state_conn = sqlite3.connect(state_uri, uri=True)
                    try:
                        state_rows = state_conn.execute("SELECT id FROM sessions WHERE id IS NOT NULL").fetchall()
                    finally:
                        state_conn.close()
                    state_sessions = {str(row[0]) for row in state_rows if row[0]}
                    state_db_read_success = True
                    stats.update({
                        "state_sessions_total": len(state_sessions),
                        "lifecycle_current_missing_in_state": len(lifecycle_current_sessions - state_sessions),
                        "lifecycle_last_finalized_missing_in_state": len(
                            lifecycle_last_finalized_sessions - state_sessions
                        ),
                        "lcm_message_sessions_missing_in_state": len(message_sessions - state_sessions),
                        "lcm_node_sessions_missing_in_state": len(node_sessions - state_sessions),
                        "state_sessions_missing_in_lcm_messages": len(state_sessions - message_sessions),
                        "state_sessions_missing_in_lcm_any": len(state_sessions - lcm_any_sessions),
                    })
                except Exception as exc:  # pragma: no cover - defensive
                    stats["state_db_error"] = str(exc)
            else:
                stats["state_db_error"] = f"state database not found: {path}"

        stats["classification"] = self._classify_fragmentation(
            lifecycle_rows=stats["lifecycle_rows"],
            lifecycle_current_sessions=lifecycle_current_sessions,
            lifecycle_last_finalized_sessions=lifecycle_last_finalized_sessions,
            message_sessions=message_sessions,
            node_sessions=node_sessions,
            lcm_any_sessions=lcm_any_sessions,
            lifecycle_referenced_sessions=lifecycle_referenced_sessions,
            state_sessions=state_sessions,
            state_db_read_success=state_db_read_success,
        )

        return stats

    @staticmethod
    def _classify_fragmentation(
        *,
        lifecycle_rows: int,
        lifecycle_current_sessions: set[str],
        lifecycle_last_finalized_sessions: set[str],
        message_sessions: set[str],
        node_sessions: set[str],
        lcm_any_sessions: set[str],
        lifecycle_referenced_sessions: set[str],
        state_sessions: set[str],
        state_db_read_success: bool,
    ) -> dict[str, Any]:
        """Bucket lifecycle mismatches into operator-readable read-only categories."""

        def sample(session_ids: set[str], limit: int = 5) -> list[str]:
            return sorted(session_ids)[:limit]

        categories: list[dict[str, Any]] = []

        def add_category(
            name: str,
            session_ids: set[str],
            *,
            severity: str,
            description: str,
            recommended_action: str,
        ) -> None:
            if not session_ids:
                return
            categories.append({
                "name": name,
                "severity": severity,
                "count": len(session_ids),
                "sample_session_ids": sample(session_ids),
                "description": description,
                "recommended_action": recommended_action,
            })

        add_category(
            "stale_lifecycle_current",
            lifecycle_current_sessions - lcm_any_sessions,
            severity="warn",
            description="Lifecycle current-session references that no longer have raw messages or summary nodes in LCM.",
            recommended_action="Inspect samples before cleanup; these are often old or ephemeral lifecycle rows, not automatic corruption.",
        )
        add_category(
            "stale_lifecycle_finalized",
            lifecycle_last_finalized_sessions - lcm_any_sessions,
            severity="warn",
            description="Lifecycle finalized-session references that no longer have raw messages or summary nodes in LCM.",
            recommended_action="Inspect samples before cleanup; only remove with an explicit backup-first lifecycle cleanup flow.",
        )
        if lifecycle_rows > 0:
            add_category(
                "lcm_message_sessions_without_lifecycle_reference",
                message_sessions - lifecycle_referenced_sessions,
                severity="notice",
                description="Raw-message sessions exist in LCM but are not referenced by current or finalized lifecycle state.",
                recommended_action="Usually safe as historical retained context; investigate only if the sessions should belong to an active conversation.",
            )
            add_category(
                "lcm_node_sessions_without_lifecycle_reference",
                node_sessions - lifecycle_referenced_sessions,
                severity="notice",
                description="Summary-node sessions exist in LCM but are not referenced by current or finalized lifecycle state.",
                recommended_action="Usually safe as historical retained context; verify expand/search still work before considering cleanup.",
            )

        if state_db_read_success:
            add_category(
                "lcm_message_sessions_missing_in_state",
                message_sessions - state_sessions,
                severity="notice",
                description="LCM raw-message sessions are absent from the Hermes session database.",
                recommended_action="Treat as retained or imported context unless the session should still be browsable in host session history.",
            )
            add_category(
                "lcm_node_sessions_missing_in_state",
                node_sessions - state_sessions,
                severity="notice",
                description="LCM summary-node sessions are absent from the Hermes session database.",
                recommended_action="Keep read-only; this can happen after host session pruning while LCM retained summaries remain useful.",
            )
            add_category(
                "state_only_sessions",
                state_sessions - lcm_any_sessions,
                severity="notice",
                description="Hermes host sessions exist without raw messages or summary nodes in LCM.",
                recommended_action="Usually benign for sessions outside LCM scope, ignored sessions, or sessions that never reached durable LCM ingest.",
            )

        warn_count = sum(1 for item in categories if item["severity"] == "warn")
        status = "warn" if warn_count else ("notice" if categories else "pass")
        summary = (
            "no lifecycle fragmentation categories detected"
            if not categories
            else f"{len(categories)} lifecycle fragmentation categories need review"
        )
        return {
            "read_only": True,
            "status": status,
            "summary": summary,
            "categories": categories,
        }

    @_synchronized
    def record_debt(
        self,
        conversation_id: str | None,
        *,
        kind: str,
        size_estimate: int,
    ) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET debt_kind = ?,
                debt_size_estimate = ?,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (kind, max(0, int(size_estimate or 0)), now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def clear_debt(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET debt_kind = NULL,
                debt_size_estimate = 0,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def record_maintenance_attempt(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET last_maintenance_attempt_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def record_reset(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET last_reset_at = ?,
                debt_kind = NULL,
                debt_size_estimate = 0,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    def _collect_empty_session_candidates(
        self,
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        max_age_hours: float | None = None,
        max_candidates: int = 100,
    ) -> list[tuple[str, str, str, float]]:
        """Find a bounded set of empty rows without holding the writer lock."""
        limit = max(0, int(max_candidates))
        if limit == 0:
            return []
        protected = sorted({str(item) for item in (protected_session_ids or ()) if item})
        uri = self.db_path.expanduser().resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only=ON")
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            predicates: list[str] = []
            params: list[Any] = []
            for column in ("current_session_id", "last_finalized_session_id"):
                if "messages" in tables:
                    predicates.append(
                        f"(COALESCE(state.{column}, '') = '' OR NOT EXISTS ("
                        f"SELECT 1 FROM messages WHERE session_id = state.{column}))"
                    )
                if "summary_nodes" in tables:
                    predicates.append(
                        f"(COALESCE(state.{column}, '') = '' OR NOT EXISTS ("
                        f"SELECT 1 FROM summary_nodes WHERE session_id = state.{column}))"
                    )
            if protected:
                placeholders = ", ".join("?" for _ in protected)
                predicates.extend((
                    f"COALESCE(state.current_session_id, '') NOT IN ({placeholders})",
                    f"COALESCE(state.last_finalized_session_id, '') NOT IN ({placeholders})",
                ))
                params.extend(protected)
                params.extend(protected)
            if max_age_hours is not None:
                predicates.append(
                    "COALESCE(state.current_bound_at, state.last_finalized_at, "
                    "state.updated_at, 0) <= ?"
                )
                params.append(time.time() - (float(max_age_hours) * 3600.0))
            params.append(limit)
            rows = conn.execute(
                "SELECT state.conversation_id, state.current_session_id, "
                "state.last_finalized_session_id, state.updated_at "
                "FROM lcm_lifecycle_state AS state WHERE "
                + (" AND ".join(predicates) if predicates else "1")
                + " ORDER BY state.updated_at, state.conversation_id LIMIT ?",
                params,
            ).fetchall()
            return [
                (
                    str(row["conversation_id"]),
                    str(row["current_session_id"] or ""),
                    str(row["last_finalized_session_id"] or ""),
                    float(row["updated_at"] or 0.0),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def _snapshot_sessions_with_data(self) -> set[str]:
        """Compatibility helper used by older diagnostics and tests."""
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            tables = {
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            sessions: set[str] = set()
            for table in ("messages", "summary_nodes"):
                if table in tables:
                    sessions.update(
                        str(row[0])
                        for row in conn.execute(
                            f"SELECT DISTINCT session_id FROM {table}"
                        ).fetchall()
                        if row[0]
                    )
            return sessions
        finally:
            conn.close()

    def prune_empty_sessions(
        self,
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        protected_session_ids_provider: Callable[[], Iterable[str]] | None = None,
        protection_revision_provider: Callable[[], int] | None = None,
        protection_commit_guard: Callable[[], ContextManager[None]] | None = None,
        max_age_hours: float | None = None,
        max_candidates: int | None = None,
    ) -> int:
        """Delete empty lifecycle rows using transactions bounded to 100 rows.

        The default preserves the legacy complete-cleanup contract by running
        multiple bounded transactions. An explicit limit requests one bounded
        pass and is clamped to the hard safety ceiling.
        """
        if max_candidates is not None:
            return self._prune_empty_sessions_batch(
                protected_session_ids=protected_session_ids,
                protected_session_ids_provider=protected_session_ids_provider,
                protection_revision_provider=protection_revision_provider,
                protection_commit_guard=protection_commit_guard,
                max_age_hours=max_age_hours,
                max_candidates=min(
                    _MAX_EMPTY_LIFECYCLE_GC_BATCH_SIZE,
                    max(0, int(max_candidates)),
                ),
            )

        deleted_total = 0
        while True:
            deleted = self._prune_empty_sessions_batch(
                protected_session_ids=protected_session_ids,
                protected_session_ids_provider=protected_session_ids_provider,
                protection_revision_provider=protection_revision_provider,
                protection_commit_guard=protection_commit_guard,
                max_age_hours=max_age_hours,
                max_candidates=_MAX_EMPTY_LIFECYCLE_GC_BATCH_SIZE,
            )
            deleted_total += deleted
            if deleted < _MAX_EMPTY_LIFECYCLE_GC_BATCH_SIZE:
                return deleted_total

    def _prune_empty_sessions_batch(
        self,
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None,
        protected_session_ids_provider: Callable[[], Iterable[str]] | None,
        protection_revision_provider: Callable[[], int] | None,
        protection_commit_guard: Callable[[], ContextManager[None]] | None,
        max_age_hours: float | None,
        max_candidates: int,
    ) -> int:
        """Delete one hard-bounded batch after read-only candidate discovery."""
        max_candidates = min(
            _MAX_EMPTY_LIFECYCLE_GC_BATCH_SIZE,
            max(0, int(max_candidates)),
        )
        protection_revision = (
            int(protection_revision_provider())
            if protection_revision_provider is not None
            else None
        )
        candidates = self._collect_empty_session_candidates(
            protected_session_ids=protected_session_ids,
            max_age_hours=max_age_hours,
            max_candidates=max_candidates,
        )
        if not candidates:
            return 0
        protected = {str(item) for item in (protected_session_ids or ()) if item}
        max_age_seconds = (
            float(max_age_hours) * 3600.0
            if max_age_hours is not None
            else None
        )
        conn = self._conn
        assert conn is not None
        with self._lock.attributed("prune_empty_sessions"):
            conn.execute("BEGIN IMMEDIATE")
            try:
                if protected_session_ids_provider is not None:
                    protected.update(
                        str(item)
                        for item in protected_session_ids_provider()
                        if item
                    )
                tables = {
                    str(row[0]) for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }

                def session_has_data(session_id: str) -> bool:
                    if not session_id:
                        return False
                    for table in ("messages", "summary_nodes"):
                        if table in tables and conn.execute(
                            f"SELECT 1 FROM {table} WHERE session_id = ? LIMIT 1",
                            (session_id,),
                        ).fetchone():
                            return True
                    return False

                def session_is_protected(current: str | None, finalized: str | None) -> int:
                    live = set(protected)
                    if protected_session_ids_provider is not None:
                        live.update(
                            str(item)
                            for item in protected_session_ids_provider()
                            if item
                        )
                    return int(bool({str(item) for item in (current, finalized) if item} & live))

                conn.create_function(
                    "lcm_gc_session_is_protected",
                    2,
                    session_is_protected,
                )

                deleted = 0
                now = time.time()
                for conversation_id, snapshot_cur, snapshot_fin, snapshot_updated_at in candidates:
                    if protected_session_ids_provider is not None:
                        protected.update(
                            str(item)
                            for item in protected_session_ids_provider()
                            if item
                        )
                    row = conn.execute(
                        "SELECT * FROM lcm_lifecycle_state WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    cur = str(row["current_session_id"] or "")
                    fin = str(row["last_finalized_session_id"] or "")
                    updated_at = float(row["updated_at"] or 0.0)
                    if (cur, fin, updated_at) != (
                        snapshot_cur,
                        snapshot_fin,
                        snapshot_updated_at,
                    ):
                        continue
                    if {ref for ref in (cur, fin) if ref} & protected:
                        continue
                    if max_age_seconds is not None:
                        row_age = (
                            row["current_bound_at"]
                            or row["last_finalized_at"]
                            or row["updated_at"]
                        )
                        if row_age is not None and (now - float(row_age)) < max_age_seconds:
                            continue
                    if session_has_data(cur) or session_has_data(fin):
                        continue
                    cursor = conn.execute(
                        """
                        DELETE FROM lcm_lifecycle_state
                        WHERE conversation_id = ?
                          AND lcm_gc_session_is_protected(
                              current_session_id,
                              last_finalized_session_id
                          ) = 0
                        """,
                        (conversation_id,),
                    )
                    deleted += max(0, int(cursor.rowcount or 0))
                if deleted:
                    commit_guard = (
                        protection_commit_guard()
                        if protection_commit_guard is not None
                        else nullcontext()
                    )
                    with commit_guard:
                        if (
                            protection_revision is not None
                            and protection_revision_provider is not None
                            and int(protection_revision_provider()) != protection_revision
                        ):
                            conn.rollback()
                            return 0
                        conn.commit()
                else:
                    conn.rollback()
                return deleted
            except Exception:
                conn.rollback()
                raise

    @_synchronized
    def delete_safe_rows_for_sessions(
        self,
        session_ids: set[str] | list[str] | tuple[str, ...],
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> tuple[int, int]:
        candidates = {str(s) for s in session_ids if s}
        if not candidates:
            return 0, 0
        protected = {str(s) for s in (protected_session_ids or ()) if s}
        deleted = 0
        skipped = 0
        rows = self._conn.execute("SELECT * FROM lcm_lifecycle_state").fetchall()
        for row in rows:
            refs = {
                str(value)
                for value in (row["current_session_id"], row["last_finalized_session_id"])
                if value
            }
            if not refs or not (refs & candidates):
                continue
            if refs & protected:
                skipped += 1
                continue
            if refs <= candidates:
                self._conn.execute(
                    "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
                    (row["conversation_id"],),
                )
                deleted += 1
                continue
            skipped += 1
        if deleted:
            self._conn.commit()
        return deleted, skipped

    def advance_frontier(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int,
    ) -> LifecycleState | None:
        if not conversation_id:
            return None
        with self._lock:
            state = self.get_by_conversation(conversation_id)
            if state is None or state.current_session_id != session_id:
                return state
            now = time.time()
            conn = self._conn
            assert conn is not None
            # MAX() in SQL keeps the advance monotonic even if a concurrent
            # writer bumped the frontier between the read above and this write.
            # A Python-side max() over the stale read could otherwise regress
            # the checkpoint and force the same range to be compacted twice.
            cursor = conn.execute(
                """
                UPDATE lcm_lifecycle_state
                SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                    updated_at = ?
                WHERE conversation_id = ? AND current_session_id = ?
                """,
                (int(frontier_store_id or 0), now, conversation_id, session_id),
            )
            if cursor.rowcount == 0:
                return self.get_by_conversation(conversation_id)
            conn.commit()
            return self.get_by_conversation(conversation_id)
