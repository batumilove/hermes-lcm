"""Protected RED contracts for MessageStore transaction cleanup."""
from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.store import MessageStore


class _FailFirstCommitConnection:
    """Delegate SQL to a real connection but inject one commit failure."""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection
        self._fail_commit = True

    def execute(self, *args, **kwargs):
        return self._connection.execute(*args, **kwargs)

    def commit(self) -> None:
        if self._fail_commit:
            self._fail_commit = False
            raise sqlite3.OperationalError("injected commit failure")
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.rollback()
            return False
        try:
            self.commit()
        except BaseException:
            self.rollback()
            raise
        return False

    def __getattr__(self, name):
        return getattr(self._connection, name)


@pytest.mark.parametrize(
    "operation",
    [
        "append",
        "reassign_session_messages",
        "delete_session_messages",
        "gc_externalized_tool_result",
        "pin",
        "unpin",
        "write_metadata_json",
    ],
)
def test_manual_commit_failure_rolls_back_before_releasing_process_writer_gate(
    tmp_path, operation
):
    """A failed commit must never leave a hidden write transaction behind.

    A leaked transaction outlives the process writer-lock context and blocks all
    other MessageStore/SummaryDAG/LifecycleStore connections. Production showed
    this exact aftermath: no current process writer owner, no external holder,
    yet repeated same-process SQLite lock failures.
    """
    store = MessageStore(tmp_path / f"transaction-cleanup-{operation}.db")
    seed_store_id = store.append(
        "transaction-cleanup",
        {"role": "tool", "content": "seed", "tool_call_id": "seed-call"},
    )
    real_connection = store._conn
    assert real_connection is not None
    store._conn = _FailFirstCommitConnection(real_connection)
    try:
        with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
            if operation == "append":
                store.append(
                    "transaction-cleanup",
                    {"role": "user", "content": "must roll back"},
                )
            elif operation == "reassign_session_messages":
                store.reassign_session_messages("transaction-cleanup", "reassigned")
            elif operation == "delete_session_messages":
                store.delete_session_messages("transaction-cleanup")
            elif operation == "gc_externalized_tool_result":
                store.gc_externalized_tool_result(seed_store_id, "[GC placeholder]")
            elif operation == "pin":
                store.pin(seed_store_id)
            elif operation == "unpin":
                store.unpin(seed_store_id)
            else:
                store.write_metadata_json(
                    ["transaction-cleanup"],
                    '{"state":"must roll back"}',
                )

        assert real_connection.in_transaction is False
        # A subsequent writer on the same DB must proceed immediately and leave
        # exactly one committed row, proving the failed write did not leak.
        peer = MessageStore(store.db_path)
        try:
            peer.append(
                "peer-writer",
                {"role": "assistant", "content": "writer after rollback"},
            )
            rows = peer.get_session_messages("peer-writer")
            assert [row["content"] for row in rows] == ["writer after rollback"]
        finally:
            peer.close()
    finally:
        store._conn = real_connection
        if real_connection.in_transaction:
            real_connection.rollback()
        store.close()
