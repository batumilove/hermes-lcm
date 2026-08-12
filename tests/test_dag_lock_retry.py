"""Regression tests for transient SQLite contention in SummaryDAG writes."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.compaction import CompactionMixin
from hermes_lcm.dag import SummaryDAG, SummaryNode


class _LockingConnection:
    """Delegate to a real connection but fail the first node INSERTs."""

    def __init__(self, connection: sqlite3.Connection, failures: int):
        self._connection = connection
        self._remaining = failures
        self.rollback_calls = 0

    def execute(self, sql, parameters=()):
        if sql.lstrip().startswith("INSERT INTO summary_nodes") and self._remaining:
            self._remaining -= 1
            raise sqlite3.OperationalError("database is locked")
        return self._connection.execute(sql, parameters)

    def commit(self):
        return self._connection.commit()

    def rollback(self):
        self.rollback_calls += 1
        return self._connection.rollback()

    def __getattr__(self, name):
        return getattr(self._connection, name)


def _node() -> SummaryNode:
    return SummaryNode(
        session_id="session-a",
        depth=0,
        summary="summary",
        token_count=3,
        source_token_count=10,
        source_ids=[1, 2],
        source_type="messages",
    )


def test_add_node_retries_transient_lock_and_rolls_back(tmp_path, monkeypatch):
    dag = SummaryDAG(tmp_path / "lcm.db")
    wrapped = _LockingConnection(dag._conn, failures=2)
    dag._conn = wrapped
    monkeypatch.setattr("hermes_lcm.dag.time.sleep", lambda _seconds: None)

    try:
        node_id = dag.add_node(_node())

        assert node_id > 0
        assert wrapped.rollback_calls == 2
        row = wrapped.execute(
            "SELECT summary FROM summary_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        assert row == ("summary",)
    finally:
        dag.close()


def test_add_node_recovers_from_real_multiwriter_contention(tmp_path, monkeypatch):
    db_path = tmp_path / "lcm.db"
    dag = SummaryDAG(db_path)
    competing_writer = sqlite3.connect(str(db_path), timeout=1.0)
    competing_writer.execute("BEGIN IMMEDIATE")
    competing_writer.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('holder', 'active')"
    )
    released = False

    def release_during_backoff(_seconds):
        nonlocal released
        competing_writer.commit()
        released = True

    monkeypatch.setattr("hermes_lcm.dag.time.sleep", release_during_backoff)

    try:
        node_id = dag.add_node(_node())
        assert released is True
        assert node_id > 0
    finally:
        competing_writer.close()
        dag.close()


def test_add_node_does_not_retry_unrelated_sqlite_errors(tmp_path, monkeypatch):
    dag = SummaryDAG(tmp_path / "lcm.db")
    wrapped = _LockingConnection(dag._conn, failures=0)
    dag._conn = wrapped

    def fail_unrelated(sql, parameters=()):
        if sql.lstrip().startswith("INSERT INTO summary_nodes"):
            raise sqlite3.OperationalError("disk I/O error")
        return wrapped._connection.execute(sql, parameters)

    monkeypatch.setattr(wrapped, "execute", fail_unrelated)

    try:
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            dag.add_node(_node())
        assert wrapped.rollback_calls == 0
    finally:
        dag.close()


def test_compress_defers_after_lock_retries_are_exhausted():
    class LockedCompactor:
        _last_compression_status = "running"
        _last_compression_noop_reason = ""

        def _compress_impl(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("database table is locked")

    compactor = LockedCompactor()
    messages = [{"role": "user", "content": "keep this turn alive"}]

    result = CompactionMixin.compress(compactor, messages)

    assert result is messages
    assert compactor._last_compression_status == "deferred"
    assert compactor._last_compression_noop_reason == "sqlite lock contention"
