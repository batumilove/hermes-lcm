"""Connection-level RED for session-end after coordinator acquisition.

This fixture deliberately lets a real write transaction on the engine's DAG
connection outlive the coordinator phase that would normally contain it.  The
session-end thread can therefore acquire the process coordinator while SQLite
still has a same-process writer on another LCM connection.  That is the narrow
production signature this test protects: no external process, no coordinator
owner to wait for, then SQLITE_BUSY inside ``session_end_ingest_finalize``.

The expected contract is stronger than eventual replay: transient internal
connection contention must not emit a hard ``database is locked`` marker.
Message/receipt durability, sidecar settlement, and lifecycle finalization are
reported separately so a durable message cannot mask an unfinished lifecycle.
"""
from __future__ import annotations

import threading
import time

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.session_end_pending import iter_session_end_intents


def test_session_end_settles_transient_dag_writer_without_hard_lock_marker(
    tmp_path, caplog, monkeypatch
):
    db_path = tmp_path / "connection-contention.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    engine.on_session_start("connection-contention-session", platform="telegram")
    session_id = engine._session_id
    conversation_id = engine._conversation_id

    # Warm imports, ingest protection, and the message write path before timing
    # the deliberate overlap.
    engine._ingest_messages([{"role": "user", "content": "warm-up"}])

    writer_started = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[BaseException] = []

    def hold_dag_write_transaction() -> None:
        conn = engine._dag.connection
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                ("connection-contention-holder", "active"),
            )
            writer_started.set()
            if not release_writer.wait(timeout=5.0):
                raise TimeoutError("test DAG writer was not released")
            conn.rollback()
        except BaseException as exc:  # surfaced in the test thread below
            writer_errors.append(exc)
            writer_started.set()
            try:
                conn.rollback()
            except BaseException:
                pass

    holder = threading.Thread(
        target=hold_dag_write_transaction,
        name="lcm-test-dag-connection-holder",
    )
    holder.start()
    assert writer_started.wait(timeout=3.0)
    assert writer_errors == []

    # Prove the database writer is invisible to the process coordinator: the
    # session-end path can acquire this same lock even while BEGIN IMMEDIATE is
    # live on the DAG connection.
    with engine._store._write_lock.attributed("connection_contention_red_probe"):
        assert engine._store._write_lock.owner_snapshot()["operation"] == (
            "connection_contention_red_probe"
        )
    assert engine._store._write_lock.owner_snapshot() == {}

    # Keep the SQLite writer longer than the current 50 ms busy timeout but
    # shorter than the existing 2.5 s append-overlap budget.  This is not a
    # timeout-increase test; it asks session-end to settle within its current
    # bounded recovery budget.
    def delayed_release() -> None:
        time.sleep(1.0)
        release_writer.set()

    releaser = threading.Thread(target=delayed_release, name="lcm-test-writer-release")
    releaser.start()
    monkeypatch.setattr(
        lcm_engine, "_SESSION_END_DEFERRED_RETRY_INTERVAL_SECONDS", 0.01
    )

    immediate_pending = False
    immediate_finalized = False
    try:
        with caplog.at_level("WARNING", logger="hermes_lcm.engine"):
            engine.on_session_end(
                session_id,
                [
                    {"role": "user", "content": "warm-up"},
                    {"role": "assistant", "content": "final durable turn"},
                ],
            )
        immediate_pending = bool(tuple(iter_session_end_intents(db_path)))
        immediate_state = engine._lifecycle.get_by_conversation(conversation_id)
        immediate_finalized = bool(
            immediate_state
            and immediate_state.last_finalized_session_id == session_id
        )
    finally:
        release_writer.set()
        releaser.join(timeout=3.0)
        holder.join(timeout=3.0)

    assert not holder.is_alive()
    assert not releaser.is_alive()
    assert writer_errors == []
    assert engine._session_end_drain_done.wait(timeout=5.0)

    messages = engine._store.get_session_messages(session_id)
    durable_contents = [message.get("content") for message in messages]
    receipt_count = engine._store.connection.execute(
        "SELECT COUNT(*) FROM lcm_session_end_ingest_receipts WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    pending_after_drain = tuple(iter_session_end_intents(db_path))
    final_state = engine._lifecycle.get_by_conversation(conversation_id)
    finalized_after_drain = bool(
        final_state and final_state.last_finalized_session_id == session_id
    )
    hard_markers = [
        record.getMessage()
        for record in caplog.records
        if "database is locked" in record.getMessage().lower()
        or "sqlite lock" in record.getMessage().lower()
        or "timed out waiting for process-wide sqlite writer"
        in record.getMessage().lower()
    ]

    engine.shutdown()

    evidence = {
        "hard_markers": hard_markers,
        "immediate_pending_sidecar": immediate_pending,
        "immediate_lifecycle_finalized": immediate_finalized,
        "durable_contents_after_drain": durable_contents,
        "receipt_count_after_drain": receipt_count,
        "pending_sidecars_after_drain": [path.name for path in pending_after_drain],
        "lifecycle_finalized_after_drain": finalized_after_drain,
    }
    assert hard_markers == [], evidence
    assert durable_contents == ["warm-up", "final durable turn"], evidence
    assert receipt_count == 1, evidence
    assert pending_after_drain == (), evidence
    assert finalized_after_drain, evidence


def test_session_end_settles_transient_dag_writer_before_lifecycle_finalize(
    tmp_path, caplog, monkeypatch
):
    """A committed receipt must not mask a transient lifecycle SQLITE_BUSY."""
    db_path = tmp_path / "lifecycle-connection-contention.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    engine.on_session_start("lifecycle-contention-session", platform="telegram")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    original_ingest = engine._ingest_messages
    writer_started = threading.Event()
    writer_errors: list[BaseException] = []
    armed = True

    def release_dag_transaction() -> None:
        if not writer_started.wait(timeout=3.0):
            writer_errors.append(TimeoutError("lifecycle DAG writer never started"))
            return
        time.sleep(1.0)
        try:
            engine._dag.connection.rollback()
        except BaseException as exc:
            writer_errors.append(exc)

    def ingest_then_hold_dag_writer(*args, **kwargs):
        nonlocal armed
        result = original_ingest(*args, **kwargs)
        if armed:
            armed = False
            conn = engine._dag.connection
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                ("lifecycle-contention-holder", "active"),
            )
            writer_started.set()
        return result

    monkeypatch.setattr(engine, "_ingest_messages", ingest_then_hold_dag_writer)
    monkeypatch.setattr(
        lcm_engine, "_SESSION_END_DEFERRED_RETRY_INTERVAL_SECONDS", 0.01
    )
    releaser = threading.Thread(
        target=release_dag_transaction,
        name="lcm-test-lifecycle-writer-release",
    )
    releaser.start()

    try:
        with caplog.at_level("WARNING", logger="hermes_lcm.engine"):
            engine.on_session_end(
                session_id,
                [{"role": "assistant", "content": "receipt before lifecycle"}],
            )
        assert writer_started.wait(timeout=3.0)
        immediate_pending = bool(tuple(iter_session_end_intents(db_path)))
        immediate_state = engine._lifecycle.get_by_conversation(conversation_id)
        immediate_finalized = bool(
            immediate_state
            and immediate_state.last_finalized_session_id == session_id
        )
    finally:
        releaser.join(timeout=3.0)

    assert not releaser.is_alive()
    assert writer_errors == []
    assert engine._session_end_drain_done.wait(timeout=5.0)

    durable_contents = [
        message.get("content")
        for message in engine._store.get_session_messages(session_id)
    ]
    receipt_count = engine._store.connection.execute(
        "SELECT COUNT(*) FROM lcm_session_end_ingest_receipts WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    pending_after_drain = tuple(iter_session_end_intents(db_path))
    final_state = engine._lifecycle.get_by_conversation(conversation_id)
    finalized_after_drain = bool(
        final_state and final_state.last_finalized_session_id == session_id
    )
    hard_markers = [
        record.getMessage()
        for record in caplog.records
        if "database is locked" in record.getMessage().lower()
        or "sqlite lock" in record.getMessage().lower()
    ]
    engine.shutdown()

    evidence = {
        "hard_markers": hard_markers,
        "immediate_pending_sidecar": immediate_pending,
        "immediate_lifecycle_finalized": immediate_finalized,
        "durable_contents_after_drain": durable_contents,
        "receipt_count_after_drain": receipt_count,
        "pending_sidecars_after_drain": [path.name for path in pending_after_drain],
        "lifecycle_finalized_after_drain": finalized_after_drain,
    }
    assert hard_markers == [], evidence
    assert not immediate_pending, evidence
    assert immediate_finalized, evidence
    assert durable_contents == ["receipt before lifecycle"], evidence
    assert receipt_count == 1, evidence
    assert pending_after_drain == (), evidence
    assert finalized_after_drain, evidence
