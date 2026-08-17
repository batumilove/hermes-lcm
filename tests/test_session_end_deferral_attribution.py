import logging
import re
import sqlite3
import threading
from contextlib import contextmanager

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.session_end_pending import (
    load_session_end_intent,
    pending_session_end_dir,
)


def _engine(tmp_path, name):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"{name}.db")))
    engine.on_session_start(f"{name}-session", platform="telegram")
    return engine


def _event_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if "event=session_end_deferred" in record.getMessage()
    ]


def test_structured_deferral_event_is_single_line_and_bounded(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine_module._log_session_end_deferred_event(
            logging.WARNING,
            stage="raw\ningest" + "s" * 1000,
            outcome="scheduled" + "o" * 1000,
            operation="session_end_raw_message_ingest" + "p" * 1000,
            intent_sha256="d" * 5000,
            receipt_id="r" * 5000,
            session_id="session\n" + "x" * 5000,
            conversation_id="conversation\r" + "y" * 5000,
            wait_seconds=0.1,
            error=RuntimeError("locked\n" + "z" * 5000),
        )

    events = _event_messages(caplog)
    assert len(events) == 1
    assert "\n" not in events[0]
    assert "\r" not in events[0]
    assert len(events[0]) <= 2048


def test_structured_deferral_event_redacts_arbitrary_exception_payload(caplog):
    secret_marker = "TOP_SECRET_USER_TEXT"
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine_module._log_session_end_deferred_event(
            logging.WARNING,
            stage="bounded_flush",
            outcome="scheduled",
            operation="session_end_ingest_finalize",
            intent_sha256="d" * 64,
            session_id="session-id",
            conversation_id="conversation-id",
            wait_seconds=0.1,
            error=RuntimeError(
                "database is locked: owner_operation=compress_worker "
                f"payload={secret_marker}"
            ),
        )

    events = _event_messages(caplog)
    assert len(events) == 1
    assert "error_type=RuntimeError" in events[0]
    assert "owner_operation=compress_worker" in events[0]
    assert secret_marker not in events[0]
    assert "payload=" not in events[0]


def test_bounded_flush_deferral_logs_exact_durable_intent_identity(
    tmp_path, monkeypatch, caplog
):
    engine = _engine(tmp_path, "bounded-deferral-attribution")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    scheduled = []

    @contextmanager
    def locked_before_flush(*args, **kwargs):
        del args, kwargs
        raise sqlite3.OperationalError(
            "database is locked: timed out waiting for process-wide SQLite writer; "
            "owner_thread_id=42 owner_thread_name=compress-worker "
            "owner_operation=wrapper owner_age_s=7.332"
        )
        yield  # pragma: no cover

    monkeypatch.setattr(engine_module, "_temporary_sqlite_busy_timeout", locked_before_flush)
    monkeypatch.setattr(engine, "_schedule_session_end_drain", lambda: scheduled.append(True))

    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
            engine.on_session_end(
                session_id,
                [{"role": "user", "content": "durable deferred turn"}],
            )

        pending = list(pending_session_end_dir(engine._store.db_path).glob("*.json"))
        assert len(pending) == 1
        intent = load_session_end_intent(pending[0])
        events = _event_messages(caplog)
        assert scheduled == [True]
        assert len(events) == 1
        event = events[0]
        assert "stage=bounded_flush" in event
        assert "outcome=scheduled" in event
        assert "operation=session_end_ingest_finalize" in event
        assert f"intent_sha256={intent['intent_sha256']}" in event
        assert f"session_id={session_id!r}" in event
        assert f"conversation_id={conversation_id!r}" in event
        assert re.search(r"\bwait_s=\d+\.\d{3}\b", event)
        assert "owner_operation=wrapper" in event
    finally:
        engine.shutdown()


def test_successful_drain_logs_exact_receipt_pairing(tmp_path, caplog):
    engine = _engine(tmp_path, "drain-receipt-attribution")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    messages = [{"role": "user", "content": "settled exactly once"}]
    path = engine._persist_session_end_intent(session_id, messages, ingest_cursor=0)
    intent = load_session_end_intent(path)
    digest = intent["intent_sha256"]

    try:
        with caplog.at_level(logging.INFO, logger="hermes_lcm.engine"):
            engine._drain_one_session_end_intent(path)

        assert not path.exists()
        assert engine._store.has_session_end_ingest_receipt(digest)
        events = _event_messages(caplog)
        assert len(events) == 1
        event = events[0]
        assert "stage=drain" in event
        assert "outcome=settled" in event
        assert "operation=session_end_deferred_drain" in event
        assert f"intent_sha256={digest}" in event
        assert f"receipt_id={digest}" in event
        assert f"session_id={session_id!r}" in event
        assert f"conversation_id={conversation_id!r}" in event
    finally:
        engine.shutdown()


def test_post_receipt_failure_logs_retry_pending_then_settles_without_duplicate(
    tmp_path, monkeypatch, caplog
):
    engine = _engine(tmp_path, "post-receipt-unresolved")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    messages = [{"role": "user", "content": "do not duplicate me"}]
    path = engine._persist_session_end_intent(session_id, messages, ingest_cursor=0)
    intent = load_session_end_intent(path)
    digest = intent["intent_sha256"]
    original_finalize = engine._lifecycle.finalize_session

    def fail_finalize(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("lifecycle unavailable after receipt commit")

    monkeypatch.setattr(engine._lifecycle, "finalize_session", fail_finalize)
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm.engine"):
            with pytest.raises(RuntimeError, match="lifecycle unavailable"):
                engine._drain_one_session_end_intent(path)

        assert path.exists()
        assert engine._store.has_session_end_ingest_receipt(digest)
        events = _event_messages(caplog)
        assert len(events) == 1
        event = events[0]
        assert "stage=drain" in event
        assert "outcome=retry_pending" in event
        assert "operation=session_end_deferred_drain" in event
        assert "error_type=RuntimeError" in event
        assert f"intent_sha256={digest}" in event
        assert f"receipt_id={digest}" in event
        assert f"session_id={session_id!r}" in event
        assert f"conversation_id={conversation_id!r}" in event

        monkeypatch.setattr(engine._lifecycle, "finalize_session", original_finalize)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="hermes_lcm.engine"):
            engine._drain_one_session_end_intent(path)

        persisted = engine._store.get_session_messages(session_id)
        assert [message.get("content") for message in persisted] == ["do not duplicate me"]
        assert not path.exists()
        retry_events = _event_messages(caplog)
        assert len(retry_events) == 1
        assert "outcome=settled" in retry_events[0]
        assert f"receipt_id={digest}" in retry_events[0]
    finally:
        monkeypatch.setattr(engine._lifecycle, "finalize_session", original_finalize)
        engine.shutdown()


def test_bounded_drain_exhaustion_logs_one_exact_unresolved_outcome(
    tmp_path, monkeypatch, caplog
):
    engine = _engine(tmp_path, "bounded-drain-exhaustion")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    path = engine._persist_session_end_intent(
        session_id,
        [{"role": "user", "content": "remain durable while unresolved"}],
        ingest_cursor=0,
    )
    digest = load_session_end_intent(path)["intent_sha256"]

    def fail_finalize(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("lifecycle remains unavailable")

    monkeypatch.setattr(engine._lifecycle, "finalize_session", fail_finalize)
    monkeypatch.setattr(
        engine_module, "_SESSION_END_DEFERRED_RETRY_BUDGET_SECONDS", 0.03
    )
    monkeypatch.setattr(
        engine_module, "_SESSION_END_DEFERRED_RETRY_INTERVAL_SECONDS", 0.005
    )
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm.engine"):
            engine._schedule_session_end_drain()
            assert engine._session_end_drain_done.wait(timeout=2.0)

        events = _event_messages(caplog)
        retry_events = [event for event in events if "outcome=retry_pending" in event]
        unresolved_events = [event for event in events if "outcome=unresolved" in event]
        assert retry_events
        assert len(unresolved_events) == 1
        event = unresolved_events[0]
        assert f"intent_sha256={digest}" in event
        assert f"receipt_id={digest}" in event
        assert f"session_id={session_id!r}" in event
        assert f"conversation_id={conversation_id!r}" in event
        assert path.exists()
    finally:
        engine.shutdown()


def test_raw_ingest_lock_deferral_logs_exact_intent_identity(
    tmp_path, monkeypatch, caplog
):
    engine = _engine(tmp_path, "raw-ingest-deferral")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    scheduled = []

    def fail_ingest(*args, **kwargs):
        del args, kwargs
        raise sqlite3.OperationalError("database is locked during raw ingest")

    monkeypatch.setattr(engine, "_ingest_messages", fail_ingest)
    monkeypatch.setattr(engine, "_schedule_session_end_drain", lambda: scheduled.append(True))
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
            engine.on_session_end(
                session_id,
                [{"role": "user", "content": "raw ingest deferred"}],
            )

        path = next(pending_session_end_dir(engine._store.db_path).glob("*.json"))
        digest = load_session_end_intent(path)["intent_sha256"]
        events = _event_messages(caplog)
        assert scheduled == [True]
        assert len(events) == 1
        assert "stage=raw_ingest" in events[0]
        assert "operation=session_end_raw_message_ingest" in events[0]
        assert "outcome=scheduled" in events[0]
        assert f"intent_sha256={digest}" in events[0]
        assert "receipt_id=-" in events[0]
        assert f"session_id={session_id!r}" in events[0]
        assert f"conversation_id={conversation_id!r}" in events[0]
    finally:
        engine.shutdown()


def test_lifecycle_lock_deferral_logs_exact_receipt_pairing(
    tmp_path, monkeypatch, caplog
):
    engine = _engine(tmp_path, "lifecycle-deferral")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    scheduled = []

    def fail_finalize(*args, **kwargs):
        del args, kwargs
        raise sqlite3.OperationalError("database is locked during lifecycle finalize")

    monkeypatch.setattr(engine._lifecycle, "finalize_session", fail_finalize)
    monkeypatch.setattr(engine, "_schedule_session_end_drain", lambda: scheduled.append(True))
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
            engine.on_session_end(
                session_id,
                [{"role": "user", "content": "receipt already committed"}],
            )

        path = next(pending_session_end_dir(engine._store.db_path).glob("*.json"))
        digest = load_session_end_intent(path)["intent_sha256"]
        assert engine._store.has_session_end_ingest_receipt(digest)
        events = _event_messages(caplog)
        assert scheduled == [True]
        assert len(events) == 1
        assert "stage=lifecycle_finalize" in events[0]
        assert "operation=session_end_lifecycle_finalize" in events[0]
        assert "outcome=scheduled" in events[0]
        assert f"intent_sha256={digest}" in events[0]
        assert f"receipt_id={digest}" in events[0]
        assert f"session_id={session_id!r}" in events[0]
        assert f"conversation_id={conversation_id!r}" in events[0]
    finally:
        engine.shutdown()


def test_session_end_writer_owner_reports_exact_active_phase(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "writer-phase-attribution")
    ingest_entered = threading.Event()
    release_ingest = threading.Event()
    lifecycle_entered = threading.Event()
    release_lifecycle = threading.Event()
    original_finalize = engine._lifecycle.finalize_session

    def hold_ingest(*args, **kwargs):
        del args, kwargs
        ingest_entered.set()
        assert release_ingest.wait(timeout=2.0)

    def hold_finalize(*args, **kwargs):
        lifecycle_entered.set()
        assert release_lifecycle.wait(timeout=2.0)
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(engine, "_ingest_messages", hold_ingest)
    monkeypatch.setattr(engine._lifecycle, "finalize_session", hold_finalize)
    worker = threading.Thread(
        target=engine.on_session_end,
        args=(
            engine._session_id,
            [{"role": "user", "content": "attribute each writer phase"}],
        ),
        name="session-end-phase-probe",
    )
    worker.start()
    try:
        assert ingest_entered.wait(timeout=1.0)
        ingest_owner = engine._store._write_lock.owner_snapshot()
        assert ingest_owner["operation"] == "session_end_raw_message_ingest"
        assert ingest_owner["operation_age_seconds"] >= 0.0
        release_ingest.set()

        assert lifecycle_entered.wait(timeout=1.0)
        lifecycle_owner = engine._store._write_lock.owner_snapshot()
        assert lifecycle_owner["operation"] == "session_end_lifecycle_finalize"
        assert lifecycle_owner["operation_age_seconds"] >= 0.0
    finally:
        release_ingest.set()
        release_lifecycle.set()
        worker.join(timeout=2.0)
        engine.shutdown()

    assert not worker.is_alive()
