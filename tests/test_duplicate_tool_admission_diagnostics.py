"""Bounded, behavior-neutral diagnostics for exact duplicate tool admission."""

import hashlib
import json
import logging

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


EVENT_PREFIX = "LCM_DUPLICATE_TOOL_ADMISSION_DIAGNOSTIC "


def _tool(call_id: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "tool_name": "inspect",
        "content": content,
    }


def test_exact_duplicate_reaching_storage_admission_emits_bounded_receipt_without_filtering(
    tmp_path, monkeypatch, caplog
):
    db_path = tmp_path / "duplicate-admission.db"
    config = LCMConfig(database_path=str(db_path))
    session_id = "sensitive-session-id"
    conversation_id = "sensitive-conversation-id"
    call_id = "call_sensitive_exact_duplicate"
    content = "sensitive exact result body"

    seed = LCMEngine(config=config)
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    seed._ingest_messages([_tool(call_id, content)])
    seed.shutdown()

    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    # Model a missed earlier replay scan. The admission diagnostic must be an
    # independent observer and must not turn into another behavior filter.
    monkeypatch.setattr(
        engine,
        "_find_tool_anchored_replay_indexes",
        lambda *_args, **_kwargs: (set(), 0),
    )

    def fail_if_git_is_probed(_root):
        raise AssertionError("diagnostic admission path must not probe git")

    monkeypatch.setattr(engine_module, "_git_runtime_identity", fail_if_git_is_probed)

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine._ingest_messages([_tool(call_id, content)])

    rows = engine._store.get_session_messages(session_id)
    engine.shutdown()

    events = [
        json.loads(record.message[len(EVENT_PREFIX) :])
        for record in caplog.records
        if record.message.startswith(EVENT_PREFIX)
    ]
    assert len(rows) == 2, "instrumentation must not alter storage behavior"
    assert len(events) == 1
    event = events[0]
    assert event["schema"] == "lcm_duplicate_tool_admission_v1"
    assert event["duplicate_count"] == 1
    assert event["incoming_count"] == 1
    assert event["cursor"] == 0
    assert event["cursor_before_reconcile"] == 0
    assert event["reconcile_requested"] is False
    assert event["session_end"] is False
    assert event["duplicates"] == [
        {
            "incoming_index": 0,
            "tool_call_id_sha256": hashlib.sha256(call_id.encode()).hexdigest(),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "durable_store_ids": [1],
        }
    ]
    serialized = json.dumps(event, sort_keys=True)
    assert len(serialized) < 4096
    for secret in (session_id, conversation_id, call_id, content):
        assert secret not in serialized


def test_new_standalone_tool_content_with_reused_id_stays_silent_and_persists(
    tmp_path, caplog
):
    config = LCMConfig(database_path=str(tmp_path / "changed-content.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "changed-session",
        platform="telegram",
        conversation_id="changed-conversation",
        context_length=200000,
    )
    engine._ingest_messages([_tool("call_reused", "first result")])
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine._ingest_messages([_tool("call_reused", "different result")])

    rows = engine._store.get_session_messages("changed-session")
    engine.shutdown()

    assert [row["content"] for row in rows] == ["first result", "different result"]
    assert EVENT_PREFIX not in caplog.text


def test_receipt_is_hard_bounded_with_an_adversarial_engine_class_name(
    tmp_path, monkeypatch, caplog
):
    config = LCMConfig(database_path=str(tmp_path / "bounded-receipt.db"))
    engine_type = type("X" * 10000, (LCMEngine,), {})
    engine = engine_type(config=config)
    engine.on_session_start("bounded-session", context_length=200000)
    engine._ingest_messages([_tool("call_bounded", "same result")])
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    monkeypatch.setattr(
        engine,
        "_find_tool_anchored_replay_indexes",
        lambda *_args, **_kwargs: (set(), 0),
    )

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine._ingest_messages([_tool("call_bounded", "same result")])

    engine.shutdown()
    records = [record.message for record in caplog.records if record.message.startswith(EVENT_PREFIX)]
    assert len(records) == 1
    assert len(records[0].encode("utf-8")) <= 4096
    assert json.loads(records[0][len(EVENT_PREFIX) :])["receipt_truncated"] is True


def test_failing_diagnostic_logging_handler_does_not_block_storage(tmp_path, monkeypatch):
    class FailingHandler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if EVENT_PREFIX in message or message.startswith(
                "LCM duplicate-tool admission diagnostic failed:"
            ):
                raise RuntimeError("diagnostic sink unavailable")

    config = LCMConfig(database_path=str(tmp_path / "failing-handler.db"))
    session_id = "failing-handler-session"
    engine = LCMEngine(config=config)
    engine.on_session_start(session_id, context_length=200000)
    engine._ingest_messages([_tool("call_handler", "same result")])
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    monkeypatch.setattr(
        engine,
        "_find_tool_anchored_replay_indexes",
        lambda *_args, **_kwargs: (set(), 0),
    )
    handler = FailingHandler()
    logger = logging.getLogger("hermes_lcm.engine")
    prior_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        engine._ingest_messages([_tool("call_handler", "same result")])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)

    rows = engine._store.get_session_messages(session_id)
    engine.shutdown()
    assert len(rows) == 2


def test_diagnostic_store_lookup_is_indexed_and_result_bounded(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "bounded-lookup.db"))
    engine = LCMEngine(config=config)
    engine.on_session_start("lookup-session", context_length=200000)
    for idx in range(100):
        engine._store.append(
            "lookup-session",
            _tool("reused-call", f"result-{idx}"),
        )

    rows = engine._store.get_bounded_tool_call_rows(
        "lookup-session",
        {"reused-call"},
        max_call_ids=8,
        max_rows=16,
    )
    plan = engine._store.connection.execute(
        "EXPLAIN QUERY PLAN SELECT store_id FROM messages "
        "INDEXED BY idx_msg_session WHERE session_id = ? "
        "ORDER BY store_id DESC LIMIT ?",
        ("lookup-session", 256),
    ).fetchall()
    engine.shutdown()

    assert len(rows) == 16
    assert [row["content"] for row in rows] == [f"result-{idx}" for idx in range(84, 100)]
    assert any("idx_msg_session" in str(row) for row in plan)
