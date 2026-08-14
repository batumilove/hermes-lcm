import hashlib
import json

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.session_end_pending import (
    build_session_end_intent,
    load_session_end_intent,
    persist_session_end_intent,
)


def _engine(tmp_path, name):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"{name}.db")))
    engine.on_session_start(f"{name}-session", platform="telegram")
    return engine


def test_v2_intent_persists_immutable_ingest_cursor(tmp_path):
    intent = build_session_end_intent(
        session_id="session-a",
        conversation_id="conversation-a",
        source="telegram",
        frontier_store_id=42,
        messages=[
            {"role": "system", "content": "compressed context"},
            {"role": "user", "content": "final turn"},
        ],
        ingest_cursor=1,
    )
    path = persist_session_end_intent(tmp_path / "lcm.db", intent)

    loaded = load_session_end_intent(path)

    assert loaded["version"] == 2
    assert loaded["ingest_cursor"] == 1
    assert loaded["intent_sha256"] == intent["intent_sha256"]


def test_loader_preserves_legacy_v1_digest_contract(tmp_path):
    identity = {
        "version": 1,
        "session_id": "legacy-session",
        "conversation_id": "legacy-conversation",
        "source": "telegram",
        "frontier_store_id": 7,
        "messages": [{"role": "user", "content": "legacy final turn"}],
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    payload = {
        **identity,
        "intent_sha256": hashlib.sha256(encoded).hexdigest(),
        "created_at": 1.0,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_session_end_intent(path)

    assert loaded == payload
    assert "ingest_cursor" not in loaded


def test_v2_intent_rejects_cursor_outside_immutable_snapshot():
    with pytest.raises(ValueError, match="cursor is out of range"):
        build_session_end_intent(
            session_id="session-a",
            conversation_id="conversation-a",
            source="telegram",
            frontier_store_id=0,
            messages=[{"role": "user", "content": "only message"}],
            ingest_cursor=2,
        )


def test_engine_clamps_runtime_cursor_to_immutable_snapshot(tmp_path):
    engine = _engine(tmp_path, "cursor-clamp")
    try:
        engine._ingest_cursor = 99
        path = engine._persist_session_end_intent(
            engine._session_id,
            [{"role": "user", "content": "only message"}],
        )

        loaded = load_session_end_intent(path)

        assert loaded["ingest_cursor"] == 1
    finally:
        engine.shutdown()


def test_deferred_recovery_replays_only_uningested_cursor_suffix(tmp_path):
    engine = _engine(tmp_path, "cursor-replay")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    try:
        engine._store.append_batch(
            session_id,
            [{"role": "tool", "content": "historical persisted row"}],
            source="telegram",
            conversation_id=conversation_id,
        )
        messages = [
            {"role": "system", "content": "compressed context not stored as a prefix"},
            {"role": "user", "content": "final unpersisted turn"},
        ]
        path = engine._persist_session_end_intent(
            session_id,
            messages,
            ingest_cursor=1,
        )
        digest = load_session_end_intent(path)["intent_sha256"]

        engine._drain_one_session_end_intent(path)

        persisted = engine._store.get_session_messages(session_id)
        assert [message.get("content") for message in persisted] == [
            "historical persisted row",
            "final unpersisted turn",
        ]
        assert engine._store.has_session_end_ingest_receipt(digest)
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.last_finalized_session_id == session_id
        assert not path.exists()
    finally:
        engine.shutdown()


def test_synchronous_raw_commit_records_receipt_before_lifecycle_failure(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, "sync-receipt")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    original_finalize = engine._lifecycle.finalize_session
    try:
        def fail_finalize(*args, **kwargs):
            raise RuntimeError("lifecycle unavailable after raw commit")

        monkeypatch.setattr(engine._lifecycle, "finalize_session", fail_finalize)
        messages = [{"role": "user", "content": "committed exactly once"}]
        with pytest.raises(RuntimeError, match="lifecycle unavailable"):
            engine.on_session_end(session_id, messages)

        pending = list(
            (engine._store.db_path.parent / f"{engine._store.db_path.name}-session-end-pending").glob(
                "*.json"
            )
        )
        assert len(pending) == 1
        digest = load_session_end_intent(pending[0])["intent_sha256"]
        assert engine._store.has_session_end_ingest_receipt(digest)

        monkeypatch.setattr(engine._lifecycle, "finalize_session", original_finalize)
        engine._drain_one_session_end_intent(pending[0])
        persisted = engine._store.get_session_messages(session_id)
        assert [message.get("content") for message in persisted] == [
            "committed exactly once"
        ]
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.last_finalized_session_id == session_id
    finally:
        engine.shutdown()


def test_receipt_prevents_duplicate_after_raw_commit_before_lifecycle(tmp_path):
    engine = _engine(tmp_path, "receipt-retry")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    try:
        messages = [
            {"role": "system", "content": "active context absent from immutable store"},
            {"role": "user", "content": "already committed final turn"},
        ]
        path = engine._persist_session_end_intent(
            session_id,
            messages,
            ingest_cursor=1,
        )
        digest = load_session_end_intent(path)["intent_sha256"]
        engine._store._append_protected_batch(
            session_id,
            [messages[1]],
            [1],
            source="telegram",
            conversation_id=conversation_id,
            session_end_intent_sha256=digest,
        )

        engine._drain_one_session_end_intent(path)

        persisted = engine._store.get_session_messages(session_id)
        assert [message.get("content") for message in persisted] == [
            "already committed final turn"
        ]
        assert not path.exists()
    finally:
        engine.shutdown()


def test_restart_coalesces_extending_cursor_zero_intents_exactly_once(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "restart-coalesce.db"))
    ending = LCMEngine(config=config)
    ending.on_session_start("shared-session", platform="telegram")
    try:
        ending._persist_session_end_intent(
            "shared-session",
            [{"role": "user", "content": "one"}],
            ingest_cursor=0,
        )
        ending._persist_session_end_intent(
            "shared-session",
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
            ingest_cursor=0,
        )
    finally:
        ending.shutdown()

    restarted = LCMEngine(config=config)
    try:
        assert restarted._session_end_drain_done.wait(timeout=2.0)
        persisted = restarted._store.get_session_messages("shared-session")
        assert [message.get("content") for message in persisted] == ["one", "two"]
    finally:
        restarted.shutdown()


def test_deferred_drain_exhaustion_is_bounded_and_retains_intent(
    tmp_path,
    monkeypatch,
    caplog,
):
    engine = _engine(tmp_path, "bounded-exhaustion")
    release = False
    path = None
    original = engine._drain_one_session_end_intent
    try:
        path = engine._persist_session_end_intent(
            engine._session_id,
            [{"role": "user", "content": "retain until recoverable"}],
            ingest_cursor=0,
        )

        def unresolved(intent_path):
            if release:
                return original(intent_path)
            raise RuntimeError("permanently unresolved in this worker budget")

        monkeypatch.setattr(engine, "_drain_one_session_end_intent", unresolved)
        monkeypatch.setattr(
            "hermes_lcm.engine._SESSION_END_DEFERRED_RETRY_BUDGET_SECONDS",
            0.08,
        )
        monkeypatch.setattr(
            "hermes_lcm.engine._SESSION_END_DEFERRED_RETRY_INTERVAL_SECONDS",
            0.01,
        )
        with caplog.at_level("WARNING"):
            engine._schedule_session_end_drain()
            finished = engine._session_end_drain_done.wait(timeout=0.5)
        retained = path.exists()
        thread_cleared = engine._session_end_drain_thread is None
        exhaustion_count = caplog.text.count(
            "LCM deferred session-end drain exhausted bounded retry budget"
        )
        pending_failures = engine._session_end_pending_failures
        release = True
        engine._schedule_session_end_drain()
        recovered_finished = engine._session_end_drain_done.wait(timeout=1.0)
        recovered = not path.exists()
    finally:
        release = True
        if not engine._session_end_drain_done.is_set():
            engine._session_end_drain_done.wait(timeout=2.0)
        if path is not None:
            path.unlink(missing_ok=True)
        engine.shutdown()

    assert finished
    assert retained
    assert thread_cleared
    assert exhaustion_count == 1
    assert pending_failures == 1
    assert recovered_finished
    assert recovered


def test_message_batch_and_receipt_roll_back_together(tmp_path):
    engine = _engine(tmp_path, "receipt-atomicity")
    session_id = engine._session_id
    conversation_id = engine._conversation_id
    digest = "a" * 64
    try:
        with pytest.raises(TypeError):
            engine._store._append_protected_batch(
                session_id,
                [
                    {"role": "user", "content": "must roll back"},
                    {"role": "assistant", "content": None, "tool_calls": object()},
                ],
                [1, 1],
                source="telegram",
                conversation_id=conversation_id,
                session_end_intent_sha256=digest,
            )

        assert engine._store.get_session_messages(session_id) == []
        assert not engine._store.has_session_end_ingest_receipt(digest)
    finally:
        engine.shutdown()
