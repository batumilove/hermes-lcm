"""Regression for partial durability within one assistant tool-call message."""

from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize("delivery_path", ["direct", "deferred"])
def test_session_end_preserves_only_new_calls_from_mixed_assistant_message(
    tmp_path,
    delivery_path,
):
    session_id = "partial-assistant-tool-call-replay"
    conversation_id = "agent:main:telegram:dm:partial-tool-calls"
    config = LCMConfig(database_path=str(tmp_path / "partial-tool-calls.db"))
    call_a = "call_already_durable"
    call_b = "call_genuinely_new"

    durable_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_a,
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }
        ],
    }
    durable_result = {
        "role": "tool",
        "tool_call_id": call_a,
        "tool_name": "inspect",
        "content": "already durable result",
    }
    seed = LCMEngine(config=config)
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    seed._store.append_batch(
        session_id,
        [durable_assistant, durable_result],
        source="telegram",
        conversation_id=conversation_id,
    )
    seed.shutdown()

    mixed_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            *durable_assistant["tool_calls"],
            {
                "id": call_b,
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            },
        ],
    }
    incoming = [
        mixed_assistant,
        durable_result,
        {
            "role": "tool",
            "tool_call_id": call_b,
            "tool_name": "inspect",
            "content": "new result",
        },
        {"role": "user", "content": "new suffix"},
    ]

    after = LCMEngine(config=config)
    after.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    if delivery_path == "direct":
        after.on_session_end(session_id, incoming)
    else:
        pending = after._persist_session_end_intent(
            session_id,
            incoming,
            ingest_cursor=0,
        )
        after._drain_one_session_end_intent(pending)

    rows = after._store.get_session_messages(session_id)
    assistant_call_counts = Counter(
        str(tool_call.get("id"))
        for row in rows
        for tool_call in row.get("tool_calls") or []
        if isinstance(tool_call, dict) and tool_call.get("id")
    )
    result_counts = Counter(
        str(row["tool_call_id"])
        for row in rows
        if row.get("tool_call_id")
    )
    mixed_rows = [
        row
        for row in rows
        if row.get("role") == "assistant"
        and any(
            isinstance(tool_call, dict) and tool_call.get("id") == call_b
            for tool_call in row.get("tool_calls") or []
        )
    ]
    after.shutdown()

    assert assistant_call_counts == Counter({call_a: 1, call_b: 1})
    assert result_counts == Counter({call_a: 1, call_b: 1})
    assert len(mixed_rows) == 1
    assert [call["id"] for call in mixed_rows[0]["tool_calls"]] == [call_b]


def test_session_end_tool_identity_lookup_does_not_page_full_store(tmp_path, monkeypatch):
    session_id = "bounded-tool-identity-lookup"
    conversation_id = "agent:main:telegram:dm:bounded-lookup"
    config = LCMConfig(database_path=str(tmp_path / "bounded-lookup.db"))
    call_id = "call_already_durable"
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    engine._store.append_batch(
        session_id,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function"}],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "durable result",
            },
        ],
        source="telegram",
        conversation_id=conversation_id,
    )

    def reject_full_store_page(*args, **kwargs):
        raise AssertionError("session-end lookup must not page the full durable store")

    monkeypatch.setattr(engine._store, "get_range", reject_full_store_page)
    replayed = engine._session_end_replayed_tool_indexes(
        session_id,
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function"}],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "replay"},
        ],
        conversation_id=conversation_id,
    )
    engine.shutdown()

    assert replayed == {0, 1}


@pytest.mark.parametrize("delivery_path", ["direct", "deferred"])
def test_session_end_recognizes_durable_object_form_assistant_tool_call(
    tmp_path,
    delivery_path,
):
    session_id = "object-form-assistant-tool-call-replay"
    conversation_id = "agent:main:telegram:dm:object-tool-call"
    config = LCMConfig(database_path=str(tmp_path / "object-tool-call.db"))
    call_id = "call_object_already_durable"
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": {
            "id": call_id,
            "type": "function",
            "function": {"name": "inspect", "arguments": "{}"},
        },
    }

    seed = LCMEngine(config=config)
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    seed._store.append_batch(
        session_id,
        [assistant],
        source="telegram",
        conversation_id=conversation_id,
    )
    seed.shutdown()

    after = LCMEngine(config=config)
    after.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    incoming = [
        assistant,
        {
            "role": "tool",
            "tool_call_id": call_id,
            "tool_name": "inspect",
            "content": "new result",
        },
        {"role": "user", "content": "new suffix"},
    ]
    if delivery_path == "direct":
        after.on_session_end(session_id, incoming)
    else:
        pending = after._persist_session_end_intent(
            session_id,
            incoming,
            ingest_cursor=0,
        )
        after._drain_one_session_end_intent(pending)

    rows = after._store.get_session_messages(session_id)
    object_call_rows = [
        row
        for row in rows
        if row.get("role") == "assistant"
        and isinstance(row.get("tool_calls"), dict)
        and row["tool_calls"].get("id") == call_id
    ]
    result_rows = [
        row
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id") == call_id
    ]
    after.shutdown()

    assert len(object_call_rows) == 1
    assert len(result_rows) == 1
