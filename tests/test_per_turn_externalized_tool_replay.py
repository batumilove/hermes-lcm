"""Regression for per-turn replay after a bound runtime loses its ingest cursor."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize("runtime_shape", ["same_instance_reset", "same_bound_clone"])
def test_per_turn_ingest_filters_old_durable_tool_id_but_preserves_every_new_row(
    tmp_path,
    runtime_shape,
):
    session_id = "production-shaped-per-turn-replay"
    conversation_id = "agent:main:telegram:dm:sanitized:thread"
    payload_dir = tmp_path / "payloads"
    config = LCMConfig(
        database_path=str(tmp_path / "per-turn-replay.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(payload_dir),
    )
    stale_assistant_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_production_externalized",
                "type": "function",
                "function": {"name": "session_search", "arguments": "{}"},
            }
        ],
    }
    stale_tool_result = {
        "role": "tool",
        "tool_call_id": "call_production_externalized",
        "tool_name": "session_search",
        "content": "production-shaped externalized result " * 100,
    }
    durable_history = [
        {"role": "user", "content": "legitimate repeated user text"},
        stale_assistant_call,
        stale_tool_result,
        {"role": "assistant", "content": "inspection complete"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    first.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    first.ingest(durable_history)
    assert len(list(payload_dir.glob("*.json"))) == 1

    # Move the durable tool identity outside the replay scanner's bounded tail.
    # The contract is keyed by durable tool ID, not by accidental tail proximity.
    first._store.append_batch(
        session_id,
        [
            {"role": "assistant", "content": f"durable filler row {index}"}
            for index in range(4200)
        ],
        source="telegram",
        conversation_id=conversation_id,
    )

    if runtime_shape == "same_instance_reset":
        replay = first
        replay._ingest_cursor = 0
        replay._ingest_cursor_needs_reconcile = False
    else:
        first.shutdown()
        replay = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        # Production post-turn binding treats this clone as already bound and does
        # not call on_session_start again; its process-local cursor is therefore 0.
        replay._session_id = session_id
        replay._conversation_id = conversation_id
        replay._session_platform = "telegram"
        replay._ingest_cursor = 0
        replay._ingest_cursor_needs_reconcile = False

    fresh_assistant_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_genuinely_new",
                "type": "function",
                "function": {"name": "lcm_status", "arguments": "{}"},
            }
        ],
    }
    fresh_tool_result = {
        "role": "tool",
        "tool_call_id": "call_genuinely_new",
        "tool_name": "lcm_status",
        "content": "genuinely new tool result",
    }
    legitimate_new_rows = [
        {"role": "user", "content": "legitimate repeated user text"},
        {"role": "user", "content": "genuinely new request"},
        {"role": "assistant", "content": "genuinely new response"},
        fresh_assistant_call,
        fresh_tool_result,
    ]
    replay.ingest([stale_assistant_call, stale_tool_result, *legitimate_new_rows])

    rows = replay._store.get_session_messages(session_id)

    def assistant_call_count(call_id):
        return sum(
            1
            for row in rows
            if row.get("role") == "assistant"
            and any(
                isinstance(call, dict) and call.get("id") == call_id
                for call in (row.get("tool_calls") or [])
            )
        )

    evidence = {
        "runtime_shape": runtime_shape,
        "stale_tool_count": sum(
            row.get("role") == "tool"
            and row.get("tool_call_id") == "call_production_externalized"
            for row in rows
        ),
        "stale_assistant_call_count": assistant_call_count(
            "call_production_externalized"
        ),
        "fresh_assistant_call_count": assistant_call_count("call_genuinely_new"),
        "fresh_tool_count": sum(
            row.get("role") == "tool"
            and row.get("tool_call_id") == "call_genuinely_new"
            for row in rows
        ),
        "repeated_user_count": sum(
            row.get("role") == "user"
            and row.get("content") == "legitimate repeated user text"
            for row in rows
        ),
        "new_request_count": sum(
            row.get("role") == "user" and row.get("content") == "genuinely new request"
            for row in rows
        ),
        "new_response_count": sum(
            row.get("role") == "assistant"
            and row.get("content") == "genuinely new response"
            for row in rows
        ),
        "payload_count": len(list(payload_dir.glob("*.json"))),
    }
    replay.shutdown()

    assert evidence == {
        "runtime_shape": runtime_shape,
        "stale_tool_count": 1,
        "stale_assistant_call_count": 1,
        "fresh_assistant_call_count": 1,
        "fresh_tool_count": 1,
        "repeated_user_count": 2,
        "new_request_count": 1,
        "new_response_count": 1,
        "payload_count": 1,
    }
