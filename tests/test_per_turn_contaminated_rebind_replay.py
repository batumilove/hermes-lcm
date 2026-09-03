"""Regression for stale durable tool IDs in non-session-end ingestion.

A compression/rebind delivery can contain a valid active prefix, one stale tool
result copied from durable history, and genuinely new turn rows.  The ordinary
ingest path must preserve the new suffix without amplifying the old tool ID.
"""

from collections import Counter

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_per_turn_rebind_filters_stale_durable_tool_id_and_keeps_new_suffix(tmp_path):
    session_id = "production-shaped-per-turn-rebind"
    conversation_id = "agent:main:telegram:dm:sanitized:thread"
    config = LCMConfig(database_path=str(tmp_path / "per-turn-rebind.db"))
    old_call_id = "call_sanitized_old"
    new_call_id = "call_sanitized_new"

    canonical_history = [
        {"role": "user", "content": "preserved active prefix"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": old_call_id,
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": old_call_id,
            "tool_name": "inspect",
            "content": "already durable old result",
        },
        {"role": "assistant", "content": "old result acknowledged"},
    ]

    seed = LCMEngine(config=config)
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    durable_filler = [
        # Keep the stale tool identity outside the bounded generic tail scanner,
        # matching the production replay where thousands of rows intervened.
        {"role": "user", "content": f"later durable history row {index}"}
        for index in range(300)
    ]
    seed._store.append_batch(
        session_id,
        [*canonical_history, *durable_filler],
        source="telegram",
        conversation_id=conversation_id,
    )
    seed.shutdown()

    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )

    # Overflow recovery leaves a cursor indexing the short active assembly.
    # A later ordinary (non-session-end) delivery keeps that prefix but carries
    # one stale durable tool result in front of genuinely new turn rows.
    compressed_active = canonical_history[:1]
    engine._finalize_forced_overflow_result(canonical_history, compressed_active)
    assert engine._ingest_cursor == 1
    assert not engine._ingest_cursor_needs_reconcile

    # A same-prefix callback has no suffix to inspect and must not consume the
    # one-shot recovery guard before the next delivery arrives.
    engine._ingest_messages(compressed_active)
    assert engine._overflow_recovery_ingest_pending

    rebound_delivery = [
        *compressed_active,
        dict(canonical_history[2]),
        {"role": "user", "content": "genuinely new user turn"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": new_call_id,
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": new_call_id,
            "tool_name": "inspect",
            "content": "genuinely new result",
        },
        {"role": "assistant", "content": "genuinely new answer"},
    ]

    engine._ingest_messages(rebound_delivery)
    assert not engine._overflow_recovery_ingest_pending
    rows = engine._store.get_session_messages(session_id)
    tool_counts = Counter(
        str(row["tool_call_id"])
        for row in rows
        if row.get("tool_call_id")
    )
    content_counts = Counter(str(row.get("content") or "") for row in rows)
    evidence = {
        "tool_counts": dict(tool_counts),
        "new_user_count": content_counts["genuinely new user turn"],
        "new_answer_count": content_counts["genuinely new answer"],
        "row_count": len(rows),
    }
    engine.shutdown()

    assert tool_counts[old_call_id] == 1, evidence
    assert tool_counts[new_call_id] == 1, evidence
    assert content_counts["genuinely new user turn"] == 1, evidence
    assert content_counts["genuinely new answer"] == 1, evidence


def test_session_reset_clears_pending_overflow_recovery_ingest(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "reset-overflow-pending.db"))
    )
    engine._overflow_recovery_ingest_pending = True

    engine._reset_session_scoped_runtime_state()

    assert not engine._overflow_recovery_ingest_pending
    engine.shutdown()
