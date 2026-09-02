"""Regression for session-end replay against an already contaminated LCM store.

Production evidence from generation 272952e showed a full canonical host transcript
whose durable identities existed in LCM, but not as one clean leading prefix: older
partial/full replay bursts had reordered and duplicated parts of the session.  The
session-end durable-prefix check consequently returned no proof and appended the
canonical tool identities again immediately before recording a unique receipt.
"""

from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize("delivery_path", ["direct", "deferred"])
def test_full_snapshot_does_not_amplify_tool_ids_in_contaminated_store(
    tmp_path,
    delivery_path,
):
    session_id = "production-shaped-replayed-session"
    conversation_id = "agent:main:telegram:dm:sanitized:thread"
    db_path = tmp_path / "contaminated-session-end-replay.db"
    config = LCMConfig(database_path=str(db_path))

    call_a = "call_sanitized_A"
    call_b = "call_sanitized_B"
    canonical = [
        {"role": "user", "content": "sanitized recovered objective"},
        {"role": "assistant", "content": "sanitized recovered context"},
        {"role": "user", "content": "inspect first object"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_a,
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_a,
            "tool_name": "inspect",
            "content": "sanitized first result",
        },
        {"role": "assistant", "content": "first result acknowledged"},
        {"role": "user", "content": "inspect second object"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_b,
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_b,
            "tool_name": "inspect",
            "content": "sanitized second result",
        },
        {"role": "assistant", "content": "second result acknowledged"},
    ]

    # Freeze the production shape: a partial replay burst precedes a later full
    # canonical copy. One old tool result has different content for the same ID,
    # matching the small non-byte-identical subset in the live evidence.
    historical_partial = [
        dict(canonical[6]),
        dict(canonical[7]),
        {
            **canonical[8],
            "content": "sanitized second result from an older representation",
        },
        dict(canonical[9]),
    ]
    seed = LCMEngine(config=config)
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    seed._store.append_batch(
        session_id,
        [*historical_partial, *canonical],
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

    # Overflow recovery leaves a short active assembly and a cursor that indexes
    # that assembly, not the subsequently delivered full host transcript.
    compressed_active = canonical[:2]
    after._finalize_forced_overflow_result(canonical, compressed_active)
    assert after._ingest_cursor == 2
    assert not after._ingest_cursor_needs_reconcile

    full_snapshot = [
        *canonical,
        {"role": "user", "content": "genuinely new session-end suffix"},
    ]

    def durable_tool_counts():
        result_counts = Counter()
        assistant_call_counts = Counter()
        for row in after._store.get_session_messages(session_id):
            if row.get("tool_call_id"):
                result_counts[str(row["tool_call_id"])] += 1
            for tool_call in row.get("tool_calls") or []:
                if isinstance(tool_call, dict) and tool_call.get("id"):
                    assistant_call_counts[str(tool_call["id"])] += 1
        return result_counts, assistant_call_counts

    before_result_counts, before_assistant_counts = durable_tool_counts()
    if delivery_path == "direct":
        after.on_session_end(session_id, full_snapshot)
    else:
        pending = after._persist_session_end_intent(
            session_id,
            full_snapshot,
            ingest_cursor=after._ingest_cursor,
        )
        after._drain_one_session_end_intent(pending)

    rows = after._store.get_session_messages(session_id)
    after_result_counts, after_assistant_counts = durable_tool_counts()
    suffix_count = sum(
        row.get("content") == "genuinely new session-end suffix" for row in rows
    )
    receipt_count = after._store._conn.execute(
        """SELECT COUNT(*) FROM lcm_session_end_ingest_receipts
           WHERE session_id = ? AND conversation_id = ?""",
        (session_id, conversation_id),
    ).fetchone()[0]
    evidence = {
        "delivery_path": delivery_path,
        "before_result_counts": dict(before_result_counts),
        "after_result_counts": dict(after_result_counts),
        "before_assistant_counts": dict(before_assistant_counts),
        "after_assistant_counts": dict(after_assistant_counts),
        "suffix_count": suffix_count,
        "receipt_count": receipt_count,
        "row_count": len(rows),
    }
    after.shutdown()

    # Existing contamination is evidence, not permission to amplify it further.
    assert (
        after_result_counts,
        after_assistant_counts,
    ) == (
        before_result_counts,
        before_assistant_counts,
    ), evidence
    assert suffix_count == 1, evidence
    assert receipt_count == 1, evidence
