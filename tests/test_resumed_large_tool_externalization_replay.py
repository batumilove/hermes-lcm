"""Regression for replayed large tool results after durable externalization.

The active transcript retains provider-usable raw tool content while storage keeps
an externalized marker. A resumed/rebound delivery can append the same raw tool
result after the ingest cursor. Durable replay matching must compare against the
storage form, otherwise the old tool ID is amplified while new suffix rows arrive.
"""

from collections import Counter

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_rebound_raw_large_tool_result_matches_durable_externalized_marker(tmp_path):
    session_id = "production-shaped-resumed-large-tool-replay"
    conversation_id = "agent:main:telegram:dm:sanitized:thread"
    old_call_id = "call_sanitized_large_old"
    config = LCMConfig(
        database_path=str(tmp_path / "resumed-large-tool.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )

    prefix = {"role": "user", "content": "durable active prefix"}
    raw_large_tool = {
        "role": "tool",
        "tool_call_id": old_call_id,
        "tool_name": "session_search",
        "content": "production-shaped large result " + ("x" * 4096),
    }

    first_active = engine._ingest_messages([prefix, raw_large_tool])
    assert first_active[1]["content"] == raw_large_tool["content"]
    assert engine._ingest_cursor == 2
    first_rows = engine._store.get_session_messages(session_id)
    first_tool = next(row for row in first_rows if row.get("tool_call_id") == old_call_id)
    assert str(first_tool["content"]).startswith("[Externalized tool output:")

    # A resumed callback preserves the already-consumed prefix but appends the
    # same provider-visible raw tool result after the cursor, followed by a real
    # new user row. This is the production shape missed by generation 09514edb.
    rebound_delivery = [
        *first_active,
        dict(raw_large_tool),
        {"role": "user", "content": "genuinely new resumed suffix"},
    ]
    engine._ingest_messages(rebound_delivery)

    rows = engine._store.get_session_messages(session_id)
    tool_counts = Counter(
        str(row["tool_call_id"])
        for row in rows
        if row.get("tool_call_id")
    )
    content_counts = Counter(str(row.get("content") or "") for row in rows)
    evidence = {
        "tool_counts": dict(tool_counts),
        "new_suffix_count": content_counts["genuinely new resumed suffix"],
        "tool_contents": [
            row.get("content") for row in rows if row.get("tool_call_id") == old_call_id
        ],
        "row_count": len(rows),
    }
    engine.shutdown()

    assert tool_counts[old_call_id] == 1, evidence
    assert content_counts["genuinely new resumed suffix"] == 1, evidence


def test_cursor_rewind_rescans_pre_cursor_durable_tools_by_key(tmp_path):
    """A late prefix rewind must not bypass durable-key replay matching.

    The first durable-key pass starts at the current cursor. A changed active
    prefix can subsequently make the tail scanner rewind that cursor to zero.
    Tool results before the old cursor are then storage candidates too, so they
    require the same durable-key scan; a bounded tail scan cannot prove old
    identities once hundreds of later rows separate them from the tail.
    """
    session_id = "production-shaped-late-cursor-rewind"
    conversation_id = "agent:main:telegram:dm:sanitized:thread"
    old_call_ids = [f"call_sanitized_old_{index}" for index in range(4)]
    recent_call_id = "call_sanitized_recent_anchor"
    config = LCMConfig(
        database_path=str(tmp_path / "late-cursor-rewind.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized-rewind"),
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )

    active_prefix = [{"role": "user", "content": "compressed active prefix"}]
    stale_raw_tools = [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "tool_name": "session_search",
            "content": f"large durable result {call_id} " + ("x" * 4096),
        }
        for call_id in old_call_ids
    ]
    filler = [
        {"role": "user", "content": f"later durable history row {index}"}
        for index in range(300)
    ]
    recent_tool = {
        "role": "tool",
        "tool_call_id": recent_call_id,
        "tool_name": "inspect",
        "content": "recent durable tail anchor",
    }
    canonical_history = [*active_prefix, *stale_raw_tools, *filler, recent_tool]
    engine._ingest_messages(canonical_history)

    compressed_active = [*active_prefix, *stale_raw_tools]
    engine._finalize_forced_overflow_result(canonical_history, compressed_active)
    assert engine._ingest_cursor == len(compressed_active)
    engine._ingest_messages(compressed_active)
    assert engine._overflow_recovery_ingest_pending

    rebound_delivery = [
        {"role": "user", "content": "changed resumed prefix"},
        *[dict(message) for message in stale_raw_tools],
        dict(recent_tool),
        {"role": "user", "content": "genuinely new post-rewind suffix"},
    ]
    engine._ingest_messages(rebound_delivery)

    rows = engine._store.get_session_messages(session_id)
    tool_counts = Counter(
        str(row["tool_call_id"])
        for row in rows
        if row.get("tool_call_id")
    )
    content_counts = Counter(str(row.get("content") or "") for row in rows)
    evidence = {
        "tool_counts": dict(tool_counts),
        "new_suffix_count": content_counts["genuinely new post-rewind suffix"],
        "row_count": len(rows),
    }
    engine.shutdown()

    assert all(tool_counts[call_id] == 1 for call_id in old_call_ids), evidence
    assert tool_counts[recent_call_id] == 1, evidence
    assert content_counts["genuinely new post-rewind suffix"] == 1, evidence
