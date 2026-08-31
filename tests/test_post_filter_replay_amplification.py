"""Post-filter full-snapshot replay amplification (RED).

Live failure 2026-08-31, session 20260831_044058_5cafeb (gateway started on
9c01f08): after a tool-anchored snapshot was correctly filtered on replay, a
LATER full snapshot re-delivered most of the same durable identities plus a
new suffix, and the re-delivered tool IDs were appended again. lcm.db grew
1291 duplicate (session_id, tool_call_id) groups / 1306 excess rows across 47
sessions while state.db held no corresponding duplicate IDs — so the
duplicates live in the LCM store, not in gateway bookkeeping.

Proven burst shape in the representative session:
  rows 1003202-1003301  (100 rows, 54 tool results)
  rows 1003777-1003823  (47 rows, 20 tool results)
  rows 1005969-1006009  (41 rows, 20 tool results)
  rows 1007586-1007725  (140 rows, 76 tool results)
The final burst repeats 97/100 identities from the first burst and 38/41 from
the third while preserving some genuinely new rows.

Hypothesis under test: after overflow recovery reshapes active context and resets
the process cursor to ``len(compressed)``, the session-end callback receives the
host's full transcript. The direct final-flush path trusts that active-context
cursor as though it indexed the full transcript, bypasses durable-prefix proof,
and reappends old tool IDs before recording the receipt. Expected behavior: every
previously stored tool_call_id remains exactly once and only the genuine suffix
persists.
"""
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize("delivery_path", ["direct", "deferred"])
def test_full_snapshot_after_filtered_replay_does_not_reappend_durable_tool_ids(
    tmp_path,
    delivery_path,
):
    db_path = tmp_path / "post-filter-replay-amplification.db"
    config = LCMConfig(database_path=str(db_path))

    before = LCMEngine(config=config)
    before.on_session_start(
        "post-filter-replay-session",
        platform="telegram",
        conversation_id="post-filter-replay-conversation",
        context_length=200000,
    )
    durable = []
    for idx in range(3):
        call_id = f"call_dur_{idx}"
        durable.extend(
            [
                {"role": "user", "content": f"burst request {idx}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "tool_name": "inspect",
                    "content": f"durable result {idx}",
                },
                {"role": "assistant", "content": f"durable answer {idx}"},
            ]
        )
    durable.extend(
        {"role": "user", "content": f"later durable message {i}"} for i in range(8)
    )
    before._ingest_messages(durable)
    before.shutdown()

    after = LCMEngine(config=config)
    after.on_session_start(
        "post-filter-replay-session",
        platform="telegram",
        conversation_id="post-filter-replay-conversation",
        context_length=200000,
    )

    # Phase 1: a tool-anchored snapshot replay. The scanner must suppress the
    # already-durable identities; only the one genuinely new row persists.
    # This mirrors the mid-session compaction reshaping seen live.
    phase1 = list(durable) + [{"role": "user", "content": "phase one new tail"}]
    after._ingest_messages(phase1)

    rows_after_phase1 = after._store.get_session_messages("post-filter-replay-session")
    phase1_tool_rows = sum(
        1
        for row in rows_after_phase1
        if row.get("tool_call_id")
        in {"call_dur_0", "call_dur_1", "call_dur_2"}
    )
    assert phase1_tool_rows == 3, {
        "phase": "post-filtered-replay",
        "tool_rows": phase1_tool_rows,
        "row_count": len(rows_after_phase1),
    }

    # Production transition: forced overflow recovery replaces active context
    # with a shorter assembly and deliberately sets the process cursor to that
    # compressed length with reconciliation disabled. That cursor does not index
    # the full host transcript subsequently delivered to on_session_end().
    compressed_active = phase1[-5:]
    after._finalize_forced_overflow_result(phase1, compressed_active)
    assert after._ingest_cursor == len(compressed_active)
    assert not after._ingest_cursor_needs_reconcile

    # Phase 2: session-end receives a FULL host snapshot re-delivering nearly
    # every prior identity plus a new suffix. Durable-prefix proof, not the
    # active-context cursor, must decide what is already stored.
    phase2 = [
        *durable,
        {"role": "user", "content": "phase one new tail"},
        {"role": "user", "content": "phase two genuinely new follow-up"},
    ]
    if delivery_path == "direct":
        after.on_session_end("post-filter-replay-session", phase2)
    else:
        pending = after._persist_session_end_intent(
            "post-filter-replay-session",
            phase2,
            ingest_cursor=after._ingest_cursor,
        )
        after._drain_one_session_end_intent(pending)

    rows = after._store.get_session_messages("post-filter-replay-session")
    contents = [row["content"] for row in rows]
    tool_row_count = sum(
        1
        for row in rows
        if row.get("tool_call_id")
        in {"call_dur_0", "call_dur_1", "call_dur_2"}
    )
    new_count = sum(1 for c in contents if c == "phase two genuinely new follow-up")

    evidence = {
        "row_count": len(rows),
        "expected_rows_max": len(phase2),
        "tool_row_count": tool_row_count,
        "new_count": new_count,
        "duplicate_contents": sorted(
            {c for c in contents if contents.count(c) > 1}
        ),
        "reconciliation": after.get_status().get("ingest_reconciliation"),
    }
    after.shutdown()

    # Every prior tool_call_id remains exactly once; nothing is reappended.
    assert tool_row_count == 3, evidence
    for idx in range(3):
        assert contents.count(f"durable result {idx}") == 1, evidence
        assert contents.count(f"durable answer {idx}") == 1, evidence
        assert contents.count(f"burst request {idx}") == 1, evidence
    for i in range(8):
        assert contents.count(f"later durable message {i}") == 1, evidence
    assert contents.count("phase one new tail") == 1, evidence
    # The new suffix persists exactly once.
    assert new_count == 1, evidence
    assert rows[-1]["content"] == "phase two genuinely new follow-up", evidence
    # The store never exceeds one copy of any delivered identity.
    assert len(rows) <= len(phase2), evidence
