"""Interleaved full-snapshot replay after Telegram session resume (RED).

Live failure 2026-08-26, session 20260826_152359_ea48c8: a durable session is
resumed and the incoming context re-delivers already-stored rows interleaved
(1:1 alternating replayed/new-shape rows) rather than as a contiguous prefix or
segment. No cursor-based matcher can align it, so the terminal
``persisted ambiguous delta`` fallback (cursor=0) persists the whole batch.
Expected: already-stored identities are suppressed per-row; genuinely new rows
persist.
"""
import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_existing_session_restart_suppresses_interleaved_snapshot_replay(tmp_path):
    db_path = tmp_path / "interleaved-snapshot-replay.db"
    config = LCMConfig(database_path=str(db_path))

    before = LCMEngine(config=config)
    before.on_session_start(
        "interleaved-replay-session",
        platform="telegram",
        conversation_id="interleaved-replay-conversation",
        context_length=200000,
    )
    durable = []
    for idx in range(3):
        call_id = f"call_dur_{idx}"
        durable.extend(
            [
                {"role": "user", "content": f"resume request {idx}"},
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
        {"role": "user", "content": f"later durable message {i}"} for i in range(80)
    )
    before._ingest_messages(durable)
    before.shutdown()

    after = LCMEngine(config=config)
    after.on_session_start(
        "interleaved-replay-session",
        platform="telegram",
        conversation_id="interleaved-replay-conversation",
        context_length=200000,
    )

    # Interleaved snapshot: replayed durable rows alternate 1:1 with
    # never-stored filler rows, so the replayed rows never form a contiguous
    # block. One genuinely new row closes the batch.
    replayed_rows = [
        msg
        for msg in durable
        if msg.get("tool_call_id") in {"call_dur_0", "call_dur_1", "call_dur_2"}
        or msg.get("content") in {f"resume request {i}" for i in range(3)}
        or msg.get("content") in {f"durable answer {i}" for i in range(3)}
    ]
    interleaved = []
    for idx, replayed in enumerate(replayed_rows):
        interleaved.append(replayed)
        interleaved.append({"role": "user", "content": f"[compacted filler {idx}]"})
    interleaved.append({"role": "user", "content": "genuinely new follow-up"})

    after._ingest_messages(interleaved)

    rows = after._store.get_session_messages("interleaved-replay-session")
    contents = [row["content"] for row in rows]
    filler_count = sum(1 for c in contents if str(c).startswith("[compacted filler"))
    new_count = sum(1 for c in contents if c == "genuinely new follow-up")
    tool_row_count = sum(
        1
        for row in rows
        if row.get("tool_call_id") in {"call_dur_0", "call_dur_1", "call_dur_2"}
    )

    # Every replayed durable identity appears exactly once (the original),
    # fillers and the new row persist, and nothing else leaks in.
    assert len(rows) == len(durable) + filler_count + new_count
    assert tool_row_count == 3
    for idx in range(3):
        assert contents.count(f"durable result {idx}") == 1
        assert contents.count(f"durable answer {idx}") == 1
        assert contents.count(f"resume request {idx}") == 1
    assert new_count == 1
    assert rows[-1]["content"] == "genuinely new follow-up"

    reconciliation = after.get_status()["ingest_reconciliation"]
    assert reconciliation["reason"] != "persisted ambiguous delta" or (
        reconciliation.get("effective_incoming") == filler_count + new_count
    )
