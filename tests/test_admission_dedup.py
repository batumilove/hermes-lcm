"""Admission-level hard filter for exact durable tool-row duplicates.

Generation #45 bound the duplicate diagnostic to scoped identity but kept the
admission observer behavior-neutral, so byte-identical tool rows could still
be stored (live evidence 2026-09-12: 99 byte-identical duplicate groups in one
post-restart window). These tests pin the new contract: an incoming tool row
whose exact identity already exists in the durable store for the same session
AND conversation scope is dropped at admission, with a bounded receipt; a
row with different content under the same tool_call_id is still stored.
"""

import hashlib
import json
import logging

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


EVENT_PREFIX = "LCM_DUPLICATE_TOOL_ADMISSION_FILTERED "

def _tool(call_id: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "tool_name": "inspect",
        "content": content,
    }


def _seed_and_rewind(tmp_path, db_name, session_id, conversation_id):
    config = LCMConfig(database_path=str(tmp_path / db_name))
    seed = LCMEngine(config=config)
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    seed._ingest_messages([_tool("call_seed", "seed result")])
    seed.shutdown()
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    # Model a missed replay scan: disable the replay filter so the duplicate
    # reaches admission — exactly the residual per-turn replay class observed
    # live on 2026-09-12.
    engine._find_tool_anchored_replay_indexes = (
        lambda *_args, **_kwargs: (set(), 0)
    )
    return engine


def test_exact_duplicate_tool_row_is_dropped_at_admission(tmp_path, caplog):
    session_id = "dedup-session"
    conversation_id = "dedup-conversation"
    call_id = "call_exact_dup"
    content = "exact durable result body"

    engine = _seed_and_rewind(
        tmp_path, "dedup.db", session_id, conversation_id
    )
    engine._ingest_messages([_tool(call_id, content)])
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine._ingest_messages([_tool(call_id, content)])

    rows = [
        row
        for row in engine._store.get_session_messages(session_id)
        if row["role"] == "tool"
    ]
    engine.shutdown()

    # Exactly one durable copy of the exact-identity tool row; the seed row
    # for call_seed is expected and untouched.
    stored_exact = [
        row for row in rows if row["tool_call_id"] == call_id
    ]
    assert len(stored_exact) == 1, (
        "exact duplicate must be dropped at admission, "
        f"got {len(stored_exact)} durable rows: {rows}"
    )

    events = [
        json.loads(record.message[len(EVENT_PREFIX):])
        for record in caplog.records
        if record.message.startswith(EVENT_PREFIX)
    ]
    assert len(events) == 1
    event = events[0]
    assert event["schema"] == "lcm_duplicate_tool_admission_filtered_v1"
    assert event["dropped_count"] == 1
    assert event["duplicates"][0]["tool_call_id_sha256"] == hashlib.sha256(
        call_id.encode()
    ).hexdigest()
    serialized = json.dumps(event, sort_keys=True)
    for secret in (session_id, conversation_id, call_id, content):
        assert secret not in serialized


def test_different_content_same_call_id_is_stored_not_dropped(tmp_path, caplog):
    session_id = "changed-session"
    conversation_id = "changed-conversation"

    engine = _seed_and_rewind(
        tmp_path, "changed.db", session_id, conversation_id
    )
    engine._ingest_messages([_tool("call_reused", "first result")])
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.engine"):
        engine._ingest_messages([_tool("call_reused", "different result")])

    rows = engine._store.get_session_messages(session_id)
    engine.shutdown()
    assert [
        row["content"] for row in rows if row["role"] == "tool"
    ] == ["seed result", "first result", "different result"]
    assert EVENT_PREFIX not in caplog.text
