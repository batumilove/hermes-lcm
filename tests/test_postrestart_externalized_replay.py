"""Regression for a full state transcript replayed after a fresh-process rebind."""

from collections import Counter

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _assistant_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "fixture_tool", "arguments": "{}"},
            }
        ],
    }


def _tool_result(call_id: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "tool_name": "fixture_tool",
        "content": content,
    }


def test_fresh_process_rebind_does_not_reexternalize_one_old_result(tmp_path, monkeypatch):
    """A 77-row durable transcript plus one new user row stores only that row.

    This mirrors the observed production boundary: the fresh engine receives the
    complete 78-row canonical transcript, reconciliation reports a nonzero cursor,
    and an old large tool result near the end must not survive replay filtering
    merely because its final storage representation is an externalized marker.
    """
    session_id = "postrestart-full-transcript-rebind"
    conversation_id = "agent:main:telegram:dm:sanitized:thread"
    target_call_id = "call_old_large_result_near_end"
    monkeypatch.setattr(
        "hermes_lcm.ingest_protection.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    config = LCMConfig(
        database_path=str(tmp_path / "rebind.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )

    initial = [{"role": "user", "content": "initial imported context"}]
    for index in range(36):
        call_id = f"call_old_{index:02d}"
        initial.extend(
            [
                _assistant_call(call_id),
                _tool_result(call_id, f"durable result {index}"),
            ]
        )
    target_raw_content = "large session-search result " + ("x" * 29_333)
    initial.extend(
        [
            _assistant_call(target_call_id),
            _tool_result(target_call_id, target_raw_content),
            {"role": "assistant", "content": "old final answer"},
            {"role": "assistant", "content": "old handoff"},
        ]
    )
    assert len(initial) == 77

    seed = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=272000,
    )
    seed.ingest(initial)
    assert seed._store.get_session_count(session_id) == 77
    seed.shutdown()

    # After restart, Hermes' context transport can present the old canonical raw
    # result as a live <persisted-output> marker.  The first LCM storage happened
    # from raw content, so its payload intentionally has no persisted-source
    # provenance even though both representations recover to identical bytes.
    host_storage = tmp_path / "hermes-results"
    host_storage.mkdir()
    persisted_path = host_storage / "call_old_large_result_near_end.txt"
    persisted_path.write_text(target_raw_content, encoding="utf-8")
    preview = target_raw_content[:30]
    persisted_marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(target_raw_content):,} characters, 28.7 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )
    rebound_transcript = [dict(message) for message in initial]
    rebound_transcript[-3] = _tool_result(target_call_id, persisted_marker)

    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=272000,
    )
    rebound.ingest([*rebound_transcript, {"role": "user", "content": "new verify request"}])

    rows = rebound._store.get_session_messages(session_id)
    counts = Counter(
        str(row.get("tool_call_id"))
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id")
    )
    evidence = {
        "target_count": counts[target_call_id],
        "row_count": len(rows),
        "reconciliation": rebound._last_ingest_reconciliation,
    }
    rebound.shutdown()

    assert counts[target_call_id] == 1, evidence
    assert len(rows) == 78, evidence
