"""Regression for a full state transcript replayed after a fresh-process rebind."""

from collections import Counter

import pytest

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


@pytest.mark.parametrize("delivery_path", ["direct", "deferred"])
def test_final_form_replay_filter_records_session_end_receipt(
    tmp_path,
    monkeypatch,
    delivery_path,
):
    session_id = "postrestart-final-filter-session-end"
    conversation_id = "agent:main:telegram:dm:sanitized:receipt"
    call_id = "call_replayed_at_session_end"
    raw_content = "large replayed result " + ("x" * 1024)
    monkeypatch.setattr(
        "hermes_lcm.ingest_protection.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    config = LCMConfig(
        database_path=str(tmp_path / "receipt.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )

    seed = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=272000,
    )
    seed.ingest([_tool_result(call_id, raw_content)])
    assert seed._store.get_session_count(session_id) == 1
    seed.shutdown()

    host_storage = tmp_path / "hermes-results"
    host_storage.mkdir()
    persisted_path = host_storage / "call_replayed_at_session_end.txt"
    persisted_path.write_text(raw_content, encoding="utf-8")
    persisted_marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(raw_content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{raw_content[:30]}\n...\n"
        "</persisted-output>"
    )

    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=272000,
    )
    monkeypatch.setattr(
        rebound,
        "_session_end_tool_replay_plan",
        lambda *_args, **_kwargs: (set(), {}),
    )
    recorded_receipts = []
    original_record_receipt = rebound._store.record_session_end_ingest_receipt

    def record_receipt(intent_sha256, **kwargs):
        original_record_receipt(intent_sha256, **kwargs)
        recorded_receipts.append(intent_sha256)

    monkeypatch.setattr(
        rebound._store,
        "record_session_end_ingest_receipt",
        record_receipt,
    )
    final_snapshot = [_tool_result(call_id, persisted_marker)]
    pending = None
    if delivery_path == "direct":
        rebound.on_session_end(session_id, final_snapshot)
    else:
        pending = rebound._persist_session_end_intent(
            session_id,
            final_snapshot,
            ingest_cursor=rebound._ingest_cursor,
        )
        rebound._drain_one_session_end_intent(pending)

    rows = rebound._store.get_session_messages(session_id)
    evidence = {
        "row_count": len(rows),
        "reconciliation": rebound._last_ingest_reconciliation,
    }
    receipt_recorded = (
        len(recorded_receipts) == 1
        and rebound._store.has_session_end_ingest_receipt(recorded_receipts[0])
    )
    rebound.shutdown()

    assert len(rows) == 1, evidence
    if delivery_path == "direct":
        # Direct delivery first traverses the canonical strict-replay scan,
        # which proves and filters this persisted output by its durable tool
        # anchor before the later persisted-output fallback is reached.
        assert evidence["reconciliation"] == {
            "action": "filtered replay",
            "reason": "replayed durable tool-anchored segment",
            "cursor": 0,
            "incoming": 1,
            "session_count": 1,
            "stored_tail_count": 1,
            "effective_incoming": 0,
        }, evidence
    else:
        assert pending is not None and not pending.exists(), evidence
    assert receipt_recorded, evidence


def test_deferred_final_form_filter_uses_intent_session_identity_when_bound_elsewhere(
    tmp_path,
    monkeypatch,
):
    ended_session_id = "postrestart-deferred-ended-session"
    ended_conversation_id = "agent:main:telegram:dm:sanitized:ended"
    active_session_id = "postrestart-active-successor-session"
    active_conversation_id = "agent:main:telegram:dm:sanitized:successor"
    call_id = "call_replayed_from_ended_session"
    raw_content = "large ended-session result " + ("x" * 1024)
    monkeypatch.setattr(
        "hermes_lcm.ingest_protection.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    config = LCMConfig(
        database_path=str(tmp_path / "off-current-receipt.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )

    host_storage = tmp_path / "hermes-results"
    host_storage.mkdir()
    persisted_path = host_storage / "call_replayed_from_ended_session.txt"
    persisted_path.write_text(raw_content, encoding="utf-8")
    persisted_marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(raw_content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{raw_content[:30]}\n...\n"
        "</persisted-output>"
    )

    seed = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    seed.on_session_start(
        ended_session_id,
        platform="telegram",
        conversation_id=ended_conversation_id,
        context_length=272000,
    )
    seed.ingest([_tool_result(call_id, raw_content)])
    seed.shutdown()

    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound._session_id = active_session_id
    rebound._conversation_id = active_conversation_id
    monkeypatch.setattr(
        rebound,
        "_session_end_tool_replay_plan",
        lambda *_args, **_kwargs: (set(), {}),
    )
    appended = rebound._append_off_current_session_end_suffix(
        ended_session_id,
        [_tool_result(call_id, persisted_marker)],
        source="telegram",
        conversation_id=ended_conversation_id,
    )

    rows = rebound._store.get_session_messages(ended_session_id)
    tool_rows = [
        row
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id") == call_id
    ]
    evidence = {
        "row_count": len(rows),
        "tool_row_count": len(tool_rows),
        "appended": appended,
    }
    rebound.shutdown()

    assert len(rows) == 1, evidence
    assert len(tool_rows) == 1, evidence
    assert appended == [], evidence


def test_lossy_durable_replay_uses_explicit_session_for_exact_generation_lookup(
    tmp_path,
    monkeypatch,
):
    ended_session_id = "postrestart-lossy-ended-session"
    active_session_id = "postrestart-lossy-active-session"
    call_id = "call_lossy_replayed_from_ended_session"
    raw_content = "password=super-secret " + ("x" * 1024)
    monkeypatch.setattr(
        "hermes_lcm.ingest_protection.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    config = LCMConfig(
        database_path=str(tmp_path / "lossy-explicit-session.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    host_storage = tmp_path / "hermes-results"
    host_storage.mkdir()
    persisted_path = host_storage / "call_lossy_replayed_from_ended_session.txt"
    persisted_path.write_text(raw_content, encoding="utf-8")
    persisted_marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(raw_content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{raw_content[:30]}\n...\n"
        "</persisted-output>"
    )
    lossy_content = "[LCM sensitive redaction: name=password_assignment]"
    lookup_session_ids = []

    def find_durable_content(**kwargs):
        lookup_session_ids.append(kwargs["session_id"])
        if len(lookup_session_ids) == 1:
            return lossy_content
        if len(lookup_session_ids) == 2:
            return None
        return lossy_content if kwargs["session_id"] == ended_session_id else None

    monkeypatch.setattr(
        "hermes_lcm.reconcile.find_externalized_tool_result_content_for_call",
        find_durable_content,
    )
    monkeypatch.setattr(
        "hermes_lcm.reconcile.redact_sensitive_value",
        lambda *_args, **_kwargs: lossy_content,
    )
    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound._session_id = active_session_id
    monkeypatch.setattr(
        rebound,
        "_recovered_content_matches_durable_identity",
        lambda *_args, **_kwargs: True,
    )

    replayed = rebound._has_durable_persisted_output_replay_identity(
        _tool_result(call_id, persisted_marker),
        session_id=ended_session_id,
    )
    rebound.shutdown()

    assert replayed is True, lookup_session_ids
    assert lookup_session_ids == [ended_session_id, ended_session_id, ended_session_id]


def test_legacy_replay_uses_message_conversation_when_explicit_value_is_omitted(
    tmp_path,
    monkeypatch,
):
    ended_session_id = "postrestart-legacy-ended-session"
    ended_conversation_id = "postrestart-legacy-ended-conversation"
    active_conversation_id = "postrestart-legacy-active-conversation"
    call_id = "call_legacy_replayed_from_ended_conversation"
    raw_content = "legacy durable result " + ("x" * 1024)
    config = LCMConfig(database_path=str(tmp_path / "legacy-message-conversation.db"))
    persisted_path = tmp_path / "call_legacy_replayed_from_ended_conversation.txt"
    persisted_path.write_text(raw_content, encoding="utf-8")
    persisted_marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(raw_content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{raw_content[:30]}\n...\n"
        "</persisted-output>"
    )
    message = _tool_result(call_id, persisted_marker)
    message["conversation_id"] = ended_conversation_id
    lookup_conversation_ids = []

    monkeypatch.setattr(
        "hermes_lcm.reconcile._expected_persisted_output_chars",
        lambda _content: len(raw_content),
    )
    monkeypatch.setattr(
        "hermes_lcm.reconcile._persisted_output_saved_path",
        lambda _content: str(persisted_path),
    )
    monkeypatch.setattr(
        "hermes_lcm.reconcile.recover_hermes_persisted_output_with_file_stat",
        lambda _content: (
            raw_content,
            {"size": len(raw_content), "mtime_ns": 1, "ctime_ns": 1},
        ),
    )
    monkeypatch.setattr(
        "hermes_lcm.reconcile.find_externalized_tool_result_content_for_call",
        lambda **_kwargs: None,
    )
    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound._session_id = "postrestart-legacy-active-session"
    rebound._conversation_id = active_conversation_id
    monkeypatch.setattr(
        rebound,
        "_persisted_output_marker_replay_proof",
        lambda _content: ("preview-sha256", False),
    )

    def find_neighborhoods(session_id, call_ids, *, conversation_id=None):
        lookup_conversation_ids.append(conversation_id)
        if conversation_id != ended_conversation_id:
            return []
        return [_tool_result(call_id, raw_content)]

    monkeypatch.setattr(
        rebound._store,
        "get_tool_call_replay_neighborhoods",
        find_neighborhoods,
    )

    replayed = rebound._has_durable_persisted_output_replay_identity(
        message,
        session_id=ended_session_id,
    )
    rebound.shutdown()

    assert replayed is True, lookup_conversation_ids
    assert lookup_conversation_ids == [ended_conversation_id]


@pytest.mark.parametrize(
    ("intervening_messages", "expected_row_count", "expected_tool_rows"),
    [
        ([], 3, 2),
        ([{"role": "user", "content": "intervening turn"}], 2, 1),
    ],
)
def test_deferred_final_form_filter_requires_original_tool_call_adjacency(
    tmp_path,
    monkeypatch,
    intervening_messages,
    expected_row_count,
    expected_tool_rows,
):
    session_id = "postrestart-deferred-original-adjacency"
    conversation_id = "agent:main:telegram:dm:sanitized:adjacency"
    call_id = "call_reused_at_deferred_session_end"
    raw_content = "large durable result " + ("x" * 1024)
    monkeypatch.setattr(
        "hermes_lcm.ingest_protection.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    config = LCMConfig(
        database_path=str(tmp_path / "adjacency.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )

    seed = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=272000,
    )
    seed.ingest([_tool_result(call_id, raw_content)])
    seed.shutdown()

    host_storage = tmp_path / "hermes-results"
    host_storage.mkdir()
    persisted_path = host_storage / "call_reused_at_deferred_session_end.txt"
    persisted_path.write_text(raw_content, encoding="utf-8")
    persisted_marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(raw_content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{raw_content[:30]}\n...\n"
        "</persisted-output>"
    )

    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=272000,
    )
    monkeypatch.setattr(
        rebound,
        "_session_end_tool_replay_plan",
        lambda _session_id, messages, **_kwargs: (
            {1} if len(messages) == 3 else set(),
            {},
        ),
    )
    snapshot = [
        _assistant_call(call_id),
        *intervening_messages,
        _tool_result(call_id, persisted_marker),
    ]
    pending = rebound._persist_session_end_intent(
        session_id,
        snapshot,
        ingest_cursor=rebound._ingest_cursor,
    )
    rebound._drain_one_session_end_intent(pending)

    rows = rebound._store.get_session_messages(session_id)
    tool_rows = [
        row
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id") == call_id
    ]
    evidence = {
        "row_count": len(rows),
        "tool_row_count": len(tool_rows),
        "pending_exists": pending.exists(),
    }
    rebound.shutdown()

    assert len(rows) == expected_row_count, evidence
    assert len(tool_rows) == expected_tool_rows, evidence
    assert not pending.exists(), evidence
