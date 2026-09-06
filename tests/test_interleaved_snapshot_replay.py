"""Interleaved full-snapshot replay after Telegram session resume.

Live failure 2026-08-26, session 20260826_152359_ea48c8: a durable session is
resumed and the incoming context re-delivers already-stored rows interleaved
(1:1 alternating replayed/new-shape rows) rather than as a contiguous prefix or
segment. No cursor-based matcher can align it, so the terminal
``persisted ambiguous delta`` fallback (cursor=0) persists the whole batch.
Exact tool identities are replay-safe to suppress. Tool-less identities across
unmatched gaps are ambiguous and must remain preserved: equal user/assistant
text can be a legitimate new turn.
"""
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_existing_session_restart_preserves_tool_less_rows_across_interleaved_gaps(tmp_path):
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

    # Exact tool identities are suppressed, but six tool-less user/assistant
    # rows cross unmatched gaps and therefore remain ambiguous. Preserving them
    # is the fail-closed behavior introduced by the replay-gap safety rules.
    ambiguous_tool_less_count = 6
    assert len(rows) == len(durable) + filler_count + new_count + ambiguous_tool_less_count
    assert tool_row_count == 3
    for idx in range(3):
        assert contents.count(f"durable result {idx}") == 1
        assert contents.count(f"durable answer {idx}") == 2
        assert contents.count(f"resume request {idx}") == 2
    assert new_count == 1
    assert rows[-1]["content"] == "genuinely new follow-up"

    reconciliation = after.get_status()["ingest_reconciliation"]
    assert reconciliation["reason"] == "replayed durable tool-anchored segment"
    assert reconciliation.get("effective_incoming") == (
        filler_count + new_count + ambiguous_tool_less_count
    )


def test_existing_session_expired_marker_preserves_ambiguous_prefix_only(tmp_path):
    """An expired persisted-output pointer must not weaken gap safety.

    Production failure 2026-08-31 resumed a durable Telegram session containing
    an expired persisted-output marker plus older tool-anchored snapshot rows.
    The exact orphan marker and anchored tool cycles remain replay-filterable,
    but a tool-less row separated from the first anchor by the reordered marker
    is ambiguous and must be preserved.
    """
    db_path = tmp_path / "partial-tail-interleaved-replay.db"
    externalized_path = tmp_path / "externalized"
    hermes_home = tmp_path / "hermes"
    config = LCMConfig(
        database_path=str(db_path),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(externalized_path),
    )

    before = LCMEngine(config=config, hermes_home=str(hermes_home))
    before.on_session_start(
        "partial-tail-replay-session",
        platform="telegram",
        conversation_id="partial-tail-replay-conversation",
        context_length=200000,
    )
    durable = []
    for idx in range(3):
        call_id = f"call_partial_{idx}"
        durable.extend(
            [
                {"role": "user", "content": f"partial request {idx}"},
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
                    "content": f"partial durable result {idx}",
                },
                {"role": "assistant", "content": f"partial durable answer {idx}"},
            ]
        )

    persisted_content = "PARTIAL_PERSISTED_OUTPUT:" + ("x" * 1000)
    persisted_path = tmp_path / "partial-persisted-output.txt"
    persisted_path.write_text(persisted_content, encoding="utf-8")
    preview = persisted_content[:40]
    marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(persisted_content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )
    marker_call_id = "call_partial_marker"
    # The gateway spillover file has expired before LCM sees the resumed
    # snapshot. The marker itself remains byte-identical and already durable.
    persisted_path.unlink()
    durable.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": marker_call_id,
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": marker_call_id,
                "tool_name": "inspect",
                "content": marker,
            },
        ]
    )
    before._ingest_messages(durable)
    before.shutdown()

    after = LCMEngine(config=config, hermes_home=str(hermes_home))
    after.on_session_start(
        "partial-tail-replay-session",
        platform="telegram",
        conversation_id="partial-tail-replay-conversation",
        context_length=200000,
    )
    replayed_tool_cycles = durable[:12]
    incoming = [
        durable[-1],
        *replayed_tool_cycles,
        {"role": "user", "content": "partial genuinely new follow-up"},
    ]

    after._ingest_messages(incoming)

    rows = after._store.get_session_messages("partial-tail-replay-session")
    contents = [row["content"] for row in rows]
    tool_row_count = sum(
        1
        for row in rows
        if row.get("tool_call_id") in {
            "call_partial_0",
            "call_partial_1",
            "call_partial_2",
        }
    )
    evidence = {
        "row_count": len(rows),
        "tool_row_count": tool_row_count,
        "reconciliation": after.get_status()["ingest_reconciliation"],
    }
    after.shutdown()

    assert len(rows) == len(durable) + 2, evidence
    assert tool_row_count == 3, evidence
    for idx in range(3):
        assert contents.count(f"partial durable result {idx}") == 1, evidence
        assert contents.count(f"partial durable answer {idx}") == 1, evidence
        assert contents.count(f"partial request {idx}") == (2 if idx == 0 else 1), evidence
    assert contents.count(marker) == 1, evidence
    assert contents.count("partial genuinely new follow-up") == 1, evidence
    assert rows[-1]["content"] == "partial genuinely new follow-up", evidence


def _run_unproven_expired_marker_case(
    tmp_path, *, durable_order, incoming_order, gap_after, marker_position="head"
):
    db_path = tmp_path / "unproven-expired-marker.db"
    config = LCMConfig(
        database_path=str(db_path),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    session_id = "unproven-expired-marker-session"
    conversation_id = "unproven-expired-marker-conversation"

    persisted_content = "UNPROVEN_EXPIRED_MARKER:" + ("x" * 1000)
    persisted_path = tmp_path / "expired-output.txt"
    preview = persisted_content[:40]
    marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(persisted_content):,} characters, 1.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )

    def cycle(name):
        call_id = f"call_{name}"
        return [
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
                "content": f"durable {name} result",
            },
        ]

    marker_call_id = "call_expired_marker"
    durable_marker = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": marker_call_id,
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": marker_call_id,
            "tool_name": "inspect",
            "content": marker,
        },
    ]
    cycles = {name: cycle(name) for name in {"a1", "a2", "a3"}}
    durable = []
    for name in durable_order:
        durable.extend(cycles[name])
    durable.extend(durable_marker)

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    before.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    before._ingest_messages(durable)
    before.shutdown()

    incoming = [durable_marker[-1]] if marker_position == "head" else []
    if gap_after == "marker":
        incoming.append({"role": "user", "content": "unmatched new gap"})
    for name in incoming_order:
        incoming.extend(cycles[name])
        if gap_after == name:
            incoming.append({"role": "user", "content": "unmatched new gap"})
    if marker_position == "tail":
        incoming.append(durable_marker[-1])

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    after.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    after._ingest_messages(incoming)
    rows = after._store.get_session_messages(session_id)
    after.shutdown()
    return rows, marker


def test_expired_marker_is_preserved_when_later_anchors_are_out_of_order(tmp_path):
    rows, marker = _run_unproven_expired_marker_case(
        tmp_path,
        durable_order=("a2", "a1", "a3"),
        incoming_order=("a1", "a2", "a3"),
        gap_after=None,
    )

    assert sum(row["content"] == marker for row in rows) == 2


def test_expired_marker_is_preserved_across_unmatched_gap_before_later_anchors(tmp_path):
    rows, marker = _run_unproven_expired_marker_case(
        tmp_path,
        durable_order=("a1", "a2", "a3"),
        incoming_order=("a1", "a2", "a3"),
        gap_after="marker",
    )

    assert sum(row["content"] == marker for row in rows) == 2
    assert sum(row["content"] == "unmatched new gap" for row in rows) == 1


def test_expired_marker_is_preserved_when_gap_separates_later_anchors(tmp_path):
    rows, marker = _run_unproven_expired_marker_case(
        tmp_path,
        durable_order=("a1", "a2", "a3"),
        incoming_order=("a1", "a2", "a3"),
        gap_after="a1",
    )

    assert sum(row["content"] == marker for row in rows) == 2
    assert sum(row["content"] == "unmatched new gap" for row in rows) == 1


def test_expired_marker_at_incoming_tail_is_not_anchored_without_later_proof(tmp_path):
    rows, marker = _run_unproven_expired_marker_case(
        tmp_path,
        durable_order=("a1", "a2", "a3"),
        incoming_order=("a1", "a2", "a3"),
        gap_after=None,
        marker_position="tail",
    )

    assert sum(row["content"] == marker for row in rows) == 2


def test_repeated_exact_generation_anchor_cannot_supply_three_anchor_proof(tmp_path):
    db_path = tmp_path / "repeated-generation-anchor.db"
    config = LCMConfig(
        database_path=str(db_path),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    session_id = "repeated-generation-anchor-session"
    conversation_id = "repeated-generation-anchor-conversation"

    def marker(path, content):
        preview = content[:40]
        return (
            "<persisted-output>\n"
            f"This tool result was too large ({len(content):,} characters, 1.0 KB).\n"
            f"Full output saved to: {path}\n"
            "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
            f"Preview (first {len(preview)} chars):\n"
            f"{preview}\n...\n"
            "</persisted-output>"
        )

    anchor_content = "EXACT_GENERATION_ANCHOR:" + ("a" * 1000)
    anchor_path = tmp_path / "exact-anchor.txt"
    anchor_path.write_text(anchor_content, encoding="utf-8")
    anchor_call_id = "call_exact_anchor"
    anchor_cycle = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": anchor_call_id,
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": anchor_call_id,
            "tool_name": "inspect",
            "content": marker(anchor_path, anchor_content),
        },
    ]

    expired_content = "REPEATED_ANCHOR_EXPIRED:" + ("x" * 1000)
    expired_path = tmp_path / "expired-marker.txt"
    expired_marker = marker(expired_path, expired_content)
    expired_call_id = "call_repeated_anchor_expired"
    expired_cycle = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": expired_call_id,
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": expired_call_id,
            "tool_name": "inspect",
            "content": expired_marker,
        },
    ]

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    before.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    before._ingest_messages([*anchor_cycle, *expired_cycle])
    before.shutdown()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    after.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    after._ingest_messages([expired_cycle[-1], *anchor_cycle, *anchor_cycle, *anchor_cycle])
    rows = after._store.get_session_messages(session_id)
    after.shutdown()

    assert sum(row["content"] == expired_marker for row in rows) == 2
    assert sum(row.get("tool_call_id") == anchor_call_id for row in rows) == 1
