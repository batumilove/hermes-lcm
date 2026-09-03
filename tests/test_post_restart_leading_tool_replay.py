"""Regression probe for leading tool-only replay after fresh-process rebind."""

from collections import Counter
from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_fresh_process_rebind_filters_leading_externalized_tool_only_replay(tmp_path):
    session_id = "production-shaped-post-restart-leading-tool"
    conversation_id = "agent:main:telegram:dm:sanitized:thread"
    old_call_id = "call_sanitized_old_externalized"
    new_call_id = "call_sanitized_new_standalone"
    config = LCMConfig(
        database_path=str(tmp_path / "post-restart-leading-tool.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )

    raw_old_tool = {
        "role": "tool",
        "tool_call_id": old_call_id,
        "tool_name": "session_search",
        "content": "old session_search result " + ("x" * 4096),
    }
    import tempfile

    persisted_dir = Path(tempfile.gettempdir()) / "hermes-results"
    persisted_dir.mkdir(exist_ok=True)
    persisted_path = persisted_dir / "call_sanitized_old_externalized.txt"
    persisted_path.write_text(raw_old_tool["content"], encoding="utf-8")
    preview = raw_old_tool["content"][:64]
    persisted_marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(raw_old_tool['content']):,} characters, 4.0 KB).\n"
        f"Full output saved to: {persisted_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        f"{preview}\n...\n"
        "</persisted-output>"
    )
    host_old_tool = dict(raw_old_tool, content=persisted_marker)
    seed = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    seed.ingest([host_old_tool])
    stored_old_tool = next(
        row
        for row in seed._store.get_session_messages(session_id)
        if row.get("tool_call_id") == old_call_id
    )
    assert str(stored_old_tool["content"]).startswith("[Externalized tool output:"), repr(stored_old_tool["content"])
    seed._store.append_batch(
        session_id,
        [
            {"role": "user", "content": f"durable filler row {index}"}
            for index in range(4200)
        ],
        source="telegram",
        conversation_id=conversation_id,
    )
    seed.shutdown()

    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    assert rebound._ingest_cursor == 0
    assert rebound._ingest_cursor_needs_reconcile

    # The production failure proved that the earlier cursor/tail scanners can
    # miss a row whose identity converges only after final ingest protection.
    # Model that bypass while leaving a later protected-form defense available.
    original_replay_scan = rebound._find_tool_anchored_replay_indexes
    replay_scan_calls = 0

    def miss_pre_protection_scans(*args, **kwargs):
        nonlocal replay_scan_calls
        replay_scan_calls += 1
        if replay_scan_calls <= 2:
            return set(), 0
        return original_replay_scan(*args, **kwargs)

    rebound._find_tool_anchored_replay_indexes = miss_pre_protection_scans

    new_tool = {
        "role": "tool",
        "tool_call_id": new_call_id,
        "tool_name": "terminal",
        "content": "genuinely new standalone result",
    }
    rebound.ingest(
        [
            {
                "role": "tool",
                "tool_call_id": old_call_id,
                "tool_name": "session_search",
                "content": persisted_marker,
            },
            new_tool,
            {"role": "user", "content": "genuinely new post-restart request"},
        ]
    )

    rows = rebound._store.get_session_messages(session_id)
    tool_counts = Counter(
        str(row.get("tool_call_id"))
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id")
    )
    evidence = {
        "old_tool_count": tool_counts[old_call_id],
        "new_tool_count": tool_counts[new_call_id],
        "new_user_count": sum(
            row.get("role") == "user"
            and row.get("content") == "genuinely new post-restart request"
            for row in rows
        ),
        "reconciliation": rebound._last_ingest_reconciliation,
    }
    rebound.shutdown()

    assert evidence["old_tool_count"] == 1, evidence
    assert evidence["new_tool_count"] == 1, evidence
    assert evidence["new_user_count"] == 1, evidence
