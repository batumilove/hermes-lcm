"""Portable full-shape regression for post-restart middle raw-tool replay."""

import json
from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import protect_messages_for_ingest


_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "post_restart_middle_raw_replay.json").read_text(
        encoding="utf-8"
    )
)["case"]


def test_fresh_rebind_does_not_reappend_middle_raw_tool_after_protection(tmp_path):
    case = _FIXTURE
    db_path = tmp_path / "lcm.db"
    hermes_home = tmp_path / "hermes"
    externalized_path = hermes_home / "lcm-large-outputs"
    config = LCMConfig(
        database_path=str(db_path),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=12_000,
        large_output_externalization_path=str(externalized_path),
    )

    seed_messages = [dict(message) for message in case["seed_messages"]]
    incoming_messages = [dict(message) for message in case["incoming_messages"]]
    target_call_id = case["target_tool_call_id"]
    target_raw = case["target_raw_content"]
    target_template = next(
        message
        for message in incoming_messages
        if message.get("tool_call_id") == target_call_id
    )
    protected_target = protect_messages_for_ingest(
        [{**target_template, "content": target_raw}],
        config=config,
        hermes_home=str(hermes_home),
        session_id=case["session_id"],
    )[0]["content"]
    for message in seed_messages:
        if (
            message.get("tool_call_id") == target_call_id
            and message.get("content") == "__TARGET_STUB__"
        ):
            message["content"] = protected_target
    for message in incoming_messages:
        if (
            message.get("tool_call_id") == target_call_id
            and message.get("content") == "__TARGET_RAW__"
        ):
            message["content"] = target_raw

    before = LCMEngine(config=config, hermes_home=str(hermes_home))
    before.on_session_start(
        case["session_id"],
        platform="telegram",
        conversation_id=case["conversation_id"],
        context_length=200_000,
    )
    before._store._append_protected_batch(
        case["session_id"],
        seed_messages,
        source="telegram",
        conversation_id=case["conversation_id"],
    )
    before.shutdown()

    expected_old_tool_count = sum(
        message.get("tool_call_id") == target_call_id for message in seed_messages
    )
    after = LCMEngine(config=config, hermes_home=str(hermes_home))
    after.on_session_start(
        case["session_id"],
        platform="telegram",
        conversation_id=case["conversation_id"],
        context_length=200_000,
    )
    assert after._ingest_cursor_needs_reconcile
    after._ingest_messages(incoming_messages)

    rows = after._store.get_session_messages(case["session_id"])
    old_tool_count = sum(row.get("tool_call_id") == target_call_id for row in rows)
    new_user_count = sum(
        row.get("role") == "user" and row.get("content") == case["new_user_content"]
        for row in rows
    )
    evidence = {
        "expected_old_tool_count": expected_old_tool_count,
        "old_tool_count": old_tool_count,
        "new_user_count": new_user_count,
        "seed_count": len(seed_messages),
        "incoming_count": len(incoming_messages),
        "stored_count": len(rows),
        "reconciliation": after.get_status()["ingest_reconciliation"],
    }
    after.shutdown()

    assert old_tool_count == expected_old_tool_count, evidence
    assert new_user_count == 1, evidence


def _prepare_rebind_case(tmp_path):
    case = _FIXTURE
    db_path = tmp_path / "lcm.db"
    hermes_home = tmp_path / "hermes"
    config = LCMConfig(
        database_path=str(db_path),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=12_000,
        large_output_externalization_path=str(hermes_home / "lcm-large-outputs"),
    )
    seed_messages = [dict(message) for message in case["seed_messages"]]
    incoming_messages = [dict(message) for message in case["incoming_messages"]]
    target_call_id = case["target_tool_call_id"]
    target_raw = case["target_raw_content"]
    target_template = next(
        message
        for message in incoming_messages
        if message.get("tool_call_id") == target_call_id
    )
    protected_target = protect_messages_for_ingest(
        [{**target_template, "content": target_raw}],
        config=config,
        hermes_home=str(hermes_home),
        session_id=case["session_id"],
    )[0]["content"]
    for message in seed_messages:
        if (
            message.get("tool_call_id") == target_call_id
            and message.get("content") == "__TARGET_STUB__"
        ):
            message["content"] = protected_target
    for message in incoming_messages:
        if (
            message.get("tool_call_id") == target_call_id
            and message.get("content") == "__TARGET_RAW__"
        ):
            message["content"] = target_raw
    return case, config, hermes_home, seed_messages, incoming_messages


def _seed_then_rebind(case, config, hermes_home, seed_messages, incoming_messages):
    before = LCMEngine(config=config, hermes_home=str(hermes_home))
    before.on_session_start(
        case["session_id"],
        platform="telegram",
        conversation_id=case["conversation_id"],
        context_length=200_000,
    )
    before._store._append_protected_batch(
        case["session_id"],
        seed_messages,
        source="telegram",
        conversation_id=case["conversation_id"],
    )
    before.shutdown()

    after = LCMEngine(config=config, hermes_home=str(hermes_home))
    after.on_session_start(
        case["session_id"],
        platform="telegram",
        conversation_id=case["conversation_id"],
        context_length=200_000,
    )
    assert after._ingest_cursor_needs_reconcile
    after._ingest_messages(incoming_messages)
    rows = after._store.get_session_messages(case["session_id"])
    after.shutdown()
    return rows


def test_fresh_rebind_preserves_nonadjacent_repeated_user_after_tool_anchor(tmp_path):
    case, config, hermes_home, seed_messages, incoming_messages = _prepare_rebind_case(
        tmp_path
    )
    target_seed_index = next(
        index
        for index, message in enumerate(seed_messages)
        if message.get("tool_call_id") == case["target_tool_call_id"]
    )
    repeated_content = next(
        message["content"]
        for message in seed_messages[:target_seed_index]
        if message.get("role") == "user"
    )
    incoming_messages[:] = [
        message
        for message in incoming_messages
        if not (
            message.get("role") == "user"
            and message.get("content") == repeated_content
        )
    ]
    original_count = sum(
        message.get("role") == "user" and message.get("content") == repeated_content
        for message in seed_messages
    )
    incoming_messages.append({"role": "user", "content": repeated_content})

    rows = _seed_then_rebind(
        case, config, hermes_home, seed_messages, incoming_messages
    )

    assert (
        sum(
            row.get("role") == "user" and row.get("content") == repeated_content
            for row in rows
        )
        == original_count + 1
    )


def test_fresh_rebind_preserves_scaffold_tool_with_changed_explicit_name(tmp_path):
    case, config, hermes_home, seed_messages, incoming_messages = _prepare_rebind_case(
        tmp_path
    )
    target = next(
        message
        for message in incoming_messages
        if message.get("tool_call_id") == case["target_tool_call_id"]
    )
    target["tool_name"] = f"{target.get('tool_name') or 'tool'}_changed"
    original_count = sum(
        message.get("tool_call_id") == case["target_tool_call_id"]
        for message in seed_messages
    )

    rows = _seed_then_rebind(
        case, config, hermes_home, seed_messages, incoming_messages
    )

    target_rows = [
        row
        for row in rows
        if row.get("tool_call_id") == case["target_tool_call_id"]
    ]
    assert len(target_rows) == original_count + 1
    assert sum(row.get("tool_name") == target["tool_name"] for row in target_rows) == 1
