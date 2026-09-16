"""Production-shaped RED contracts for raw persisted-output replay bursts.

These fixtures model the observed live discriminator: exact raw Hermes
``<persisted-output>`` markers with tool names reach admission after their
source file generation changes.  A multi-marker replay burst is sufficiently
specific to suppress; a single unpaired marker remains ambiguous.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter

import hermes_lcm.reconcile as reconcile_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import recover_hermes_persisted_output_with_file_stat
from hermes_lcm.reconcile import _proven_distinct_durable_burst_offsets


def _marker(path, raw: str) -> str:
    return (
        "<persisted-output>\n"
        f"This tool result was too large ({len(raw):,} characters, 1.0 KB).\n"
        f"Full output saved to: {path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{raw[:30]}\n...\n"
        "</persisted-output>"
    )


def _tool(call_id: str, tool_name: str, marker: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "tool_name": tool_name,
        "content": marker,
    }


def _assistant_call(call_id: str, tool_name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
    }


def _replace_same_bytes_with_new_generation(path, raw: str) -> None:
    before = path.stat()
    path.unlink()
    path.write_text(raw, encoding="utf-8")
    target = max(before.st_mtime_ns + 1_000_000_000, path.stat().st_mtime_ns + 1_000_000_000)
    os.utime(path, ns=(target, target))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )


def _fixture(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=256,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    host = tmp_path / "hermes-results"
    host.mkdir()
    rows = []
    raw_by_path = {}
    for index in range(2):
        call_id = f"call_replayed_marker_{index}"
        tool_name = "session_search" if index == 0 else "read_file"
        raw = f"durable raw result {index} " + (chr(65 + index) * 1024)
        path = host / f"result-{index}.txt"
        path.write_text(raw, encoding="utf-8")
        raw_by_path[path] = raw
        rows.append(_tool(call_id, tool_name, _marker(path, raw)))
    return config, rows, raw_by_path


def _tool_counts(rows) -> Counter:
    return Counter(
        str(row.get("tool_call_id"))
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id")
    )


def test_burst_proof_requires_distinct_durable_occurrences():
    identity_a = ("tool", "call", "session_search", "hash-a", "")
    identity_b = ("tool", "call", "session_search", "hash-b", "")

    assert (
        _proven_distinct_durable_burst_offsets(
            [(0, identity_a, {7}), (1, identity_b, {7})]
        )
        == set()
    )
    assert _proven_distinct_durable_burst_offsets(
        [(0, identity_a, {7}), (1, identity_b, {8})]
    ) == {0, 1}
    assert _proven_distinct_durable_burst_offsets(
        [(0, identity_a, {7}), (1, identity_a, {8}), (2, identity_b, {7})]
    ) == {1, 2}

    oversized = [
        (
            offset,
            ("tool", f"call-{offset}", "session_search", f"hash-{offset}", ""),
            {offset},
        )
        for offset in range(129)
    ]
    assert _proven_distinct_durable_burst_offsets(oversized) == set()


def test_per_turn_suppresses_two_unpaired_exact_markers_after_source_generation_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config, replayed_tools, raw_by_path = _fixture(tmp_path)
    session_id = "duplicate-admission-per-turn-burst"
    conversation_id = "agent:main:telegram:dm:sanitized:per-turn"

    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    engine.ingest(replayed_tools)
    for path, raw in raw_by_path.items():
        _replace_same_bytes_with_new_generation(path, raw)
    assert all(
        recover_hermes_persisted_output_with_file_stat(message["content"]) is not None
        for message in replayed_tools
    )

    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    fresh_call = "call_genuinely_new_per_turn"
    incoming = [
        replayed_tools[0],
        {"role": "user", "content": "genuinely new request"},
        replayed_tools[1],
        _assistant_call(fresh_call, "lcm_status"),
        _tool(fresh_call, "lcm_status", "genuinely new tool result"),
        {"role": "assistant", "content": "genuinely new response"},
    ]
    engine.ingest(incoming)

    rows = engine._store.get_session_messages(session_id)
    counts = _tool_counts(rows)
    contents = [row.get("content") for row in rows]
    evidence = {
        "counts": dict(counts),
        "row_count": len(rows),
        "reconciliation": engine._last_ingest_reconciliation,
    }
    engine.shutdown()

    assert counts["call_replayed_marker_0"] == 1, evidence
    assert counts["call_replayed_marker_1"] == 1, evidence
    assert counts[fresh_call] == 1, evidence
    assert contents.count("genuinely new request") == 1, evidence
    assert contents.count("genuinely new response") == 1, evidence


def test_per_turn_suppresses_distinct_marker_identities_that_reuse_one_call_id(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config, replayed_tools, raw_by_path = _fixture(tmp_path)
    shared_call_id = "call_reused_across_replay_burst"
    for message in replayed_tools:
        message["tool_call_id"] = shared_call_id
    session_id = "duplicate-admission-per-turn-shared-call-id"

    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id="agent:main:telegram:dm:sanitized:shared-call-id",
        context_length=200000,
    )
    engine.ingest(replayed_tools)
    for path, raw in raw_by_path.items():
        _replace_same_bytes_with_new_generation(path, raw)
    assert all(
        recover_hermes_persisted_output_with_file_stat(message["content"]) is not None
        for message in replayed_tools
    )

    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    engine.ingest(replayed_tools)

    rows = engine._store.get_session_messages(session_id)
    matching_rows = [
        row
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id") == shared_call_id
    ]
    evidence = {
        "matching_contents": [row.get("content") for row in matching_rows],
        "row_count": len(rows),
        "reconciliation": engine._last_ingest_reconciliation,
    }
    engine.shutdown()

    assert len(matching_rows) == 2, evidence


def test_two_marker_identities_cannot_reuse_one_durable_occurrence_as_burst_proof(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config, replayed_tools, raw_by_path = _fixture(tmp_path)
    shared_call_id = "call_one_durable_occurrence"
    first, second = replayed_tools
    first["tool_call_id"] = shared_call_id
    second["tool_call_id"] = shared_call_id
    second["tool_name"] = first["tool_name"]
    session_id = "duplicate-admission-one-durable-occurrence"

    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id="agent:main:telegram:dm:sanitized:one-durable-occurrence",
        context_length=200000,
    )
    engine.ingest([first])

    stored = engine._store.get_session_messages(session_id)
    assert len(stored) == 1
    ref = stored[0]["content"].rsplit("ref=", 1)[1].rstrip("]")
    payload_path = tmp_path / "externalized" / ref
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    second_path = next(path for path in raw_by_path if str(path) in second["content"])
    first_path = next(path for path in raw_by_path if str(path) in first["content"])
    second_raw = raw_by_path[first_path]
    second_path.write_text(second_raw, encoding="utf-8")
    second["content"] = _marker(second_path, second_raw)
    raw_by_path[second_path] = second_raw
    second_stat = second_path.stat()
    payload["persisted_output_markers"].append(
        {
            "source_path": str(second_path),
            "expected_chars": len(second_raw),
            "file_size": second_stat.st_size,
            "file_mtime_ns": second_stat.st_mtime_ns,
            "file_ctime_ns": second_stat.st_ctime_ns,
        }
    )
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        reconcile_module,
        "load_externalized_payload",
        lambda *args, **kwargs: payload,
    )

    for path, raw in raw_by_path.items():
        _replace_same_bytes_with_new_generation(path, raw)
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    engine.ingest([first, second])

    rows = engine._store.get_session_messages(session_id)
    matching_rows = [
        row
        for row in rows
        if row.get("role") == "tool" and row.get("tool_call_id") == shared_call_id
    ]
    evidence = {
        "matching_contents": [row.get("content") for row in matching_rows],
        "row_count": len(rows),
        "reconciliation": engine._last_ingest_reconciliation,
    }
    engine.shutdown()

    # One durable row cannot prove a two-occurrence replay burst. Both incoming
    # marker identities therefore remain ambiguous and are retained.
    assert len(matching_rows) == 3, evidence


def test_session_end_suppresses_same_two_marker_burst_and_retains_new_suffix(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config, replayed_tools, raw_by_path = _fixture(tmp_path)
    session_id = "duplicate-admission-session-end-burst"
    conversation_id = "agent:main:telegram:dm:sanitized:session-end"

    seed = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    seed.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    seed.ingest(replayed_tools)
    seed.shutdown()
    for path, raw in raw_by_path.items():
        _replace_same_bytes_with_new_generation(path, raw)
    assert all(
        recover_hermes_persisted_output_with_file_stat(message["content"]) is not None
        for message in replayed_tools
    )

    rebound = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    rebound.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    fresh_call = "call_genuinely_new_session_end"
    final_snapshot = [
        replayed_tools[0],
        {"role": "user", "content": "genuinely new terminal request"},
        replayed_tools[1],
        _assistant_call(fresh_call, "lcm_status"),
        _tool(fresh_call, "lcm_status", "genuinely new terminal tool result"),
        {"role": "assistant", "content": "genuinely new terminal response"},
    ]
    rebound.on_session_end(session_id, final_snapshot)

    rows = rebound._store.get_session_messages(session_id)
    counts = _tool_counts(rows)
    contents = [row.get("content") for row in rows]
    evidence = {
        "counts": dict(counts),
        "row_count": len(rows),
        "reconciliation": rebound._last_ingest_reconciliation,
    }
    rebound.shutdown()

    assert counts["call_replayed_marker_0"] == 1, evidence
    assert counts["call_replayed_marker_1"] == 1, evidence
    assert counts[fresh_call] == 1, evidence
    assert contents.count("genuinely new terminal request") == 1, evidence
    assert contents.count("genuinely new terminal response") == 1, evidence


def test_single_unpaired_marker_with_changed_source_generation_remains_ambiguous(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    config, replayed_tools, raw_by_path = _fixture(tmp_path)
    session_id = "duplicate-admission-singleton-ambiguity"
    conversation_id = "agent:main:telegram:dm:sanitized:singleton"
    only = replayed_tools[0]

    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine.on_session_start(
        session_id,
        platform="telegram",
        conversation_id=conversation_id,
        context_length=200000,
    )
    engine.ingest([only])
    path = next(iter(raw_by_path))
    _replace_same_bytes_with_new_generation(path, raw_by_path[path])
    assert recover_hermes_persisted_output_with_file_stat(only["content"]) is not None

    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    engine.ingest([only])
    rows = engine._store.get_session_messages(session_id)
    counts = _tool_counts(rows)
    engine.shutdown()

    assert counts["call_replayed_marker_0"] == 2
