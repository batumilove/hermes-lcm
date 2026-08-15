import json
import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def test_compression_boundary_carries_summaries_without_moving_raw_messages(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start(
            "parent-session",
            platform="discord",
            conversation_id="discord-thread",
            context_length=200_000,
        )
        store_ids = engine._store.append_batch(
            "parent-session",
            [
                {"role": "user", "content": "raw parent payload"},
                {"role": "assistant", "content": "raw assistant payload"},
            ],
            source="discord",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="parent-session",
                depth=0,
                summary="summary carried across compression boundary",
                token_count=8,
                source_token_count=12,
                source_ids=store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=time.time(),
                latest_at=time.time(),
                expand_hint="Expand for raw parent payload",
            )
        )
        externalized_dir = tmp_path / "externalized"
        externalized_dir.mkdir()
        payload_path = externalized_dir / "payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "kind": "ingest_payload",
                    "role": "tool",
                    "session_id": "parent-session",
                    "content": "large raw payload",
                    "created_at": time.time(),
                }
            ),
            encoding="utf-8",
        )

        engine.on_session_start(
            "child-session",
            platform="discord",
            conversation_id="discord-thread",
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="parent-session",
        )

        assert engine._store.get_session_count("parent-session") == 2
        assert engine._store.get_session_count("child-session") == 0
        assert engine._dag.get_session_nodes("parent-session") == []
        child_nodes = engine._dag.get_session_nodes("child-session")
        assert len(child_nodes) == 1
        assert child_nodes[0].source_ids == store_ids
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["session_id"] == "parent-session"
    finally:
        engine.shutdown()


def test_replayed_parent_boundary_reuses_existing_child_conversation(tmp_path):
    """A late duplicate parent boundary must not mint a child-keyed lifecycle row."""
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start(
            "parent-session",
            platform="telegram",
            conversation_id="telegram-thread",
            context_length=200_000,
        )
        store_id = engine._store.append(
            "parent-session",
            {"role": "user", "content": "durable parent payload"},
            source="telegram",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="parent-session",
                depth=0,
                summary="parent summary",
                token_count=4,
                source_token_count=4,
                source_ids=[store_id],
                source_type="messages",
                created_at=time.time(),
            )
        )
        engine.on_session_start(
            "first-child",
            platform="telegram",
            conversation_id="telegram-thread",
            boundary_reason="compression",
            old_session_id="parent-session",
        )
        assert engine._dag.get_session_nodes("parent-session") == []
        assert engine._dag.get_session_nodes("first-child")

        # A side-channel can temporarily own the shared engine before a delayed
        # duplicate compression callback arrives for the already-finalized parent.
        engine.on_session_start(
            "auxiliary-session",
            platform="subagent",
            conversation_id="auxiliary-conversation",
        )
        engine.on_session_start(
            "abandoned-second-child",
            platform="telegram",
            boundary_reason="compression",
            old_session_id="parent-session",
        )

        stable = engine._lifecycle.get_by_conversation("telegram-thread")
        assert stable is not None
        assert stable.current_session_id == "first-child"
        assert engine._lifecycle.get_by_conversation("abandoned-second-child") is None
    finally:
        engine.shutdown()
