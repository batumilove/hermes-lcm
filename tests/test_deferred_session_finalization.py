"""Focused lifecycle contracts for deferred session-end finalization."""

import logging
import threading
import time

from hermes_lcm import lifecycle_metrics
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, name="deferred"):
    return LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"{name}.db")))


class _WriterHolder:
    def __init__(self, engine):
        self.engine = engine
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._run, name="deferred-test-writer", daemon=True)

    def _run(self):
        with self.engine._store._write_lock.attributed("deferred_test_writer"):
            self.entered.set()
            self.release.wait(timeout=10)

    def __enter__(self):
        self.thread.start()
        assert self.entered.wait(timeout=2)
        return self

    def __exit__(self, *_args):
        self.release.set()
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()


def _wait_empty(coordinator, timeout=3.0):
    deadline = time.monotonic() + timeout
    while coordinator.pending_count() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert coordinator.pending_count() == 0


def test_long_overlap_defers_then_persists_and_finalizes(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.on_session_start("session-a", platform="telegram", conversation_id="conv-a")
        messages = [{"role": "user", "content": "terminal message"}]
        with _WriterHolder(engine):
            started = time.monotonic()
            engine.on_session_end("session-a", messages)
            assert time.monotonic() - started < 0.75
            coordinator = engine._storage_owner._finalization_coordinator
            assert coordinator.has_pending("session-a")
        _wait_empty(coordinator)
        assert [row["content"] for row in engine._store.get_session_messages("session-a")] == ["terminal message"]
        state = engine._lifecycle.get_by_conversation("conv-a")
        assert state is not None and state.last_finalized_session_id == "session-a"
    finally:
        engine.shutdown()


def test_duplicate_callbacks_coalesce_to_latest_payload(tmp_path):
    engine = _engine(tmp_path, "coalesce")
    try:
        engine.on_session_start("session-a", platform="telegram", conversation_id="conv-a")
        first = [{"role": "user", "content": "one"}]
        latest = first + [{"role": "assistant", "content": "two"}]
        with _WriterHolder(engine):
            engine.on_session_end("session-a", first)
            engine.on_session_end("session-a", latest)
            coordinator = engine._storage_owner._finalization_coordinator
            assert coordinator.pending_count() == 1
        _wait_empty(coordinator)
        assert [row["content"] for row in engine._store.get_session_messages("session-a")] == ["one", "two"]
    finally:
        engine.shutdown()


def test_lifecycle_retry_does_not_duplicate_successful_raw_ingest(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "phase")
    try:
        engine.on_session_start("session-a", platform="telegram", conversation_id="conv-a")
        original = engine._lifecycle.finalize_session
        attempts = 0

        def transient(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient callback failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(engine._lifecycle, "finalize_session", transient)
        coordinator = engine._storage_owner._finalization_coordinator
        coordinator.enqueue(
            session_id="session-a", conversation_id="conv-a", source="telegram",
            protected_messages=[{"role": "user", "content": "once"}],
            frontier_store_id=0, ingest_pending=True,
        )
        _wait_empty(coordinator)
        assert [row["content"] for row in engine._store.get_session_messages("session-a")] == ["once"]
        state = engine._lifecycle.get_by_conversation("conv-a")
        assert state is not None and state.last_finalized_session_id == "session-a"
    finally:
        engine.shutdown()


def test_synchronous_lifecycle_lock_defers_only_finalization(tmp_path, monkeypatch):
    engine = _engine(tmp_path, "sync-phase")
    try:
        engine.on_session_start("session-a", platform="telegram", conversation_id="conv-a")
        original = engine._lifecycle.finalize_session
        attempts = 0

        def locked_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                import sqlite3

                raise sqlite3.OperationalError("database is locked")
            return original(*args, **kwargs)

        monkeypatch.setattr(engine._lifecycle, "finalize_session", locked_once)
        engine.on_session_end("session-a", [{"role": "user", "content": "once"}])
        coordinator = engine._storage_owner._finalization_coordinator
        _wait_empty(coordinator)
        assert [row["content"] for row in engine._store.get_session_messages("session-a")] == ["once"]
        state = engine._lifecycle.get_by_conversation("conv-a")
        assert state is not None and state.last_finalized_session_id == "session-a"
    finally:
        engine.shutdown()


def test_clone_close_keeps_shared_coordinator_alive(tmp_path):
    root = _engine(tmp_path, "clone")
    clone = root.clone_for_agent()
    coordinator = root._storage_owner._finalization_coordinator
    try:
        assert clone._storage_owner._finalization_coordinator is coordinator
        assert coordinator._store is None
        assert coordinator._lifecycle is None
        clone.shutdown()
        assert coordinator.enqueue(
            session_id="later", conversation_id="", source="telegram",
            protected_messages=[], frontier_store_id=0, ingest_pending=False,
        )
    finally:
        root.shutdown()


def test_exhaustion_is_truthful_and_counted(tmp_path, caplog):
    lifecycle_metrics.configure(enabled=True)
    engine = _engine(tmp_path, "exhaust")
    try:
        engine.on_session_start("session-a", platform="telegram", conversation_id="conv-a")
        coordinator = engine._storage_owner._finalization_coordinator
        coordinator.exhaustion_budget_seconds = 0.15
        before = lifecycle_metrics.snapshot()["runtime_lifecycle"]
        with _WriterHolder(engine):
            with caplog.at_level(logging.WARNING):
                engine.on_session_end("session-a", [{"role": "user", "content": "blocked"}])
                _wait_empty(coordinator, timeout=2.0)
        after = lifecycle_metrics.snapshot()["runtime_lifecycle"]
        assert "exhausted" in caplog.text.lower()
        assert after["session_finalizations_deferred_total"] > before["session_finalizations_deferred_total"]
        assert after["session_finalizations_deferred_exhausted_total"] > before["session_finalizations_deferred_exhausted_total"]
    finally:
        engine.shutdown()
        lifecycle_metrics.configure(enabled=False)
