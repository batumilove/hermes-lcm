import threading
import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore


def _gc_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        empty_lifecycle_gc_enabled=True,
        empty_lifecycle_gc_threshold=1,
        empty_lifecycle_gc_max_age_hours=0,
    )
    engine = LCMEngine(config=config)
    for index in range(3):
        engine._lifecycle.bind_session(f"orphan-{index}")
    return engine


def test_session_bind_does_not_wait_for_empty_lifecycle_gc(tmp_path, monkeypatch):
    engine = _gc_engine(tmp_path)
    gc_started = threading.Event()
    release_gc = threading.Event()
    bind_returned = threading.Event()
    original = LifecycleStateStore.prune_empty_sessions

    def blocked_prune(self, **kwargs):
        gc_started.set()
        assert release_gc.wait(timeout=2.0)
        return original(self, **kwargs)

    monkeypatch.setattr(LifecycleStateStore, "prune_empty_sessions", blocked_prune)

    def bind():
        engine.on_session_start("live-session", platform="cli", context_length=200_000)
        bind_returned.set()

    caller = threading.Thread(target=bind, name="bind-caller")
    caller.start()
    try:
        assert gc_started.wait(timeout=1.0)
        assert bind_returned.wait(timeout=0.2), "session bind waited for lifecycle GC"
    finally:
        release_gc.set()
        caller.join(timeout=2.0)
        engine.shutdown()
    assert not caller.is_alive()


def test_empty_lifecycle_gc_is_single_flight_per_database(tmp_path, monkeypatch):
    engine = _gc_engine(tmp_path)
    gc_started = threading.Event()
    release_gc = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def blocked_prune(self, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        gc_started.set()
        assert release_gc.wait(timeout=2.0)
        return 0

    monkeypatch.setattr(LifecycleStateStore, "prune_empty_sessions", blocked_prune)
    try:
        engine.on_session_start("live-one", platform="cli", context_length=200_000)
        assert gc_started.wait(timeout=1.0)
        engine.on_session_start("live-two", platform="cli", context_length=200_000)
        engine.on_session_start("live-three", platform="cli", context_length=200_000)
        time.sleep(0.05)
        assert calls == 1
    finally:
        release_gc.set()
        engine.shutdown()


def test_empty_lifecycle_gc_is_rate_limited_after_completion(tmp_path, monkeypatch):
    engine = _gc_engine(tmp_path)
    completed = threading.Event()
    calls = 0

    def counted_prune(self, **kwargs):
        nonlocal calls
        calls += 1
        completed.set()
        return 0

    monkeypatch.setattr(LifecycleStateStore, "prune_empty_sessions", counted_prune)
    try:
        engine.on_session_start("live-one", platform="cli", context_length=200_000)
        assert completed.wait(timeout=1.0)
        engine.on_session_start("live-two", platform="cli", context_length=200_000)
        time.sleep(0.1)
        assert calls == 1
    finally:
        engine.shutdown()


def test_session_bind_survives_gc_thread_start_failure(tmp_path, monkeypatch):
    engine = _gc_engine(tmp_path)

    def fail_start(self):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        engine.on_session_start("live-session", platform="cli", context_length=200_000)
        assert engine._session_id == "live-session"
    finally:
        engine.shutdown()


def test_prune_empty_sessions_bounds_final_write_transaction(tmp_path):
    db_path = tmp_path / "large-lcm.db"
    lifecycle = LifecycleStateStore(db_path)
    try:
        for index in range(5_200):
            lifecycle.bind_session(f"orphan-{index}")
        deleted = lifecycle.prune_empty_sessions(max_age_hours=0, max_candidates=100)
        assert deleted == 100
        assert lifecycle.row_count() == 5_100
    finally:
        lifecycle.close()


def test_prune_preserves_conversation_rebound_after_candidate_snapshot(tmp_path):
    db_path = tmp_path / "rebound-lcm.db"
    lifecycle = LifecycleStateStore(db_path)
    concurrent = LifecycleStateStore(db_path)
    rebound = threading.Event()
    try:
        lifecycle.bind_session("stale-session", conversation_id="conversation")

        def rebind_before_write_transaction(statement):
            normalized = " ".join(str(statement).upper().split())
            if normalized == "BEGIN IMMEDIATE" and not rebound.is_set():
                concurrent.bind_session("new-session", conversation_id="conversation")
                rebound.set()

        lifecycle._conn.set_trace_callback(rebind_before_write_transaction)
        deleted = lifecycle.prune_empty_sessions(max_age_hours=None)
        assert rebound.is_set()
        assert deleted == 0
        state = lifecycle.get_by_conversation("conversation")
        assert state is not None
        assert state.current_session_id == "new-session"
    finally:
        lifecycle.close()
        concurrent.close()
