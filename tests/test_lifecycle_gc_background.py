import threading
import time

from hermes_lcm import engine as engine_mod
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
    original_start = threading.Thread.start
    gc_started = threading.Event()

    def fail_start(self):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    engine.on_session_start("failed-start", platform="cli", context_length=200_000)
    assert engine._session_id == "failed-start"

    monkeypatch.setattr(threading.Thread, "start", original_start)
    monkeypatch.setattr(
        LifecycleStateStore,
        "prune_empty_sessions",
        lambda self, **kwargs: gc_started.set() or 0,
    )
    try:
        engine.on_session_start("retry-session", platform="cli", context_length=200_000)
        assert gc_started.wait(timeout=1.0), "failed thread start consumed the retry window"
    finally:
        engine.shutdown()


def test_gc_request_includes_all_active_registry_sessions(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "active-registry.db"),
        empty_lifecycle_gc_enabled=True,
        empty_lifecycle_gc_threshold=1,
        empty_lifecycle_gc_max_age_hours=0,
    )
    requests = []
    monkeypatch.setattr(
        engine_mod,
        "request_empty_lifecycle_gc",
        lambda *args, **kwargs: requests.append(kwargs) or True,
    )
    engine_a = LCMEngine(config=config)
    engine_b = LCMEngine(config=config)
    try:
        engine_a.on_session_start("long-active", platform="cli", context_length=200_000)
        engine_b.on_session_start("gc-trigger", platform="cli", context_length=200_000)
        provider = requests[-1]["protected_session_ids_provider"]
        assert {"long-active", "gc-trigger"} <= set(provider())
    finally:
        engine_a.shutdown()
        engine_b.shutdown()


def test_inflight_gc_preserves_session_bound_while_candidate_scan_runs(tmp_path, monkeypatch):
    engine_a = _gc_engine(tmp_path)
    engine_b = LCMEngine(config=engine_a._config)
    scan_started = threading.Event()
    release_scan = threading.Event()
    gc_finished = threading.Event()
    original_collect = LifecycleStateStore._collect_empty_session_candidates
    original_prune = LifecycleStateStore.prune_empty_sessions

    def blocked_collect(self, **kwargs):
        scan_started.set()
        assert release_scan.wait(timeout=2.0)
        return original_collect(self, **kwargs)

    def observed_prune(self, **kwargs):
        try:
            return original_prune(self, **kwargs)
        finally:
            gc_finished.set()

    monkeypatch.setattr(
        LifecycleStateStore,
        "_collect_empty_session_candidates",
        blocked_collect,
    )
    monkeypatch.setattr(LifecycleStateStore, "prune_empty_sessions", observed_prune)
    try:
        engine_a.on_session_start("gc-trigger", platform="cli", context_length=200_000)
        assert scan_started.wait(timeout=1.0)
        engine_b.on_session_start("concurrently-bound", platform="cli", context_length=200_000)
        release_scan.set()
        assert gc_finished.wait(timeout=2.0)
        assert engine_b._lifecycle.get_by_session("concurrently-bound") is not None
    finally:
        release_scan.set()
        engine_a.shutdown()
        engine_b.shutdown()


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


def test_prune_clamps_explicit_batch_but_default_completes_cleanup(tmp_path):
    bounded = LifecycleStateStore(tmp_path / "bounded.db")
    complete = LifecycleStateStore(tmp_path / "complete.db")
    try:
        for index in range(150):
            bounded.bind_session(f"bounded-{index}")
            complete.bind_session(f"complete-{index}")

        assert bounded.prune_empty_sessions(max_age_hours=0, max_candidates=1_000) == 100
        assert bounded.row_count() == 50

        assert complete.prune_empty_sessions(max_age_hours=0) == 150
        assert complete.row_count() == 0
    finally:
        bounded.close()
        complete.close()


def test_prune_refreshes_protection_for_each_final_delete(tmp_path):
    lifecycle = LifecycleStateStore(tmp_path / "final-protection.db")
    try:
        lifecycle.bind_session("first")
        lifecycle.bind_session("late-protected")
        calls = 0

        def live_protection():
            nonlocal calls
            calls += 1
            return {"late-protected"} if calls >= 2 else set()

        deleted = lifecycle.prune_empty_sessions(
            max_age_hours=0,
            max_candidates=100,
            protected_session_ids_provider=live_protection,
        )
        assert calls >= 2
        assert deleted == 1
        assert lifecycle.get_by_session("late-protected") is not None
    finally:
        lifecycle.close()


def test_delete_statement_rechecks_protection_at_sql_linearization_point(tmp_path):
    lifecycle = LifecycleStateStore(tmp_path / "sql-protection.db")
    try:
        lifecycle.bind_session("late-protected")
        calls = 0

        def live_protection():
            nonlocal calls
            calls += 1
            return {"late-protected"} if calls >= 3 else set()

        deleted = lifecycle.prune_empty_sessions(
            max_age_hours=0,
            max_candidates=100,
            protected_session_ids_provider=live_protection,
        )
        assert calls >= 3
        assert deleted == 0
        assert lifecycle.get_by_session("late-protected") is not None
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
