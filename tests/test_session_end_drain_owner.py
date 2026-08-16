import threading
import time

import hermes_lcm.engine as engine_module
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.session_end_pending import (
    build_session_end_intent,
    pending_session_end_dir,
    persist_session_end_intent,
)


def _v2_intent(db_path, *, session_id):
    return persist_session_end_intent(
        db_path,
        build_session_end_intent(
            session_id=session_id,
            conversation_id=f"conversation:{session_id}",
            source="telegram",
            frontier_store_id=0,
            messages=[{"role": "user", "content": f"deferred:{session_id}"}],
            ingest_cursor=0,
        ),
    )


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_overlapping_clone_session_ends_handoff_without_writer_timeout(
    tmp_path, monkeypatch, caplog
):
    db_path = tmp_path / "overlapping-session-ends.db"
    root = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    clone = root.clone_for_agent()
    root.on_session_start("first-session", platform="telegram")
    clone.on_session_start("second-session", platform="telegram")
    first_finalize_entered = threading.Event()
    release_first_finalize = threading.Event()
    first_errors = []
    original_finalize = root._lifecycle.finalize_session

    def hold_first_finalize(conversation_id, session_id, frontier_store_id=0):
        if session_id == "first-session":
            first_finalize_entered.set()
            assert release_first_finalize.wait(timeout=2.0)
        return original_finalize(
            conversation_id,
            session_id,
            frontier_store_id=frontier_store_id,
        )

    def end_first_session():
        try:
            root.on_session_end(
                "first-session",
                [{"role": "user", "content": "first final message"}],
            )
        except BaseException as exc:
            first_errors.append(exc)

    monkeypatch.setattr(root._lifecycle, "finalize_session", hold_first_finalize)
    monkeypatch.setattr(
        engine_module,
        "_SESSION_END_PROCESS_WRITE_TIMEOUT_MS",
        20,
    )
    first = threading.Thread(target=end_first_session, name="first-session-end")
    try:
        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            first.start()
            assert first_finalize_entered.wait(timeout=1.0)
            clone.on_session_end(
                "second-session",
                [{"role": "user", "content": "second final message"}],
            )
            handoff_records = [
                record.getMessage()
                for record in caplog.records
                if "event=session_end_deferred" in record.getMessage()
                and "session_id='second-session'" in record.getMessage()
            ]
            assert len(handoff_records) == 1
            assert "stage=singleflight" in handoff_records[0]
            assert "outcome=scheduled" in handoff_records[0]
            assert "operation=session_end_singleflight_handoff" in handoff_records[0]
            assert "error_type=-" in handoff_records[0]
            assert not any(
                "database is locked" in record.getMessage().lower()
                or "timed out waiting for process-wide sqlite writer"
                in record.getMessage().lower()
                for record in caplog.records
            )

            release_first_finalize.set()
            first.join(timeout=2.0)
            assert root._session_end_drain_done.wait(timeout=2.0)

        assert not first.is_alive()
        assert first_errors == []
        assert tuple(pending_session_end_dir(db_path).glob("*.json")) == ()
        assert [
            message.get("content")
            for message in root._store.get_session_messages("first-session")
        ] == ["first final message"]
        assert [
            message.get("content")
            for message in root._store.get_session_messages("second-session")
        ] == ["second final message"]
    finally:
        release_first_finalize.set()
        first.join(timeout=2.0)
        root._session_end_drain_done.wait(timeout=2.0)
        clone.shutdown()
        root.shutdown()


def test_session_end_intent_persistence_failure_releases_singleflight_lock(
    tmp_path, monkeypatch
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "persist-failure.db"))
    )
    engine.on_session_start("persist-failure", platform="telegram")

    def fail_persist(*args, **kwargs):
        del args, kwargs
        raise OSError("intent persistence failed")

    monkeypatch.setattr(engine, "_persist_session_end_intent", fail_persist)
    try:
        with pytest.raises(OSError, match="intent persistence failed"):
            engine.on_session_end(
                "persist-failure",
                [{"role": "user", "content": "must not strand the lease"}],
            )

        lock = engine._storage_owner._session_end_flush_lock
        assert lock.acquire(blocking=False)
        lock.release()
    finally:
        engine.shutdown()


def test_gate_waiting_drain_survives_scheduling_clone_shutdown(tmp_path):
    db_path = tmp_path / "gate-waiting-clone-shutdown.db"
    root = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    clone = root.clone_for_agent()
    owner = root._storage_owner
    intent_path = None
    test_gate_held = False
    owner._session_end_flush_lock.acquire()
    test_gate_held = True
    try:
        intent_path = _v2_intent(db_path, session_id="waiting-clone-session")
        clone._schedule_session_end_drain()
        assert not root._session_end_drain_done.wait(timeout=0.1)

        clone.shutdown()
        assert not owner._closed
        assert intent_path.exists()

        owner._session_end_flush_lock.release()
        test_gate_held = False
        assert root._session_end_drain_done.wait(timeout=2.0)
        assert not intent_path.exists()
        assert [
            message.get("content")
            for message in root._store.get_session_messages("waiting-clone-session")
        ] == ["deferred:waiting-clone-session"]
    finally:
        if test_gate_held:
            owner._session_end_flush_lock.release()
        root._session_end_drain_done.wait(timeout=2.0)
        if not clone._storage_released:
            clone.shutdown()
        root.shutdown()
        if intent_path is not None:
            intent_path.unlink(missing_ok=True)


def test_clones_share_one_serialized_session_end_drain_worker(tmp_path, monkeypatch):
    db_path = tmp_path / "shared.db"
    intent_path = _v2_intent(db_path, session_id="session-a")
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    max_active = 0
    calls = 0

    def blocked_drain(self, path):
        nonlocal active, max_active, calls
        with state_lock:
            active += 1
            calls += 1
            max_active = max(max_active, active)
        try:
            release.wait(timeout=2.0)
            path.unlink(missing_ok=True)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(LCMEngine, "_drain_one_session_end_intent", blocked_drain)
    root = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    clones = []
    try:
        assert _wait_until(lambda: calls >= 1)
        clones = [root.clone_for_agent() for _ in range(5)]
        worker_ids = {
            id(engine._session_end_drain_thread)
            for engine in [root, *clones]
            if engine._session_end_drain_thread is not None
        }
        _wait_until(lambda: calls >= 2, timeout=0.3)
        observed_max_active = max_active
    finally:
        release.set()
        for engine in [*clones, root]:
            engine._session_end_drain_done.wait(timeout=2.0)
        for engine in clones:
            engine.shutdown()
        root.shutdown()
        intent_path.unlink(missing_ok=True)

    assert len(worker_ids) == 1
    assert observed_max_active == 1


def test_independent_storage_owners_can_drain_concurrently(tmp_path, monkeypatch):
    paths = [
        _v2_intent(tmp_path / "first.db", session_id="first"),
        _v2_intent(tmp_path / "second.db", session_id="second"),
    ]
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def blocked_drain(self, path):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            release.wait(timeout=2.0)
            path.unlink(missing_ok=True)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(LCMEngine, "_drain_one_session_end_intent", blocked_drain)
    engines = [
        LCMEngine(config=LCMConfig(database_path=str(tmp_path / "first.db"))),
        LCMEngine(config=LCMConfig(database_path=str(tmp_path / "second.db"))),
    ]
    try:
        assert _wait_until(lambda: max_active == 2)
        assert engines[0]._storage_owner is not engines[1]._storage_owner
        assert engines[0]._session_end_drain_thread is not engines[1]._session_end_drain_thread
    finally:
        release.set()
        for engine in engines:
            engine._session_end_drain_done.wait(timeout=2.0)
            engine.shutdown()
        for path in paths:
            path.unlink(missing_ok=True)


def test_worker_owner_reference_survives_initiating_clone_shutdown(tmp_path, monkeypatch):
    db_path = tmp_path / "clone-close.db"
    root = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    clone = root.clone_for_agent()
    owner = root._storage_owner
    intent_path = _v2_intent(db_path, session_id="clone-session")
    entered = threading.Event()
    release = threading.Event()

    def blocked_drain(self, path):
        entered.set()
        release.wait(timeout=2.0)
        path.unlink(missing_ok=True)

    monkeypatch.setattr(LCMEngine, "_drain_one_session_end_intent", blocked_drain)
    try:
        clone._schedule_session_end_drain()
        assert entered.wait(timeout=1.0)
        clone.shutdown()
        assert not owner._closed
        root._store.get_session_messages("clone-session")
        release.set()
        assert root._session_end_drain_done.wait(timeout=2.0)
        assert not intent_path.exists()
    finally:
        release.set()
        root._session_end_drain_done.wait(timeout=2.0)
        if not clone._storage_released:
            clone.shutdown()
        root.shutdown()
        intent_path.unlink(missing_ok=True)


def test_worker_signals_done_when_owner_release_raises(tmp_path, monkeypatch):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "release-error.db"))
    )
    owner = engine._storage_owner.acquire()
    original_release = owner.release

    def fail_release():
        raise RuntimeError("helper close failed")

    monkeypatch.setattr(owner, "release", fail_release)
    owner._session_end_drain_done.clear()
    try:
        with pytest.raises(RuntimeError, match="helper close failed"):
            engine._drain_pending_session_ends(owner)
        assert owner._session_end_drain_done.is_set()
    finally:
        monkeypatch.setattr(owner, "release", original_release)
        original_release()
        engine.shutdown()


def test_retirement_start_failure_never_signals_done_or_replaces_live_worker(
    tmp_path, monkeypatch
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "retirement-start-error.db"))
    )
    owner = engine._storage_owner.acquire()
    original_start = threading.Thread.start
    original_release = owner.release
    release_entered = threading.Event()
    allow_release = threading.Event()

    def selective_start(thread):
        if thread.name == "lcm-session-end-drain-retire":
            raise RuntimeError("retirement helper unavailable")
        return original_start(thread)

    def blocked_release():
        release_entered.set()
        assert allow_release.wait(timeout=2.0)
        return original_release()

    monkeypatch.setattr(threading.Thread, "start", selective_start)
    monkeypatch.setattr(owner, "release", blocked_release)
    owner._session_end_drain_done.clear()
    worker = threading.Thread(
        target=engine._drain_pending_session_ends,
        args=(owner,),
        name="lcm-session-end-drain",
        daemon=True,
    )
    owner._session_end_drain_thread = worker
    worker.start()
    try:
        assert release_entered.wait(timeout=1.0)
        assert worker.is_alive()
        assert owner._session_end_drain_thread is worker
        assert not owner._session_end_drain_done.is_set()

        allow_release.set()
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        assert owner._session_end_drain_thread is worker
        assert not owner._session_end_drain_done.is_set()

        monkeypatch.setattr(threading.Thread, "start", original_start)
        monkeypatch.setattr(owner, "release", original_release)
        engine._schedule_session_end_drain()
        assert owner._session_end_drain_done.wait(timeout=1.0)
    finally:
        allow_release.set()
        monkeypatch.setattr(threading.Thread, "start", original_start)
        monkeypatch.setattr(owner, "release", original_release)
        worker.join(timeout=1.0)
        engine.shutdown()


def test_successful_zero_budget_drain_does_not_record_exhaustion(tmp_path, monkeypatch):
    db_path = tmp_path / "successful-zero-budget.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    intent_path = _v2_intent(db_path, session_id="completed-session")
    exhausted_calls = []

    monkeypatch.setattr(
        engine_module,
        "_SESSION_END_DEFERRED_RETRY_BUDGET_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        "hermes_lcm.lifecycle_metrics.record_session_end_drain_exhausted",
        lambda: exhausted_calls.append(True),
    )
    try:
        engine._schedule_session_end_drain()
        assert engine._session_end_drain_done.wait(timeout=1.0)
        assert not intent_path.exists()
        assert exhausted_calls == []
    finally:
        engine.shutdown()
        intent_path.unlink(missing_ok=True)


def test_started_metric_failure_cannot_undo_a_running_drain_worker(tmp_path, monkeypatch):
    db_path = tmp_path / "started-metric-error.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    intent_path = _v2_intent(db_path, session_id="metric-session")
    entered = threading.Event()
    release = threading.Event()

    def blocked_drain(self, path):
        entered.set()
        assert release.wait(timeout=2.0)
        path.unlink(missing_ok=True)

    def fail_metric():
        raise RuntimeError("metrics backend unavailable")

    monkeypatch.setattr(LCMEngine, "_drain_one_session_end_intent", blocked_drain)
    monkeypatch.setattr(
        "hermes_lcm.lifecycle_metrics.record_session_end_drain_started",
        fail_metric,
    )
    try:
        engine._schedule_session_end_drain()
        assert entered.wait(timeout=1.0)
        worker = engine._session_end_drain_thread
        assert worker is not None
        assert worker.is_alive()
        assert not engine._session_end_drain_done.is_set()

        release.set()
        assert engine._session_end_drain_done.wait(timeout=1.0)
        assert not intent_path.exists()
    finally:
        release.set()
        engine._session_end_drain_done.wait(timeout=1.0)
        engine.shutdown()
        intent_path.unlink(missing_ok=True)


def test_exhaustion_rescan_waits_for_retiring_generation_to_exit(tmp_path, monkeypatch):
    db_path = tmp_path / "exhaustion-handoff.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    stuck_path = _v2_intent(db_path, session_id="stuck-session")
    retiring = threading.Event()
    release_retiring = threading.Event()
    replacement_processed = threading.Event()
    overlap = threading.Event()
    active_lock = threading.Lock()
    active_generations = 0
    original_drain = LCMEngine._drain_pending_session_ends

    def tracked_drain(self, owner):
        nonlocal active_generations
        with active_lock:
            active_generations += 1
            if active_generations > 1:
                overlap.set()
        try:
            return original_drain(self, owner)
        finally:
            with active_lock:
                active_generations -= 1

    def drain_with_one_stuck_intent(self, path):
        if path == stuck_path:
            raise RuntimeError("temporary lifecycle outage")
        replacement_processed.set()
        path.unlink(missing_ok=True)

    def block_retiring_generation():
        retiring.set()
        release_retiring.wait(timeout=2.0)

    monkeypatch.setattr(LCMEngine, "_drain_pending_session_ends", tracked_drain)
    monkeypatch.setattr(
        LCMEngine,
        "_drain_one_session_end_intent",
        drain_with_one_stuck_intent,
    )
    monkeypatch.setattr(
        engine_module,
        "_SESSION_END_DEFERRED_RETRY_BUDGET_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        "hermes_lcm.lifecycle_metrics.record_session_end_drain_exhausted",
        block_retiring_generation,
    )

    new_path = None
    try:
        engine._schedule_session_end_drain()
        assert retiring.wait(timeout=1.0)
        new_path = _v2_intent(db_path, session_id="new-session")
        engine._schedule_session_end_drain()

        assert not overlap.wait(timeout=0.2)
        release_retiring.set()
        assert engine._session_end_drain_done.wait(timeout=1.0)
        assert replacement_processed.is_set()
        assert not new_path.exists()
        assert stuck_path.exists()
    finally:
        release_retiring.set()
        engine._session_end_drain_done.wait(timeout=1.0)
        engine.shutdown()
        stuck_path.unlink(missing_ok=True)
        if new_path is not None:
            new_path.unlink(missing_ok=True)


def test_new_intent_rescan_is_not_lost_at_exhaustion(tmp_path, monkeypatch):
    db_path = tmp_path / "exhaustion-race.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    stuck_path = _v2_intent(db_path, session_id="stuck-session")
    retained_snapshot_taken = threading.Event()
    release_retained_snapshot = threading.Event()
    processed_new = threading.Event()
    original_iter = engine_module.iter_session_end_intents
    drain_scan_count = 0

    def controlled_iter(path):
        nonlocal drain_scan_count
        result = tuple(original_iter(path))
        if threading.current_thread().name == "lcm-session-end-drain":
            drain_scan_count += 1
            if drain_scan_count == 2:
                retained_snapshot_taken.set()
                release_retained_snapshot.wait(timeout=2.0)
        return result

    def drain_with_one_stuck_intent(self, path):
        if path == stuck_path:
            raise RuntimeError("temporary lifecycle outage")
        processed_new.set()
        path.unlink(missing_ok=True)

    monkeypatch.setattr(engine_module, "iter_session_end_intents", controlled_iter)
    monkeypatch.setattr(
        LCMEngine,
        "_drain_one_session_end_intent",
        drain_with_one_stuck_intent,
    )
    monkeypatch.setattr(
        engine_module,
        "_SESSION_END_DEFERRED_RETRY_BUDGET_SECONDS",
        0.0,
    )

    new_path = None
    try:
        engine._schedule_session_end_drain()
        assert retained_snapshot_taken.wait(timeout=1.0)
        new_path = _v2_intent(db_path, session_id="new-session")
        engine._schedule_session_end_drain()
        release_retained_snapshot.set()

        assert engine._session_end_drain_done.wait(timeout=1.0)
        assert processed_new.is_set()
        assert not new_path.exists()
        assert stuck_path.exists()
    finally:
        release_retained_snapshot.set()
        engine._session_end_drain_done.wait(timeout=1.0)
        engine.shutdown()
        stuck_path.unlink(missing_ok=True)
        if new_path is not None:
            new_path.unlink(missing_ok=True)
