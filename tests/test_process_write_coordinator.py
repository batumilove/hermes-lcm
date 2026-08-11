"""Regression tests for process-wide SQLite writer coordination."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from hermes_lcm import db_bootstrap, maintenance, sqlite_util
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import VectorStore


def test_process_write_lock_api_exists():
    assert callable(getattr(sqlite_util, "process_sqlite_write_lock", None))


def test_same_database_path_reuses_process_write_lock(tmp_path):
    db_path = tmp_path / "lcm.db"

    assert sqlite_util.process_sqlite_write_lock(db_path) is sqlite_util.process_sqlite_write_lock(db_path)


def test_different_database_paths_use_different_process_write_locks(tmp_path):
    assert sqlite_util.process_sqlite_write_lock(tmp_path / "one.db") is not sqlite_util.process_sqlite_write_lock(
        tmp_path / "two.db"
    )


def test_write_capable_helpers_share_one_database_coordinator(tmp_path):
    db_path = tmp_path / "lcm.db"
    store = MessageStore(db_path)
    dag = SummaryDAG(db_path)
    lifecycle = LifecycleStateStore(db_path)
    rollups = RollupStore(db_path)
    vectors = VectorStore(db_path)

    try:
        coordinator = sqlite_util.process_sqlite_write_lock(db_path)
        assert store._write_lock is coordinator
        assert dag._db_lock is coordinator
        assert lifecycle._lock is coordinator
        assert rollups._write_lock is coordinator
        assert vectors._write_lock is coordinator
    finally:
        vectors.close()
        rollups.close()
        lifecycle.close()
        dag.close()
        store.close()


@pytest.mark.parametrize(
    ("helper_class", "lock_attribute"),
    [
        (MessageStore, "_write_lock"),
        (SummaryDAG, "_db_lock"),
        (LifecycleStateStore, "_lock"),
        (RollupStore, "_write_lock"),
        (VectorStore, "_write_lock"),
    ],
)
def test_helper_initialization_holds_database_coordinator(
    tmp_path, monkeypatch, helper_class, lock_attribute
):
    observed = []

    def probe_initialization(self):
        observed.append(getattr(self, lock_attribute)._is_owned())

    monkeypatch.setattr(helper_class, "_init_db", probe_initialization)

    helper_class(tmp_path / "lcm.db")

    assert observed == [True]


def test_temporary_busy_timeout_holds_database_coordinator(tmp_path):
    db_path = tmp_path / "lcm.db"
    connection = sqlite3.connect(db_path)
    coordinator = sqlite_util.process_sqlite_write_lock(db_path)

    try:
        with sqlite_util._temporary_sqlite_busy_timeout(
            [connection], 50, write_lock=coordinator
        ):
            assert coordinator._is_owned()
            assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 50
    finally:
        connection.close()


def test_temporary_busy_timeout_uses_requested_bound_for_coordinator(
    tmp_path, monkeypatch
):
    coordinator = sqlite_util.process_sqlite_write_lock(tmp_path / "lcm.db")
    attempted_timeouts = []

    def reject_acquire(*, timeout):
        attempted_timeouts.append(timeout)
        return False

    monkeypatch.setattr(coordinator, "acquire", reject_acquire)

    with pytest.raises(sqlite3.OperationalError, match="process-wide SQLite writer"):
        with sqlite_util._temporary_sqlite_busy_timeout(
            [], 50, write_lock=coordinator
        ):
            pass

    assert attempted_timeouts == [0.05]


def test_process_write_lock_wait_is_bounded(tmp_path, monkeypatch):
    coordinator = sqlite_util.process_sqlite_write_lock(tmp_path / "lcm.db")
    holder_entered = threading.Event()

    def hold_coordinator():
        with coordinator:
            holder_entered.set()
            time.sleep(0.2)

    holder = threading.Thread(target=hold_coordinator)
    holder.start()
    assert holder_entered.wait(timeout=1.0)
    monkeypatch.setattr(
        sqlite_util, "_PROCESS_SQLITE_WRITE_LOCK_TIMEOUT_SECONDS", 0.05, raising=False
    )

    try:
        with pytest.raises(sqlite3.OperationalError, match="process-wide SQLite writer"):
            with coordinator:
                pass
    finally:
        holder.join(timeout=1.0)

    assert not holder.is_alive()


def test_concurrent_helper_writes_are_serialized_and_integral(tmp_path):
    db_path = tmp_path / "lcm.db"
    thread_count = 12
    writes_per_thread = 25
    start = threading.Barrier(thread_count)
    errors = []
    errors_lock = threading.Lock()

    def write_messages(index):
        store = dag = lifecycle = None
        try:
            store = MessageStore(db_path)
            dag = SummaryDAG(db_path)
            lifecycle = LifecycleStateStore(db_path)
            session_id = f"stress-{index}"
            conversation_id = f"stress-conversation-{index}"
            lifecycle.bind_session(session_id, conversation_id=conversation_id)
            start.wait(timeout=30.0)
            for turn in range(writes_per_thread):
                store.append(
                    session_id,
                    {"role": "user", "content": f"message-{index}-{turn}"},
                    source="stress",
                    conversation_id=conversation_id,
                )
                lifecycle.advance_frontier(conversation_id, session_id, turn + 1)
                if turn % 5 == 0:
                    dag.add_node(
                        SummaryNode(
                            session_id=session_id,
                            depth=0,
                            summary=f"summary-{index}-{turn}",
                            source_ids=[turn + 1],
                            source_type="messages",
                        )
                    )
        except BaseException as exc:
            with errors_lock:
                errors.append(f"thread={index} {type(exc).__name__}: {exc}")
        finally:
            for helper in (lifecycle, dag, store):
                if helper is not None:
                    helper.close()

    threads = [threading.Thread(target=write_messages, args=(index,)) for index in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120.0)

    assert not [thread for thread in threads if thread.is_alive()]
    assert errors == []

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 300
        assert connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 60
        assert connection.execute("SELECT COUNT(*) FROM lcm_lifecycle_state").fetchone()[0] == 12
    finally:
        connection.close()


def test_lifecycle_clear_debt_holds_database_coordinator(tmp_path):
    lifecycle = LifecycleStateStore(tmp_path / "lcm.db")
    bound = lifecycle.bind_session("session", conversation_id="conversation")
    lifecycle.record_debt(bound.conversation_id, kind="raw_backlog", size_estimate=1)
    coordinator = sqlite_util.process_sqlite_write_lock(tmp_path / "lcm.db")
    real_connection = lifecycle._conn
    update_lock_states = []

    class ProbeConnection:
        def execute(self, sql, *args, **kwargs):
            if "UPDATE lcm_lifecycle_state" in sql:
                update_lock_states.append(coordinator._is_owned())
            return real_connection.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_connection, name)

    lifecycle._conn = ProbeConnection()
    try:
        lifecycle.clear_debt(bound.conversation_id)
    finally:
        lifecycle._conn = real_connection
        lifecycle.close()

    assert update_lock_states == [True]


def test_lifecycle_delete_safe_rows_holds_database_coordinator(tmp_path):
    lifecycle = LifecycleStateStore(tmp_path / "lcm.db")
    lifecycle.bind_session("session", conversation_id="conversation")
    coordinator = sqlite_util.process_sqlite_write_lock(tmp_path / "lcm.db")
    real_connection = lifecycle._conn
    delete_lock_states = []

    class ProbeConnection:
        def execute(self, sql, *args, **kwargs):
            if "DELETE FROM lcm_lifecycle_state" in sql:
                delete_lock_states.append(coordinator._is_owned())
            return real_connection.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(real_connection, name)

    lifecycle._conn = ProbeConnection()
    try:
        lifecycle.delete_safe_rows_for_sessions({"session"})
    finally:
        lifecycle._conn = real_connection
        lifecycle.close()

    assert delete_lock_states == [True]


def _fts_spec():
    return db_bootstrap.ExternalContentFtsSpec(
        table_name="messages_fts",
        content_table="messages",
        content_rowid="store_id",
        indexed_column="content",
        trigger_sqls=(),
    )


def test_background_fts_scan_writes_hold_database_coordinator(tmp_path, monkeypatch):
    db_path = tmp_path / "lcm.db"
    store = MessageStore(db_path)
    store.close()
    coordinator = sqlite_util.process_sqlite_write_lock(db_path)
    observations = []

    def observe(name, result=None):
        def probe(*args, **kwargs):
            observations.append((name, coordinator._is_owned()))
            return result

        return probe

    monkeypatch.setattr(db_bootstrap, "_record_scan_started", observe("scan-started"))
    monkeypatch.setattr(
        db_bootstrap,
        "check_external_content_fts_integrity",
        observe("integrity-check", {"status": "pass", "detail": ""}),
    )
    monkeypatch.setattr(db_bootstrap, "_record_integrity_checked", observe("checked"))
    monkeypatch.setattr(db_bootstrap, "_clear_integrity_failed", observe("clear-failed"))
    monkeypatch.setattr(db_bootstrap, "_clear_scan_started", observe("clear-started"))

    db_bootstrap._run_background_integrity_scan(str(db_path), _fts_spec(), time.time())

    assert observations
    assert all(owned for _, owned in observations), observations


def test_background_fts_dispatch_claim_holds_database_coordinator(tmp_path, monkeypatch):
    db_path = tmp_path / "lcm.db"
    store = MessageStore(db_path)
    coordinator = sqlite_util.process_sqlite_write_lock(db_path)
    observations = []
    real_record = db_bootstrap._record_scan_started

    def probe_record(*args, **kwargs):
        observations.append(coordinator._is_owned())
        return real_record(*args, **kwargs)

    monkeypatch.setattr(db_bootstrap, "_record_scan_started", probe_record)
    monkeypatch.setattr(db_bootstrap, "_run_background_integrity_scan", lambda *args: None)
    try:
        assert db_bootstrap._dispatch_background_integrity_scan(store._conn, _fts_spec()) is True
        db_bootstrap.join_background_integrity_scans(timeout=5)
    finally:
        store.close()

    assert observations == [True]


@pytest.mark.parametrize(
    "helper_class",
    [MessageStore, SummaryDAG, LifecycleStateStore, RollupStore, VectorStore],
)
def test_helper_close_holds_database_coordinator(tmp_path, helper_class):
    helper = helper_class(tmp_path / "lcm.db")
    coordinator = sqlite_util.process_sqlite_write_lock(tmp_path / "lcm.db")
    real_connection = helper._conn
    observations = []

    class ProbeConnection:
        def execute(self, *args, **kwargs):
            observations.append(("execute", coordinator._is_owned()))
            return real_connection.execute(*args, **kwargs)

        def close(self):
            observations.append(("close", coordinator._is_owned()))
            return real_connection.close()

        def __getattr__(self, name):
            return getattr(real_connection, name)

    helper._conn = ProbeConnection()
    helper.close()

    assert observations
    assert all(owned for _, owned in observations)


@pytest.mark.parametrize(
    "operation", [maintenance.backup_database, maintenance.rotate_backup_database]
)
def test_backup_flush_and_snapshot_hold_database_coordinator(tmp_path, operation):
    db_path = tmp_path / "lcm.db"
    db_path.touch()
    coordinator = sqlite_util.process_sqlite_write_lock(db_path)
    observations = []

    class ProbeConnection:
        def commit(self):
            observations.append(("commit", coordinator._is_owned()))

    class ProbeStore:
        def __init__(self):
            self.db_path = db_path
            self._write_lock = coordinator

        def commit(self):
            observations.append(("store_commit", coordinator._is_owned()))

        def backup(self, destination):
            observations.append(("backup", coordinator._is_owned()))
            destination.execute("CREATE TABLE snapshot_probe(value INTEGER)")
            destination.commit()

    class ProbeEngine:
        def __init__(self):
            self._store = ProbeStore()
            self._dag = type("ProbeDAG", (), {"_conn": ProbeConnection()})()
            self._lifecycle = type("ProbeLifecycle", (), {"_conn": ProbeConnection()})()

        def backup_dir(self):
            return tmp_path / "backups"

        def rotate_backup_path(self):
            return tmp_path / "backups" / "rotate-latest.sqlite3"

    result = operation(ProbeEngine())

    assert result["ok"] is True
    assert observations
    assert all(owned for _, owned in observations)


def _run_close_during_active_transaction(tmp_path):
    db_path = tmp_path / "lcm.db"
    writer = MessageStore(db_path)
    closer = SummaryDAG(db_path)
    coordinator = sqlite_util.process_sqlite_write_lock(db_path)
    writer._conn.execute("CREATE TABLE IF NOT EXISTS close_probe(value TEXT NOT NULL)")
    writer._conn.commit()
    transaction_started = threading.Event()
    release_transaction = threading.Event()
    close_finished = threading.Event()
    errors = []

    def hold_transaction():
        try:
            with coordinator:
                writer._conn.execute("BEGIN IMMEDIATE")
                writer._conn.execute("INSERT INTO close_probe(value) VALUES ('committed')")
                transaction_started.set()
                assert release_transaction.wait(timeout=5)
                writer._conn.commit()
        except BaseException as exc:
            errors.append(exc)

    def close_other_helper():
        try:
            closer.close()
            close_finished.set()
        except BaseException as exc:
            errors.append(exc)

    holder = threading.Thread(target=hold_transaction)
    closer_thread = threading.Thread(target=close_other_helper)
    holder.start()
    assert transaction_started.wait(timeout=5)
    closer_thread.start()
    assert not close_finished.wait(timeout=0.1)
    release_transaction.set()
    holder.join(timeout=5)
    closer_thread.join(timeout=5)

    assert not holder.is_alive()
    assert not closer_thread.is_alive()
    assert close_finished.is_set()
    assert errors == []
    writer.close()

    check = sqlite3.connect(db_path)
    try:
        row_count = check.execute("SELECT COUNT(*) FROM close_probe").fetchone()[0]
        integrity = check.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        check.close()
    return row_count, integrity


def test_concurrent_close_during_live_write_is_serialized(tmp_path):
    row_count, integrity = _run_close_during_active_transaction(tmp_path)
    assert row_count == 1
    assert integrity == "ok"


def test_close_during_active_transaction_preserves_integrity(tmp_path):
    row_count, integrity = _run_close_during_active_transaction(tmp_path)
    assert (row_count, integrity) == (1, "ok")


def test_backup_holds_coordinator_lock(tmp_path):
    db_path = tmp_path / "lcm.db"
    backup_path = tmp_path / "backup.db"
    store = MessageStore(db_path)
    coordinator = sqlite_util.process_sqlite_write_lock(db_path)
    real_connection = store._conn
    observations = []

    class ProbeConnection:
        def backup(self, destination):
            observations.append(coordinator._is_owned())
            return real_connection.backup(destination)

        def __getattr__(self, name):
            return getattr(real_connection, name)

    store._conn = ProbeConnection()
    destination = sqlite3.connect(backup_path)
    try:
        store.backup(destination)
    finally:
        destination.close()
        store._conn = real_connection
        store.close()

    assert observations == [True]


def test_concurrent_backup_during_write_is_serialized(tmp_path):
    db_path = tmp_path / "lcm.db"
    backup_path = tmp_path / "backup.db"
    writer = MessageStore(db_path)
    snapshot_source = MessageStore(db_path)
    coordinator = sqlite_util.process_sqlite_write_lock(db_path)
    writer._conn.execute("CREATE TABLE IF NOT EXISTS backup_probe(value TEXT NOT NULL)")
    writer._conn.commit()
    transaction_started = threading.Event()
    release_transaction = threading.Event()
    backup_finished = threading.Event()
    errors = []

    def hold_transaction():
        try:
            with coordinator:
                writer._conn.execute("BEGIN IMMEDIATE")
                writer._conn.execute("INSERT INTO backup_probe(value) VALUES ('committed')")
                transaction_started.set()
                assert release_transaction.wait(timeout=5)
                writer._conn.commit()
        except BaseException as exc:
            errors.append(exc)

    def take_backup():
        destination = None
        try:
            destination = sqlite3.connect(backup_path)
            snapshot_source.backup(destination)
            backup_finished.set()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if destination is not None:
                destination.close()

    holder = threading.Thread(target=hold_transaction)
    backup_thread = threading.Thread(target=take_backup)
    holder.start()
    assert transaction_started.wait(timeout=5)
    backup_thread.start()
    assert not backup_finished.wait(timeout=0.1)
    release_transaction.set()
    holder.join(timeout=5)
    backup_thread.join(timeout=5)

    assert not holder.is_alive()
    assert not backup_thread.is_alive()
    assert backup_finished.is_set()
    assert errors == []
    snapshot_source.close()
    writer.close()

    snapshot = sqlite3.connect(backup_path)
    try:
        assert snapshot.execute("SELECT COUNT(*) FROM backup_probe").fetchone()[0] == 1
        assert snapshot.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        snapshot.close()
