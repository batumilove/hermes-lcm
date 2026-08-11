"""Regression tests for process-wide SQLite writer coordination."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from hermes_lcm import sqlite_util
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
