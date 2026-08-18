"""Owner-history attribution for post-release lock diagnostics.

Class B field evidence: preflight ``BEGIN IMMEDIATE`` fails with
``database is locked`` while ``process_writer_owner`` is empty. All in-process
lcm.db writers coordinate through ``ProcessSQLiteWriteLock``, and ``release()``
erases owner metadata, so diagnostics captured after the holder released see
``{}``. These tests pin the fix: a bounded ring of recently released owners.
"""

from __future__ import annotations

import threading
import time

from hermes_lcm.sqlite_util import process_sqlite_write_lock


def test_recent_owners_empty_without_history(tmp_path):
    lock = process_sqlite_write_lock(tmp_path / "owner-history-empty.db")
    assert lock.recent_owners() == []


def test_release_records_owner_history(tmp_path):
    lock = process_sqlite_write_lock(tmp_path / "owner-history-basic.db")
    with lock.attributed("test_operation_alpha"):
        pass
    history = lock.recent_owners()
    assert len(history) == 1
    entry = history[0]
    assert entry["operation"] == "test_operation_alpha"
    assert entry["thread_id"] == threading.get_ident()
    assert entry["held_seconds"] >= 0.0
    assert entry["released_monotonic"] > 0.0


def test_reentrant_release_records_once(tmp_path):
    lock = process_sqlite_write_lock(tmp_path / "owner-history-reentrant.db")
    with lock.attributed("outer_op"):
        with lock.attributed("inner_op"):
            pass
    history = lock.recent_owners()
    assert len(history) == 1
    # The innermost completed operation label is what diagnostics should see
    # at release time; assert the recorded operation is one of the two labels.
    assert history[0]["operation"] in {"outer_op", "inner_op"}


def test_history_is_bounded(tmp_path):
    lock = process_sqlite_write_lock(tmp_path / "owner-history-bound.db")
    for index in range(40):
        with lock.attributed(f"op_{index:02d}"):
            pass
    history = lock.recent_owners()
    assert 0 < len(history) <= 16
    assert history[-1]["operation"] == "op_39"


def test_concurrent_release_does_not_corrupt_history(tmp_path):
    lock = process_sqlite_write_lock(tmp_path / "owner-history-concurrent.db")

    def worker(name: str) -> None:
        for _ in range(25):
            with lock.attributed(name):
                time.sleep(0.0005)

    threads = [
        threading.Thread(target=worker, args=(f"worker_{i}",)) for i in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    history = lock.recent_owners()
    assert 0 < len(history) <= 16
    for entry in history:
        assert entry["operation"].startswith("worker_")
        assert isinstance(entry["thread_id"], int)
