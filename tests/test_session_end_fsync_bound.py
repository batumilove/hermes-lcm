"""Hostile test: session-end intent persistence must be time-bounded.

Reproduces the 2026-08-23 watchdog hard-exit: an unbounded fsync (file or
directory) inside persist_session_end_intent allowed a storage-level stall
(10GB state.db WAL checkpoint storm) to block the memory-provider shutdown
path for >180s, tripping the gateway loop-liveness watchdog.

Contract: every fsync in the intent persistence path must be wrapped with a
bounded wait that gives up and surfaces TimeoutError within a small budget
(default <=30s per call). The sidecar file is a best-effort pre-SQLite
durability boundary; losing it merely defers session-end processing to
SQLite - it must never block gateway shutdown.
"""
import os
import time
from pathlib import Path
from unittest import mock

import pytest

from session_end_pending import persist_session_end_intent, _FSYNC_BUDGET_S


def _intent(tmp_path: Path) -> tuple[Path, dict]:
    db = tmp_path / "state.db"
    db.write_bytes(b"x")
    return db, {
        "version": 2,
        "session_id": "s1",
        "conversation_id": "c1",
        "source": "telegram",
        "frontier_store_id": None,
        "messages": [],
        "ingest_cursor": 0,
        "intent_sha256": "a" * 64,
    }


def test_fsync_budget_constant_exists_and_bounded():
    assert 0 < _FSYNC_BUDGET_S <= 30


def test_file_fsync_stall_times_out(tmp_path):
    db, intent = _intent(tmp_path)
    real_fsync = os.fsync
    def hanging_fsync(fd):
        if fd > 2:
            time.sleep(600)
        return real_fsync(fd)
    with mock.patch("session_end_pending.os.fsync", side_effect=hanging_fsync):
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            persist_session_end_intent(db, intent)
    assert time.monotonic() - start < _FSYNC_BUDGET_S + 5


def test_directory_fsync_stall_times_out(tmp_path):
    db, intent = _intent(tmp_path)
    calls = {"n": 0}
    real_fsync = os.fsync
    def dir_only_hang(fd):
        calls["n"] += 1
        if calls["n"] >= 2:
            time.sleep(600)
        return real_fsync(fd)
    with mock.patch("session_end_pending.os.fsync", side_effect=dir_only_hang):
        with pytest.raises(TimeoutError):
            persist_session_end_intent(db, intent)
