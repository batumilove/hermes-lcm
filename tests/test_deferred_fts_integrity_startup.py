"""Protected contracts for deferring deep FTS scans past engine construction."""

from __future__ import annotations

import sqlite3
import threading
import time

from hermes_lcm import db_bootstrap, engine as engine_mod, sqlite_util
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _delete_integrity_markers(db_path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "DELETE FROM metadata WHERE key LIKE 'fts_integrity_checked_at:%'"
        )
        connection.execute(
            "DELETE FROM metadata WHERE key LIKE 'fts_integrity_scan_started_at:%'"
        )
        connection.commit()
    finally:
        connection.close()


def test_engine_construction_finishes_before_due_background_scan_starts(
    tmp_path, monkeypatch
):
    """A due deep scan must not race the remaining helper constructors."""

    db_path = tmp_path / "lcm.db"
    config = LCMConfig(database_path=str(db_path))

    # Build a structurally complete database without a background thread, then
    # make both FTS checks due for the construction under test. The contract is
    # about startup ordering, not the host filesystem's free-space policy.
    monkeypatch.setattr(db_bootstrap, "_check_disk_space", lambda _path: True)
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "false")
    seed = LCMEngine(config=config)
    seed.shutdown()
    _delete_integrity_markers(db_path)
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "true")

    all_scans_started = threading.Event()
    release_scan = threading.Event()
    started_tables = set()
    started_lock = threading.Lock()
    real_check = db_bootstrap.check_external_content_fts_integrity

    def slow_check(connection, spec):
        with started_lock:
            started_tables.add(spec.table_name)
            if started_tables == {"messages_fts", "nodes_fts"}:
                all_scans_started.set()
        assert release_scan.wait(timeout=5)
        return real_check(connection, spec)

    monkeypatch.setattr(
        db_bootstrap, "check_external_content_fts_integrity", slow_check
    )
    # Make the historical construction failure deterministic and fast.
    monkeypatch.setattr(
        sqlite_util, "_PROCESS_SQLITE_WRITE_LOCK_TIMEOUT_SECONDS", 0.1
    )
    real_summary_dag = engine_mod.SummaryDAG

    def delayed_summary_dag(*args, **kwargs):
        time.sleep(0.05)
        return real_summary_dag(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "SummaryDAG", delayed_summary_dag)

    engine = None
    try:
        engine = LCMEngine(
            config=config,
            _defer_integrity_scans_until_activation=True,
        )
        assert not started_tables
        connection = sqlite3.connect(db_path)
        try:
            assert connection.execute(
                "SELECT COUNT(*) FROM metadata "
                "WHERE key LIKE 'fts_integrity_scan_started_at:%'"
            ).fetchone()[0] == 0
        finally:
            connection.close()

        engine.start_deferred_integrity_scans()
        assert all_scans_started.wait(timeout=2)
        assert started_tables == {"messages_fts", "nodes_fts"}
    finally:
        release_scan.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        if engine is not None:
            engine.shutdown()


def test_standalone_engine_starts_due_background_scans_after_binding(
    tmp_path, monkeypatch
):
    """Direct construction preserves the historical maintenance behavior."""

    db_path = tmp_path / "standalone-lcm.db"
    config = LCMConfig(database_path=str(db_path))
    monkeypatch.setattr(db_bootstrap, "_check_disk_space", lambda _path: True)
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "false")
    seed = LCMEngine(config=config)
    seed.shutdown()
    _delete_integrity_markers(db_path)
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "true")

    all_scans_started = threading.Event()
    release_scan = threading.Event()
    started_tables = set()
    started_lock = threading.Lock()
    real_check = db_bootstrap.check_external_content_fts_integrity

    def slow_check(connection, spec):
        with started_lock:
            started_tables.add(spec.table_name)
            if started_tables == {"messages_fts", "nodes_fts"}:
                all_scans_started.set()
        assert release_scan.wait(timeout=5)
        return real_check(connection, spec)

    monkeypatch.setattr(
        db_bootstrap, "check_external_content_fts_integrity", slow_check
    )

    engine = None
    try:
        engine = LCMEngine(config=config)
        assert all_scans_started.wait(timeout=2)
        assert started_tables == {"messages_fts", "nodes_fts"}
    finally:
        release_scan.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        if engine is not None:
            engine.shutdown()
