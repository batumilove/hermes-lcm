"""Regression coverage for externally held preflight SQLite locks."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sqlite3

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _hold_external_sqlite_writer(
    database_path: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("external-lock-probe", "held"),
        )
        ready.set()
        if not release.wait(timeout=60.0):
            raise RuntimeError("external lock probe was not released")
    finally:
        connection.rollback()
        connection.close()


def test_preflight_lock_warning_identifies_external_writer(tmp_path, caplog):
    database_path = tmp_path / "lcm.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(database_path)))
    engine.on_session_start("preflight-lock-session", platform="telegram")
    expected_busy_timeout_ms = engine._store._conn.execute(
        "PRAGMA busy_timeout"
    ).fetchone()[0]

    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_external_sqlite_writer,
        args=(str(database_path), ready, release),
        name="lcm-external-lock-probe",
    )
    holder.start()
    try:
        assert ready.wait(timeout=5.0)
        with caplog.at_level(logging.WARNING):
            assert (
                engine.should_compress_preflight(
                    [{"role": "user", "content": "must remain durable"}]
                )
                is False
            )

        warning = next(
            record.getMessage()
            for record in caplog.records
            if "LCM ingest failed (preflight): database is locked" in record.getMessage()
        )
        diagnostic_json = warning.split(" sqlite_lock_diagnostics=", 1)[1]
        diagnostics = json.loads(diagnostic_json)

        assert diagnostics["operation"] == "ingest_messages"
        assert diagnostics["phase"] == "preflight"
        assert diagnostics["process_writer_owner"] == {}
        assert diagnostics["connection"]["in_transaction"] is False
        assert diagnostics["connection"]["busy_timeout_ms"] == expected_busy_timeout_ms
        assert diagnostics["files"]["db"]["inode"] == database_path.stat().st_ino
        assert diagnostics["files"]["db"]["device"] == database_path.stat().st_dev
        assert diagnostics["external_lock_holders_truncated"] is False
        assert any(
            holder_info["pid"] == holder.pid
            and holder_info["target"] in {"db", "wal", "shm"}
            for holder_info in diagnostics["external_lock_holders"]
        )
    finally:
        release.set()
        holder.join(timeout=5.0)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5.0)
        engine.shutdown()

    assert holder.exitcode == 0


def test_preflight_lock_warning_identifies_uncoordinated_same_process_writer(
    tmp_path, caplog
):
    database_path = tmp_path / "lcm.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(database_path),
            empty_lifecycle_gc_enabled=False,
        )
    )
    engine.on_session_start("same-process-lock-session", platform="telegram")
    engine._store._conn.execute("PRAGMA busy_timeout = 100")

    holder = sqlite3.connect(database_path)
    try:
        holder.execute("BEGIN IMMEDIATE")
        holder.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("same-process-lock-probe", "held"),
        )
        with caplog.at_level(logging.WARNING):
            assert (
                engine.should_compress_preflight(
                    [{"role": "user", "content": "must remain durable"}]
                )
                is False
            )

        warning = next(
            record.getMessage()
            for record in caplog.records
            if "LCM ingest failed (preflight): database is locked" in record.getMessage()
        )
        diagnostics = json.loads(warning.split(" sqlite_lock_diagnostics=", 1)[1])

        assert diagnostics["process_writer_owner"] == {}
        assert diagnostics["external_lock_holders"] == []
        assert diagnostics["same_process_lock_holders_truncated"] is False
        assert any(
            holder_info["pid"] == os.getpid()
            and holder_info["target"] in {"db", "wal", "shm"}
            for holder_info in diagnostics["same_process_lock_holders"]
        )
    finally:
        holder.rollback()
        holder.close()
        engine.shutdown()
