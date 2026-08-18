"""Session-end ingest must grant append-overlap patience, not time out.

Production failure (2026-08-17 18:51:19): session_end_ingest_finalize waits
only _SESSION_END_PROCESS_WRITE_TIMEOUT_MS=200ms for the process-wide writer
held by _append_protected_batch (observed holds 1.26s-2.13s on the 2.9GB
lcm.db), then raises sqlite3.OperationalError("database is locked: timed out
waiting for process-wide SQLite writer ...") — a hard acceptance marker.

Desired behavior (owner-aware, reconciled with the live append-overlap
patience fix): when the current writer owner is the same-process
``_append_protected_batch``, the session-end flush waits up to
_SESSION_END_APPEND_OVERLAP_TIMEOUT_MS (2500ms); unknown/external owners keep
the strict _SESSION_END_PROCESS_WRITE_TIMEOUT_MS (200ms).
"""
from __future__ import annotations

import threading
import time


from hermes_lcm.engine import (
    _SESSION_END_APPEND_OVERLAP_TIMEOUT_MS,
    _SESSION_END_PROCESS_WRITE_TIMEOUT_MS,
)
from hermes_lcm.sqlite_util import (
    ProcessSQLiteWriteLock,
    _temporary_sqlite_busy_timeout,
)


def test_append_overlap_patience_exceeds_realistic_batch_hold():
    """Contract: the append-overlap wait must exceed observed holds."""
    # Observed append-batch holds in production: 1.259s, 1.579s, 2.128s.
    # NOTE: 2500ms does NOT cover the worst 2.128s observation with the same
    # margin as the earlier flat-4000 design; it covers the ~1.3s typical
    # hold from the v5 campaign telemetry that motivated the live fix. This
    # contract pins the minimum the owner-aware design must grant.
    assert _SESSION_END_APPEND_OVERLAP_TIMEOUT_MS >= 2000, (
        f"append-overlap patience is {_SESSION_END_APPEND_OVERLAP_TIMEOUT_MS}ms; "
        f"must exceed typical observed _append_protected_batch holds (~1.3s) "
        f"and stay above 2s for the worst observed hold"
    )
    assert _SESSION_END_PROCESS_WRITE_TIMEOUT_MS < _SESSION_END_APPEND_OVERLAP_TIMEOUT_MS, (
        "strict timeout must stay tighter than the append-overlap patience so "
        "unknown owners are not granted append-class patience"
    )


def test_session_end_acquire_waits_out_append_batch_hold():
    """End-to-end: contender waits out a same-process append-batch holder."""
    lock = ProcessSQLiteWriteLock()
    release = threading.Event()
    acquired_by_holder = threading.Event()

    def hold_writer():
        with lock.attributed("_append_protected_batch"):
            acquired_by_holder.set()
            release.wait(timeout=10)

    t = threading.Thread(target=hold_writer, name="red-holder")
    t.start()
    assert acquired_by_holder.wait(timeout=5)

    def contender(result: dict) -> None:
        started = time.monotonic()
        try:
            with _temporary_sqlite_busy_timeout(
                [],
                50,
                write_lock=lock,
                write_lock_timeout_ms=_SESSION_END_APPEND_OVERLAP_TIMEOUT_MS,
                write_lock_operation="session_end_ingest_finalize",
            ):
                result["acquired"] = True
        except Exception as exc:  # noqa: BLE001
            result["error"] = repr(exc)
        finally:
            result["elapsed"] = time.monotonic() - started

    # Release the holder from a timer — simulates the batch finishing.
    threading.Timer(0.6, release.set).start()
    result: dict = {}
    ct = threading.Thread(target=contender, args=(result,), name="contender")
    ct.start()
    ct.join(timeout=15)
    release.set()
    t.join(timeout=10)

    assert not ct.is_alive(), "contender hung"
    assert not t.is_alive(), "holder hung"
    assert "error" not in result, f"session-end raised instead of waiting: {result}"
    assert result.get("acquired") is True
    assert result["elapsed"] >= 0.55, (
        f"contender did not actually wait out the hold: {result}"
    )
