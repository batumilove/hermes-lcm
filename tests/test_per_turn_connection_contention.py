"""Protected RED contract for foreground ingest under internal SQLite contention."""
from __future__ import annotations

import threading
import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_per_turn_ingest_settles_internal_writer_within_existing_overlap_budget(
    tmp_path, caplog
):
    """A short same-process writer must not become a foreground ingest failure.

    The timings scale production's long SQLite wait down to milliseconds while
    retaining the same shape: two immediate/busy-timeout attempts are not enough,
    but the writer releases within the existing 2.5 s append-overlap budget.
    """
    db_path = tmp_path / "per-turn-connection-contention.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    engine.on_session_start("per-turn-contention", platform="telegram")
    engine._store.connection.execute("PRAGMA busy_timeout=50")

    writer_started = threading.Event()
    release_writer = threading.Event()
    writer_errors: list[BaseException] = []

    def hold_uncoordinated_dag_writer() -> None:
        conn = engine._dag.connection
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                ("per-turn-contention-holder", "active"),
            )
            writer_started.set()
            if not release_writer.wait(timeout=3.0):
                raise TimeoutError("test writer was not released")
            conn.rollback()
        except BaseException as exc:
            writer_errors.append(exc)
            writer_started.set()
            try:
                conn.rollback()
            except BaseException:
                pass

    holder = threading.Thread(
        target=hold_uncoordinated_dag_writer,
        name="lcm-test-per-turn-dag-writer",
    )
    holder.start()
    assert writer_started.wait(timeout=1.0)
    assert writer_errors == []

    def delayed_release() -> None:
        time.sleep(0.35)
        release_writer.set()

    releaser = threading.Thread(target=delayed_release, name="lcm-test-per-turn-release")
    releaser.start()
    messages = [{"role": "user", "content": "persist after transient contention"}]

    try:
        started = time.monotonic()
        with caplog.at_level("INFO", logger="hermes_lcm.engine"):
            engine.ingest(messages)
        elapsed = time.monotonic() - started
    finally:
        release_writer.set()
        releaser.join(timeout=2.0)
        holder.join(timeout=2.0)

    assert not holder.is_alive()
    assert not releaser.is_alive()
    assert writer_errors == []
    rows = engine._store.get_session_messages("per-turn-contention")
    hard_failures = [
        record.getMessage()
        for record in caplog.records
        if "LCM ingest failed (per-turn ingest())" in record.getMessage()
    ]
    evidence = {
        "elapsed": elapsed,
        "rows": [(row.get("role"), row.get("content")) for row in rows],
        "hard_failures": hard_failures,
        "ingest_failures": engine._ingest_failure_count,
        "cursor": engine._ingest_cursor,
    }
    engine.shutdown()

    assert elapsed < 2.5, evidence
    assert hard_failures == [], evidence
    assert engine._ingest_failure_count == 0, evidence
    assert [(row.get("role"), row.get("content")) for row in rows] == [
        ("user", "persist after transient contention")
    ], evidence
