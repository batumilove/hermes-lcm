"""Journal-visibility contract for session-end deferred events.

The fixed-grid acceptance sampler relies on journal records to pair
``outcome=scheduled`` deferred intents with ``outcome=settled`` drain
receipts. The gateway's journal capture drops plugin INFO-level records,
so settlement must be logged at WARNING for both v1 and v2 intents —
otherwise settlement is unverifiable and the acceptance rule
"scheduled deferred intent lacking settlement at closeout" fails even
when the drain completed.
"""

import hashlib
import logging
import threading
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm import engine as _engine_module

LOGGER_NAME = _engine_module.logger.name
from hermes_lcm.session_end_pending import (
    build_session_end_intent,
    persist_session_end_intent,
)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _drain_one_and_collect(root, path):
    records = []
    handler = logging.Handler()

    def emit(record):
        records.append(record)

    handler.emit = emit
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    try:
        root._drain_one_session_end_intent(path)
    finally:
        logger.removeHandler(handler)
    return [
        r
        for r in records
        if "outcome=settled" in r.getMessage()
        and "operation=session_end_deferred_drain" in r.getMessage()
    ]


@pytest.mark.filterwarnings("ignore")
def test_settled_drain_event_logged_at_warning_for_v2_intent(tmp_path, caplog):
    db_path = tmp_path / "settled-visibility-v2.db"
    root = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    root.on_session_start("v2-session", platform="telegram")
    path = persist_session_end_intent(
        db_path,
        build_session_end_intent(
            session_id="v2-session",
            conversation_id="conversation:v2-session",
            source="telegram",
            frontier_store_id=0,
            messages=[{"role": "user", "content": "v2 deferred"}],
            ingest_cursor=0,
        ),
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        settled = _drain_one_and_collect(root, path)
    assert settled, "v2 settlement must emit a journal-visible record"
    assert all(r.levelno >= logging.WARNING for r in settled)


def test_settled_drain_event_logged_at_warning_for_v1_intent(tmp_path, caplog):
    db_path = tmp_path / "settled-visibility-v1.db"
    root = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    root.on_session_start("v1-session", platform="telegram")
    intent = build_session_end_intent(
        session_id="v1-session",
        conversation_id="conversation:v1-session",
        source="telegram",
        frontier_store_id=0,
        messages=[{"role": "user", "content": "v1 deferred"}],
        ingest_cursor=0,
    )
    # Downgrade to a *valid* legacy v1 intent: re-sign over the v1 identity
    # keys (version/session_id/conversation_id/source/frontier_store_id/
    # messages) exactly as load_session_end_intent verifies them.
    intent["version"] = 1
    identity = {
        key: intent.get(key)
        for key in (
            "version",
            "session_id",
            "conversation_id",
            "source",
            "frontier_store_id",
            "messages",
        )
    }
    from hermes_lcm.session_end_pending import _canonical_payload

    intent["intent_sha256"] = hashlib.sha256(
        _canonical_payload(identity)
    ).hexdigest()
    path = persist_session_end_intent(db_path, intent)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        settled = _drain_one_and_collect(root, path)
    assert settled, "v1 settlement must also emit a journal-visible record"
    assert all(r.levelno >= logging.WARNING for r in settled)


def test_singleflight_scheduled_event_logged_at_warning(tmp_path):
    # The singleflight handoff schedules the drain; its scheduled event must
    # be journal-visible to pair with the settled record.
    db_path = tmp_path / "singleflight-visibility.db"
    root = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    root.on_session_start("singleflight-session", platform="telegram")

    records = []
    handler = logging.Handler()

    def emit(record):
        records.append(record)

    handler.emit = emit
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    try:
        flush_lock = root._storage_owner._session_end_flush_lock
        assert flush_lock.acquire(blocking=False)
        try:
            root.on_session_end(
                "singleflight-session",
                [{"role": "user", "content": "singleflight visibility"}],
            )
        finally:
            flush_lock.release()
    finally:
        logger.removeHandler(handler)
    scheduled = [
        r
        for r in records
        if "outcome=scheduled" in r.getMessage()
        and "operation=session_end_singleflight_handoff" in r.getMessage()
    ]
    assert scheduled, "singleflight handoff must emit a journal-visible record"
    assert all(r.levelno >= logging.WARNING for r in scheduled)
    root.shutdown()
