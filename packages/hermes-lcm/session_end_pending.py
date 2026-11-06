"""Crash-durable pending session-end intent files.

SQLite cannot record an outbox row while another writer owns the database.  These
small sidecars are therefore the pre-SQLite durability boundary for bounded
session-end hooks.  Each intent is content-addressed, written with O_EXCL, fsynced,
and published with an atomic rename plus parent-directory fsync.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable

_PENDING_DIR_SUFFIX = "-session-end-pending"
_INTENT_VERSION = 3
_SUPPORTED_INTENT_VERSIONS = (1, 2, 3)


# Upper bound for any single durability fsync on the session-end path.
# The sidecar intent file is a best-effort pre-SQLite durability boundary:
# losing one merely defers session-end processing to SQLite. A storage-level
# stall (e.g. a multi-GB state.db WAL checkpoint storm observed 2026-08-23)
# must never block the memory-provider shutdown path past the gateway
# loop-liveness watchdog horizon, so every fsync here is time-bounded.
_FSYNC_BUDGET_S = 10.0


def _bounded_fsync(fd: int) -> None:
    """fsync with a hard time budget; raises TimeoutError past the budget.

    fsync on a file descriptor cannot be interrupted portably, so run it in a
    daemon helper thread and wait at most _FSYNC_BUDGET_S. A hung kernel-side
    flush keeps the helper thread (daemon: process exit is not blocked), while
    the caller surfaces TimeoutError and abandons the sidecar durability.
    """
    import threading

    done = threading.Event()

    def _run() -> None:
        try:
            os.fsync(fd)
        finally:
            done.set()

    helper = threading.Thread(target=_run, daemon=True, name="lcm-bounded-fsync")
    helper.start()
    if not done.wait(timeout=_FSYNC_BUDGET_S):
        raise TimeoutError(
            f"fsync exceeded {_FSYNC_BUDGET_S}s budget; abandoning session-end "
            "sidecar durability (SQLite remains the durable boundary)"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        _bounded_fsync(fd)
    finally:
        os.close(fd)


def pending_session_end_dir(db_path: str | Path) -> Path:
    database = Path(db_path).expanduser().resolve()
    return database.parent / f"{database.name}{_PENDING_DIR_SUFFIX}"


def _canonical_payload(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def session_end_message_fingerprints(messages: list[Dict[str, Any]]) -> list[str]:
    return [
        hashlib.sha256(_canonical_payload(message)).hexdigest()
        for message in messages
    ]


def build_session_end_intent(
    *,
    session_id: str,
    conversation_id: str,
    source: str,
    frontier_store_id: int,
    messages: list[Dict[str, Any]],
    ingest_cursor: int = 0,
    represented_prefix_fingerprints: list[str] | None = None,
) -> Dict[str, Any]:
    cursor = int(ingest_cursor or 0)
    if cursor < 0 or cursor > len(messages):
        raise ValueError("pending session-end ingest cursor is out of range")
    message_fingerprints = session_end_message_fingerprints(messages)
    represented_fingerprints = list(represented_prefix_fingerprints or [])
    if represented_fingerprints != message_fingerprints[: len(represented_fingerprints)]:
        raise ValueError("pending session-end represented prefix fingerprints mismatch")
    if len(represented_fingerprints) > cursor:
        raise ValueError("pending session-end represented prefix exceeds ingest cursor")
    identity = {
        "version": _INTENT_VERSION,
        "session_id": str(session_id),
        "conversation_id": str(conversation_id),
        "source": str(source or ""),
        "frontier_store_id": int(frontier_store_id or 0),
        "messages": messages,
        "ingest_cursor": cursor,
        "message_fingerprints": message_fingerprints,
        "represented_prefix_fingerprints": represented_fingerprints,
    }
    digest = hashlib.sha256(_canonical_payload(identity)).hexdigest()
    return {**identity, "intent_sha256": digest, "created_at": time.time()}


def intent_path(directory: Path, intent: Dict[str, Any]) -> Path:
    session_digest = hashlib.sha256(
        str(intent.get("session_id") or "").encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    return directory / f"{session_digest}-{intent['intent_sha256']}.json"


def persist_session_end_intent(db_path: str | Path, intent: Dict[str, Any]) -> Path:
    directory = pending_session_end_dir(db_path)
    directory_was_missing = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    if directory_was_missing:
        _fsync_directory(directory.parent)
    destination = intent_path(directory, intent)
    if destination.exists():
        return destination
    temporary = destination.with_name(f".{destination.name}.{time.time_ns():x}.tmp")
    data = json.dumps(intent, ensure_ascii=False, sort_keys=True, indent=2, default=str)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            _bounded_fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(directory)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def load_session_end_intent(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version") or 0)
    if version not in _SUPPORTED_INTENT_VERSIONS:
        raise ValueError("unsupported pending session-end intent version")
    supplied = str(payload.get("intent_sha256") or "")
    identity_keys = [
        "version",
        "session_id",
        "conversation_id",
        "source",
        "frontier_store_id",
        "messages",
    ]
    if version >= 2:
        identity_keys.extend(("ingest_cursor", "message_fingerprints"))
    if version >= 3:
        identity_keys.append("represented_prefix_fingerprints")
    identity = {key: payload.get(key) for key in identity_keys}
    expected = hashlib.sha256(_canonical_payload(identity)).hexdigest()
    if not supplied or supplied != expected:
        raise ValueError("pending session-end intent digest mismatch")
    if not isinstance(payload.get("messages"), list):
        raise ValueError("pending session-end intent messages must be a list")
    if version >= 2:
        cursor = payload.get("ingest_cursor")
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise ValueError("pending session-end ingest cursor must be an integer")
        if cursor < 0 or cursor > len(payload["messages"]):
            raise ValueError("pending session-end ingest cursor is out of range")
        fingerprints = payload.get("message_fingerprints")
        if fingerprints != session_end_message_fingerprints(payload["messages"]):
            raise ValueError("pending session-end message fingerprints mismatch")
    if version >= 3:
        represented_fingerprints = payload.get("represented_prefix_fingerprints")
        if not isinstance(represented_fingerprints, list) or not all(
            isinstance(item, str) for item in represented_fingerprints
        ):
            raise ValueError(
                "pending session-end represented prefix fingerprints must be a list of strings"
            )
        if represented_fingerprints != payload["message_fingerprints"][: len(represented_fingerprints)]:
            raise ValueError("pending session-end represented prefix fingerprints mismatch")
        if len(represented_fingerprints) > payload["ingest_cursor"]:
            raise ValueError("pending session-end represented prefix exceeds ingest cursor")
    return payload


def iter_session_end_intents(db_path: str | Path) -> Iterable[Path]:
    directory = pending_session_end_dir(db_path)
    if not directory.exists():
        return ()
    return tuple(sorted(directory.glob("*.json")))


def remove_session_end_intent(path: Path) -> None:
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def quarantine_session_end_intent(path: Path) -> Path:
    """Atomically exclude a permanently invalid intent from normal discovery."""
    destination = path.with_suffix(path.suffix + ".invalid")
    counter = 0
    while destination.exists():
        counter += 1
        destination = path.with_suffix(path.suffix + f".{counter}.invalid")
    os.replace(path, destination)
    _fsync_directory(path.parent)
    return destination
