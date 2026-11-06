"""Receipts for admitted native rows; never infer identities for legacy history.

The caller must pass the original host projection, before payload protection.
Only explicit persisted native UIDs qualify. Content changes create a different
receipt. A receipt is useful only while its exact bound durable row survives.
"""

import hashlib
import json

from .message_content import normalize_content_value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def receipt_key(session_id, conversation_id, message):
    uid = message.get('message_uid')
    if message.get('_db_persisted') is not True or not isinstance(uid, str) or not uid.strip():
        return None
    try:
        # Bind to the original host projection, not just a reused tool-call ID.
        identity = [message.get(key) for key in
                    ('role', 'content', 'tool_call_id', 'tool_calls', 'tool_name')]
        return 'native_ingest_identity_v1:' + _digest([
            session_id, (conversation_id or '').strip(), uid, identity])
    except (TypeError, ValueError):
        return None


def durable_digest(row):
    return _digest([
        row['store_id'], row['session_id'], row.get('conversation_id') or '',
        row['timestamp'], row.get('source'), row.get('role'),
        normalize_content_value(row.get('content')), row.get('tool_call_id'),
        row.get('tool_calls'), row.get('tool_name'),
    ])


def bound_row(store, key):
    if key is None:
        return None
    try:
        receipt = store.read_metadata_json(key)
        if not isinstance(receipt, dict) or receipt.get('version') != 1 or receipt.get('key') != key:
            return None
        row_id = receipt.get('store_id')
        if type(row_id) is not int:
            return None
        row = store.get(row_id)
        if row is not None and durable_digest(row) == receipt.get('durable_sha256'):
            return row_id
    except (TypeError, ValueError, KeyError):
        # Malformed/stale receipts are not evidence for dropping a message.
        return None
    return None


def record_in_transaction(store, key, row_id):
    if key is None:
        return
    row = store.get(row_id)
    receipt = {'version': 1, 'key': key, 'store_id': row_id,
               'durable_sha256': durable_digest(row)}
    store._conn.execute(
        'INSERT INTO metadata(key,value) VALUES (?,?) '
        'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
        (key, json.dumps(receipt, separators=(',', ':'))),
    )


def bind_proven_replay_in_transaction(store, session_id, conversation_id, message):
    """Bind a row already proved replay by the existing engine, if unambiguous.

    This must not be called for ignored, quarantined, or unexamined messages.
    Exact raw equality is deliberately conservative for externalized payloads.
    """
    key = receipt_key(session_id, conversation_id, message)
    if key is None or bound_row(store, key) is not None:
        return
    calls = message.get('tool_calls')
    matches = store._conn.execute(
        'SELECT store_id FROM messages WHERE session_id=? AND conversation_id=? '
        'AND role=? AND content IS ? AND tool_call_id IS ? AND tool_calls IS ? '
        'AND tool_name IS ? LIMIT 2',
        (session_id, (conversation_id or '').strip(), message.get('role', 'unknown'),
         normalize_content_value(message.get('content')), message.get('tool_call_id'),
         json.dumps(calls) if calls else None, message.get('tool_name')),
    ).fetchall()
    if len(matches) == 1:
        record_in_transaction(store, key, matches[0][0])
