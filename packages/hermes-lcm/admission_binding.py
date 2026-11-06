"""Bind late native IDs to this engine's proven, unchanged live admissions.

No content-only lookup, legacy inference, deletion, or cross-session matching.
Strong references prove object identity and prevent Python object-ID reuse.
"""

from .native_identity import (
    _digest,
    bound_row,
    durable_digest,
    receipt_key,
    record_in_transaction,
)


def projection(message):
    try:
        return _digest([
            message.get(key)
            for key in ('role', 'content', 'tool_call_id', 'tool_calls', 'tool_name')
        ])
    except (TypeError, ValueError):
        return None


def remember_admissions(engine, messages, indexed, row_ids):
    """Remember returned row IDs only for the original admitted dictionaries."""
    cache = getattr(engine, '_pending_native_admissions', {})
    # A private compression snapshot must not erase original-object proof.
    cache = {key: value for key, value in cache.items()
             if value[2:4] == (engine._session_id, engine._conversation_id)}
    from .snapshot_provenance import original_for
    for (index, _), row_id in zip(indexed, row_ids):
        message = original_for(engine, messages[index])
        if receipt_key(engine._session_id, engine._conversation_id, messages[index]) is not None:
            continue
        signature = projection(message)
        if signature is None:
            continue
        row = engine._store.get(row_id)
        if row is None:
            continue
        cache[id(message)] = (
            message, signature, engine._session_id, engine._conversation_id,
            row_id, durable_digest(row),
        )
    # Bound memory retention. Eviction loses only a dedup opportunity; it never
    # manufactures proof or discards a message. Native receipts remain durable.
    while len(cache) > 4096:
        cache.pop(next(iter(cache)))
    engine._pending_native_admissions = cache


def bind_late_identities(engine, messages):
    """Upgrade proven admissions before the ingest cursor can return early."""
    cache = getattr(engine, '_pending_native_admissions', {})
    if not cache:
        return
    from .snapshot_provenance import original_for
    originals = [original_for(engine, message) for message in messages]
    active = {id(message): message for message in originals}
    cache = {key: value for key, value in cache.items()
             if value[2:4] == (engine._session_id, engine._conversation_id)}
    engine._pending_native_admissions = cache
    eligible = []
    for object_id, (message, signature, session, conversation, row_id, digest) in cache.items():
        if (
            active.get(object_id) is not message
            or session != engine._session_id
            or conversation != engine._conversation_id
            or projection(message) != signature
        ):
            continue
        key = receipt_key(session, conversation, message)
        if key is not None:
            eligible.append((object_id, key, row_id, digest))
    if not eligible:
        return

    store = engine._store
    completed = []
    with store._write_lock, store._conn:
        if not store._conn.in_transaction:
            store._conn.execute('BEGIN IMMEDIATE')
        for object_id, key, row_id, digest in eligible:
            row = store.get(row_id)
            if row is None or durable_digest(row) != digest:
                continue
            # Never replace an already valid binding with a competing row.
            if bound_row(store, key) is None:
                record_in_transaction(store, key, row_id)
            completed.append(object_id)
    # Preserve pending evidence if the transaction fails, so a retry is safe.
    for object_id in completed:
        cache.pop(object_id, None)
