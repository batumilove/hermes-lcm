"""Process-local proof supplied only at the host's actual deepcopy boundary.

No markers enter message dictionaries or provider requests. The context is local
to the worker and disappears on success, exception or cancellation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from .admission_binding import projection

_snapshots = ContextVar('lcm_snapshot_provenance', default=())


def original_for(engine, message):
    for owner, session, conversation, pairs in reversed(_snapshots.get()):
        if owner is not engine or session != engine._session_id or conversation != engine._conversation_id:
            continue
        pair = pairs.get(id(message))
        if pair is None or pair[0] is not message:
            continue
        copied, original, signature = pair
        if projection(copied) == signature == projection(original):
            return original
    return message


@contextmanager
def snapshot_scope(engine, originals, copies):
    pairs = {}
    if (len(originals) == len(copies)
        and len({id(m) for m in originals}) == len(originals)
        and len({id(m) for m in copies}) == len(copies)):
        for original, copied in zip(originals, copies):
            signature = projection(original)
            if signature is not None and signature == projection(copied):
                pairs[id(copied)] = (copied, original_for(engine, original), signature)
    entry = (engine, engine._session_id, engine._conversation_id, pairs)
    token = _snapshots.set((*_snapshots.get(), entry))
    try:
        yield
    finally:
        # Revoke this proof even in a child that captured the worker context.
        pairs.clear()
        _snapshots.reset(token)
