import copy
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def open_engine(path):
    cfg = LCMConfig(database_path=str(path), embeddings_enabled=False,
                    empty_lifecycle_gc_enabled=False,
                    large_output_externalization_enabled=False)
    engine = LCMEngine(config=cfg, hermes_home=str(path.parent))
    engine.on_session_start('identity-session', platform='test',
                            conversation_id='identity-conversation', context_length=200000)
    return engine


def message(uid, content):
    return dict(role='user', content=content, message_uid=uid, _db_persisted=True)


def rows(engine):
    return engine._store.get_session_messages('identity-session', limit=10000)


def test_restart_after_interleaved_admission_is_idempotent(tmp_path):
    path = tmp_path / 'replay.db'
    engine = open_engine(path)
    durable = []
    for i in range(3):
        durable += [
            message(f'u{i}', f'request {i}'),
            dict(role='assistant', content='', message_uid=f'a{i}', _db_persisted=True,
                 tool_calls=[dict(id=f'call{i}', type='function',
                                 function=dict(name='inspect', arguments='{}'))]),
            dict(role='tool', content=f'result {i}', tool_call_id=f'call{i}',
                 tool_name='inspect', message_uid=f't{i}', _db_persisted=True),
            dict(role='assistant', content=f'answer {i}', message_uid=f'b{i}', _db_persisted=True),
        ]
    engine.ingest(copy.deepcopy(durable))
    engine.shutdown()
    interleaved = []
    for i, item in enumerate(durable):
        interleaved += [item, message(f'filler{i}', f'new filler {i}')]
    engine = open_engine(path)
    engine.ingest(copy.deepcopy(interleaved))
    before = len(rows(engine))
    engine.shutdown()
    engine = open_engine(path)
    engine.ingest(copy.deepcopy(interleaved))
    assert len(rows(engine)) == before
    engine.shutdown()


def test_new_uid_repeated_text_is_preserved(tmp_path):
    engine = open_engine(tmp_path / 'new.db')
    first = [message('one', 'repeat this')]
    engine.ingest(copy.deepcopy(first))
    engine.ingest(copy.deepcopy(first + [message('two', 'repeat this')]))
    assert [r['content'] for r in rows(engine)] == ['repeat this', 'repeat this']
    engine.shutdown()


def test_atomic_receipt_and_row_rollback(tmp_path):
    engine = open_engine(tmp_path / 'rollback.db')
    conn = engine._store._conn
    conn.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON metadata WHEN NEW.key LIKE 'native_ingest_identity_v1:%' BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END")
    conn.commit()
    # The public hook logs storage errors instead of propagating them. The
    # transaction must still roll back, and retry must retain the message.
    engine.ingest([message('atomic', 'must roll back')])
    assert not rows(engine)
    assert conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'native_ingest_identity_v1:%'").fetchone()[0] == 0
    conn.execute('DROP TRIGGER reject_receipt')
    conn.commit()
    engine.ingest([message('atomic', 'must roll back')])
    assert len(rows(engine)) == 1
    engine.shutdown()


def test_identity_receipt_validates_row_and_input(tmp_path):
    from hermes_lcm.native_identity import receipt_key
    engine = open_engine(tmp_path / 'identity.db')
    store = engine._store
    msg = message('stable', 'original')
    def append(raw, **scope):
        return store._append_protected_batch(scope.get('session', 'identity-session'), [raw], [1],
             conversation_id=scope.get('conversation', 'identity-conversation'),
             native_identity_messages=[raw])
    first = append(msg)
    assert append(msg) == first
    assert len(rows(engine)) == 1
    assert append(message('new-id', 'original')) != first
    assert append(message('stable', 'changed')) != first
    tool = dict(role='tool', content='old result', tool_call_id='reused',
                message_uid='tool-id', _db_persisted=True)
    tool_id = append(tool)
    assert append(dict(tool, content='new result')) != tool_id
    assert append(dict(tool, tool_name='different')) != tool_id
    assert append(msg, conversation='other') != first
    assert append(msg, session='other') != first
    key = receipt_key('identity-session', 'identity-conversation', msg)
    other_key = receipt_key('identity-session', 'identity-conversation', message('new-id', 'original'))
    other_value = store._conn.execute('SELECT value FROM metadata WHERE key=?', (other_key,)).fetchone()[0]
    original_value = store._conn.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()[0]
    store._conn.execute('UPDATE metadata SET value=? WHERE key=?', (other_value, key))
    store._conn.commit()
    before = len(rows(engine))
    append(msg)
    assert len(rows(engine)) == before + 1
    store._conn.execute('UPDATE metadata SET value=? WHERE key=?', (original_value, key))
    store._conn.commit()
    store._conn.execute('UPDATE messages SET content=? WHERE store_id=?', ('changed durable row', first[0]))
    store._conn.commit()
    replacement = append(msg)
    assert replacement != first
    store._conn.execute('DELETE FROM messages WHERE store_id=?', (replacement[0],))
    store._conn.commit()
    append(msg)
    store._conn.execute('UPDATE metadata SET value=? WHERE key=?', ('bad json', key))
    store._conn.commit()
    before = len(rows(engine))
    append(msg)
    assert len(rows(engine)) == before + 1
    untrusted = dict(msg, _db_persisted=False)
    before = len(rows(engine))
    append(untrusted); append(untrusted)
    assert len(rows(engine)) == before + 2
    assert store._conn.execute('PRAGMA integrity_check').fetchall()[0][0] == 'ok'
    engine.shutdown()


def test_receipt_lookup_starts_inside_write_transaction(tmp_path, monkeypatch):
    import hermes_lcm.store as store_module
    engine = open_engine(tmp_path / 'transaction.db')
    original = store_module.bound_row
    lookups = []
    def checked(store, key):
        if key is not None:
            lookups.append(store._conn.in_transaction)
        return original(store, key)
    monkeypatch.setattr(store_module, 'bound_row', checked)
    engine.ingest([message('transaction', 'one')])
    assert lookups and all(lookups)
    engine.shutdown()


def test_proven_legacy_binding_requires_unique_exact_row(tmp_path):
    from hermes_lcm.native_identity import receipt_key, bound_row
    engine = open_engine(tmp_path / 'legacy.db')
    store = engine._store
    msg = message('legacy-uid', 'already represented')
    scope = dict(conversation_id='identity-conversation')
    first = store._append_protected_batch('identity-session', [msg], [1], **scope)
    key = receipt_key('identity-session', 'identity-conversation', msg)
    assert bound_row(store, key) is None
    store._append_protected_batch('identity-session', [], [], native_replay_messages=[msg], **scope)
    assert bound_row(store, key) == first[0]
    store._conn.execute('DELETE FROM metadata WHERE key=?', (key,))
    store._conn.commit()
    store._append_protected_batch('identity-session', [msg], [1], **scope)
    store._append_protected_batch('identity-session', [], [], native_replay_messages=[msg], **scope)
    assert bound_row(store, key) is None
    mismatch = message('other', 'no durable match')
    store._append_protected_batch('identity-session', [], [], native_replay_messages=[mismatch], **scope)
    assert bound_row(store, receipt_key('identity-session', 'identity-conversation', mismatch)) is None
    assert len(rows(engine)) == 2
    engine.shutdown()
