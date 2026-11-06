import copy
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
def open_engine(path):
    e=LCMEngine(config=LCMConfig(database_path=str(path),embeddings_enabled=False,empty_lifecycle_gc_enabled=False,large_output_externalization_enabled=False),hermes_home=str(path.parent))
    e.on_session_start('identity-session',platform='test',conversation_id='identity-conversation',context_length=200000)
    return e
def message(uid,content):return dict(role='user',content=content,message_uid=uid,_db_persisted=True)
def rows(e):return e._store.get_session_messages('identity-session',limit=10000)

def test_unproven_restart_is_not_content_deduplicated(tmp_path):
    path=tmp_path/'initial-gap.db'
    initial=[{'role':'user','content':'Update'}, {'role':'assistant','content':'answer one'}, {'role':'user','content':'Continue'}, {'role':'assistant','content':'answer two'}]
    engine=open_engine(path);engine.ingest(copy.deepcopy(initial));assert len(rows(engine))==4;engine.shutdown()
    persisted=[dict(m,message_uid=f'native-{i}',_db_persisted=True) for i,m in enumerate(initial)]
    replay=[]
    for i,m in enumerate(persisted):replay += [m,message(f'new-{i}',f'interleaved new {i}')]
    engine=open_engine(path);engine.ingest(copy.deepcopy(replay))
    contents=[r['content'] for r in rows(engine)];engine.shutdown()
    assert contents.count('Update')==2
    assert contents.count('Continue')==2

def test_same_live_objects_gain_receipts_before_cursor_returns(tmp_path):
    path=tmp_path/'late-native.db';initial=[{'role':'user','content':'Update'},{'role':'assistant','content':'reply'}]
    engine=open_engine(path);engine.ingest(initial)
    for i,m in enumerate(initial):m.update(message_uid=f'native-{i}',_db_persisted=True)
    engine.ingest(initial) # persistence stamped the SAME dicts, cursor already at end
    count=engine._store._conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'native_ingest_identity_v1:%'").fetchone()[0]
    engine.shutdown();assert count==2
    engine=open_engine(path);engine.ingest([initial[0],message('fresh','different'),initial[1]])
    assert [r['content'] for r in rows(engine)].count('Update')==1
    engine.shutdown()

def test_late_binding_does_not_merge_new_occurrence(tmp_path):
    e=open_engine(tmp_path/'repeat.db');m={'role':'user','content':'Continue'};e.ingest([m]);m.update(message_uid='old',_db_persisted=True);e.ingest([m]);e.ingest([m,message('new','Continue')]);assert len(rows(e))==2;e.shutdown()

def test_changed_message_or_row_cannot_gain_late_receipt(tmp_path):
    from hermes_lcm.admission_binding import bind_late_identities
    for mode in ['message','row','scope','copied']:
        e=open_engine(tmp_path/(mode+'.db'));m={'role':'user','content':'original'};e.ingest([m]);m.update(message_uid='uid',_db_persisted=True)
        if mode=='message':m['content']='changed'
        if mode=='row':e._store._conn.execute("UPDATE messages SET content='changed'");e._store._conn.commit()
        if mode=='scope':e._conversation_id='other'
        if mode=='copied':m=dict(m)
        bind_late_identities(e,[m])
        assert e._store._conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'native_ingest_identity_v1:%'").fetchone()[0]==0
        e.shutdown()

def test_late_receipt_failure_is_atomic_and_retryable(tmp_path):
    from hermes_lcm.admission_binding import bind_late_identities
    import sqlite3,pytest
    e=open_engine(tmp_path/'failure.db');msgs=[{'role':'user','content':'one'},{'role':'assistant','content':'two'}];e.ingest(msgs)
    for i,m in enumerate(msgs):m.update(message_uid=str(i),_db_persisted=True)
    e._store._conn.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON metadata WHEN NEW.key LIKE 'native_ingest_identity_v1:%' BEGIN SELECT RAISE(ABORT,'test'); END");e._store._conn.commit()
    with pytest.raises(sqlite3.IntegrityError):bind_late_identities(e,msgs)
    assert len(rows(e))==2;assert len(e._pending_native_admissions)==2
    e._store._conn.execute('DROP TRIGGER reject_receipt');e._store._conn.commit();bind_late_identities(e,msgs)
    assert e._store._conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'native_ingest_identity_v1:%'").fetchone()[0]==2
    assert not e._pending_native_admissions;e.shutdown()

def test_real_gateway_flush_binds_original_objects(tmp_path, monkeypatch):
    # The compatible Hermes host is an explicit test supplier (e.g. PYTHONPATH),
    # not a private deployment checkout. Ordinary imports fail if it is absent.
    from hermes_state import SessionDB
    from agent.session_persistence import _db_flush_write
    from types import SimpleNamespace
    native = SessionDB(db_path=tmp_path/'native.db')
    native.create_session('identity-session',source='test')
    e=open_engine(tmp_path/'lcm.db')
    msgs=[{'role':'user','content':'natural fixture text'},{'role':'assistant','content':'fixture answer'}]
    e.ingest(msgs)
    agent=SimpleNamespace(_session_db=native,session_id='identity-session')
    _db_flush_write(agent,[dict(m) for m in msgs],msgs,msgs)
    assert all(m.get('_db_persisted') is True and m.get('message_uid') for m in msgs)
    e.ingest(msgs)
    assert e._store._conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'native_ingest_identity_v1:%'").fetchone()[0]==2
    assert len(rows(e))==2
    e.shutdown()
    native.close()

def test_generated_provenance_requires_actual_object_and_is_pruned(tmp_path):
    e=open_engine(tmp_path/'provenance.db')
    generated={'role':'user','content':'same placeholder text'}
    literal=dict(generated)
    # Simulate an ID collision deterministically; an ID key alone is no proof.
    e._generated_ignored_active_replay_placeholder_messages={id(literal):generated}
    copied=e._copy_active_replay_messages_preserving_generated_ids([literal])[0]
    assert id(copied) not in e._generated_ignored_active_replay_placeholder_messages
    e._generated_ignored_active_replay_placeholder_messages={id(generated):generated}
    copied=e._copy_active_replay_messages_preserving_generated_ids([generated])[0]
    assert e._generated_ignored_active_replay_placeholder_messages[id(copied)] is copied
    e._remember_active_replay_messages([literal],[literal])
    assert not e._generated_ignored_active_replay_placeholder_messages
    e.shutdown()
