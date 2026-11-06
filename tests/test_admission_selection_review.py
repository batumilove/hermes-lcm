import copy

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def engine_at(path):
    engine = LCMEngine(config=LCMConfig(database_path=str(path), embeddings_enabled=False,
                        empty_lifecycle_gc_enabled=False, large_output_externalization_enabled=False),
                        hermes_home=str(path.parent))
    engine.on_session_start('reuse-session', platform='test', conversation_id='reuse-conversation', context_length=200000)
    return engine


def exchange(argument, uid):
    return [
        dict(role='assistant', content='', message_uid=uid+'-a', _db_persisted=True,
             tool_calls=[dict(id='reused-id', type='function', function=dict(name='inspect', arguments=argument))]),
        dict(role='tool', content='same successful result', tool_name='inspect', tool_call_id='reused-id',
             message_uid=uid+'-t', _db_persisted=True),
    ]


def test_changed_arguments_same_result_survives_restart(tmp_path):
    path = tmp_path/'reused.db'
    engine = engine_at(path)
    engine.ingest(copy.deepcopy(exchange('{"target":"old"}', 'old')))
    engine.shutdown()
    engine = engine_at(path)
    engine.ingest(copy.deepcopy(exchange('{"target":"new"}', 'new')))
    rows = engine._store.get_session_messages('reuse-session')
    assert len([row for row in rows if row['role']=='assistant']) == 2
    assert len([row for row in rows if row['role']=='tool']) == 2
    engine.shutdown()


def test_hard_filter_uses_deterministic_bounded_call_order(tmp_path, monkeypatch):
    engine = engine_at(tmp_path/'order.db')
    ids = {f'call-{n:02}' for n in range(30)}
    observed = []
    def capture(session_id, call_ids, **kwargs):
        observed.append(list(call_ids)[:kwargs['max_call_ids']])
        return []
    monkeypatch.setattr(engine._store, 'get_bounded_tool_call_rows', capture)
    messages = [(i, dict(role='tool', content='result', tool_name='inspect', tool_call_id=call_id))
                for i, call_id in enumerate(sorted(ids, reverse=True))]
    engine._drop_exact_duplicate_tool_admission(messages)
    assert observed == [sorted(ids)[:8]]
    engine.shutdown()
