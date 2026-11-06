import json,logging,os
from pathlib import Path
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.engine as engine_module

def records(caplog):
 return [json.loads(r.getMessage().split('LCM_RUNTIME_IDENTITY ',1)[1]) for r in caplog.records if r.getMessage().startswith('LCM_RUNTIME_IDENTITY ')]

def test_storage_identity_is_process_bound_and_excludes_conversation_content(tmp_path,caplog,monkeypatch):
 monkeypatch.setenv('INVOCATION_ID','test-invocation');caplog.set_level(logging.INFO)
 db=tmp_path/'lcm.db';e=LCMEngine(config=LCMConfig(database_path=str(db),embeddings_enabled=False,empty_lifecycle_gc_enabled=False),hermes_home=str(tmp_path))
 try:
  rows=records(caplog);assert len(rows)==1
  assert rows[0]=={'pid':os.getpid(),'invocation_id':'test-invocation','module_path':str(Path(engine_module.__file__).resolve()),'database_path':str(db.resolve()),'hermes_home':str(tmp_path.resolve()),'embeddings_enabled':False}
  e.on_session_start('identity-fixture',platform='test',context_length=200000)
  e.ingest([{'role':'user','content':'DO_NOT_LOG_THIS_MESSAGE'}])
  assert len(records(caplog))==1
  assert 'DO_NOT_LOG_THIS_MESSAGE' not in json.dumps(records(caplog))
 finally:e.shutdown()

def test_identity_follows_rebound_database(tmp_path,caplog):
 caplog.set_level(logging.INFO)
 first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
 e=LCMEngine(config=LCMConfig(embeddings_enabled=False,empty_lifecycle_gc_enabled=False),hermes_home=str(first))
 try:
  assert e._rebind_storage_for_home(str(second))
  assert records(caplog)[-1]['database_path']==str(second/'lcm.db')
  assert records(caplog)[-1]['hermes_home']==str(second)
 finally:e.shutdown()
