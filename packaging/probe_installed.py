"""Fresh installed-tree behavioral probe; no development/source-parent imports."""
import importlib.util
import json
from pathlib import Path
import sys

import os
os.environ['LCM_DATABASE_PATH'] = '/opt/fixture/home/.hermes/lcm.db'
Path(os.environ['HERMES_HOME']).mkdir(parents=True, exist_ok=True)
p = Path('/opt/delivered')
spec = importlib.util.spec_from_file_location('hermes_lcm', p/'__init__.py', submodule_search_locations=[str(p)])
module = importlib.util.module_from_spec(spec)
sys.modules['hermes_lcm'] = module
spec.loader.exec_module(module)
class Context:
    runtime_role = 'cli'
    context_engine_tool_handlers_receive_messages = True
    def __init__(self): self.tools = {}; self.hooks = {}; self.commands = {}
    def register_context_engine(self, engine): self.engine = engine
    def register_tool(self, **kw): self.tools[kw['name']] = kw
    def register_hook(self, *args, **kw): self.hooks[str(args)] = kw
    def register_command(self, *args, **kw): self.commands[str(args)] = kw
ctx = Context()
module.register(ctx)
try:
    assert ctx.engine is not None
    ctx.engine.on_session_start('distribution-probe')
    result = json.loads(ctx.engine.handle_tool_call('lcm_status', {}))
    assert 'error' not in result, result
    messages = [{'role':'user','content':'distribution canary'}, {'role':'assistant','content':'canary acknowledged'}]
    ctx.engine.on_session_end('distribution-probe', messages)
    for source in p.glob('*.py'):
        __import__('hermes_lcm.' + source.stem) if source.stem != '__init__' else None
    from hermes_lcm.tokens import count_tokens
    assert count_tokens('canary') > 0
    assert len(ctx.tools) >= 8, list(ctx.tools)
    paths = {name: m.__file__ for name,m in sys.modules.items() if name.startswith('hermes_lcm') and getattr(m,'__file__',None)}
    assert all(Path(x).is_relative_to(p) for x in paths.values()), paths
    print(json.dumps({'registration':True,'tools':sorted(ctx.tools),'checkpoint':True,'status':result,'modules':paths},default=str))
finally:
    ctx.engine.shutdown()
