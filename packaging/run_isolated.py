"""Explicit-supplier networkless delivered-tree probe (Linux bubblewrap required)."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('host-source', 'python-runtime', 'dependencies', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sudo', action='store_true')
    a = p.parse_args()
    host, runtime, deps = [x.resolve(strict=True) for x in (a.host_source, a.python_runtime, a.dependencies)]
    if not (runtime/'bin/python3').resolve(strict=True).is_relative_to(runtime):
        p.error('Python executable must resolve inside supplied self-contained runtime')
    if not list((runtime/'lib').glob('python*/encodings/__init__.py')):
        p.error('Runtime must include its own stdlib')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'home/.hermes').mkdir(parents=True)
    (out/'home').chmod(0o700)
    (out/'tmp').mkdir()
    if a.sudo:
        subprocess.run(['sudo', '-n', 'chown', '-R', '0:0', str(out/'home'), str(out/'tmp')], check=True)
    # Privileged namespace setup retains actual UID 0 for private supplier trees.
    # A user namespace cannot traverse owner-only inputs after the UID mapping.
    namespace = [] if a.sudo else ['--unshare-user']
    fixture_uid, fixture_gid = (0, 0) if a.sudo else (os.getuid(), os.getgid())
    base = (['sudo','-n'] if a.sudo else []) + ['bwrap'] + namespace + ['--unshare-net','--unshare-pid','--die-with-parent','--ro-bind','/','/','--tmpfs','/home','--tmpfs','/opt']
    for source, target in ((runtime,'/opt/python'),(deps,'/opt/dependencies'),(host,'/opt/host'),(ROOT/'packages/hermes-lcm','/opt/delivered'),(ROOT/'packaging','/opt/probes')):
        base += ['--ro-bind',str(source),target]
    base += ['--bind',str(out),'/opt/fixture','--proc','/proc','--dev','/dev','--tmpfs','/run','--chdir','/opt/fixture','--cap-drop','ALL','--uid',str(fixture_uid),'--gid',str(fixture_gid),'--clearenv']
    env = {'PATH':'/opt/python/bin:/usr/bin:/bin','HOME':'/opt/fixture/home','HERMES_HOME':'/opt/fixture/home/.hermes','HERMES_RUNTIME_DIR':'/opt/fixture/runtime','TMPDIR':'/opt/fixture/tmp','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/opt/host:/opt/dependencies','UV_OFFLINE':'1','PIP_NO_INDEX':'1','HERMES_DISABLE_LAZY_INSTALLS':'1','LCM_EMBEDDING_BACKEND':'openai','LCM_EMBEDDINGS_ENABLED':'false','HF_HUB_OFFLINE':'1','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
    for k,v in env.items():
        base += ['--setenv',k,v]
    checks = {'scanner': """import json,dataclasses
from pathlib import Path
from tools.plugin_guard import scan_plugin,should_allow_plugin_install
r=scan_plugin(Path('/opt/delivered'),source='file:///opt/delivered')
print(json.dumps({'verdict':r.verdict,'policy':should_allow_plugin_install(r),'findings':[dataclasses.asdict(f) for f in r.findings]},default=str))
assert r.verdict == 'safe', r.verdict
""", 'installed': "exec(compile(open('/opt/probes/probe_installed.py').read(),'/opt/probes/probe_installed.py','exec'))"}
    for label, code in checks.items():
        cmd = base + ['/opt/python/bin/python3','-B','-c',code]
        r = subprocess.run(cmd,capture_output=True,text=True,timeout=100)
        (out/(label+'.stdout')).write_text(r.stdout)
        (out/(label+'.stderr')).write_text(r.stderr)
        (out/(label+'.command.json')).write_text(json.dumps({'argv':cmd,'rc':r.returncode},indent=2))
        print(label,r.returncode,flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
