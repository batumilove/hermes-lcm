"""Deterministic allowlisted runtime distribution. No source rewriting or imports."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def build(destination):
    declaration = json.loads((ROOT / 'packaging/runtime-inputs.json').read_text())
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    payload = {}
    for rel, expected in declaration['files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError(rel)
        source = ROOT / rel
        if source.is_symlink() or not source.is_file():
            raise ValueError(rel)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source hash mismatch: ' + rel)
        payload[rel] = data
    payload['plugin.yaml'] = (ROOT / 'packaging/plugin.yaml').read_bytes()
    receipt = {'schema': 1, 'source_commit': declaration['source_commit'],
               'files': {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(payload.items())}}
    payload['runtime-build.json'] = (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode()
    destination.mkdir(parents=True)
    for rel, data in sorted(payload.items()):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o644)

if __name__ == '__main__':
    build(sys.argv[1])
