"""Distribution contract; run with stdlib, independent of source test conftest."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class DistributionTests(unittest.TestCase):
    def test_delivered_runtime_has_complete_byte_identical_allowlist(self):
        package = ROOT / 'packages/hermes-lcm'
        self.assertTrue((package / 'plugin.yaml').is_file(), 'installable runtime package absent')
        manifest = json.loads((ROOT / 'packaging/runtime-inputs.json').read_text())
        actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
        self.assertEqual(actual, set(manifest['files']) | {'plugin.yaml', 'runtime-build.json'})
        for rel, digest in manifest['files'].items():
            p = package / rel
            self.assertFalse(p.is_symlink())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
            self.assertEqual(p.read_bytes(), (ROOT / rel).read_bytes())
        self.assertFalse(any((package / x).exists() for x in ('tests', 'benchmarking', 'benchmarks', 'scripts')))

if __name__ == '__main__':
    unittest.main()
