"""Portable runner CLI does not depend on retained operational receipts."""
import subprocess
import sys
from pathlib import Path


def test_explicit_supplier_cli():
    runner = Path(__file__).resolve().parents[1] / 'packaging/run_isolated.py'
    result = subprocess.run([sys.executable, str(runner), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for option in ('--host-source', '--python-runtime', '--dependencies', '--output'):
        assert option in result.stdout
