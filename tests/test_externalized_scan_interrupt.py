"""Cooperative cancellation preserves the owning host's interrupt signal."""
import sys
from types import SimpleNamespace

import pytest
from hermes_lcm import externalize


class ScanInterrupted(Exception):
    pass


def test_checkpoint_propagates_owning_host_cancellation(monkeypatch):
    host = SimpleNamespace(is_interrupted=lambda: True, TurnInterrupted=ScanInterrupted)
    monkeypatch.setitem(sys.modules, "tools.interrupt", host)
    with pytest.raises(ScanInterrupted, match="scan interrupted"):
        externalize._checkpoint_externalized_scan()
    assert host.is_interrupted()


def test_checkpoint_does_not_bootstrap_standalone_host(monkeypatch):
    monkeypatch.delitem(sys.modules, "tools.interrupt", raising=False)
    externalize._checkpoint_externalized_scan()
    assert "tools.interrupt" not in sys.modules
