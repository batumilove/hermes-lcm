from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_module(name: str):
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dashboard_surface_does_not_construct_or_register_engine(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    module = load_module("hermes_lcm_passive_surface_test")

    class DashboardContext:
        runtime_role = "dashboard"

        def register_context_engine(self, engine):
            raise AssertionError("passive dashboard must not register LCM")

    module.register(DashboardContext())

    assert not (tmp_path / ".hermes" / "lcm.db").exists()
    assert "passive surface" in caplog.text.lower()


def test_unknown_modern_surface_fails_closed_without_constructing_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    module = load_module("hermes_lcm_unknown_surface_test")

    class UnknownContext:
        runtime_role = "unknown"

        def register_context_engine(self, engine):
            raise AssertionError("unknown modern surface must not register LCM")

    module.register(UnknownContext())

    assert not (tmp_path / ".hermes" / "lcm.db").exists()
