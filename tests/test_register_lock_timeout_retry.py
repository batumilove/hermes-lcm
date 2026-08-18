"""RED: register() must retry engine construction on SQLite lock timeout.

Incident 2026-08-17 20:14:38Z: the LCM plugin's own startup background FTS
integrity scan (lcm-fts-integrity-messages_fts, _run_background_integrity_scan)
held the process-wide SQLite writer for the full 30s timeout while register()
was still constructing LCMEngine. The constructor's writer acquisition timed
out, register() propagated, and the host aborted plugin load with no retry.
The gateway then ran ~12h with 'Context engine lcm not found' fallback.

Contract: register() retries engine construction a bounded number of times
when construction fails with a database-locked/timeout error, and does NOT
retry on unrelated errors.
"""

import importlib.util
import sys
import types
from pathlib import Path

import hermes_lcm.engine as engine_mod  # noqa: F401  (conftest path shim)

# conftest registers submodules but never executes the plugin __init__. Exec
# it as the canonical "hermes_lcm" package (same name conftest registered) so
# register()'s lazy `from .engine import LCMEngine` resolves to the exact
# module object tests monkeypatch.
_plugin_dir = Path(engine_mod.__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "hermes_lcm", str(_plugin_dir / "__init__.py"),
    submodule_search_locations=[str(_plugin_dir)],
)
lcm_plugin = importlib.util.module_from_spec(_spec)
lcm_plugin.__package__ = "hermes_lcm"
sys.modules["hermes_lcm"] = lcm_plugin
_spec.loader.exec_module(lcm_plugin)


class _FakeCtx:
    """Minimal plugin ctx satisfying register()."""

    runtime_role = "gateway"

    def __init__(self):
        self.registered_engine = None
        self.registered_tools = []

    def register_context_engine(self, engine):
        self.registered_engine = engine

    # Force Path B fallback: no message-aware tool handlers on this host.
    context_engine_tool_handlers_receive_messages = False


class _LockTimeoutError(RuntimeError):
    """Matches the incident error class/text."""

    def __init__(self):
        super().__init__(
            "database is locked: timed out waiting for process-wide SQLite writer; "
            "owner_thread_name=lcm-fts-integrity-messages_fts "
            "owner_operation=_run_background_integrity_scan owner_age_s=30.001"
        )


def test_register_retries_engine_construction_on_lock_timeout(monkeypatch):
    attempts = {"n": 0}

    def flaky_engine_factory(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise _LockTimeoutError()
        return types.SimpleNamespace(shutdown=lambda: None)

    # register() imports LCMEngine lazily from .engine; patch at source module.
    monkeypatch.setattr(engine_mod, "LCMEngine", flaky_engine_factory)

    ctx = _FakeCtx()
    lcm_plugin.register(ctx)
    assert attempts["n"] == 3
    assert ctx.registered_engine is not None


def test_register_does_not_retry_unrelated_errors(monkeypatch):
    attempts = {"n": 0}

    def bad_engine_factory(*args, **kwargs):
        attempts["n"] += 1
        raise ValueError("unrelated config error")

    monkeypatch.setattr(engine_mod, "LCMEngine", bad_engine_factory)

    ctx = _FakeCtx()
    try:
        lcm_plugin.register(ctx)
    except ValueError:
        pass
    assert attempts["n"] == 1, "unrelated errors must not be retried"
