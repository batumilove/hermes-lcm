"""TDD tests for process-local LCM lifecycle metrics in ``lcm_status``.

These tests assert behavioural deltas and invariants — never frozen global
counts.  They verify:

* A thread-safe weak-ref registry tracks live engine/helper objects without
  retaining them.
* Cumulative created/closed/shutdown/storage-bind counters increment correctly.
* Helper ``close()`` idempotency does not double-count.
* Engine initialization failure does not leave false-open counts.
* ``lcm_status`` exposes the metrics under a stable ``runtime_lifecycle`` key
  with privacy-safe values (no DB paths, session/conversation ids, content,
  or object ids).
"""

from __future__ import annotations

import gc
import json
import sys
import threading
from pathlib import Path

import pytest

# The conftest already registers the hermes_lcm package.
import hermes_lcm.lifecycle_metrics as lcm_metrics
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tools import lcm_status


@pytest.fixture(autouse=True)
def _enable_lifecycle_diagnostics():
    lcm_metrics.configure(enabled=True)
    yield
    lcm_metrics.configure(enabled=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path, tag: str = "eng") -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / f"{tag}.db"))
    return LCMEngine(config=config)


def _snapshot() -> dict:
    """Return a deep-ish copy of the lifecycle snapshot for delta tests."""
    snap = lcm_metrics.snapshot()
    return json.loads(json.dumps(snap))


# ---------------------------------------------------------------------------
# Registry existence & shape
# ---------------------------------------------------------------------------

class TestRegistryShape:
    def test_disabled_by_default_is_noop_without_changing_close_semantics(
        self, tmp_path
    ):
        from hermes_lcm.store import MessageStore

        lcm_metrics.configure(enabled=False)
        store = MessageStore(tmp_path / "disabled.db")
        store.close()
        runtime = _snapshot()["runtime_lifecycle"]
        assert runtime["enabled"] == 0
        assert all(value == 0 for value in runtime.values())

    def test_disabled_engine_shutdown_closes_all_sqlite_helpers(self, tmp_path):
        lcm_metrics.configure(enabled=False)
        engine = _make_engine(tmp_path, "disabled-engine")

        engine.shutdown()

        assert engine._store._conn is None
        assert engine._dag._conn is None
        assert engine._lifecycle._conn is None
        runtime = _snapshot()["runtime_lifecycle"]
        assert runtime["enabled"] == 0
        assert all(value == 0 for value in runtime.values())

    def test_configure_from_host_policy_reuses_shared_metrics_opt_in(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config_readonly",
            lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
        )
        lcm_metrics.configure(enabled=False)
        lcm_metrics.configure_from_host_policy()
        assert _snapshot()["runtime_lifecycle"]["enabled"] == 1

        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config_readonly",
            lambda: {"telemetry": {"shared_metrics": {"enabled": False}}},
        )
        lcm_metrics.configure_from_host_policy()
        assert _snapshot()["runtime_lifecycle"]["enabled"] == 0

    def test_snapshot_has_runtime_lifecycle_key(self):
        snap = lcm_metrics.snapshot()
        assert "runtime_lifecycle" in snap

    def test_snapshot_contains_expected_counter_keys(self):
        rl = lcm_metrics.snapshot()["runtime_lifecycle"]
        for key in (
            "engine_objects_live",
            "engines_open",
            "engines_registered_unique",
            "message_store_objects_live",
            "message_stores_open",
            "summary_dag_objects_live",
            "summary_dags_open",
            "lifecycle_state_store_objects_live",
            "lifecycle_state_stores_open",
            "engines_created_total",
            "engine_shutdown_calls_total",
            "engines_shutdown_completed_total",
            "engine_shutdown_idempotent_total",
            "engine_shutdown_failures_total",
            "message_stores_created_total",
            "message_store_close_calls_total",
            "message_stores_closed_total",
            "message_store_close_idempotent_total",
            "message_store_close_failures_total",
            "summary_dags_created_total",
            "summary_dag_close_calls_total",
            "summary_dags_closed_total",
            "summary_dag_close_idempotent_total",
            "summary_dag_close_failures_total",
            "lifecycle_state_stores_created_total",
            "lifecycle_state_store_close_calls_total",
            "lifecycle_state_stores_closed_total",
            "lifecycle_state_store_close_idempotent_total",
            "lifecycle_state_store_close_failures_total",
            "storage_binds_total",
        ):
            assert key in rl, f"missing key {key!r}"

    def test_snapshot_is_a_copy(self):
        snap1 = lcm_metrics.snapshot()
        snap1["runtime_lifecycle"]["engines_created_total"] += 9999
        snap2 = lcm_metrics.snapshot()
        assert snap2["runtime_lifecycle"]["engines_created_total"] != \
            snap1["runtime_lifecycle"]["engines_created_total"]


# ---------------------------------------------------------------------------
# Engine lifecycle counters
# ---------------------------------------------------------------------------

class TestEngineCounters:
    def test_engine_creation_increments_live_open_and_total(self, tmp_path):
        before = _snapshot()["runtime_lifecycle"]
        e = _make_engine(tmp_path, "create")
        try:
            after = _snapshot()["runtime_lifecycle"]
            assert after["engines_created_total"] == before["engines_created_total"] + 1
            assert after["engine_objects_live"] == before["engine_objects_live"] + 1
            assert after["engines_open"] == before["engines_open"] + 1
        finally:
            e.shutdown()

    def test_engine_shutdown_separates_call_completion_and_live_object(self, tmp_path):
        e = _make_engine(tmp_path, "shutdown")
        before = _snapshot()["runtime_lifecycle"]
        e.shutdown()
        after = _snapshot()["runtime_lifecycle"]
        assert after["engine_objects_live"] == before["engine_objects_live"]
        assert after["engines_open"] == before["engines_open"] - 1
        assert after["engine_shutdown_calls_total"] == before["engine_shutdown_calls_total"] + 1
        assert after["engines_shutdown_completed_total"] == before["engines_shutdown_completed_total"] + 1

    def test_engine_shutdown_idempotent_call_is_counted_without_second_completion(self, tmp_path):
        e = _make_engine(tmp_path, "idem")
        e.shutdown()
        after_first = _snapshot()["runtime_lifecycle"]
        e.shutdown()
        after_second = _snapshot()["runtime_lifecycle"]
        assert after_second["engine_shutdown_calls_total"] == after_first["engine_shutdown_calls_total"] + 1
        assert after_second["engine_shutdown_idempotent_total"] == after_first["engine_shutdown_idempotent_total"] + 1
        assert after_second["engines_shutdown_completed_total"] == after_first["engines_shutdown_completed_total"]
        assert after_second["engines_open"] == after_first["engines_open"]


# ---------------------------------------------------------------------------
# Helper counters (MessageStore / SummaryDAG / LifecycleStateStore)
# ---------------------------------------------------------------------------

class TestHelperCounters:
    def test_helper_creation_and_close(self, tmp_path):
        from hermes_lcm.store import MessageStore
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.lifecycle_state import LifecycleStateStore

        before = _snapshot()["runtime_lifecycle"]
        ms = MessageStore(tmp_path / "ms.db")
        dag = SummaryDAG(tmp_path / "dag.db")
        lss = LifecycleStateStore(tmp_path / "lss.db")
        try:
            after_open = _snapshot()["runtime_lifecycle"]
            assert after_open["message_store_objects_live"] == before["message_store_objects_live"] + 1
            assert after_open["message_stores_open"] == before["message_stores_open"] + 1
            assert after_open["summary_dag_objects_live"] == before["summary_dag_objects_live"] + 1
            assert after_open["summary_dags_open"] == before["summary_dags_open"] + 1
            assert after_open["lifecycle_state_store_objects_live"] == before["lifecycle_state_store_objects_live"] + 1
            assert after_open["lifecycle_state_stores_open"] == before["lifecycle_state_stores_open"] + 1
            assert after_open["message_stores_created_total"] == before["message_stores_created_total"] + 1
            assert after_open["summary_dags_created_total"] == before["summary_dags_created_total"] + 1
            assert after_open["lifecycle_state_stores_created_total"] == before["lifecycle_state_stores_created_total"] + 1
        finally:
            ms.close()
            dag.close()
            lss.close()
        after_close = _snapshot()["runtime_lifecycle"]
        assert after_close["message_store_objects_live"] == before["message_store_objects_live"] + 1
        assert after_close["message_stores_open"] == before["message_stores_open"]
        assert after_close["summary_dag_objects_live"] == before["summary_dag_objects_live"] + 1
        assert after_close["summary_dags_open"] == before["summary_dags_open"]
        assert after_close["lifecycle_state_store_objects_live"] == before["lifecycle_state_store_objects_live"] + 1
        assert after_close["lifecycle_state_stores_open"] == before["lifecycle_state_stores_open"]
        assert after_close["message_store_close_calls_total"] == before["message_store_close_calls_total"] + 1
        assert after_close["message_stores_closed_total"] == before["message_stores_closed_total"] + 1
        assert after_close["summary_dag_close_calls_total"] == before["summary_dag_close_calls_total"] + 1
        assert after_close["summary_dags_closed_total"] == before["summary_dags_closed_total"] + 1
        assert after_close["lifecycle_state_store_close_calls_total"] == before["lifecycle_state_store_close_calls_total"] + 1
        assert after_close["lifecycle_state_stores_closed_total"] == before["lifecycle_state_stores_closed_total"] + 1

    def test_helper_close_idempotent_no_double_count(self, tmp_path):
        from hermes_lcm.store import MessageStore
        ms = MessageStore(tmp_path / "ms_idem.db")
        ms.close()
        after_first = _snapshot()["runtime_lifecycle"]
        ms.close()
        ms.close()
        after_multi = _snapshot()["runtime_lifecycle"]
        assert after_multi["message_store_close_calls_total"] == after_first["message_store_close_calls_total"] + 2
        assert after_multi["message_store_close_idempotent_total"] == after_first["message_store_close_idempotent_total"] + 2
        assert after_multi["message_stores_closed_total"] == after_first["message_stores_closed_total"]
        assert after_multi["message_stores_open"] == after_first["message_stores_open"]


# ---------------------------------------------------------------------------
# Weak-ref non-retention
# ---------------------------------------------------------------------------

class TestWeakRefNonRetention:
    def test_engine_gc_reduces_live_count(self, tmp_path):
        gc.collect()
        before = _snapshot()["runtime_lifecycle"]
        e = _make_engine(tmp_path, "gc")
        e.shutdown()
        del e
        gc.collect()
        after = _snapshot()["runtime_lifecycle"]
        assert after["engine_objects_live"] == before["engine_objects_live"]

    def test_helper_gc_reduces_live_count(self, tmp_path):
        from hermes_lcm.store import MessageStore
        gc.collect()
        before = _snapshot()["runtime_lifecycle"]
        ms = MessageStore(tmp_path / "gc.db")
        ms.close()
        del ms
        gc.collect()
        after = _snapshot()["runtime_lifecycle"]
        assert after["message_store_objects_live"] == before["message_store_objects_live"]


# ---------------------------------------------------------------------------
# Storage bind counter
# ---------------------------------------------------------------------------

class TestStorageBindCounter:
    def test_engine_init_binds_storage_once(self, tmp_path):
        before = _snapshot()["runtime_lifecycle"]
        e = _make_engine(tmp_path, "bind")
        try:
            after = _snapshot()["runtime_lifecycle"]
            assert after["storage_binds_total"] == before["storage_binds_total"] + 1
        finally:
            e.shutdown()


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_engine_creation_and_shutdown(self, tmp_path):
        before = _snapshot()["runtime_lifecycle"]
        errors = []

        def worker(idx):
            try:
                e = _make_engine(tmp_path, f"conc_{idx}")
                e.shutdown()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        gc.collect()
        after = _snapshot()["runtime_lifecycle"]
        # All 8 created and shutdown, net open-engine change should be zero.
        assert after["engines_created_total"] == before["engines_created_total"] + 8
        assert after["engine_shutdown_calls_total"] == before["engine_shutdown_calls_total"] + 8
        assert after["engines_shutdown_completed_total"] == before["engines_shutdown_completed_total"] + 8
        assert after["engines_open"] == before["engines_open"]

    def test_snapshot_is_concurrency_safe(self):
        """Call snapshot from many threads simultaneously — no exception."""
        errors = []

        def reader():
            try:
                for _ in range(50):
                    lcm_metrics.snapshot()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestFailureAndRegistryCounters:
    def test_helper_close_failure_is_not_reported_as_closed(self):
        from hermes_lcm.store import MessageStore

        class BrokenConnection:
            def execute(self, _sql):
                return None

            def close(self):
                raise RuntimeError("synthetic close failure")

        store = object.__new__(MessageStore)
        store._conn = BrokenConnection()
        lcm_metrics.register_message_store_created(store)
        before = _snapshot()["runtime_lifecycle"]

        with pytest.raises(RuntimeError, match="synthetic close failure"):
            store.close()

        after = _snapshot()["runtime_lifecycle"]
        assert after["message_store_close_calls_total"] == before["message_store_close_calls_total"] + 1
        assert after["message_store_close_failures_total"] == before["message_store_close_failures_total"] + 1
        assert after["message_stores_closed_total"] == before["message_stores_closed_total"]
        assert after["message_stores_open"] == before["message_stores_open"]
        store._conn = None

    def test_engine_shutdown_failure_is_not_reported_as_completed(self):
        class BrokenHelper:
            def close(self):
                raise RuntimeError("synthetic shutdown failure")

        engine = object.__new__(LCMEngine)
        engine._unregister_active_engine_binding = lambda: None
        engine._store = BrokenHelper()
        engine._dag = BrokenHelper()
        engine._lifecycle = BrokenHelper()
        lcm_metrics.register_engine_created(engine)
        before = _snapshot()["runtime_lifecycle"]

        with pytest.raises(RuntimeError, match="synthetic shutdown failure"):
            engine.shutdown()

        after = _snapshot()["runtime_lifecycle"]
        assert after["engine_shutdown_calls_total"] == before["engine_shutdown_calls_total"] + 1
        assert after["engine_shutdown_failures_total"] == before["engine_shutdown_failures_total"] + 1
        assert after["engines_shutdown_completed_total"] == before["engines_shutdown_completed_total"]
        assert after["engines_open"] == before["engines_open"]

    def test_registered_unique_counts_registry_objects_not_all_open_engines(self, tmp_path):
        before = _snapshot()["runtime_lifecycle"]
        engine = _make_engine(tmp_path, "registry")
        try:
            unbound = _snapshot()["runtime_lifecycle"]
            assert unbound["engines_registered_unique"] == before["engines_registered_unique"]
            engine.on_session_start("registry-session")
            bound = _snapshot()["runtime_lifecycle"]
            assert bound["engines_registered_unique"] == before["engines_registered_unique"] + 1
        finally:
            engine.shutdown()


# ---------------------------------------------------------------------------
# lcm_status integration
# ---------------------------------------------------------------------------

class TestLcmStatusIntegration:
    def test_lcm_status_includes_runtime_lifecycle(self, tmp_path):
        e = _make_engine(tmp_path, "status")
        try:
            e._session_id = "lifecycle-status-test"
            result = json.loads(lcm_status({}, engine=e))
            assert "runtime_lifecycle" in result
            rl = result["runtime_lifecycle"]
            assert "engines_open" in rl
            assert "engines_created_total" in rl
        finally:
            e.shutdown()

    def test_lcm_status_privacy_no_secrets(self, tmp_path):
        """runtime_lifecycle must not leak DB paths, session ids, etc."""
        e = _make_engine(tmp_path, "privacy")
        try:
            e._session_id = "secret-session-id-123"
            result = json.loads(lcm_status({}, engine=e))
            rl = result["runtime_lifecycle"]
            blob = json.dumps(rl)
            # Must not contain the session id, db path, or conversation id.
            assert "secret-session-id-123" not in blob
            assert str(tmp_path) not in blob
            # Should be plain counters — no object repr / id().
            assert "object at 0x" not in blob
        finally:
            e.shutdown()

    def test_lcm_status_reflects_engine_shutdown_delta(self, tmp_path):
        e = _make_engine(tmp_path, "delta")
        e._session_id = "delta-session"
        before = json.loads(lcm_status({}, engine=e))["runtime_lifecycle"]
        e.shutdown()
        # After shutdown, we need a different engine for lcm_status since
        # the original is shut down.  Use a fresh one just to read.
        e2 = _make_engine(tmp_path, "delta_reader")
        e2._session_id = "delta-reader"
        try:
            after = json.loads(lcm_status({}, engine=e2))["runtime_lifecycle"]
            assert after["engines_shutdown_completed_total"] >= before["engines_shutdown_completed_total"] + 1
        finally:
            e2.shutdown()


# ---------------------------------------------------------------------------
# Initialization failure does not leave false-open counts
# ---------------------------------------------------------------------------

class TestInitFailure:
    def test_failed_store_init_no_live_leak(self, tmp_path):
        """If MessageStore.__init__ raises, live count must not be inflated."""
        from hermes_lcm.store import MessageStore
        before = _snapshot()["runtime_lifecycle"]
        with pytest.raises(Exception):
            # An unwritable / invalid path should cause init to fail.
            MessageStore("/dev/null/impossible/path.db")
        after = _snapshot()["runtime_lifecycle"]
        assert after["message_stores_open"] == before["message_stores_open"]
