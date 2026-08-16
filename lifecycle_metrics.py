"""Privacy-safe, process-local lifecycle counters for hermes-lcm.

Only aggregate integers are exposed. Weak registries ensure that observing an
engine or SQLite helper cannot extend its lifetime.
"""

from __future__ import annotations

from collections import Counter
import threading
import weakref
from typing import Any

__all__ = [
    "register_engine_created",
    "begin_engine_shutdown",
    "complete_engine_shutdown",
    "fail_engine_shutdown",
    "register_message_store_created",
    "begin_message_store_close",
    "complete_message_store_close",
    "fail_message_store_close",
    "register_summary_dag_created",
    "begin_summary_dag_close",
    "complete_summary_dag_close",
    "fail_summary_dag_close",
    "register_lifecycle_state_store_created",
    "begin_lifecycle_state_store_close",
    "complete_lifecycle_state_store_close",
    "fail_lifecycle_state_store_close",
    "record_storage_bind",
    "record_session_end_drain_started",
    "record_session_end_drain_exhausted",
    "configure",
    "configure_from_host_policy",
    "snapshot",
]

_LOCK = threading.RLock()
_ENABLED = False
_TOTALS: Counter[str] = Counter()
_ENGINE_OBJECTS: weakref.WeakSet[Any] = weakref.WeakSet()
_ENGINES_OPEN: weakref.WeakSet[Any] = weakref.WeakSet()
_HELPER_OBJECTS: dict[str, weakref.WeakSet[Any]] = {
    "message_store": weakref.WeakSet(),
    "summary_dag": weakref.WeakSet(),
    "lifecycle_state_store": weakref.WeakSet(),
}
_HELPERS_OPEN: dict[str, weakref.WeakSet[Any]] = {
    name: weakref.WeakSet() for name in _HELPER_OBJECTS
}


def _enabled() -> bool:
    return _ENABLED


def configure(*, enabled: bool) -> None:
    """Set the process-local collection gate from an existing host policy."""
    global _ENABLED
    with _LOCK:
        _ENABLED = bool(enabled)


def configure_from_host_policy() -> None:
    """Reuse Hermes' existing local shared-metrics opt-in; fail closed."""
    try:
        from hermes_cli.config import read_raw_config_readonly

        config = read_raw_config_readonly() or {}
    except Exception:
        enabled = False
    else:
        telemetry = config.get("telemetry") if isinstance(config, dict) else None
        shared_metrics = (
            telemetry.get("shared_metrics") if isinstance(telemetry, dict) else None
        )
        enabled = (
            isinstance(shared_metrics, dict)
            and shared_metrics.get("enabled") is True
        )
    configure(enabled=enabled)


def register_engine_created(engine: Any) -> None:
    """Register one successfully constructed engine without retaining it."""
    if not _enabled():
        return
    with _LOCK:
        _ENGINE_OBJECTS.add(engine)
        _ENGINES_OPEN.add(engine)
        _TOTALS["engines_created_total"] += 1


def begin_engine_shutdown(engine: Any) -> bool:
    """Record a shutdown call and return whether it starts an open transition."""
    if not _enabled():
        return False
    with _LOCK:
        _TOTALS["engine_shutdown_calls_total"] += 1
        if engine not in _ENGINES_OPEN:
            _TOTALS["engine_shutdown_idempotent_total"] += 1
            return False
        return True


def complete_engine_shutdown(engine: Any) -> None:
    if not _enabled():
        return
    with _LOCK:
        if engine in _ENGINES_OPEN:
            _ENGINES_OPEN.discard(engine)
            _TOTALS["engines_shutdown_completed_total"] += 1


def fail_engine_shutdown(engine: Any) -> None:
    if not _enabled():
        return
    with _LOCK:
        if engine in _ENGINES_OPEN:
            _TOTALS["engine_shutdown_failures_total"] += 1


def _register_helper(kind: str, helper: Any) -> None:
    if not _enabled():
        return
    with _LOCK:
        _HELPER_OBJECTS[kind].add(helper)
        _HELPERS_OPEN[kind].add(helper)
        _TOTALS[f"{kind}s_created_total"] += 1


def _begin_helper_close(kind: str, helper: Any) -> bool:
    if not _enabled():
        return False
    with _LOCK:
        _TOTALS[f"{kind}_close_calls_total"] += 1
        if helper not in _HELPERS_OPEN[kind]:
            _TOTALS[f"{kind}_close_idempotent_total"] += 1
            return False
        return True


def _complete_helper_close(kind: str, helper: Any) -> None:
    if not _enabled():
        return
    with _LOCK:
        if helper in _HELPERS_OPEN[kind]:
            _HELPERS_OPEN[kind].discard(helper)
            _TOTALS[f"{kind}s_closed_total"] += 1


def _fail_helper_close(kind: str, helper: Any) -> None:
    if not _enabled():
        return
    with _LOCK:
        if helper in _HELPERS_OPEN[kind]:
            _TOTALS[f"{kind}_close_failures_total"] += 1


def register_message_store_created(store: Any) -> None:
    _register_helper("message_store", store)


def begin_message_store_close(store: Any) -> bool:
    return _begin_helper_close("message_store", store)


def complete_message_store_close(store: Any) -> None:
    _complete_helper_close("message_store", store)


def fail_message_store_close(store: Any) -> None:
    _fail_helper_close("message_store", store)


def register_summary_dag_created(dag: Any) -> None:
    _register_helper("summary_dag", dag)


def begin_summary_dag_close(dag: Any) -> bool:
    return _begin_helper_close("summary_dag", dag)


def complete_summary_dag_close(dag: Any) -> None:
    _complete_helper_close("summary_dag", dag)


def fail_summary_dag_close(dag: Any) -> None:
    _fail_helper_close("summary_dag", dag)


def register_lifecycle_state_store_created(store: Any) -> None:
    _register_helper("lifecycle_state_store", store)


def begin_lifecycle_state_store_close(store: Any) -> bool:
    return _begin_helper_close("lifecycle_state_store", store)


def complete_lifecycle_state_store_close(store: Any) -> None:
    _complete_helper_close("lifecycle_state_store", store)


def fail_lifecycle_state_store_close(store: Any) -> None:
    _fail_helper_close("lifecycle_state_store", store)


def record_storage_bind() -> None:
    if not _enabled():
        return
    with _LOCK:
        _TOTALS["storage_binds_total"] += 1


def record_session_end_drain_started() -> None:
    if not _enabled():
        return
    with _LOCK:
        _TOTALS["session_end_drains_started_total"] += 1


def record_session_end_drain_exhausted() -> None:
    if not _enabled():
        return
    with _LOCK:
        _TOTALS["session_end_drains_exhausted_total"] += 1


def _registered_unique_engine_count() -> int:
    """Count unique currently bound engines without exposing registry keys."""
    if not _enabled():
        return 0
    try:
        from .engine_registry import (
            _ACTIVE_ENGINE_REGISTRY_LOCK,
            _ACTIVE_ENGINES_BY_CONVERSATION_ID,
            _ACTIVE_ENGINES_BY_SESSION_ID,
        )

        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            engines = list(_ACTIVE_ENGINES_BY_SESSION_ID.values())
            engines.extend(_ACTIVE_ENGINES_BY_CONVERSATION_ID.values())
        return len({id(engine) for engine in engines})
    except Exception:
        return 0


def snapshot() -> dict[str, dict[str, int]]:
    """Return a fresh aggregate-only snapshot suitable for ``lcm_status``."""
    is_enabled = _enabled()
    with _LOCK:
        values = {
            "enabled": int(is_enabled),
            "engine_objects_live": len(_ENGINE_OBJECTS) if is_enabled else 0,
            "engines_open": len(_ENGINES_OPEN) if is_enabled else 0,
            "message_store_objects_live": len(_HELPER_OBJECTS["message_store"]) if is_enabled else 0,
            "message_stores_open": len(_HELPERS_OPEN["message_store"]) if is_enabled else 0,
            "summary_dag_objects_live": len(_HELPER_OBJECTS["summary_dag"]) if is_enabled else 0,
            "summary_dags_open": len(_HELPERS_OPEN["summary_dag"]) if is_enabled else 0,
            "lifecycle_state_store_objects_live": (len(
                _HELPER_OBJECTS["lifecycle_state_store"]
            ) if is_enabled else 0),
            "lifecycle_state_stores_open": (len(
                _HELPERS_OPEN["lifecycle_state_store"]
            ) if is_enabled else 0),
        }
        for key in (
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
            "session_end_drains_started_total",
            "session_end_drains_exhausted_total",
        ):
            values[key] = int(_TOTALS[key]) if is_enabled else 0

    values["engines_registered_unique"] = _registered_unique_engine_count()
    return {"runtime_lifecycle": values}
