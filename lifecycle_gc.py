"""Asynchronous, rate-limited lifecycle garbage collection."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .lifecycle_state import LifecycleStateStore

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 300.0
_DEFAULT_BATCH_SIZE = 100


@dataclass
class _GCState:
    running: bool = False
    next_allowed_at: float = 0.0
    thread: threading.Thread | None = None
    protected_session_ids: set[str] = field(default_factory=set)


class EmptyLifecycleGCCoordinator:
    """Coordinate at most one bounded GC worker per database path."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        store_factory: Callable[[str | Path], LifecycleStateStore] = LifecycleStateStore,
    ) -> None:
        self._clock = clock
        self._store_factory = store_factory
        self._lock = threading.Lock()
        self._states: dict[str, _GCState] = {}

    @staticmethod
    def _key(db_path: str | Path) -> str:
        return str(Path(db_path).expanduser().resolve())

    def request(
        self,
        db_path: str | Path,
        *,
        threshold: int,
        max_age_hours: float | None,
        protected_session_ids: Iterable[str] = (),
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> bool:
        """Schedule one best-effort GC pass and return without waiting for it."""
        key = self._key(db_path)
        now = self._clock()
        protected = {str(item) for item in protected_session_ids if item}
        with self._lock:
            state = self._states.setdefault(key, _GCState())
            state.protected_session_ids.update(protected)
            if state.running or now < state.next_allowed_at:
                return False
            state.running = True
            state.next_allowed_at = now + max(0.0, float(interval_seconds))
            thread = threading.Thread(
                target=self._run,
                args=(key, int(threshold), max_age_hours, max(1, int(batch_size))),
                name="lcm-empty-lifecycle-gc",
                daemon=True,
            )
            state.thread = thread
            try:
                thread.start()
            except BaseException:
                state.running = False
                state.thread = None
                logger.warning(
                    "LCM could not start asynchronous empty-lifecycle GC",
                    exc_info=True,
                )
                return False
        return True

    def _protected_snapshot(self, key: str) -> set[str]:
        with self._lock:
            state = self._states.get(key)
            return set(state.protected_session_ids) if state is not None else set()

    def _run(
        self,
        key: str,
        threshold: int,
        max_age_hours: float | None,
        batch_size: int,
    ) -> None:
        store: LifecycleStateStore | None = None
        try:
            store = self._store_factory(key)
            if store.row_count() <= threshold:
                return
            deleted = store.prune_empty_sessions(
                protected_session_ids=self._protected_snapshot(key),
                max_age_hours=max_age_hours,
                max_candidates=batch_size,
            )
            if deleted:
                logger.info(
                    "LCM asynchronously pruned %d empty lifecycle rows "
                    "(threshold=%d batch_size=%d)",
                    deleted,
                    threshold,
                    batch_size,
                )
        except Exception:
            logger.warning("LCM asynchronous empty-lifecycle GC failed", exc_info=True)
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:
                    logger.debug("LCM failed closing lifecycle GC store", exc_info=True)
            with self._lock:
                state = self._states.get(key)
                if state is not None:
                    state.running = False
                    state.thread = None
                    state.protected_session_ids.clear()


EMPTY_LIFECYCLE_GC_COORDINATOR = EmptyLifecycleGCCoordinator()


def request_empty_lifecycle_gc(
    db_path: str | Path,
    **kwargs,
) -> bool:
    return EMPTY_LIFECYCLE_GC_COORDINATOR.request(db_path, **kwargs)
