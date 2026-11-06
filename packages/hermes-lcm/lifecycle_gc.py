"""Asynchronous, rate-limited lifecycle garbage collection."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
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
    protected_session_ids_provider: Callable[[], Iterable[str]] | None = None
    protection_revision: int = 0


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
        self._lock = threading.RLock()
        self._states: dict[str, _GCState] = {}

    @staticmethod
    def _key(db_path: str | Path) -> str:
        return str(Path(db_path).expanduser().resolve())

    def protect(self, db_path: str | Path, session_ids: Iterable[str]) -> None:
        """Protect sessions before their bind can race an in-flight GC pass."""
        key = self._key(db_path)
        protected = {str(item) for item in session_ids if item}
        if not protected:
            return
        with self._lock:
            state = self._states.setdefault(key, _GCState())
            state.protected_session_ids.update(protected)
            state.protection_revision += 1

    def request(
        self,
        db_path: str | Path,
        *,
        threshold: int,
        max_age_hours: float | None,
        protected_session_ids: Iterable[str] = (),
        protected_session_ids_provider: Callable[[], Iterable[str]] | None = None,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> bool:
        """Schedule one best-effort GC pass and return without waiting for it."""
        key = self._key(db_path)
        now = self._clock()
        protected = {str(item) for item in protected_session_ids if item}
        with self._lock:
            state = self._states.setdefault(key, _GCState())
            if protected:
                state.protected_session_ids.update(protected)
                state.protection_revision += 1
            if protected_session_ids_provider is not None:
                state.protected_session_ids_provider = protected_session_ids_provider
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
                state.next_allowed_at = 0.0
                logger.warning(
                    "LCM could not start asynchronous empty-lifecycle GC",
                    exc_info=True,
                )
                return False
        return True

    def _protected_snapshot(self, key: str) -> set[str]:
        with self._lock:
            state = self._states.get(key)
            protected = set(state.protected_session_ids) if state is not None else set()
            provider = state.protected_session_ids_provider if state is not None else None
        if provider is not None:
            protected.update(str(item) for item in provider() if item)
        return protected

    def _protection_revision(self, key: str) -> int:
        with self._lock:
            state = self._states.get(key)
            return state.protection_revision if state is not None else 0

    @contextmanager
    def _protection_commit_guard(self, key: str):
        """Linearize GC commit against protect-before-bind registration."""
        with self._lock:
            yield

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
            # Gate on ELIGIBLE empty candidates, not total lifecycle rows:
            # a table dominated by healthy data-bearing sessions must not
            # schedule pointless prune passes. Discovery is bounded
            # (threshold + 1 rows) and runs on the store's read-only path.
            eligible = store.empty_session_candidate_count(
                protected_session_ids=self._protected_snapshot(key),
                max_age_hours=max_age_hours,
                limit=threshold + 1,
            )
            if eligible <= threshold:
                return
            deleted = store.prune_empty_sessions(
                protected_session_ids=self._protected_snapshot(key),
                protected_session_ids_provider=lambda: self._protected_snapshot(key),
                protection_revision_provider=lambda: self._protection_revision(key),
                protection_commit_guard=lambda: self._protection_commit_guard(key),
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


def protect_empty_lifecycle_sessions(
    db_path: str | Path,
    session_ids: Iterable[str],
) -> None:
    EMPTY_LIFECYCLE_GC_COORDINATOR.protect(db_path, session_ids)


def request_empty_lifecycle_gc(
    db_path: str | Path,
    **kwargs,
) -> bool:
    return EMPTY_LIFECYCLE_GC_COORDINATOR.request(db_path, **kwargs)
