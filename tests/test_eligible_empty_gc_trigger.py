"""Eligible-empty GC trigger contracts.

The async GC worker must wake only when ELIGIBLE empty lifecycle rows exceed
the threshold — not when total lifecycle rows (including healthy data-bearing
sessions) exceed it. Otherwise every bind schedules a pointless bounded
background pass that deletes nothing.
"""
from __future__ import annotations

import time

from hermes_lcm.lifecycle_gc import EmptyLifecycleGCCoordinator
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def _populate(db_path, *, empty: int, live: int):
    state = LifecycleStateStore(db_path)
    messages = MessageStore(db_path)
    for index in range(empty):
        state.bind_session(f"orphan-{index}")
    for index in range(live):
        session_id = f"live-{index}"
        messages.append(
            session_id, {"role": "user", "content": f"keep {index}"}, source="cli"
        )
        state.bind_session(session_id)
    return state, messages


class TestEligibleCandidateCount:
    def test_count_only_counts_empty_eligible_rows(self, tmp_path):
        db = tmp_path / "count.db"
        state, messages = _populate(db, empty=2, live=50)
        try:
            assert state.row_count() == 52
            assert state.empty_session_candidate_count(max_age_hours=None) == 2
        finally:
            state.close(); messages.close()

    def test_count_respects_limit_bound(self, tmp_path):
        db = tmp_path / "limit.db"
        state, messages = _populate(db, empty=30, live=0)
        try:
            assert state.empty_session_candidate_count(
                max_age_hours=None, limit=6
            ) == 6
        finally:
            state.close(); messages.close()

    def test_count_respects_max_age(self, tmp_path):
        db = tmp_path / "age.db"
        state, messages = _populate(db, empty=3, live=0)
        try:
            assert state.empty_session_candidate_count(max_age_hours=1.0) == 0
        finally:
            state.close(); messages.close()

    def test_count_excludes_protected_sessions(self, tmp_path):
        db = tmp_path / "prot.db"
        state, messages = _populate(db, empty=3, live=0)
        try:
            assert state.empty_session_candidate_count(
                protected_session_ids={"orphan-0", "orphan-1"},
                max_age_hours=None,
            ) == 1
        finally:
            state.close(); messages.close()


class TestCoordinatorEligibleTrigger:
    def test_gc_worker_skips_prune_when_no_eligible_candidates(self, tmp_path, monkeypatch):
        """Many data-bearing rows over threshold must not trigger a prune pass."""
        db = tmp_path / "trigger.db"
        state, messages = _populate(db, empty=0, live=60)
        try:
            prune_calls = []

            def fake_prune(**kwargs):
                prune_calls.append(kwargs)
                return 0

            monkeypatch.setattr(state, "prune_empty_sessions", fake_prune)

            coordinator = EmptyLifecycleGCCoordinator(store_factory=lambda path: state)
            started = coordinator.request(
                str(state.db_path), threshold=10, max_age_hours=None
            )
            assert started, "worker should start (rate-limit gate is separate)"
            deadline = time.monotonic() + 5.0
            while (
                coordinator._states[coordinator._key(state.db_path)].running
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert prune_calls == [], (
                "prune must not run when eligible candidates (0) <= threshold (10) "
                "even though total rows (60) exceed it"
            )
        finally:
            state.close(); messages.close()
