"""Tests for AIAgent.wait_for_background_reviews() and its tracking list.

The wait helper exists because background-review threads are spawned as
``daemon=True`` (so they die on process exit) — and non-interactive
callers (`cli.py -q`, mcp_serve one-shots, batch_runner, cron) exit
faster than an LLM round-trip takes, silently dropping every skill /
memory update the review would have made.  These tests pin the wait's
behavior so a future refactor can't quietly revert to "fire and forget."
"""

import threading
import time

import pytest

from run_agent import AIAgent


class _StubAgent:
    """Minimal stand-in for AIAgent so we can call the unbound method.

    wait_for_background_reviews only reads ``self._bg_review_threads``;
    no other agent state is touched.
    """


def _thread_that_runs_for(seconds: float) -> threading.Thread:
    t = threading.Thread(target=time.sleep, args=(seconds,), daemon=True)
    t.start()
    return t


def _thread_that_returns_immediately() -> threading.Thread:
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    t.join()  # ensure terminated before caller observes it
    return t


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


def test_no_tracking_attribute_returns_zero():
    """Agents that never spawned a review have no ``_bg_review_threads``."""
    agent = _StubAgent()
    assert AIAgent.wait_for_background_reviews(agent, timeout_sec=1.0) == 0


def test_empty_list_returns_zero():
    agent = _StubAgent()
    agent._bg_review_threads = []
    assert AIAgent.wait_for_background_reviews(agent, timeout_sec=1.0) == 0


def test_zero_timeout_skips_wait_but_still_prunes():
    """timeout_sec <= 0 is the documented "don't wait" knob — but we
    still want stale dead threads pruned so the list doesn't grow."""
    agent = _StubAgent()
    dead = _thread_that_returns_immediately()
    agent._bg_review_threads = [dead]
    assert AIAgent.wait_for_background_reviews(agent, timeout_sec=0) == 0
    # Dead thread pruned even though we didn't wait
    assert agent._bg_review_threads == []


# ---------------------------------------------------------------------------
# Completion / pruning
# ---------------------------------------------------------------------------


def test_completed_thread_counted_without_blocking():
    agent = _StubAgent()
    agent._bg_review_threads = [_thread_that_returns_immediately()]
    started = time.monotonic()
    assert AIAgent.wait_for_background_reviews(agent, timeout_sec=10.0) == 1
    # Should return instantly — the thread was already dead
    assert time.monotonic() - started < 0.5
    assert agent._bg_review_threads == []


def test_running_thread_joined_within_budget():
    agent = _StubAgent()
    agent._bg_review_threads = [_thread_that_runs_for(0.2)]
    started = time.monotonic()
    assert AIAgent.wait_for_background_reviews(agent, timeout_sec=2.0) == 1
    elapsed = time.monotonic() - started
    assert 0.15 < elapsed < 1.5, (
        f"expected to wait ~0.2s for the sleeping thread, took {elapsed:.2f}s"
    )
    assert agent._bg_review_threads == []


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_hung_thread_times_out_and_is_abandoned():
    agent = _StubAgent()
    hung = _thread_that_runs_for(30.0)
    agent._bg_review_threads = [hung]
    started = time.monotonic()
    result = AIAgent.wait_for_background_reviews(agent, timeout_sec=0.3)
    elapsed = time.monotonic() - started
    assert result == 0, "hung thread should not be counted as completed"
    # We waited only as long as the timeout
    assert 0.25 < elapsed < 1.0, (
        f"expected to wait ~0.3s for the hung thread, took {elapsed:.2f}s"
    )
    # The thread is still alive — list retains it (daemon will die on exit)
    assert agent._bg_review_threads == [hung]


def test_multiple_threads_share_the_deadline():
    """Total wait is bounded by ``timeout_sec``, not N × timeout — so
    one slow review can't make every other review wait its full budget."""
    agent = _StubAgent()
    agent._bg_review_threads = [
        _thread_that_runs_for(5.0),  # hangs past the deadline
        _thread_that_runs_for(5.0),
        _thread_that_runs_for(5.0),
    ]
    started = time.monotonic()
    result = AIAgent.wait_for_background_reviews(agent, timeout_sec=0.3)
    elapsed = time.monotonic() - started
    assert result == 0
    # Without the shared deadline this would be ~3 × 0.3 = 0.9s.  With it,
    # we should bail right at the deadline.
    assert elapsed < 0.8, (
        f"3 hung threads should share one 0.3s budget, took {elapsed:.2f}s"
    )


def test_mixed_quick_and_slow_threads():
    """Quick threads succeed; slow ones time out.  Verifies the per-call
    count reflects only the threads that actually finished."""
    agent = _StubAgent()
    agent._bg_review_threads = [
        _thread_that_runs_for(0.1),  # finishes
        _thread_that_runs_for(0.1),  # finishes
        _thread_that_runs_for(5.0),  # hangs past deadline
    ]
    result = AIAgent.wait_for_background_reviews(agent, timeout_sec=1.0)
    assert result == 2, "only the two quick threads should be counted"
    # Hung thread retained
    assert len(agent._bg_review_threads) == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_safe_to_call_multiple_times():
    agent = _StubAgent()
    agent._bg_review_threads = [_thread_that_returns_immediately()]
    AIAgent.wait_for_background_reviews(agent, timeout_sec=0.5)
    # Second call: list is empty
    assert AIAgent.wait_for_background_reviews(agent, timeout_sec=0.5) == 0
    assert agent._bg_review_threads == []
