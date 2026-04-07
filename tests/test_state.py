"""Tests for SupervisorState running_jobs_by_bundle tracking and BundleLaunchCondition.

ADR-0011 D5: Per-team launch conditions, not scheduling policy.
"""

import pytest


def test_supervisor_state_has_running_jobs_by_bundle():
    """SupervisorState has running_jobs_by_bundle field."""
    from trenni.state import SupervisorState

    state = SupervisorState()
    assert hasattr(state, "running_jobs_by_bundle")
    assert state.running_jobs_by_bundle == {}


def test_supervisor_state_increment_bundle_running():
    """SupervisorState.increment_bundle_running increments counter for team."""
    from trenni.state import SupervisorState

    state = SupervisorState()

    # First increment creates entry
    state.increment_bundle_running("factorio")
    assert state.running_jobs_by_bundle == {"factorio": 1}

    # Second increment increases count
    state.increment_bundle_running("factorio")
    assert state.running_jobs_by_bundle == {"factorio": 2}

    # Different team has separate counter
    state.increment_bundle_running("default")
    assert state.running_jobs_by_bundle == {"factorio": 2, "default": 1}


def test_supervisor_state_decrement_bundle_running():
    """SupervisorState.decrement_bundle_running decrements counter for team."""
    from trenni.state import SupervisorState

    state = SupervisorState()
    state.increment_bundle_running("factorio")
    state.increment_bundle_running("factorio")
    assert state.running_jobs_by_bundle == {"factorio": 2}

    # Decrement reduces count
    state.decrement_bundle_running("factorio")
    assert state.running_jobs_by_bundle == {"factorio": 1}

    # Decrement to zero removes or leaves at 0
    state.decrement_bundle_running("factorio")
    assert state.running_jobs_by_bundle.get("factorio", 0) == 0


def test_supervisor_state_decrement_bundle_running_does_not_go_negative():
    """Decrementing a team with zero running jobs does not go negative."""
    from trenni.state import SupervisorState

    state = SupervisorState()

    # Decrementing non-existent team should not create negative count
    state.decrement_bundle_running("factorio")
    assert state.running_jobs_by_bundle.get("factorio", 0) == 0

    # Decrementing team at zero should stay at zero
    state.increment_bundle_running("factorio")
    state.decrement_bundle_running("factorio")
    assert state.running_jobs_by_bundle.get("factorio", 0) == 0
    state.decrement_bundle_running("factorio")  # Extra decrement
    assert state.running_jobs_by_bundle.get("factorio", 0) == 0


def test_supervisor_state_running_count_for_bundle():
    """SupervisorState.running_count_for_bundle returns count or 0."""
    from trenni.state import SupervisorState

    state = SupervisorState()

    # Non-existent team returns 0
    assert state.running_count_for_bundle("factorio") == 0

    # Existing team returns count
    state.increment_bundle_running("factorio")
    state.increment_bundle_running("factorio")
    assert state.running_count_for_bundle("factorio") == 2

    # Different team returns its count
    assert state.running_count_for_bundle("default") == 0


def test_bundle_launch_condition_is_satisfied():
    """BundleLaunchCondition.is_satisfied checks running count against max_concurrent."""
    from trenni.state import SupervisorState, BundleLaunchCondition

    state = SupervisorState()

    # max_concurrent=1, 0 running -> satisfied
    condition = BundleLaunchCondition(bundle="factorio", max_concurrent=1)
    assert condition.is_satisfied(state) is True

    # max_concurrent=1, 1 running -> not satisfied
    state.increment_bundle_running("factorio")
    assert condition.is_satisfied(state) is False


def test_bundle_launch_condition_unlimited_when_max_concurrent_zero():
    """BundleLaunchCondition with max_concurrent=0 or negative means no limit."""
    from trenni.state import SupervisorState, BundleLaunchCondition

    state = SupervisorState()

    # max_concurrent=0 -> always satisfied (no limit)
    condition_zero = BundleLaunchCondition(bundle="factorio", max_concurrent=0)
    state.increment_bundle_running("factorio")
    state.increment_bundle_running("factorio")
    state.increment_bundle_running("factorio")
    assert condition_zero.is_satisfied(state) is True

    # max_concurrent=-1 -> also no limit (negative treated as unlimited)
    condition_negative = BundleLaunchCondition(bundle="factorio", max_concurrent=-1)
    assert condition_negative.is_satisfied(state) is True


def test_bundle_launch_condition_different_teams_independent():
    """BundleLaunchCondition checks are independent per team."""
    from trenni.state import SupervisorState, BundleLaunchCondition

    state = SupervisorState()

    # factorio has max_concurrent=1, default has max_concurrent=2
    factorio_condition = BundleLaunchCondition(bundle="factorio", max_concurrent=1)
    default_condition = BundleLaunchCondition(bundle="default", max_concurrent=2)

    # Both satisfied initially
    assert factorio_condition.is_satisfied(state) is True
    assert default_condition.is_satisfied(state) is True

    # Add factorio job -> factorio blocked, default still available
    state.increment_bundle_running("factorio")
    assert factorio_condition.is_satisfied(state) is False
    assert default_condition.is_satisfied(state) is True

    # Add default job -> both at capacity? factorio blocked, default has room
    state.increment_bundle_running("default")
    assert factorio_condition.is_satisfied(state) is False
    assert default_condition.is_satisfied(state) is True  # 1 < 2

    # Add another default job -> default now blocked too
    state.increment_bundle_running("default")
    assert factorio_condition.is_satisfied(state) is False
    assert default_condition.is_satisfied(state) is False  # 2 >= 2