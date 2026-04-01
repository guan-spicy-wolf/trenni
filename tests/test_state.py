"""Tests for SupervisorState running_jobs_by_team tracking and TeamLaunchCondition.

ADR-0011 D5: Per-team launch conditions, not scheduling policy.
"""

import pytest


def test_supervisor_state_has_running_jobs_by_team():
    """SupervisorState has running_jobs_by_team field."""
    from trenni.state import SupervisorState

    state = SupervisorState()
    assert hasattr(state, "running_jobs_by_team")
    assert state.running_jobs_by_team == {}


def test_supervisor_state_increment_team_running():
    """SupervisorState.increment_team_running increments counter for team."""
    from trenni.state import SupervisorState

    state = SupervisorState()

    # First increment creates entry
    state.increment_team_running("factorio")
    assert state.running_jobs_by_team == {"factorio": 1}

    # Second increment increases count
    state.increment_team_running("factorio")
    assert state.running_jobs_by_team == {"factorio": 2}

    # Different team has separate counter
    state.increment_team_running("default")
    assert state.running_jobs_by_team == {"factorio": 2, "default": 1}


def test_supervisor_state_decrement_team_running():
    """SupervisorState.decrement_team_running decrements counter for team."""
    from trenni.state import SupervisorState

    state = SupervisorState()
    state.increment_team_running("factorio")
    state.increment_team_running("factorio")
    assert state.running_jobs_by_team == {"factorio": 2}

    # Decrement reduces count
    state.decrement_team_running("factorio")
    assert state.running_jobs_by_team == {"factorio": 1}

    # Decrement to zero removes or leaves at 0
    state.decrement_team_running("factorio")
    assert state.running_jobs_by_team.get("factorio", 0) == 0


def test_supervisor_state_decrement_team_running_does_not_go_negative():
    """Decrementing a team with zero running jobs does not go negative."""
    from trenni.state import SupervisorState

    state = SupervisorState()

    # Decrementing non-existent team should not create negative count
    state.decrement_team_running("factorio")
    assert state.running_jobs_by_team.get("factorio", 0) == 0

    # Decrementing team at zero should stay at zero
    state.increment_team_running("factorio")
    state.decrement_team_running("factorio")
    assert state.running_jobs_by_team.get("factorio", 0) == 0
    state.decrement_team_running("factorio")  # Extra decrement
    assert state.running_jobs_by_team.get("factorio", 0) == 0


def test_supervisor_state_running_count_for_team():
    """SupervisorState.running_count_for_team returns count or 0."""
    from trenni.state import SupervisorState

    state = SupervisorState()

    # Non-existent team returns 0
    assert state.running_count_for_team("factorio") == 0

    # Existing team returns count
    state.increment_team_running("factorio")
    state.increment_team_running("factorio")
    assert state.running_count_for_team("factorio") == 2

    # Different team returns its count
    assert state.running_count_for_team("default") == 0


def test_team_launch_condition_is_satisfied():
    """TeamLaunchCondition.is_satisfied checks running count against max_concurrent."""
    from trenni.state import SupervisorState, TeamLaunchCondition

    state = SupervisorState()

    # max_concurrent=1, 0 running -> satisfied
    condition = TeamLaunchCondition(team="factorio", max_concurrent=1)
    assert condition.is_satisfied(state) is True

    # max_concurrent=1, 1 running -> not satisfied
    state.increment_team_running("factorio")
    assert condition.is_satisfied(state) is False


def test_team_launch_condition_unlimited_when_max_concurrent_zero():
    """TeamLaunchCondition with max_concurrent=0 or negative means no limit."""
    from trenni.state import SupervisorState, TeamLaunchCondition

    state = SupervisorState()

    # max_concurrent=0 -> always satisfied (no limit)
    condition_zero = TeamLaunchCondition(team="factorio", max_concurrent=0)
    state.increment_team_running("factorio")
    state.increment_team_running("factorio")
    state.increment_team_running("factorio")
    assert condition_zero.is_satisfied(state) is True

    # max_concurrent=-1 -> also no limit (negative treated as unlimited)
    condition_negative = TeamLaunchCondition(team="factorio", max_concurrent=-1)
    assert condition_negative.is_satisfied(state) is True


def test_team_launch_condition_different_teams_independent():
    """TeamLaunchCondition checks are independent per team."""
    from trenni.state import SupervisorState, TeamLaunchCondition

    state = SupervisorState()

    # factorio has max_concurrent=1, default has max_concurrent=2
    factorio_condition = TeamLaunchCondition(team="factorio", max_concurrent=1)
    default_condition = TeamLaunchCondition(team="default", max_concurrent=2)

    # Both satisfied initially
    assert factorio_condition.is_satisfied(state) is True
    assert default_condition.is_satisfied(state) is True

    # Add factorio job -> factorio blocked, default still available
    state.increment_team_running("factorio")
    assert factorio_condition.is_satisfied(state) is False
    assert default_condition.is_satisfied(state) is True

    # Add default job -> both at capacity? factorio blocked, default has room
    state.increment_team_running("default")
    assert factorio_condition.is_satisfied(state) is False
    assert default_condition.is_satisfied(state) is True  # 1 < 2

    # Add another default job -> default now blocked too
    state.increment_team_running("default")
    assert factorio_condition.is_satisfied(state) is False
    assert default_condition.is_satisfied(state) is False  # 2 >= 2