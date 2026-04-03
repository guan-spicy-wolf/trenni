"""Tests for external event handling in supervisor."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from trenni.state import SupervisorState
from yoitsu_contracts.events import TriggerData


class TestExternalEventHandling:
    """Tests for external event to trigger conversion."""

    @pytest.mark.asyncio
    async def test_handle_ci_failure_event(self):
        """CI failure event is converted to trigger and processed."""
        from trenni.supervisor import Supervisor

        state = SupervisorState()
        client = AsyncMock()

        # Create minimal supervisor
        class TestSupervisor:
            def __init__(self):
                self.state = state
                self.client = client
                self._launched_event_ids = set()
                self.scheduler = MagicMock()
                self.scheduler.record_task_submission = MagicMock()

            async def _process_trigger(self, event, data, *, replay=False):
                """Record that trigger was processed."""
                self.processed_trigger = (event, data)

        supervisor = TestSupervisor()

        # Simulate CI failure event
        event = SimpleNamespace(
            id="evt-ci-001",
            type="external.event",
            data={
                "event_type": "ci_failure",
                "repo": "owner/repo",
                "branch": "main",
                "commit_sha": "abc123",
                "workflow": "CI",
                "message": "Tests failed",
            },
        )

        # Import and call handler
        from trenni.supervisor import Supervisor as RealSupervisor
        await RealSupervisor._handle_external_event(supervisor, event, replay=False)

        # Verify trigger was created and processed
        assert hasattr(supervisor, 'processed_trigger')
        _, trigger_data = supervisor.processed_trigger
        assert isinstance(trigger_data, TriggerData)
        assert trigger_data.role == "implementer"
        assert trigger_data.repo == "owner/repo"
        assert trigger_data.init_branch == "main"

    @pytest.mark.asyncio
    async def test_handle_issue_labeled_event(self):
        """Issue labeled event is converted to trigger."""
        from trenni.supervisor import Supervisor

        state = SupervisorState()
        client = AsyncMock()

        class TestSupervisor:
            def __init__(self):
                self.state = state
                self.client = client
                self._launched_event_ids = set()

            async def _process_trigger(self, event, data, *, replay=False):
                self.processed_trigger = (event, data)

        supervisor = TestSupervisor()

        event = SimpleNamespace(
            id="evt-issue-001",
            type="external.event",
            data={
                "event_type": "issue_labeled",
                "repo": "owner/repo",
                "issue_number": 42,
                "label": "needs-review",
                "title": "Bug in feature X",
            },
        )

        from trenni.supervisor import Supervisor as RealSupervisor
        await RealSupervisor._handle_external_event(supervisor, event, replay=False)

        assert hasattr(supervisor, 'processed_trigger')
        _, trigger_data = supervisor.processed_trigger
        assert isinstance(trigger_data, TriggerData)
        assert trigger_data.role == "reviewer"
        assert "Bug in feature X" in trigger_data.goal

    @pytest.mark.asyncio
    async def test_handle_pr_labeled_event(self):
        """PR labeled event is converted to trigger."""
        from trenni.supervisor import Supervisor

        state = SupervisorState()
        client = AsyncMock()

        class TestSupervisor:
            def __init__(self):
                self.state = state
                self.client = client
                self._launched_event_ids = set()

            async def _process_trigger(self, event, data, *, replay=False):
                self.processed_trigger = (event, data)

        supervisor = TestSupervisor()

        event = SimpleNamespace(
            id="evt-pr-001",
            type="external.event",
            data={
                "event_type": "pr_labeled",
                "repo": "owner/repo",
                "pr_number": 42,
                "label": "ready-for-review",
                "title": "Add new feature",
                "head_branch": "feature/new",
            },
        )

        from trenni.supervisor import Supervisor as RealSupervisor
        await RealSupervisor._handle_external_event(supervisor, event, replay=False)

        assert hasattr(supervisor, 'processed_trigger')
        _, trigger_data = supervisor.processed_trigger
        assert isinstance(trigger_data, TriggerData)
        assert trigger_data.role == "reviewer"
        assert "Add new feature" in trigger_data.goal

    @pytest.mark.asyncio
    async def test_unknown_event_type_ignored(self):
        """Unknown external event type is ignored."""
        from trenni.supervisor import Supervisor

        state = SupervisorState()
        client = AsyncMock()

        class TestSupervisor:
            def __init__(self):
                self.state = state
                self.client = client

        supervisor = TestSupervisor()

        event = SimpleNamespace(
            id="evt-unknown",
            type="external.event",
            data={
                "event_type": "unknown_type",
            },
        )

        from trenni.supervisor import Supervisor as RealSupervisor
        await RealSupervisor._handle_external_event(supervisor, event, replay=False)

        # Should not crash, just return silently

    @pytest.mark.asyncio
    async def test_unmapped_label_ignored(self):
        """Label not in mapping is ignored."""
        from trenni.supervisor import Supervisor

        state = SupervisorState()
        client = AsyncMock()

        class TestSupervisor:
            def __init__(self):
                self.state = state
                self.client = client

        supervisor = TestSupervisor()

        event = SimpleNamespace(
            id="evt-label-001",
            type="external.event",
            data={
                "event_type": "issue_labeled",
                "repo": "owner/repo",
                "issue_number": 42,
                "label": "wontfix",  # Not in default mapping
                "title": "Not a bug",
            },
        )

        from trenni.supervisor import Supervisor as RealSupervisor
        await RealSupervisor._handle_external_event(supervisor, event, replay=False)

        # Should not crash, just return silently (trigger_data is None)