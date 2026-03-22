"""Tests for Supervisor pause/resume behaviour."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from trenni.config import TrenniConfig
from trenni.supervisor import Supervisor


def _sup(**kw) -> Supervisor:
    return Supervisor(TrenniConfig(**kw))


class TestPauseResume:
    def test_initial_state_not_paused(self):
        s = _sup()
        assert s.paused is False

    async def test_pause_sets_flag(self):
        s = _sup()
        await s.pause()
        assert s.paused is True

    async def test_resume_clears_flag(self):
        s = _sup()
        await s.pause()
        await s.resume()
        assert s.paused is False

    async def test_pause_emits_pasloe_event(self):
        s = _sup()
        s.client.emit = AsyncMock()
        await s.pause()
        s.client.emit.assert_called_once_with("supervisor.paused", {})

    async def test_resume_emits_pasloe_event(self):
        s = _sup()
        s._resume_event.clear()
        s.client.emit = AsyncMock()
        await s.resume()
        s.client.emit.assert_called_once_with("supervisor.resumed", {})

    async def test_drain_queue_blocks_while_paused(self):
        """While paused, ready jobs stay in queue."""
        s = _sup()
        await s.pause()

        from trenni.supervisor import SpawnedJob
        job = SpawnedJob("j1", "e1", "task", "default", "/r", "main", None)
        await s._ready_queue.put(job)

        drain_task = asyncio.create_task(s._drain_queue())
        await asyncio.sleep(0.05)
        assert s._ready_queue.qsize() == 1  # still in queue
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass


class TestStatusProperty:
    def test_status_includes_paused(self):
        s = _sup()
        st = s.status
        assert "paused" in st
        assert st["paused"] is False
