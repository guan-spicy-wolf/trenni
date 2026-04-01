from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from yoitsu_contracts.conditions import evaluate_condition
from yoitsu_contracts.events import EvalSpec

from .config import TeamConfig
from .state import SpawnedJob, SupervisorState, TaskRecord, TeamLaunchCondition

_TASK_TERMINAL_STATES = {"complete", "failed", "cancelled"}


class Scheduler:
    def __init__(
        self,
        state: SupervisorState,
        *,
        max_workers: int,
        teams: Mapping[str, TeamConfig] | None = None,
    ) -> None:
        self.state = state
        self.max_workers = max_workers
        self.teams: Mapping[str, TeamConfig] = teams or {}

    def has_capacity(self) -> bool:
        return len(self.state.running_jobs) < self.max_workers

    def has_team_capacity(self, team: str) -> bool:
        """Check if team has capacity for another job.

        Returns True if:
        - Team not configured (no limit)
        - Team max_concurrent_jobs is 0 or negative (unlimited)
        - Current running count < max_concurrent_jobs
        """
        team_config = self.teams.get(team)
        if team_config is None:
            return True  # No config for team = no limit

        condition = TeamLaunchCondition(
            team=team, max_concurrent=team_config.scheduling.max_concurrent_jobs
        )
        return condition.is_satisfied(self.state)

    def evaluate_job(self, job: SpawnedJob) -> bool | None:
        if job.condition is not None:
            return evaluate_condition(job.condition, self.state.task_states())

        if job.depends_on:
            failed_deps = job.depends_on & self.state.failed_jobs
            if failed_deps:
                return False
            unsatisfied = job.depends_on - self.state.completed_jobs
            return True if not unsatisfied else None

        return True

    async def enqueue(self, job: SpawnedJob) -> list[SpawnedJob]:
        task_id = job.task_id or job.job_id
        if not job.task_id:
            job = replace(job, task_id=task_id)

        # Intake replays can redeliver the same logical job. Keep enqueue idempotent
        # by short-circuiting when the job is already scheduled or terminal.
        if job.job_id in self.state.completed_jobs or job.job_id in self.state.cancelled_jobs:
            self.state.jobs_by_id[job.job_id] = job
            return []
        if job.job_id in self.state.running_jobs:
            self.state.jobs_by_id[job.job_id] = job
            return []
        if job.job_id in self.state.pending_jobs or self.state.has_ready_job(job.job_id):
            self.state.jobs_by_id[job.job_id] = job
            return []

        self.state.jobs_by_id[job.job_id] = job

        outcome = self.evaluate_job(job)
        if outcome is True:
            # Check team capacity before putting in ready queue
            if self.has_team_capacity(job.team):
                await self.state.ready_queue.put(job)
                return []
            else:
                # Team at capacity - keep pending
                self.state.pending_jobs[job.job_id] = job
                return []
        if outcome is False:
            return [job]

        self.state.pending_jobs[job.job_id] = job
        return []

    def record_task_submission(
        self,
        *,
        task_id: str,
        goal: str,
        source_event_id: str,
        spec: dict,
        eval_spec: EvalSpec | None = None,
    ) -> None:
        self.state.tasks[task_id] = TaskRecord(
            task_id=task_id,
            goal=goal,
            source_event_id=source_event_id,
            spec=spec,
            eval_spec=eval_spec,
        )

    async def mark_task_terminal(
        self,
        *,
        task_id: str,
        state: str,
    ) -> tuple[list[SpawnedJob], list[SpawnedJob]]:
        record = self.state.tasks.get(task_id)
        if record is None:
            return [], []
        
        if record.terminal:
            return [], []

        record.terminal = True
        record.terminal_state = state

        return await self._resolve_pending()

    async def record_job_terminal(
        self,
        *,
        job_id: str,
        summary: str = "",
        failed: bool = False,
        cancelled: bool = False,
    ) -> tuple[list[SpawnedJob], list[SpawnedJob]]:
        self.state.completed_jobs.add(job_id)
        if summary:
            self.state.job_summaries[job_id] = summary

        if failed:
            self.state.failed_jobs.add(job_id)
        if cancelled:
            self.state.cancelled_jobs.add(job_id)

        return await self._resolve_pending()



    def status_snapshot(self, *, runtime_kind: str, running: bool, paused: bool) -> dict:
        return {
            "running": running,
            "paused": paused,
            "running_jobs": len(self.state.running_jobs),
            "max_workers": self.max_workers,
            "pending_jobs": len(self.state.pending_jobs),
            "ready_queue_size": self.state.ready_queue.qsize(),
            "runtime_kind": runtime_kind,
            "tasks": self.state.task_states(),
        }

    async def _resolve_pending(self) -> tuple[list[SpawnedJob], list[SpawnedJob]]:
        ready: list[SpawnedJob] = []
        cancelled: list[SpawnedJob] = []

        # Track virtual team capacity consumed during this iteration.
        # Jobs promoted to ready queue haven't launched yet, so their
        # team running count isn't incremented. We need to track this
        # to prevent over-promoting from same team in one iteration.
        virtual_team_running: dict[str, int] = {}

        for job_id, job in list(self.state.pending_jobs.items()):
            outcome = self.evaluate_job(job)
            if outcome is True:
                # Check team capacity including virtual jobs already promoted
                team_config = self.teams.get(job.team)
                if team_config is None or team_config.scheduling.max_concurrent_jobs <= 0:
                    # No limit for this team
                    del self.state.pending_jobs[job_id]
                    await self.state.ready_queue.put(job)
                    ready.append(job)
                else:
                    max_concurrent = team_config.scheduling.max_concurrent_jobs
                    actual_running = self.state.running_count_for_team(job.team)
                    virtual_running = virtual_team_running.get(job.team, 0)
                    total_running = actual_running + virtual_running

                    if total_running < max_concurrent:
                        # Has capacity - promote and track virtual running
                        del self.state.pending_jobs[job_id]
                        await self.state.ready_queue.put(job)
                        ready.append(job)
                        virtual_team_running[job.team] = virtual_running + 1
                    # else: keep in pending, team at capacity
            elif outcome is False:
                del self.state.pending_jobs[job_id]
                cancelled.append(job)

        return ready, cancelled
