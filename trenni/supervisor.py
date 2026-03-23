"""Core supervisor: event polling loop + Podman-backed job launcher."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import TrenniConfig
from .pasloe_client import Event, PasloeClient
from .podman_backend import PodmanBackend
from .runtime_builder import RuntimeSpecBuilder, build_runtime_defaults
from .runtime_types import ContainerState, JobHandle

logger = logging.getLogger(__name__)


@dataclass
class SpawnedJob:
    """A job recognised by the supervisor, possibly waiting on dependencies."""
    job_id: str
    source_event_id: str
    task: str
    role: str
    repo: str
    init_branch: str
    evo_sha: str | None
    llm_overrides: dict[str, Any] = field(default_factory=dict)
    workspace_overrides: dict[str, Any] = field(default_factory=dict)
    publication_overrides: dict[str, Any] = field(default_factory=dict)
    depends_on: frozenset[str] = field(default_factory=frozenset)


@dataclass
class SpawnDefaults:
    repo: str
    init_branch: str
    role: str
    evo_sha: str | None
    llm_overrides: dict[str, Any] = field(default_factory=dict)
    workspace_overrides: dict[str, Any] = field(default_factory=dict)
    publication_overrides: dict[str, Any] = field(default_factory=dict)


_DEFAULT_CHECKPOINT_CYCLES = 30
_ACTIVE_CONTAINER_STATES = {"created", "configured", "running", "paused"}
_CONTROL_EVENT_TIMEOUT_S = 1.0


class Supervisor:
    def __init__(self, config: TrenniConfig) -> None:
        self.config = config
        self.client = PasloeClient(
            base_url=config.pasloe_url,
            api_key_env=config.pasloe_api_key_env,
            source_id=config.source_id,
        )
        self.running: bool = False

        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()

        self.runtime_defaults = build_runtime_defaults(config)
        self.runtime_builder = RuntimeSpecBuilder(config, self.runtime_defaults)
        self.backend = PodmanBackend(self.runtime_defaults)

        self.event_cursor: str | None = None
        self.jobs: dict[str, JobHandle] = {}

        self._ready_queue: asyncio.Queue[SpawnedJob] = asyncio.Queue()
        self._pending: dict[str, SpawnedJob] = {}
        self._completed_jobs: set[str] = set()
        self._failed_jobs: set[str] = set()
        self._job_summaries: dict[str, str] = {}
        self._processed_event_ids: set[str] = set()
        self._launched_event_ids: set[str] = set()
        self._spawn_defaults_by_job: dict[str, SpawnDefaults] = {}

        self._checkpoint_cycles = _DEFAULT_CHECKPOINT_CYCLES
        self._reap_timeout = self.runtime_defaults.cleanup_timeout_seconds

        self._webhook_id: str | None = None
        self._webhook_active: bool = False
        self._webhook_poll_not_before: float = 0.0

    @property
    def paused(self) -> bool:
        return not self._resume_event.is_set()

    async def start(self) -> None:
        logger.info(
            "Supervisor starting (max_workers=%d, runtime=%s)",
            self.config.max_workers, self.runtime_defaults.kind,
        )
        self.running = True
        drain_task: asyncio.Task | None = None
        try:
            await self.backend.ensure_ready()
            await self.client.register_source()
            logger.info("Registered source '%s' with Pasloe", self.config.source_id)

            await self._replay_unfinished_tasks()
            await self._try_register_webhook()

            drain_task = asyncio.create_task(self._drain_queue())
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Supervisor loop cancelled")
        finally:
            self.running = False
            if drain_task is not None:
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass
            await self.client.close()
            await self.backend.close()

    async def stop(self, force: bool = False) -> None:
        logger.info("Supervisor stopping (force=%s)", force)
        self.running = False
        if self._webhook_id:
            try:
                await self.client.delete_webhook(self._webhook_id)
                logger.info("Deregistered webhook %s", self._webhook_id)
            except Exception as exc:
                logger.warning("Could not deregister webhook: %s", exc)

        for job_id, handle in list(self.jobs.items()):
            try:
                if force:
                    await self.backend.remove(handle, force=True)
                else:
                    await self.backend.stop(handle, self.runtime_defaults.stop_grace_seconds)
                    await self.backend.remove(handle)
            except Exception as exc:
                logger.warning("Could not stop job %s cleanly: %s", job_id, exc)
        self.jobs.clear()

    async def pause(self) -> None:
        logger.info("Supervisor pausing")
        self._resume_event.clear()
        await self._emit_control_event("supervisor.paused")

    async def resume(self) -> None:
        logger.info("Supervisor resuming")
        self._resume_event.set()
        await self._emit_control_event("supervisor.resumed")

    async def _try_register_webhook(self) -> None:
        url = self.config.trenni_webhook_url
        if not url:
            return
        try:
            self._webhook_id = await self.client.register_webhook(
                url=url,
                secret=self.config.webhook_secret,
                event_types=["task.submit", "job.spawn.request",
                             "job.completed", "job.failed", "job.started"],
            )
            self._webhook_active = True
            self._reset_webhook_poll_deadline()
            logger.info(
                "Registered webhook (id=%s) → %s (poll fallback every %.0fs)",
                self._webhook_id, url, self.config.webhook_poll_interval,
            )
        except Exception as exc:
            logger.warning(
                "Could not register webhook with Pasloe (%s) — falling back to pure polling",
                exc,
            )

    async def _run_loop(self) -> None:
        polls_since_checkpoint = 0
        while self.running:
            if self._poll_due_now():
                try:
                    await self._poll_and_handle()
                except Exception:
                    logger.exception("Error in poll cycle")

            try:
                await self._mark_exited_jobs()
            except Exception:
                logger.exception("Error while inspecting job containers")

            polls_since_checkpoint += 1
            if polls_since_checkpoint >= self._checkpoint_cycles:
                await self._checkpoint()
                polls_since_checkpoint = 0

            await asyncio.sleep(self.config.poll_interval)

    async def _drain_queue(self) -> None:
        while True:
            await self._resume_event.wait()
            job = await self._ready_queue.get()
            while not self._has_capacity() or not self._resume_event.is_set():
                await asyncio.sleep(1.0)
            try:
                await self._launch_from_spawned(job)
            except Exception:
                logger.exception(
                    "Failed to launch queued job %s, dropping (recoverable on restart)",
                    job.job_id,
                )

    async def _launch_from_spawned(self, job: SpawnedJob) -> None:
        await self._launch(
            job_id=job.job_id,
            task=job.task,
            role=job.role,
            repo=job.repo,
            init_branch=job.init_branch,
            evo_sha=job.evo_sha,
            llm_overrides=job.llm_overrides,
            workspace_overrides=job.workspace_overrides,
            publication_overrides=job.publication_overrides,
            source_event_id=job.source_event_id,
        )

    async def _poll_and_handle(self) -> None:
        events, next_cursor = await self.client.poll(
            cursor=self.event_cursor,
            limit=100,
        )
        if next_cursor:
            self.event_cursor = next_cursor
        elif events:
            last = events[-1]
            self.event_cursor = f"{last.ts.isoformat()}|{last.id}"

        for event in events:
            try:
                await self._handle_event(event)
            except Exception:
                logger.exception("Error handling event %s (type=%s)", event.id, event.type)

    async def _handle_event(self, event: Event, *, realtime: bool = False) -> None:
        if realtime:
            self._advance_cursor_from_event(event)
            self._reset_webhook_poll_deadline()

        if event.id in self._processed_event_ids:
            logger.debug("Skipping already-processed event %s (type=%s)", event.id, event.type)
            return

        self._processed_event_ids.add(event.id)

        match event.type:
            case "task.submit":
                await self._handle_task_submit(event)
            case "job.spawn.request":
                await self._handle_spawn(event)
            case "job.completed" | "job.failed":
                await self._handle_job_done(event)
            case "job.started":
                self._handle_job_started(event)

    async def _handle_task_submit(self, event: Event) -> None:
        if event.id in self._launched_event_ids:
            logger.debug("Skipping already-processed task.submit %s", event.id)
            return

        data = event.data
        task = data.get("task", "")
        if not task:
            logger.warning("Ignoring task.submit with empty task (event=%s)", event.id)
            return

        self._launched_event_ids.add(event.id)

        job = SpawnedJob(
            job_id=self._generate_job_id(),
            source_event_id=event.id,
            task=task,
            role=data.get("role", "default"),
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", data.get("branch", "main")),
            evo_sha=data.get("evo_sha"),
            llm_overrides=dict(data.get("llm") or {}),
            workspace_overrides=dict(data.get("workspace") or {}),
            publication_overrides=dict(data.get("publication") or {}),
        )
        await self._enqueue(job)
        logger.info(
            "Queued task %s (job_id=%s, queue_size=%d)",
            event.id, job.job_id, self._ready_queue.qsize(),
        )

    async def _handle_spawn(self, event: Event) -> None:
        data = event.data
        parent_job_id = data.get("job_id", "")
        tasks = data.get("tasks", [])
        parent_defaults = self._spawn_defaults_by_job.get(parent_job_id)

        if not tasks:
            logger.warning("Empty spawn request from job %s", parent_job_id)
            return

        child_ids: list[str] = []

        for i, child_def in enumerate(tasks):
            child_id = f"{parent_job_id}-c{i}"
            prompt = (child_def.get("prompt") or child_def.get("task") or "").strip()
            if not prompt:
                logger.warning("Ignoring spawned child %s with empty prompt", child_id)
                continue

            child_ids.append(child_id)

            job_spec = dict(child_def.get("job_spec") or {})

            role = job_spec.get("role")
            if not role and child_def.get("role_file"):
                role = str(child_def["role_file"]).replace("roles/", "").replace(".py", "")
            if not role and parent_defaults:
                role = parent_defaults.role

            repo = job_spec.get("repo") or child_def.get("repo", "")
            if not repo and parent_defaults:
                repo = parent_defaults.repo

            init_branch = job_spec.get("init_branch") or child_def.get("init_branch") or child_def.get("branch")
            if not init_branch and parent_defaults:
                init_branch = parent_defaults.init_branch

            evo_sha = job_spec.get("evo_sha") or child_def.get("evo_sha") or child_def.get("role_sha")
            if not evo_sha and parent_defaults:
                evo_sha = parent_defaults.evo_sha

            llm_overrides = dict(parent_defaults.llm_overrides if parent_defaults else {})
            llm_overrides.update(dict(job_spec.get("llm") or child_def.get("llm") or {}))

            workspace_overrides = dict(parent_defaults.workspace_overrides if parent_defaults else {})
            workspace_overrides.update(dict(job_spec.get("workspace") or child_def.get("workspace") or {}))

            publication_overrides = dict(parent_defaults.publication_overrides if parent_defaults else {})
            publication_overrides.update(dict(job_spec.get("publication") or child_def.get("publication") or {}))

            child = SpawnedJob(
                job_id=child_id,
                source_event_id=event.id,
                task=prompt,
                role=role or "default",
                repo=repo,
                init_branch=init_branch or "main",
                evo_sha=evo_sha,
                llm_overrides=llm_overrides,
                workspace_overrides=workspace_overrides,
                publication_overrides=publication_overrides,
            )
            await self._enqueue(child)

        logger.info(
            "Spawn: parent=%s children=%s (fan-out only, no continuation)",
            parent_job_id, child_ids,
        )

    async def _enqueue(self, job: SpawnedJob) -> None:
        unsatisfied = job.depends_on - self._completed_jobs
        if not unsatisfied:
            await self._ready_queue.put(job)
        else:
            self._pending[job.job_id] = job

    async def _handle_job_done(self, event: Event) -> None:
        job_id = event.data.get("job_id", "")
        is_failure = event.type == "job.failed"
        logger.info("Job %s %s", job_id, "failed" if is_failure else "completed")

        handle = self.jobs.pop(job_id, None)
        self._completed_jobs.add(job_id)
        if is_failure:
            self._failed_jobs.add(job_id)
        self._job_summaries[job_id] = event.data.get("summary", "")

        if handle is not None:
            await self._cleanup_handle(handle, failed=is_failure)

        newly_ready: list[str] = []
        propagate_failed: list[str] = []

        for pending_id, pending_job in list(self._pending.items()):
            failed_deps = pending_job.depends_on & self._failed_jobs
            if failed_deps:
                propagate_failed.append(pending_id)
                continue

            unsatisfied = pending_job.depends_on - self._completed_jobs
            if not unsatisfied:
                newly_ready.append(pending_id)
                await self._ready_queue.put(pending_job)

        for jid in newly_ready:
            del self._pending[jid]

        for jid in propagate_failed:
            pending_job = self._pending.pop(jid)
            failed_deps = pending_job.depends_on & self._failed_jobs
            logger.warning(
                "Propagating failure to job %s (failed deps: %s)",
                jid, sorted(failed_deps),
            )
            self._completed_jobs.add(jid)
            self._failed_jobs.add(jid)
            await self.client.emit("job.failed", {
                "job_id": jid,
                "error": f"Dependency failed: {', '.join(sorted(failed_deps))}",
                "code": "dependency_failed",
            })

    async def _mark_exited_jobs(self) -> None:
        now = time.monotonic()
        for handle in self.jobs.values():
            state = await self.backend.inspect(handle)
            if not state.exists:
                if handle.exited_at is None:
                    handle.exited_at = now
                continue

            if state.running or state.status in _ACTIVE_CONTAINER_STATES:
                continue

            if handle.exited_at is None:
                handle.exited_at = now
                handle.exit_code = state.exit_code
                logger.info(
                    "Container for job %s exited (status=%s, rc=%s)",
                    handle.job_id,
                    state.status or "unknown",
                    "?" if state.exit_code is None else state.exit_code,
                )

    async def _checkpoint(self) -> None:
        now = time.monotonic()
        reaped: list[JobHandle] = []

        for handle in list(self.jobs.values()):
            if handle.exited_at is None or (now - handle.exited_at) <= self._reap_timeout:
                continue

            logs = await self.backend.logs(handle)
            logger.warning(
                "Job %s container exited %ds ago without terminal event",
                handle.job_id, int(now - handle.exited_at),
            )
            await self.client.emit("job.failed", {
                "job_id": handle.job_id,
                "error": (
                    "Container exited without emitting terminal event "
                    f"(exit_code={handle.exit_code})"
                ),
                "code": "runtime_lost",
                "logs_tail": logs[-4000:],
            })
            reaped.append(handle)

        for handle in reaped:
            self.jobs.pop(handle.job_id, None)
            self._completed_jobs.add(handle.job_id)
            self._failed_jobs.add(handle.job_id)
            await self._cleanup_handle(handle, failed=True)

        await self.client.emit("supervisor.checkpoint", {
            "cursor": self.event_cursor,
            "running_jobs": list(self.jobs.keys()),
            "pending_jobs": list(self._pending.keys()),
            "ready_queue_size": self._ready_queue.qsize(),
            "completed_count": len(self._completed_jobs),
        })

    async def _fetch_all(
        self,
        type_: str,
        source: str | None = None,
    ) -> list[Event]:
        results: list[Event] = []
        cursor = None
        while True:
            events, next_cursor = await self.client.poll(
                cursor=cursor,
                source=source,
                type_=type_,
                limit=100,
            )
            results.extend(events)
            if not next_cursor:
                break
            cursor = next_cursor
        return results

    async def _replay_unfinished_tasks(self) -> None:
        logger.info("Replaying unfinished tasks from Pasloe...")

        checkpoints = await self._fetch_all(
            "supervisor.checkpoint", source=self.config.source_id
        )
        replay_cursor: str | None = None
        if checkpoints:
            latest_cp = checkpoints[-1]
            replay_cursor = latest_cp.data.get("cursor")
            logger.info("Found checkpoint, replay from cursor=%s", replay_cursor)

        launched_events = await self._fetch_all(
            "supervisor.job.launched", source=self.config.source_id
        )
        started_events = await self._fetch_all("job.started")
        completed_events = await self._fetch_all("job.completed")
        failed_events = await self._fetch_all("job.failed")
        submit_events = await self._fetch_all("task.submit")

        launched_map: dict[str, Event] = {
            e.data["source_event_id"]: e
            for e in launched_events
            if e.data.get("source_event_id")
        }
        for e in launched_events:
            job_id = e.data.get("job_id", "")
            if not job_id:
                continue
            self._spawn_defaults_by_job[job_id] = SpawnDefaults(
                repo=e.data.get("repo", ""),
                init_branch=e.data.get("init_branch", "main"),
                role=e.data.get("role", "default"),
                evo_sha=e.data.get("evo_sha") or None,
                llm_overrides=dict(e.data.get("llm") or {}),
                workspace_overrides=dict(e.data.get("workspace") or {}),
                publication_overrides=dict(e.data.get("publication") or {}),
            )
        started_job_ids = {
            e.data["job_id"] for e in started_events if e.data.get("job_id")
        }
        finished_job_ids = {
            e.data["job_id"]
            for e in (completed_events + failed_events)
            if e.data.get("job_id")
        }

        for e in completed_events:
            jid = e.data.get("job_id", "")
            if jid:
                self._completed_jobs.add(jid)
                self._job_summaries[jid] = e.data.get("summary", "")
        for e in failed_events:
            jid = e.data.get("job_id", "")
            if jid:
                self._completed_jobs.add(jid)
                self._failed_jobs.add(jid)

        all_events = launched_events + started_events + completed_events + failed_events + submit_events
        if replay_cursor:
            self.event_cursor = replay_cursor
        else:
            latest = max(all_events, key=lambda e: (e.ts, e.id), default=None)
            if latest:
                self.event_cursor = f"{latest.ts.isoformat()}|{latest.id}"
        logger.info("Replay cursor set to %s", self.event_cursor)

        enqueued = skipped = reattached = lost = 0
        for event in submit_events:
            data = event.data
            task = data.get("task", "")
            if not task:
                continue

            launched = launched_map.get(event.id)
            if launched is None:
                self._launched_event_ids.add(event.id)
                await self._enqueue(self._spawned_job_from_event(event))
                enqueued += 1
                continue

            job_id = launched.data.get("job_id", "")
            container_id = launched.data.get("container_id", "")
            container_name = launched.data.get("container_name", "")

            if job_id in finished_job_ids:
                self._launched_event_ids.add(event.id)
                skipped += 1
                continue

            state = await self._inspect_replay_state(container_id, container_name)
            handle = self._handle_from_replay(job_id, container_id, container_name)

            if job_id in started_job_ids:
                if state.exists and (state.running or state.status in _ACTIVE_CONTAINER_STATES):
                    self.jobs[job_id] = handle
                    self._launched_event_ids.add(event.id)
                    reattached += 1
                    continue

                self._completed_jobs.add(job_id)
                self._failed_jobs.add(job_id)
                self._launched_event_ids.add(event.id)
                await self.client.emit("job.failed", {
                    "job_id": job_id,
                    "error": "Container disappeared before a terminal event was emitted",
                    "code": "runtime_lost",
                })
                if state.exists:
                    await self._cleanup_handle(handle, failed=True)
                lost += 1
                continue

            if state.exists and (state.running or state.status in _ACTIVE_CONTAINER_STATES):
                self.jobs[job_id] = handle
                self._launched_event_ids.add(event.id)
                reattached += 1
                continue

            if state.exists:
                await self._cleanup_handle(handle, failed=True)

            self._launched_event_ids.add(event.id)
            await self._enqueue(self._spawned_job_from_event(event))
            enqueued += 1

        logger.info(
            "Replay complete: %d enqueued, %d skipped, %d reattached, %d lost",
            enqueued, skipped, reattached, lost,
        )

    def _handle_job_started(self, event: Event) -> None:
        job_id = event.data.get("job_id", "")
        evo_sha = event.data.get("evo_sha", "")
        logger.info("Job %s started (evo_sha=%s)", job_id, evo_sha or "unknown")

    async def _launch(
        self,
        job_id: str,
        task: str,
        role: str,
        repo: str,
        init_branch: str,
        evo_sha: str | None,
        llm_overrides: dict[str, Any] | None = None,
        workspace_overrides: dict[str, Any] | None = None,
        publication_overrides: dict[str, Any] | None = None,
        source_event_id: str = "",
    ) -> None:
        logger.info("Launching job %s (role=%s, source=%s)", job_id, role, source_event_id or "?")

        spec = self.runtime_builder.build(
            job_id=job_id,
            source_event_id=source_event_id,
            task=task,
            role=role,
            repo=repo,
            init_branch=init_branch,
            evo_sha=evo_sha,
            llm_overrides=llm_overrides,
            workspace_overrides=workspace_overrides,
            publication_overrides=publication_overrides,
        )
        handle = await self.backend.create(spec)
        try:
            await self.backend.start(handle)
        except Exception:
            await self.backend.remove(handle, force=True)
            raise

        self.jobs[job_id] = handle
        self._spawn_defaults_by_job[job_id] = SpawnDefaults(
            repo=repo,
            init_branch=init_branch,
            role=role,
            evo_sha=evo_sha,
            llm_overrides=dict(llm_overrides or {}),
            workspace_overrides=dict(workspace_overrides or {}),
            publication_overrides=dict(publication_overrides or {}),
        )

        await self.client.emit("supervisor.job.launched", {
            "job_id": job_id,
            "source_event_id": source_event_id,
            "task": task,
            "role": role,
            "repo": repo,
            "init_branch": init_branch,
            "evo_sha": evo_sha or "",
            "llm": dict(llm_overrides or {}),
            "workspace": dict(workspace_overrides or {}),
            "publication": dict(publication_overrides or {}),
            "runtime_kind": self.runtime_defaults.kind,
            "container_id": handle.container_id,
            "container_name": handle.container_name,
        })

    async def _cleanup_handle(self, handle: JobHandle, *, failed: bool) -> None:
        if failed and self.runtime_defaults.retain_on_failure:
            return
        try:
            await self.backend.stop(handle, self.runtime_defaults.stop_grace_seconds)
        except Exception:
            pass
        try:
            await self.backend.remove(handle, force=failed)
        except Exception as exc:
            logger.warning("Could not remove container for job %s: %s", handle.job_id, exc)

    async def _emit_control_event(self, event_type: str) -> None:
        try:
            await self.client.emit(
                event_type,
                {},
                timeout=_CONTROL_EVENT_TIMEOUT_S,
            )
        except Exception:
            logger.warning("Could not emit %s event", event_type)

    def _poll_due_now(self) -> bool:
        if not self._webhook_active:
            return True
        return time.monotonic() >= self._webhook_poll_not_before

    def _reset_webhook_poll_deadline(self) -> None:
        if not self._webhook_active:
            return
        self._webhook_poll_not_before = time.monotonic() + self.config.webhook_poll_interval

    def _advance_cursor_from_event(self, event: Event) -> None:
        current = self._cursor_key(self.event_cursor)
        candidate = (event.ts, event.id)
        if current is None or candidate > current:
            self.event_cursor = f"{event.ts.isoformat()}|{event.id}"

    @staticmethod
    def _cursor_key(cursor: str | None) -> tuple[datetime, str] | None:
        if not cursor:
            return None
        try:
            ts_raw, event_id = cursor.split("|", 1)
        except ValueError:
            return None
        try:
            return datetime.fromisoformat(ts_raw), event_id
        except ValueError:
            return None

    async def _inspect_replay_state(self, container_id: str, container_name: str) -> ContainerState:
        ref = container_id or container_name
        if not ref:
            return ContainerState(exists=False)
        return await self.backend.inspect(
            JobHandle(job_id="", container_id=container_id or ref, container_name=container_name or ref)
        )

    def _handle_from_replay(self, job_id: str, container_id: str, container_name: str) -> JobHandle:
        ref = container_id or container_name or f"yoitsu-job-{job_id}"
        return JobHandle(
            job_id=job_id,
            container_id=container_id or ref,
            container_name=container_name or ref,
        )

    def _spawned_job_from_event(self, event: Event) -> SpawnedJob:
        data = event.data
        return SpawnedJob(
            source_event_id=event.id,
            job_id=self._generate_job_id(),
            task=data.get("task", ""),
            role=data.get("role", "default"),
            repo=data.get("repo", ""),
            init_branch=data.get("init_branch", data.get("branch", "main")),
            evo_sha=data.get("evo_sha"),
            llm_overrides=dict(data.get("llm") or {}),
            workspace_overrides=dict(data.get("workspace") or {}),
            publication_overrides=dict(data.get("publication") or {}),
        )

    def _has_capacity(self) -> bool:
        return len(self.jobs) < self.config.max_workers

    def _generate_job_id(self) -> str:
        import uuid_utils
        return str(uuid_utils.uuid7())

    @property
    def status(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "running_jobs": len(self.jobs),
            "max_workers": self.config.max_workers,
            "pending_jobs": len(self._pending),
            "ready_queue_size": self._ready_queue.qsize(),
            "runtime_kind": self.runtime_defaults.kind,
        }
