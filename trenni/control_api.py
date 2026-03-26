"""FastAPI control plane for Trenni supervisor.

Exposes:
  GET  /control/status           — supervisor state
  POST /control/pause            — stop dispatching new jobs
  POST /control/resume           — re-enable dispatch
  POST /control/stop             — graceful shutdown
  POST /control/tasks/{task_id}/retry  — retry a failed task
  POST /control/jobs/{job_id}/replay   — replay a failed job
  GET  /control/tasks            — list all tasks
  GET  /control/tasks/{task_id}  — get task details
  POST /hooks/events             — Pasloe webhook delivery intake
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from .supervisor import Supervisor

logger = logging.getLogger(__name__)


class EventPayload(BaseModel):
    id: str
    source_id: str
    type: str
    ts: datetime
    data: dict = {}


class RetryTaskRequest(BaseModel):
    reason: str = "operator_retry"


class ReplayJobRequest(BaseModel):
    reason: str = "operator_replay"


def build_control_app(supervisor: "Supervisor") -> FastAPI:
    app = FastAPI(title="Trenni Control API", docs_url=None, redoc_url=None)

    @app.get("/control/status")
    async def status():
        return supervisor.status

    @app.post("/control/pause")
    async def pause():
        await supervisor.pause()
        return {"ok": True, "paused": True}

    @app.post("/control/resume")
    async def resume():
        await supervisor.resume()
        return {"ok": True, "paused": False}

    @app.post("/control/stop")
    async def stop():
        await supervisor.stop()
        return {"ok": True}

    @app.get("/control/tasks")
    async def list_tasks():
        """List all tasks with their current state."""
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "state": task.state,
                    "terminal": task.terminal,
                    "terminal_state": task.terminal_state,
                    "team": task.team,
                    "eval_spawned": task.eval_spawned,
                    "eval_job_id": task.eval_job_id,
                }
                for task in supervisor.state.tasks.values()
            ]
        }

    @app.get("/control/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get detailed information about a specific task."""
        task = supervisor.state.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return {
            "task_id": task.task_id,
            "goal": task.goal,
            "state": task.state,
            "terminal": task.terminal,
            "terminal_state": task.terminal_state,
            "source_event_id": task.source_event_id,
            "team": task.team,
            "spec": task.spec,
            "eval_spec": task.eval_spec.model_dump(mode="json") if task.eval_spec else None,
            "eval_spawned": task.eval_spawned,
            "eval_job_id": task.eval_job_id,
            "job_order": task.job_order,
            "result": task.result.model_dump(mode="json") if task.result else None,
        }

    @app.post("/control/tasks/{task_id}/retry")
    async def retry_task(task_id: str, req: RetryTaskRequest):
        """Retry a failed or terminal task from the beginning.
        
        This creates a new task with the same goal and spec as the original.
        The original task remains in the history for reference.
        """
        result = await supervisor.retry_task(task_id, reason=req.reason)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        if result is False:
            raise HTTPException(
                status_code=400, 
                detail=f"Task {task_id} cannot be retried - it may be running or already complete"
            )
        return {"ok": True, "task_id": task_id, "new_task_id": result}

    @app.get("/control/jobs")
    async def list_jobs():
        """List all jobs with their current state."""
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "task_id": job.task_id,
                    "task": job.task,
                    "role": job.role,
                    "repo": job.repo,
                    "init_branch": job.init_branch,
                    "state": (
                        "running" if job.job_id in supervisor.jobs
                        else "completed" if job.job_id in supervisor.state.completed_jobs
                        else "failed" if job.job_id in supervisor.state.failed_jobs
                        else "cancelled" if job.job_id in supervisor.state.cancelled_jobs
                        else "pending" if job.job_id in supervisor.state.pending_jobs
                        else "ready" if supervisor.state.has_ready_job(job.job_id)
                        else "unknown"
                    ),
                }
                for job in supervisor.state.jobs_by_id.values()
            ]
        }

    @app.get("/control/jobs/{job_id}")
    async def get_job(job_id: str):
        """Get detailed information about a specific job."""
        job = supervisor.state.jobs_by_id.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Determine job state
        if job_id in supervisor.jobs:
            state = "running"
        elif job_id in supervisor.state.completed_jobs:
            state = "completed"
        elif job_id in supervisor.state.failed_jobs:
            state = "failed"
        elif job_id in supervisor.state.cancelled_jobs:
            state = "cancelled"
        elif job_id in supervisor.state.pending_jobs:
            state = "pending"
        elif supervisor.state.has_ready_job(job_id):
            state = "ready"
        else:
            state = "unknown"
        
        return {
            "job_id": job.job_id,
            "task_id": job.task_id,
            "source_event_id": job.source_event_id,
            "task": job.task,
            "role": job.role,
            "repo": job.repo,
            "init_branch": job.init_branch,
            "evo_sha": job.evo_sha,
            "team": job.team,
            "parent_job_id": job.parent_job_id,
            "state": state,
            "summary": supervisor.state.job_summaries.get(job_id),
            "completion_code": supervisor.state.job_completion_codes.get(job_id),
            "git_ref": supervisor.state.job_git_refs.get(job_id),
        }

    @app.post("/control/jobs/{job_id}/replay")
    async def replay_job(job_id: str, req: ReplayJobRequest):
        """Replay a failed job.
        
        This re-enqueues the job with the same configuration, allowing
        an operator to retry work from a failed branch.
        """
        result = await supervisor.replay_job(job_id, reason=req.reason)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if result is False:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} cannot be replayed - it may be running, completed, or not failed"
            )
        return {"ok": True, "job_id": job_id}

    @app.post("/hooks/events")
    async def receive_event(request: Request):
        body = await request.body()
        secret = getattr(supervisor.config, "webhook_secret", "")
        if secret:
            sig_header = request.headers.get("X-Pasloe-Signature", "")
            expected = "sha256=" + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, sig_header):
                raise HTTPException(status_code=401, detail="Invalid signature")

        try:
            payload = EventPayload.model_validate_json(body)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid payload")

        from .pasloe_client import Event
        event = Event(id=payload.id, source_id=payload.source_id, type=payload.type, ts=payload.ts, data=payload.data)
        try:
            await supervisor._handle_event(event, realtime=True)
        except Exception:
            logger.exception("Error handling webhook event %s", payload.id)
        return {"ok": True}

    return app
