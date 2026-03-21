"""Process isolation for Palimpsest job subprocesses."""
from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class JobProcess:
    job_id: str
    proc: asyncio.subprocess.Process
    work_dir: Path
    config_path: Path


async def launch_job(
    *,
    job_id: str,
    task: str,
    role: str,
    repo: str,
    branch: str,
    evo_sha: str | None,
    palimpsest_command: str,
    work_dir: Path,
    eventstore_url: str,
    eventstore_api_key_env: str,
    eventstore_source: str,
    llm_defaults: dict,
    workspace_defaults: dict,
    publication_defaults: dict,
) -> JobProcess:
    job_dir = work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "job_id": job_id,
        "task": task,
        "role": role,
        "workspace": {
            "repo": repo,
            "branch": branch,
            **workspace_defaults,
        },
        "llm": {**llm_defaults} if llm_defaults else {},
        "publication": {**publication_defaults} if publication_defaults else {},
        "eventstore": {
            "url": eventstore_url,
            "api_key_env": eventstore_api_key_env,
            "source_id": eventstore_source,
        },
    }

    config_path = job_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    # Minimal environment: inherit only what's needed
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        eventstore_api_key_env: os.environ.get(eventstore_api_key_env, ""),
    }
    # Pass through LLM API key
    for key in ("ANTHROPIC_API_KEY", "GIT_TOKEN"):
        val = os.environ.get(key)
        if val:
            env[key] = val

    proc = await asyncio.create_subprocess_exec(
        palimpsest_command, "run", str(config_path),
        cwd=str(job_dir),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group for cleanup
    )

    return JobProcess(
        job_id=job_id,
        proc=proc,
        work_dir=job_dir,
        config_path=config_path,
    )
