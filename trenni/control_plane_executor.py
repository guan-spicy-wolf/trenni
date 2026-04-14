"""Control-plane capability executor per ADR-0021.

Single subprocess per job, multi-frame JSON-lines protocol.
All setup and finalize frames share one 300s total deadline.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

from yoitsu_contracts import (
    ControlPlaneContext,
    EventData,
    FinalizeResult,
    HostPathView,
    ContainerPathView,
)

logger = logging.getLogger(__name__)


class ControlPlaneTimeout(Exception):
    """Raised when control-plane subprocess exceeds total deadline."""
    pass


class ControlPlaneFrameError(Exception):
    """Raised when a control-plane frame returns ok=false."""
    pass


class SharedDeadline:
    """Shared deadline tracker for multiple operations.

    Python's asyncio.timeout() creates a new timeout for each use.
    We need to track elapsed time across multiple frames manually.
    """

    def __init__(self, total_seconds: float):
        self._deadline = time.monotonic() + total_seconds

    def remaining(self) -> float:
        """Return remaining seconds, or 0 if expired."""
        remaining = self._deadline - time.monotonic()
        return max(0, remaining)

    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        return time.monotonic() >= self._deadline

    async def wrap(self, coro):
        """Wrap a coroutine with remaining timeout."""
        remaining = self.remaining()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        async with asyncio.timeout(remaining):
            return await coro


class ControlPlaneExecutor:
    """Executes control-plane capabilities in a subprocess.

    Per ADR-0021 A.3:
    - Single subprocess per job
    - Multi-frame JSON-lines over stdin/stdout
    - Shared 300s deadline across setup + finalize
    - stderr forwarded to Trenni logs (not part of protocol)

    Usage:
        proc = await executor.start_subprocess(bundle, master_worktree)
        deadline = SharedDeadline(300)
        events, success = await executor.send_frame(proc, "setup", "factorio_mount", ctx, deadline)
        events, success = await executor.send_frame(proc, "finalize", "factorio_mount", ctx, deadline)
        await executor.stop_subprocess(proc)
    """

    TOTAL_TIMEOUT_SECONDS = 300  # ADR-0021 A.8

    async def start_subprocess(
        self,
        bundle: str,
        master_worktree: Path,
    ) -> asyncio.subprocess.Process:
        """Start the control-plane subprocess.

        cwd = master_worktree (contains bundle capabilities/).
        sys.path is prepended by the subprocess entry point.

        Args:
            bundle: Bundle name (for logging)
            master_worktree: Path to worktree at master@switched_sha

        Returns:
            asyncio.subprocess.Process handle
        """
        logger.info(f"Starting control-plane subprocess for {bundle} at {master_worktree}")

        # Use current interpreter to ensure venv packages are available
        # (Trenni runs in venv; system Python may not have trenni installed)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m", "trenni.capability_subprocess",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,  # Forwarded to logs
            cwd=str(master_worktree),
        )

        return proc

    async def send_frame(
        self,
        proc: asyncio.subprocess.Process,
        op: Literal["setup", "finalize"],
        capability: str,
        context: ControlPlaneContext,
        deadline: SharedDeadline | None = None,
    ) -> tuple[list[EventData], bool]:
        """Send a single frame to the subprocess and read response.

        Args:
            proc: Subprocess handle
            op: Operation type ("setup" or "finalize")
            capability: Capability name to execute
            context: ControlPlaneContext for this frame
            deadline: Shared deadline across all frames (created by caller)

        Returns:
            (events, success) tuple from capability

        Raises:
            ControlPlaneTimeout: If deadline exceeded
            ControlPlaneFrameError: If frame returned ok=false
        """
        if deadline is None:
            deadline = SharedDeadline(self.TOTAL_TIMEOUT_SECONDS)

        # Build frame
        frame = {
            "op": op,
            "capability": capability,
            "context": context.model_dump(mode="json"),
        }
        frame_json = json.dumps(frame) + "\n"

        # Send with deadline
        try:
            proc.stdin.write(frame_json.encode())
            await deadline.wrap(proc.stdin.drain())
        except asyncio.TimeoutError:
            raise ControlPlaneTimeout(
                f"Control-plane {op} for {capability} exceeded deadline"
            )

        # Read response with deadline
        try:
            response_line = await deadline.wrap(proc.stdout.readline())
        except asyncio.TimeoutError:
            raise ControlPlaneTimeout(
                f"Control-plane {op} for {capability} response exceeded deadline"
            )

        if not response_line:
            raise ControlPlaneFrameError(
                f"Control-plane subprocess returned empty response for {capability} {op}"
            )

        # Parse response
        try:
            response = json.loads(response_line.decode())
        except json.JSONDecodeError as e:
            raise ControlPlaneFrameError(
                f"Control-plane subprocess returned invalid JSON: {e}"
            )

        if not response.get("ok", False):
            error_msg = response.get("error", "unknown error")
            raise ControlPlaneFrameError(
                f"Control-plane {op} for {capability} failed: {error_msg}"
            )

        # Extract events and success
        events_data = response.get("events", [])
        events = [EventData.model_validate(e) for e in events_data]
        success = response.get("success", True)

        logger.debug(
            f"Control-plane {op} for {capability} returned {len(events)} events, success={success}"
        )

        return events, success

    async def stop_subprocess(
        self,
        proc: asyncio.subprocess.Process,
        deadline: SharedDeadline | None = None,
    ) -> None:
        """Stop the subprocess gracefully.

        Closes stdin, waits for exit with deadline, kills if needed.

        Args:
            proc: Subprocess handle
            deadline: Optional deadline (uses remaining from shared deadline)
        """
        if deadline is None:
            deadline = SharedDeadline(10)  # Graceful shutdown timeout

        try:
            proc.stdin.close()
            await deadline.wrap(proc.wait())
            logger.debug(f"Control-plane subprocess exited with code {proc.returncode}")
        except asyncio.TimeoutError:
            logger.warning("Control-plane subprocess did not exit gracefully, killing")
            proc.kill()
            await proc.wait()
        except Exception as e:
            logger.warning(f"Error stopping control-plane subprocess: {e}")
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass

    async def forward_stderr(
        self,
        proc: asyncio.subprocess.Process,
        bundle: str,
    ) -> None:
        """Forward subprocess stderr to Trenni logs.

        Runs as a background task alongside frame execution.

        Args:
            proc: Subprocess handle
            bundle: Bundle name (for log prefix)
        """
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                logger.info(f"[cp-{bundle}] {line.decode().rstrip()}")
        except Exception as e:
            logger.warning(f"Error forwarding stderr for {bundle}: {e}")