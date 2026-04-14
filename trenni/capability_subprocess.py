"""Control-plane capability subprocess entry point per ADR-0021 A.3.

Runs in master@switched_sha worktree, receives JSON frames over stdin,
executes control-plane capabilities, returns JSON frames over stdout.

Usage:
    python -m trenni.capability_subprocess

Protocol:
    Input frame: {"op": "setup"|"finalize", "capability": "<name>", "context": {...}}
    Output frame: {"ok": true|false, "events": [...], "success": bool, "error": str|None}
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

from yoitsu_contracts import (
    ControlPlaneContext,
    EventData,
    FinalizeResult,
)

# Configure logging to stderr (stdout is for protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _load_control_plane_capabilities(worktree_root: Path) -> dict[str, Any]:
    """Load capabilities from worktree, filtering for surface="control_plane".

    Args:
        worktree_root: Path to master@switched_sha worktree (cwd)

    Returns:
        Dict mapping capability name -> capability instance
    """
    caps_dir = worktree_root / "capabilities"
    if not caps_dir.is_dir():
        logger.warning(f"No capabilities directory at {caps_dir}")
        return {}

    result: dict[str, Any] = {}

    for py_file in caps_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(
                f"cp_cap_{py_file.stem}",
                py_file,
            )
            if not spec or not spec.loader:
                continue

            module = importlib.util.module_from_spec(spec)
            # CRITICAL: prepend worktree to sys.path BEFORE loading
            # This ensures bundle imports resolve correctly
            sys.path.insert(0, str(worktree_root))
            spec.loader.exec_module(module)

            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if obj_name.startswith("_"):
                    continue
                if not isinstance(obj, type):
                    continue
                # Check Capability protocol attributes
                if not hasattr(obj, "name"):
                    continue
                if not hasattr(obj, "surface"):
                    continue
                if not hasattr(obj, "setup") or not hasattr(obj, "finalize"):
                    continue
                # Filter for control-plane only
                if getattr(obj, "surface") != "control_plane":
                    continue
                # Instantiate and register
                try:
                    instance = obj()
                    name = getattr(obj, "name")
                    result[name] = instance
                    logger.info(f"Loaded control-plane capability: {name}")
                except Exception as e:
                    logger.warning(f"Failed to instantiate {obj_name}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load capabilities from {py_file}: {e}")

    return result


def main() -> None:
    """Main entry point for control-plane subprocess."""
    worktree_root = Path.cwd()
    logger.info(f"Control-plane subprocess started at {worktree_root}")

    # Load control-plane capabilities
    capabilities = _load_control_plane_capabilities(worktree_root)
    logger.info(f"Loaded {len(capabilities)} control-plane capabilities")

    if not capabilities:
        logger.error("No control-plane capabilities found")
        # Exit gracefully - supervisor will handle missing capabilities
        sys.exit(0)

    # Process frames from stdin
    for line in sys.stdin:
        if not line.strip():
            continue

        try:
            frame = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON frame: {e}")
            _write_error_frame(f"Invalid JSON: {e}")
            continue

        op = frame.get("op")
        cap_name = frame.get("capability")
        ctx_data = frame.get("context", {})

        if op not in ("setup", "finalize"):
            logger.error(f"Invalid op: {op}")
            _write_error_frame(f"Invalid op: {op}")
            continue

        if not cap_name:
            logger.error("Missing capability name")
            _write_error_frame("Missing capability name")
            continue

        cap = capabilities.get(cap_name)
        if cap is None:
            logger.error(f"Capability not found: {cap_name}")
            _write_error_frame(f"Capability not found: {cap_name}")
            continue

        # Build context
        try:
            ctx = ControlPlaneContext.model_validate(ctx_data)
        except Exception as e:
            logger.error(f"Invalid context: {e}")
            _write_error_frame(f"Invalid context: {e}")
            continue

        # Execute capability
        try:
            if op == "setup":
                # setup() returns list[EventData], not FinalizeResult
                events = cap.setup(ctx)
                _write_success_frame(events, success=True)
                logger.info(f"{op} {cap_name}: success=True, events={len(events)}")
            else:
                # finalize() returns FinalizeResult
                result: FinalizeResult = cap.finalize(ctx)
                _write_success_frame(result.events, result.success)
                logger.info(f"{op} {cap_name}: success={result.success}, events={len(result.events)}")

        except Exception as e:
            logger.error(f"Capability {op} failed: {e}")
            _write_error_frame(f"Capability error: {e}")
            continue

    logger.info("Control-plane subprocess finished")


def _write_success_frame(events: list[EventData], success: bool) -> None:
    """Write a success response frame to stdout."""
    frame = {
        "ok": True,
        "events": [e.model_dump(mode="json") for e in events],
        "success": success,
        "error": None,
    }
    sys.stdout.write(json.dumps(frame) + "\n")
    sys.stdout.flush()


def _write_error_frame(error: str) -> None:
    """Write an error response frame to stdout."""
    frame = {
        "ok": False,
        "events": [],
        "success": False,
        "error": error,
    }
    sys.stdout.write(json.dumps(frame) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()