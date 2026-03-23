"""Compatibility exports for the Podman job runtime.

This module used to host subprocess and bubblewrap launch code. The deployed
runtime is now Podman-only; keep this file as a narrow import shim while the
rest of the repo finishes migrating to the new runtime module names.
"""
from __future__ import annotations

from .podman_backend import PodmanBackend
from .runtime_builder import RuntimeSpecBuilder, build_git_credential_env, build_runtime_defaults
from .runtime_types import ContainerExit, ContainerState, JobHandle, JobRuntimeSpec, RuntimeDefaults

__all__ = [
    "ContainerExit",
    "ContainerState",
    "JobHandle",
    "JobRuntimeSpec",
    "PodmanBackend",
    "RuntimeDefaults",
    "RuntimeSpecBuilder",
    "build_git_credential_env",
    "build_runtime_defaults",
]
