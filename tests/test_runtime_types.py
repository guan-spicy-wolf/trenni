"""Tests for runtime_types module."""
from __future__ import annotations

import pytest

from trenni.runtime_types import JobRuntimeSpec


class TestJobRuntimeSpec:
    """Tests for JobRuntimeSpec dataclass."""

    def test_pod_name_can_be_none(self) -> None:
        """pod_name should accept None to indicate no pod."""
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="yoitsu-job-job-1",
            image="localhost/test:latest",
            pod_name=None,  # No pod - standalone container
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
        )
        assert spec.pod_name is None

    def test_pod_name_can_be_string(self) -> None:
        """pod_name should still accept string for pod-based containers."""
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="yoitsu-job-job-1",
            image="localhost/test:latest",
            pod_name="yoitsu-dev",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
        )
        assert spec.pod_name == "yoitsu-dev"

    def test_extra_networks_defaults_to_empty_tuple(self) -> None:
        """extra_networks should default to empty tuple when not provided."""
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="yoitsu-job-job-1",
            image="localhost/test:latest",
            pod_name="yoitsu-dev",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
        )
        assert spec.extra_networks == ()

    def test_extra_networks_can_be_provided(self) -> None:
        """extra_networks should accept tuple of network names."""
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="yoitsu-job-job-1",
            image="localhost/test:latest",
            pod_name="yoitsu-dev",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
            extra_networks=("network-a", "network-b"),
        )
        assert spec.extra_networks == ("network-a", "network-b")

    def test_is_frozen(self) -> None:
        """JobRuntimeSpec should be immutable (frozen dataclass)."""
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="yoitsu-job-job-1",
            image="localhost/test:latest",
            pod_name="yoitsu-dev",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
        )
        with pytest.raises(AttributeError):
            spec.job_id = "job-2"  # type: ignore[misc]