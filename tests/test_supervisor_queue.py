"""Tests for Trenni supervisor queue and replay logic."""
import re
import pytest


def test_generate_job_id_is_uuid_v7():
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    sup = Supervisor(TrenniConfig())
    job_id = sup._generate_job_id()
    # UUID v7: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        job_id,
    ), f"Not a UUID v7: {job_id}"
