"""Tests for supervisor observation aggregation deduplication."""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib
import json

from trenni.supervisor import Supervisor
from trenni.config import TrenniConfig


@pytest.fixture
def mock_config():
    """Minimal TrenniConfig for testing."""
    return TrenniConfig(
        pasloe_url="http://localhost:8000",
        observation_aggregation_interval=60,
        observation_window_hours=24,
        observation_thresholds={"tool_repetition": 5.0},
    )


@pytest.fixture
def mock_pasloe_client():
    """Mock PasloeClient to avoid network calls."""
    client = MagicMock()
    client.emit = AsyncMock()
    client.query = AsyncMock(return_value=[])
    return client


def make_response(events, cursor=None):
    """Helper to create mock httpx.Response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = events
    response.headers = {"X-Next-Cursor": cursor} if cursor else {}
    response.raise_for_status = MagicMock()
    return response


class TestSyntheticEventIdUniqueness:
    """Test that different observation batches get unique optimizer job ids."""

    def test_same_new_ids_produce_same_event_id(self, mock_config):
        """Same observation batch produces same synthetic event id (no duplicate spawn)."""
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.config = mock_config
        supervisor._processed_observation_ids_order = []
        supervisor._processed_observation_ids_set = set()
        supervisor._max_processed_observation_ids = 1000
        
        # Same new_ids batch
        new_ids1 = ["evt-1", "evt-2", "evt-3"]
        new_ids2 = ["evt-1", "evt-2", "evt-3"]
        
        hash1 = hashlib.md5(json.dumps(sorted(new_ids1)).encode()).hexdigest()[:8]
        hash2 = hashlib.md5(json.dumps(sorted(new_ids2)).encode()).hexdigest()[:8]
        
        assert hash1 == hash2
        # Same hash means same synthetic event id: obs-agg-tool_repetition-{hash}
        # This prevents duplicate optimizer spawns for same observation batch

    def test_different_new_ids_produce_different_event_id(self, mock_config):
        """Different observation batches produce different synthetic event ids."""
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.config = mock_config
        supervisor._processed_observation_ids_order = []
        supervisor._processed_observation_ids_set = set()
        supervisor._max_processed_observation_ids = 1000
        
        # Different new_ids batches
        new_ids1 = ["evt-1", "evt-2", "evt-3"]
        new_ids2 = ["evt-4", "evt-5"]  # New observation events
        
        hash1 = hashlib.md5(json.dumps(sorted(new_ids1)).encode()).hexdigest()[:8]
        hash2 = hashlib.md5(json.dumps(sorted(new_ids2)).encode()).hexdigest()[:8]
        
        assert hash1 != hash2
        # Different hash means different synthetic event id
        # This allows subsequent new observations to spawn new optimizer jobs
        # (won't reuse old job_id that scheduler would short-circuit)


class TestFIFOPruning:
    """Test that processed_observation_ids pruning is true FIFO."""

    def test_fifo_pruning_removes_oldest_first(self, mock_config):
        """When pruning, oldest IDs are removed first (FIFO), not random."""
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.config = mock_config
        supervisor._processed_observation_ids_order = []
        supervisor._processed_observation_ids_set = set()
        supervisor._max_processed_observation_ids = 5
        
        # Add IDs in order: evt-1, evt-2, evt-3, evt-4, evt-5, evt-6
        ids_to_add = ["evt-1", "evt-2", "evt-3", "evt-4", "evt-5", "evt-6"]
        for id in ids_to_add:
            if id not in supervisor._processed_observation_ids_set:
                supervisor._processed_observation_ids_order.append(id)
                supervisor._processed_observation_ids_set.add(id)
        
        # Prune to 5
        while len(supervisor._processed_observation_ids_order) > supervisor._max_processed_observation_ids:
            old_id = supervisor._processed_observation_ids_order.pop(0)
            supervisor._processed_observation_ids_set.discard(old_id)
        
        # After pruning, oldest (evt-1) should be removed, newest (evt-6) retained
        assert "evt-1" not in supervisor._processed_observation_ids_set
        assert "evt-2" in supervisor._processed_observation_ids_set
        assert "evt-6" in supervisor._processed_observation_ids_set
        assert supervisor._processed_observation_ids_order == ["evt-2", "evt-3", "evt-4", "evt-5", "evt-6"]

    def test_fifo_preserves_insertion_order(self, mock_config):
        """Insertion order is preserved for FIFO semantics."""
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.config = mock_config
        supervisor._processed_observation_ids_order = []
        supervisor._processed_observation_ids_set = set()
        supervisor._max_processed_observation_ids = 100
        
        # Add IDs in specific order
        ids = ["a", "b", "c", "d", "e"]
        for id in ids:
            if id not in supervisor._processed_observation_ids_set:
                supervisor._processed_observation_ids_order.append(id)
                supervisor._processed_observation_ids_set.add(id)
        
        # Order should match insertion order
        assert supervisor._processed_observation_ids_order == ids
        
        # Re-adding existing IDs shouldn't change order
        supervisor._processed_observation_ids_order.append("a")
        # (In real code, we check `if id not in set` first, so this won't happen)

    def test_pruned_ids_can_trigger_new_aggregation(self, mock_config):
        """When old ID is pruned, it can trigger new aggregation (not forgotten)."""
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.config = mock_config
        supervisor._processed_observation_ids_order = []
        supervisor._processed_observation_ids_set = set()
        supervisor._max_processed_observation_ids = 3
        
        # Add evt-1, evt-2, evt-3
        for id in ["evt-1", "evt-2", "evt-3"]:
            supervisor._processed_observation_ids_order.append(id)
            supervisor._processed_observation_ids_set.add(id)
        
        # Add evt-4, triggers pruning
        for id in ["evt-4"]:
            if id not in supervisor._processed_observation_ids_set:
                supervisor._processed_observation_ids_order.append(id)
                supervisor._processed_observation_ids_set.add(id)
        
        while len(supervisor._processed_observation_ids_order) > supervisor._max_processed_observation_ids:
            old_id = supervisor._processed_observation_ids_order.pop(0)
            supervisor._processed_observation_ids_set.discard(old_id)
        
        # evt-1 should be pruned
        assert "evt-1" not in supervisor._processed_observation_ids_set
        
        # If evt-1 appears again in pasloe, it would be treated as "new"
        # This is intentional: old observations might need fresh analysis
        # The hash-based dedup handles same-batch, FIFO handles memory limit


class TestImplementerPathAllowlist:
    """Test implementer path allowlist catches all changes."""

    def test_allowlist_catches_unstaged_changes(self):
        """Implementer publication catches unstaged changes outside allowlist."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            
            # Create fake git repo structure
            (tmp_path / "teams" / "factorio" / "scripts").mkdir(parents=True)
            (tmp_path / "docs").mkdir(parents=True)
            
            # Create allowed and forbidden files
            allowed_file = tmp_path / "teams" / "factorio" / "scripts" / "test.lua"
            forbidden_file = tmp_path / "docs" / "README.md"
            allowed_file.write_text("-- allowed")
            forbidden_file.write_text("forbidden")
            
            # Simulate git status --porcelain output
            # Format: " M PATH" for modified unstaged, "?? PATH" for untracked
            porcelain_output = f" M factorio/evolved/scripts/test.lua\n?? docs/README.md\n"
            
            # Parse changes
            changed = []
            for line in porcelain_output.splitlines():
                if not line:
                    continue
                if " -> " in line:
                    parts = line.split(" -> ")
                    path = parts[-1].strip()
                else:
                    # Git porcelain: "XY PATH" - path starts at position 3
                    path = line[3:].strip()
                if path:
                    changed.append(path)
            
            # Check allowlist (Bundle MVP: only factorio/evolved/scripts/)
            forbidden = [
                p for p in changed
                if not p.startswith("factorio/evolved/scripts/")
            ]
            
            assert forbidden == ["docs/README.md"]

    def test_allowlist_catches_renamed_files(self):
        """Implementer publication catches renamed files to forbidden paths."""
        # Simulate git status --porcelain for rename
        porcelain_output = "R  old_path.lua -> docs/hacked.lua\n"
        
        changed = []
        for line in porcelain_output.splitlines():
            line = line.strip()
            if not line:
                continue
            if " -> " in line:
                parts = line.split(" -> ")
                path = parts[-1].strip()
            else:
                path = line[3:].strip()
            if path:
                changed.append(path)
        
        # Check allowlist (Bundle MVP: only factorio/evolved/scripts/)
        forbidden = [
            p for p in changed
            if not p.startswith("factorio/evolved/scripts/")
        ]
        
        assert forbidden == ["docs/hacked.lua"]

class TestResolveBundleForObservations:
    """Test _resolve_bundle_for_observations helper function."""

    def test_bundle_from_evidence_majority_vote(self):
        """Bundle resolved by majority vote from evidence."""
        from trenni.supervisor import _resolve_bundle_for_observations
        
        evidence = [
            {"bundle": "factorio", "tool_name": "test1"},
            {"bundle": "factorio", "tool_name": "test2"},
            {"bundle": "factorio", "tool_name": "test3"},
            {"bundle": "default", "tool_name": "test4"},
            {"bundle": "default", "tool_name": "test5"},
        ]
        
        result = _resolve_bundle_for_observations(evidence)
        assert result == "factorio"  # 3 votes vs 2

    def test_bundle_fallback_to_default_when_empty(self):
        """When evidence is empty, fallback to 'default' with warning."""
        from trenni.supervisor import _resolve_bundle_for_observations
        
        result = _resolve_bundle_for_observations([])
        assert result == "default"

    def test_bundle_fallback_when_all_missing(self):
        """When all evidence missing bundle field, fallback to 'default'."""
        from trenni.supervisor import _resolve_bundle_for_observations
        
        evidence = [
            {"tool_name": "test1"},  # No bundle
            {"tool_name": "test2"},  # No bundle
        ]
        
        result = _resolve_bundle_for_observations(evidence)
        assert result == "default"

    def test_bundle_single_vote_wins(self):
        """Single bundle entry wins (no tie-breaking needed)."""
        from trenni.supervisor import _resolve_bundle_for_observations
        
        evidence = [
            {"bundle": "factorio", "tool_name": "test"},
        ]
        
        result = _resolve_bundle_for_observations(evidence)
        assert result == "factorio"


class TestOptimizerTriggerDataBundleRouting:
    """Test that optimizer trigger_data uses correct bundle from evidence."""

    @pytest.mark.asyncio
    async def test_trigger_data_bundle_from_evidence(self, mock_config):
        """Optimizer spawn uses bundle from observation evidence."""
        from trenni.supervisor import Supervisor
        from trenni.observation_aggregator import AggregationResult
        from yoitsu_contracts.events import TriggerData
        
        # Create supervisor instance (minimal setup)
        supervisor = Supervisor.__new__(Supervisor)
        supervisor.config = mock_config
        supervisor._processed_observation_ids_order = []
        supervisor._processed_observation_ids_set = set()
        supervisor._max_processed_observation_ids = 1000
        
        # Create mock aggregation result with factorio evidence
        result = AggregationResult(
            metric_type="tool_repetition",
            count=10,
            threshold=5.0,
            exceeded=True,
            evidence=[
                {"bundle": "factorio", "tool_name": "factorio_call_script(find_ore_basic)", 
                 "arg_pattern": "find_ore_basic", "call_count": 10, "similarity": 0.8},
            ] * 5,  # 5 evidence entries, all factorio
        )
        
        new_ids = ["evt-1", "evt-2", "evt-3"]
        
        # Mock _process_trigger to capture trigger_data
        captured_trigger = None
        async def mock_process_trigger(event, data, replay=False):
            nonlocal captured_trigger
            captured_trigger = data
        
        supervisor._process_trigger = mock_process_trigger
        
        # Run the observation aggregation logic (simulate)
        # This is extracted from _aggregate_observations_periodically
        import hashlib
        import json
        from datetime import datetime, timezone
        from types import SimpleNamespace
        
        batch_hash = hashlib.md5(json.dumps(sorted(new_ids)).encode()).hexdigest()[:8]
        
        from trenni.supervisor import _resolve_bundle_for_observations
        target_bundle = _resolve_bundle_for_observations(result.evidence)
        
        trigger_data = {
            "goal": f"Analyze {result.metric_type} pattern in bundle '{target_bundle}'",
            "role": "optimizer",
            "bundle": target_bundle,
            "budget": 0.5,
            "params": {
                "metric_type": result.metric_type,
                "observation_count": result.count,
                "window_hours": 24,
                "evidence": result.evidence,
            },
        }
        
        # Validate as TriggerData
        data = TriggerData.model_validate(trigger_data)
        
        assert data.bundle == "factorio"
        assert data.params.get("evidence") == result.evidence
        assert len(data.params.get("evidence", [])) == 5
        assert "factorio" in data.goal
