"""Observation analyzers for post-job analysis (ADR-0017).

Analyzers examine job events and produce observation data.
Trenni calls analyzers after job reaches terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any

# Event is dict-like in supervisor context
# from yoitsu_contracts.events import Event  -- Event is from pasloe_client, not exported

# Use Any for Event type (it's a SimpleNamespace or dict)

@runtime_checkable
class ObservationAnalyzer(Protocol):
    """Analyzer protocol per ADR-0017.
    
    Analyzers examine job events and return observation data.
    Trenni emits observation.* events from the returned data.
    """
    name: str  # Analyzer identifier (e.g., "tool_repetition", "budget_variance")
    
    def analyze(self, job_events: list[Event], job_id: str, task_id: str, role: str, bundle: str) -> list[dict[str, Any]]:
        """Analyze job events and return observation data.
        
        Args:
            job_events: List of events from this job
            job_id: Job identifier
            task_id: Task identifier  
            role: Role type
            bundle: Bundle name
            
        Returns:
            List of observation data dicts (will be emitted as observation.* events)
        """
        ...


class ToolRepetitionAnalyzer:
    """Analyze tool call patterns for repetition.
    
    Detects when a tool is called multiple times with similar arguments.
    This indicates an abstraction opportunity for bundle evolution.
    """
    name = "tool_repetition"
    
    def analyze(self, job_events: list[Event], job_id: str, task_id: str, role: str, bundle: str) -> list[dict[str, Any]]:
        """Analyze tool call history from job.completed event.
        
        Args:
            job_events: Events from this job (includes job.completed with tool_call_history)
            
        Returns:
            List of tool repetition observations
        """
        # Find job.completed event with tool_call_history
        # PasloeEvent is a dataclass with .type and .data attributes
        completed_event = None
        for evt in job_events:
            # Handle both dict (replay) and PasloeEvent dataclass (realtime)
            evt_type = evt.type if hasattr(evt, 'type') else evt.get('type', '')
            if evt_type == "agent.job.completed":
                completed_event = evt
                break
        
        if not completed_event:
            return []
        
        # Handle both dict and PasloeEvent dataclass
        if hasattr(completed_event, 'data'):
            tool_call_history = completed_event.data.get('tool_call_history', [])
        else:
            tool_call_history = completed_event.get('data', {}).get('tool_call_history', [])
        if not tool_call_history:
            return []
        
        # Detect repetition patterns
        # Import is inside try block because palimpsest is not available in trenni container
        try:
            from palimpsest.runtime.tool_pattern import detect_repetition
            repetitions = detect_repetition(tool_call_history)
        except Exception:
            # If detect_repetition not available, use simple detection
            repetitions = self._simple_detect(tool_call_history)
        
        results = []
        for r in repetitions:
            results.append({
                "tool_name": r.tool_name,
                "call_count": r.call_count,
                "arg_pattern": r.arg_pattern,
                "similarity": r.similarity,
            })
        
        return results
    
    def _simple_detect(self, tool_call_history: list[dict]) -> list[Any]:
        """Simple repetition detection fallback.
        
        Detects tools called more than threshold times.
        """
        from dataclasses import dataclass
        
        @dataclass
        class Repetition:
            tool_name: str
            call_count: int
            arg_pattern: str
            similarity: float
        
        threshold = 5
        tool_counts: dict[str, int] = {}
        
        for call in tool_call_history:
            tool_name = call.get("name", call.get("tool_name", ""))
            if tool_name:
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        results = []
        for tool_name, count in tool_counts.items():
            if count >= threshold:
                results.append(Repetition(
                    tool_name=tool_name,
                    call_count=count,
                    arg_pattern="",  # Simple detection doesn't analyze args
                    similarity=0.0,
                ))
        
        return results


class BudgetVarianceAnalyzer:
    """Analyze budget prediction accuracy.
    
    Compares estimated budget with actual cost.
    """
    name = "budget_variance"
    
    def analyze(self, job_events: list[Event], job_id: str, task_id: str, role: str, bundle: str) -> list[dict[str, Any]]:
        """Analyze budget variance from spawn_defaults and job.completed.
        
        Note: This analyzer needs access to spawn_defaults which is in Supervisor.
        In practice, budget_variance is handled directly in supervisor for now.
        
        Returns:
            Budget variance observation (if applicable)
        """
        # Budget variance requires spawn_defaults (estimated budget)
        # which is not in job_events. This analyzer is informational only.
        # Real budget_variance emission stays in supervisor._emit_budget_variance
        return []


# === Analyzer Registry ===

BUILTIN_ANALYZERS: dict[str, ObservationAnalyzer] = {
    "tool_repetition": ToolRepetitionAnalyzer(),
    "budget_variance": BudgetVarianceAnalyzer(),  # Placeholder
}


def get_analyzer(name: str) -> ObservationAnalyzer | None:
    """Get an analyzer by name from registry."""
    return BUILTIN_ANALYZERS.get(name)