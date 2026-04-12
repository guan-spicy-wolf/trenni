"""Observation event aggregator for autonomous optimization loop.

Periodically queries pasloe for observation.* events, aggregates by metric_type,
and spawns optimizer tasks when thresholds are exceeded.

Per ADR-0010 extension for Factorio Tool Evolution MVP.

Key semantics:
- window_count: count of ALL observation events in the window (for threshold check)
- new_ids: IDs of events NOT YET processed (for dedup, spawn control, and hash)

This separation ensures:
- Threshold is based on total window activity (not just "new this round")
- Dedup prevents duplicate spawns for same batch of new events
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import httpx
import logging

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    metric_type: str
    count: int  # Window-wide count (ALL events in window, for threshold)
    threshold: float
    exceeded: bool
    evidence: list[dict]  # NEW: list of observation event payloads (latest N)
    role: str | None = None


async def aggregate_observations(
    pasloe_url: str,
    window_hours: int,
    thresholds: dict[str, float],
    api_key: str = "",
    processed_ids: set[str] | None = None,
) -> tuple[list[AggregationResult], dict[str, list[str]]]:
    """Query pasloe for observation.* events in window, aggregate by metric_type.
    
    Returns TWO things with distinct semantics:
    
    1. results[].count: Window-wide total (ALL observation events in window)
       - Used for threshold check ("5 occurrences in 24h window")
       - NOT affected by processed_ids filtering
       
    2. metric_new_ids: Dict mapping metric_type to list of unprocessed event IDs
       - Per-metric dedup (each metric gets its own batch_members)
       - Used for spawn decision (only spawn if metric has new events)
    
    This separation is critical for correct threshold behavior:
    - If threshold=5, and events arrive: round1=3, round2=3
    - round1: count=3, new_ids=[e1,e2,e3], not exceeded
    - round2: count=6 (window total), new_ids=[e4,e5,e6], exceeded → spawn optimizer
    
    Args:
        pasloe_url: Pasloe API base URL
        window_hours: How many hours back to query
        thresholds: Threshold dict (metric_type -> threshold value)
        api_key: Pasloe API key for authentication
        processed_ids: Set of already-processed observation event IDs
        
    Returns:
        (results, new_ids) tuple:
        - results: AggregationResult with window-wide counts
        - new_ids: List of event IDs not yet processed
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    all_events = []  # ALL observation events in window (for count)
    metric_event_ids: dict[str, list[str]] = {}  # metric_type -> event IDs
    cursor = None
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    async with httpx.AsyncClient() as client:
        while True:
            params = {"since": cutoff.isoformat(), "limit": 1000, "order": "asc"}
            if cursor:
                params["cursor"] = cursor
            
            resp = await client.get(f"{pasloe_url}/events", params=params, headers=headers)
            resp.raise_for_status()
            batch = resp.json()
            
            # Collect ALL observation.* events in window (no filtering yet)
            for evt in batch:
                if evt.get("type", "").startswith("observation."):
                    all_events.append(evt)
                    event_id = evt.get("id", "")
                    if event_id:
                        event_type = evt.get("type", "")
                        metric = event_type.split(".", 1)[1] if "." in event_type else ""
                        if metric:
                            metric_event_ids.setdefault(metric, []).append(event_id)
            
            cursor = resp.headers.get("X-Next-Cursor")
            if not cursor:
                break
    
    # Compute window-wide counts (ALL events, NOT filtered by processed_ids)
    # Per tool_repetition threshold semantics: count per tool_name, not per metric_type
    # Each tool is tracked independently - threshold=5 means "5 jobs called this tool repeatedly"
    counts: dict[str, int] = {}  # metric_type -> count (for non-tool metrics)
    tool_counts: dict[str, dict[str, int]] = {}  # metric_type -> tool_name -> count
    
    for evt in all_events:
        event_type = evt.get("type", "")
        if not event_type.startswith("observation."):
            continue
        metric = event_type.split(".", 1)[1] if "." in event_type else ""
        
        # For tool_repetition, count per tool_name
        if metric == "tool_repetition":
            tool_name = evt.get("data", {}).get("tool_name", "unknown")
            if metric not in tool_counts:
                tool_counts[metric] = {}
            tool_counts[metric][tool_name] = tool_counts[metric].get(tool_name, 0) + 1
        else:
            # For other metrics, count per metric_type
            counts[metric] = counts.get(metric, 0) + 1
    
    # Build results - one per tool for tool_repetition, one per metric for others
    results = []
    
    # Handle tool_repetition: emit one result per tool
    if "tool_repetition" in tool_counts:
        threshold = thresholds.get("tool_repetition", float("inf"))
        for tool_name, count in tool_counts["tool_repetition"].items():
            # Extract evidence for this specific tool
            tool_events = [
                evt for evt in all_events
                if evt.get("type", "") == "observation.tool_repetition"
                and evt.get("data", {}).get("tool_name", "") == tool_name
            ]
            tool_events.sort(key=lambda e: e.get("ts", ""), reverse=True)
            evidence = [
                {
                    "role": evt.get("data", {}).get("role", ""),
                    "bundle": evt.get("data", {}).get("bundle", ""),
                    "tool_name": evt.get("data", {}).get("tool_name", ""),
                    "call_count": evt.get("data", {}).get("call_count", 0),
                    "arg_pattern": evt.get("data", {}).get("arg_pattern", ""),
                    "similarity": evt.get("data", {}).get("similarity", 0.0),
                }
                for evt in tool_events[:5]
            ]
            # metric_type includes tool_name for aggregation result
            results.append(AggregationResult(
                metric_type=f"tool_repetition:{tool_name}",  # Include tool_name for clarity
                count=count,
                threshold=threshold,
                exceeded=(count >= threshold),
                evidence=evidence,
            ))
    
    # Handle other metrics (budget_variance, etc.)
    for metric, count in counts.items():
        if metric == "tool_repetition":
            continue  # Already handled above
        threshold = thresholds.get(metric, float("inf"))
        
        # Extract evidence events for this metric (latest 5)
        metric_events = [
            evt for evt in all_events
            if evt.get("type", "").endswith(metric)
        ]
        metric_events.sort(key=lambda e: e.get("ts", ""), reverse=True)
        evidence = [
            {
                "role": evt.get("data", {}).get("role", ""),
                "bundle": evt.get("data", {}).get("bundle", ""),
            }
            for evt in metric_events[:5]
        ]
        
        results.append(AggregationResult(
            metric_type=metric,
            count=count,
            threshold=threshold,
            exceeded=(count >= threshold),
            evidence=evidence,
        ))
    
    # Compute per-metric new_ids (events NOT yet processed)
    # For tool_repetition, include tool_name in the key
    processed_set = processed_ids or set()
    metric_new_ids: dict[str, list[str]] = {}
    
    for evt in all_events:
        event_id = evt.get("id", "")
        if not event_id or event_id in processed_set:
            continue
        event_type = evt.get("type", "")
        if not event_type.startswith("observation."):
            continue
        metric = event_type.split(".", 1)[1] if "." in event_type else ""
        
        # For tool_repetition, use metric:tool_name as key
        if metric == "tool_repetition":
            tool_name = evt.get("data", {}).get("tool_name", "unknown")
            key = f"tool_repetition:{tool_name}"
        else:
            key = metric
        
        metric_new_ids.setdefault(key, []).append(event_id)
    
    return results, metric_new_ids