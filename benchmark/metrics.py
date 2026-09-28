"""Aggregation of raw per-episode results. No magic composite score."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional, Sequence

from arena.stats import mean, percentile


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    succ = sum(1 for r in results if r["success"])
    lat = [r["mean_decision_latency_ms"] for r in results if r["mean_decision_latency_ms"] is not None]
    p95 = [r["p95_latency_ms"] for r in results if r["p95_latency_ms"] is not None]
    lo, hi = wilson_interval(succ, n)
    return {
        "episodes": n,
        "successes": succ,
        "success_rate": succ / n if n else 0.0,
        "success_rate_ci95": [lo, hi],
        "mean_targets": mean([r["targets_collected"] for r in results]),
        "mean_survival_time": mean([r["survival_time"] for r in results]),
        "mean_distance": mean([r["distance_travelled"] for r in results]),
        "mean_latency_ms": mean(lat),
        "p95_latency_ms": percentile(p95, 50) if p95 else None,  # median of per-episode p95
        "mean_missed_slots": mean([r["missed_slots"] for r in results]),
        "mean_delayed_slots": mean([r.get("delayed_slots", 0) for r in results]),
        "failure_reasons": dict(Counter(r["reason"] for r in results if not r["success"])),
    }


def threshold_level(curve: Sequence[tuple[float, float]], p: float) -> Optional[float]:
    """Largest difficulty level at which the success rate is still >= p.

    ``curve`` is [(level, success_rate), ...]. Linear interpolation between
    the last level above p and the first level below it. This is the hook for
    D95 / D50 / D10 style failure-frontier summaries; a proper psychometric
    fit can replace it later.
    """
    pts = sorted(curve)
    if not pts or pts[0][1] < p:
        return None
    for (l0, s0), (l1, s1) in zip(pts, pts[1:]):
        if s0 >= p > s1:
            return l0 + (s0 - p) / (s0 - s1) * (l1 - l0)
    return pts[-1][0]  # never dropped below p in the tested range
