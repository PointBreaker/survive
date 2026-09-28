"""Decision inspector: turns the runner's event stream into a view model.

It is fed the exact events the runner logs (live, through ``TeeRecorder``;
offline, from ``events.jsonl``). The live GUI and the replay viewer show
the same thing, and both are pure observers: nothing here reaches back into
the controller or the environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from arena.recorder import Recorder
from arena.stats import mean, percentile

TICK_MS = 1000.0 / 60.0


@dataclass
class DecisionRecord:
    id: int
    request_tick: int
    resolve_tick: int
    status: str  # applied | failed | dropped
    action: Optional[str]
    latency_ms: float
    meta: Optional[dict] = None
    error: Optional[str] = None

    @property
    def delay_ticks(self) -> int:
        return self.resolve_tick - self.request_tick


def event_tick(ev: dict[str, Any]) -> float:
    """The physics tick at which an event happened (for offline seeking)."""
    t = ev.get("type")
    if t == "episode_start":
        return -1
    if t == "episode_end":
        return math.inf
    if t == "decision":
        return ev["applied_tick"]
    return ev.get("tick", -1)


@dataclass
class Inspector:
    history: int = 400
    requests: dict[int, dict[str, Any]] = field(default_factory=dict)  # id -> request event
    records: list[DecisionRecord] = field(default_factory=list)
    inflight_id: Optional[int] = None
    missed_slots: int = 0
    targets: list[int] = field(default_factory=list)  # ticks
    collision: Optional[dict[str, Any]] = None
    start: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None

    def feed(self, ev: dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "episode_start":
            self.start = ev
        elif t == "request":
            self.requests[ev["id"]] = ev
            self.inflight_id = ev["id"]
            if len(self.requests) > self.history:
                for k in sorted(self.requests)[: len(self.requests) - self.history]:
                    del self.requests[k]
        elif t in ("decision", "decision_failed", "decision_dropped"):
            req = self.requests.get(ev["id"], {})
            status = {"decision": "applied", "decision_failed": "failed", "decision_dropped": "dropped"}[t]
            self.records.append(
                DecisionRecord(
                    id=ev["id"],
                    request_tick=ev.get("request_tick", req.get("tick", 0)),
                    resolve_tick=ev.get("applied_tick", ev.get("tick", 0)),
                    status=status,
                    action=ev.get("action"),
                    latency_ms=ev.get("latency_ms", 0.0),
                    meta=ev.get("meta"),
                    error=ev.get("error"),
                )
            )
            if self.inflight_id == ev["id"]:
                self.inflight_id = None
        elif t == "missed_slot":
            self.missed_slots += 1
        elif t == "target_collected":
            self.targets.append(ev["tick"])
        elif t == "collision":
            self.collision = ev
        elif t == "episode_end":
            self.result = ev.get("result")

    # ------------------------------------------------------------- queries
    def inflight(self) -> Optional[dict[str, Any]]:
        return self.requests.get(self.inflight_id) if self.inflight_id is not None else None

    def last_applied(self) -> Optional[DecisionRecord]:
        for r in reversed(self.records):
            if r.status == "applied":
                return r
        return None

    def observation_behind(self, record: Optional[DecisionRecord]) -> Optional[dict[str, Any]]:
        if record is None:
            return None
        req = self.requests.get(record.id)
        return req.get("observation") if req else None

    def stats(self) -> dict[str, Any]:
        lat = [r.latency_ms for r in self.records]
        return {
            "requests": len(self.requests) if len(self.requests) < self.history else None,
            "applied": sum(r.status == "applied" for r in self.records),
            "failed": sum(r.status == "failed" for r in self.records),
            "dropped": sum(r.status == "dropped" for r in self.records),
            "missed": self.missed_slots,
            "mean_ms": mean(lat),
            "p95_ms": percentile(lat, 95),
            "recent_ms": [r.latency_ms for r in self.records[-60:]],
        }


class TeeRecorder(Recorder):
    """Forward every event to ``inner`` and to an Inspector."""

    def __init__(self, inner: Recorder, inspector: Inspector):
        self.inner = inner
        self.inspector = inspector

    def event(self, data: dict[str, Any]) -> None:
        self.inspector.feed(data)
        self.inner.event(data)

    def write_result(self, result: dict[str, Any]) -> None:
        self.inner.write_result(result)

    def close(self) -> None:
        self.inner.close()


def timeline_data(ins: Inspector, total_ticks: int, current: int, label: str = "") -> dict[str, Any]:
    return {
        "total_ticks": total_ticks,
        "current": current,
        "requests": [r["tick"] for r in ins.requests.values()],
        "applied": [r.resolve_tick for r in ins.records if r.status == "applied"],
        "failed": [r.resolve_tick for r in ins.records if r.status != "applied"],
        "targets": list(ins.targets),
        "collision_tick": ins.collision["tick"] if ins.collision else None,
        "label": label,
    }
