"""Closed-loop episode runner: Environment -> Observation -> Controller -> Action.

Timing model
------------
* Physics runs in ticks of 1/60 s of controller (wall-clock) time. Each tick
  advances the world by ``world_speed_scale / 60`` world seconds.
* Decision slots occur every ``60 / decision_hz`` ticks. At a slot, if fewer
  than ``max_inflight`` requests are outstanding, the current observation is
  captured and handed to ``controller.request``. Otherwise the slot waits
  and is served (with a fresh observation) on the tick a request frees up
  (counted as *delayed*); if the next slot arrives first, the waiting one
  is *missed* (never served). So ``decision_hz`` is a maximum rate.
* With several requests in flight, answers can arrive out of order. At each
  tick the newest ready answer is applied; an answer to an older request
  than the one already applied (or than another ready one) is *superseded*
  and never applied. Stale information never overwrites fresher.
* A decision with measured wall-clock latency L (from the moment its
  observation was handed over until the runner received the decision,
  timed by the runner's own clock, never self-reported) takes
  effect at tick ``request_tick + max(1, ceil(L / tick))``. Until then the
  previous action keeps being applied. The world is never paused.
* With ``decision_deadline_ms`` set, a decision whose latency exceeds the
  deadline is discarded; the previous action simply continues.

Pacing
------
``realtime=True`` paces ticks to the wall clock (interactive play).
``realtime=False`` (headless) runs as fast as possible *except* while a
request is outstanding: then the simulation never runs ahead of the wall
clock since that request was issued, so "has the answer arrived by tick j?"
is always decided by real elapsed time. Fast controllers therefore run
faster than real time, while slow or remote controllers are charged exactly
what they cost. Both modes apply identical rules. In realtime mode a threaded controller's
answer is only noticed at the next UI frame, which can add up to one frame
(~16 ms) of measured latency; headless mode is the measurement mode.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from arena import physics
from arena.action import Action
from arena.environment import Environment
from arena.recorder import NullRecorder, Recorder
from arena.stats import mean, percentile
from controllers.base import Controller, Decision

TICK = physics.TICK_SECONDS


@dataclass
class _Pending:
    id: int
    tick: int
    wall: float
    decision: Optional[Decision] = None
    latency_s: Optional[float] = None
    effective_tick: Optional[int] = None


@dataclass
class DecisionStats:
    requests: int = 0
    applied: int = 0
    missed_slots: int = 0  # slots that never got their own request
    delayed_slots: int = 0  # slots served late because all requests were busy
    late_dropped: int = 0
    failed: int = 0
    superseded: int = 0
    latencies_s: list[float] = field(default_factory=list)
    delay_ticks: list[int] = field(default_factory=list)  # applied: effective - request tick


class EpisodeRunner:
    def __init__(
        self,
        env: Environment,
        controller: Controller,
        recorder: Optional[Recorder] = None,
        realtime: bool = False,
    ):
        self.env = env
        self.controller = controller
        self.recorder: Recorder = recorder or NullRecorder()
        self.realtime = realtime
        self.started = False

    # ----------------------------------------------------------------- setup
    def start(self) -> None:
        env, cfg = self.env, self.env.config
        self.current_action = Action.STAY
        self.last_latency_s: Optional[float] = None
        self._applied_request_tick: Optional[int] = None
        self.stats = DecisionStats()
        self.action_changes: list[tuple[int, str]] = [(0, Action.STAY.value)]
        self._pendings: dict[int, _Pending] = {}
        self._last_applied_id = -1
        self._max_inflight = cfg.max_inflight
        self._next_id = 0
        self._slot_period = physics.PHYSICS_HZ / cfg.decision_hz
        self._next_slot = 0.0
        self._slot_owed = False
        self._slot_owed_since: Optional[int] = None
        self._deadline_s = None if cfg.decision_deadline_ms is None else cfg.decision_deadline_ms / 1000.0

        info = env.public_info()
        self.controller.reset(info)
        self.recorder.event(
            {
                "type": "episode_start",
                "controller": self.controller.name,
                "seed": env.seed,
                "config": cfg.to_dict(),
                "info": info.to_dict(),
                "realtime": self.realtime,
            }
        )
        self.wall_origin = time.perf_counter()
        self.started = True

    # ------------------------------------------------------------------ loop
    def run(self) -> dict[str, Any]:
        if not self.started:
            self.start()
        while not self.env.done:
            if self.realtime:
                target = self.wall_origin + self.env.tick * TICK
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            self.step_tick()
        return self.finish()

    def advance_realtime(self, max_ticks: int = 30) -> int:
        """Run every tick that is due by the wall clock (for a UI frame loop)."""
        n = 0
        due = int((time.perf_counter() - self.wall_origin) / TICK)
        while not self.env.done and self.env.tick < due and n < max_ticks:
            self.step_tick()
            n += 1
        if self.env.tick < due and n >= max_ticks:
            # Rendering fell behind; re-anchor rather than spiral. The world
            # still only ever advances tick by tick with the same rules.
            self.wall_origin = time.perf_counter() - self.env.tick * TICK
        return n

    def step_tick(self) -> None:
        env = self.env
        if env.done:
            return
        k = env.tick

        if self._pendings:
            self._resolve(k)

        if k + 1e-9 >= self._next_slot:
            while self._next_slot <= k + 1e-9:
                self._next_slot += self._slot_period
            if len(self._pendings) >= self._max_inflight:
                if self._slot_owed:  # an earlier owed slot is superseded by this one: lost
                    self.stats.missed_slots += 1
                    self.recorder.event({"type": "missed_slot", "tick": k, "pending_ids": list(self._pendings)})
                self._slot_owed_since = k
            self._slot_owed = True
        if self._slot_owed and len(self._pendings) < self._max_inflight:
            if self._slot_owed_since is not None:  # served late, but served
                self.stats.delayed_slots += 1
            self._slot_owed = False
            self._slot_owed_since = None
            self._issue_request(k)

        events = env.step(self.current_action)

        if events.targets_collected:
            self.recorder.event(
                {"type": "target_collected", "tick": k, "t": env.world_time, "score": env.score}
            )
        for oid in events.spawned_obstacle_ids:
            self.recorder.event({"type": "obstacle_spawned", "tick": k, "id": oid})
        if events.collision_obstacle_id is not None:
            self.recorder.event(
                {
                    "type": "collision",
                    "tick": k,
                    "t": env.world_time,
                    "obstacle_id": events.collision_obstacle_id,
                    "action": self.current_action.value,
                }
            )

    # ------------------------------------------------------------- decisions
    def _control_feedback(self) -> dict[str, Any]:
        lat = self.last_latency_s
        return {
            "applied_request_tick": self._applied_request_tick,
            "applied_latency_s": lat,
            "applied_latency_world_s": None if lat is None else lat * self.env.config.world_speed_scale,
        }

    def _issue_request(self, k: int) -> None:
        obs = self.env.observe(self._control_feedback())
        rid = self._next_id
        self._next_id += 1
        self.stats.requests += 1
        self.recorder.event({"type": "request", "id": rid, "tick": k, "observation": obs.to_dict()})
        wall = time.perf_counter()
        self._pendings[rid] = _Pending(rid, k, wall)
        self.controller.request(obs, rid)
        # Inline (synchronous) controllers are done by now; time them exactly.
        self._receive_all()

    def _receive_all(self) -> None:
        while True:
            d = self.controller.poll()
            if d is None:
                return
            received = time.perf_counter()  # runner's clock is the only authority
            rid = d.request_id
            if rid is None:  # controller did not tag it: oldest unanswered request
                rid = next((p.id for p in self._pendings.values() if p.decision is None), None)
            pr = self._pendings.get(rid) if rid is not None else None
            if pr is None or pr.decision is not None:
                self.recorder.event({"type": "unmatched_decision", "request_id": d.request_id})
                continue
            pr.decision = d
            pr.latency_s = (received - pr.wall) + max(0.0, d.extra_latency_s)
            pr.effective_tick = pr.tick + max(1, math.ceil(pr.latency_s / TICK - 1e-9))
            self.stats.latencies_s.append(pr.latency_s)

    def _resolve(self, j: int) -> None:
        # 1. Establish the truth: which answers exist by the time of tick j.
        while True:
            checked_at = time.perf_counter()
            self._receive_all()
            waits = [
                (j - p.tick) * TICK - (checked_at - p.wall)
                for p in self._pendings.values()
                if p.decision is None and checked_at - p.wall < (j - p.tick) * TICK
            ]
            if not waits:
                break
            # Simulation is ahead of the wall clock for some request: wait.
            time.sleep(min(min(waits), 0.0005))

        # 2. Deadline drops, failures, superseded answers; apply the newest.
        ready: list[_Pending] = []
        for pr in sorted(self._pendings.values(), key=lambda p: p.id):
            if pr.decision is None:
                continue
            assert pr.latency_s is not None and pr.effective_tick is not None
            if self._deadline_s is not None and pr.latency_s > self._deadline_s:
                self._close(pr, "decision_dropped", j, reason="deadline")
                self.stats.late_dropped += 1
            elif j >= pr.effective_tick:
                if pr.decision.action is None:
                    self._close(pr, "decision_failed", j)
                    self.stats.failed += 1
                elif pr.id < self._last_applied_id:
                    self._close(pr, "decision_superseded", j)
                    self.stats.superseded += 1
                else:
                    ready.append(pr)
        if not ready:
            return
        for old in ready[:-1]:  # several ready at once: only the newest counts
            self._close(old, "decision_superseded", j)
            self.stats.superseded += 1
        self._apply(ready[-1], j)

    def _close(self, pr: _Pending, kind: str, j: int, **extra: Any) -> None:
        d = pr.decision
        assert d is not None and pr.latency_s is not None
        self.recorder.event(
            {
                "type": kind,
                "id": pr.id,
                "request_tick": pr.tick,
                "tick": j,
                "action": None if d.action is None else d.action.value,
                "latency_ms": pr.latency_s * 1000,
                **({"error": d.error} if d.error else {}),
                **({"meta": d.meta} if d.meta else {}),
                **extra,
            }
        )
        del self._pendings[pr.id]

    def _apply(self, pr: _Pending, j: int) -> None:
        d = pr.decision
        assert d is not None and d.action is not None and pr.latency_s is not None
        action = d.action
        self.stats.applied += 1
        self.stats.delay_ticks.append(j - pr.tick)
        self.last_latency_s = pr.latency_s
        self._applied_request_tick = pr.tick
        self._last_applied_id = pr.id
        if action != self.current_action:
            self.action_changes.append((j, action.value))
        self.current_action = action
        self.recorder.event(
            {
                "type": "decision",
                "id": pr.id,
                "request_tick": pr.tick,
                "applied_tick": j,
                "action": action.value,
                "latency_ms": pr.latency_s * 1000,
                **({"meta": d.meta} if d.meta else {}),
            }
        )
        del self._pendings[pr.id]

    # ---------------------------------------------------------------- result
    def finish(self) -> dict[str, Any]:
        result = self.result()
        self.recorder.event({"type": "episode_end", "result": result})
        self.recorder.write_result(result)
        return result

    def result(self) -> dict[str, Any]:
        env, cfg, st = self.env, self.env.config, self.stats
        o = env.outcome
        lat_ms = [x * 1000 for x in st.latencies_s]
        survival = env.world_time
        return {
            "controller": self.controller.name,
            "seed": env.seed,
            "done": o.done,
            "success": o.success,
            "reason": o.reason,
            "survival_time": survival,
            "targets_collected": env.score,
            "distance_travelled": env.distance_travelled,
            "average_speed": env.distance_travelled / survival if survival > 0 else 0.0,
            "collision_time": o.time if o.reason == "collision" else None,
            "collision_obstacle_id": o.collision_obstacle_id,
            "ticks": env.tick,
            "decision_count": st.applied,
            "request_count": st.requests,
            "missed_slots": st.missed_slots,
            "delayed_slots": st.delayed_slots,
            "late_dropped": st.late_dropped,
            "failed_decisions": st.failed,
            "superseded_decisions": st.superseded,
            "mean_decision_latency_ms": mean(lat_ms),
            "p50_latency_ms": percentile(lat_ms, 50),
            "p95_latency_ms": percentile(lat_ms, 95),
            "mean_delay_ticks": mean(st.delay_ticks),
            "world_speed_scale": cfg.world_speed_scale,
            "decision_hz": cfg.decision_hz,
            "max_inflight": cfg.max_inflight,
            "obstacle_count": cfg.obstacle_count,
            "final_obstacle_count": len(env.obstacles),
            "action_changes": self.action_changes,
        }
