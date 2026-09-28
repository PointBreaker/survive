"""Closed-loop episode runner: Environment -> Observation -> Controller -> Action.

Timing model
------------
* Physics runs in ticks of 1/60 s of controller (wall-clock) time. Each tick
  advances the world by ``world_speed_scale / 60`` world seconds.
* Decision slots occur every ``60 / decision_hz`` ticks. At a slot, if the
  controller is idle, the current observation is captured and handed to
  ``controller.request``. A controller has at most one request in flight.
  If it is still busy when a slot comes, the slot is counted as missed and
  served (with a fresh observation) on the tick the controller frees up.
  So ``decision_hz`` is a maximum rate; a slow controller simply decides
  as often as its own latency allows.
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
    missed_slots: int = 0
    late_dropped: int = 0
    failed: int = 0
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
        self.stats = DecisionStats()
        self.action_changes: list[tuple[int, str]] = [(0, Action.STAY.value)]
        self._pending: Optional[_Pending] = None
        self._next_id = 0
        self._slot_period = physics.PHYSICS_HZ / cfg.decision_hz
        self._next_slot = 0.0
        self._slot_owed = False
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

        if self._pending is not None:
            self._resolve(k)

        if k + 1e-9 >= self._next_slot:
            while self._next_slot <= k + 1e-9:
                self._next_slot += self._slot_period
            if self._pending is not None:
                self.stats.missed_slots += 1
                self.recorder.event({"type": "missed_slot", "tick": k, "pending_id": self._pending.id})
            self._slot_owed = True
        if self._slot_owed and self._pending is None:
            self._slot_owed = False
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
    def _issue_request(self, k: int) -> None:
        obs = self.env.observe()
        rid = self._next_id
        self._next_id += 1
        self.stats.requests += 1
        self.recorder.event({"type": "request", "id": rid, "tick": k, "observation": obs.to_dict()})
        wall = time.perf_counter()
        self._pending = _Pending(rid, k, wall)
        self.controller.request(obs)
        # Inline (synchronous) controllers are done by now; time them exactly.
        self._try_receive(self._pending)

    def _try_receive(self, pr: _Pending) -> bool:
        d = self.controller.poll()
        if d is None:
            return False
        received = time.perf_counter()  # runner's clock is the only authority
        pr.decision = d
        pr.latency_s = (received - pr.wall) + max(0.0, d.extra_latency_s)
        pr.effective_tick = pr.tick + max(1, math.ceil(pr.latency_s / TICK - 1e-9))
        self.stats.latencies_s.append(pr.latency_s)
        return True

    def _resolve(self, j: int) -> None:
        pr = self._pending
        assert pr is not None
        sim_elapsed = (j - pr.tick) * TICK
        while pr.decision is None:
            checked_at = time.perf_counter()
            if self._try_receive(pr):
                break
            if checked_at - pr.wall >= sim_elapsed:
                return  # truly not available by this tick: keep previous action
            # Simulation is ahead of the wall clock: wait for the truth.
            time.sleep(min(sim_elapsed - (checked_at - pr.wall), 0.0005))

        assert pr.latency_s is not None and pr.effective_tick is not None
        if self._deadline_s is not None and pr.latency_s > self._deadline_s:
            self.stats.late_dropped += 1
            self.recorder.event(
                {
                    "type": "decision_dropped",
                    "id": pr.id,
                    "tick": j,
                    "reason": "deadline",
                    "action": None if pr.decision.action is None else pr.decision.action.value,
                    "latency_ms": pr.latency_s * 1000,
                }
            )
            self._pending = None
            return
        if j < pr.effective_tick:
            return

        if pr.decision.action is None:
            # Failed request: nothing to apply, previous action continues.
            self.stats.failed += 1
            self.recorder.event(
                {
                    "type": "decision_failed",
                    "id": pr.id,
                    "tick": j,
                    "error": pr.decision.error,
                    "latency_ms": pr.latency_s * 1000,
                    "meta": pr.decision.meta,
                }
            )
            self._pending = None
            return

        action = pr.decision.action
        self.stats.applied += 1
        self.stats.delay_ticks.append(j - pr.tick)
        self.last_latency_s = pr.latency_s
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
                **({"meta": pr.decision.meta} if pr.decision.meta else {}),
            }
        )
        self._pending = None

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
            "late_dropped": st.late_dropped,
            "failed_decisions": st.failed,
            "mean_decision_latency_ms": mean(lat_ms),
            "p50_latency_ms": percentile(lat_ms, 50),
            "p95_latency_ms": percentile(lat_ms, 95),
            "mean_delay_ticks": mean(st.delay_ticks),
            "world_speed_scale": cfg.world_speed_scale,
            "decision_hz": cfg.decision_hz,
            "obstacle_count": cfg.obstacle_count,
            "final_obstacle_count": len(env.obstacles),
            "action_changes": self.action_changes,
        }
