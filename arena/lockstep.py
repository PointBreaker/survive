"""Lockstep runner: measure decision quality with latency taken out.

This is a *separate, explicitly labeled* measurement mode. The real-time
benchmark (``arena.runner.EpisodeRunner``: the world never waits) is
unchanged and remains the only mode for real-time capability.

In lockstep:
* a decision point occurs every ``interval_s`` of world time, for every
  controller alike (decision rate is pinned, not bounded by each controller's
  own latency);
* at a decision point the world waits until the answers arrive; wall-clock
  latency is still measured and logged, but costs nothing in world time;
* ``delay_s`` optionally inserts a fixed world-time delay between the snapshot
  and the moment its answer takes effect. That's a controlled, jitter-free
  latency for robustness curves (success vs delay).

Shadow mode: one ``driver`` acts; any number of ``shadows`` receive the
identical observation at every decision point and answer, but never act.
Their answers are logged; decision quality is graded by counterfactual
takeover branches (``run_takeover``, ``benchmark.takeover``). All
controllers receive exactly the public Observation/ArenaInfo, as usual.

Replay: the driver's applied actions are logged as ``action_changes``
(per-tick), so ``arena.replay`` re-simulates lockstep episodes exactly.
"""
from __future__ import annotations

import collections
import time
from typing import Any, Optional

from arena.action import Action
from arena.environment import Environment
from arena.recorder import NullRecorder, Recorder
from arena.stats import mean, percentile
from controllers.base import Controller

TICKS_PER_WORLD_S = 60  # lockstep configs use world_speed_scale = 1: one tick = 1/60 world s


class LockstepRunner:
    def __init__(
        self,
        env: Environment,
        driver: tuple[str, Controller],
        shadows: Optional[dict[str, Controller]] = None,
        interval_s: float = 0.1,
        delay_s: float = 0.0,
        recorder: Optional[Recorder] = None,
        answer_timeout_s: float = 60.0,
    ):
        if env.config.world_speed_scale != 1.0:
            raise ValueError("lockstep configs use world_speed_scale = 1 (time is decision-driven)")
        self.env = env
        self.driver_name, self.driver = driver
        self.shadows = dict(shadows or {})
        self.interval_ticks = max(1, round(interval_s * TICKS_PER_WORLD_S))
        self.delay_ticks = max(0, round(delay_s * TICKS_PER_WORLD_S))
        self.interval_s = self.interval_ticks / TICKS_PER_WORLD_S
        self.delay_s = self.delay_ticks / TICKS_PER_WORLD_S
        self.recorder: Recorder = recorder or NullRecorder()
        self.timeout_s = answer_timeout_s

    @property
    def controllers(self) -> dict[str, Controller]:
        return {self.driver_name: self.driver, **self.shadows}

    def _collect(self, obs, rid: int) -> dict[str, dict[str, Any]]:
        """Hand the same observation to every controller and wait for all answers."""
        answers: dict[str, dict[str, Any]] = {}
        t0: dict[str, float] = {}
        for name, c in self.controllers.items():
            t0[name] = time.perf_counter()
            c.request(obs, rid)
        pending = set(self.controllers)
        deadline = time.perf_counter() + self.timeout_s
        while pending:
            for name in list(pending):
                d = self.controllers[name].poll()
                if d is None:
                    continue
                pending.discard(name)
                meta = d.meta or {}
                answers[name] = {
                    "action": None if d.action is None else d.action.value,
                    "latency_ms": (time.perf_counter() - t0[name]) * 1000 + d.extra_latency_s * 1000,
                    **({"error": d.error} if d.error else {}),
                    **({"confidence": meta["confidence"]} if "confidence" in meta else {}),
                    **({"probabilities": meta["probabilities"]} if "probabilities" in meta else {}),
                }
            if pending:
                if time.perf_counter() > deadline:
                    for name in pending:
                        answers[name] = {"action": None, "latency_ms": self.timeout_s * 1000, "error": "timeout"}
                    break
                time.sleep(0.0005)
        return answers

    def run(self) -> dict[str, Any]:
        env = self.env
        info = env.public_info()
        for c in self.controllers.values():
            c.reset(info)
        self.recorder.event({
            "type": "episode_start", "mode": "lockstep", "controller": self.driver_name, "driver": self.driver_name,
            "shadows": list(self.shadows), "seed": env.seed, "config": env.config.to_dict(), "info": info.to_dict(),
            "interval_s": self.interval_s, "delay_s": self.delay_s,
        })
        current = Action.STAY
        action_changes: list[tuple[int, str]] = [(0, Action.STAY.value)]
        pending: collections.deque = collections.deque()  # (effective_tick, action, request_tick)
        applied_request_tick: Optional[int] = None
        latencies: dict[str, list[float]] = {n: [] for n in self.controllers}
        failures: dict[str, int] = {n: 0 for n in self.controllers}
        rid = 0

        while not env.done:
            k = env.tick
            if k % self.interval_ticks == 0:
                control = {
                    "applied_request_tick": applied_request_tick,
                    "applied_latency_s": self.delay_s if applied_request_tick is not None else None,
                    "applied_latency_world_s": self.delay_s if applied_request_tick is not None else None,
                }
                obs = env.observe(control)
                answers = self._collect(obs, rid)
                for n, a in answers.items():
                    latencies[n].append(a["latency_ms"])
                    if a["action"] is None:
                        failures[n] += 1
                self.recorder.event({"type": "lockstep_decision", "id": rid, "tick": k,
                                     "effective_tick": k + self.delay_ticks, "answers": answers})
                drv = answers.get(self.driver_name, {})
                if drv.get("action"):
                    pending.append((k + self.delay_ticks, Action(drv["action"]), k))
                rid += 1
            while pending and pending[0][0] <= k:
                _, act, req_tick = pending.popleft()
                applied_request_tick = req_tick
                if act != current:
                    if action_changes[-1][0] == k:  # never two entries for one tick (replay sorts them)
                        action_changes.pop()
                    action_changes.append((k, act.value))
                current = act

            events = env.step(current)
            if events.targets_collected:
                self.recorder.event({"type": "target_collected", "tick": k, "t": env.world_time, "score": env.score})
            if events.collision_obstacle_id is not None:
                self.recorder.event({"type": "collision", "tick": k, "t": env.world_time,
                                     "obstacle_id": events.collision_obstacle_id, "action": current.value})

        o = env.outcome
        result = {
            "mode": "lockstep",
            "controller": self.driver_name,
            "driver": self.driver_name,
            "shadows": list(self.shadows),
            "seed": env.seed,
            "done": o.done,
            "success": o.success,
            "reason": o.reason,
            "survival_time": env.world_time,
            "targets_collected": env.score,
            "distance_travelled": env.distance_travelled,
            "collision_time": o.time if o.reason == "collision" else None,
            "collision_obstacle_id": o.collision_obstacle_id,
            "ticks": env.tick,
            "decision_points": rid,
            "interval_s": self.interval_s,
            "delay_s": self.delay_s,
            "latency_ms": {n: {"mean": mean(v), "p50": percentile(v, 50), "p95": percentile(v, 95)}
                           for n, v in latencies.items()},
            "failed_answers": failures,
            "obstacle_count": env.config.obstacle_count,
            "action_changes": action_changes,
            # fields shared with real-time results so aggregate() applies unchanged
            "mean_decision_latency_ms": mean(latencies[self.driver_name]),
            "p95_latency_ms": percentile(latencies[self.driver_name], 95),
            "missed_slots": 0,
            "delayed_slots": 0,
        }
        self.recorder.event({"type": "episode_end", "result": result})
        self.recorder.write_result(result)
        return result


def _clearance(env: Environment) -> float:
    import math

    p = env.player
    return min((math.hypot(p.x - o.x, p.y - o.y) - p.radius - o.radius for o in env.obstacles), default=1e9)


def run_takeover(env: Environment, controller: Controller, interval_ticks: int, delay_ticks: int,
                 max_ticks: int, path_every: int = 3, answer_timeout_s: float = 60.0,
                 frames_every: int = 0, initial_action: Action = Action.STAY) -> dict[str, Any]:
    """Let one controller act, in lockstep, from the state ``env`` is in (``env`` is modified).

    Used for counterfactual takeover branches: every controller starts from an
    identical copy of the same moment and is judged on its own consequences.
    The controller only sees the public observation, as always.

    ``frames_every`` > 0 also records obstacle positions (``[[id, x, y], ...]``
    every n ticks) for visualisation. Obstacles are recorded per branch because
    spawns can depend on the player's position, so branches may diverge.
    """
    import math

    controller.reset(env.public_info())
    d0 = math.hypot(env.player.x - env.target.x, env.player.y - env.target.y)
    score0 = env.score
    current = initial_action  # the action already in effect (matters when delay_ticks > 0)
    pending: collections.deque = collections.deque()
    path = [[round(env.player.x, 1), round(env.player.y, 1)]]
    min_clear = _clearance(env)
    latencies: list[float] = []
    failures = 0
    applied_tick: Optional[int] = None
    radii: dict[int, float] = {}
    frames: list[list[list[float]]] = []

    def snap() -> None:
        for o in env.obstacles:
            radii.setdefault(o.id, round(o.radius, 1))
        frames.append([[o.id, round(o.x), round(o.y)] for o in env.obstacles])

    if frames_every:
        snap()
    ticks = 0
    for i in range(max_ticks):
        if env.done:
            break
        ticks = i + 1
        if i % interval_ticks == 0:
            obs = env.observe({"applied_request_tick": applied_tick,
                               "applied_latency_s": delay_ticks / TICKS_PER_WORLD_S if applied_tick is not None else None,
                               "applied_latency_world_s": delay_ticks / TICKS_PER_WORLD_S if applied_tick is not None else None})
            t0 = time.perf_counter()
            controller.request(obs, i)
            d = None
            deadline = t0 + answer_timeout_s
            while d is None and time.perf_counter() < deadline:
                d = controller.poll()
                if d is None:
                    time.sleep(0.0005)
            latencies.append((time.perf_counter() - t0) * 1000)
            if d is not None and d.action is not None:
                pending.append((i + delay_ticks, d.action, env.tick))
            else:
                failures += 1
        while pending and pending[0][0] <= i:
            _, current, applied_tick = pending.popleft()
        env.step(current)
        min_clear = min(min_clear, _clearance(env))
        if (i + 1) % path_every == 0 or env.done:
            path.append([round(env.player.x, 1), round(env.player.y, 1)])
        if frames_every and ((i + 1) % frames_every == 0 or env.done):
            snap()
    collided = env.done and env.outcome.reason == "collision"
    targets = env.score - score0
    d1 = math.hypot(env.player.x - env.target.x, env.player.y - env.target.y)
    return {
        "collided": collided,
        "end_reason": env.outcome.reason,
        "ticks": ticks,
        "targets": targets,
        # approach: fraction of the initial target distance closed (only meaningful without a pickup)
        "approach": max(-1.0, min(1.0, (d0 - d1) / max(d0, 50.0))) if targets == 0 else 0.0,
        "min_clearance": round(min_clear, 2),
        "path": path,
        "latency_ms_p50": percentile(latencies, 50),
        "failed_answers": failures,
        **({"frames": frames, "radii": {str(k): v for k, v in radii.items()}} if frames_every else {}),
    }
