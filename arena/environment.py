"""Survival + Collect arena.

The environment owns physics, collisions, spawning, observation generation
and scoring. It never decides for the controller: it only applies whatever
action it is given, and it never overrides or "saves" the player.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from arena import physics
from arena.action import ALL_ACTIONS, Action
from arena.difficulty import DifficultyConfig
from arena.entities import Obstacle, Player, Target
from arena.observation import ArenaInfo, Observation


@dataclass
class StepEvents:
    targets_collected: int = 0
    collision_obstacle_id: Optional[int] = None
    spawned_obstacle_ids: list[int] = field(default_factory=list)


@dataclass
class EpisodeOutcome:
    done: bool = False
    success: bool = False
    reason: Optional[str] = None  # collision | target_timeout | max_duration | target_goal
    time: Optional[float] = None
    collision_obstacle_id: Optional[int] = None


class Environment:
    """Deterministic given (config, seed, per-tick action sequence)."""

    MIN_TARGET_DISTANCE = 150.0
    SPAWN_ATTEMPTS = 500

    def __init__(self, config: DifficultyConfig, seed: int):
        config.validate()
        self.config = config
        self.seed = seed
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        cfg = self.config
        self._rng = random.Random(self.seed)
        self.tick = 0
        self.world_time = 0.0
        self.score = 0
        self.distance_travelled = 0.0
        self.last_target_time = 0.0
        self.outcome = EpisodeOutcome()
        self._next_obstacle_id = 0
        self._spawn_accumulator = 0.0

        self.player = Player(
            x=cfg.arena_width / 2,
            y=cfg.arena_height / 2,
            vx=0.0,
            vy=0.0,
            radius=cfg.player_radius,
            max_speed=cfg.player_max_speed,
        )
        self.obstacles: list[Obstacle] = []
        for _ in range(cfg.obstacle_count):
            ob = self._sample_initial_obstacle()
            if ob is not None:
                self.obstacles.append(ob)
        self.target = self._sample_target()

    def public_info(self) -> ArenaInfo:
        cfg = self.config
        return ArenaInfo(
            arena_width=cfg.arena_width,
            arena_height=cfg.arena_height,
            physics_hz=physics.PHYSICS_HZ,
            world_speed_scale=cfg.world_speed_scale,
            decision_hz=cfg.decision_hz,
            decision_deadline_ms=cfg.decision_deadline_ms,
            player_radius=cfg.player_radius,
            player_max_speed=cfg.player_max_speed,
            player_acceleration=cfg.player_acceleration,
            player_drag=cfg.player_drag,
            target_radius=cfg.target_radius,
            target_timeout=cfg.target_timeout,
            target_goal=cfg.target_goal,
            max_duration=cfg.max_duration,
            actions=tuple(a.value for a in ALL_ACTIONS),
        )

    # ------------------------------------------------------------ observation
    def observe(self) -> Observation:
        """Snapshot of current objective state, built from fresh primitives."""
        p = self.player
        t = self.target
        return Observation(
            timestamp=self.world_time,
            tick=self.tick,
            player={"x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy, "radius": p.radius},
            target={"x": t.x, "y": t.y, "radius": t.radius},
            obstacles=tuple(
                {"id": o.id, "x": o.x, "y": o.y, "vx": o.vx, "vy": o.vy, "radius": o.radius}
                for o in self.obstacles
            ),
            arena={"width": self.config.arena_width, "height": self.config.arena_height},
            score=self.score,
        )

    # ------------------------------------------------------------------- step
    @property
    def done(self) -> bool:
        return self.outcome.done

    @property
    def substeps(self) -> int:
        # Keep the integration step roughly constant in world time.
        return max(1, math.ceil(self.config.world_speed_scale))

    def step(self, action: Action) -> StepEvents:
        """Advance one physics tick (1/60 s of controller time)."""
        events = StepEvents()
        if self.outcome.done:
            return events
        cfg = self.config
        n = self.substeps
        dt = cfg.world_speed_scale * physics.TICK_SECONDS / n
        thrust = action.direction
        W, H = cfg.arena_width, cfg.arena_height

        for _ in range(n):
            self.distance_travelled += physics.integrate_player(
                self.player, thrust, cfg.player_acceleration, cfg.player_drag, dt, W, H
            )
            for ob in self.obstacles:
                physics.move_obstacle(ob, dt, W, H)
            self.world_time += dt

            self._maybe_spawn(dt, events)

            hit = physics.first_collision(self.player, self.obstacles)
            if hit is not None:
                events.collision_obstacle_id = hit.id
                self._finish(False, "collision", collision_obstacle_id=hit.id)
                break

            t = self.target
            if physics.circles_overlap(self.player.x, self.player.y, self.player.radius, t.x, t.y, t.radius):
                self.score += 1
                events.targets_collected += 1
                self.last_target_time = self.world_time
                self.target = self._sample_target()
                if cfg.target_goal is not None and self.score >= cfg.target_goal:
                    self._finish(True, "target_goal")
                    break

            if self.world_time - self.last_target_time > cfg.target_timeout:
                self._finish(False, "target_timeout")
                break
            if self.world_time >= cfg.max_duration - 1e-9:
                self._finish(True, "max_duration")
                break

        self.tick += 1
        return events

    def _finish(self, success: bool, reason: str, collision_obstacle_id: Optional[int] = None) -> None:
        self.outcome = EpisodeOutcome(
            done=True,
            success=success,
            reason=reason,
            time=self.world_time,
            collision_obstacle_id=collision_obstacle_id,
        )

    # --------------------------------------------------------------- spawning
    def _new_id(self) -> int:
        i = self._next_obstacle_id
        self._next_obstacle_id += 1
        return i

    def _random_velocity(self) -> tuple[float, float]:
        cfg = self.config
        speed = self._rng.uniform(cfg.obstacle_speed_min, cfg.obstacle_speed_max)
        ang = self._rng.uniform(0.0, 2 * math.pi)
        return speed * math.cos(ang), speed * math.sin(ang)

    def _violates_spawn_safety(self, x: float, y: float, vx: float, vy: float, r: float) -> bool:
        """Generator legality check: is the fresh obstacle trivially unfair?

        Rejects obstacles that start inside the player's safe bubble, or whose
        straight-line path passes through that bubble within the grace period.
        This is environment-internal and is never shown to controllers.
        """
        cfg = self.config
        p = self.player
        dx, dy = x - p.x, y - p.y
        clearance = cfg.spawn_safe_radius + r
        if dx * dx + dy * dy < clearance * clearance:
            return True
        vv = vx * vx + vy * vy
        if vv == 0:
            return False
        t = max(0.0, min(cfg.spawn_grace_period, -(dx * vx + dy * vy) / vv))
        cx, cy = dx + vx * t, dy + vy * t
        near = p.radius + r + 0.5 * cfg.spawn_safe_radius
        return cx * cx + cy * cy < near * near

    def _sample_initial_obstacle(self) -> Optional[Obstacle]:
        cfg = self.config
        for _ in range(self.SPAWN_ATTEMPTS):
            r = self._rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)
            x = self._rng.uniform(r, cfg.arena_width - r)
            y = self._rng.uniform(r, cfg.arena_height - r)
            vx, vy = self._random_velocity()
            if not self._violates_spawn_safety(x, y, vx, vy, r):
                return Obstacle(self._new_id(), x, y, vx, vy, r)
        return None

    def _maybe_spawn(self, dt: float, events: StepEvents) -> None:
        cfg = self.config
        if cfg.spawn_rate <= 0 or len(self.obstacles) >= cfg.max_obstacles:
            return
        self._spawn_accumulator += cfg.spawn_rate * dt
        while self._spawn_accumulator >= 1.0 and len(self.obstacles) < cfg.max_obstacles:
            self._spawn_accumulator -= 1.0
            ob = self._sample_edge_obstacle()
            if ob is not None:
                self.obstacles.append(ob)
                events.spawned_obstacle_ids.append(ob.id)

    def _sample_edge_obstacle(self) -> Optional[Obstacle]:
        """Enter from a random edge, heading inward."""
        cfg = self.config
        W, H = cfg.arena_width, cfg.arena_height
        for _ in range(self.SPAWN_ATTEMPTS):
            r = self._rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)
            vx, vy = self._random_velocity()
            edge = self._rng.randrange(4)
            if edge == 0:  # top
                x, y, vy = self._rng.uniform(r, W - r), r, abs(vy)
            elif edge == 1:  # bottom
                x, y, vy = self._rng.uniform(r, W - r), H - r, -abs(vy)
            elif edge == 2:  # left
                x, y, vx = r, self._rng.uniform(r, H - r), abs(vx)
            else:  # right
                x, y, vx = W - r, self._rng.uniform(r, H - r), -abs(vx)
            if not self._violates_spawn_safety(x, y, vx, vy, r):
                return Obstacle(self._new_id(), x, y, vx, vy, r)
        return None

    def _sample_target(self) -> Target:
        cfg = self.config
        tr = cfg.target_radius
        margin = tr + cfg.player_radius * 2
        p = self.player
        fallback: Optional[tuple[float, float]] = None
        for attempt in range(self.SPAWN_ATTEMPTS):
            x = self._rng.uniform(margin, cfg.arena_width - margin)
            y = self._rng.uniform(margin, cfg.arena_height - margin)
            fallback = (x, y)
            relax = attempt > self.SPAWN_ATTEMPTS // 2
            if not relax and math.hypot(x - p.x, y - p.y) < self.MIN_TARGET_DISTANCE:
                continue
            if any(
                math.hypot(x - o.x, y - o.y) < o.radius + tr + 2 * cfg.player_radius for o in self.obstacles
            ):
                continue
            return Target(x, y, tr)
        assert fallback is not None
        return Target(fallback[0], fallback[1], tr)
