"""Structured difficulty configuration.

Difficulty is a vector of independent knobs, not a single easy/medium/hard
label. The presets at the bottom are only convenient starting points.

Units
-----
* Distances are in world units (1 unit == 1 pixel at default zoom).
* Speeds are world units per *world* second.
* Durations (``max_duration``, ``target_timeout``, ``spawn_rate``) are in
  *world* seconds.
* ``decision_hz`` and ``decision_deadline_ms`` are in *controller clock*
  (real wall-clock) time. ``world_speed_scale`` never touches them.

World speed
-----------
``world_speed_scale`` dilates world time relative to the controller clock:
one physics tick (1/60 s of controller time) advances the world by
``world_speed_scale / 60`` world seconds. At 8x, obstacles cover 8x more
ground per real second while a 150 ms controller still takes 150 real ms.
Geometry of the task is unchanged; only temporal pressure grows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Optional


@dataclass(frozen=True)
class DifficultyConfig:
    # Arena
    arena_width: float = 1200.0
    arena_height: float = 800.0

    # Obstacles
    obstacle_count: int = 12
    obstacle_speed_min: float = 40.0
    obstacle_speed_max: float = 130.0
    obstacle_radius_min: float = 12.0
    obstacle_radius_max: float = 30.0
    spawn_rate: float = 0.0  # extra obstacles per world second (enter from edges)
    max_obstacles: int = 400  # cap for spawn_rate growth

    # Player dynamics
    player_radius: float = 12.0
    player_max_speed: float = 260.0
    player_acceleration: float = 900.0
    player_drag: float = 1.5  # 1/s, linear drag coefficient

    # Objective
    target_radius: float = 16.0
    target_timeout: float = 15.0  # world seconds without a pickup -> failure
    target_goal: Optional[int] = None  # optional early success after N targets

    # Episode
    max_duration: float = 60.0  # world seconds

    # Time
    world_speed_scale: float = 1.0
    decision_hz: float = 10.0
    decision_deadline_ms: Optional[float] = None  # late results are dropped

    # Generator safety (environment-internal, never exposed as advice)
    spawn_safe_radius: float = 180.0
    spawn_grace_period: float = 1.5  # world s: no obstacle heads straight at spawn

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DifficultyConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown difficulty fields: {sorted(unknown)}")
        return cls(**data)

    def with_overrides(self, **overrides: Any) -> "DifficultyConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)

    def validate(self) -> None:
        if self.obstacle_count < 0:
            raise ValueError("obstacle_count must be >= 0")
        if self.obstacle_speed_min > self.obstacle_speed_max:
            raise ValueError("obstacle_speed_min > obstacle_speed_max")
        if self.obstacle_radius_min > self.obstacle_radius_max:
            raise ValueError("obstacle_radius_min > obstacle_radius_max")
        if self.world_speed_scale <= 0:
            raise ValueError("world_speed_scale must be > 0")
        if self.decision_hz <= 0:
            raise ValueError("decision_hz must be > 0")


PRESETS: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(obstacle_count=6, obstacle_speed_min=30.0, obstacle_speed_max=100.0),
    "medium": DifficultyConfig(),
    "hard": DifficultyConfig(obstacle_count=25, obstacle_speed_min=60.0, obstacle_speed_max=180.0),
}
