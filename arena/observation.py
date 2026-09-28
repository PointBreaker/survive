"""The public controller interface: ``Observation`` and ``ArenaInfo``.

Everything a controller may know comes through these two types.

* ``ArenaInfo`` is given once at reset: the fixed, public rules of the game
  (arena size, player dynamics constants, timing). It never contains the
  seed or any random-generator state.
* ``Observation`` is a snapshot of the *current* objective world state:
  positions, velocities and radii. It contains no predictions, no risk
  scores, no recommendations and no ordering by danger. Obstacles are listed
  in id (spawn) order.
* ``Observation.control`` is feedback about the controller's *own* control
  loop, measured by the runner: which of its observations the action now in
  force was answering, and how long that answer took. It says nothing about
  the world. It lets any controller (not only one that can time its own
  calls) account for its latency.

Both are built from fresh primitive values, so a controller holding or
mutating them cannot reach or alter environment internals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


EMPTY_CONTROL: dict[str, Any] = {
    "applied_request_tick": None,  # tick of the observation the action in force answered
    "applied_latency_s": None,  # its measured latency, controller seconds
    "applied_latency_world_s": None,  # the same latency in world seconds
}


@dataclass(frozen=True)
class ArenaInfo:
    arena_width: float
    arena_height: float
    physics_hz: int
    world_speed_scale: float
    decision_hz: float
    decision_deadline_ms: Optional[float]
    player_radius: float
    player_max_speed: float
    player_acceleration: float
    player_drag: float
    target_radius: float
    target_timeout: float
    target_goal: Optional[int]
    max_duration: float
    actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["actions"] = list(self.actions)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArenaInfo":
        return cls(**{**d, "actions": tuple(d["actions"])})


@dataclass(frozen=True)
class Observation:
    timestamp: float  # world seconds since episode start
    tick: int  # physics tick at which the snapshot was taken
    player: dict[str, float]  # x, y, vx, vy, radius
    target: dict[str, float]  # x, y, radius
    obstacles: tuple[dict[str, float], ...]  # id, x, y, vx, vy, radius (id order)
    arena: dict[str, float]  # width, height
    score: int  # targets collected so far
    # Own-loop feedback: {"applied_request_tick", "applied_latency_s", "applied_latency_world_s"}
    control: dict[str, Any] = field(default_factory=lambda: dict(EMPTY_CONTROL))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tick": self.tick,
            "player": dict(self.player),
            "target": dict(self.target),
            "obstacles": [dict(o) for o in self.obstacles],
            "arena": dict(self.arena),
            "score": self.score,
            "control": dict(self.control),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        return cls(
            timestamp=d["timestamp"],
            tick=d["tick"],
            player=dict(d["player"]),
            target=dict(d["target"]),
            obstacles=tuple(dict(o) for o in d["obstacles"]),
            arena=dict(d["arena"]),
            score=d["score"],
            control=dict(d.get("control") or EMPTY_CONTROL),
        )


# Keys that must never appear anywhere in an observation. Checked by tests.
FORBIDDEN_KEYS = frozenset(
    {
        "collision_risk",
        "time_to_collision",
        "ttc",
        "safe_direction",
        "recommended_direction",
        "recommended_speed",
        "recommended_action",
        "danger",
        "danger_level",
        "best_action",
        "predicted_trajectory",
        "future",
        "distance_to_future_collision",
        "escape_direction",
        "nearest_safe_point",
        "most_dangerous_obstacle",
        "threat",
        "safe",
        "unsafe",
        "seed",
        "rng",
    }
)
